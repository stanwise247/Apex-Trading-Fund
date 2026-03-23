"""
APEX Trading Engine - Phase 1 Backend Server
Provides live market data, news, macro analysis, historical OHLCV, and deep AI news thesis.
Run with: python3 server.py
"""

import os
import re
import json
import time
import sqlite3
import logging
import threading
import requests
from datetime import datetime, timedelta, timezone
from flask import Flask, jsonify, request
from flask_cors import CORS

logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('APEX')

# Prevents concurrent backfill threads from contending on write locks
_backfill_lock = threading.Lock()

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

CONFIG_FILE = 'config.json'

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}

def save_config(cfg):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(cfg, f, indent=2)

cfg           = load_config()
# Environment variables take priority — allows Railway/cloud deployment
import os as _os
POLYGON_KEY     = _os.environ.get('POLYGON_KEY',    cfg.get('polygon_key',   ''))
NEWS_KEY        = _os.environ.get('NEWS_KEY',        cfg.get('news_key',      ''))
ANTHROPIC_KEY   = _os.environ.get('ANTHROPIC_KEY',  cfg.get('anthropic_key', ''))
TELEGRAM_TOKEN  = _os.environ.get('TELEGRAM_TOKEN', cfg.get('telegram_token', ''))
TELEGRAM_CHAT   = _os.environ.get('TELEGRAM_CHAT_ID',cfg.get('telegram_chat_id',''))
DB_PATH         = _os.environ.get('DB_PATH',        cfg.get('db_path', 'apex_market.db'))
# Write env vars back so other modules (telegram_alerts etc) can read config.json
if TELEGRAM_TOKEN and not cfg.get('telegram_token'):
    cfg['telegram_token']   = TELEGRAM_TOKEN
    cfg['telegram_chat_id'] = TELEGRAM_CHAT
    save_config(cfg)

INSTRUMENTS = {
    'NQ':  {'yahoo': 'NQ=F',     'polygon_paid': 'NQ:CME',   'databento': 'NQ.c.0', 'type': 'future', 'name': 'Nasdaq 100 Futures'},
    'ES':  {'yahoo': 'ES=F',     'polygon_paid': 'ES:CME',   'databento': 'ES.c.0', 'type': 'future', 'name': 'S&P 500 E-Mini'},
    'GC':  {'yahoo': 'GC=F',     'polygon_paid': 'GC:COMEX', 'databento': 'GC.c.0', 'type': 'future', 'name': 'Gold Futures'},
    'CL':  {'yahoo': 'CL=F',     'polygon_paid': 'CL:NYMEX', 'databento': None,     'type': 'future', 'name': 'Crude Oil Futures'},
    'ZN':  {'yahoo': 'ZN=F',     'polygon_paid': 'ZN:CBOT',  'databento': None,     'type': 'future', 'name': '10Y T-Note Futures'},
    'VIX': {'yahoo': '^VIX',     'polygon_paid': None,       'databento': None,     'type': 'index',  'name': 'CBOE VIX'},
    'DXY': {'yahoo': 'DX-Y.NYB', 'polygon_paid': None,       'databento': None,     'type': 'forex',  'name': 'Dollar Index'},
    'BTC': {'yahoo': 'BTC-USD',  'polygon_paid': None,       'databento': None,     'type': 'crypto', 'name': 'Bitcoin USD'},
}

# Databento dataset for CME futures
DATABENTO_DATASET  = 'GLBX.MDP3'
DATABENTO_API_KEY  = os.environ.get('DATABENTO_API_KEY', '')

REFERENCE_PRICES = {
    'NQ': 21500, 'ES': 5780,  'GC': 3050,  'CL': 71.5,
    'ZN': 109.5, 'VIX': 19.5, 'DXY': 103.8,'BTC': 85000,
    'TNX': 4.35, 'IRX': 4.25,
}

YF_INTERVAL_MAP = {'1min': '1m', '5min': '5m', '15min': '15m', '1hour': '1h', '4hour': '1h', '1day': '1d', '1week': '1wk'}
YF_PERIOD_MAP   = {'1min': '7d', '5min': '60d', '15min': '60d', '1hour': '2y', '4hour': '2y', '1day': '10y', '1week': '10y'}


def yf_fetch(yahoo_ticker, tf_label):
    import yfinance as yf
    import pandas as _pd
    interval = YF_INTERVAL_MAP.get(tf_label, '1d')
    period   = YF_PERIOD_MAP.get(tf_label, '5y')
    try:
        df = yf.Ticker(yahoo_ticker).history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            logger.warning('yfinance empty: ' + yahoo_ticker + ' ' + tf_label)
            return []
        # yfinance has no 4h interval — fetched as 1h, must aggregate to 4h
        if tf_label == '4hour':
            df = df.resample('4h').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min',
                'Close': 'last', 'Volume': 'sum',
            }).dropna(subset=['Open'])
        bars = []
        for ts, row in df.iterrows():
            bars.append({
                't': int(ts.timestamp()),
                'o': round(float(row['Open']), 4),
                'h': round(float(row['High']), 4),
                'l': round(float(row['Low']), 4),
                'c': round(float(row['Close']), 4),
                'v': int(row['Volume']) if row['Volume'] else 0,
            })
        logger.info('yfinance OK: ' + yahoo_ticker + ' ' + tf_label + ' ' + str(len(bars)) + ' bars')
        return bars
    except Exception as e:
        logger.warning('yfinance failed ' + yahoo_ticker + ' ' + tf_label + ': ' + str(e))
        return []


def yf_get_quote(yahoo_ticker):
    import yfinance as yf
    try:
        info  = yf.Ticker(yahoo_ticker).fast_info
        price = info.last_price or info.previous_close
        prev  = info.previous_close or price
        if price:
            chg = price - prev if prev else 0
            pct = (chg / prev * 100) if prev else 0
            return {
                'price':      round(float(price), 4),
                'change':     round(float(chg), 4),
                'change_pct': round(float(pct), 3),
                'live':       True,
            }
    except Exception as e:
        logger.debug('yf quote failed ' + yahoo_ticker + ': ' + str(e))
    return None


def databento_fetch(symbol, tf_label, years=3):
    """
    Fetch OHLCV bars from Databento for NQ, ES, GC.
    Uses CME Globex MDP3.0 dataset with continuous front-month contracts.
    Falls back to yfinance if Databento key not set or request fails.
    """
    import requests
    from datetime import datetime, timedelta, timezone as _tz

    api_key = DATABENTO_API_KEY or os.environ.get('DATABENTO_API_KEY', '')
    if not api_key:
        logger.debug('Databento key not set — falling back to yfinance')
        return None

    db_symbol = INSTRUMENTS.get(symbol, {}).get('databento')
    if not db_symbol:
        return None

    # Map our timeframe labels to Databento schemas
    schema_map = {
        '1min':  'ohlcv-1m',
        '5min':  'ohlcv-1m',
        '15min': 'ohlcv-1m',
        '1hour': 'ohlcv-1h',
        '4hour': 'ohlcv-1h',
        '1day':  'ohlcv-1d',
        '1week': 'ohlcv-1d',
    }

    # Aggregation multipliers (how many base bars to combine)
    agg_map = {
        '1min':  1,
        '5min':  5,
        '15min': 15,
        '1hour': 1,
        '4hour': 4,
        '1day':  1,
        '1week': 7,
    }

    schema = schema_map.get(tf_label, 'ohlcv-1d')
    agg    = agg_map.get(tf_label, 1)

    now       = datetime.now(_tz.utc)
    # Databento historical API only serves data up to midnight UTC
    end       = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start     = end - timedelta(days=365 * years)
    start_str = start.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_str   = end.strftime('%Y-%m-%dT%H:%M:%SZ')

    try:
        logger.info(f'Databento fetch: {symbol} ({db_symbol}) {tf_label} schema={schema}')

        resp = requests.get(
            'https://hist.databento.com/v0/timeseries.get_range',
            params={
                'dataset':    DATABENTO_DATASET,
                'symbols':    db_symbol,
                'schema':     schema,
                'start':      start_str,
                'end':        end_str,
                'stype_in':   'continuous',
                'encoding':   'json',
            },
            auth=(api_key, ''),
            timeout=60,
        )

        if resp.status_code != 200:
            logger.warning(f'Databento error {resp.status_code}: {resp.text[:200]}')
            return None

        # Databento returns newline-delimited JSON (one object per line)
        raw = []
        for line in resp.text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                import json as _json
                obj = _json.loads(line)
                if isinstance(obj, list):
                    raw.extend(obj)
                elif isinstance(obj, dict):
                    raw.append(obj)
            except Exception:
                continue

        if not raw:
            logger.warning(f'Databento empty response for {symbol} {tf_label}')
            return None

        # Parse raw bars
        # Databento format: {"hd":{"ts_event":"1234567890000000000",...},"open":"24435250000000",...}
        # Prices are fixed-point integers, divide by 1e9 to get actual price
        # Timestamps are nanoseconds since epoch
        parsed = []
        for r in raw:
            try:
                # Get timestamp from nested hd object
                hd = r.get('hd', {})
                ts_raw = hd.get('ts_event', 0)
                ts = int(ts_raw) // 1_000_000_000  # nanoseconds -> seconds

                # Prices are strings of fixed-point integers (divide by 1e9)
                def parse_price(val):
                    if val is None: return 0.0
                    return round(float(val) / 1e9, 4)

                parsed.append({
                    't': ts,
                    'o': parse_price(r.get('open')),
                    'h': parse_price(r.get('high')),
                    'l': parse_price(r.get('low')),
                    'c': parse_price(r.get('close')),
                    'v': int(r.get('volume', 0)),
                })
            except Exception as pe:
                logger.debug(f'Databento parse error: {pe} row={r}')
                continue

        if not parsed:
            logger.warning(f'Databento: no parseable bars for {symbol} {tf_label}')
            return None

        # Aggregate if needed — use clock-aligned resample, not positional chunking
        if agg > 1:
            import pandas as _pd
            _df = _pd.DataFrame(parsed)
            _df['dt'] = _pd.to_datetime(_df['t'], unit='s', utc=True)
            _df.set_index('dt', inplace=True)
            resample_rule = {1: '1min', 5: '5min', 15: '15min', 4: '4h', 7: '1W'}.get(agg, f'{agg}min')
            _agg = _df.resample(resample_rule).agg(
                o=('o', 'first'), h=('h', 'max'), l=('l', 'min'),
                c=('c', 'last'), v=('v', 'sum')
            ).dropna(subset=['o'])
            parsed = [
                {'t': int(ts.timestamp()), 'o': row['o'], 'h': row['h'],
                 'l': row['l'], 'c': row['c'], 'v': int(row['v'])}
                for ts, row in _agg.iterrows()
            ]

        logger.info(f'Databento OK: {symbol} {tf_label} {len(parsed)} bars')
        return parsed

    except Exception as e:
        logger.warning(f'Databento fetch failed {symbol} {tf_label}: {e}')
        return None


def fetch_bars(symbol, tf_label, years=3):
    """
    Unified bar fetcher — tries Databento first for NQ/ES/GC,
    falls back to yfinance for everything else or on failure.
    """
    db_symbol = INSTRUMENTS.get(symbol, {}).get('databento')
    if db_symbol and DATABENTO_API_KEY:
        bars = databento_fetch(symbol, tf_label, years)
        if bars:
            return bars
        logger.info(f'Databento failed for {symbol} {tf_label} — falling back to yfinance')

    yahoo = INSTRUMENTS.get(symbol, {}).get('yahoo')
    if yahoo:
        return yf_fetch(yahoo, tf_label)
    return []


def init_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute('PRAGMA journal_mode=WAL')   # allow concurrent reads during writes
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS ohlcv (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL, timeframe TEXT NOT NULL, ts INTEGER NOT NULL,
        open REAL, high REAL, low REAL, close REAL, volume REAL,
        UNIQUE(symbol, timeframe, ts))''')
    c.execute('''CREATE TABLE IF NOT EXISTS macro_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER NOT NULL, date TEXT NOT NULL,
        vix REAL, tnx REAL, irx REAL, dxy REAL, gold REAL, oil REAL, regime TEXT, notes TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS patterns (
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE, symbol TEXT, conditions TEXT,
        occurrences INTEGER DEFAULT 0, wins INTEGER DEFAULT 0, losses INTEGER DEFAULT 0,
        avg_rr REAL, expectancy REAL, best_regime TEXT, edge_score REAL DEFAULT 0,
        last_updated INTEGER, active INTEGER DEFAULT 1)''')
    c.execute('''CREATE TABLE IF NOT EXISTS scan_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER NOT NULL, symbol TEXT,
        pattern_name TEXT, score REAL, direction TEXT, entry REAL, stop REAL,
        target1 REAL, target2 REAL, outcome TEXT DEFAULT 'pending',
        outcome_rr REAL, regime TEXT, notes TEXT)''')
    conn.commit()
    conn.close()
    logger.info('Database initialised: ' + DB_PATH)


def store_ohlcv(symbol, timeframe, bars):
    if not bars:
        return 0
    conn = sqlite3.connect(DB_PATH, timeout=30)
    c = conn.cursor()
    stored = 0
    for bar in bars:
        try:
            ts = int(bar.get('t', 0))
            if ts > 1e12:
                ts = ts // 1000
            c.execute(
                'INSERT OR IGNORE INTO ohlcv (symbol,timeframe,ts,open,high,low,close,volume) VALUES (?,?,?,?,?,?,?,?)',
                (symbol, timeframe, ts, bar.get('o'), bar.get('h'), bar.get('l'), bar.get('c'), bar.get('v'))
            )
            stored += c.rowcount
        except Exception as e:
            logger.debug('OHLCV skip: ' + str(e))
    conn.commit()
    conn.close()
    return stored


def get_ohlcv(symbol, timeframe, limit=500):
    conn = sqlite3.connect(DB_PATH, timeout=30)
    c = conn.cursor()
    c.execute(
        'SELECT ts,open,high,low,close,volume FROM ohlcv WHERE symbol=? AND timeframe=? ORDER BY ts DESC LIMIT ?',
        (symbol, timeframe, limit)
    )
    rows = c.fetchall()
    conn.close()
    return [{'t': r[0], 'o': r[1], 'h': r[2], 'l': r[3], 'c': r[4], 'v': r[5]} for r in reversed(rows)]


def get_db_stats():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    c = conn.cursor()
    c.execute('SELECT symbol,timeframe,COUNT(*),MIN(ts),MAX(ts) FROM ohlcv GROUP BY symbol,timeframe')
    rows = c.fetchall()
    conn.close()
    out = []
    for r in rows:
        f = datetime.fromtimestamp(r[3], tz=timezone.utc).strftime('%Y-%m-%d') if r[3] else '--'
        l = datetime.fromtimestamp(r[4], tz=timezone.utc).strftime('%Y-%m-%d') if r[4] else '--'
        out.append({'symbol': r[0], 'timeframe': r[1], 'bars': r[2], 'from': f, 'to': l})
    return out


def backfill_history(symbol, years=5):
    info = INSTRUMENTS.get(symbol, {})
    if not info:
        logger.warning('Unknown symbol: ' + symbol)
        return
    if not _backfill_lock.acquire(blocking=False):
        logger.warning(f'Backfill skipped for {symbol} — another backfill already running')
        return
    try:
        source = 'Databento' if (info.get('databento') and DATABENTO_API_KEY) else 'yfinance'
        logger.info(f'Backfill started: {symbol} (source={source})')
        for tf in ['1week', '1day', '4hour', '1hour', '15min', '5min']:
            try:
                logger.info('  Fetching ' + symbol + ' ' + tf + '...')
                bars = fetch_bars(symbol, tf, years)
                if bars:
                    stored = store_ohlcv(symbol, tf, bars)
                    logger.info('  OK ' + symbol + ' ' + tf + ': ' + str(stored) + ' new bars (' + str(len(bars)) + ' total)')
                else:
                    logger.warning('  FAIL ' + symbol + ' ' + tf + ': no data')
                time.sleep(1)
            except Exception as e:
                logger.error('  Error ' + symbol + ' ' + tf + ': ' + str(e))
        logger.info('Backfill complete: ' + symbol)
    finally:
        _backfill_lock.release()


def daily_update(symbol):
    yahoo = INSTRUMENTS.get(symbol, {}).get('yahoo')
    if not yahoo:
        return
    for tf in ['1day', '1hour', '5min']:
        try:
            bars = yf_fetch(yahoo, tf)
            if bars:
                stored = store_ohlcv(symbol, tf, bars)
                if stored:
                    logger.info('Daily update ' + symbol + ' ' + tf + ': ' + str(stored) + ' new bars')
        except Exception as e:
            logger.debug('Daily update failed ' + symbol + ' ' + tf + ': ' + str(e))
        time.sleep(1)


def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i-1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    ag = sum(gains[:period]) / period
    al = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        ag = (ag * (period-1) + gains[i]) / period
        al = (al * (period-1) + losses[i]) / period
    return round(100 - (100 / (1 + ag/al)), 2) if al != 0 else 100


def calc_ema(closes, period):
    if not closes:
        return []
    k = 2 / (period + 1)
    ema = [closes[0]]
    for p in closes[1:]:
        ema.append(p * k + ema[-1] * (1 - k))
    return ema


def calc_macd(closes):
    if len(closes) < 26:
        return None, None, None
    e12 = calc_ema(closes, 12)
    e26 = calc_ema(closes, 26)
    ml  = [a - b for a, b in zip(e12, e26)]
    sig = calc_ema(ml[25:], 9)
    hist = [m - s for m, s in zip(ml[25+len(ml[25:])-len(sig):], sig)]
    return round(ml[-1], 4), (round(sig[-1], 4) if sig else None), (round(hist[-1], 4) if hist else None)


def calc_atr(highs, lows, closes, period=14):
    trs = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])) for i in range(1, len(closes))]
    return round(sum(trs[-period:]) / period, 4) if len(trs) >= period else None


def calc_ma(closes, period):
    return round(sum(closes[-period:]) / period, 4) if len(closes) >= period else None


def calc_bollinger(closes, period=20, sd=2):
    if len(closes) < period:
        return None, None, None
    r = closes[-period:]
    ma = sum(r) / period
    std = (sum((x - ma)**2 for x in r) / period) ** 0.5
    return round(ma + sd*std, 4), round(ma, 4), round(ma - sd*std, 4)


def get_technicals(symbol, timeframe='1day'):
    bars = get_ohlcv(symbol, timeframe, limit=200)
    if not bars or len(bars) < 20:
        return {}
    closes = [b['c'] for b in bars if b['c']]
    highs  = [b['h'] for b in bars if b['h']]
    lows   = [b['l'] for b in bars if b['l']]
    vols   = [b['v'] for b in bars if b['v']]
    price  = closes[-1]
    ma20   = calc_ma(closes, 20)
    ma50   = calc_ma(closes, 50)
    ma200  = calc_ma(closes, 200)
    macd, sig, hist = calc_macd(closes)
    atr    = calc_atr(highs, lows, closes)
    bb_u, bb_m, bb_l = calc_bollinger(closes)
    avg_vol   = round(sum(vols[-20:]) / 20) if len(vols) >= 20 else None
    vol_ratio = round(vols[-1] / avg_vol, 2) if avg_vol and vols else None
    return {
        'price': round(price, 4), 'rsi': calc_rsi(closes),
        'ma20': ma20, 'ma50': ma50, 'ma200': ma200,
        'ma_cross': ('golden' if ma20 and ma50 and ma20 > ma50 else 'death'),
        'macd': macd, 'macd_signal': sig, 'macd_hist': hist,
        'atr': atr, 'bb_upper': bb_u, 'bb_mid': bb_m, 'bb_lower': bb_l,
        'vol_ratio': vol_ratio, 'avg_volume': avg_vol,
        'trend_20':  ('up' if price > closes[-20] else 'down'),
        'trend_50':  ('up' if len(closes) >= 50 and price > closes[-50] else 'down'),
        'trend_200': ('up' if ma200 and price > ma200 else 'down'),
        'bars_available': len(bars), 'timeframe': timeframe,
    }


def classify_regime(macro):
    vix   = macro.get('vix', 19)
    tnx   = macro.get('tnx', 4.3)
    dxy   = macro.get('dxy', 103)
    curve = macro.get('curve_2s10s', 0.1)
    vr = 'low_vol' if vix < 15 else 'normal_vol' if vix < 20 else 'elevated_vol' if vix < 30 else 'extreme_vol'
    rr = 'high_rates' if tnx > 4.5 else 'elevated_rates' if tnx > 3.5 else 'low_rates'
    dr = 'strong_dollar' if dxy > 106 else 'neutral_dollar' if dxy > 100 else 'weak_dollar'
    cr = 'deeply_inverted' if curve < -0.5 else 'inverted' if curve < 0 else 'flat' if curve < 0.5 else 'normal'
    risk = 'risk_off' if (vix > 25 or cr == 'deeply_inverted') else 'risk_on' if (vix < 18 and cr in ['flat','normal']) else 'neutral'
    return {
        'label': risk + '_' + vr + '_' + rr,
        'risk_appetite': risk, 'vol_regime': vr, 'rate_regime': rr,
        'dollar_regime': dr, 'curve_regime': cr, 'vix': vix,
        'description': (risk.replace('_',' ').title() + ' | ' + vr.replace('_',' ').title() +
                        ' | ' + rr.replace('_',' ').title() + ' | ' + cr.replace('_',' ').title() + ' curve'),
    }


_cache, _cache_ttl = {}, {}

def cache_get(key):
    return _cache[key] if key in _cache and time.time() < _cache_ttl.get(key, 0) else None

def cache_set(key, val, ttl=60):
    _cache[key] = val
    _cache_ttl[key] = time.time() + ttl


def fetch_live_price(symbol):
    cached = cache_get('px_' + symbol)
    if cached:
        return cached
    info      = INSTRUMENTS.get(symbol, {})
    ref       = REFERENCE_PRICES.get(symbol, 100)
    result    = {'symbol': symbol, 'price': ref, 'change': 0, 'change_pct': 0,
                 'source': 'reference', 'live': False, 'name': info.get('name', symbol)}
    yahoo = info.get('yahoo')
    if yahoo:
        q = yf_get_quote(yahoo)
        if q and q.get('price'):
            result.update(q)
            result['symbol'] = symbol
            result['name']   = info.get('name', symbol)
    cache_set('px_' + symbol, result, ttl=20)
    return result


def fetch_macro_live():
    cached = cache_get('macro')
    if cached:
        return cached
    macro = {
        'vix': REFERENCE_PRICES['VIX'], 'tnx': REFERENCE_PRICES['TNX'],
        'irx': REFERENCE_PRICES['IRX'], 'dxy': REFERENCE_PRICES['DXY'],
        'gold': REFERENCE_PRICES['GC'], 'oil': REFERENCE_PRICES['CL'],
        'fed_funds': '4.25-4.50%', 'source': 'reference', 'live': False,
        'ts': int(time.time()),
    }
    for key, ticker in [('vix','^VIX'),('tnx','^TNX'),('dxy','DX-Y.NYB'),('gold','GC=F'),('oil','CL=F')]:
        try:
            q = yf_get_quote(ticker)
            if q and q.get('price'):
                macro[key]  = round(q['price'], 3)
                macro['live'] = True
                macro['source'] = 'yfinance'
        except Exception:
            pass
    macro['curve_2s10s'] = round(macro['tnx'] - macro['irx'], 3)
    macro['regime']      = classify_regime(macro)
    cache_set('macro', macro, ttl=60)
    return macro


# =============================================================
#  RSS FEED SOURCES — free, real-time, no API key needed
# =============================================================
RSS_FEEDS = [
    # Geopolitical & breaking news
    ('Reuters World',      'geopolitical', 'https://feeds.reuters.com/reuters/worldNews'),
    ('Reuters Business',   'macro',        'https://feeds.reuters.com/reuters/businessNews'),
    ('BBC World',          'geopolitical', 'https://feeds.bbci.co.uk/news/world/rss.xml'),
    ('BBC Business',       'macro',        'https://feeds.bbci.co.uk/news/business/rss.xml'),
    # Markets & Finance
    ('CNBC Markets',       'markets',      'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069'),
    ('CNBC Finance',       'macro',        'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664'),
    ('MarketWatch',        'markets',      'https://feeds.marketwatch.com/marketwatch/topstories/'),
    ('Investing.com',      'markets',      'https://www.investing.com/rss/news.rss'),
    # Macro & Fed
    ('WSJ Economy',        'macro',        'https://feeds.a.dj.com/rss/RSSEconomics.xml'),
    ('FT Markets',         'markets',      'https://www.ft.com/markets?format=rss'),
    # Commodities & Energy
    ('OilPrice.com',       'commodities',  'https://oilprice.com/rss/main'),
    ('Kitco Gold',         'commodities',  'https://www.kitco.com/rss/news.xml'),
    # Geopolitical deep
    ('Al Jazeera',         'geopolitical', 'https://www.aljazeera.com/xml/rss/all.xml'),
    ('Foreign Policy',     'geopolitical', 'https://foreignpolicy.com/feed/'),
]

def fetch_rss_feed(name, category, url):
    """Fetch and parse a single RSS feed — no API key, always free"""
    try:
        r = requests.get(url, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/rss+xml, application/xml, text/xml',
        }, timeout=8)
        if r.status_code != 200:
            return []

        import xml.etree.ElementTree as ET
        # Strip namespaces for easier parsing
        xml_content = r.text
        xml_content = re.sub(r' xmlns[^=]*="[^"]*"', '', xml_content)
        xml_content = re.sub(r'<[a-z]+:', '<', xml_content)
        xml_content = re.sub(r'</[a-z]+:', '</', xml_content)

        root = ET.fromstring(xml_content)
        articles = []

        # Handle both RSS and Atom formats
        items = root.findall('.//item') or root.findall('.//entry')
        for item in items[:8]:
            title = item.findtext('title') or ''
            title = re.sub(r'<[^>]+>', '', title).strip()
            if not title or len(title) < 10:
                continue
            pub = item.findtext('pubDate') or item.findtext('published') or item.findtext('updated') or ''
            link = item.findtext('link') or ''
            desc = item.findtext('description') or item.findtext('summary') or ''
            desc = re.sub(r'<[^>]+>', '', desc).strip()[:200]
            articles.append({
                'headline':    title,
                'source':      name,
                'url':         link,
                'published':   pub,
                'description': desc,
                'category':    category,
            })
        return articles
    except Exception as e:
        logger.debug('RSS failed ' + name + ': ' + str(e))
        return []


def fetch_all_rss():
    """Fetch all RSS feeds in parallel"""
    import concurrent.futures
    articles = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(fetch_rss_feed, name, cat, url): name
                   for name, cat, url in RSS_FEEDS}
        for future in concurrent.futures.as_completed(futures, timeout=15):
            try:
                result = future.result()
                articles.extend(result)
            except Exception as e:
                logger.debug('RSS future failed: ' + str(e))
    return articles


def fetch_news_category(query, count):
    """Fetch from NewsAPI by query"""
    if not NEWS_KEY:
        return []
    try:
        r = requests.get('https://newsapi.org/v2/everything', params={
            'q': query, 'language': 'en', 'sortBy': 'publishedAt',
            'pageSize': count, 'apiKey': NEWS_KEY,
        }, timeout=10)
        return [{'headline': a.get('title',''), 'source': a.get('source',{}).get('name',''),
                 'url': a.get('url',''), 'published': a.get('publishedAt',''),
                 'description': a.get('description','')}
                for a in r.json().get('articles',[]) if a.get('title')]
    except Exception as e:
        logger.warning('NewsAPI error: ' + str(e))
        return []


def get_all_news(symbol='NQ'):
    """
    Pull news from two layers:
    1. RSS feeds — free, real-time, broad geopolitical and macro coverage
    2. NewsAPI — targeted queries by category and instrument
    Combine, deduplicate, and return up to 50 articles.
    """
    cached = cache_get('all_news_' + symbol)
    if cached:
        return cached

    symbol_queries = {
        'NQ': 'nasdaq 100 technology AI semiconductors earnings big tech',
        'ES': 'S&P 500 corporate earnings economic growth outlook',
        'GC': 'gold price safe haven inflation dollar hedge',
        'CL': 'crude oil price OPEC supply production energy',
        'BTC': 'bitcoin price crypto institutional adoption',
        'VIX': 'market volatility fear options hedging',
    }

    all_articles = []

    # Layer 1 — RSS feeds (free, real-time, no limits)
    logger.info('Fetching RSS feeds...')
    rss_articles = fetch_all_rss()
    logger.info('RSS: ' + str(len(rss_articles)) + ' articles from ' + str(len(RSS_FEEDS)) + ' sources')
    all_articles.extend(rss_articles)

    # Layer 2 — NewsAPI targeted queries
    if NEWS_KEY:
        newsapi_categories = [
            ('geopolitical', 'geopolitical risk war conflict sanctions Iran Russia China Middle East military', 8),
            ('macro',        'federal reserve inflation CPI GDP recession unemployment economic outlook', 8),
            ('markets',      'stock market futures nasdaq S&P rally selloff correction', 6),
            ('fed',          'Fed interest rates monetary policy rate hike cut pivot balance sheet', 6),
            ('commodities',  'oil gold energy commodities OPEC supply demand prices', 5),
            ('symbol',       symbol_queries.get(symbol, 'futures market trading'), 6),
        ]
        for cat_name, query, count in newsapi_categories:
            articles = fetch_news_category(query, count)
            for a in articles:
                a['category'] = cat_name
            all_articles.extend(articles)
            time.sleep(0.2)

    # Deduplicate by headline
    seen, unique = set(), []
    for a in all_articles:
        key = a.get('headline', '')[:60].lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(a)

    # Sort by recency — most recent first
    def parse_date(a):
        pub = a.get('published', '')
        try:
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(pub).timestamp()
        except Exception:
            try:
                return datetime.fromisoformat(pub.replace('Z','+00:00')).timestamp()
            except Exception:
                return 0
    unique.sort(key=parse_date, reverse=True)

    result = unique[:50]
    cache_set('all_news_' + symbol, result, ttl=300)
    logger.info('Total unique news articles: ' + str(len(result)))
    return result


def call_anthropic(api_key, prompt, max_tokens=2500):
    r = requests.post(
        'https://api.anthropic.com/v1/messages',
        headers={'Content-Type': 'application/json', 'x-api-key': api_key, 'anthropic-version': '2023-06-01'},
        json={'model': 'claude-sonnet-4-20250514', 'max_tokens': max_tokens,
              'messages': [{'role': 'user', 'content': prompt}]},
        timeout=90
    )
    data = r.json()
    if data.get('error'):
        raise Exception(data['error']['message'])
    text  = data['content'][0]['text'].strip()
    match = re.search(r'\{[\s\S]*\}', text)
    if not match:
        raise Exception('Could not parse JSON from AI response')
    return json.loads(match.group())


@app.route('/')
def dashboard():
    from flask import send_from_directory
    return send_from_directory('.', 'apex_dashboard_v4.html')

@app.route('/api/debug')
def debug():
    import flask
    rules = [str(r) for r in flask.current_app.url_map.iter_rules()]
    return flask.jsonify({'routes': sorted(rules), 'count': len(rules)})


@app.route('/api/status')
def status():
    return jsonify({'status': 'online', 'version': '1.1.0',
                    'polygon': bool(POLYGON_KEY), 'news': bool(NEWS_KEY),
                    'anthropic': bool(ANTHROPIC_KEY), 'db_path': DB_PATH, 'ts': int(time.time())})


@app.route('/api/config', methods=['GET','POST'])
def config_route():
    global POLYGON_KEY, NEWS_KEY, ANTHROPIC_KEY
    if request.method == 'POST':
        data = request.json or {}
        c = load_config()
        if data.get('polygon_key'):   c['polygon_key']   = POLYGON_KEY   = data['polygon_key']
        if data.get('news_key'):      c['news_key']       = NEWS_KEY      = data['news_key']
        if data.get('anthropic_key'): c['anthropic_key']  = ANTHROPIC_KEY = data['anthropic_key']
        save_config(c)
        return jsonify({'ok': True, 'message': 'Configuration saved'})
    return jsonify({
        'polygon_key':   ('****' + POLYGON_KEY[-4:])   if POLYGON_KEY   else '',
        'news_key':      ('****' + NEWS_KEY[-4:])      if NEWS_KEY      else '',
        'anthropic_key': ('****' + ANTHROPIC_KEY[-4:]) if ANTHROPIC_KEY else '',
    })


@app.route('/api/prices')
def prices():
    symbols = request.args.get('symbols', 'NQ,ES,GC,CL,BTC,VIX').split(',')
    return jsonify({s.strip(): fetch_live_price(s.strip()) for s in symbols})


@app.route('/api/price/<symbol>')
def price(symbol):
    return jsonify(fetch_live_price(symbol.upper()))


@app.route('/api/macro')
def macro_route():
    return jsonify(fetch_macro_live())


@app.route('/api/regime')
def regime_route():
    m = fetch_macro_live()
    return jsonify(m.get('regime', classify_regime(m)))


@app.route('/api/chart/<symbol>')
def chart(symbol):
    tf    = request.args.get('tf', '1day')
    limit = int(request.args.get('limit', 300))
    sym   = symbol.upper()
    bars  = get_ohlcv(sym, tf, limit)
    if not bars:
        yahoo = INSTRUMENTS.get(sym, {}).get('yahoo')
        if yahoo:
            raw = yf_fetch(yahoo, tf)
            if raw:
                store_ohlcv(sym, tf, raw)
                bars = get_ohlcv(sym, tf, limit)
    return jsonify({'symbol': sym, 'timeframe': tf, 'bars': bars, 'count': len(bars)})


@app.route('/api/technicals/<symbol>')
def technicals(symbol):
    tf   = request.args.get('tf', '1day')
    data = get_technicals(symbol.upper(), tf)
    if not data:
        data = {'price': REFERENCE_PRICES.get(symbol.upper(), 100), 'source': 'reference'}
    return jsonify(data)


@app.route('/api/multitf/<symbol>')
def multitf(symbol):
    sym = symbol.upper()
    return jsonify({tf: get_technicals(sym, tf) for tf in ['5min','1hour','1day','1week']})


@app.route('/api/news')
def news_route():
    symbol = request.args.get('symbol', 'NQ')
    cached = cache_get('news_' + symbol)
    if cached:
        return jsonify(cached)
    articles = get_all_news(symbol)
    result   = {'articles': articles, 'count': len(articles),
                'source': 'newsapi' if NEWS_KEY else 'unavailable', 'symbol': symbol}
    cache_set('news_' + symbol, result, ttl=600)
    return jsonify(result)


@app.route('/api/news/analysis')
def news_analysis():
    """
    Deep AI analysis of ALL current news events.
    Produces: market impact thesis, predictive outlook, tail risks, trading implications.
    This runs BEFORE the daily brief and feeds into it.
    """
    symbol  = request.args.get('symbol', 'NQ')
    api_key = request.args.get('key', '') or ANTHROPIC_KEY

    if not api_key:
        return jsonify({'error': 'Anthropic API key required'})

    articles   = get_all_news(symbol)
    if not articles:
        return jsonify({'error': 'No news available. Add a NewsAPI key in Setup.'})

    macro      = fetch_macro_live()
    regime     = macro.get('regime', {})
    price_data = fetch_live_price(symbol)
    inst_name  = INSTRUMENTS.get(symbol, {}).get('name', symbol)

    by_cat = {}
    for a in articles:
        cat = a.get('category', 'general')
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append('- ' + a['headline'] + ' [' + a.get('source', '') + ']')

    news_block = ''
    for cat, lines in by_cat.items():
        news_block += '\n[' + cat.upper() + ']\n' + '\n'.join(lines[:6])

    prompt = '\n'.join([
        'You are the head of macro research at a major hedge fund.',
        'Read ALL the news below and produce a deep market impact thesis for ' + symbol + ' (' + inst_name + ').',
        'Think like a professional trader — be specific, directional, and actionable.',
        '',
        'LIVE MARKET CONTEXT:',
        'Price: ' + str(price_data.get('price', 'n/a')),
        'Regime: ' + regime.get('description', 'unknown'),
        'VIX: ' + str(macro.get('vix')) + ' | 10Y: ' + str(macro.get('tnx')) + '% | DXY: ' + str(macro.get('dxy')),
        'Curve 2s/10s: ' + str(macro.get('curve_2s10s')) + '% | Gold: ' + str(macro.get('gold')) + ' | Oil: ' + str(macro.get('oil')),
        '',
        'ALL CURRENT NEWS (last 24 hours):',
        news_block,
        '',
        'TASK:',
        '1. Find the 3 most market-moving events and explain EXACTLY how each impacts ' + symbol,
        '   Geopolitical events: explain the risk-off transmission to equity futures',
        '   Macro data: explain the rate/dollar/liquidity channel',
        '   Sector news: explain the index-level impact and magnitude',
        '2. Write a coherent market narrative synthesising ALL events',
        '3. Give a SPECIFIC predictive outlook for ' + symbol + ' today and this week',
        '   Include: expected direction, key levels, catalysts to watch, conviction level',
        '4. State the biggest tail risk that could blow up the base case',
        '5. State whether news supports or opposes the current technical regime',
        '',
        'Return ONLY a valid JSON object — no markdown, no backticks, no explanation:',
        '{',
        '  "overall_news_bias": "bullish or bearish or mixed",',
        '  "news_conviction": <integer 0-100>,',
        '  "top_events": [',
        '    {"headline": "short event name", "category": "geopolitical or macro or fed or sector or commodity",',
        '     "impact_direction": "bullish or bearish or neutral", "impact_magnitude": "high or medium or low",',
        '     "mechanism": "exact transmission channel to ' + symbol + ' — be specific",',
        '     "timeframe": "intraday or days or weeks"},',
        '    {"headline": "event 2", "category": "cat", "impact_direction": "dir", "impact_magnitude": "mag", "mechanism": "mech", "timeframe": "tf"},',
        '    {"headline": "event 3", "category": "cat", "impact_direction": "dir", "impact_magnitude": "mag", "mechanism": "mech", "timeframe": "tf"}',
        '  ],',
        '  "market_impact_thesis": "4-5 sentences. Deep synthesis of ALL news into a coherent narrative. What is the dominant force? How do geopolitical, macro, and sector themes combine?",',
        '  "predictive_outlook": "4-5 sentences. Specific directional prediction for ' + symbol + ' today and this week. What price action do you expect? What levels matter most? What catalysts could accelerate or reverse the move?",',
        '  "expected_range_today": "e.g. 21200-21600",',
        '  "tail_risks": ["specific tail risk 1 with price implication", "tail risk 2 with scenario", "tail risk 3"],',
        '  "news_vs_technical": "2-3 sentences. Does the news flow confirm or contradict the technical picture? If conflicting, which takes precedence and why?",',
        '  "trading_implication": "2-3 sentences. The single most important thing a trader needs to know. The highest conviction trade idea emerging from this news backdrop."',
        '}',
    ])

    try:
        result = call_anthropic(api_key, prompt, max_tokens=2500)
        result['articles'] = articles
        result['ts']       = int(time.time())
        result['symbol']   = symbol
        cache_set('news_analysis_' + symbol, result, ttl=1800)
        return jsonify(result)
    except Exception as e:
        logger.error('News analysis error: ' + str(e))
        return jsonify({'error': str(e)})


@app.route('/api/db/stats')
def db_stats():
    return jsonify({'stats': get_db_stats()})


@app.route('/api/db/backfill', methods=['POST'])
def db_backfill():
    data   = request.json or {}
    symbol = data.get('symbol', 'NQ').upper()
    years  = int(data.get('years', 5))
    if symbol not in INSTRUMENTS:
        return jsonify({'ok': False, 'message': 'Unknown symbol: ' + symbol})
    threading.Thread(target=backfill_history, args=(symbol, years), daemon=True).start()
    return jsonify({'ok': True, 'message': 'Backfill started for ' + symbol})


@app.route('/api/patterns')
def patterns():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    c    = conn.cursor()
    c.execute('SELECT * FROM patterns WHERE active=1 ORDER BY edge_score DESC')
    cols = [d[0] for d in c.description]
    rows = [dict(zip(cols, r)) for r in c.fetchall()]
    conn.close()
    return jsonify({'patterns': rows, 'count': len(rows)})


@app.route('/api/scan/log')
@app.route('/api/signals')
def scan_log():
    limit = int(request.args.get('limit', 50))
    conn  = sqlite3.connect(DB_PATH, timeout=30)
    c     = conn.cursor()
    c.execute('SELECT * FROM scan_log ORDER BY ts DESC LIMIT ?', (limit,))
    cols = [d[0] for d in c.description]
    rows = [dict(zip(cols, r)) for r in c.fetchall()]
    conn.close()
    return jsonify({'signals': rows, 'count': len(rows)})


@app.route('/api/<path:path>', methods=['OPTIONS'])
def options(path):
    return '', 200



def send_session_open_alert(sess_name, sess_quality, vix, dt_ny):
    try:
        from telegram_alerts import send_telegram, load_telegram_config
        import zoneinfo as _zi
        token, chat_id = load_telegram_config()
        if not token or not chat_id:
            return
        mode     = 'Swing' if 'London' in sess_name else 'Scalp'
        uk_time  = dt_ny.astimezone(_zi.ZoneInfo('Europe/London'))
        time_str = uk_time.strftime('%H:%M')
        day_str  = uk_time.strftime('%A')
        vix_str  = f'{vix:.1f}' if vix else 'N/A'
        vix_ok   = vix and vix <= 25
        vix_emoji= '\u2705' if vix and vix <= 20 else ('\u26a0\ufe0f' if vix_ok else '\U0001f534')
        msg = (
            f'\U0001f50d <b>APEX Scanner Active</b>\n'
            f'\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n'
            f'\U0001f4c5 {day_str} \u00b7 {time_str} UK\n'
            f'\U0001f4ca Session: <b>{sess_name}</b>\n'
            f'\U0001f3af Mode: <b>{mode}</b>\n'
            f'\u2b50 Quality: {sess_quality}/100\n'
            f'{vix_emoji} VIX: {vix_str}\n'
            f'\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n'
            f'Scanning NQ + ES \u00b7 Setups scoring 70+...'
        )
        send_telegram(msg, token, chat_id)
        logger.info(f'Session open alert sent: {sess_name}')
    except Exception as e:
        logger.warning(f'Session open alert failed: {e}')


def background_scheduler():
    logger.info('Background scheduler started')
    last_daily, last_macro_log = time.time(), time.time()
    last_session_alerted = {}
    while True:
        now = time.time()
        if now - last_daily > 86400:
            logger.info('Running daily data update...')
            for sym in INSTRUMENTS:
                try:
                    daily_update(sym)
                except Exception as e:
                    logger.warning('Daily update failed ' + sym + ': ' + str(e))
            last_daily = now

        # ── APEX Data Feed — refresh every 5 minutes ─────────────
        if not hasattr(background_scheduler, '_last_feed'):
            background_scheduler._last_feed    = 0
            background_scheduler._last_htf     = 0
            background_scheduler._last_nq_1min = 0
        # NQ 1min refresh — every 60 seconds for FVG scanner
        if now - background_scheduler._last_nq_1min > 60:
            try:
                from data_feed import refresh_symbol
                refresh_symbol('NQ', ['1min'], lookback_hours=1)
            except Exception as e:
                logger.debug(f'NQ 1min refresh error: {e}')
            background_scheduler._last_nq_1min = now
        # Full refresh — every 5 minutes
        if now - background_scheduler._last_feed > 300:
            try:
                from data_feed import refresh_all
                include_htf = (now - background_scheduler._last_htf) > 1800
                results     = refresh_all(include_htf=include_htf)
                if include_htf:
                    background_scheduler._last_htf = now
                total = sum(v for sym in results.values() for v in sym.values())
                if total > 0:
                    logger.info(f'DataFeed: +{total} new bars')
            except Exception as e:
                logger.warning(f'DataFeed error: {e}')
            background_scheduler._last_feed = now
        if now - last_macro_log > 14400:
            try:
                m    = fetch_macro_live()
                conn = sqlite3.connect(DB_PATH, timeout=30)
                c    = conn.cursor()
                c.execute(
                    'INSERT INTO macro_log (ts,date,vix,tnx,irx,dxy,gold,oil,regime,notes) VALUES (?,?,?,?,?,?,?,?,?,?)',
                    (int(now), datetime.now().strftime('%Y-%m-%d %H:%M'),
                     m.get('vix'), m.get('tnx'), m.get('irx'),
                     m.get('dxy'), m.get('gold'), m.get('oil'),
                     json.dumps(m.get('regime', {})), 'auto')
                )
                conn.commit()
                conn.close()
                logger.info('Macro snapshot logged')
            except Exception as e:
                logger.warning('Macro log failed: ' + str(e))
            last_macro_log = now

        # ── APEX Trade Monitor ───────────────────────────────────
        try:
            from trade_tracker import monitor_trades
            monitor_trades()
        except Exception as e:
            logger.debug(f'Trade monitor error: {e}')

        # ── APEX Session Alerts ──────────────────────────────────
        try:
            check_session_alerts()
        except Exception as e:
            logger.warning(f'Session alert error: {e}')

        # ── APEX Daily P&L Summary — fires at 19:00 UTC ──────────
        try:
            from datetime import datetime, timezone
            _now = datetime.now(timezone.utc)
            if _now.hour == 19 and _now.minute < 5:
                if not hasattr(background_scheduler, '_daily_summary_date') or                    background_scheduler._daily_summary_date != str(_now.date()):
                    background_scheduler._daily_summary_date = str(_now.date())
                    from trade_tracker import get_stats, init_trades_table
                    from live_scanner import send_telegram
                    from zoneinfo import ZoneInfo
                    NY = ZoneInfo('America/New_York')
                    init_trades_table()
                    import sqlite3
                    conn = sqlite3.connect(DB_PATH, timeout=30)
                    today = str(_now.date())
                    trades = conn.execute(
                        'SELECT symbol, direction, setup, pnl_r, exit_reason FROM apex_trades '
                        'WHERE status=? AND entry_time LIKE ?',
                        ('closed', today + '%')
                    ).fetchall()
                    conn.close()
                    stats = get_stats()
                    if trades:
                        wins   = [t for t in trades if t[3] and t[3] > 0]
                        losses = [t for t in trades if t[3] and t[3] <= 0]
                        total_r = round(sum(t[3] for t in trades if t[3]), 2)
                        sign = '+' if total_r >= 0 else ''
                        sep = chr(9473) * 20
                        now_ny = _now.astimezone(NY).strftime('%Y-%m-%d')
                        trade_lines = chr(10).join([
                            f'{"+" if t[3]>0 else ""}{t[3]:.2f}R {t[0]} {t[1].upper()} [{t[2]}]'
                            for t in trades if t[3] is not None
                        ])
                        msg = (
                            chr(128202) + ' <b>APEX Daily Summary</b>' + chr(10) +
                            sep + chr(10) +
                            f'<b>Date:</b> {now_ny}' + chr(10) +
                            f'<b>Trades:</b> {len(trades)} ({len(wins)}W / {len(losses)}L)' + chr(10) +
                            f'<b>Day P&L:</b> {sign}{total_r}R' + chr(10) +
                            f'<b>Total P&L:</b> {stats.get("total_r", 0):+.2f}R' + chr(10) +
                            sep + chr(10) +
                            trade_lines
                        )
                    else:
                        msg = (
                            chr(128202) + ' <b>APEX Daily Summary</b>' + chr(10) +
                            chr(9473)*20 + chr(10) +
                            f'<b>Date:</b> {str(_now.astimezone(NY).date())}' + chr(10) +
                            '<b>Trades:</b> 0 — no signals today' + chr(10) +
                            f'<b>Total P&L:</b> {stats.get("total_r", 0):+.2f}R'
                        )
                    send_telegram(msg)
                    logger.info('Daily P&L summary sent')
        except Exception as e:
            logger.warning(f'Daily summary error: {e}')

        # ── APEX FVG Scanner — runs every minute ─────────────────
        try:
            from fvg_engine import scan_fvg, format_fvg_alert, FVG_PARAMS
            from live_scanner import send_telegram
            from trade_tracker import log_trade
            from datetime import datetime, timezone
            now_utc = datetime.now(timezone.utc)
            _fvg_windows = FVG_PARAMS.get('session_windows', {}).get('NQ', [{'start': 13, 'end': 19}])
            _in_fvg_session = any(w['start'] <= now_utc.hour < w['end'] for w in _fvg_windows)
            if _in_fvg_session:
                fvg_signals = scan_fvg('NQ', now_utc)
                for sig in fvg_signals:
                    msg = format_fvg_alert(sig)
                    send_telegram(msg)
                    log_trade(sig)
                    logger.info(f'FVG signal: NQ {sig["direction"].upper()} entry={sig["entry"]}')
        except Exception as e:
            logger.warning(f'FVG scanner error: {e}')

        # ── APEX Engine v2 — Setup B Scanner ──────────────────────
        try:
            from live_scanner import run_scan, send_telegram, format_alert, SignalTracker
            from trade_tracker import log_trade
            if not hasattr(background_scheduler, '_apex_tracker'):
                background_scheduler._apex_tracker = SignalTracker()
            tracker = background_scheduler._apex_tracker
            signals = run_scan()
            for result in signals:
                if tracker.is_new(result):
                    send_telegram(format_alert(result))
                    log_trade({
                        'symbol':    result.symbol,
                        'direction': result.direction,
                        'setup':     result.setup,
                        'mode':      'swing',
                        'entry':     result.entry,
                        'stop':      result.stop,
                        'target':    result.target,
                        'rr':        result.rr,
                        'session':   getattr(result, 'session', ''),
                        'quality':   result.quality,
                    })
                    logger.info(f'APEX signal: {result.symbol} {result.direction} {result.setup}')
            if not signals:
                logger.debug('APEX scan: no signals')
        except Exception as e:
            logger.debug(f'APEX scanner error: {e}')

        time.sleep(60)


# startup moved to end of file


# =============================================================
#  PHASE 2 — EDGE ENGINE API ROUTES
# =============================================================

@app.route('/api/edge/run', methods=['POST'])
def edge_run():
    """Run the pattern engine on historical data"""
    data      = request.json or {}
    symbol    = data.get('symbol', 'NQ').upper()
    timeframe = data.get('timeframe', '1day')
    def run():
        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(__file__))
            from patterns import run_pattern_engine
            run_pattern_engine(symbol, timeframe)
            logger.info(f'Pattern engine complete: {symbol} {timeframe}')
        except Exception as e:
            logger.error(f'Pattern engine error: {e}')
    threading.Thread(target=run, daemon=True).start()
    return jsonify({'ok': True, 'message': f'Pattern engine started for {symbol} {timeframe}'})


@app.route('/api/edge/backtest', methods=['POST'])
def edge_backtest():
    """Run full backtest simulation"""
    data      = request.json or {}
    symbol    = data.get('symbol', 'NQ').upper()
    timeframe = data.get('timeframe', '1day')
    balance   = float(data.get('balance', 1000))
    risk      = float(data.get('risk', 2.0))
    def run():
        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(__file__))
            from backtest import run_backtest
            run_backtest(symbol, timeframe, balance, risk)
            logger.info(f'Backtest complete: {symbol} {timeframe} ${balance} {risk}%')
        except Exception as e:
            logger.error(f'Backtest error: {e}')
    threading.Thread(target=run, daemon=True).start()
    return jsonify({'ok': True, 'message': f'Backtest started: {symbol} ${balance} {risk}%/trade'})


@app.route('/api/edge/results/<symbol>')
def edge_results(symbol):
    """Return pattern engine results"""
    tf = request.args.get('tf', '1day')
    try:
        with open(f'edge_results_{symbol.upper()}_{tf}.json') as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({'error': 'No results yet. Run the pattern engine first.'})


@app.route('/api/edge/backtest_results/<symbol>')
def backtest_results(symbol):
    """Return backtest simulation results"""
    tf      = request.args.get('tf', '1day')
    balance = request.args.get('balance', '1000')
    try:
        with open(f'backtest_{symbol.upper()}_{tf}_{balance}.json') as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({'error': 'No backtest results yet. Run the backtest first.'})


@app.route('/api/edge/scan/<symbol>')
def edge_scan(symbol):
    """Real-time pattern scan — returns active setups right now"""
    tf = request.args.get('tf', '1day')
    cached = cache_get(f'scan_{symbol}_{tf}')
    if cached:
        return jsonify(cached)
    try:
        from scanner import scan_for_setups, get_market_conditions_summary
        setups    = scan_for_setups(symbol.upper(), tf)
        conditions= get_market_conditions_summary(symbol.upper(), tf)
        result    = {'setups': setups, 'conditions': conditions, 'ts': int(time.time())}
        cache_set(f'scan_{symbol}_{tf}', result, ttl=60)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e), 'setups': [], 'conditions': {}})


@app.route('/api/edge/fvg/<symbol>')
def edge_fvg(symbol):
    """Return open Fair Value Gaps for symbol"""
    tf = request.args.get('tf', '1day')
    cached = cache_get(f'fvg_{symbol}_{tf}')
    if cached:
        return jsonify(cached)
    try:
        from scanner import load_recent_bars, add_indicators, detect_open_fvgs
        df  = load_recent_bars(symbol.upper(), tf)
        if df is None:
            return jsonify({'fvgs': [], 'error': 'No data'})
        df  = add_indicators(df)
        fvgs= detect_open_fvgs(df)
        result = {'fvgs': fvgs, 'symbol': symbol.upper(), 'timeframe': tf, 'ts': int(time.time())}
        cache_set(f'fvg_{symbol}_{tf}', result, ttl=120)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e), 'fvgs': []})


@app.route('/api/edge/sr/<symbol>')
def edge_sr(symbol):
    """Return support and resistance levels"""
    tf = request.args.get('tf', '1day')
    cached = cache_get(f'sr_{symbol}_{tf}')
    if cached:
        return jsonify(cached)
    try:
        from patterns import load_ohlcv, add_indicators, find_sr_levels, find_round_numbers
        df = load_ohlcv(symbol.upper(), tf)
        if df is None:
            return jsonify({'levels': [], 'error': 'No data'})
        df = add_indicators(df)
        sr = find_sr_levels(df)
        rn = find_round_numbers(float(df['close'].iloc[-1]))
        all_levels = sr + [l for l in rn if not any(abs(l['price']-s['price'])/s['price'] < 0.002 for s in sr)]
        all_levels.sort(key=lambda x: abs(x['dist_pct']))
        result = {'levels': all_levels[:12], 'symbol': symbol.upper(), 'timeframe': tf, 'ts': int(time.time())}
        cache_set(f'sr_{symbol}_{tf}', result, ttl=300)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e), 'levels': []})



# =============================================================
#  MTF ENGINE API ROUTES
# =============================================================

@app.route('/api/mtf/analysis/<symbol>')
def mtf_analysis(symbol):
    """Full multi-timeframe confluence analysis"""
    cached = cache_get('mtf_' + symbol)
    if cached:
        return jsonify(cached)
    try:
        from mtf_engine import load_all_timeframes, add_indicators, get_mtf_confluence
        dfs, available = load_all_timeframes(symbol.upper())
        if not dfs:
            return jsonify({'error': 'No data. Run backfill for all timeframes first.'})
        mtf_dfs = {k: add_indicators(v.copy()) for k, v in dfs.items()}
        result  = get_mtf_confluence(mtf_dfs, available)
        result['symbol']    = symbol.upper()
        result['timestamp'] = int(time.time())
        result['available_timeframes'] = available
        cache_set('mtf_' + symbol, result, ttl=120)
        return jsonify(result)
    except Exception as e:
        logger.error('MTF analysis error: ' + str(e))
        return jsonify({'error': str(e)})


@app.route('/api/mtf/scan/<symbol>')
def mtf_scan(symbol):
    """Live MTF setup scanner"""
    cached = cache_get('mtf_scan_' + symbol)
    if cached:
        return jsonify(cached)
    try:
        from mtf_engine import load_all_timeframes, add_indicators, scan_live
        dfs, available = load_all_timeframes(symbol.upper())
        if not dfs:
            return jsonify({'error': 'No data', 'setups': []})
        result = scan_live(symbol.upper(), dfs, available)
        result['symbol']    = symbol.upper()
        result['timestamp'] = int(time.time())
        cache_set('mtf_scan_' + symbol, result, ttl=60)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e), 'setups': []})


@app.route('/api/mtf/backtest', methods=['POST'])
def mtf_backtest():
    """Run full MTF backtest simulation"""
    data    = request.json or {}
    symbol  = data.get('symbol', 'NQ').upper()
    balance = float(data.get('balance', 1000))
    risk    = float(data.get('risk', 2.0))
    mc      = int(data.get('mc', 1000))
    def run():
        try:
            from mtf_engine import run_mtf_engine
            run_mtf_engine(symbol, balance, risk, do_backtest=True, n_monte_carlo=mc)
            logger.info('MTF backtest complete: ' + symbol)
        except Exception as e:
            logger.error('MTF backtest error: ' + str(e))
    threading.Thread(target=run, daemon=True).start()
    return jsonify({'ok': True, 'message': 'MTF backtest started for ' + symbol})


@app.route('/api/mtf/results/<symbol>')
def mtf_results(symbol):
    """Return MTF backtest results"""
    try:
        with open('mtf_results_' + symbol.upper() + '.json') as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({'error': 'No results yet. Run MTF backtest first.'})


@app.route('/api/mtf/backfill_all', methods=['POST'])
def mtf_backfill_all():
    """Backfill all timeframes for a symbol"""
    data   = request.json or {}
    symbol = data.get('symbol', 'NQ').upper()
    def run():
        all_tfs = ['1week', '1day', '4hour', '1hour', '15min', '5min']
        for tf in all_tfs:
            try:
                logger.info('Backfilling ' + symbol + ' ' + tf + '...')
                backfill_history(symbol)
                time.sleep(3)
            except Exception as e:
                logger.error('Backfill error ' + symbol + ' ' + tf + ': ' + str(e))
    threading.Thread(target=run, daemon=True).start()
    return jsonify({'ok': True, 'message': 'Full backfill started for ' + symbol + ' (all timeframes)'})



# =============================================================
#  DEEP EDGE + PAPER TRADING + TELEGRAM API ROUTES
# =============================================================

@app.route('/api/deep/scan/<symbol>')
def deep_scan(symbol):
    cached = cache_get('deep_' + symbol)
    if cached:
        return jsonify(cached)
    try:
        from deep_edge import run_live_scan
        result = run_live_scan(symbol.upper(), min_score=55,
                               alert=True, paper_trade=True)
        if result:
            cache_set('deep_' + symbol, result, ttl=60)
        return jsonify(result or {'message': 'No qualifying setup found'})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/deep/backtest', methods=['POST'])
def deep_backtest():
    data    = request.json or {}
    symbol  = data.get('symbol', 'NQ').upper()
    balance = float(data.get('balance', 10000))
    risk    = float(data.get('risk', 2.0))
    min_sc  = float(data.get('min_score', 55))
    def run():
        try:
            from deep_edge import run_deep_backtest
            run_deep_backtest(symbol, balance, risk, min_sc)
            logger.info('Deep backtest complete: ' + symbol)
        except Exception as e:
            logger.error('Deep backtest error: ' + str(e))
    threading.Thread(target=run, daemon=True).start()
    return jsonify({'ok': True, 'message': 'Deep backtest started'})


@app.route('/api/deep/results/<symbol>')
def deep_results(symbol):
    try:
        with open('deep_edge_' + symbol.upper() + '.json') as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({'error': 'No results yet'})


# Paper Trading
@app.route('/api/paper/account')
def paper_account():
    try:
        from paper_trader import get_performance_report, init_paper_db
        init_paper_db()
        return jsonify(get_performance_report())
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/paper/open', methods=['POST'])
def paper_open():
    data = request.json or {}
    try:
        from paper_trader import open_position, init_paper_db
        init_paper_db()
        result = open_position(
            symbol      = data.get('symbol','NQ'),
            direction   = data.get('direction','long'),
            entry_price = float(data.get('entry', 0)),
            stop        = float(data.get('stop', 0)),
            target1     = float(data.get('target1', 0)),
            target2     = float(data.get('target2', 0)),
            setup_name  = data.get('setup', ''),
            setup_score = int(data.get('score', 0)),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/paper/close/<int:position_id>', methods=['POST'])
def paper_close(position_id):
    data  = request.json or {}
    price = float(data.get('price', 0))
    try:
        from paper_trader import close_position
        close_position(position_id, price, 'manual')
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/paper/update', methods=['POST'])
def paper_update():
    """Update all open positions with current prices"""
    try:
        from paper_trader import get_account_state, update_position_price
        state = get_account_state()
        results = []
        for pos in state['open_positions']:
            sym   = pos['symbol']
            price = get_latest_price(sym)
            if price:
                trigger = update_position_price(pos['id'], price)
                if trigger:
                    results.append({'id': pos['id'], 'trigger': trigger})
        return jsonify({'ok': True, 'triggers': results})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/paper/init')
def paper_init():
    """Initialise paper account with $10k if not already set"""
    try:
        from paper_trader import init_paper_db, get_account_value, set_account_value
        init_paper_db()
        bal = get_account_value('balance')
        if not bal or float(bal) == 0:
            set_account_value('balance', 10000)
            set_account_value('starting_balance', 10000)
            set_account_value('peak_balance', 10000)
            return jsonify({'ok': True, 'message': 'Account initialised', 'balance': 10000})
        return jsonify({'ok': True, 'message': 'Already initialised', 'balance': float(bal)})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})

@app.route('/api/paper/reset', methods=['POST'])
def paper_reset():
    data    = request.json or {}
    balance = float(data.get('balance', 10000))
    try:
        from paper_trader import set_account_value
        set_account_value('balance', balance)
        set_account_value('starting_balance', balance)
        set_account_value('peak_balance', balance)
        set_account_value('total_trades', 0)
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.execute("UPDATE paper_positions SET status='closed' WHERE status='open'")
        conn.commit()
        conn.close()
        return jsonify({'ok': True, 'balance': balance})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


# Strategy
@app.route('/api/strategy/status', methods=['GET'])
def strategy_status():
    try:
        from strategy_config import get_strategy_summary, is_tradeable_session, STRATEGY
        from datetime import datetime, timezone
        from zoneinfo import ZoneInfo
        dt_ny = datetime.now(timezone.utc).astimezone(ZoneInfo('America/New_York'))
        tradeable, sess_name, quality = is_tradeable_session(dt_ny)
        return jsonify({
            'ok': True,
            'tradeable_now': tradeable,
            'session': sess_name,
            'quality': quality,
            'ny_time': dt_ny.strftime('%H:%M %Z'),
            'day': dt_ny.strftime('%A'),
            'settings': get_strategy_summary(),
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})



# Telegram
@app.route('/api/telegram/test', methods=['GET','POST'])
def telegram_test():
    try:
        from telegram_alerts import test_telegram
        ok, msg = test_telegram()
        return jsonify({'ok': ok, 'message': msg})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/telegram/config', methods=['POST'])
def telegram_config():
    data  = request.json or {}
    token = data.get('token', '').strip()
    chat  = data.get('chat_id', '').strip()
    if not token or not chat:
        return jsonify({'ok': False, 'error': 'Token and chat_id required'})
    try:
        with open('config.json') as f:
            cfg = json.load(f)
        cfg['telegram_token']   = token
        cfg['telegram_chat_id'] = chat
        with open('config.json', 'w') as f:
            json.dump(cfg, f, indent=2)
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})



# =============================================================
#  STARTUP
# =============================================================

def _startup():
    logger.info('=' * 55)
    logger.info('  APEX Trading Engine - Phase 1 v1.1')
    logger.info('=' * 55)
    logger.info('  Polygon API:   ' + ('configured' if POLYGON_KEY   else 'not set'))
    logger.info('  News API:      ' + ('configured' if NEWS_KEY      else 'not set'))
    logger.info('  Anthropic API: ' + ('configured' if ANTHROPIC_KEY else 'not set'))
    logger.info('  Database:      ' + DB_PATH)
    logger.info('=' * 55)
    init_db()
    try:
        from paper_trader import init_paper_db, get_account_value, set_account_value
        init_paper_db()
        bal = get_account_value('balance')
        if not bal or float(bal) == 0:
            set_account_value('balance', 10000)
            set_account_value('starting_balance', 10000)
            set_account_value('peak_balance', 10000)
        logger.info('  Paper account seeded OK')
    except Exception as e:
        logger.warning('  Paper account seed failed: ' + str(e))
    def startup_backfill():
        import time as _t
        _t.sleep(10)
        # Check all Databento-supported instruments, not just ES
        _db_syms = [s for s, info in INSTRUMENTS.items() if info.get('databento')]
        for _sym in _db_syms:
            try:
                import sqlite3 as _sq
                conn = _sq.connect(DB_PATH, timeout=30)
                count = conn.execute(
                    'SELECT COUNT(*) FROM ohlcv WHERE symbol=?', (_sym,)
                ).fetchone()[0]
                conn.close()
                if count < 100:
                    logger.info(f'  {_sym} data missing — running backfill...')
                    backfill_history(_sym, years=2)
                    logger.info(f'  {_sym} backfill complete')
                else:
                    logger.info(f'  {_sym} data OK ({count} bars)')
            except Exception as e:
                logger.warning(f'  {_sym} backfill check failed: ' + str(e))
    threading.Thread(target=startup_backfill, daemon=True).start()
    threading.Thread(target=background_scheduler, daemon=True).start()
    logger.info('  Server running at: http://localhost:5000')
    logger.info('  Open apex_dashboard_v4.html in your browser')
    logger.info('=' * 55)

_startup()


# ─────────────────────────────────────────────────────────────
#  APEX DASHBOARD API ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.route('/api/apex/scan', methods=['GET'])
def apex_scan():
    """Run full gate check and return results."""
    from setup_engine import check_setup, check_setup_a, check_setup_c
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    NY = ZoneInfo('America/New_York')
    now = datetime.now(timezone.utc)
    results = []
    for sym in ('NQ', 'ES', 'GC'):
        for direction in ('long', 'short'):
            for check_fn, setup_name in [
                (lambda s,d: check_setup(s,d,'swing',now), 'B'),
                (lambda s,d: check_setup_a(s,d,'swing',now), 'A'),
                (lambda s,d: check_setup_c(s,d,'swing',now), 'C'),
            ]:
                try:
                    r = check_fn(sym, direction)
                    gates = [{'gate': g.gate, 'name': g.name, 'passed': g.passed, 'detail': g.detail} for g in r.gates]
                    results.append({
                        'symbol':    sym,
                        'direction': direction,
                        'setup':     setup_name,
                        'valid':     r.valid,
                        'gates':     gates,
                        'entry':     r.entry,
                        'stop':      r.stop,
                        'target':    r.target,
                        'rr':        r.rr,
                        'quality':   r.quality,
                        'failed_at': next((g['name'] for g in gates if not g['passed']), None),
                    })
                except Exception as e:
                    pass

    # FVG signals — NQ only
    fvg_signals = []
    try:
        from fvg_engine import scan_fvg
        sigs = scan_fvg('NQ', now)
        for s in sigs:
            fvg_signals.append({
                'symbol':    s['symbol'],
                'direction': s['direction'],
                'setup':     'D',
                'valid':     True,
                'gates':     [{'gate': i+1, 'name': f'Gate {i+1}', 'passed': True, 'detail': ''} for i in range(4)],
                'entry':     s['entry'],
                'stop':      s['stop'],
                'target':    s['target'],
                'rr':        s['rr'],
                'quality':   'primary',
                'fvg_top':   s.get('fvg_top'),
                'fvg_bottom':s.get('fvg_bottom'),
                'bias':      s.get('bias'),
                'failed_at': None,
            })
    except Exception as e:
        pass

    return jsonify({
        'ok':          True,
        'results':     results,
        'fvg_signals': fvg_signals,
        'time':        now.astimezone(NY).strftime('%Y-%m-%d %H:%M ET')
    })


@app.route('/api/apex/trades', methods=['GET'])
def apex_trades():
    """Return open trades and stats."""
    try:
        from trade_tracker import get_open_trades, get_stats
        return jsonify({
            'ok':         True,
            'open_trades': get_open_trades(),
            'stats':      get_stats(),
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/apex/market', methods=['GET'])
def apex_market():
    """Return current market structure per instrument."""
    from market_structure import load_bars, find_swings, detect_structure, compute_bias
    from zoneinfo import ZoneInfo
    NY = ZoneInfo('America/New_York')
    results = {}
    for sym in ('NQ', 'ES', 'GC'):
        try:
            df = load_bars(sym, '1hour', limit=200)
            sh, sl = find_swings(df, lookback=5)
            events, _ = detect_structure(df, sh, sl)
            bias, strength = compute_bias(events)
            last = df.iloc[-1]
            prev = df.iloc[-2]
            chg  = round(float(last['close']) - float(prev['close']), 2)
            pct  = round(chg / float(prev['close']) * 100, 2)
            last_bar_time = df.index[-1].astimezone(NY).strftime('%H:%M ET')
            results[sym] = {
                'bias':      bias,
                'strength':  strength,
                'close':     round(float(last['close']), 2),
                'change':    chg,
                'pct':       pct,
                'last_bar':  last_bar_time,
                'last_event': str(events[-1]) if events else None,
            }
        except Exception as e:
            results[sym] = {'bias': 'unknown', 'error': str(e)}
    return jsonify({'ok': True, 'market': results})


@app.route('/api/apex/candles/<symbol>', methods=['GET'])
def apex_candles(symbol):
    """Return last 100 5min OHLCV bars for lightweight-charts (time, open, high, low, close, volume)."""
    symbol = symbol.upper()
    if symbol not in INSTRUMENTS:
        return jsonify({'ok': False, 'error': 'Unknown symbol'})
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        rows = conn.execute(
            'SELECT ts, open, high, low, close, volume FROM ohlcv '
            'WHERE symbol=? AND timeframe=? ORDER BY ts DESC LIMIT 100',
            (symbol, '5min')
        ).fetchall()
        conn.close()
        bars = [
            {'time': int(r[0]), 'open': float(r[1]), 'high': float(r[2]),
             'low': float(r[3]), 'close': float(r[4]), 'volume': float(r[5])}
            for r in reversed(rows)
        ]
        return jsonify({'ok': True, 'symbol': symbol, 'bars': bars})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/apex/equity', methods=['GET'])
def apex_equity():
    """Return compounding equity curve: $10k start, 1% risk per trade."""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        trades = conn.execute(
            'SELECT entry_time, exit_time, pnl_r FROM apex_trades '
            'WHERE status=? AND pnl_r IS NOT NULL ORDER BY exit_time ASC',
            ('closed',)
        ).fetchall()
        conn.close()

        START = 10000.0
        RISK  = 0.01
        balance = START
        peak    = START
        points  = [{'label': 'Start', 'equity': START, 'drawdown': 0.0}]

        for i, (entry_time, exit_time, pnl_r) in enumerate(trades, 1):
            if pnl_r is None:
                continue
            balance = round(balance + float(pnl_r) * balance * RISK, 2)
            if balance > peak:
                peak = balance
            dd = round((balance - peak) / peak * 100, 2) if peak > 0 else 0.0
            ts_label = (exit_time or entry_time or '')[:10]
            points.append({'label': f'T{i} {ts_label}', 'equity': balance, 'drawdown': dd})

        max_dd = round(min((p['drawdown'] for p in points), default=0.0), 2)
        return jsonify({
            'ok': True,
            'points': points,
            'current_balance': round(balance, 2),
            'max_drawdown': max_dd,
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/apex/telegram_test', methods=['POST'])
def apex_telegram_test():
    """Send a test Telegram message."""
    try:
        from live_scanner import send_telegram
        from datetime import datetime, timezone
        from zoneinfo import ZoneInfo
        NY = ZoneInfo('America/New_York')
        now = datetime.now(timezone.utc).astimezone(NY).strftime('%Y-%m-%d %H:%M')
        result = send_telegram(f'🧪 <b>APEX Test</b>\nScanner is live and connected\n<i>{now} ET</i>')
        return jsonify({'ok': result})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


# ─────────────────────────────────────────────────────────────
#  APEX SESSION ALERTS
# ─────────────────────────────────────────────────────────────

APEX_SESSIONS = [
    {'name': 'London',     'syms': 'NQ/ES', 'start': 7,  'end': 11},
    {'name': 'NY Primary', 'syms': 'NQ/ES', 'start': 13, 'end': 19},
    {'name': 'GC Primary', 'syms': 'GC',    'start': 12, 'end': 17},
]
_session_state = {}

def check_session_alerts():
    try:
        from live_scanner import send_telegram
        from datetime import datetime, timezone
        from zoneinfo import ZoneInfo
        NY = ZoneInfo('America/New_York')
        now = datetime.now(timezone.utc)
        if now.weekday() >= 5:
            return
        hour     = now.hour
        date_str = str(now.date())
        now_ny   = now.astimezone(NY).strftime('%H:%M')
        sep = chr(9473) * 10
        for sess in APEX_SESSIONS:
            key_open  = sess['name'] + '_open_'  + date_str
            key_close = sess['name'] + '_close_' + date_str
            if hour == sess['start'] and not _session_state.get(key_open):
                _session_state[key_open] = True
                parts = [
                    chr(128276) + ' <b>' + sess['name'] + ' Session Open</b>',
                    sep,
                    '<b>Instruments:</b> ' + sess['syms'],
                    '<b>Window:</b> {:02d}:00-{:02d}:00 UTC'.format(sess['start'], sess['end']),
                    '<b>Scanner:</b> Active',
                    '<i>' + now_ny + ' ET</i>',
                ]
                send_telegram(chr(10).join(parts))
                logger.info('Session open alert: ' + sess['name'])
            if hour == sess['end'] and not _session_state.get(key_close):
                _session_state[key_close] = True
                parts = [
                    chr(128277) + ' <b>' + sess['name'] + ' Session Closed</b>',
                    sep,
                    '<b>Instruments:</b> ' + sess['syms'],
                    '<b>Scanner:</b> Paused until next session',
                    '<i>' + now_ny + ' ET</i>',
                ]
                send_telegram(chr(10).join(parts))
                logger.info('Session close alert: ' + sess['name'])
    except Exception as e:
        logger.warning('Session alert error: ' + str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
