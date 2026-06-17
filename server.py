"""
APEX Trading Engine - Phase 1 Backend Server
Provides live market data, news, macro analysis, historical OHLCV, and deep AI news thesis.
Run with: python3 server.py
"""

import os
import re
import json
import time
import logging
import threading
import requests
import db as _db
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
POLYGON_KEY     = os.environ.get('POLYGON_KEY',    cfg.get('polygon_key',   ''))
NEWS_KEY        = os.environ.get('NEWS_KEY',        cfg.get('news_key',      ''))
ANTHROPIC_KEY   = os.environ.get('ANTHROPIC_KEY',  cfg.get('anthropic_key', ''))
TELEGRAM_TOKEN  = os.environ.get('TELEGRAM_TOKEN', cfg.get('telegram_token', ''))
TELEGRAM_CHAT   = os.environ.get('TELEGRAM_CHAT_ID',cfg.get('telegram_chat_id',''))
DB_PATH         = os.environ.get('DB_PATH',        cfg.get('db_path', 'apex_market.db'))
# Write env vars back so other modules (telegram_alerts etc) can read config.json
if TELEGRAM_TOKEN and not cfg.get('telegram_token'):
    cfg['telegram_token']   = TELEGRAM_TOKEN
    cfg['telegram_chat_id'] = TELEGRAM_CHAT
    save_config(cfg)

# ─────────────────────────────────────────────────────────────
#  SIGNAL QUALITY FILTERS — set False to disable individually
# ─────────────────────────────────────────────────────────────
SIGNAL_FILTERS = {
    'max_concurrent_per_instrument': True,   # block if same instrument already has open trade
    'primary_session_only_bcd':      True,   # B/C/D: require quality=='primary' session
    'dual_htf_bias':                 True,   # A/B/C/D/E: 1h EMA20 must align with 4h direction
    'setup_e_min_atr':               True,   # E: skip if 5min ATR14 < minimum (slow markets)
    'economic_calendar':             True,   # block signals in ±30/60 min window around HIGH-impact events
}

# Setup-level enable flags — set to False to fully disable a setup's signal path.
# Checked at the top of each scheduler block before any model/feature work.
setup_f_enabled: bool = True    # Enabled: RF models trained (AUC 0.59/0.61), DB-persisted
SETUP_E_MIN_ATR = {'MNQ': 25.0, 'NQ': 25.0, 'ES': 8.0, 'GC': 5.0}   # pts on 5min chart

# ── Execution limits — configurable via Railway env vars without redeployment ──
DAILY_LOSS_LIMIT    = int(os.environ.get('DAILY_LOSS_LIMIT',    '100'))
MAX_PORTFOLIO_HEAT  = int(os.environ.get('MAX_PORTFOLIO_HEAT',  '3'))   # max concurrent open trades
# Setups that scan, log, and Telegram but never send a Tradovate order.
PAPER_ONLY_SETUPS = set(
    s.strip().upper() for s in
    os.environ.get('PAPER_ONLY_SETUPS', 'E').split(',') if s.strip()
)

# ── Strategy Control Centre — in-memory cache ─────────────────────────────
# Refreshed every 60 s so there is no DB hit on every scan tick.
# None means "not yet loaded"; each value is True=enabled, False=disabled.
_strategy_enabled_cache: dict = {}   # {'A': {'enabled': True, 'optimal_regimes': [...], 'regime_gating_enabled': False}}
_strategy_cache_ts: float = 0.0      # last refresh time (time.time())
_STRATEGY_CACHE_TTL = 60             # seconds

_STRATEGY_DEFAULTS = {
    'A': True, 'B': True, 'C': True, 'D': True,
    'E': False,            # E disabled — confirmed losing record
    'F': True,             # F live — RF models trained (AUC 0.59/0.61), DB-persisted
    'H': True, 'I': True,
    'J': True,             # J Value Area Continuation — promoted from shadow lab
}


def _refresh_strategy_cache():
    """Reload strategy_config from DB into memory. Called at most once per 60 s."""
    global _strategy_enabled_cache, _strategy_cache_ts
    try:
        conn = _db.connect()
        rows = conn.execute(
            "SELECT setup_id, enabled, optimal_regimes, regime_gating_enabled, "
            "       paper_instruments FROM strategy_config"
        ).fetchall()
        conn.close()
        _strategy_cache_ts = time.time()  # always advance — prevents hammering on empty table
        if rows:
            _strategy_enabled_cache = {
                r[0]: {
                    'enabled':               bool(r[1]),
                    'optimal_regimes':       r[2] or '',
                    'regime_gating_enabled': bool(r[3]),
                    'paper_instruments':     r[4] or '',
                }
                for r in rows
            }
    except Exception as _sce:
        _strategy_cache_ts = time.time()  # advance even on failure to prevent per-call hammering
        logger.warning(f'Strategy config: cache refresh failed — {_sce}')


def is_setup_enabled(setup_id: str, symbol: str = None) -> bool:
    """Return True if the setup is enabled in strategy_config.

    When symbol is provided and regime_gating_enabled is set, also checks
    get_current_regime(symbol) against optimal_regimes. Fails open on any
    exception or low-confidence regime — a missing row must never disable a
    live strategy.
    """
    global _strategy_cache_ts
    sid = setup_id.upper().strip()
    try:
        if time.time() - _strategy_cache_ts > _STRATEGY_CACHE_TTL:
            _refresh_strategy_cache()
        if sid in _strategy_enabled_cache:
            row     = _strategy_enabled_cache[sid]
            enabled = row['enabled']
            logger.info(f'Strategy config: Setup {sid} enabled={enabled}')
            if not enabled:
                return False
            # Regime gate — only when symbol provided and gating active
            if row.get('regime_gating_enabled') and symbol:
                try:
                    from regime_engine import get_current_regime
                    regime_info = get_current_regime(symbol)
                    conf = regime_info.get('confidence', 0) if regime_info else 0
                    # Pre-compute optimal regimes — needed for both branches
                    _optimal = [
                        r.strip()
                        for r in (row.get('optimal_regimes') or '').split(',')
                        if r.strip()
                    ]
                    _trending_only = (_optimal == ['TRENDING'])
                    if regime_info and conf >= 0.50:
                        current_regime = regime_info.get('regime', 'UNKNOWN')
                        if _optimal and current_regime not in _optimal:
                            logger.info(
                                f'Setup {sid}: regime gate blocked — '
                                f'current={current_regime} optimal={_optimal}'
                            )
                            return False
                        elif _optimal and current_regime in _optimal:
                            logger.debug(
                                f'Setup {sid}: regime gate passed — '
                                f'{current_regime} in {_optimal}'
                            )
                    elif _trending_only:
                        # Low confidence — only block TRENDING-only setups (A/B/C/I/E).
                        # Multi-regime setups (H=CHOPPY/MR, D/F=TRENDING+CHOPPY, J=all)
                        # fail open: their own signal conditions validate the entry.
                        logger.info(
                            f'Setup {sid}: regime gate blocked — '
                            f'low confidence ({conf:.2f} < 0.50), TRENDING-only setup'
                        )
                        return False
                    else:
                        # Multi-regime setup + low confidence → fail open
                        logger.debug(
                            f'Setup {sid}: regime gate passed — multi-regime fail-open '
                            f'(conf={conf:.2f} < 0.50, optimal={_optimal})'
                        )
                except Exception as _rge:
                    logger.debug(f'Setup {sid}: regime gate error (fail open): {_rge}')
            return True
        # No row in cache — safe default, never block a live strategy
        default = _STRATEGY_DEFAULTS.get(sid, True)
        logger.info(f'Strategy config: Setup {sid} enabled={default} (default — no DB row)')
        return default
    except Exception as _e:
        logger.warning(
            f'Strategy config: is_setup_enabled failed for {sid} — defaulting to ENABLED'
        )
        return True


SETUP_REGIME_CONFIG = {
    'A': ({'TRENDING'},                          True),
    'B': ({'TRENDING'},                          True),
    'C': ({'TRENDING'},                          True),
    'D': ({'TRENDING', 'CHOPPY'},                True),
    'E': ({'TRENDING'},                          True),   # disabled anyway
    'H': ({'CHOPPY', 'MEAN_REVERTING'},          True),
    'I': ({'TRENDING'},                          True),
    'F': ({'TRENDING', 'CHOPPY'},                False),  # RF model is its own gate; no regime_gating needed
    'J': ({'TRENDING', 'CHOPPY', 'MEAN_REVERTING'}, True),  # works in all regimes
}

# ── Instrument-level paper/live defaults ─────────────────────────────────
# Format: {setup_id: set_of_paper_instruments}
# Empty set = all instruments execute live (subject to PAPER_ONLY_SETUPS env).
SETUP_PAPER_INSTRUMENTS: dict = {
    'E': {'MNQ', 'ES'},         # Setup E disabled — all paper
    # H, I, J: MNQ now live — all instruments execute via Tradovate when TRADOVATE_ENABLED=true
}

# Static fallback stop caps (points). Used when ATR data is unavailable.
_MAX_STOP_PTS: dict = {'MNQ': 60, 'ES': 15, 'GC': 8}

# Simple per-symbol ATR cache: (atr_value, computed_at_epoch)
_atr_cache: dict = {}
_ATR_CACHE_TTL = 300  # seconds


def _get_stop_cap(symbol: str) -> int:
    """
    Dynamic stop cap scaled to current ATR:
      MNQ: ATR<45=60pts | ATR 45-60=75pts | ATR>60=90pts
      ES:  ATR<12=15pts | ATR 12-16=20pts  | ATR>16=25pts
      GC:  static 8pts
    Falls back to _MAX_STOP_PTS on any data error.
    """
    static = _MAX_STOP_PTS.get(symbol)
    if symbol not in ('MNQ', 'ES'):
        return static if static is not None else 9999
    now_ts = time.time()
    cached_atr, cached_at = _atr_cache.get(symbol, (None, 0))
    if cached_atr is not None and (now_ts - cached_at) < _ATR_CACHE_TTL:
        atr = cached_atr
    else:
        try:
            from market_structure import load_bars
            import pandas as _pd_atr
            _df = load_bars(symbol, '5min', limit=300)
            if _df.empty or len(_df) < 20:
                return static if static is not None else 9999
            _h  = _df['high'].astype(float)
            _lo = _df['low'].astype(float)
            _pc = _df['close'].astype(float).shift(1)
            _tr = _pd_atr.concat([_h - _lo, (_h - _pc).abs(), (_lo - _pc).abs()], axis=1).max(axis=1)
            atr = float(_tr.rolling(14).mean().iloc[-1])
            _atr_cache[symbol] = (atr, now_ts)
        except Exception:
            return static if static is not None else 9999
    if symbol == 'MNQ':
        if atr < 45:   return 60
        elif atr < 60: return 75
        else:          return 90
    else:  # ES
        if atr < 12:   return 15
        elif atr < 16: return 20
        else:          return 25


def _stop_cap_ok(sig: dict) -> tuple:
    """Return (True, '') or (False, warning_msg) when stop distance exceeds dynamic ATR-based cap."""
    sym = sig.get('symbol', '')
    max_pts = _get_stop_cap(sym)
    if max_pts is None:
        return True, ''
    try:
        dist = abs(float(sig['entry']) - float(sig['stop']))
    except (KeyError, TypeError, ValueError):
        return True, ''
    if dist > max_pts:
        setup = sig.get('setup', '?')
        return False, (
            f'Setup {setup} {sym}: stop={dist:.1f}pts exceeds max {max_pts}pts '
            f'— signal REJECTED (high volatility)'
        )
    return True, ''


def _write_scan_log(symbol: str, pattern_name: str, score, direction: str,
                    entry, stop, target1, target2, outcome: str, notes: str = ''):
    """Write one row to scan_log (SQLite). Silently swallows all errors."""
    try:
        conn = _db.connect()
        conn.execute(
            'INSERT INTO scan_log '
            '(ts, symbol, pattern_name, score, direction, entry, stop, target1, target2, outcome, notes) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (int(time.time()), symbol, pattern_name, score, direction,
             entry, stop, target1, target2, outcome, (notes or '')[:255])
        )
        conn.commit()
        conn.close()
    except Exception as _sl_e:
        logger.debug(f'scan_log write failed: {_sl_e}')


def _seed_strategy_config():
    """Upsert all setups into strategy_config, enforcing correct defaults on every startup."""
    try:
        conn = _db.connect()
        for sid, enabled in _STRATEGY_DEFAULTS.items():
            regime_regimes, regime_gating = SETUP_REGIME_CONFIG.get(sid, (set(), False))
            optimal_str   = ','.join(sorted(regime_regimes)) if regime_regimes else ''
            gating_int    = 1 if regime_gating else 0
            paper_instrs  = ','.join(sorted(SETUP_PAPER_INSTRUMENTS.get(sid, set())))
            if _db.IS_POSTGRES:
                conn.execute(
                    "INSERT INTO strategy_config "
                    "(setup_id, enabled, optimal_regimes, regime_gating_enabled, paper_instruments) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT (setup_id) DO UPDATE SET "
                    "enabled = EXCLUDED.enabled, "
                    "optimal_regimes = EXCLUDED.optimal_regimes, "
                    "regime_gating_enabled = EXCLUDED.regime_gating_enabled, "
                    "paper_instruments = EXCLUDED.paper_instruments",
                    (sid, enabled, optimal_str, gating_int, paper_instrs)
                )
            else:
                conn.execute(
                    "INSERT OR REPLACE INTO strategy_config "
                    "(setup_id, enabled, optimal_regimes, regime_gating_enabled, paper_instruments) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (sid, enabled, optimal_str, gating_int, paper_instrs)
                )
        conn.commit()
        conn.close()
        _refresh_strategy_cache()
        for sid, enabled in _STRATEGY_DEFAULTS.items():
            _, gating   = SETUP_REGIME_CONFIG.get(sid, (set(), False))
            regimes, _  = SETUP_REGIME_CONFIG.get(sid, (set(), False))
            paper_i     = sorted(SETUP_PAPER_INSTRUMENTS.get(sid, set()))
            logger.info(
                f'Strategy config: Setup {sid} seeded enabled={enabled} '
                f'regime_gating={gating} optimal_regimes={sorted(regimes)} '
                f'paper_instruments={paper_i}'
            )
    except Exception as e:
        logger.warning(f'Strategy Control Centre seed failed: {e}')

INSTRUMENTS = {
    'NQ':  {'yahoo': 'NQ=F',     'polygon_paid': 'NQ:CME',   'databento': 'NQ.c.0',  'type': 'future', 'name': 'Nasdaq 100 Futures'},
    'ES':  {'yahoo': 'ES=F',     'polygon_paid': 'ES:CME',   'databento': 'ES.c.0',  'type': 'future', 'name': 'S&P 500 E-Mini'},
    'GC':  {'yahoo': 'GC=F',     'polygon_paid': 'GC:COMEX', 'databento': 'GC.c.0',  'type': 'future', 'name': 'Gold Futures'},
    'MNQ': {'yahoo': None,       'polygon_paid': None,       'databento': 'MNQ.c.0', 'type': 'future', 'name': 'Micro E-mini Nasdaq 100 (data only)'},
    'CL':  {'yahoo': 'CL=F',     'polygon_paid': 'CL:NYMEX', 'databento': None,      'type': 'future', 'name': 'Crude Oil Futures'},
    'ZN':  {'yahoo': 'ZN=F',     'polygon_paid': 'ZN:CBOT',  'databento': None,      'type': 'future', 'name': '10Y T-Note Futures'},
    'VIX': {'yahoo': '^VIX',     'polygon_paid': None,       'databento': None,      'type': 'index',  'name': 'CBOE VIX'},
    'DXY': {'yahoo': 'DX-Y.NYB', 'polygon_paid': None,       'databento': None,      'type': 'forex',  'name': 'Dollar Index'},
    'BTC': {'yahoo': 'BTC-USD',  'polygon_paid': None,       'databento': None,      'type': 'crypto', 'name': 'Bitcoin USD'},
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
    _db.init_schema()   # create all tables in PostgreSQL (no-op for SQLite)
    conn = _db.connect()
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
    # ── Research Division tables (SQLite-compatible) ──────────
    c.execute('''CREATE TABLE IF NOT EXISTS strategy_health_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        setup_id TEXT, week_start TEXT,
        sharpe_30d REAL, sharpe_benchmark REAL,
        win_rate REAL, win_rate_benchmark REAL,
        signal_count_week REAL, expectancy REAL,
        health_score INTEGER, alert_level TEXT, notes TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS shadow_lab (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        strategy_name TEXT, description TEXT,
        entered_date TEXT, week_number INTEGER, total_weeks INTEGER DEFAULT 8,
        paper_sharpe REAL, paper_win_rate REAL,
        paper_total_r REAL, paper_signal_count INTEGER,
        backtest_sharpe REAL, backtest_win_rate REAL,
        status TEXT, promotion_eligible_date TEXT, notes TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS research_decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        decision_type TEXT, subject TEXT,
        recommendation TEXT, supporting_data TEXT,
        status TEXT, decided_at TEXT, outcome TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS backtest_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        setup_id TEXT, lookback_days INTEGER, run_date TEXT,
        total_signals INTEGER, win_rate REAL, sharpe REAL,
        avg_r REAL, expectancy REAL, max_drawdown REAL, profit_factor REAL,
        benchmark_sharpe REAL, benchmark_win_rate REAL,
        sharpe_vs_benchmark REAL, wr_vs_benchmark REAL,
        edge_score INTEGER, bars_analysed INTEGER,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS research_state (
        key TEXT PRIMARY KEY,
        value TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS hypothesis_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        hypothesis_id TEXT UNIQUE NOT NULL,
        description TEXT,
        category TEXT,
        instrument TEXT,
        lookback_days INTEGER,
        signals_generated INTEGER,
        win_rate REAL,
        sharpe REAL,
        avg_r REAL,
        information_coefficient REAL,
        p_value REAL,
        status TEXT DEFAULT 'TESTING',
        run_date TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS feature_combinations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        features TEXT NOT NULL,
        oos_ic REAL,
        oos_auc REAL,
        run_date TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS pattern_library (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pattern_id TEXT UNIQUE NOT NULL,
        name TEXT,
        description TEXT,
        discovery_source TEXT,
        instrument TEXT,
        signals_observed INTEGER DEFAULT 0,
        win_rate REAL,
        sharpe REAL,
        information_coefficient REAL,
        first_observed TEXT,
        last_validated TEXT,
        decay_score REAL DEFAULT 1.0,
        status TEXT DEFAULT 'ACTIVE',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP)''')
    # Migrate: add dual-score columns to strategy_health_log if missing
    for col in ('backtest_score INTEGER', 'live_score INTEGER'):
        try:
            c.execute(f'ALTER TABLE strategy_health_log ADD COLUMN {col}')
        except Exception:
            pass  # already exists
    # Migrate: add reason column to research_decisions if missing
    try:
        c.execute('ALTER TABLE research_decisions ADD COLUMN reason TEXT')
    except Exception:
        pass  # already exists
    conn.commit()
    # strategy_config — isolated commit so any prior aborted tx cannot block it
    try:
        conn.rollback()  # clear any aborted transaction state from above migrations
        c.execute('''CREATE TABLE IF NOT EXISTS strategy_config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            setup_id TEXT UNIQUE NOT NULL,
            enabled INTEGER DEFAULT 1,
            disabled_reason TEXT,
            disabled_at TEXT,
            enabled_at TEXT,
            updated_by TEXT DEFAULT 'dashboard',
            optimal_regimes TEXT,
            regime_gating_enabled INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()
    except Exception as _scfg_e:
        try:
            conn.rollback()
        except Exception:
            pass
        logger.error(f'strategy_config table creation failed (non-fatal): {_scfg_e}')
    # Phase 2.4 — migrate strategy_config: add regime gating columns if missing
    for col in ('optimal_regimes TEXT', 'regime_gating_enabled INTEGER DEFAULT 0',
                'paper_instruments TEXT'):  # 2.6 instrument-level paper/live control
        try:
            c.execute(f'ALTER TABLE strategy_config ADD COLUMN {col}')
            conn.commit()
        except Exception:
            pass  # already exists
    # Phase 2.3 — ml_models table: persist trained ML models across restarts
    try:
        conn.rollback()
        c.execute('''CREATE TABLE IF NOT EXISTS ml_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT UNIQUE NOT NULL,
            model_bytes BLOB NOT NULL,
            trained_at TEXT NOT NULL,
            oos_auc REAL,
            feature_count INTEGER,
            training_samples INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()
    except Exception as _ml_e:
        try:
            conn.rollback()
        except Exception:
            pass
        logger.error(f'ml_models table creation failed (non-fatal): {_ml_e}')
    conn.close()
    logger.info('Database initialised (' + ('PostgreSQL' if _db.IS_POSTGRES else DB_PATH) + ')')
    _migrate_htf_prices()


def _migrate_htf_prices():
    """
    Startup migration: fix ×1000 price bug on 1hour/4hour bars introduced by
    the Databento Python SDK ohlcv-1h path (SDK divides by 1e9; raw price is ×1e12
    so result was ×1000 too large).  Idempotent — only rows with close>100000
    are touched, which is impossible for NQ/ES/GC at correct prices.
    """
    try:
        conn = _db.connect()
        total = 0
        for sym in ('NQ', 'ES'):
            for tf in ('1hour', '4hour'):
                cur = conn.execute(
                    'UPDATE ohlcv SET open=open/1000, high=high/1000, low=low/1000, close=close/1000 '
                    'WHERE symbol=? AND timeframe=? AND close>100000',
                    (sym, tf)
                )
                if cur.rowcount:
                    logger.warning(f'_migrate_htf_prices: fixed {cur.rowcount} rows for {sym} {tf}')
                    total += cur.rowcount
        # GC: 1hour only (4hour correct — gold ~$3000-6000, threshold safe at 100000)
        cur = conn.execute(
            'UPDATE ohlcv SET open=open/1000, high=high/1000, low=low/1000, close=close/1000 '
            'WHERE symbol=? AND timeframe=? AND close>100000',
            ('GC', '1hour')
        )
        if cur.rowcount:
            logger.warning(f'_migrate_htf_prices: fixed {cur.rowcount} rows for GC 1hour')
            total += cur.rowcount
        conn.commit()
        conn.close()
        if total:
            logger.warning(f'_migrate_htf_prices: fixed {total} rows total')
        else:
            logger.info('_migrate_htf_prices: no bad rows found (already clean)')
    except Exception as e:
        logger.error(f'_migrate_htf_prices error: {e}')


def store_ohlcv(symbol, timeframe, bars):
    if not bars:
        return 0
    conn = _db.connect()
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
    conn = _db.connect()
    c = conn.cursor()
    c.execute(
        'SELECT ts,open,high,low,close,volume FROM ohlcv WHERE symbol=? AND timeframe=? ORDER BY ts DESC LIMIT ?',
        (symbol, timeframe, limit)
    )
    rows = c.fetchall()
    conn.close()
    return [{'t': r[0], 'o': r[1], 'h': r[2], 'l': r[3], 'c': r[4], 'v': r[5]} for r in reversed(rows)]


def get_db_stats():
    conn = _db.connect()
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
    return send_from_directory('.', 'apex_dashboard_v8.html')

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


@app.route('/api/db/refresh_intraday', methods=['POST'])
def db_refresh_intraday():
    """Directly invoke build_intraday_from_1min for diagnostics. Returns detailed result."""
    try:
        from data_feed import build_intraday_from_1min
        import pandas as _pd
        from db import connect as _dbc, read_sql as _drs

        symbol = (request.json or {}).get('symbol', 'NQ').upper()

        # Mirror what build_intraday_from_1min does, step by step
        conn = _dbc()
        df = _drs(
            'SELECT ts, open, high, low, close, volume FROM ohlcv '
            'WHERE symbol=? AND timeframe=? ORDER BY ts DESC LIMIT 1000',
            conn, params=(symbol, '1min')
        )
        conn.close()

        diag = {
            'symbol': symbol,
            '1min_rows_loaded': len(df),
            '1min_empty': df.empty,
        }

        if not df.empty:
            df = df.sort_values('ts')
            df['dt'] = _pd.to_datetime(df['ts'], unit='s', utc=True)
            df.set_index('dt', inplace=True)
            diag['1min_first_ts'] = int(df.index[0].timestamp())
            diag['1min_last_ts']  = int(df.index[-1].timestamp())

            for tf, rule, lookback in [('5min', '5min', 500), ('15min', '15min', 1000)]:
                src = df.tail(lookback)
                agg = src.resample(rule).agg({
                    'open': 'first', 'high': 'max', 'low': 'min',
                    'close': 'last', 'volume': 'sum',
                }).dropna()
                agg['ts'] = agg.index.map(lambda x: int(x.timestamp()))

                bars = [(int(r['ts']), float(r['open']), float(r['high']),
                         float(r['low']), float(r['close']), float(r['volume']))
                        for _, r in agg.iterrows()]

                conn2 = _dbc()
                last_row = conn2.execute(
                    'SELECT MAX(ts) FROM ohlcv WHERE symbol=? AND timeframe=?',
                    (symbol, tf)
                ).fetchone()
                conn2.close()
                last_stored_ts = int(last_row[0]) if last_row and last_row[0] else 0

                bars_to_write = [b for b in bars if b[0] >= last_stored_ts]

                diag[tf] = {
                    'resampled_bars': len(bars),
                    'last_stored_ts': last_stored_ts,
                    'bars_to_write': len(bars_to_write),
                    'first_bar_ts': bars[0][0] if bars else None,
                    'last_bar_ts':  bars[-1][0] if bars else None,
                    'first_to_write_ts': bars_to_write[0][0] if bars_to_write else None,
                }

        # Now actually run it
        result = build_intraday_from_1min(symbol)
        diag['build_result'] = result

        return jsonify({'ok': True, 'diag': diag})
    except Exception as e:
        import traceback
        return jsonify({'ok': False, 'error': str(e), 'trace': traceback.format_exc()})


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
    conn = _db.connect()
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
    conn  = _db.connect()
    c     = conn.cursor()
    c.execute('SELECT * FROM scan_log ORDER BY ts DESC LIMIT ?', (limit,))
    cols = [d[0] for d in c.description]
    rows = [dict(zip(cols, r)) for r in c.fetchall()]
    conn.close()
    return jsonify({'signals': rows, 'count': len(rows)})


@app.route('/api/<path:path>', methods=['OPTIONS'])
def options(path):
    return '', 200



def _execute_via_tradovate(signal: dict, trade_id: int):
    """
    Fire-and-forget Tradovate execution after a signal is logged.
    Only runs when TRADOVATE_ENABLED=true. Silently skips otherwise.
    Updates broker_order_id on the apex_trades row if execution succeeds.
    """
    # SAFETY: block all order placement in demo/test mode.
    # TRADOVATE_DEMO=true → local test account. APEX_TESTING=true → test suite running.
    if (os.environ.get('TRADOVATE_DEMO', 'false').lower() != 'false' or
            os.environ.get('APEX_TESTING', 'false').lower() == 'true'):
        logger.info('Tradovate: execution blocked — test/demo mode')
        return None
    try:
        from tradovate import execute_apex_signal, TRADOVATE_ENABLED, TRADING_ENABLED
        logger.info(
            f'_execute_via_tradovate: CALLED — {signal.get("symbol")} {signal.get("direction")} '
            f'{signal.get("setup")} trade_id={trade_id} '
            f'TRADOVATE_ENABLED={TRADOVATE_ENABLED} TRADING_ENABLED={TRADING_ENABLED} '
            f'kill_switch={_kill_switch_active}'
        )
        if not TRADOVATE_ENABLED:
            logger.warning('_execute_via_tradovate: TRADOVATE_ENABLED=false — skipping (set TRADOVATE_ENABLED=true in Railway env to enable live execution)')
            return
        if not TRADING_ENABLED:
            logger.warning('_execute_via_tradovate: TRADING_ENABLED=false — order blocked')
            return
        if _kill_switch_active:
            logger.warning(
                f'_execute_via_tradovate: kill switch active (balance=${_kill_switch_balance:.0f}) — order blocked'
            )
            return
        logger.info(f'_execute_via_tradovate: calling execute_apex_signal for {signal.get("symbol")} {signal.get("direction")}')
        result = execute_apex_signal(signal)
        if result['ok'] and trade_id:
            try:
                conn = _db.connect()
                conn.execute(
                    'UPDATE apex_trades SET broker_order_id=?, contracts=? WHERE id=?',
                    (result['order_id'], result.get('contracts', 1), trade_id)
                )
                conn.commit()
                conn.close()
            except Exception as _dbe:
                logger.warning(f'broker_order_id update failed: {_dbe}')
            # Send execution confirmation via Telegram
            try:
                from live_scanner import send_telegram
                send_telegram(
                    f'📋 <b>Order placed</b> — {signal.get("symbol")} {signal.get("direction","").upper()}\n'
                    f'Contracts: {result["contracts"]} {result["instrument"]}\n'
                    f'Fill: {result["fill_price"]:.2f} | Risk: ${result["dollar_risk"]:.0f}'
                )
            except Exception:
                pass
        elif not result['ok']:
            logger.warning(
                f'Tradovate execution skipped: {result.get("skipped_reason")} — {result.get("detail","")}'
            )
    except Exception as e:
        logger.error(f'_execute_via_tradovate EXCEPTION: {e}', exc_info=True)


# Module-level in-memory dedup set — survives DB write failures, resets on redeploy.
# Keys: "{symbol}_{direction}_{setup}"
_active_signals: set = set()

# Per-day dedup dict — primary signal gate for ALL setups.
# Key: f"{symbol}_{setup}_{direction}_{utc_date}" — expires naturally at midnight UTC.
# Marked BEFORE send_telegram so even if log_trade fails, signal stays blocked all day.
_fired_today: dict = {}

# Kill switch — set to True when account balance drops to/below tier threshold.
# Cleared automatically when balance recovers, or by KILL_SWITCH_OVERRIDE=true.
_kill_switch_active:       bool  = False
_kill_switch_balance:      float = 0.0
_kill_switch_threshold:    float = 400.0


def _cal_block(symbol: str, setup: str) -> bool:
    """
    Return True (and log) if economic calendar blackout is active.
    Called before every signal fires. Single gate — protects all setups.
    """
    if not SIGNAL_FILTERS.get('economic_calendar', True):
        return False

    _now_utc = datetime.now(timezone.utc)

    # NY-open soft blackout 13:00–14:05 UTC (11.6% TRENDING probability — worst hour of day).
    # Applies to momentum setups A/B/C/D/I only. H and J are unaffected.
    _setup_letter = (setup or '')[:1].upper()
    if _setup_letter in ('A', 'B', 'C', 'D', 'I'):
        if _now_utc.hour == 13 or (_now_utc.hour == 14 and _now_utc.minute <= 5):
            logger.info(f'{setup} {symbol}: skipped — NY open soft blackout 13:00-14:05 UTC')
            return True

    try:
        from calendar_filter import get_filter as _gcf
        _cf = _gcf()
        _blocked, _reason = _cf.is_blocked(symbol, _now_utc)
        if _blocked:
            logger.warning(f'{setup} {symbol}: skipped — economic calendar blackout: {_reason}')
        return _blocked
    except Exception as _e:
        logger.debug(f'_cal_block check failed (allowing signal): {_e}')
        return False


def _is_signal_already_active(symbol: str, direction: str, setup: str) -> bool:
    """
    Return True if this signal is already active.
    Checks in-memory set FIRST — never auto-clears it (only explicit reconciliation clears it).
    DB check is secondary: used for restart recovery (re-populates set from DB).
    """
    key = f'{symbol}_{direction}_{setup}'
    # Fast path: in-memory set — if key is here, signal is active (trust it)
    if key in _active_signals:
        logger.debug(f'Signal dedup (mem): {symbol} {direction} {setup}')
        return True
    # DB check (persistent, survives redeploys)
    try:
        conn = _db.connect()
        row  = conn.execute(
            "SELECT id FROM apex_trades WHERE symbol=? AND direction=? AND setup=? AND status='open' AND quality != 'test' LIMIT 1",
            (symbol, direction, setup)
        ).fetchone()
        conn.close()
        if row:
            _active_signals.add(key)  # re-populate set from DB after redeploy
            logger.debug(f'Signal dedup (db): {symbol} {direction} {setup} — trade #{row[0]} already open')
            return True
        return False
    except Exception:
        return False


def _check_and_mark_fired(symbol: str, setup: str, direction: str) -> bool:
    """
    Primary signal dedup — returns True (already fired today, skip) or
    False (new signal — marks as fired immediately so caller can send telegram).
    Key includes UTC date so dict auto-expires at midnight.
    Marked BEFORE send_telegram: if log_trade fails, signal stays blocked all day.
    """
    key = f'{symbol}_{setup}_{direction}_{datetime.now(timezone.utc).date()}'
    if key in _fired_today:
        logger.debug(f'Dedup: {key} already fired today')
        return True
    _fired_today[key] = True
    return False


def _has_opposite_swing_trade(symbol: str, direction: str) -> bool:
    """Return True if an open swing trade (A/B/C) exists on symbol in the OPPOSITE direction.
    Used to block FVG and Setup E signals that conflict with an existing position."""
    try:
        opp = 'short' if direction == 'long' else 'long'
        conn = _db.connect()
        row = conn.execute(
            "SELECT id FROM apex_trades "
            "WHERE symbol=? AND direction=? AND setup NOT LIKE 'FVG%' AND status='open' AND quality != 'test' LIMIT 1",
            (symbol, opp)
        ).fetchone()
        conn.close()
        return row is not None
    except Exception:
        return False


def _has_open_trade_on_instrument(symbol: str) -> tuple:
    """Return (True, trade_id) if ANY open trade exists on symbol regardless of direction/setup."""
    try:
        conn = _db.connect()
        row = conn.execute(
            "SELECT id FROM apex_trades WHERE symbol=? AND status='open' AND quality != 'test' LIMIT 1",
            (symbol,)
        ).fetchone()
        conn.close()
        return (True, row[0]) if row else (False, None)
    except Exception:
        return (False, None)


def _count_open_trades() -> int:
    """Return total number of open trades across all setups and instruments.
    Used to enforce MAX_PORTFOLIO_HEAT. Fails open (returns 0) on any error."""
    try:
        conn = _db.connect()
        n = conn.execute(
            "SELECT COUNT(*) FROM apex_trades WHERE status='open' AND quality != 'test'"
        ).fetchone()[0]
        conn.close()
        return int(n)
    except Exception:
        return 0


def _check_1h_bias(symbol: str, direction: str) -> tuple:
    """
    1h EMA20 bias check (in-memory resample from 5min bars — same pattern as gate1_htf_bias).
    Returns (aligned: bool, detail: str). Fails open on data errors — allowed through.
    BULLISH: 1h close > EMA20 × 1.001  |  BEARISH: 1h close < EMA20 × 0.999
    """
    try:
        from market_structure import load_bars
        import pandas as _pd
        df5 = load_bars(symbol, '5min', limit=1500)
        if df5.empty or len(df5) < 25:
            return (True, f'1h bias: no data ({len(df5)} bars) — allowed')
        df1h = df5['close'].resample('1h').last().dropna().to_frame('close')
        if len(df1h) < 21:
            return (True, f'1h bias: only {len(df1h)} 1h bars — allowed')
        ema20      = df1h['close'].ewm(span=20, adjust=False).mean()
        last_close = float(df1h['close'].iloc[-1])
        last_ema   = float(ema20.iloc[-1])
        if last_close > last_ema * 1.001:
            bias_1h = 'bullish'
        elif last_close < last_ema * 0.999:
            bias_1h = 'bearish'
        else:
            bias_1h = 'neutral'
        aligned = (
            (direction == 'long'  and bias_1h == 'bullish') or
            (direction == 'short' and bias_1h == 'bearish')
        )
        return (aligned, f'1h bias={bias_1h.upper()} (close={last_close:.2f} EMA20={last_ema:.2f})')
    except Exception as _e:
        return (True, f'1h bias error: {_e} — allowed')


def background_scheduler():
    logger.info('Background scheduler started')
    logger.info('Regime confidence threshold: 0.50 (non-J setups fail closed below threshold)')
    logger.info('NY open blackout: 13:00-14:05 UTC active for momentum setups A/B/C/D/I')
    # Load Meridian L3 models on startup
    try:
        import meridian_l3 as _ml3_init
        for _l3_sym in ('MNQ', 'ES'):
            _l3_loaded = _ml3_init.load_model(_l3_sym)
            _l3_info = _ml3_init._l3_cache.get(_l3_sym, {})
            if _l3_loaded:
                logger.info(
                    f'Meridian L3: loaded {_l3_sym} '
                    f'(AUC={_l3_info.get("auc", 0):.4f} '
                    f'mode={_l3_info.get("mode", "?")} '
                    f'trained={_l3_info.get("trained_at", "?")})'
                )
            else:
                logger.warning(f'Meridian L3: no model for {_l3_sym} — position sizing uses 1.0×')
    except Exception as _ml3_startup_e:
        logger.warning(f'Meridian L3: startup load failed — {_ml3_startup_e}')
    last_daily, last_macro_log = time.time(), time.time()
    last_session_alerted = {}
    _tick = 0
    while True:
        now   = time.time()
        _tick += 1

        # ── Heartbeat — log every 5 min so Railway logs show liveness ──
        if not hasattr(background_scheduler, '_last_heartbeat') or \
                now - background_scheduler._last_heartbeat >= 300:
            background_scheduler._last_heartbeat = now
            _utc_hr = datetime.now(timezone.utc).hour
            _in_sess = 13 <= _utc_hr < 19
            logger.info(
                f'Scheduler heartbeat tick={_tick} '
                f'utc_hour={_utc_hr} in_session={_in_sess}'
            )

        # ── Strategy Control Centre — refresh cache every 60 s ───────
        _refresh_strategy_cache()

        # ── Kill switch — check account balance every 5 min ─────────
        if not hasattr(background_scheduler, '_last_kill_check') or \
                now - background_scheduler._last_kill_check >= 300:
            background_scheduler._last_kill_check = now
            try:
                global _kill_switch_active, _kill_switch_balance, _kill_switch_threshold
                _ks_balance = None
                try:
                    from tradovate import TRADOVATE_ENABLED as _TV_EN, get_account as _tv_acct
                    if _TV_EN:
                        _acct_r = _tv_acct()
                        if _acct_r.get('ok'):
                            _ks_balance = _acct_r['balance']
                except Exception:
                    pass
                if _ks_balance is None:
                    try:
                        from paper_trader import get_account_value as _gav
                        _ks_balance = float(_gav('balance') or 10000)
                    except Exception:
                        _ks_balance = 10000.0

                from tradovate import get_risk_tier as _get_tier
                _ks_tier = _get_tier(_ks_balance)
                _ks_threshold = _ks_tier['kill_switch']
                _kill_switch_threshold = _ks_threshold

                _ks_override = os.environ.get('KILL_SWITCH_OVERRIDE', 'false').lower() == 'true'

                if _ks_balance <= _ks_threshold and not _ks_override:
                    if not _kill_switch_active:
                        _kill_switch_active   = True
                        _kill_switch_balance  = _ks_balance
                        logger.critical(
                            f'Kill switch triggered — account ${_ks_balance:.0f} <= ${_ks_threshold} threshold'
                        )
                        try:
                            from live_scanner import send_telegram as _ks_tg
                            _ks_tg(
                                f'🛑 <b>APEX KILL SWITCH</b> — Account at ${_ks_balance:.0f}, '
                                f'below ${_ks_threshold} threshold. All trading halted.'
                            )
                        except Exception:
                            pass
                else:
                    if _kill_switch_active:
                        _kill_switch_active = False
                        logger.info(
                            f'Kill switch cleared — balance ${_ks_balance:.0f} > ${_ks_threshold}'
                            + (' (KILL_SWITCH_OVERRIDE)' if _ks_override else '')
                        )
            except Exception as _ks_e:
                logger.warning(f'Kill switch check failed: {_ks_e}')

        # ── Economic calendar refresh (every 6 hours) + 15-min warnings ─
        if not hasattr(background_scheduler, '_last_cal_refresh') or \
                now - background_scheduler._last_cal_refresh >= 21600:
            background_scheduler._last_cal_refresh = now
            try:
                from calendar_filter import get_filter as _gcf_r
                _gcf_r().refresh_calendar()
            except Exception as _cre:
                logger.warning(f'Calendar refresh failed: {_cre}')

        # 15-min pre-event Telegram warnings (checked every scheduler tick)
        try:
            from calendar_filter import get_filter as _gcf_w
            from live_scanner import send_telegram as _cal_tg
            _cf_w = _gcf_w()
            for _warn_ev in _cf_w.check_warnings():
                _cal_tg(
                    f'⚠️ <b>APEX Calendar Alert</b> — {_warn_ev["name"]} in 15 minutes '
                    f'({_warn_ev["utc_time"]}). All signals blocked until {_warn_ev["block_end"]}.'
                )
                logger.info(f'Calendar warning sent: {_warn_ev["name"]}')
            for _lifted_ev in _cf_w.check_blackout_lifted():
                _cal_tg(f'✅ <b>APEX Calendar</b> — Blackout lifted ({_lifted_ev}). Normal trading resumed.')
                logger.info(f'Calendar blackout lifted: {_lifted_ev}')
        except Exception as _cwe:
            logger.debug(f'Calendar warning check failed: {_cwe}')

        # ── Clear _fired_today at midnight UTC ────────────────────
        _today_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        if not hasattr(background_scheduler, '_fired_today_date') or \
                background_scheduler._fired_today_date != _today_utc:
            _fired_today.clear()
            background_scheduler._fired_today_date = _today_utc
            logger.info(f'_fired_today reset for new day: {_today_utc}')

        # ── Research Division — daily backtest (02:00 UTC weekdays) ──
        _now_bt = datetime.now(timezone.utc)
        if _now_bt.weekday() < 5 and _now_bt.hour == 2:
            if not hasattr(background_scheduler, '_last_backtest_day') or \
                    background_scheduler._last_backtest_day != _today_utc:
                background_scheduler._last_backtest_day = _today_utc
                try:
                    import research_division as _rd_mod
                    _rd_mod.run_daily_backtest()
                    logger.info('Research Division: daily backtest complete')
                except Exception as _bte:
                    logger.error(f'Research Division daily backtest failed: {_bte}')

        # ── Research Division — shadow lab live scanning (every 5 min during session) ──
        if not hasattr(background_scheduler, '_last_shadow_scan'):
            background_scheduler._last_shadow_scan = 0
        _now_ss = datetime.now(timezone.utc)
        _in_session_ss = 7 <= _now_ss.hour < 21
        if _in_session_ss and now - background_scheduler._last_shadow_scan >= 300:
            background_scheduler._last_shadow_scan = now
            try:
                import research_division as _rd_ss
                shadow_signals = _rd_ss.run_shadow_lab_scans()
                if shadow_signals:
                    logger.info(f'Shadow lab: {len(shadow_signals)} signal(s) generated this tick')
                    for _ss_item in shadow_signals:
                        _ss_sig = _ss_item.get('signal', {})
                        _ss_msg = (
                            f'[SHADOW LAB] {_ss_item.get("strategy_name", "")}\n'
                            f'{_ss_sig.get("symbol", "")} {str(_ss_sig.get("direction", "")).upper()} '
                            f'@ {_ss_sig.get("entry", "?")}\n'
                            f'Stop: {_ss_sig.get("stop", "?")} | '
                            f'Target: {_ss_sig.get("target", "?")} | '
                            f'RR: {_ss_sig.get("rr", "?")} | '
                            f'trade_id={_ss_item.get("trade_id", "?")}'
                        )
                        send_telegram(_ss_msg)
            except Exception as _ss_e:
                logger.warning(f'Shadow lab scan error: {_ss_e}')

        # ── Research Division — hypothesis engine (02:00 UTC daily) ──
        _now_hyp = datetime.now(timezone.utc)
        if _now_hyp.hour == 2:
            if not hasattr(background_scheduler, '_last_hypothesis_day') or \
                    background_scheduler._last_hypothesis_day != _today_utc:
                background_scheduler._last_hypothesis_day = _today_utc
                try:
                    import research_division as _rd_mod
                    _rd_mod.run_hypothesis_engine()
                    _rd_mod.update_pattern_library()
                    logger.info('Research Division: hypothesis engine complete')
                except Exception as _hye:
                    logger.error(f'Research Division hypothesis engine failed: {_hye}')

        # ── Meridian Direction — resolve predictions every 5 min ──────
        try:
            import meridian_direction as _mdir_res
            _mdir_res.resolve_predictions()
        except Exception as _mdir_res_e:
            logger.debug(f'meridian_direction resolve_predictions: {_mdir_res_e}')

        # ── Meridian Direction — log predictions every 30 min ──────
        if not hasattr(background_scheduler, '_last_mdir_log') or \
                now - background_scheduler._last_mdir_log >= 1800:
            background_scheduler._last_mdir_log = now
            try:
                import meridian_direction as _mdir_log
                for _mdir_sym in ('MNQ', 'ES'):
                    _mdir_preds = _mdir_log.predict_all(_mdir_sym)
                    # Get current price for logging
                    try:
                        _mdir_conn = _db.connect()
                        _mdir_pr = _mdir_conn.execute(
                            'SELECT close FROM ohlcv WHERE symbol=? AND timeframe=? ORDER BY ts DESC LIMIT 1',
                            (_mdir_sym, '5min')
                        ).fetchone()
                        _mdir_conn.close()
                        _mdir_price = float(_mdir_pr[0]) if _mdir_pr else 0.0
                    except Exception:
                        _mdir_price = 0.0
                    for _mdir_h, _mdir_hr in _mdir_preds.items():
                        if _mdir_hr.get('deployed') and _mdir_price > 0:
                            _mdir_log.log_prediction(
                                _mdir_sym, _mdir_h,
                                _mdir_hr['direction'], _mdir_hr['probability'], _mdir_price
                            )
            except Exception as _mdir_log_e:
                logger.debug(f'meridian_direction log_prediction: {_mdir_log_e}')

        # ── Meridian Direction — weekly retrain (Sunday 03:00 UTC) ──
        _now_mdir = datetime.now(timezone.utc)
        if _now_mdir.weekday() == 6 and _now_mdir.hour == 3:
            if not hasattr(background_scheduler, '_last_mdir_retrain_day') or \
                    background_scheduler._last_mdir_retrain_day != _today_utc:
                background_scheduler._last_mdir_retrain_day = _today_utc
                try:
                    import meridian_direction as _mdir_rt
                    _mdir_report = _mdir_rt.retrain_all()
                    deployed = [f'{s}/{h}' for s, hrs in _mdir_report.items()
                                for h, r in hrs.items() if r.get('deployed')]
                    send_telegram(
                        f'[MERIDIAN DIR] Weekly retrain complete\n'
                        f'Deployed: {", ".join(deployed) or "none"}'
                    )
                except Exception as _mdir_rt_e:
                    logger.error(f'Meridian Direction retrain failed: {_mdir_rt_e}')

        # ── Meridian L3 — weekly retrain (Sunday 02:00 UTC) ──
        _now_l3 = datetime.now(timezone.utc)
        if _now_l3.weekday() == 6 and _now_l3.hour == 2:
            if not hasattr(background_scheduler, '_last_l3_retrain_day') or \
                    background_scheduler._last_l3_retrain_day != _today_utc:
                background_scheduler._last_l3_retrain_day = _today_utc
                try:
                    import meridian_l3 as _ml3_rt
                    for _rt_sym in ('MNQ', 'ES'):
                        _rt_ok = _ml3_rt.retrain(_rt_sym)
                        if _rt_ok:
                            _rt_info = _ml3_rt._l3_cache.get(_rt_sym, {})
                            send_telegram(
                                f'[MERIDIAN L3] Weekly retrain complete — {_rt_sym}\n'
                                f'AUC={_rt_info.get("auc", 0):.4f} '
                                f'mode={_rt_info.get("mode", "?")} '
                                f'trained={_rt_info.get("trained_at", "?")}'
                            )
                except Exception as _l3_rt_e:
                    logger.error(f'Meridian L3 retrain failed: {_l3_rt_e}')

        # ── Research Division — combination explorer (Saturday 03:00 UTC) ──
        _now_combo = datetime.now(timezone.utc)
        if _now_combo.weekday() == 5 and _now_combo.hour == 3:
            if not hasattr(background_scheduler, '_last_combo_day') or \
                    background_scheduler._last_combo_day != _today_utc:
                background_scheduler._last_combo_day = _today_utc
                try:
                    import research_division as _rd_mod
                    _rd_mod.run_combination_explorer()
                    logger.info('Research Division: combination explorer complete')
                except Exception as _cbe:
                    logger.error(f'Research Division combination explorer failed: {_cbe}')

        # ── Research Division — weekly health check (Monday 06:00 UTC) ──
        # DB-backed guard: persists across restarts so the check runs exactly once
        # per week even if Railway restarts the server multiple times on Monday morning.
        _now_rd = datetime.now(timezone.utc)
        if _now_rd.weekday() == 0 and _now_rd.hour == 6:
            if not hasattr(background_scheduler, '_last_research_monday') or \
                    background_scheduler._last_research_monday != _today_utc:
                background_scheduler._last_research_monday = _today_utc
                try:
                    import research_division as _rd_mod
                    if _rd_mod._health_check_ran_recently():
                        logger.info('Research Division: weekly check skipped — ran within last 24h')
                    else:
                        _rd_mod.run_weekly_health_check()
                        _rd_mod.score_shadow_lab()
                        _rd_mod.generate_weekly_telegram_report()
                        _rd_mod._mark_health_check_ran()
                        logger.info('Research Division: weekly check complete')
                except Exception as _rde:
                    logger.error(f'Research Division weekly check failed: {_rde}')

        if now - last_daily > 86400:
            logger.info('Running daily data update...')
            for sym in INSTRUMENTS:
                try:
                    daily_update(sym)
                except Exception as e:
                    logger.warning('Daily update failed ' + sym + ': ' + str(e))
            last_daily = now

        # ── APEX Data Feed ────────────────────────────────────────
        if not hasattr(background_scheduler, '_last_feed'):
            background_scheduler._last_feed = 0
            background_scheduler._last_htf  = 0

        # Live bar feed — covers both London (07:00–11:00) and NY (13:00–20:00) sessions.
        # Feed runs 07:00–21:00 UTC (1h buffer after NY close).
        # Databento Live streams ohlcv-1m bars as they close.
        _now_utc_hr = datetime.now(timezone.utc).hour
        _in_session = 7 <= _now_utc_hr < 21
        try:
            from data_feed import start_live_feed, stop_live_feed, is_live_feed_running, \
                                  get_live_feed_stats, restart_live_feed
            if _in_session and not is_live_feed_running():
                started = start_live_feed()
                if started:
                    logger.info('LiveBarFeed: session open — live 1min stream started')
            elif not _in_session and not is_live_feed_running():
                # Log once per hour so logs show feed is intentionally idle
                if not hasattr(background_scheduler, '_last_offsession_log') or \
                        now - background_scheduler._last_offsession_log >= 3600:
                    background_scheduler._last_offsession_log = now
                    logger.info(
                        f'LiveBarFeed: off-session (UTC hour={_now_utc_hr}) — '
                        f'feed starts at 13:00 UTC'
                    )
            elif _in_session and is_live_feed_running():
                # Watchdog: thread alive but no bar in >5 min means silent stall — force restart
                _feed_stats = get_live_feed_stats()
                _secs = _feed_stats.get('seconds_since_last_bar')
                if _secs is not None and _secs > 300:
                    logger.warning(
                        f'LiveBarFeed: watchdog — no bar in {_secs}s during session, '
                        f'force-restarting feed'
                    )
                    restart_live_feed()
            elif not _in_session and is_live_feed_running():
                stop_live_feed()
                logger.info('LiveBarFeed: session closed — live feed stopped')
        except Exception as e:
            logger.warning(f'LiveBarFeed management error: {e}')

        # Historical refresh — 5min/15min every 5 min; HTF every 30 min
        # When live feed is running, refresh_all() skips 1min (live handles it).
        # When live feed is down, refresh_all() includes 1min via historical.
        if now - background_scheduler._last_feed > 300:
            try:
                from data_feed import refresh_all
                include_htf = (now - background_scheduler._last_htf) > 1800
                results     = refresh_all(include_htf=include_htf)
                if include_htf:
                    background_scheduler._last_htf = now
                for _sym, _tfs in results.items():
                    for _tf, _cnt in _tfs.items():
                        if _cnt > 0:
                            logger.info(f'DataFeed: {_sym} {_tf} +{_cnt} new bars')

            except Exception as e:
                logger.warning(f'DataFeed refresh error: {e}')
            background_scheduler._last_feed = now

            # REGIME RESEARCH — observation only, no signal impact
            try:
                from regime_engine import calculate_and_store as _regime_calc
                for _reg_sym in ('MNQ', 'ES'):
                    _regime_calc(_reg_sym)
            except Exception as _re:
                logger.warning(f'Regime engine error: {_re}')

        if now - last_macro_log > 14400:
            try:
                m    = fetch_macro_live()
                conn = _db.connect()
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
            logger.warning(f'Trade monitor error: {e}', exc_info=True)

        # _active_signals reconciliation removed — replaced by _fired_today dedup.
        # _fired_today expires automatically at midnight; no per-cycle DB reconciliation needed.

        # ── APEX Session Alerts ──────────────────────────────────
        try:
            check_session_alerts()
        except Exception as e:
            logger.warning(f'Session alert error: {e}')

        # ── APEX Daily P&L Summary — fires at 19:00 UTC ──────────
        try:
            _now = datetime.now(timezone.utc)
            if _now.hour == 19 and _now.minute < 5:
                if not hasattr(background_scheduler, '_daily_summary_date') or                    background_scheduler._daily_summary_date != str(_now.date()):
                    background_scheduler._daily_summary_date = str(_now.date())
                    from trade_tracker import get_stats, init_trades_table
                    from live_scanner import send_telegram
                    from zoneinfo import ZoneInfo
                    NY = ZoneInfo('America/New_York')
                    init_trades_table()
                    conn = _db.connect()
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
            now_utc = datetime.now(timezone.utc)
            _fvg_windows = FVG_PARAMS.get('session_windows', {}).get('MNQ', [{'start': 13, 'end': 19}])
            _in_fvg_session = any(w['start'] <= now_utc.hour < w['end'] for w in _fvg_windows)
            if _in_fvg_session:
                if not hasattr(background_scheduler, '_risk_gate'):
                    from risk_manager import RiskGate
                    background_scheduler._risk_gate = RiskGate()
                _rg = background_scheduler._risk_gate
                if _rg.daily.is_daily_limit_hit():
                    logger.info('FVG scanner: daily loss limit hit — signals suppressed')
                else:
                    _regime  = _rg.regime.get_regime()
                    _dd_mult = _rg.dd.get_risk_multiplier()
                    _risk_footer = (
                        f'\n⚙️ <i>Regime: {_regime.label} | '
                        f'Risk: {_dd_mult:.2f}× | DD: {_rg.dd.get_drawdown_pct():.1f}%</i>'
                    )
                    logger.info('Scanning FVG for MNQ')
                    fvg_signals = scan_fvg('MNQ', now_utc)
                    for sig in fvg_signals:
                        _fvg_setup = sig.get('setup', 'D_fvg_fill')
                        _fvg_sym   = sig.get('symbol', 'MNQ')
                        # ── Regime gate (mirrors Setup D) ─────────────────
                        if not is_setup_enabled('D', _fvg_sym):
                            logger.info(f'FVG: regime gate blocked — skipping {_fvg_sym}')
                            _write_scan_log(_fvg_sym, _fvg_setup, None, sig.get('direction', ''),
                                            sig.get('entry'), sig.get('stop'),
                                            sig.get('target'), None,
                                            'BLOCKED_REGIME', 'FVG regime gate')
                            continue
                        # ── FILTER 1: max 1 concurrent per instrument ─────
                        if SIGNAL_FILTERS['max_concurrent_per_instrument']:
                            _f1fvg_has, _f1fvg_id = _has_open_trade_on_instrument(_fvg_sym)
                            if _f1fvg_has:
                                logger.info(f'FVG: skipped — {_fvg_sym} already has open trade #{_f1fvg_id}')
                                _write_scan_log(_fvg_sym, _fvg_setup, None, sig.get('direction', ''),
                                                sig.get('entry'), sig.get('stop'),
                                                sig.get('target'), None,
                                                'BLOCKED_CONCURRENT', f'open trade #{_f1fvg_id}')
                                continue
                        if _has_opposite_swing_trade(_fvg_sym, sig['direction']):
                            logger.info(
                                f'FVG {sig["direction"]} suppressed — '
                                f'opposite swing trade open on {_fvg_sym}'
                            )
                            continue
                        if _cal_block(_fvg_sym, _fvg_setup):
                            _write_scan_log(_fvg_sym, _fvg_setup, None, sig.get('direction', ''),
                                            sig.get('entry'), sig.get('stop'),
                                            sig.get('target'), None,
                                            'BLOCKED_CALENDAR', 'calendar gate')
                            continue
                        if _check_and_mark_fired(_fvg_sym, _fvg_setup, sig['direction']):
                            continue
                        if _is_signal_already_active(_fvg_sym, sig['direction'], _fvg_setup):
                            continue
                        _fvg_stop_pts = round(abs(sig['entry'] - sig['stop']), 1)
                        _fvg_risk_usd = round(_fvg_stop_pts * 2, 0)
                        logger.info(
                            f'Signal: {_fvg_sym} {sig["direction"]} {_fvg_setup} | '
                            f'stop={_fvg_stop_pts} pts | risk=${_fvg_risk_usd:.0f} | contracts=1'
                        )
                        _fvg_cap_ok, _fvg_cap_msg = _stop_cap_ok(sig)
                        if not _fvg_cap_ok:
                            logger.warning(_fvg_cap_msg)
                            _write_scan_log(_fvg_sym, _fvg_setup, None, sig.get('direction', ''),
                                            sig.get('entry'), sig.get('stop'),
                                            sig.get('target'), None,
                                            'BLOCKED_STOP_CAP', _fvg_cap_msg[:200])
                            continue
                        _write_scan_log(_fvg_sym, _fvg_setup, None, sig.get('direction', ''),
                                        sig.get('entry'), sig.get('stop'),
                                        sig.get('target'), None,
                                        'SIGNAL', 'FVG scan generated')
                        _fvg_tid = None
                        try:
                            _fvg_tid = log_trade(sig)
                            if _fvg_tid:
                                logger.info(f'Trade logged: id={_fvg_tid} {_fvg_sym} {sig["direction"]} {_fvg_setup}')
                                _write_scan_log(_fvg_sym, _fvg_setup, None, sig.get('direction', ''),
                                                sig.get('entry'), sig.get('stop'),
                                                sig.get('target'), None,
                                                'EXECUTED', f'trade_id={_fvg_tid}')
                            else:
                                logger.error(f'CRITICAL: FVG log_trade() returned None — {_fvg_sym} {sig["direction"]} NOT in DB')
                        except Exception as _fvg_lte:
                            logger.error(f'CRITICAL: FVG log_trade() EXCEPTION — {_fvg_sym} {sig["direction"]}: {_fvg_lte}', exc_info=True)
                        # ── Telegram only after confirmed log_trade ───────
                        if not _fvg_tid:
                            logger.critical(f'CRITICAL: Skipping Telegram for FVG {_fvg_sym} — log_trade returned None')
                        else:
                            try:
                                msg = format_fvg_alert(sig) + _risk_footer
                                send_telegram(msg)
                            except Exception as _fvg_te:
                                logger.error(f'FVG send_telegram failed {_fvg_sym} {sig["direction"]}: {_fvg_te}')
                            _execute_via_tradovate(sig, _fvg_tid)
                        logger.info(
                            f'FVG signal: {_fvg_sym} {sig["direction"].upper()} entry={sig["entry"]} '
                            f'regime={_regime.label} dd_mult={_dd_mult:.2f}×'
                        )
        except Exception as e:
            logger.warning(f'FVG scanner error: {e}')

        # ── APEX Engine v2 — Setup A/B/C/E Scanner ──────────────────────
        try:
            from live_scanner import run_scan, send_telegram, format_alert, SignalTracker
            from setup_e import format_alert as format_alert_e
            from trade_tracker import log_trade
            if not hasattr(background_scheduler, '_apex_tracker'):
                background_scheduler._apex_tracker = SignalTracker()
            if not hasattr(background_scheduler, '_risk_gate'):
                from risk_manager import RiskGate
                background_scheduler._risk_gate = RiskGate()
            tracker = background_scheduler._apex_tracker
            _rg     = background_scheduler._risk_gate
            _heat_abc = _count_open_trades()
            if _heat_abc >= MAX_PORTFOLIO_HEAT:
                logger.info(f'Portfolio heat: {_heat_abc} trades open — new signal blocked for Setup A/B/C/E')
                _write_scan_log('ALL', 'A_B_C_E', None, '', None, None, None, None,
                                'BLOCKED_HEAT', f'heat={_heat_abc}/{MAX_PORTFOLIO_HEAT}')
            elif _rg.daily.is_daily_limit_hit():
                logger.info('APEX scanner: daily loss limit hit — signals suppressed')
            else:
                _regime  = _rg.regime.get_regime()
                _dd_mult = _rg.dd.get_risk_multiplier()
                _risk_footer = (
                    f'\n⚙️ <i>Regime: {_regime.label} | '
                    f'Risk: {_dd_mult:.2f}× | DD: {_rg.dd.get_drawdown_pct():.1f}%</i>'
                )
                logger.info('Scanning Setups A/B/C/E...')
                signals = run_scan()
                for result in signals:
                    _sym   = result.symbol
                    _dirn  = result.direction
                    _setup = getattr(result, 'setup', '')

                    # ── Control Centre: check setup enabled ───────────────
                    _ctrl_sid = _setup[0].upper() if _setup else ''
                    if _ctrl_sid and not is_setup_enabled(_ctrl_sid, _sym):
                        logger.info(f'Setup {_ctrl_sid}: disabled via Control Centre — skipping')
                        _write_scan_log(_sym, _setup, None, _dirn, None, None, None, None,
                                        'BLOCKED_REGIME', f'regime gate blocked {_ctrl_sid}')
                        continue

                    # ── FILTER 1: max 1 concurrent trade per instrument ───
                    if SIGNAL_FILTERS['max_concurrent_per_instrument']:
                        _f1_has, _f1_id = _has_open_trade_on_instrument(_sym)
                        if _f1_has:
                            logger.info(f'{_setup}: skipped — {_sym} already has open trade #{_f1_id}')
                            _write_scan_log(_sym, _setup, None, _dirn, None, None, None, None,
                                            'BLOCKED_CONCURRENT', f'open trade #{_f1_id}')
                            continue

                    # ── FILTER 2: primary session only for B and C ────────
                    if SIGNAL_FILTERS['primary_session_only_bcd']:
                        if _setup in ('B_choch_breaker', 'C_bos_ob'):
                            _qual = getattr(result, 'quality', 'primary')
                            if _qual != 'primary':
                                logger.info(
                                    f'{_setup}: skipped — secondary session quality, requires primary '
                                    f'({getattr(result, "session", "")})'
                                )
                                continue

                    # ── FILTER 3: dual HTF bias (1h must align with 4h) ──
                    if SIGNAL_FILTERS['dual_htf_bias']:
                        _f3_ok, _f3_detail = _check_1h_bias(_sym, _dirn)
                        if not _f3_ok:
                            logger.info(
                                f'{_setup}: skipped — 1h bias conflicts with {_dirn} direction '
                                f'({_f3_detail})'
                            )
                            _write_scan_log(_sym, _setup, None, _dirn, None, None, None, None,
                                            'BLOCKED_HTF_BIAS', _f3_detail[:200])
                            continue

                    # ── FILTER 4: Setup E minimum ATR ────────────────────
                    if SIGNAL_FILTERS['setup_e_min_atr'] and _setup == 'E_ema50_pullback':
                        _min_atr = SETUP_E_MIN_ATR.get(_sym, 0.0)
                        if result.entry is not None and result.stop is not None:
                            _atr_impl = abs(result.entry - result.stop) / 1.5
                            if _atr_impl < _min_atr:
                                logger.info(
                                    f'Setup E: skipped — ATR too low '
                                    f'(ATR={_atr_impl:.2f} pts, minimum={_min_atr})'
                                )
                                continue

                    # Correlation filter: suppress Setup E if opposite swing trade open
                    if _setup == 'E_ema50_pullback':
                        if _has_opposite_swing_trade(_sym, _dirn):
                            logger.info(f'Setup E {_dirn} suppressed — opposite swing trade open on {_sym}')
                            continue
                    # Economic calendar gate — skip if blackout active
                    if _cal_block(_sym, _setup):
                        _write_scan_log(_sym, _setup, None, _dirn, None, None, None, None,
                                        'BLOCKED_CALENDAR', 'calendar gate')
                        continue
                    # Unified dedup: mark fired BEFORE telegram
                    if _check_and_mark_fired(_sym, _setup, _dirn):
                        continue
                    if _is_signal_already_active(_sym, _dirn, _setup):
                        continue
                    _write_scan_log(_sym, _setup, None, _dirn,
                                    getattr(result, 'entry', None), getattr(result, 'stop', None),
                                    getattr(result, 'target', None), None,
                                    'SCANNING', f'signal found dir={_dirn}')
                    if result.entry is not None and result.stop is not None:
                        _apex_stop_pts = round(abs(result.entry - result.stop), 1)
                        _apex_risk_usd = round(_apex_stop_pts * (2 if _sym == 'MNQ' else 50), 0)
                        logger.info(
                            f'Signal: {_sym} {_dirn} {_setup} | '
                            f'stop={_apex_stop_pts} pts | risk=${_apex_risk_usd:.0f} | contracts=1'
                        )
                        _apex_cap_ok, _apex_cap_msg = _stop_cap_ok(
                            {'symbol': _sym, 'entry': result.entry, 'stop': result.stop, 'setup': _setup}
                        )
                        if not _apex_cap_ok:
                            logger.warning(_apex_cap_msg)
                            _write_scan_log(_sym, _setup, None, _dirn,
                                            result.entry, result.stop, getattr(result, 'target', None), None,
                                            'BLOCKED_STOP_CAP', _apex_cap_msg[:200])
                            continue
                    tracker.mark_sent(result)
                    _write_scan_log(_sym, _setup, None, _dirn,
                                    getattr(result, 'entry', None), getattr(result, 'stop', None),
                                    getattr(result, 'target', None), None,
                                    'SIGNAL', f'A/B/C/E signal ready')
                    _sig_dict = {
                        'symbol':    _sym,
                        'direction': _dirn,
                        'setup':     _setup,
                        'mode':      'swing',
                        'entry':     result.entry,
                        'stop':      result.stop,
                        'target':    result.target,
                        'rr':        result.rr,
                        'session':   getattr(result, 'session', 'NY Primary'),
                        'quality':   getattr(result, 'quality', 'primary'),
                    }
                    _apex_tid = None
                    try:
                        _apex_tid = log_trade(_sig_dict)
                        if _apex_tid:
                            logger.info(f'Trade logged: id={_apex_tid} {_sym} {_dirn} {_setup}')
                            _write_scan_log(_sym, _setup, None, _dirn,
                                            result.entry, result.stop, getattr(result, 'target', None), None,
                                            'EXECUTED', f'trade_id={_apex_tid}')
                        else:
                            logger.error(f'CRITICAL: log_trade() returned None — {_sym} {_dirn} {_setup} NOT in DB')
                    except Exception as _apex_lte:
                        logger.error(f'CRITICAL: log_trade() EXCEPTION — {_sym} {_dirn} {_setup}: {_apex_lte}', exc_info=True)
                    # Gate Telegram on successful log_trade — never alert for unlogged trade
                    if not _apex_tid:
                        logger.critical(
                            f'[A/B/C/E] CRITICAL: Skipping Telegram for {_sym} {_setup} — '
                            f'log_trade did not return a trade_id. Signal not in DB.'
                        )
                    else:
                        try:
                            msg = (format_alert_e(result) if _setup == 'E_ema50_pullback'
                                   else format_alert(result))
                            send_telegram(msg + _risk_footer)
                        except Exception as _apex_te:
                            logger.error(f'{_setup} send_telegram failed {_sym}: {_apex_te}')
                    if _apex_tid:
                        if _ctrl_sid in PAPER_ONLY_SETUPS:
                            logger.info(
                                f'Setup {_ctrl_sid}: paper-only — '
                                f'Tradovate execution skipped for {_sym} {_dirn}'
                            )
                        else:
                            logger.info(f'Calling _execute_via_tradovate: {_sym} {_dirn} {_setup} trade_id={_apex_tid}')
                            _execute_via_tradovate(_sig_dict, _apex_tid)
                    else:
                        logger.warning(f'Skipping _execute_via_tradovate — log_trade returned None for {_sym} {_dirn} {_setup}')
                    logger.info(f'APEX signal: {_sym} {_dirn} {_setup} regime={_regime.label} dd_mult={_dd_mult:.2f}×')
                if not signals:
                    logger.info('APEX scan: no signals this tick')
        except Exception as e:
            logger.warning(f'APEX scanner error: {e}', exc_info=True)

        # ── Setup F — Random Forest ML (every 5 min) ─────────────
        # setup_f_enabled=False: fully disabled, dedup/log_trade gap under investigation
        if not setup_f_enabled or not is_setup_enabled('F'):
            logger.debug('Setup F: disabled (setup_f_enabled=False or Control Centre) — skipping')
        else:
            if not hasattr(background_scheduler, '_last_setup_f'):
                background_scheduler._last_setup_f = 0
            if now - background_scheduler._last_setup_f >= 300:
                background_scheduler._last_setup_f = now
                try:
                    from setup_f_ml import scan_setup_f, format_f_alert, check_model_degradation
                    from live_scanner import send_telegram
                    from trade_tracker import log_trade
                    _now_utc = datetime.now(timezone.utc)
                    if not hasattr(background_scheduler, '_risk_gate'):
                        from risk_manager import RiskGate
                        background_scheduler._risk_gate = RiskGate()
                    _rg = background_scheduler._risk_gate
                    _regime  = _rg.regime.get_regime()
                    _dd_mult = _rg.dd.get_risk_multiplier()
                    _risk_footer = (
                        f'\n⚙️ <i>Regime: {_regime.label} | '
                        f'Risk: {_dd_mult:.2f}× | DD: {_rg.dd.get_drawdown_pct():.1f}%</i>'
                    )
                    for _sym in ['MNQ', 'ES']:  # GC paper only — no live signals
                        try:
                            if not is_setup_enabled('F', _sym):
                                logger.info(f'Setup F {_sym}: regime gate — BLOCKED')
                                _write_scan_log(_sym, 'F_ml_regime', None, '', None, None, None, None,
                                                'BLOCKED_REGIME', 'Setup F regime gate')
                                continue
                            # ── FILTER 1: max 1 concurrent per instrument ─────
                            if SIGNAL_FILTERS['max_concurrent_per_instrument']:
                                _f1f_has, _f1f_id = _has_open_trade_on_instrument(_sym)
                                if _f1f_has:
                                    logger.info(f'Setup F: skipped — {_sym} already has open trade #{_f1f_id}')
                                    _write_scan_log(_sym, 'F_rf_signal', None, '', None, None, None, None,
                                                    'BLOCKED_CONCURRENT', f'open trade #{_f1f_id}')
                                    continue
                            if check_model_degradation(_sym):
                                logger.warning(f'Setup F {_sym} model degraded — skipping')
                                _write_scan_log(_sym, 'F_rf_signal', None, '', None, None, None, None,
                                                'BLOCKED_REGIME', 'model degraded')
                                continue
                            if _rg.daily.is_daily_limit_hit():
                                logger.info(f'Setup F {_sym}: daily limit hit — suppressed')
                                continue
                            _write_scan_log(_sym, 'F_rf_signal', None, '', None, None, None, None,
                                            'SCANNING', 'Setup F entering scan')
                            sig = scan_setup_f(_sym, _now_utc)
                            if sig:
                                _write_scan_log(_sym, sig.get('setup', 'F_rf_signal'),
                                                sig.get('confidence'), sig['direction'],
                                                sig.get('entry'), sig.get('stop'),
                                                sig.get('target'), None, 'SIGNAL', 'scan_setup_f generated')
                                if _has_opposite_swing_trade(_sym, sig['direction']):
                                    logger.info(f'Setup F {_sym} {sig["direction"]} suppressed — opposite swing trade open')
                                    _write_scan_log(_sym, sig.get('setup', 'F_rf_signal'),
                                                    sig.get('confidence'), sig['direction'],
                                                    sig.get('entry'), sig.get('stop'),
                                                    sig.get('target'), None, 'BLOCKED_CONCURRENT', 'opposite swing trade')
                                    continue
                                _f_stop_pts = round(abs(sig['entry'] - sig['stop']), 1)
                                _f_risk_usd = round(_f_stop_pts * (2 if _sym == 'MNQ' else 50), 0)
                                logger.info(
                                    f'Signal: {_sym} {sig["direction"]} {sig["setup"]} | '
                                    f'stop={_f_stop_pts} pts | risk=${_f_risk_usd:.0f} | contracts=1'
                                )
                                # Economic calendar gate
                                if _cal_block(_sym, sig['setup']):
                                    _write_scan_log(_sym, sig.get('setup', 'F_rf_signal'),
                                                    sig.get('confidence'), sig['direction'],
                                                    sig.get('entry'), sig.get('stop'),
                                                    sig.get('target'), None, 'BLOCKED_CALENDAR', 'calendar gate')
                                    continue
                                # Unified dedup: mark fired BEFORE telegram
                                # _fired_today expires at midnight — log_trade failure does NOT cause re-fire
                                if _check_and_mark_fired(_sym, sig['setup'], sig['direction']):
                                    continue
                                if _is_signal_already_active(_sym, sig['direction'], sig['setup']):
                                    continue
                                _f_cap_ok, _f_cap_msg = _stop_cap_ok(sig)
                                if not _f_cap_ok:
                                    logger.warning(_f_cap_msg)
                                    _write_scan_log(_sym, sig.get('setup', 'F_rf_signal'),
                                                    sig.get('confidence'), sig['direction'],
                                                    sig.get('entry'), sig.get('stop'),
                                                    sig.get('target'), None, 'BLOCKED_STOP_CAP', _f_cap_msg[:200])
                                    continue
                                _f_tid = None
                                try:
                                    logger.info(
                                        f'Setup F: attempting log_trade {_sym} {sig["direction"]} '
                                        f'entry={sig["entry"]} sig_keys={list(sig.keys())}'
                                    )
                                    _f_tid = log_trade(sig)
                                    if _f_tid:
                                        logger.info(f'Setup F: trade logged id={_f_tid} {_sym} {sig["direction"]}')
                                        _write_scan_log(_sym, sig.get('setup', 'F_rf_signal'),
                                                        sig.get('confidence'), sig['direction'],
                                                        sig.get('entry'), sig.get('stop'),
                                                        sig.get('target'), None,
                                                        'EXECUTED', f'trade_id={_f_tid}')
                                    else:
                                        logger.error(
                                            f'CRITICAL: Setup F log_trade() returned None — {_sym} {sig["direction"]} NOT in DB'
                                        )
                                except Exception as _lte:
                                    logger.error(
                                        f'CRITICAL: Setup F log_trade() EXCEPTION — {_sym} {sig["direction"]}: {_lte}',
                                        exc_info=True
                                    )
                                # ── Telegram only after confirmed log_trade ──────
                                if not _f_tid:
                                    logger.critical(f'CRITICAL: Skipping Telegram for Setup F {_sym} — log_trade returned None')
                                else:
                                    try:
                                        msg = format_f_alert(sig) + _risk_footer
                                        send_telegram(msg)
                                    except Exception as _te:
                                        logger.error(f'Setup F: send_telegram failed {_sym}: {_te}')
                                if _f_tid:
                                    _execute_via_tradovate(sig, _f_tid)
                                logger.info(
                                    f'Setup F signal: {_sym} {sig["direction"].upper()} '
                                    f'conf={sig["confidence"]:.0%} regime={_regime.label} db_id={_f_tid}'
                                )
                        except Exception as _fe:
                            logger.warning(f'Setup F {_sym} error: {_fe}')
                except Exception as e:
                    logger.warning(f'Setup F scanner error: {e}')

        # ── Setup I — Mathematical Alpha (every 5 min) ─────────────
        # MNQ: live | ES: live | XGB dual-model, Tue/Wed/Thu, 13-20 UTC MNQ / 13-19 UTC ES
        if not hasattr(background_scheduler, '_last_setup_i'):
            background_scheduler._last_setup_i = 0
        if now - background_scheduler._last_setup_i >= 300:
            background_scheduler._last_setup_i = now
            if not is_setup_enabled('I'):
                logger.info('Setup I: disabled via Control Centre — skipping')
            else:
                try:
                    from setup_i_mathematical import scan_setup_i, format_i_alert
                    from live_scanner import send_telegram
                    from trade_tracker import log_trade
                    _now_utc_i = datetime.now(timezone.utc)
                    if not hasattr(background_scheduler, '_risk_gate'):
                        from risk_manager import RiskGate
                        background_scheduler._risk_gate = RiskGate()
                    _rg_i = background_scheduler._risk_gate
                    for _sym in ['MNQ', 'ES']:
                        try:
                            # ── Pre-scan gates — every outcome logged ─────────────
                            _heat_i = _count_open_trades()
                            if _heat_i >= MAX_PORTFOLIO_HEAT:
                                logger.info(f'[I-pre] Setup I {_sym}: heat {_heat_i}/{MAX_PORTFOLIO_HEAT} — BLOCKED')
                                _write_scan_log(_sym, 'I_mathematical', None, '', None, None, None, None,
                                                'BLOCKED_HEAT', f'heat={_heat_i}/{MAX_PORTFOLIO_HEAT}')
                                continue
                            logger.info(f'[I-pre] Setup I {_sym}: heat {_heat_i}/{MAX_PORTFOLIO_HEAT} — OK')

                            if SIGNAL_FILTERS['max_concurrent_per_instrument']:
                                _i_has, _i_id = _has_open_trade_on_instrument(_sym)
                                if _i_has:
                                    logger.info(f'[I-pre] Setup I {_sym}: open trade #{_i_id} — BLOCKED')
                                    _write_scan_log(_sym, 'I_mathematical', None, '', None, None, None, None,
                                                    'BLOCKED_CONCURRENT', f'open trade #{_i_id}')
                                    continue
                                logger.info(f'[I-pre] Setup I {_sym}: no open trade — OK')

                            if _rg_i.daily.is_daily_limit_hit():
                                logger.info(f'[I-pre] Setup I {_sym}: daily limit hit — BLOCKED')
                                continue

                            if _now_utc_i.weekday() not in {1, 2, 3}:
                                _i_day = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][_now_utc_i.weekday()]
                                logger.info(
                                    f'[I-pre] Setup I {_sym}: day={_i_day} '
                                    f'(weekday={_now_utc_i.weekday()}) not Tue/Wed/Thu — BLOCKED'
                                )
                                continue
                            logger.info(
                                f'[I-pre] Setup I {_sym}: '
                                f'day={["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][_now_utc_i.weekday()]} — OK'
                            )

                            if not is_setup_enabled('I', _sym):
                                logger.info(f'[I-pre] Setup I {_sym}: regime gate — BLOCKED')
                                _write_scan_log(_sym, 'I_mathematical', None, '', None, None, None, None,
                                                'BLOCKED_REGIME', 'Setup I regime gate')
                                continue
                            logger.info(f'[I-pre] Setup I {_sym}: regime gate — OK')
                            _write_scan_log(_sym, 'I_mathematical', None, '', None, None, None, None,
                                            'SCANNING', 'Setup I entering scan')

                            # ── [I-1/6] Scan ─────────────────────────────────────
                            logger.info(f'[I-1/6] Setup I {_sym}: scanning...')
                            sig = scan_setup_i(_sym, _now_utc_i)
                            logger.info(
                                f'[I-1/6] scan_setup_i {_sym} returned: '
                                f'{"SIGNAL dir=" + sig["direction"] + " entry=" + str(sig.get("entry")) if sig else "None — no signal conditions met"}'
                            )
                            if not sig:
                                continue
                            _write_scan_log(_sym, sig.get('setup', 'I_mathematical'),
                                            sig.get('xgb_prob'), sig.get('direction', ''),
                                            sig.get('entry'), sig.get('stop'),
                                            sig.get('target'), None,
                                            'SIGNAL', f'xgb={sig.get("xgb_prob")} lr={sig.get("lr_prob")}')

                            # ── [I-2a/6] Calendar ────────────────────────────────
                            if _cal_block(_sym, sig['setup']):
                                logger.info(f'[I-2a/6] Setup I {_sym}: calendar blackout — BLOCKED')
                                _write_scan_log(_sym, sig.get('setup', 'I_mathematical'),
                                                sig.get('xgb_prob'), sig.get('direction', ''),
                                                sig.get('entry'), sig.get('stop'),
                                                sig.get('target'), None,
                                                'BLOCKED_CALENDAR', 'calendar blackout')
                                continue
                            logger.info(f'[I-2a/6] Setup I {_sym}: calendar clear — OK')

                            # ── [I-2b/6] Dedup ───────────────────────────────────
                            if _is_signal_already_active(_sym, sig['direction'], sig['setup']):
                                logger.info(f'[I-2b/6] Setup I {_sym} {sig["direction"]}: already active in DB — BLOCKED')
                                continue
                            logger.info(f'[I-2b/6] Setup I {_sym} {sig["direction"]}: dedup — OK')

                            # ── [I-2c/6] Stop cap ────────────────────────────────
                            _i_cap_ok, _i_cap_msg = _stop_cap_ok(sig)
                            if not _i_cap_ok:
                                logger.warning(f'[I-2c/6] {_i_cap_msg} — BLOCKED')
                                _write_scan_log(_sym, sig.get('setup', 'I_mathematical'),
                                                sig.get('xgb_prob'), sig.get('direction', ''),
                                                sig.get('entry'), sig.get('stop'),
                                                sig.get('target'), None,
                                                'BLOCKED_STOP_CAP', _i_cap_msg[:200])
                                continue
                            _i_stop_dist = abs(float(sig.get('entry', 0)) - float(sig.get('stop', 0)))
                            logger.info(
                                f'[I-2c/6] Setup I {_sym}: stop cap OK — '
                                f'{_i_stop_dist:.1f}pts ≤ {_MAX_STOP_PTS.get(_sym, "?")}pts'
                            )

                            # ── [I-3/6] log_trade — CRITICAL on any failure ───────
                            _i_tid = None
                            try:
                                logger.info(
                                    f'[I-3/6] log_trade: {_sym} {sig["direction"]} '
                                    f'entry={sig.get("entry")} stop={sig.get("stop")} '
                                    f'target={sig.get("target")} setup={sig.get("setup")}'
                                )
                                _i_tid = log_trade(sig)
                                if _i_tid:
                                    logger.info(f'[I-4/6] log_trade returned trade_id={_i_tid}')
                                    _check_and_mark_fired(_sym, sig['setup'], sig['direction'])
                                    _write_scan_log(_sym, sig.get('setup', 'I_mathematical'),
                                                    sig.get('xgb_prob'), sig.get('direction', ''),
                                                    sig.get('entry'), sig.get('stop'),
                                                    sig.get('target'), None,
                                                    'EXECUTED', f'trade_id={_i_tid}')
                                else:
                                    logger.critical(
                                        f'[I-4/6] CRITICAL: log_trade() returned None — '
                                        f'{_sym} {sig["direction"]} entry={sig.get("entry")} NOT in DB'
                                    )
                            except Exception as _i_lte:
                                logger.critical(
                                    f'[I-4/6] CRITICAL: log_trade() EXCEPTION — '
                                    f'{_sym} {sig["direction"]}: {_i_lte}',
                                    exc_info=True
                                )

                            # ── [I-5/6] send_telegram — only after log_trade ──────
                            if not _i_tid:
                                logger.critical(
                                    f'[I-5/6] CRITICAL: Skipping Telegram — '
                                    f'log_trade did not return trade_id for {_sym}'
                                )
                            else:
                                try:
                                    logger.info(f'[I-5/6] send_telegram: {_sym} trade_id={_i_tid}')
                                    try:
                                        _i_risk_footer = (
                                            f'\n⚙️ <i>Regime: {_rg_i.regime.get_regime().label} | '
                                            f'Risk: {_rg_i.dd.get_risk_multiplier():.2f}× | '
                                            f'DD: {_rg_i.dd.get_drawdown_pct():.1f}%</i>'
                                        )
                                    except Exception:
                                        _i_risk_footer = ''
                                    send_telegram(format_i_alert(sig) + _i_risk_footer)
                                    logger.info(f'[I-5/6] Telegram sent OK — {_sym} trade_id={_i_tid}')
                                except Exception as _i_te:
                                    logger.critical(
                                        f'[I-5/6] CRITICAL: send_telegram failed {_sym}: {_i_te}',
                                        exc_info=True
                                    )

                            # ── [I-6/6] Tradovate execution ───────────────────────
                            if _i_tid:
                                logger.info(
                                    f'[I-6/6] {_sym} Setup I: executing on Tradovate '
                                    f'(trade_id={_i_tid})'
                                )
                                try:
                                    _execute_via_tradovate(sig, _i_tid)
                                    logger.info(f'[I-6/6] {_sym} Setup I: Tradovate call complete')
                                except Exception as _i_exe:
                                    logger.critical(
                                        f'[I-6/6] CRITICAL: _execute_via_tradovate raised '
                                        f'for {_sym}: {_i_exe}',
                                        exc_info=True
                                    )
                            else:
                                logger.critical(
                                    f'[I-6/6] CRITICAL: {_sym} Setup I — skipping Tradovate, '
                                    f'no trade_id (log_trade failed)'
                                )

                            logger.info(
                                f'Setup I signal complete: {_sym} {sig["direction"].upper()} '
                                f'xgb={sig["xgb_prob"]:.2f} lr={sig["lr_prob"]:.2f} '
                                f'db_id={_i_tid}'
                            )
                        except Exception as _i_sym_e:
                            logger.critical(
                                f'[I-FAIL] CRITICAL: Setup I {_sym} unhandled exception: {_i_sym_e}',
                                exc_info=True
                            )
                except Exception as _i_e:
                    logger.critical(
                        f'[I-FAIL] CRITICAL: Setup I scanner fatal error: {_i_e}',
                        exc_info=True
                    )

        # ── Setup J — Value Area Continuation (every 5 min, Tue-Fri session) ──
        # ES=LIVE | MNQ=LIVE | Backtest: ES Sharpe 8.34 | WR 54.7% | 11/11 months positive | MaxDD 4R
        if not hasattr(background_scheduler, '_last_setup_j'):
            background_scheduler._last_setup_j = 0
        if now - background_scheduler._last_setup_j >= 300:
            background_scheduler._last_setup_j = now
            if not is_setup_enabled('J'):
                logger.info('Setup J: disabled via Control Centre — skipping')
            else:
                try:
                    from setup_j_value_area import scan_setup_j, format_j_alert
                    from live_scanner import send_telegram
                    from trade_tracker import log_trade
                    _now_utc_j = datetime.now(timezone.utc)
                    if not hasattr(background_scheduler, '_risk_gate'):
                        from risk_manager import RiskGate
                        background_scheduler._risk_gate = RiskGate()
                    _rg_j = background_scheduler._risk_gate
                    for _sym_j in ['ES', 'MNQ']:
                        try:
                            # ── Pre-scan gates — every outcome logged ─────────────
                            _heat_j = _count_open_trades()
                            if _heat_j >= MAX_PORTFOLIO_HEAT:
                                logger.info(f'[J-pre] Setup J {_sym_j}: heat {_heat_j}/{MAX_PORTFOLIO_HEAT} — BLOCKED')
                                _write_scan_log(_sym_j, 'J_value_area', None, '', None, None, None, None,
                                                'BLOCKED_HEAT', f'heat={_heat_j}/{MAX_PORTFOLIO_HEAT}')
                                continue
                            logger.info(f'[J-pre] Setup J {_sym_j}: heat {_heat_j}/{MAX_PORTFOLIO_HEAT} — OK')

                            if SIGNAL_FILTERS['max_concurrent_per_instrument']:
                                _j_has, _j_id = _has_open_trade_on_instrument(_sym_j)
                                if _j_has:
                                    logger.info(f'[J-pre] Setup J {_sym_j}: open trade #{_j_id} — BLOCKED')
                                    _write_scan_log(_sym_j, 'J_value_area', None, '', None, None, None, None,
                                                    'BLOCKED_CONCURRENT', f'open trade #{_j_id}')
                                    continue
                                logger.info(f'[J-pre] Setup J {_sym_j}: no open trade — OK')

                            if _rg_j.daily.is_daily_limit_hit():
                                logger.info(f'[J-pre] Setup J {_sym_j}: daily limit hit — BLOCKED')
                                continue

                            if not is_setup_enabled('J', _sym_j):
                                logger.info(f'[J-pre] Setup J {_sym_j}: regime gate — BLOCKED')
                                _write_scan_log(_sym_j, 'J_value_area', None, '', None, None, None, None,
                                                'BLOCKED_REGIME', 'Setup J regime gate')
                                continue
                            logger.info(f'[J-pre] Setup J {_sym_j}: regime gate — OK')
                            try:
                                from setup_j_value_area import get_setup_j_state as _gjs
                                _js = _gjs(_sym_j)
                                _j_scan_note = (
                                    f'vah={_js.get("vah")} val={_js.get("val")} '
                                    f'bias={_js.get("htf_bias")} '
                                    f'session={_js.get("in_session")}'
                                )
                            except Exception:
                                _j_scan_note = 'Setup J entering scan'
                            _write_scan_log(_sym_j, 'J_value_area', None, '', None, None, None, None,
                                            'SCANNING', _j_scan_note)

                            # ── [J-1/6] Scan ─────────────────────────────────────
                            logger.info(f'[J-1/6] Setup J {_sym_j}: scanning...')
                            sig_j = scan_setup_j(_sym_j, _now_utc_j)
                            logger.info(
                                f'[J-1/6] scan_setup_j {_sym_j} returned: '
                                f'{"SIGNAL dir=" + sig_j["direction"] + " entry=" + str(sig_j.get("entry")) if sig_j else "None — no signal conditions met"}'
                            )
                            if not sig_j:
                                continue
                            _write_scan_log(_sym_j, sig_j.get('setup', 'J_value_area'),
                                            sig_j.get('score'), sig_j.get('direction', ''),
                                            sig_j.get('entry'), sig_j.get('stop'),
                                            sig_j.get('target'), None,
                                            'SIGNAL', f'VAH={sig_j.get("vah")} VAL={sig_j.get("val")}')

                            # ── [J-2a/6] Calendar ────────────────────────────────
                            if _cal_block(_sym_j, sig_j['setup']):
                                logger.info(f'[J-2a/6] Setup J {_sym_j}: calendar blackout — BLOCKED')
                                _write_scan_log(_sym_j, sig_j.get('setup', 'J_value_area'),
                                                sig_j.get('score'), sig_j.get('direction', ''),
                                                sig_j.get('entry'), sig_j.get('stop'),
                                                sig_j.get('target'), None,
                                                'BLOCKED_CALENDAR', 'calendar blackout')
                                continue
                            logger.info(f'[J-2a/6] Setup J {_sym_j}: calendar clear — OK')

                            # ── [J-2b/6] Dedup ───────────────────────────────────
                            if _is_signal_already_active(_sym_j, sig_j['direction'], sig_j['setup']):
                                logger.info(f'[J-2b/6] Setup J {_sym_j} {sig_j["direction"]}: already active in DB — BLOCKED')
                                continue
                            logger.info(f'[J-2b/6] Setup J {_sym_j} {sig_j["direction"]}: dedup — OK')

                            # ── [J-2c/6] Stop cap ────────────────────────────────
                            _j_cap_ok, _j_cap_msg = _stop_cap_ok(sig_j)
                            if not _j_cap_ok:
                                logger.warning(f'[J-2c/6] {_j_cap_msg} — BLOCKED')
                                _write_scan_log(_sym_j, sig_j.get('setup', 'J_value_area'),
                                                sig_j.get('score'), sig_j.get('direction', ''),
                                                sig_j.get('entry'), sig_j.get('stop'),
                                                sig_j.get('target'), None,
                                                'BLOCKED_STOP_CAP', _j_cap_msg[:200])
                                continue
                            _j_stop_dist = abs(float(sig_j.get('entry', 0)) - float(sig_j.get('stop', 0)))
                            logger.info(
                                f'[J-2c/6] Setup J {_sym_j}: stop cap OK — '
                                f'{_j_stop_dist:.1f}pts ≤ {_MAX_STOP_PTS.get(_sym_j, "?")}pts'
                            )

                            # ── [J-3/6] log_trade ────────────────────────────────
                            _j_tid = None
                            try:
                                logger.info(
                                    f'[J-3/6] log_trade: {_sym_j} {sig_j["direction"]} '
                                    f'entry={sig_j.get("entry")} stop={sig_j.get("stop")} '
                                    f'target={sig_j.get("target")} setup={sig_j.get("setup")}'
                                )
                                _j_tid = log_trade(sig_j)
                                if _j_tid:
                                    logger.info(f'[J-4/6] log_trade returned trade_id={_j_tid}')
                                    _check_and_mark_fired(_sym_j, sig_j['setup'], sig_j['direction'])
                                    _write_scan_log(_sym_j, sig_j.get('setup', 'J_value_area'),
                                                    sig_j.get('score'), sig_j.get('direction', ''),
                                                    sig_j.get('entry'), sig_j.get('stop'),
                                                    sig_j.get('target'), None,
                                                    'EXECUTED', f'trade_id={_j_tid}')
                                else:
                                    logger.critical(
                                        f'[J-4/6] CRITICAL: log_trade() returned None — '
                                        f'{_sym_j} {sig_j["direction"]} NOT in DB'
                                    )
                            except Exception as _j_lte:
                                logger.critical(
                                    f'[J-4/6] CRITICAL: log_trade() EXCEPTION — '
                                    f'{_sym_j}: {_j_lte}', exc_info=True
                                )

                            # ── [J-5/6] send_telegram — only after log_trade ──────
                            if not _j_tid:
                                logger.critical(
                                    f'[J-5/6] CRITICAL: Skipping Telegram — '
                                    f'log_trade did not return trade_id for {_sym_j}'
                                )
                            else:
                                try:
                                    logger.info(f'[J-5/6] send_telegram: {_sym_j} trade_id={_j_tid}')
                                    try:
                                        _j_risk_footer = (
                                            f'\n⚙️ <i>Regime: {_rg_j.regime.get_regime().label} | '
                                            f'Risk: {_rg_j.dd.get_risk_multiplier():.2f}× | '
                                            f'DD: {_rg_j.dd.get_drawdown_pct():.1f}%</i>'
                                        )
                                    except Exception:
                                        _j_risk_footer = ''
                                    send_telegram(format_j_alert(sig_j) + _j_risk_footer)
                                    logger.info(f'[J-5/6] Telegram sent OK — {_sym_j} trade_id={_j_tid}')
                                except Exception as _j_te:
                                    logger.critical(
                                        f'[J-5/6] CRITICAL: send_telegram failed {_sym_j}: {_j_te}',
                                        exc_info=True
                                    )

                            # ── [J-6/6] Tradovate execution ───────────────────────
                            if _j_tid:
                                logger.info(
                                    f'[J-6/6] {_sym_j} Setup J: executing on Tradovate '
                                    f'(trade_id={_j_tid})'
                                )
                                try:
                                    _execute_via_tradovate(sig_j, _j_tid)
                                    logger.info(f'[J-6/6] {_sym_j} Setup J: Tradovate call complete')
                                except Exception as _j_exe:
                                    logger.critical(
                                        f'[J-6/6] CRITICAL: _execute_via_tradovate raised for {_sym_j}: {_j_exe}',
                                        exc_info=True
                                    )
                            else:
                                logger.critical(
                                    f'[J-6/6] CRITICAL: {_sym_j} Setup J — skipping Tradovate, no trade_id'
                                )

                            logger.info(
                                f'Setup J signal complete: {_sym_j} {sig_j["direction"].upper()} '
                                f'VAH={sig_j.get("vah")} VAL={sig_j.get("val")} '
                                f'db_id={_j_tid}'
                            )
                        except Exception as _j_sym_e:
                            logger.critical(
                                f'[J-FAIL] CRITICAL: Setup J {_sym_j} unhandled exception: {_j_sym_e}',
                                exc_info=True
                            )
                except Exception as _j_e:
                    logger.critical(
                        f'[J-FAIL] CRITICAL: Setup J scanner fatal error: {_j_e}',
                        exc_info=True
                    )

        # ── Setup D — FVG Fill (every 5 min, NQ+ES only) ─────────
        if not hasattr(background_scheduler, '_last_setup_d'):
            background_scheduler._last_setup_d = 0
        if now - background_scheduler._last_setup_d >= 300:
            background_scheduler._last_setup_d = now
            if not is_setup_enabled('D'):
                logger.info('Setup D: disabled via Control Centre — skipping')
            else:
                try:
                    from fvg_engine import scan_setup_d, format_d_alert
                    from live_scanner import send_telegram as _send_tg
                    from trade_tracker import log_trade as _log_trade
                    _now_utc_d = datetime.now(timezone.utc)
                    for _sym_d in ['MNQ', 'ES']:  # GC disabled — feed unverified
                        try:
                            _heat_d = _count_open_trades()
                            if _heat_d >= MAX_PORTFOLIO_HEAT:
                                logger.info(f'Portfolio heat: {_heat_d} trades open — new signal blocked for Setup D {_sym_d}')
                                _write_scan_log(_sym_d, 'D_fvg_fill', None, '', None, None, None, None,
                                                'BLOCKED_HEAT', f'heat={_heat_d}/{MAX_PORTFOLIO_HEAT}')
                                continue
                            if not is_setup_enabled('D', _sym_d):
                                logger.info(f'Setup D {_sym_d}: regime gate — BLOCKED')
                                _write_scan_log(_sym_d, 'D_fvg_fill', None, '', None, None, None, None,
                                                'BLOCKED_REGIME', 'Setup D regime gate')
                                continue
                            logger.info(f'Scanning Setup D for {_sym_d}')
                            sig_d = scan_setup_d(_sym_d, _now_utc_d)
                            if not sig_d:
                                continue
                            # CANDIDATE = FVG found, downstream gates (calendar/concurrent/stop_cap) still pending
                            _write_scan_log(_sym_d, sig_d.get('setup', 'D_fvg_fill'),
                                            sig_d.get('fvg_score'), sig_d.get('direction', ''),
                                            sig_d.get('entry'), sig_d.get('stop'),
                                            sig_d.get('target'), sig_d.get('target2'),
                                            'CANDIDATE', 'scan_setup_d generated')
                            if SIGNAL_FILTERS['max_concurrent_per_instrument']:
                                _f1d_has, _f1d_id = _has_open_trade_on_instrument(_sym_d)
                                if _f1d_has:
                                    logger.info(f'Setup D: skipped — {_sym_d} already has open trade #{_f1d_id}')
                                    _write_scan_log(_sym_d, sig_d.get('setup', 'D_fvg_fill'),
                                                    sig_d.get('fvg_score'), sig_d.get('direction', ''),
                                                    sig_d.get('entry'), sig_d.get('stop'),
                                                    sig_d.get('target'), sig_d.get('target2'),
                                                    'BLOCKED_CONCURRENT', f'open trade #{_f1d_id}')
                                    continue
                            if SIGNAL_FILTERS['primary_session_only_bcd']:
                                if sig_d.get('quality', 'primary') != 'primary':
                                    logger.info(f'Setup D: skipped — secondary session quality, requires primary')
                                    continue
                            if SIGNAL_FILTERS['dual_htf_bias']:
                                _f3d_ok, _f3d_detail = _check_1h_bias(_sym_d, sig_d['direction'])
                                if not _f3d_ok:
                                    logger.info(
                                        f'Setup D: skipped — 1h bias conflicts with {sig_d["direction"]} '
                                        f'direction ({_f3d_detail})'
                                    )
                                    _write_scan_log(_sym_d, sig_d.get('setup', 'D_fvg_fill'),
                                                    sig_d.get('fvg_score'), sig_d.get('direction', ''),
                                                    sig_d.get('entry'), sig_d.get('stop'),
                                                    sig_d.get('target'), sig_d.get('target2'),
                                                    'BLOCKED_HTF_BIAS', _f3d_detail[:200])
                                    continue
                            if _cal_block(_sym_d, sig_d['setup']):
                                _write_scan_log(_sym_d, sig_d.get('setup', 'D_fvg_fill'),
                                                sig_d.get('fvg_score'), sig_d.get('direction', ''),
                                                sig_d.get('entry'), sig_d.get('stop'),
                                                sig_d.get('target'), sig_d.get('target2'),
                                                'BLOCKED_CALENDAR', 'calendar gate')
                                continue
                            if _check_and_mark_fired(_sym_d, sig_d['setup'], sig_d['direction']):
                                continue
                            _d_stop_pts = round(abs(sig_d['entry'] - sig_d['stop']), 1)
                            _d_risk_usd = round(_d_stop_pts * (2 if _sym_d == 'MNQ' else 50), 0)
                            logger.info(
                                f'Signal: {_sym_d} {sig_d["direction"]} {sig_d["setup"]} | '
                                f'stop={_d_stop_pts} pts | risk=${_d_risk_usd:.0f} | contracts=1'
                            )
                            _d_cap_ok, _d_cap_msg = _stop_cap_ok(sig_d)
                            if not _d_cap_ok:
                                logger.warning(_d_cap_msg)
                                _write_scan_log(_sym_d, sig_d.get('setup', 'D_fvg_fill'),
                                                sig_d.get('fvg_score'), sig_d.get('direction', ''),
                                                sig_d.get('entry'), sig_d.get('stop'),
                                                sig_d.get('target'), sig_d.get('target2'),
                                                'BLOCKED_STOP_CAP', _d_cap_msg[:200])
                                continue
                            # All gates passed — log SIGNAL now
                            _write_scan_log(_sym_d, sig_d.get('setup', 'D_fvg_fill'),
                                            sig_d.get('fvg_score'), sig_d.get('direction', ''),
                                            sig_d.get('entry'), sig_d.get('stop'),
                                            sig_d.get('target'), sig_d.get('target2'),
                                            'SIGNAL', 'all gates passed — entering')
                            _d_tid = None
                            try:
                                _d_tid = _log_trade(sig_d)
                                if _d_tid:
                                    logger.info(f'Setup D: trade logged id={_d_tid} {_sym_d} {sig_d["direction"]}')
                                    _write_scan_log(_sym_d, sig_d.get('setup', 'D_fvg_fill'),
                                                    sig_d.get('fvg_score'), sig_d.get('direction', ''),
                                                    sig_d.get('entry'), sig_d.get('stop'),
                                                    sig_d.get('target'), sig_d.get('target2'),
                                                    'EXECUTED', f'trade_id={_d_tid}')
                                else:
                                    logger.error(f'CRITICAL: Setup D log_trade() returned None — {_sym_d} {sig_d["direction"]} NOT in DB')
                            except Exception as _d_lte:
                                logger.error(f'CRITICAL: Setup D log_trade() EXCEPTION — {_sym_d}: {_d_lte}', exc_info=True)
                            if not _d_tid:
                                logger.critical(f'CRITICAL: Skipping Telegram for Setup D {_sym_d} — log_trade returned None')
                            else:
                                try:
                                    _send_tg(format_d_alert(sig_d))
                                except Exception as _d_te:
                                    logger.error(f'Setup D send_telegram failed {_sym_d}: {_d_te}')
                            if _d_tid:
                                logger.info(f'Calling _execute_via_tradovate: {_sym_d} {sig_d["direction"]} {sig_d["setup"]} trade_id={_d_tid}')
                                _execute_via_tradovate(sig_d, _d_tid)
                            else:
                                logger.warning(f'Skipping _execute_via_tradovate — log_trade returned None for Setup D {_sym_d}')
                            logger.info(f'Setup D signal: {_sym_d} {sig_d["direction"].upper()} score={sig_d.get("fvg_score")} db_id={_d_tid}')
                        except Exception as _de:
                            logger.warning(f'Setup D {_sym_d} error: {_de}')
                except Exception as e:
                    logger.warning(f'Setup D scanner error: {e}')

        # ── Setup G — Wyckoff Upthrust Tracker (every 5 min) ─────
        if not hasattr(background_scheduler, '_last_wyckoff'):
            background_scheduler._last_wyckoff = 0
        if now - background_scheduler._last_wyckoff >= 300:
            background_scheduler._last_wyckoff = now
            try:
                from wyckoff_tracker import scan_and_log_wyckoff
                scan_and_log_wyckoff()
            except Exception as e:
                logger.debug(f'Wyckoff tracker error: {e}')

        # ── Setup H — VWAP 2σ Reversal (every 5 min) ─────────────
        if not hasattr(background_scheduler, '_last_setup_h'):
            background_scheduler._last_setup_h = 0
        if now - background_scheduler._last_setup_h >= 300:
            background_scheduler._last_setup_h = now
            if not is_setup_enabled('H'):
                logger.info('Setup H: disabled via Control Centre — skipping')
            else:
                try:
                    from setup_h_vwap import scan_setup_h, format_h_alert, log_h_paper
                    from live_scanner import send_telegram
                    from trade_tracker import log_trade
                    _now_utc = datetime.now(timezone.utc)
                    if not hasattr(background_scheduler, '_risk_gate'):
                        from risk_manager import RiskGate
                        background_scheduler._risk_gate = RiskGate()
                    _rg = background_scheduler._risk_gate
                    _regime  = _rg.regime.get_regime()
                    _dd_mult = _rg.dd.get_risk_multiplier()
                    _risk_footer = (
                        f'\n⚙️ <i>Regime: {_regime.label} | '
                        f'Risk: {_dd_mult:.2f}× | DD: {_rg.dd.get_drawdown_pct():.1f}%</i>'
                    )
                    try:
                        _heat_h = _count_open_trades()
                        if _heat_h >= MAX_PORTFOLIO_HEAT:
                            logger.info(f'Portfolio heat: {_heat_h} trades open — new signal blocked for Setup H')
                            _write_scan_log('ES', 'H_vwap_reversal', None, '', None, None, None, None,
                                            'BLOCKED_HEAT', f'heat={_heat_h}/{MAX_PORTFOLIO_HEAT}')
                        elif not _rg.daily.is_daily_limit_hit():
                            if not is_setup_enabled('H', 'ES'):
                                logger.info('Setup H ES: regime gate blocked — skipping')
                                _write_scan_log('ES', 'H_vwap_reversal', None, '', None, None, None, None,
                                                'BLOCKED_REGIME', 'Setup H ES regime gate')
                                sig_es = None
                            else:
                                try:
                                    from setup_h_vwap import get_h_state as _gh_es
                                    _hs_es = _gh_es('ES')
                                    _h_scan_note_es = (
                                        f'state={_hs_es.get("signal_state","?")} '
                                        f'price={_hs_es.get("price")} '
                                        f'upper={_hs_es.get("upper_band")} '
                                        f'lower={_hs_es.get("lower_band")} '
                                        f'vwap={_hs_es.get("vwap")} '
                                        f'bias={_hs_es.get("htf_bias")}'
                                    )
                                except Exception:
                                    _h_scan_note_es = 'Setup H ES entering scanner'
                                _write_scan_log('ES', 'H_vwap_reversal', None, '', None, None, None, None,
                                                'SCANNING', _h_scan_note_es)
                                sig_es = scan_setup_h('ES', _now_utc, paper_only=False)
                            logger.info(
                                f'scan_setup_h ES returned: '
                                f'{"SIGNAL dir=" + sig_es["direction"] if sig_es else "None (no signal this tick)"}'
                            )
                            if sig_es:
                                _write_scan_log('ES', sig_es.get('setup', 'H_vwap_reversal'),
                                                sig_es.get('score'), sig_es.get('direction', ''),
                                                sig_es.get('entry'), sig_es.get('stop'),
                                                sig_es.get('target'), None,
                                                'SIGNAL', 'scan_setup_h ES generated')
                                if SIGNAL_FILTERS['max_concurrent_per_instrument']:
                                    _f1h_has, _f1h_id = _has_open_trade_on_instrument('ES')
                                    if _f1h_has:
                                        logger.info(f'Setup H: skipped — ES already has open trade #{_f1h_id}')
                                        _write_scan_log('ES', sig_es.get('setup', 'H_vwap_reversal'),
                                                        sig_es.get('score'), sig_es.get('direction', ''),
                                                        sig_es.get('entry'), sig_es.get('stop'),
                                                        sig_es.get('target'), None,
                                                        'BLOCKED_CONCURRENT', f'open trade #{_f1h_id}')
                                        sig_es = None
                            if sig_es:
                                if _has_opposite_swing_trade('ES', sig_es['direction']):
                                    logger.info('Setup H ES suppressed — opposite swing trade open')
                                    _write_scan_log('ES', sig_es.get('setup', 'H_vwap_reversal'),
                                                    sig_es.get('score'), sig_es.get('direction', ''),
                                                    sig_es.get('entry'), sig_es.get('stop'),
                                                    sig_es.get('target'), None,
                                                    'BLOCKED_CONCURRENT', 'opposite swing trade open')
                                elif _cal_block('ES', sig_es['setup']):
                                    _write_scan_log('ES', sig_es.get('setup', 'H_vwap_reversal'),
                                                    sig_es.get('score'), sig_es.get('direction', ''),
                                                    sig_es.get('entry'), sig_es.get('stop'),
                                                    sig_es.get('target'), None,
                                                    'BLOCKED_CALENDAR', 'calendar gate')
                                elif not _check_and_mark_fired('ES', sig_es['setup'], sig_es['direction']):
                                    if not _is_signal_already_active('ES', sig_es['direction'], sig_es['setup']):
                                        _h_es_cap_ok, _h_es_cap_msg = _stop_cap_ok(sig_es)
                                        if not _h_es_cap_ok:
                                            logger.warning(_h_es_cap_msg)
                                            _write_scan_log('ES', sig_es.get('setup', 'H_vwap_reversal'),
                                                            sig_es.get('score'), sig_es.get('direction', ''),
                                                            sig_es.get('entry'), sig_es.get('stop'),
                                                            sig_es.get('target'), None,
                                                            'BLOCKED_STOP_CAP', _h_es_cap_msg[:200])
                                        else:
                                            _h_tid = None
                                            try:
                                                _h_tid = log_trade(sig_es)
                                                if _h_tid:
                                                    logger.info(f'Trade logged: id={_h_tid} ES {sig_es["direction"]} {sig_es["setup"]}')
                                                    _write_scan_log('ES', sig_es.get('setup', 'H_vwap_reversal'),
                                                                    sig_es.get('score'), sig_es.get('direction', ''),
                                                                    sig_es.get('entry'), sig_es.get('stop'),
                                                                    sig_es.get('target'), None,
                                                                    'EXECUTED', f'trade_id={_h_tid}')
                                                else:
                                                    logger.error('CRITICAL: Setup H log_trade() returned None — ES NOT in DB')
                                            except Exception as _lte:
                                                logger.error(f'CRITICAL: Setup H log_trade() EXCEPTION — ES {sig_es["direction"]}: {_lte}', exc_info=True)
                                            if not _h_tid:
                                                logger.critical('CRITICAL: Skipping Telegram for Setup H ES — log_trade returned None')
                                            else:
                                                try:
                                                    msg = format_h_alert(sig_es) + _risk_footer
                                                    send_telegram(msg)
                                                except Exception as _h_te:
                                                    logger.error(f'Setup H send_telegram failed ES: {_h_te}')
                                            if _h_tid:
                                                logger.info(f'Calling _execute_via_tradovate: ES {sig_es["direction"]} {sig_es["setup"]} trade_id={_h_tid}')
                                                _execute_via_tradovate(sig_es, _h_tid)
                                            else:
                                                logger.warning('Skipping _execute_via_tradovate — log_trade returned None for Setup H ES')
                                            logger.info(f'Setup H ES {sig_es["direction"].upper()} signal fired db_id={_h_tid}')
                        else:
                            logger.info('Setup H ES: daily limit hit — suppressed')
                    except Exception as _he:
                        logger.warning(f'Setup H ES error: {_he}')
                    try:
                        _heat_h_mnq = _count_open_trades()
                        if _heat_h_mnq >= MAX_PORTFOLIO_HEAT:
                            logger.info(f'Portfolio heat: {_heat_h_mnq} trades open — new signal blocked for Setup H MNQ')
                            _write_scan_log('MNQ', 'H_vwap_reversal', None, '', None, None, None, None,
                                            'BLOCKED_HEAT', f'heat={_heat_h_mnq}/{MAX_PORTFOLIO_HEAT}')
                        elif not _rg.daily.is_daily_limit_hit():
                            if not is_setup_enabled('H', 'MNQ'):
                                logger.info('Setup H MNQ: regime gate blocked — skipping')
                                _write_scan_log('MNQ', 'H_vwap_reversal', None, '', None, None, None, None,
                                                'BLOCKED_REGIME', 'Setup H MNQ regime gate')
                                sig_mnq_h = None
                            else:
                                try:
                                    from setup_h_vwap import get_h_state as _gh_mnq
                                    _hs_mnq = _gh_mnq('MNQ')
                                    _h_scan_note_mnq = (
                                        f'state={_hs_mnq.get("signal_state","?")} '
                                        f'price={_hs_mnq.get("price")} '
                                        f'upper={_hs_mnq.get("upper_band")} '
                                        f'lower={_hs_mnq.get("lower_band")} '
                                        f'vwap={_hs_mnq.get("vwap")} '
                                        f'bias={_hs_mnq.get("htf_bias")}'
                                    )
                                except Exception:
                                    _h_scan_note_mnq = 'Setup H MNQ entering scanner'
                                _write_scan_log('MNQ', 'H_vwap_reversal', None, '', None, None, None, None,
                                                'SCANNING', _h_scan_note_mnq)
                                sig_mnq_h = scan_setup_h('MNQ', _now_utc, paper_only=False)
                            logger.info(
                                f'scan_setup_h MNQ returned: '
                                f'{"SIGNAL dir=" + sig_mnq_h["direction"] if sig_mnq_h else "None (no signal this tick)"}'
                            )
                            if sig_mnq_h:
                                _write_scan_log('MNQ', sig_mnq_h.get('setup', 'H_vwap_reversal'),
                                                sig_mnq_h.get('score'), sig_mnq_h.get('direction', ''),
                                                sig_mnq_h.get('entry'), sig_mnq_h.get('stop'),
                                                sig_mnq_h.get('target'), None,
                                                'SIGNAL', 'scan_setup_h MNQ generated')
                                if SIGNAL_FILTERS['max_concurrent_per_instrument']:
                                    _f1hm_has, _f1hm_id = _has_open_trade_on_instrument('MNQ')
                                    if _f1hm_has:
                                        logger.info(f'Setup H: skipped — MNQ already has open trade #{_f1hm_id}')
                                        _write_scan_log('MNQ', sig_mnq_h.get('setup', 'H_vwap_reversal'),
                                                        sig_mnq_h.get('score'), sig_mnq_h.get('direction', ''),
                                                        sig_mnq_h.get('entry'), sig_mnq_h.get('stop'),
                                                        sig_mnq_h.get('target'), None,
                                                        'BLOCKED_CONCURRENT', f'open trade #{_f1hm_id}')
                                        sig_mnq_h = None
                            if sig_mnq_h:
                                if _has_opposite_swing_trade('MNQ', sig_mnq_h['direction']):
                                    logger.info('Setup H MNQ suppressed — opposite swing trade open')
                                    _write_scan_log('MNQ', sig_mnq_h.get('setup', 'H_vwap_reversal'),
                                                    sig_mnq_h.get('score'), sig_mnq_h.get('direction', ''),
                                                    sig_mnq_h.get('entry'), sig_mnq_h.get('stop'),
                                                    sig_mnq_h.get('target'), None,
                                                    'BLOCKED_CONCURRENT', 'opposite swing trade open')
                                elif _cal_block('MNQ', sig_mnq_h['setup']):
                                    _write_scan_log('MNQ', sig_mnq_h.get('setup', 'H_vwap_reversal'),
                                                    sig_mnq_h.get('score'), sig_mnq_h.get('direction', ''),
                                                    sig_mnq_h.get('entry'), sig_mnq_h.get('stop'),
                                                    sig_mnq_h.get('target'), None,
                                                    'BLOCKED_CALENDAR', 'calendar gate')
                                elif not _check_and_mark_fired('MNQ', sig_mnq_h['setup'], sig_mnq_h['direction']):
                                    if not _is_signal_already_active('MNQ', sig_mnq_h['direction'], sig_mnq_h['setup']):
                                        _hm_cap_ok, _hm_cap_msg = _stop_cap_ok(sig_mnq_h)
                                        if not _hm_cap_ok:
                                            logger.warning(_hm_cap_msg)
                                            _write_scan_log('MNQ', sig_mnq_h.get('setup', 'H_vwap_reversal'),
                                                            sig_mnq_h.get('score'), sig_mnq_h.get('direction', ''),
                                                            sig_mnq_h.get('entry'), sig_mnq_h.get('stop'),
                                                            sig_mnq_h.get('target'), None,
                                                            'BLOCKED_STOP_CAP', _hm_cap_msg[:200])
                                        else:
                                            _hm_tid = None
                                            try:
                                                _hm_tid = log_trade(sig_mnq_h)
                                                if _hm_tid:
                                                    logger.info(f'Trade logged: id={_hm_tid} MNQ {sig_mnq_h["direction"]} {sig_mnq_h["setup"]}')
                                                    _write_scan_log('MNQ', sig_mnq_h.get('setup', 'H_vwap_reversal'),
                                                                    sig_mnq_h.get('score'), sig_mnq_h.get('direction', ''),
                                                                    sig_mnq_h.get('entry'), sig_mnq_h.get('stop'),
                                                                    sig_mnq_h.get('target'), None,
                                                                    'EXECUTED', f'trade_id={_hm_tid}')
                                                else:
                                                    logger.error('CRITICAL: Setup H log_trade() returned None — MNQ NOT in DB')
                                            except Exception as _lte_hm:
                                                logger.error(f'CRITICAL: Setup H log_trade() EXCEPTION — MNQ: {_lte_hm}', exc_info=True)
                                            if not _hm_tid:
                                                logger.critical('CRITICAL: Skipping Telegram for Setup H MNQ — log_trade returned None')
                                            else:
                                                try:
                                                    msg = format_h_alert(sig_mnq_h) + _risk_footer
                                                    send_telegram(msg)
                                                except Exception as _h_te_m:
                                                    logger.error(f'Setup H send_telegram failed MNQ: {_h_te_m}')
                                                _execute_via_tradovate(sig_mnq_h, _hm_tid)
                                            logger.info(f'Setup H MNQ {sig_mnq_h["direction"].upper()} signal fired db_id={_hm_tid}')
                    except Exception as _hmnq:
                        logger.warning(f'Setup H MNQ error: {_hmnq}')
                except Exception as e:
                    logger.warning(f'Setup H scanner error: {e}')

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
        conn = _db.connect()
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
        conn = _db.connect()
        for _tf in ('5min', '15min'):
            n = conn.execute('SELECT COUNT(*) FROM ohlcv WHERE ts=0 AND timeframe=?', (_tf,)).fetchone()[0]
            if n > 0:
                conn.execute('DELETE FROM ohlcv WHERE ts=0 AND timeframe=?', (_tf,))
                conn.commit()
                logger.info(f'  DB cleanup: removed {n} corrupted ts=0 rows from ohlcv {_tf}')
            else:
                logger.info(f'  DB cleanup: no ts=0 rows in ohlcv {_tf} — clean')
        conn.close()
    except Exception as e:
        logger.warning(f'  DB ts=0 cleanup error: {e}')
    # ── Strategy parameter summary — visible in Railway logs ──
    logger.info('  STRATEGY PARAMETERS')
    logger.info('  ─────────────────────────────────────────────────')
    logger.info('  A/B/C  stop_atr=0.8  RR=4.0  MNQ/ES London primary (7-11 UTC) / NY secondary (13-19 UTC)')
    logger.info('  A/B/C  MNQ day filter: Tue/Wed/Thu | ES: Mon/Tue/Thu | GC: Mon-Fri')
    logger.info('  D      stop_atr=1.0  RR=2.5  MNQ+ES  session=13-19 UTC  min_score=70')
    logger.info('  E      stop_atr=1.5  RR=2.5  target=3.75×ATR  MNQ only  session=13-18 UTC')
    logger.info('  F      stop_atr=1.5  RR=2.5  target=3.75×ATR  MNQ+ES   long>0.65  short<0.35  cap=2/session')
    logger.info('  H      stop_atr=1.5  RR≥2.0  target=VWAP  ES live/NQ paper  session=13-19 UTC')
    logger.info('  ─────────────────────────────────────────────────')
    logger.info('  SIGNAL FILTERS')
    for _fk, _fv in SIGNAL_FILTERS.items():
        logger.info(f'  {"ON " if _fv else "OFF"} {_fk}')
    logger.info(f'  setup_e_min_atr = {SETUP_E_MIN_ATR}')
    logger.info('  ─────────────────────────────────────────────────')
    try:
        from trade_tracker import init_trades_table as _itt
        _itt()
        logger.info('  apex_trades table ready')
        # Migrate legacy NQ open trades → MNQ (NQ was renamed to MNQ)
        try:
            _mc = _db.connect()
            _migrated = _mc.execute(
                "UPDATE apex_trades SET symbol='MNQ' WHERE symbol='NQ' AND status='open'"
            ).rowcount
            _mc.commit()
            _mc.close()
            if _migrated:
                logger.info(f'  Migrated {_migrated} legacy NQ open trades to MNQ')
        except Exception as _me:
            logger.warning(f'  NQ→MNQ migration failed: {_me}')
        # ── Cleanup test trades that pollute P&L calculations ──────────
        # Deletes: quality='test', exit_reason in test_*, |pnl_r| > 10.
        # Real trades never match these conditions. Safe to run on every startup.
        try:
            _cc = _db.connect()
            _before = _cc.execute("SELECT COUNT(*) FROM apex_trades").fetchone()[0]
            _deleted = _cc.execute("""
                DELETE FROM apex_trades
                WHERE quality='test'
                   OR exit_reason IN ('test_endpoint','test_cleanup','test_simulation')
                   OR (pnl_r IS NOT NULL AND ABS(pnl_r) > 10)
            """).rowcount
            _cc.commit()
            _after = _cc.execute("SELECT COUNT(*) FROM apex_trades").fetchone()[0]
            _real_r = _cc.execute(
                "SELECT COALESCE(SUM(pnl_r),0) FROM apex_trades "
                "WHERE status='closed' AND pnl_r IS NOT NULL"
            ).fetchone()[0]
            _cc.close()
            logger.info(
                f'  Test-trade cleanup: removed {_deleted} rows '
                f'({_before} → {_after} remaining, total_R={_real_r:+.2f}R)'
            )
        except Exception as _ce:
            logger.warning(f'  Test-trade cleanup failed: {_ce}')
    except Exception as e:
        logger.warning('  apex_trades init failed: ' + str(e))
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
    try:
        import research_division as _rd_mod
        _rd_mod.seed_shadow_lab_candidates()
        logger.info('Research Division: initialised successfully')
    except Exception as _rde:
        logger.error(
            f'Research Division: init failed — {_rde} — trading continues normally',
            exc_info=True
        )
    _seed_strategy_config()
    # Pre-warm scan-critical modules — imports only, no inference.
    # Model .pkl loading happens on first scan request (~15-20s acceptable with 45s test timeout).
    # Do NOT call get_setup_i_state()/get_current_prediction() here — they load DB + run
    # inference synchronously, adding 20-40s to startup and can exceed Railway's port-bind timeout.
    try:
        from setup_engine import check_setup, check_setup_a, check_setup_c  # noqa: F401
        from setup_e import check_setup_e                                    # noqa: F401
        from fvg_engine import scan_setup_d                                  # noqa: F401
        from setup_h_vwap import get_h_state                                 # noqa: F401
        from setup_i_mathematical import get_setup_i_state                   # noqa: F401
        from setup_f_ml import get_current_prediction                        # noqa: F401
        logger.info('  Scan modules pre-warmed (imports cached)')
    except Exception as _pw:
        logger.warning(f'  Scan module pre-warm failed: {_pw}')
    def startup_backfill():
        import time as _t
        _t.sleep(10)
        # Check all Databento-supported instruments, not just ES
        _db_syms = [s for s, info in INSTRUMENTS.items() if info.get('databento')]
        for _sym in _db_syms:
            try:
                conn = _db.connect()
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
        # Ensure sufficient 5min history for Setup F training (needs ~10000 bars / 6 months)
        for _sym in ['MNQ', 'ES']:
            try:
                conn = _db.connect()
                count_5m = conn.execute(
                    "SELECT COUNT(*) FROM ohlcv WHERE symbol=? AND timeframe='5min'", (_sym,)
                ).fetchone()[0]
                conn.close()
                if count_5m < 10000:
                    logger.info(
                        f'Setup F: {_sym} only has {count_5m} 5min bars — '
                        f'fetching 6 months of history in monthly chunks...'
                    )
                    try:
                        from data_feed import fetch_bars_range, store_bars
                        from datetime import date
                        import calendar
                        total_stored = 0
                        # Sep 2024 → current month, one month at a time
                        cur_year, cur_month = 2024, 9
                        now_dt = datetime.now(timezone.utc)
                        while (cur_year, cur_month) <= (now_dt.year, now_dt.month):
                            month_label = date(cur_year, cur_month, 1).strftime('%b %Y')
                            try:
                                days_in_month = calendar.monthrange(cur_year, cur_month)[1]
                                chunk_start = datetime(cur_year, cur_month, 1, tzinfo=timezone.utc)
                                chunk_end   = min(
                                    datetime(cur_year, cur_month, days_in_month, 23, 59, tzinfo=timezone.utc),
                                    now_dt
                                )
                                bars = fetch_bars_range(_sym, '5min', chunk_start, chunk_end)
                                stored = store_bars(_sym, '5min', bars)
                                total_stored += stored
                                logger.info(f'Backfill: {_sym} 5min {month_label}... done ({stored} bars)')
                            except Exception as _me:
                                logger.warning(f'Backfill: {_sym} 5min {month_label} failed: {_me}')
                            # advance month
                            if cur_month == 12:
                                cur_year += 1
                                cur_month = 1
                            else:
                                cur_month += 1
                            _t.sleep(2)
                        conn = _db.connect()
                        final_count = conn.execute(
                            "SELECT COUNT(*) FROM ohlcv WHERE symbol=? AND timeframe='5min'",
                            (_sym,)
                        ).fetchone()[0]
                        conn.close()
                        logger.info(f'Backfill complete: {_sym} 5min — {final_count} total bars stored')
                    except Exception as _be:
                        logger.warning(f'Setup F chunked backfill {_sym}: {_be}')
                else:
                    logger.info(f'Setup F: {_sym} 5min data sufficient ({count_5m} bars)')
            except Exception as e:
                logger.warning(f'Setup F backfill check {_sym}: {e}')
        # Build 1h/4h aggregates from 5min bars — required by Setup F feature engineering
        try:
            from data_feed import build_htf_from_5min
            for _sym in ['MNQ', 'ES']:
                try:
                    htf = build_htf_from_5min(_sym)
                    logger.info(f'Setup F: {_sym} HTF built — {htf}')
                except Exception as _he:
                    logger.warning(f'Setup F: {_sym} HTF build error: {_he}')
        except Exception as e:
            logger.warning(f'Setup F HTF build failed: {e}')

        # ── MNQ 1min historical backfill (6 months) ──────────────
        # MNQ is data-collection only — no signals. We backfill once on startup
        # so backtesting can begin immediately. Chunked monthly to avoid timeouts.
        try:
            conn = _db.connect()
            mnq_1min_count = conn.execute(
                "SELECT COUNT(*) FROM ohlcv WHERE symbol='MNQ' AND timeframe='1min'"
            ).fetchone()[0]
            conn.close()
            if mnq_1min_count < 5000:
                logger.info(
                    f'MNQ: only {mnq_1min_count} 1min bars — '
                    f'fetching 6 months of history in monthly chunks...'
                )
                try:
                    from data_feed import fetch_bars_range, store_bars
                    from datetime import date
                    import calendar as _cal
                    mnq_total = 0
                    now_dt = datetime.now(timezone.utc)
                    # 6 months back from today
                    start_dt = now_dt - timedelta(days=183)
                    cur_year, cur_month = start_dt.year, start_dt.month
                    while (cur_year, cur_month) <= (now_dt.year, now_dt.month):
                        month_label = date(cur_year, cur_month, 1).strftime('%b %Y')
                        try:
                            days_in_month = _cal.monthrange(cur_year, cur_month)[1]
                            chunk_start = datetime(cur_year, cur_month, 1, tzinfo=timezone.utc)
                            chunk_end   = min(
                                datetime(cur_year, cur_month, days_in_month, 23, 59, tzinfo=timezone.utc),
                                now_dt
                            )
                            bars = fetch_bars_range('MNQ', '1min', chunk_start, chunk_end)
                            stored = store_bars('MNQ', '1min', bars)
                            mnq_total += stored
                            logger.info(f'MNQ backfill: 1min {month_label}... done ({stored} bars)')
                        except Exception as _me:
                            logger.warning(f'MNQ backfill: 1min {month_label} failed: {_me}')
                        if cur_month == 12:
                            cur_year += 1
                            cur_month = 1
                        else:
                            cur_month += 1
                        _t.sleep(2)
                    conn = _db.connect()
                    mnq_final = conn.execute(
                        "SELECT COUNT(*) FROM ohlcv WHERE symbol='MNQ' AND timeframe='1min'"
                    ).fetchone()[0]
                    conn.close()
                    logger.info(f'MNQ backfill complete: {mnq_final} total 1min bars stored')
                except Exception as _be:
                    logger.warning(f'MNQ chunked backfill error: {_be}')
            else:
                logger.info(f'MNQ: 1min data sufficient ({mnq_1min_count} bars) — backfill skipped')
        except Exception as _mnqe:
            logger.warning(f'MNQ backfill check failed: {_mnqe}')
        # Train Setup F ML models after data is ready
        _t.sleep(5)
        try:
            from setup_f_ml import train_model, load_or_train_model
            for _sym in ['MNQ', 'ES']:
                try:
                    logger.info(f'Setup F: Training {_sym} model...')
                    acc = train_model(_sym)
                    if acc > 0:
                        load_or_train_model(_sym)   # warm the in-memory cache
                        logger.info(f'Setup F: {_sym} model ready, accuracy={acc:.1%}')
                    else:
                        logger.warning(f'Setup F: {_sym} model training skipped (insufficient data)')
                except Exception as _te:
                    logger.warning(f'Setup F: {_sym} model training error: {_te}')
        except Exception as e:
            logger.warning(f'Setup F startup training failed: {e}')

        # Train Setup I direction models after data is ready
        _t.sleep(2)
        try:
            from setup_i_mathematical import train_model_i, load_or_train_model_i
            import os as _os_i
            for _sym in ['MNQ', 'ES']:
                try:
                    short_pkl = f'apex_xi_{_sym}_short.pkl'
                    long_pkl  = f'apex_xi_{_sym}_long.pkl'
                    if not (_os_i.path.exists(short_pkl) and _os_i.path.exists(long_pkl)):
                        logger.info(f'Setup I: Training {_sym} direction models...')
                        result = train_model_i(_sym)
                        logger.info(
                            f'Setup I: {_sym} short AUC={result["short_auc"]:.3f} '
                            f'deploying={result["short_ok"]} | '
                            f'long AUC={result["long_auc"]:.3f} deploying={result["long_ok"]}'
                        )
                    else:
                        # Load cached models and log AUC
                        xgb_s, xgb_l, _, _ = load_or_train_model_i(_sym)
                        import pickle as _pkl_i
                        s_auc = l_auc = None
                        if _os_i.path.exists(short_pkl):
                            with open(short_pkl, 'rb') as _f:
                                s_auc = _pkl_i.load(_f).get('oos_auc')
                        if _os_i.path.exists(long_pkl):
                            with open(long_pkl, 'rb') as _f:
                                l_auc = _pkl_i.load(_f).get('oos_auc')
                        logger.info(
                            f'Setup I: {_sym} short AUC={s_auc or "?"} '
                            f'deploying={xgb_s is not None} | '
                            f'long AUC={l_auc or "?"} deploying={xgb_l is not None}'
                        )
                    logger.info(
                        f'Setup I: {_sym} active directions — '
                        f'SHORT={_sym not in __import__("setup_i_mathematical")._i_disabled_short} '
                        f'LONG={_sym not in __import__("setup_i_mathematical")._i_disabled_long}'
                    )
                except Exception as _i_te:
                    logger.warning(f'Setup I: {_sym} startup error: {_i_te}')
        except Exception as e:
            logger.warning(f'Setup I startup training failed: {e}')

        # ── APEX READY verification log ──────────────────────────
        try:
            from setup_f_ml import _load_ohlcv, calculate_features, FEATURE_NAMES
            import numpy as _np
            _bias_labels = {1.0: 'BULLISH', -1.0: 'BEARISH', 0.0: 'NEUTRAL'}
            _bias_parts = []
            for _sym in ['MNQ', 'ES', 'GC']:
                try:
                    _df = _load_ohlcv(_sym, '5min', limit=2000)
                    if not _df.empty:
                        _X = calculate_features(_sym, _df)
                        _bias = _bias_labels.get(float(_X[-1, 1]), 'NEUTRAL')
                    else:
                        _bias = 'NO DATA'
                    _bias_parts.append(f'{_sym} bias={_bias}')
                except Exception:
                    _bias_parts.append(f'{_sym} bias=ERROR')
            _f_parts = []
            for _sym in ['MNQ', 'ES']:
                import os as _os
                _pkl = f'apex_rf_{_sym}.pkl'
                _f_parts.append(f'SetupF {_sym}={"loaded" if _os.path.exists(_pkl) else "MISSING"}')
            logger.info(
                'APEX READY — ' + ' | '.join(_bias_parts) + ' | ' + ' | '.join(_f_parts)
            )
        except Exception as _re:
            logger.warning(f'APEX READY check failed: {_re}')

        # Pre-load economic calendar on startup
        try:
            from calendar_filter import get_filter as _gcf_startup
            _cf_startup = _gcf_startup()
            _cal_upcoming = _cf_startup.get_upcoming_events(hours=24)
            logger.info(
                f'Economic calendar loaded — {len(_cal_upcoming)} events in next 24h'
                + (f' | next: {_cal_upcoming[0]["name"]} {_cal_upcoming[0]["utc_display"]}' if _cal_upcoming else '')
            )
        except Exception as _cse:
            logger.warning(f'Calendar startup load failed: {_cse}')

    threading.Thread(target=startup_backfill, daemon=True).start()
    threading.Thread(target=background_scheduler, daemon=True).start()

    def _mdir_startup_load():
        """Load (or trigger first-time train) of Meridian Direction models at startup."""
        try:
            import meridian_direction as _mdir_su
            for _su_sym in ('MNQ', 'ES'):
                if not _mdir_su.load_models(_su_sym):
                    logger.info(f'Meridian Direction: no model for {_su_sym} — triggering first-time train')
                    _mdir_su.train(_su_sym)
        except Exception as _mdir_su_e:
            logger.warning(f'Meridian Direction startup load: {_mdir_su_e}')

    if _db.IS_POSTGRES:
        threading.Thread(target=_mdir_startup_load, daemon=True).start()
    logger.info(f'  Daily loss limit:  ${DAILY_LOSS_LIMIT}')
    logger.info(f'  Portfolio heat limit: max {MAX_PORTFOLIO_HEAT} concurrent trades')
    logger.info(f'  Paper-only setups (no Tradovate execution): '
                f'{", ".join(sorted(PAPER_ONLY_SETUPS)) or "none"}')
    logger.info('  Setup H: MNQ=live | ES=live')
    logger.info('  Setup I: MNQ=live | ES=live')
    logger.info('  Setup J: MNQ=live | ES=live MESM6')
    logger.info('  Max stop limits: MNQ=50pts ES=12pts GC=8pts')
    logger.info('  Setup E: disabled (underperforming — confirmed losing record)')
    logger.info('  Setup J: wired to scanner — scanning every 5min')
    logger.info('  Server running at: http://localhost:5000')
    logger.info('  Open apex_dashboard_v8.html in your browser')
    logger.info('=' * 55)

_startup()


# ─────────────────────────────────────────────────────────────
#  APEX DASHBOARD API ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.route('/api/apex/scan', methods=['GET'])
def apex_scan():
    """Run full gate check and return results."""
    from setup_engine import check_setup, check_setup_a, check_setup_c
    from setup_e import check_setup_e
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    NY = ZoneInfo('America/New_York')
    now = datetime.now(timezone.utc)
    results = []
    for sym in ('MNQ', 'ES', 'GC'):
        for direction in ('long', 'short'):
            for check_fn, setup_name in [
                (lambda s,d: check_setup(s,d,'swing',now), 'B'),
                (lambda s,d: check_setup_a(s,d,'swing',now), 'A'),
                (lambda s,d: check_setup_c(s,d,'swing',now), 'C'),
            ]:
                try:
                    r = check_fn(sym, direction)
                    gates = [{'gate': g.gate, 'name': g.name, 'passed': g.passed, 'detail': g.detail} for g in r.gates]
                    _gp = sum(1 for g in gates if g['passed'])
                    _gt = len(gates)
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
                        'readiness_score': 100 if r.valid else (round(_gp / _gt * 100) if _gt else 0),
                        'next_gate_description': (
                            'All gates passed — signal ready' if r.valid
                            else next((f"{g['name']}: {g['detail']}" for g in gates if not g['passed']), '—')
                        ),
                    })
                except Exception as e:
                    logger.debug(f'Gate check error {sym} {direction} Setup {setup_name}: {e}')

    # Setup E — EMA50 Pullback, MNQ only
    for direction in ('long', 'short'):
        try:
            r = check_setup_e('MNQ', direction, now)
            gates = [{'gate': g.gate, 'name': g.name, 'passed': g.passed, 'detail': g.detail} for g in r.gates]
            _gp = sum(1 for g in gates if g['passed'])
            _gt = len(gates)
            results.append({
                'symbol':    'MNQ',
                'direction': direction,
                'setup':     'E',
                'valid':     r.valid,
                'gates':     gates,
                'entry':     r.entry,
                'stop':      r.stop,
                'target':    r.target,
                'rr':        r.rr,
                'quality':   getattr(r, 'quality', 'primary'),
                'failed_at': next((g['name'] for g in gates if not g['passed']), None),
                'readiness_score': 100 if r.valid else (round(_gp / _gt * 100) if _gt else 0),
                'next_gate_description': (
                    'All gates passed — signal ready' if r.valid
                    else next((f"{g['name']}: {g['detail']}" for g in gates if not g['passed']), '—')
                ),
            })
        except Exception as e:
            logger.debug(f'Setup E scan error ({direction}): {e}')

    # Setup D — FVG Fill (MNQ + ES, GC disabled)
    fvg_signals = []
    try:
        from fvg_engine import scan_setup_d, get_setup_d_state
        for _d_sym in ['MNQ', 'ES']:
            s = scan_setup_d(_d_sym, now)
            if s:
                fvg_signals.append({
                    'symbol':    s['symbol'],
                    'direction': s['direction'],
                    'setup':     'D_fvg_fill',
                    'valid':     True,
                    'gates':     [{'gate': i+1, 'name': f'Gate {i+1}', 'passed': True, 'detail': ''} for i in range(5)],
                    'entry':     s['entry'],
                    'stop':      s['stop'],
                    'target':    s['target'],
                    'rr':        s['rr'],
                    'quality':   'primary',
                    'fvg_score': s.get('fvg_score'),
                    'fvg_top':   s.get('fvg_top'),
                    'fvg_bottom':s.get('fvg_bottom'),
                    'bias':      s.get('bias'),
                    'failed_at': None,
                    'signal_state': 'SIGNAL READY',
                    'readiness_score': 100,
                    'next_gate_description': 'Signal ready',
                })
            else:
                # No signal — show gate-by-gate progress
                try:
                    _d_state = get_setup_d_state(_d_sym)
                    _d_gates = _d_state.get('gates', [])
                    _d_passed = sum(1 for g in _d_gates if g['passed'])
                    _d_total  = len(_d_gates)
                    fvg_signals.append({
                        'symbol':       _d_sym,
                        'direction':    _d_state.get('bias', 'neutral'),
                        'setup':        'D_fvg_fill',
                        'valid':        False,
                        'gates':        _d_gates,
                        'entry':        None,
                        'stop':         None,
                        'target':       None,
                        'rr':           None,
                        'quality':      'primary',
                        'bias':         _d_state.get('bias'),
                        'failed_at':    next((g['name'] for g in _d_gates if not g['passed']), None),
                        'signal_state': 'SCANNING' if _d_passed >= 2 else 'DEVELOPING',
                        'readiness_score': round(_d_passed / _d_total * 100) if _d_total else 0,
                        'next_gate_description': next(
                            (f"{g['name']}: {g['detail']}" for g in _d_gates if not g['passed']), '—'
                        ),
                    })
                except Exception as _dse:
                    logger.debug(f'Setup D state error {_d_sym}: {_dse}')
    except Exception as e:
        logger.debug(f'Setup D scan error in apex_scan: {e}')

    # Setup F — ML predictions for all 3 symbols (dashboard only, no trade execution)
    setup_f_predictions = []
    try:
        from setup_f_ml import get_current_prediction
        for _sym in ('MNQ', 'ES', 'GC'):
            pred = get_current_prediction(_sym)
            _f_prob   = pred.get('probability', 0) or 0
            _f_long_t = pred.get('long_threshold', 0.65)
            _f_short_t= pred.get('short_threshold', 0.35)
            if _f_prob > _f_long_t:
                _f_score = round(_f_prob * 100)
                _f_desc  = f"Signal LONG — {_f_prob*100:.1f}% (threshold {_f_long_t*100:.0f}%)"
            elif _f_prob < _f_short_t:
                _f_score = round((1 - _f_prob) * 100)
                _f_desc  = f"Signal SHORT — {(1-_f_prob)*100:.1f}% (threshold {(1-_f_short_t)*100:.0f}%)"
            else:
                _f_score = round(max(_f_prob, 1 - _f_prob) * 100)
                _f_desc  = f"P={_f_prob*100:.1f}% — long >{_f_long_t*100:.0f}%, short <{_f_short_t*100:.0f}%"
            pred['readiness_score']       = _f_score
            pred['next_gate_description'] = _f_desc
            try:
                from regime_engine import get_current_regime as _gcr_f
                _f_reg = _gcr_f(_sym)
                pred['regime']           = _f_reg.get('regime', 'UNKNOWN') if _f_reg else 'UNKNOWN'
                pred['regime_confidence']= round(float(_f_reg.get('confidence', 0) if _f_reg else 0), 3)
                pred['regime_optimal']   = (_strategy_enabled_cache.get('F', {}).get('optimal_regimes') or 'CHOPPY,TRENDING')
                pred['regime_gating']    = False  # F uses RF model as its own gate
            except Exception:
                pred['regime'] = 'UNKNOWN'; pred['regime_confidence'] = 0.0
                pred['regime_optimal'] = 'CHOPPY,TRENDING'; pred['regime_gating'] = False
            setup_f_predictions.append(pred)
    except Exception as e:
        logger.debug(f'Setup F prediction error: {e}')

    # Setup H — VWAP band state for ES and MNQ (dashboard display)
    setup_h_data = []
    try:
        from setup_h_vwap import get_h_state
        for _sym in ('ES', 'MNQ'):
            _h = get_h_state(_sym)
            # Build gate-by-gate structure from state data
            try:
                from regime_engine import get_current_regime as _gcr_h
                _h_reg_info = _gcr_h(_sym)
                _h_reg_name = _h_reg_info.get('regime', 'UNKNOWN') if _h_reg_info else 'UNKNOWN'
                _h_reg_conf = float(_h_reg_info.get('confidence', 0) if _h_reg_info else 0)
                _h_reg_optimal = [r.strip() for r in (_strategy_enabled_cache.get('H', {}).get('optimal_regimes') or 'CHOPPY,MEAN_REVERTING').split(',') if r.strip()]
                _h_reg_pass = bool(_h_reg_info and _h_reg_conf >= 0.50 and _h_reg_name in _h_reg_optimal)
                _h_reg_detail = (
                    f'regime={_h_reg_name}  conf={_h_reg_conf:.2f}  '
                    f'required={"/".join(_h_reg_optimal)}  threshold=0.50  '
                    + ('✓ PASS' if _h_reg_pass else '✗ FAIL')
                )
            except Exception:
                _h_reg_name = 'UNKNOWN'; _h_reg_conf = 0.0
                _h_reg_pass = True; _h_reg_detail = 'regime engine unavailable (fail open)'
            _h_gates = [
                {
                    'gate': 1, 'name': 'Session',
                    'passed': bool(_h.get('in_session')),
                    'detail': 'in session' if _h.get('in_session') else 'out of session',
                },
                {
                    'gate': 2, 'name': 'VWAP Bands Available',
                    'passed': _h.get('vwap') is not None,
                    'detail': f'VWAP={_h.get("vwap")} upper={_h.get("upper_band")} lower={_h.get("lower_band")}' if _h.get('vwap') else f'{_h.get("signal_state")}',
                },
                {
                    'gate': 3, 'name': 'Regime Gate',
                    'passed': _h_reg_pass,
                    'detail': _h_reg_detail,
                },
                {
                    'gate': 4, 'name': 'Outside 2σ Band',
                    'passed': _h.get('signal_state') in ('ABOVE_UPPER', 'BELOW_LOWER'),
                    'detail': (
                        f'ABOVE upper={_h.get("upper_band")} price={_h.get("price")} dist={_h.get("dist_upper_atr")}×ATR'
                        if _h.get('signal_state') == 'ABOVE_UPPER' else
                        f'BELOW lower={_h.get("lower_band")} price={_h.get("price")} dist={_h.get("dist_lower_atr")}×ATR'
                        if _h.get('signal_state') == 'BELOW_LOWER' else
                        f'price={_h.get("price")} upper={_h.get("upper_band")} lower={_h.get("lower_band")}'
                    ),
                },
                {
                    'gate': 5, 'name': 'HTF Bias Aligned',
                    'passed': (
                        (_h.get('signal_state') == 'ABOVE_UPPER' and _h.get('htf_bias') == 'bearish') or
                        (_h.get('signal_state') == 'BELOW_LOWER' and _h.get('htf_bias') == 'bullish')
                    ),
                    'detail': (
                        f'bias={_h.get("htf_bias")} '
                        f'need={"bearish" if _h.get("signal_state") == "ABOVE_UPPER" else "bullish" if _h.get("signal_state") == "BELOW_LOWER" else "n/a"}'
                    ),
                },
            ]
            _h_passed = sum(1 for g in _h_gates if g['passed'])
            _h_state_label = _h.get('signal_state', 'WATCHING')
            if not _h.get('in_session'):
                _h_state_label = 'OFF SESSION'
            elif _h_passed == len(_h_gates):
                _h_state_label = 'SIGNAL READY'
            elif _h_passed >= 2:
                _h_state_label = 'SCANNING'
            else:
                _h_state_label = 'DEVELOPING'
            _h['gates'] = _h_gates
            _h['signal_state_label'] = _h_state_label
            _h['failed_at'] = next((g['name'] for g in _h_gates if not g['passed']), None)
            _h_total = len(_h_gates)
            _h['readiness_score'] = 100 if _h_state_label == 'SIGNAL READY' else (
                round(_h_passed / _h_total * 100) if _h_total else 0
            )
            _h['next_gate_description'] = (
                'All gates passed — signal ready' if _h_state_label == 'SIGNAL READY'
                else next((f"{g['name']}: {g['detail']}" for g in _h_gates if not g['passed']), '—')
            )
            setup_h_data.append(_h)
    except Exception as e:
        logger.debug(f'Setup H state error: {e}')

    # Setup I — Mathematical Alpha state (dashboard display)
    setup_i_data = []
    try:
        from setup_i_mathematical import get_setup_i_state, _load_5min, _atr_i
        from datetime import datetime as _dt_cls
        _i_now = _dt_cls.now(__import__('datetime').timezone.utc)
        try:
            from calendar_filter import get_filter as _gcf_i
            _cal_i = _gcf_i()
        except Exception:
            _cal_i = None
        _heat_i_dash = _count_open_trades()
        for _sym in ('MNQ', 'ES'):
            _i = get_setup_i_state(_sym)
            _i_sess_end = 20 if _sym == 'MNQ' else 19
            _i_in_sess = (not _i_now.weekday() >= 5) and (13 <= _i_now.hour < _i_sess_end)
            _s_xgb = _i.get('short_xgb_prob')
            _l_xgb = _i.get('long_xgb_prob')
            _lr = _i.get('lr_prob')
            _long_xgb_ok  = _l_xgb is not None and _l_xgb > 0.58
            _long_lr_ok   = _lr is not None and _lr > 0.58
            _short_xgb_ok = _s_xgb is not None and _s_xgb > 0.58
            _short_lr_ok  = _lr is not None and _lr < 0.42
            _i_today_ok = _i_now.weekday() in {1, 2, 3}  # Tue=1, Wed=2, Thu=3
            # Regime gate — replicate is_setup_enabled('I', _sym) logic for dashboard
            try:
                from regime_engine import get_current_regime as _gcr_i
                _i_reg_info = _gcr_i(_sym)
                _i_reg_name = _i_reg_info.get('regime', 'UNKNOWN') if _i_reg_info else 'UNKNOWN'
                _i_reg_conf = float(_i_reg_info.get('confidence', 0) if _i_reg_info else 0)
                _i_reg_optimal = [r.strip() for r in (_strategy_enabled_cache.get('I', {}).get('optimal_regimes') or 'TRENDING').split(',') if r.strip()]
                _i_reg_pass = bool(_i_reg_info and _i_reg_conf >= 0.50 and _i_reg_name in _i_reg_optimal)
                _i_reg_detail = (
                    f'regime={_i_reg_name}  conf={_i_reg_conf:.2f}  '
                    f'required={"/".join(_i_reg_optimal)}  threshold=0.50  '
                    + ('✓ PASS' if _i_reg_pass else '✗ FAIL')
                )
            except Exception:
                _i_reg_name = 'UNKNOWN'; _i_reg_conf = 0.0
                _i_reg_pass = True; _i_reg_detail = 'regime engine unavailable (fail open)'
            # Gate 6: stop cap (1.5 × ATR14 must not exceed _MAX_STOP_PTS)
            try:
                _df5_i   = _load_5min(_sym, limit=30)
                _atr14_i = float(_atr_i(_df5_i, 14).iloc[-1] or 0)
                _i_stop  = round(1.5 * _atr14_i, 1)
                _i_cap   = _MAX_STOP_PTS.get(_sym, 9999)
                _stop_ok = 0 < _i_stop <= _i_cap
                _stop_det = f'ATR14={_atr14_i:.1f} stop={_i_stop}pts cap={_i_cap}pts'
            except Exception:
                _stop_ok = True; _stop_det = 'ATR unavailable'
            # Gate 7: calendar blackout
            try:
                _i_blocked, _i_cal_rsn = _cal_i.is_blocked(_sym, _i_now) if _cal_i else (False, '')
                _cal_i_ok  = not _i_blocked
                _cal_i_det = _i_cal_rsn if _i_blocked else 'No blackout'
            except Exception:
                _cal_i_ok = True; _cal_i_det = 'Calendar unavailable'
            # Gate 8: portfolio heat
            _heat_i_ok  = _heat_i_dash < MAX_PORTFOLIO_HEAT
            _heat_i_det = f'{_heat_i_dash}/{MAX_PORTFOLIO_HEAT} open trades'
            _i_gates = [
                {
                    'gate': 1, 'name': 'Models Trained',
                    'passed': bool(_i.get('ok')),
                    'detail': f'short_model={_i.get("short_enabled")} long_model={_i.get("long_enabled")}',
                },
                {
                    'gate': 2, 'name': 'Trading Day',
                    'passed': _i_today_ok,
                    'detail': (
                        f'weekday={_i_now.weekday()} '
                        f'({["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][_i_now.weekday()]}) '
                        f'— need Tue/Wed/Thu'
                    ),
                },
                {
                    'gate': 3, 'name': 'Regime Gate',
                    'passed': _i_reg_pass,
                    'detail': _i_reg_detail,
                },
                {
                    'gate': 4, 'name': 'Session',
                    'passed': _i_in_sess,
                    'detail': f'{_i_now.hour:02d}:00 UTC session 13-{_i_sess_end} UTC',
                },
                {
                    'gate': 5, 'name': 'XGB Probability > 0.58',
                    'passed': _long_xgb_ok or _short_xgb_ok,
                    'detail': f'long_xgb={_l_xgb} short_xgb={_s_xgb} (need >0.58)',
                },
                {
                    'gate': 6, 'name': 'LogReg Confirmation',
                    'passed': _long_lr_ok or _short_lr_ok,
                    'detail': (
                        f'lr={_lr:.3f} (long needs >0.58, short needs <0.42)'
                        if _lr is not None else 'lr=None'
                    ),
                },
                {
                    'gate': 7, 'name': 'Stop Cap',
                    'passed': _stop_ok,
                    'detail': _stop_det,
                },
                {
                    'gate': 8, 'name': 'Calendar',
                    'passed': _cal_i_ok,
                    'detail': _cal_i_det,
                },
                {
                    'gate': 9, 'name': 'Portfolio Heat',
                    'passed': _heat_i_ok,
                    'detail': _heat_i_det,
                },
            ]
            _i_passed = sum(1 for g in _i_gates if g['passed'])
            if not _i_in_sess:
                _i_label = 'OFF SESSION'
            elif not _i.get('ok'):
                _i_label = 'DEVELOPING'
            elif _i_passed == len(_i_gates):
                _i_label = 'SIGNAL READY'
            elif _i_passed >= 2:
                _i_label = 'SCANNING'
            else:
                _i_label = 'DEVELOPING'
            _i['gates'] = _i_gates
            _i['signal_state_label'] = _i_label
            _i['failed_at'] = next((g['name'] for g in _i_gates if not g['passed']), None)
            _i_max_xgb = max(filter(None, [_i.get('short_xgb_prob'), _i.get('long_xgb_prob')]), default=0)
            _i['readiness_score'] = round((_i_max_xgb or 0) * 100)
            _i['next_gate_description'] = (
                'All gates passed — signal ready' if _i_label == 'SIGNAL READY'
                else next((f"{g['name']}: {g['detail']}" for g in _i_gates if not g['passed']), '—')
            )
            setup_i_data.append(_i)
    except Exception as e:
        logger.debug(f'Setup I state error: {e}')

    # Setup J — Value Area Continuation gate states (dashboard display)
    setup_j_data = []
    try:
        from setup_j_value_area import get_setup_j_state
        for _sym_j_scan in ('ES', 'MNQ'):
            _j_state = get_setup_j_state(_sym_j_scan)
            # Inject regime gate at position 1 (before other gates)
            try:
                from regime_engine import get_current_regime as _gcr_j
                _j_reg_info = _gcr_j(_sym_j_scan)
                _j_reg_name = _j_reg_info.get('regime', 'UNKNOWN') if _j_reg_info else 'UNKNOWN'
                _j_reg_conf = float(_j_reg_info.get('confidence', 0) if _j_reg_info else 0)
                _j_reg_optimal = [r.strip() for r in (_strategy_enabled_cache.get('J', {}).get('optimal_regimes') or 'CHOPPY,MEAN_REVERTING,TRENDING').split(',') if r.strip()]
                _j_reg_pass = bool(_j_reg_info and _j_reg_conf >= 0.50 and _j_reg_name in _j_reg_optimal)
                _j_reg_detail = (
                    f'regime={_j_reg_name}  conf={_j_reg_conf:.2f}  '
                    f'required={"/".join(_j_reg_optimal)}  threshold=0.50  '
                    + ('✓ PASS' if _j_reg_pass else '✗ FAIL')
                )
            except Exception:
                _j_reg_name = 'UNKNOWN'; _j_reg_conf = 0.0
                _j_reg_pass = True; _j_reg_detail = 'regime engine unavailable (fail open)'
            _j_regime_gate = {'gate': 0, 'name': 'Regime Gate', 'passed': _j_reg_pass, 'detail': _j_reg_detail}
            existing_gates = _j_state.get('gates', [])
            # Renumber existing gates to start at 1 if they start at 0, else shift up by 1
            _j_min_gate = min((g.get('gate', 1) for g in existing_gates), default=1)
            if _j_min_gate == 0:
                # Already 0-based — insert regime gate before index 0 with gate=-1 then renumber
                for _jg in existing_gates:
                    _jg['gate'] = _jg.get('gate', 0) + 1
                _j_regime_gate['gate'] = 0
            else:
                # 1-based — shift all up by 1 and insert at gate=1
                for _jg in existing_gates:
                    _jg['gate'] = _jg.get('gate', 1) + 1
                _j_regime_gate['gate'] = 1
            _j_state['gates'] = [_j_regime_gate] + existing_gates
            setup_j_data.append(_j_state)
    except Exception as _j_scan_e:
        logger.debug(f'Setup J state error: {_j_scan_e}')

    return jsonify({
        'ok':                  True,
        'results':             results,
        'fvg_signals':         fvg_signals,
        'setup_f_predictions': setup_f_predictions,
        'setup_h_data':        setup_h_data,
        'setup_i_data':        setup_i_data,
        'setup_j_data':        setup_j_data,
        'time':                now.astimezone(NY).strftime('%Y-%m-%d %H:%M ET')
    })


@app.route('/api/apex/trades', methods=['GET'])
def apex_trades():
    """Return open trades, all historical trades, and stats."""
    try:
        from trade_tracker import get_open_trades, get_stats, init_trades_table
        import db as _db_t
        init_trades_table()
        cols = ['id','symbol','direction','setup','mode','entry_price','stop',
                'target','rr_planned','session','quality','entry_time','exit_price',
                'exit_time','exit_reason','pnl_r','status','bars_held','notes','broker_order_id']
        conn = _db_t.connect()
        rows = conn.execute(
            'SELECT * FROM apex_trades ORDER BY entry_time DESC LIMIT 500'
        ).fetchall()
        conn.close()
        all_trades = [dict(zip(cols, r)) for r in rows]
        return jsonify({
            'ok':         True,
            'open_trades': get_open_trades(),
            'all_trades':  all_trades,
            'stats':      get_stats(),
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/apex/trades/<int:trade_id>/close', methods=['POST'])
def apex_trade_close(trade_id):
    """
    POST /api/apex/trades/{trade_id}/close

    Manually close an open trade via dashboard.
    Places a market order on Tradovate to exit the position,
    updates apex_trades with exit details, sends Telegram alert.

    Safe to call even if Tradovate fails — DB is always updated.
    Only works on open trades. Returns 400 if already closed.

    Returns:
      {ok, trade_id, exit_price, pnl_r, telegram_sent, tradovate_order_id}
    """
    try:
        from trade_tracker import get_current_price, close_trade, init_trades_table
        import db as _mc_db
        init_trades_table()

        # 1. Verify trade is open
        cols = ['id','symbol','direction','setup','mode','entry_price','stop',
                'target','rr_planned','session','quality','entry_time','exit_price',
                'exit_time','exit_reason','pnl_r','status','bars_held','notes','broker_order_id']
        conn = _mc_db.connect()
        row  = conn.execute(
            "SELECT * FROM apex_trades WHERE id=? AND status='open'", (trade_id,)
        ).fetchone()
        conn.close()

        if not row:
            return jsonify({'ok': False, 'error': 'Trade not found or already closed'}), 400

        trade       = dict(zip(cols, row))
        symbol      = trade['symbol']
        direction   = trade['direction']
        entry_price = float(trade['entry_price'])

        # 2. Get current price from live feed
        current_price = get_current_price(symbol)
        if not current_price:
            current_price = entry_price  # safe fallback — pnl_r will be ~0

        # 3. Tradovate closing order — only if a Tradovate order was placed for this trade.
        # Trades without broker_order_id were never sent to Tradovate (paper/test trades)
        # so placing a close order would open a new short — never do this.
        tradovate_order_id = None
        tradovate_error    = None
        try:
            from tradovate import place_market_close, TRADOVATE_ENABLED as _TV
            if _TV and trade.get('broker_order_id'):
                tv = place_market_close(symbol, direction, contracts=1)
                if tv.get('ok'):
                    tradovate_order_id = tv.get('order_id')
                    if tv.get('fill_price'):
                        current_price = float(tv['fill_price'])
                else:
                    tradovate_error = tv.get('error', 'unknown')
                    logger.warning(
                        f'manual_close #{trade_id}: Tradovate failed — {tradovate_error}. '
                        f'DB update will still proceed.'
                    )
        except Exception as _tv_exc:
            tradovate_error = str(_tv_exc)
            logger.warning(f'manual_close #{trade_id}: Tradovate exception — {_tv_exc}')

        # 4. Close in DB — always happens regardless of Tradovate result
        closed = close_trade(trade_id, current_price, 'manual_close')
        if not closed:
            return jsonify({'ok': False, 'error': 'DB close failed'}), 500

        pnl_r = closed.get('pnl_r')
        if pnl_r is None:
            pnl_r = 0.0  # capped by ±10R sanity check or sub-1pt stop

        if tradovate_order_id:
            try:
                conn = _mc_db.connect()
                conn.execute('UPDATE apex_trades SET broker_order_id=? WHERE id=?',
                             (tradovate_order_id, trade_id))
                conn.commit(); conn.close()
            except Exception:
                pass

        # 5. Telegram alert
        telegram_sent = False
        try:
            from live_scanner import send_telegram as _mc_tg
            arrow = '🟢' if (pnl_r or 0) >= 0 else '🔴'
            _mc_tg(
                f'{arrow} <b>WISE MERIDIAN CAPITAL — Manual Close</b>\n'
                f'{symbol} {direction.upper()} · {trade["setup"]}\n'
                f'Entry: {entry_price:.2f} → Exit: {current_price:.2f}\n'
                f'P&amp;L: {pnl_r:+.2f}R\n'
                f'Reason: Manual close via dashboard'
            )
            telegram_sent = True
        except Exception as _tg_exc:
            logger.warning(f'manual_close #{trade_id}: Telegram failed — {_tg_exc}')

        logger.info(
            f'Manual close: #{trade_id} {symbol} {direction} {trade["setup"]} '
            f'exit={current_price:.2f} pnl={pnl_r:+.2f}R tv_order={tradovate_order_id}'
        )

        resp = {'ok': True, 'trade_id': trade_id, 'exit_price': current_price,
                'pnl_r': pnl_r, 'telegram_sent': telegram_sent,
                'tradovate_order_id': tradovate_order_id}
        if tradovate_error:
            resp['tradovate_error'] = tradovate_error
            resp['note'] = 'DB updated. Tradovate may still have open position.'
        return jsonify(resp)

    except Exception as e:
        logger.error(f'apex_trade_close error: {e}', exc_info=True)
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/apex/risk', methods=['GET'])
def apex_risk():
    """Return current risk management status — regime, daily P&L, drawdown, multiplier."""
    try:
        from risk_manager import RegimeDetector, DailyRiskMonitor, DrawdownTracker
        rd     = RegimeDetector()
        drm    = DailyRiskMonitor()
        ddt    = DrawdownTracker()
        regime = rd.get_regime()
        daily  = drm.get_status()
        dd     = ddt.get_status()
        open_count = 0
        try:
            from trade_tracker import get_open_trades
            open_count = len(get_open_trades())
        except Exception:
            pass
        return jsonify({
            'ok':               True,
            'current_regime':   regime.label,
            'regime_adx':       round(regime.adx, 1),
            'regime_atr_ratio': round(regime.atr_ratio, 2),
            'daily_pnl_r':      daily['daily_pnl_r'],
            'daily_limit_hit':  daily['limit_hit'],
            'win_protection':   daily['win_protection'],
            'drawdown_pct':     dd['drawdown_pct'],
            'risk_multiplier':  dd['risk_multiplier'],
            'open_trades_count': open_count,
            'max_open_trades':  3,
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/apex/wyckoff', methods=['GET'])
def apex_wyckoff():
    """Return Wyckoff upthrust log and stats for Setup G tracker."""
    try:
        from wyckoff_tracker import get_wyckoff_stats
        stats = get_wyckoff_stats()
        return jsonify({'ok': True, **stats})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/apex/calendar', methods=['GET'])
def apex_calendar():
    """Return economic calendar status and upcoming events."""
    try:
        from calendar_filter import get_filter as _gcf
        _cf = _gcf()
        status = _cf.get_current_status()
        hours = int(request.args.get('hours', 24))
        if hours != 24:
            status['upcoming'] = _cf.get_upcoming_events(hours=hours)
        return jsonify({'ok': True, **status})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e), 'status': 'CLEAR', 'reason': '', 'upcoming': []})


@app.route('/api/apex/tradovate', methods=['GET'])
def apex_tradovate():
    """Return Tradovate account status, positions, and today's orders."""
    try:
        from tradovate import get_status
        return jsonify(get_status())
    except Exception as e:
        logger.debug(f'Tradovate status error: {e}')
        return jsonify({
            'enabled':   False,
            'connected': False,
            'error':     str(e),
            'positions': [],
            'orders_today': [],
        })


@app.route('/api/apex/tradovate/order', methods=['POST'])
def apex_tradovate_order():
    """Place a manual market order via Tradovate (Execution tab)."""
    try:
        from tradovate import TRADOVATE_ENABLED, authenticate, _token_cache
        from tradovate import place_bracket_order, get_account, calculate_position_size
        if not TRADOVATE_ENABLED:
            return jsonify({'ok': False, 'error': 'Tradovate disabled'})
        data = request.get_json(force=True)
        sym       = data.get('symbol', '').upper()
        direction = data.get('direction', '').lower()
        contracts = int(data.get('contracts', 1))
        entry     = float(data.get('entry',  0))
        stop      = float(data.get('stop',   0))
        target    = float(data.get('target', 0))
        if not sym or direction not in ('long', 'short') or contracts < 1:
            return jsonify({'ok': False, 'error': 'Invalid params'})
        result = place_bracket_order(sym, direction, contracts, entry, stop, target)
        return jsonify(result)
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/apex/tradovate/test', methods=['GET'])
def apex_tradovate_test():
    """
    End-to-end Tradovate verification.
    1. Authenticates and logs token age.
    2. Places 1-contract MNQ market order on demo.
    3. Returns orderId, fill_price, full diagnostics.
    ONLY callable when TRADOVATE_ENABLED=true AND APEX_TESTING=false.
    """
    if os.environ.get('APEX_TESTING', 'false').lower() == 'true':
        return jsonify({'ok': False, 'reason': 'APEX_TESTING=true — order placement blocked'})
    diag = {}
    try:
        from tradovate import (
            TRADOVATE_ENABLED, TRADOVATE_DEMO, TRADOVATE_ACCOUNT,
            authenticate, _token_cache, place_bracket_order,
            get_account, _tradovate_symbol,
        )
        import time as _time
        diag['TRADOVATE_ENABLED'] = TRADOVATE_ENABLED
        diag['TRADOVATE_DEMO']    = TRADOVATE_DEMO
        diag['account_spec']      = TRADOVATE_ACCOUNT or 'not_set'

        if not TRADOVATE_ENABLED:
            return jsonify({'ok': False, 'reason': 'TRADOVATE_ENABLED=false — set it to true first', 'diag': diag})

        # Step 1 — auth
        auth = authenticate()
        diag['auth_ok']    = auth.get('ok', False)
        diag['account_id'] = auth.get('account_id') or _token_cache.get('account_id')

        _now = _time.time()
        _exp = _token_cache.get('expiry', 0)
        if _exp:
            diag['token_age_s'] = int(_now - (_exp - 80 * 60))
            diag['token_ttl_s'] = int(_exp - _now)

        if not auth.get('ok'):
            diag['auth_error'] = auth.get('error')
            return jsonify({'ok': False, 'reason': 'Auth failed', 'diag': diag})

        # Step 2 — account balance check
        acct = get_account()
        diag['balance'] = acct.get('balance')
        diag['acct_ok'] = acct.get('ok')

        # Step 3 — probe symbol lookup before placing order
        from tradovate import BASE_URL, _auth_header
        import requests as _req
        instrument = _tradovate_symbol('MNQ')
        diag['instrument']  = instrument
        diag['base_url']    = BASE_URL
        diag['order_url']   = f'{BASE_URL}/order/placeOrder'

        # Verify the symbol exists on Tradovate
        try:
            _h = _auth_header()
            _sym_resp = _req.get(
                f'{BASE_URL}/contract/find',
                params={'name': instrument},
                headers=_h, timeout=10
            )
            diag['symbol_lookup_status'] = _sym_resp.status_code
            diag['symbol_lookup_body']   = _sym_resp.json() if _sym_resp.ok else _sym_resp.text[:200]
        except Exception as _se:
            diag['symbol_lookup_error'] = str(_se)

        # Step 4 — place 1-contract MNQ market order on demo
        result = place_bracket_order(
            symbol='MNQ', direction='long', contracts=1,
            entry=0, stop=0, target=0,
        )
        diag['order_result'] = result
        diag['order_ok']     = result.get('ok', False)
        diag['order_id']     = result.get('order_id')
        diag['fill_price']   = result.get('fill_price')
        diag['http_error']   = result.get('error') if not result.get('ok') else None

        return jsonify({'ok': result.get('ok', False), 'diag': diag})

    except Exception as e:
        diag['exception'] = str(e)
        return jsonify({'ok': False, 'error': str(e), 'diag': diag})


@app.route('/api/apex/market', methods=['GET'])
def apex_market():
    """Return current market structure per instrument."""
    # Uses get_htf_bias (4h EMA20 resample from 5min) — same source as the scan gates
    # so this endpoint and /api/apex/scan always agree on HTF direction.
    from fvg_engine import get_htf_bias
    from market_structure import load_bars, find_swings, detect_structure, compute_bias
    from zoneinfo import ZoneInfo
    NY = ZoneInfo('America/New_York')
    results = {}
    for sym in ('MNQ', 'ES', 'GC'):
        try:
            # 4h EMA20 bias — identical calculation to the scan's Gate 1 HTF check
            bias = get_htf_bias(sym)

            # Load 5min bars → resample to 1h for price / structure display only
            df_5m = load_bars(sym, '5min', limit=3000)
            df_1h = df_5m[['open','high','low','close','volume']].resample('1h').agg({
                'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum',
            }).dropna()
            if len(df_1h) < 3:
                raise ValueError(f'Insufficient 1h bars after resample ({len(df_1h)})')
            sh, sl = find_swings(df_1h, lookback=5)
            events, _ = detect_structure(df_1h, sh, sl)
            last = df_1h.iloc[-1]
            prev = df_1h.iloc[-2]
            chg  = round(float(last['close']) - float(prev['close']), 2)
            pct  = round(chg / float(prev['close']) * 100, 2)
            last_bar_time = df_1h.index[-1].astimezone(NY).strftime('%H:%M ET')
            results[sym] = {
                'bias':       bias,        # 4h EMA20 — matches scan gates
                'close':      round(float(last['close']), 2),
                'change':     chg,
                'pct':        pct,
                'last_bar':   last_bar_time,
                'last_event': str(events[-1]) if events else None,
            }
        except Exception as e:
            results[sym] = {'bias': 'unknown', 'error': str(e)}
    return jsonify({'ok': True, 'market': results})


@app.route('/api/apex/candles/<symbol>', methods=['GET'])
@app.route('/api/apex/candles/<symbol>/<timeframe>', methods=['GET'])
def apex_candles(symbol, timeframe='5m'):
    """Return OHLCV bars for lightweight-charts. Timeframe: 1m 5m 15m 1h 4h."""
    symbol = symbol.upper()
    if symbol not in INSTRUMENTS:
        return jsonify({'ok': False, 'error': 'Unknown symbol'})
    TF_MAP = {
        '1m':  ('1min',   200),
        '5m':  ('5min',   100),
        '15m': ('15min',  100),
        '1h':  ('1hour',  100),
        '4h':  ('4hour',  100),
    }
    db_tf, default_limit = TF_MAP.get(timeframe, ('5min', 100))
    req_limit = request.args.get('limit', None, type=int)
    limit = req_limit if req_limit and 1 <= req_limit <= 2000 else default_limit
    try:
        conn = _db.connect()
        rows = conn.execute(
            'SELECT ts, open, high, low, close, volume FROM ohlcv '
            'WHERE symbol=? AND timeframe=? ORDER BY ts DESC LIMIT ?',
            (symbol, db_tf, limit)
        ).fetchall()
        conn.close()
        bars = [
            {'time': int(r[0]), 'open': float(r[1]), 'high': float(r[2]),
             'low': float(r[3]), 'close': float(r[4]), 'volume': float(r[5])}
            for r in reversed(rows)
        ]
        return jsonify({'ok': True, 'symbol': symbol, 'timeframe': timeframe, 'bars': bars})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/apex/equity', methods=['GET'])
def apex_equity():
    """Return compounding equity curve: $10k start, 1% risk per trade."""
    try:
        from trade_tracker import init_trades_table
        init_trades_table()
        conn = _db.connect()
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

        # Today's P&L — needed by dashboard sub-row
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        conn2 = _db.connect()
        today_rows = conn2.execute(
            'SELECT pnl_r FROM apex_trades WHERE status=? AND entry_time LIKE ?',
            ('closed', today + '%')
        ).fetchall()
        conn2.close()
        today_r      = round(sum(r[0] for r in today_rows if r[0] is not None), 2)
        today_pnl    = round(today_r * START * RISK, 2)
        total_return = round((balance - START) / START * 100, 2)

        # Add unrealised P&L from open trades as a live endpoint on the curve
        try:
            from trade_tracker import get_open_trades
            open_trades = get_open_trades()
            unrealised_r = sum(t.get('current_pnl_r', 0) or 0 for t in open_trades)
            live_balance = round(balance + unrealised_r * balance * RISK, 2)
        except Exception:
            live_balance = round(balance, 2)
            unrealised_r = 0.0

        from tradovate import get_risk_tier as _eq_get_tier
        _eq_tier = _eq_get_tier(live_balance)
        return jsonify({
            'ok':                    True,
            'points':                points,
            'current_balance':       round(balance, 2),
            'live_balance':          live_balance,
            'unrealised_r':          round(unrealised_r, 3),
            'max_drawdown':          max_dd,
            'today_r':               today_r,
            'today_pnl':             today_pnl,
            'total_return':          total_return,
            'kill_switch_active':    _kill_switch_active,
            'kill_switch_balance':   _kill_switch_balance,
            'kill_switch_threshold': _kill_switch_threshold,
            'daily_limit':           _eq_tier['daily_limit'],
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/apex/trades/seed', methods=['POST'])
def apex_trades_seed():
    """Insert sample closed trades so the equity curve and history render."""
    from trade_tracker import init_trades_table
    init_trades_table()
    SEED = [
        # symbol, dir, setup, mode, entry, stop, target, rr, session, quality,
        # entry_time, exit_time, exit_price, pnl_r, exit_reason
        ('NQ','long', 'A_sweep_ob',     'swing', 20320.0, 20245.0, 20620.0, 4.0,
         'NY Primary','primary',  '2026-03-10T14:15:00+00:00','2026-03-10T15:45:00+00:00', 20620.0,  4.0, 'target'),
        ('ES','short','B_choch_breaker','swing',  5630.0,  5660.0,  5510.0, 4.0,
         'NY Primary','primary',  '2026-03-11T14:00:00+00:00','2026-03-11T16:30:00+00:00',  5510.0,  4.0, 'target'),
        ('NQ','short','C_bos_ob',       'swing', 20580.0, 20655.0, 20280.0, 4.0,
         'London',    'secondary','2026-03-12T08:30:00+00:00','2026-03-12T09:15:00+00:00', 20655.0, -1.0, 'stop'),
        ('GC','long', 'A_sweep_ob',     'swing',  3020.0,  3005.0,  3080.0, 4.0,
         'GC Primary','primary',  '2026-03-13T13:00:00+00:00','2026-03-13T14:30:00+00:00',  3080.0,  4.0, 'target'),
        ('NQ','long', 'FVG_bull',       'scalp', 20700.0, 20650.0, 20800.0, 2.0,
         'NY Primary','primary',  '2026-03-14T14:05:00+00:00','2026-03-14T14:35:00+00:00', 20800.0,  2.0, 'target'),
        ('ES','long', 'A_sweep_ob',     'swing',  5710.0,  5670.0,  5870.0, 4.0,
         'NY Primary','primary',  '2026-03-17T14:30:00+00:00','2026-03-17T15:45:00+00:00',  5790.0,  2.0, 'partial'),
        ('NQ','long', 'B_choch_breaker','swing', 20850.0, 20775.0, 21150.0, 4.0,
         'NY Primary','primary',  '2026-03-18T13:45:00+00:00','2026-03-18T15:30:00+00:00', 20775.0, -1.0, 'stop'),
        ('GC','short','C_bos_ob',       'swing',  3090.0,  3110.0,  3010.0, 4.0,
         'GC Primary','primary',  '2026-03-19T13:00:00+00:00','2026-03-19T14:45:00+00:00',  3010.0,  4.0, 'target'),
    ]
    try:
        conn = _db.connect()
        inserted = 0
        for row in SEED:
            (sym, dr, setup, mode, entry, stop, tgt, rr, sess, qual,
             ent_t, ex_t, ex_p, pnl, reason) = row
            risk = abs(entry - stop)
            # Verify pnl_r matches prices
            if risk > 0:
                calc_pnl = round((ex_p - entry) / risk if dr == 'long' else (entry - ex_p) / risk, 3)
            else:
                calc_pnl = pnl
            cur = conn.execute(
                'INSERT OR IGNORE INTO apex_trades '
                '(symbol,direction,setup,mode,entry_price,stop,target,rr_planned,'
                ' session,quality,entry_time,exit_price,exit_time,exit_reason,pnl_r,status,bars_held,notes) '
                'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (sym, dr, setup, mode, entry, stop, tgt, rr,
                 sess, qual, ent_t, ex_p, ex_t, reason, calc_pnl, 'closed', 0, 'seeded')
            )
            inserted += cur.rowcount
        conn.commit()
        conn.close()
        logger.info(f'Seed trades inserted: {inserted}')
        return jsonify({'ok': True, 'inserted': inserted,
                        'message': f'Seeded {inserted} sample trades (skipped duplicates)'})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/apex/trades/reset', methods=['DELETE'])
def apex_trades_reset():
    """Remove all seeded test trades (notes='seeded')."""
    try:
        from trade_tracker import init_trades_table
        init_trades_table()
        conn = _db.connect()
        cur = conn.execute("DELETE FROM apex_trades WHERE notes='seeded'")
        deleted = cur.rowcount
        conn.commit()
        conn.close()
        logger.info(f'Seed trades deleted: {deleted}')
        return jsonify({'ok': True, 'deleted': deleted,
                        'message': f'Removed {deleted} seeded test trades'})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/apex/journal', methods=['GET'])
def apex_journal():
    """Return all closed trades for the trade journal table."""
    try:
        from trade_tracker import init_trades_table
        init_trades_table()
        conn = _db.connect()
        rows = conn.execute(
            'SELECT id,symbol,direction,setup,mode,entry_price,stop,target,rr_planned,'
            '       session,quality,entry_time,exit_price,exit_time,exit_reason,pnl_r,bars_held,notes '
            'FROM apex_trades WHERE status=? ORDER BY entry_time DESC',
            ('closed',)
        ).fetchall()
        conn.close()
        cols = ['id','symbol','direction','setup','mode','entry_price','stop','target','rr_planned',
                'session','quality','entry_time','exit_price','exit_time','exit_reason','pnl_r','bars_held','notes']
        trades = []
        for row in rows:
            t = dict(zip(cols, row))
            # Compute duration in minutes
            try:
                from datetime import datetime, timezone
                ent = datetime.fromisoformat(t['entry_time'].replace('Z','+00:00'))
                ext = datetime.fromisoformat(t['exit_time'].replace('Z','+00:00'))
                t['duration_mins'] = round((ext - ent).total_seconds() / 60)
            except Exception:
                t['duration_mins'] = None
            # Compute pts (price movement)
            try:
                ep, xp = float(t['entry_price']), float(t['exit_price'])
                t['pts'] = round(xp - ep if t['direction'] == 'long' else ep - xp, 2)
            except Exception:
                t['pts'] = None
            trades.append(t)
        return jsonify({'ok': True, 'trades': trades, 'count': len(trades)})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/apex/fvg_zones/<symbol>', methods=['GET'])
def apex_fvg_zones(symbol):
    """Return all detected FVG zones with scores for the FVG watch panel."""
    symbol = symbol.upper()
    if symbol not in INSTRUMENTS:
        return jsonify({'ok': False, 'error': 'Unknown symbol'})
    try:
        from fvg_engine import detect_fvgs, score_fvg, get_htf_bias, calc_atr
        from market_structure import load_bars
        from datetime import datetime, timezone

        df_15m = load_bars(symbol, '15min', limit=200)
        if df_15m.empty:
            return jsonify({'ok': True, 'symbol': symbol, 'zones': []})

        atr_15m = calc_atr(df_15m, 14)
        fvgs    = detect_fvgs(df_15m, atr_15m, min_atr_mult=0.3, lookback=96)
        bias    = get_htf_bias(symbol)
        now     = datetime.now(timezone.utc)

        current_price = None
        try:
            from data_feed import get_latest_price
            current_price = get_latest_price(symbol)
        except Exception:
            pass
        if not current_price:
            current_price = float(df_15m['close'].iloc[-1])

        # Scoring params computed once
        current_bar_time = df_15m.index[-1]
        vol_baseline     = float(df_15m['volume'].tail(20).mean())

        zones = []
        for fvg in fvgs:
            age_bars = len(df_15m) - df_15m.index.searchsorted(fvg['formed_at'])
            size_atr = round(fvg['size'] / fvg['atr'], 2) if fvg['atr'] else 0
            try:
                score = score_fvg(fvg, df_15m, current_bar_time, vol_baseline)
            except Exception:
                score = 0
            # Status
            if fvg['type'] == 'bullish':
                in_zone = fvg['bottom'] <= current_price <= fvg['top']
                status  = 'triggered' if in_zone else ('expired' if current_price > fvg['top'] * 1.002 else 'watching')
            else:
                in_zone = fvg['bottom'] <= current_price <= fvg['top']
                status  = 'triggered' if in_zone else ('expired' if current_price < fvg['bottom'] * 0.998 else 'watching')

            formed_str = str(fvg['formed_at'])
            zones.append({
                'type':      fvg['type'],
                'top':       round(fvg['top'], 2),
                'bottom':    round(fvg['bottom'], 2),
                'mid':       round(fvg['mid'], 2),
                'size_atr':  size_atr,
                'age_bars':  int(age_bars),
                'score':     int(score),
                'formed_at': formed_str[:16],
                'status':    status,
            })

        zones.sort(key=lambda z: z['score'], reverse=True)
        return jsonify({'ok': True, 'symbol': symbol, 'zones': zones[:20], 'bias': bias})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/apex/health', methods=['GET'])
def apex_health():
    """System health — scheduler state, live feed stats, DB row counts, Setup F status."""
    import time as _time_mod
    tg_ok = bool(TELEGRAM_TOKEN and TELEGRAM_CHAT)
    now_ts = _time_mod.time()

    # ── Scheduler state ──────────────────────────────────────
    sched_hb = getattr(background_scheduler, '_last_heartbeat', None)
    sched_hb_ago = round(now_ts - sched_hb) if sched_hb else None
    fired_today_keys = list(_fired_today.keys()) if _fired_today else []
    fired_today_date = getattr(background_scheduler, '_fired_today_date', None)

    # ── Live feed stats ───────────────────────────────────────
    feed_stats = {}
    try:
        from data_feed import get_live_feed_stats
        feed_stats = get_live_feed_stats() or {}
    except Exception as _fe:
        feed_stats = {'error': str(_fe)}

    # ── DB row counts ─────────────────────────────────────────
    db_counts = {}
    try:
        conn = _db.connect()
        for tbl in ('ohlcv', 'apex_trades', 'fvg_alerted_zones', 'swing_alerted_signals'):
            try:
                db_counts[tbl] = conn.execute(f'SELECT COUNT(*) FROM {tbl}').fetchone()[0]
            except Exception:
                db_counts[tbl] = None
        # Per-symbol per-timeframe 1min bar counts
        ohlcv_detail = {}
        for sym in ('NQ', 'ES', 'GC', 'MNQ'):
            for tf in ('1min', '5min', '15min'):
                try:
                    cnt = conn.execute(
                        'SELECT COUNT(*) FROM ohlcv WHERE symbol=? AND timeframe=? AND ts>0',
                        (sym, tf)
                    ).fetchone()[0]
                    ohlcv_detail[f'{sym}_{tf}'] = cnt
                except Exception:
                    ohlcv_detail[f'{sym}_{tf}'] = None
        conn.close()
        db_counts['ohlcv_detail'] = ohlcv_detail
    except Exception as _dbe:
        db_counts = {'error': str(_dbe)}

    # ── Setup F model status ──────────────────────────────────
    setup_f_status = {}
    try:
        from setup_f_ml import get_current_prediction
        for sym in ('MNQ', 'ES'):
            p = get_current_prediction(sym)
            setup_f_status[sym] = p
    except Exception as _sfe:
        setup_f_status = {'error': str(_sfe)}

    # ── PostgreSQL connection check ────────────────────────────
    pg_ok = False
    pg_error = None
    try:
        conn2 = _db.connect()
        conn2.execute('SELECT 1').fetchone()
        conn2.close()
        pg_ok = True
    except Exception as _pge:
        pg_error = str(_pge)

    return jsonify({
        'ok': True,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'telegram':  tg_ok,
        'postgres':  {'ok': pg_ok, 'is_postgres': _db.IS_POSTGRES, 'error': pg_error},
        'scheduler': {
            'last_heartbeat_ago_s': sched_hb_ago,
            'fired_today':          fired_today_keys,
            'fired_today_date':     fired_today_date,
        },
        'live_feed':  feed_stats,
        'db_counts':  db_counts,
        'setup_f':    setup_f_status,
    })


@app.route('/api/research/health', methods=['GET'])
def research_health():
    """Latest strategy health log entry per setup plus 4-week trend scores."""
    try:
        import research_division as _rd_mod
        conn = _db.connect()
        # Use MAX(id) to guarantee latest row per setup — MAX(date/week) returns
        # multiple rows when the same date appears more than once (same-day reruns)
        rows = conn.execute(
            "SELECT s.setup_id, s.health_score, s.alert_level, s.sharpe_30d, "
            "       s.sharpe_benchmark, s.win_rate, s.win_rate_benchmark, "
            "       s.signal_count_week, s.expectancy, s.week_start, s.notes, "
            "       s.backtest_score, s.live_score "
            "FROM strategy_health_log s "
            "WHERE s.id IN ("
            "  SELECT MAX(id) FROM strategy_health_log GROUP BY setup_id"
            ") ORDER BY s.setup_id"
        ).fetchall()

        latest = {r[0]: r for r in rows}

        # Backtest: latest row per setup by MAX(id)
        bt_rows = conn.execute(
            "SELECT b.setup_id, b.bars_analysed, b.edge_score "
            "FROM backtest_results b "
            "WHERE b.id IN (SELECT MAX(id) FROM backtest_results GROUP BY setup_id)"
        ).fetchall()
        bt_map = {r[0]: {'bars_analysed': r[1], 'edge_score': r[2]} for r in bt_rows}

        from datetime import date, timedelta
        four_weeks_ago = (date.today() - timedelta(weeks=4)).isoformat()
        trend_rows = conn.execute(
            "SELECT setup_id, health_score, week_start "
            "FROM strategy_health_log "
            "WHERE week_start >= ? "
            "ORDER BY setup_id, week_start ASC",
            (four_weeks_ago,)
        ).fetchall()
        conn.close()

        trend_map = {}
        for sid, score, ws in trend_rows:
            trend_map.setdefault(sid, []).append(score)

        all_setups = ['A', 'B', 'C', 'D', 'E', 'H', 'I']
        setups_out = []
        for sid in all_setups:
            r  = latest.get(sid)
            bt = bt_map.get(sid, {})
            # backtest_score: prefer the health log column, fall back to backtest_results
            bt_score = (r[11] if r and len(r) > 11 else None) or bt.get('edge_score')
            lv_score = r[12] if r and len(r) > 12 else None
            setups_out.append({
                'setup_id':          sid,
                'health_score':      r[1]  if r else None,
                'alert_level':       r[2]  if r else 'INSUFFICIENT_DATA',
                'sharpe_30d':        r[3]  if r else None,
                'sharpe_benchmark':  r[4]  if r else _rd_mod.BENCHMARKS.get(sid, {}).get('sharpe'),
                'win_rate':          r[5]  if r else None,
                'win_rate_benchmark':r[6]  if r else _rd_mod.BENCHMARKS.get(sid, {}).get('wr'),
                'signal_count_week': r[7]  if r else None,
                'expectancy':        r[8]  if r else None,
                'week_start':        r[9]  if r else None,
                'notes':             r[10] if r else None,
                'backtest_score':    bt_score,
                'live_score':        lv_score,
                'bars_analysed':     bt.get('bars_analysed', 0),
                'trend':             trend_map.get(sid, []),
            })

        return jsonify({'ok': True, 'setups': setups_out})
    except Exception as e:
        logger.error(f'research_health error: {e}')
        return jsonify({'ok': False, 'error': str(e), 'setups': []})


@app.route('/api/research/shadow', methods=['GET'])
def research_shadow():
    """All active shadow lab candidates."""
    try:
        conn = _db.connect()
        rows = conn.execute(
            "SELECT strategy_name, description, week_number, total_weeks, "
            "       paper_sharpe, paper_win_rate, paper_total_r, paper_signal_count, "
            "       backtest_sharpe, backtest_win_rate, "
            "       promotion_eligible_date, status, entered_date "
            "FROM shadow_lab WHERE status = 'ACTIVE' ORDER BY id"
        ).fetchall()
        conn.close()
        cols = [
            'strategy_name', 'description', 'week_number', 'total_weeks',
            'paper_sharpe', 'paper_win_rate', 'paper_total_r', 'paper_signal_count',
            'backtest_sharpe', 'backtest_win_rate',
            'promotion_eligible_date', 'status', 'entered_date',
        ]
        candidates = [dict(zip(cols, r)) for r in rows]
        return jsonify({'ok': True, 'candidates': candidates})
    except Exception as e:
        logger.error(f'research_shadow error: {e}')
        return jsonify({'ok': False, 'error': str(e), 'candidates': []})


@app.route('/api/control/strategies', methods=['GET'])
def control_strategies():
    """Return enabled/disabled state of all setups from strategy_config."""
    try:
        conn = _db.connect()
        rows = conn.execute(
            "SELECT setup_id, enabled, disabled_reason, disabled_at, enabled_at, "
            "       updated_by, created_at "
            "FROM strategy_config ORDER BY setup_id"
        ).fetchall()
        conn.close()
        cols = ['setup_id', 'enabled', 'disabled_reason', 'disabled_at',
                'enabled_at', 'updated_by', 'created_at']
        strategies = []
        for r in rows:
            d = dict(zip(cols, r))
            d['enabled'] = bool(d['enabled'])
            strategies.append(d)
        # Include any defaults not yet in DB
        existing = {s['setup_id'] for s in strategies}
        for sid, enabled in _STRATEGY_DEFAULTS.items():
            if sid not in existing:
                strategies.append({'setup_id': sid, 'enabled': enabled,
                                   'disabled_reason': None, 'disabled_at': None,
                                   'enabled_at': None, 'updated_by': 'default',
                                   'created_at': None})
        strategies.sort(key=lambda x: x['setup_id'])
        return jsonify({'ok': True, 'strategies': strategies})
    except Exception as e:
        logger.error(f'control_strategies error: {e}')
        return jsonify({'ok': False, 'error': str(e), 'strategies': []})


@app.route('/api/control/strategies/<setup_id>/toggle', methods=['POST'])
def control_toggle(setup_id):
    """Toggle a setup's enabled state. Optionally accepts JSON {reason: str}."""
    sid = setup_id.upper().strip()
    if sid not in _STRATEGY_DEFAULTS:
        return jsonify({'ok': False, 'error': f'Unknown setup_id: {sid}'}), 400
    try:
        data   = request.get_json(silent=True) or {}
        reason = (data.get('reason') or '').strip()[:200] or None
        conn   = _db.connect()
        row    = conn.execute(
            "SELECT enabled FROM strategy_config WHERE setup_id=?", (sid,)
        ).fetchone()
        now_ts = datetime.now(timezone.utc).isoformat()
        if row is None:
            # Insert default then toggle
            conn.execute(
                "INSERT INTO strategy_config (setup_id, enabled) VALUES (?,?)",
                (sid, _STRATEGY_DEFAULTS.get(sid, True))
            )
            conn.commit()
            row = conn.execute(
                "SELECT enabled FROM strategy_config WHERE setup_id=?", (sid,)
            ).fetchone()
        currently_enabled = bool(row[0])
        new_enabled = not currently_enabled
        if new_enabled:
            conn.execute(
                "UPDATE strategy_config SET enabled=?, enabled_at=?, "
                "disabled_reason=NULL, updated_by='dashboard' WHERE setup_id=?",
                (True, now_ts, sid)
            )
        else:
            conn.execute(
                "UPDATE strategy_config SET enabled=?, disabled_at=?, "
                "disabled_reason=?, updated_by='dashboard' WHERE setup_id=?",
                (False, now_ts, reason, sid)
            )
        conn.commit()
        conn.close()

        # Force cache refresh immediately
        _refresh_strategy_cache()

        # Send Telegram alert
        try:
            from live_scanner import send_telegram as _ctl_tg
            setup_name = {
                'A': 'Sweep + OB', 'B': 'ChoCh + OB', 'C': 'BOS + OB',
                'D': 'FVG Fill', 'E': 'EMA50 Pullback', 'F': 'ML Probability',
                'H': 'VWAP Reversion', 'I': 'Mathematical Alpha',
            }.get(sid, sid)
            emoji = '✅' if new_enabled else '⚠️'
            action = 'ENABLED' if new_enabled else 'DISABLED'
            reason_txt = f' — Reason: {reason}' if reason else ''
            _ctl_tg(
                f'{emoji} <b>WISE MERIDIAN CAPITAL</b>\n'
                f'Setup {sid} ({setup_name}) <b>{action}</b> via Control Centre{reason_txt}\n'
                f'<i>Changes take effect within 60 seconds</i>',
                message_type='kill_switch'  # use permitted type for control-centre alerts
            )
        except Exception as _tg_e:
            logger.warning(f'Control Centre Telegram alert failed: {_tg_e}')

        logger.info(
            f'Control Centre: Setup {sid} {"enabled" if new_enabled else "disabled"}'
            + (f' — {reason}' if reason else '')
        )
        return jsonify({
            'ok':       True,
            'setup_id': sid,
            'enabled':  new_enabled,
            'reason':   reason,
            'changed_at': now_ts,
        })
    except Exception as e:
        logger.error(f'control_toggle {sid}: {e}', exc_info=True)
        return jsonify({'ok': False, 'error': str(e)}), 500


def is_instrument_paper(setup_id: str, symbol: str) -> bool:
    """
    Return True if this setup/instrument combination is paper-only (no Tradovate execution).
    Checks strategy_config.paper_instruments from DB cache, falls back to SETUP_PAPER_INSTRUMENTS.
    Called from _execute_via_tradovate wrappers per setup.
    """
    sid = setup_id.upper().strip()[0]  # take first char: 'I_mathematical_alpha' → 'I'
    sym = symbol.upper().strip()
    # Check live cache first
    row = _strategy_enabled_cache.get(sid, {})
    paper_str = row.get('paper_instruments', '') if isinstance(row, dict) else ''
    if paper_str:
        return sym in {p.strip() for p in paper_str.split(',') if p.strip()}
    # Fall back to compile-time defaults
    return sym in SETUP_PAPER_INSTRUMENTS.get(sid, set())


@app.route('/api/control/strategies/<setup_id>/instruments/<symbol>/toggle', methods=['POST'])
def control_instrument_toggle(setup_id, symbol):
    """
    Toggle paper/live execution for a specific setup+instrument combination.
    E.g. POST /api/control/strategies/I/instruments/MNQ/toggle
    Paper = signal fires, trade logged, Telegram sent — but NO Tradovate order.
    Live  = full execution including Tradovate bracket order.
    """
    sid = setup_id.upper().strip()
    sym = symbol.upper().strip()
    if sid not in _STRATEGY_DEFAULTS:
        return jsonify({'ok': False, 'error': f'Unknown setup: {sid}'}), 400
    if sym not in ('MNQ', 'ES', 'GC', 'NQ'):
        return jsonify({'ok': False, 'error': f'Unknown instrument: {sym}'}), 400
    try:
        conn    = _db.connect()
        row     = conn.execute(
            "SELECT paper_instruments FROM strategy_config WHERE setup_id=?", (sid,)
        ).fetchone()
        current_paper = set(
            p.strip() for p in (row[0] or '').split(',') if p.strip()
        ) if row else set(SETUP_PAPER_INSTRUMENTS.get(sid, set()))

        if sym in current_paper:
            current_paper.discard(sym)
            new_status = 'LIVE'
        else:
            current_paper.add(sym)
            new_status = 'PAPER'

        new_paper_str = ','.join(sorted(current_paper))
        conn.execute(
            "UPDATE strategy_config SET paper_instruments=? WHERE setup_id=?",
            (new_paper_str, sid)
        )
        conn.commit()
        conn.close()
        _refresh_strategy_cache()
        logger.info(f'Control Centre: Setup {sid} {sym} set to {new_status} (paper={new_paper_str})')
        return jsonify({
            'ok': True, 'setup_id': sid, 'symbol': sym,
            'status': new_status, 'paper_instruments': sorted(current_paper),
        })
    except Exception as e:
        logger.error(f'control_instrument_toggle {sid}/{sym}: {e}', exc_info=True)
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/control/strategies/<setup_id>/regime_gate/toggle', methods=['POST'])
def control_regime_gate_toggle(setup_id):
    """Toggle regime gating on/off for a specific setup."""
    sid = setup_id.upper().strip()
    if sid not in _STRATEGY_DEFAULTS:
        return jsonify({'ok': False, 'error': f'Unknown setup: {sid}'}), 400
    try:
        conn = _db.connect()
        row  = conn.execute(
            "SELECT regime_gating_enabled FROM strategy_config WHERE setup_id=?", (sid,)
        ).fetchone()
        current = bool(row[0]) if row else False
        new_val  = not current
        conn.execute(
            "UPDATE strategy_config SET regime_gating_enabled=? WHERE setup_id=?",
            (1 if new_val else 0, sid)
        )
        conn.commit()
        conn.close()
        _refresh_strategy_cache()
        logger.info(f'Control Centre: Setup {sid} regime gating set to {new_val}')
        return jsonify({'ok': True, 'setup_id': sid, 'regime_gating_enabled': new_val})
    except Exception as e:
        logger.error(f'control_regime_gate_toggle {sid}: {e}', exc_info=True)
        return jsonify({'ok': False, 'error': str(e)}), 500


# Update control_strategies to include paper_instruments and regime fields
@app.route('/api/control/strategies/full', methods=['GET'])
def control_strategies_full():
    """Return full strategy config including paper_instruments and regime gating."""
    try:
        conn = _db.connect()
        rows = conn.execute(
            "SELECT setup_id, enabled, disabled_reason, optimal_regimes, "
            "       regime_gating_enabled, paper_instruments "
            "FROM strategy_config ORDER BY setup_id"
        ).fetchall()
        conn.close()
        SETUP_NAMES_MAP = {
            'A': 'Sweep + OB', 'B': 'ChoCh + OB', 'C': 'BOS + OB',
            'D': 'FVG Fill',   'E': 'EMA50 Pullback', 'F': 'ML Probability',
            'H': 'VWAP Reversion', 'I': 'Mathematical Alpha',
        }
        result = []
        for sid, enabled, reason, opt_r, rg_enabled, paper_i in rows:
            paper_set  = {p.strip() for p in (paper_i or '').split(',') if p.strip()}
            result.append({
                'setup_id':             sid,
                'name':                 SETUP_NAMES_MAP.get(sid, sid),
                'enabled':              bool(enabled),
                'disabled_reason':      reason,
                'optimal_regimes':      [r.strip() for r in (opt_r or '').split(',') if r.strip()],
                'regime_gating_enabled': bool(rg_enabled),
                'paper_instruments':    sorted(paper_set),
                'live_instruments':     sorted({'MNQ', 'ES', 'GC'} - paper_set),
            })
        return jsonify({'ok': True, 'strategies': result})
    except Exception as e:
        logger.error(f'control_strategies_full error: {e}')
        return jsonify({'ok': False, 'error': str(e), 'strategies': []}), 500


@app.route('/api/research/backtest_ping', methods=['GET'])
def research_backtest_ping():
    """Diagnostic: test backtest_results table with a direct INSERT/SELECT/DELETE."""
    try:
        from datetime import date as _date
        conn = _db.connect()
        # Check table exists
        try:
            count = conn.execute("SELECT COUNT(*) FROM backtest_results").fetchone()[0]
        except Exception as te:
            conn.close()
            return jsonify({'ok': False, 'step': 'count', 'error': str(te)})
        # Attempt test insert
        today_str = _date.today().isoformat()
        try:
            conn.execute(
                "INSERT INTO backtest_results "
                "(setup_id, lookback_days, run_date, total_signals, win_rate, sharpe, "
                " avg_r, expectancy, max_drawdown, profit_factor, "
                " benchmark_sharpe, benchmark_win_rate, "
                " sharpe_vs_benchmark, wr_vs_benchmark, edge_score, bars_analysed) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                ('TEST', 1, today_str, 5, 0.5, 1.5, 0.3, 0.3, 0.1, 1.2,
                 5.0, 0.5, 10.0, 5.0, 65, 100)
            )
            conn.commit()
        except Exception as ie:
            conn.close()
            return jsonify({'ok': False, 'step': 'insert', 'error': str(ie), 'type': type(ie).__name__})
        # Clean up test row
        try:
            conn.execute("DELETE FROM backtest_results WHERE setup_id='TEST'")
            conn.commit()
        except Exception:
            pass
        conn.close()
        return jsonify({'ok': True, 'existing_rows': count, 'table': 'backtest_results'})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/research/backtest', methods=['GET'])
def research_backtest():
    """Latest backtest_results per setup plus 30-day edge score trend."""
    try:
        conn = _db.connect()
        rows = conn.execute(
            "SELECT b.setup_id, b.run_date, b.sharpe, b.win_rate, b.edge_score, "
            "       b.sharpe_vs_benchmark, b.wr_vs_benchmark, "
            "       b.bars_analysed, b.total_signals, b.profit_factor, b.max_drawdown "
            "FROM backtest_results b "
            "WHERE b.id IN (SELECT MAX(id) FROM backtest_results GROUP BY setup_id) "
            "ORDER BY b.setup_id"
        ).fetchall()
        cols = ['setup_id', 'run_date', 'sharpe', 'win_rate', 'edge_score',
                'sharpe_vs_benchmark', 'wr_vs_benchmark',
                'bars_analysed', 'total_signals', 'profit_factor', 'max_drawdown']
        results = [dict(zip(cols, r)) for r in rows]

        from datetime import date, timedelta
        trend_cutoff = (date.today() - timedelta(days=30)).isoformat()
        # Trend: one edge_score per (setup_id, run_date) day — use MAX(id) per day
        trend_rows = conn.execute(
            "SELECT t.setup_id, t.edge_score, t.run_date "
            "FROM backtest_results t "
            "WHERE t.id IN ("
            "  SELECT MAX(id) FROM backtest_results "
            "  WHERE run_date >= ? GROUP BY setup_id, run_date"
            ") ORDER BY t.setup_id, t.run_date ASC",
            (trend_cutoff,)
        ).fetchall()
        conn.close()
        trends = {}
        for sid, es, rd in trend_rows:
            trends.setdefault(sid, []).append(es)

        return jsonify({'ok': True, 'results': results, 'trends': trends})
    except Exception as e:
        logger.error(f'research_backtest error: {e}')
        return jsonify({'ok': False, 'error': str(e), 'results': [], 'trends': {}})


@app.route('/api/research/run_backtest', methods=['POST'])
def research_run_backtest():
    """Manually trigger daily backtest for all 7 setups."""
    try:
        import research_division as _rd_mod
        results = _rd_mod.run_daily_backtest()
        write_errors = results.pop('_write_errors', {})
        summary = {}
        for sid, bt in results.items():
            if bt is not None:
                summary[sid] = {
                    'edge_score':    bt.edge_score,
                    'total_signals': bt.total_signals,
                    'sharpe':        round(bt.sharpe, 3) if bt.sharpe else None,
                    'win_rate':      round(bt.win_rate, 3),
                    'bars_analysed': bt.bars_analysed,
                }
            else:
                summary[sid] = None
        return jsonify({'ok': True, 'results': summary, 'write_errors': write_errors})
    except Exception as e:
        logger.error(f'research_run_backtest error: {e}', exc_info=True)
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/research/run_check', methods=['POST'])
def research_run_check():
    """Manually trigger the weekly research health check + shadow lab scoring."""
    try:
        import research_division as _rd_mod
        health  = _rd_mod.run_weekly_health_check()
        updated = _rd_mod.score_shadow_lab()
        return jsonify({
            'ok':            True,
            'health_setups': len(health),
            'shadow_updated': len(updated),
        })
    except Exception as e:
        logger.error(f'research_run_check error: {e}', exc_info=True)
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/research/send_report', methods=['POST'])
def research_send_report():
    """Manually trigger the weekly Telegram report."""
    try:
        import research_division as _rd_mod
        sent = _rd_mod.generate_weekly_telegram_report()
        return jsonify({'ok': True, 'telegram_sent': sent})
    except Exception as e:
        logger.error(f'research_send_report error: {e}', exc_info=True)
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/research/decisions', methods=['GET'])
def research_decisions():
    """All pending research decisions."""
    try:
        conn = _db.connect()
        rows = conn.execute(
            "SELECT id, decision_type, subject, recommendation, created_at "
            "FROM research_decisions WHERE status = 'PENDING' "
            "ORDER BY created_at DESC"
        ).fetchall()
        conn.close()
        cols = ['id', 'decision_type', 'subject', 'recommendation', 'created_at']
        decisions = [dict(zip(cols, r)) for r in rows]
        return jsonify({'ok': True, 'decisions': decisions})
    except Exception as e:
        logger.error(f'research_decisions error: {e}')
        return jsonify({'ok': False, 'error': str(e), 'decisions': []})


# ─── Phase 3 decision workflow endpoints ────────────────────────────────────

@app.route('/api/research/decisions/<int:decision_id>/approve', methods=['POST'])
def research_decision_approve(decision_id):
    """Approve a pending research decision (PROMOTION_REVIEW → APPROVED)."""
    try:
        conn = _db.connect()
        row = conn.execute(
            "SELECT id, decision_type, subject, status FROM research_decisions WHERE id = ?",
            (decision_id,)
        ).fetchone()
        if not row:
            conn.close()
            return jsonify({'ok': False, 'error': 'Decision not found'}), 404
        if row[3] not in ('PENDING',):
            conn.close()
            return jsonify({'ok': False, 'error': f'Cannot approve decision in status {row[3]}'}), 400
        data = request.get_json(force=True) or {}
        note = data.get('note', '')
        now_iso = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE research_decisions SET status='APPROVED', decided_at=?, outcome=? WHERE id=?",
            (now_iso, f'APPROVED{": " + note if note else ""}', decision_id)
        )
        conn.commit()
        conn.close()
        logger.info(f'Research Decision {decision_id} ({row[2]}): APPROVED')
        return jsonify({'ok': True, 'decision_id': decision_id, 'status': 'APPROVED'})
    except Exception as e:
        logger.error(f'research_decision_approve: {e}')
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/research/decisions/<int:decision_id>/reject', methods=['POST'])
def research_decision_reject(decision_id):
    """Reject a pending research decision with an optional reason."""
    try:
        conn = _db.connect()
        row = conn.execute(
            "SELECT id, decision_type, subject, status FROM research_decisions WHERE id = ?",
            (decision_id,)
        ).fetchone()
        if not row:
            conn.close()
            return jsonify({'ok': False, 'error': 'Decision not found'}), 404
        if row[3] not in ('PENDING',):
            conn.close()
            return jsonify({'ok': False, 'error': f'Cannot reject decision in status {row[3]}'}), 400
        data = request.get_json(force=True) or {}
        reason = data.get('reason', 'Manual rejection')
        now_iso = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE research_decisions SET status='REJECTED', decided_at=?, outcome=? WHERE id=?",
            (now_iso, f'REJECTED: {reason}', decision_id)
        )
        conn.commit()
        conn.close()
        logger.info(f'Research Decision {decision_id} ({row[2]}): REJECTED — {reason}')
        return jsonify({'ok': True, 'decision_id': decision_id, 'status': 'REJECTED',
                        'reason': reason})
    except Exception as e:
        logger.error(f'research_decision_reject: {e}')
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/research/decisions/<int:decision_id>/extend', methods=['POST'])
def research_decision_extend(decision_id):
    """Extend shadow lab by 4 weeks: reset week_number to current-4 and continue paper trading."""
    try:
        conn = _db.connect()
        row = conn.execute(
            "SELECT id, decision_type, subject, status FROM research_decisions WHERE id = ?",
            (decision_id,)
        ).fetchone()
        if not row:
            conn.close()
            return jsonify({'ok': False, 'error': 'Decision not found'}), 404
        subject = row[2]
        # Find matching shadow lab entry
        shadow_row = conn.execute(
            "SELECT id, week_number, total_weeks FROM shadow_lab WHERE strategy_name = ?",
            (subject,)
        ).fetchone()
        now_iso = datetime.now(timezone.utc).isoformat()
        if shadow_row:
            shadow_id, week_num, total_weeks = shadow_row
            new_week = max(0, int(week_num or 0) - 4)
            conn.execute(
                "UPDATE shadow_lab SET week_number=?, status='ACTIVE' WHERE id=?",
                (new_week, shadow_id)
            )
        conn.execute(
            "UPDATE research_decisions SET status='EXTENDED', decided_at=?, "
            "outcome='Extended by 4 weeks for continued paper trading' WHERE id=?",
            (now_iso, decision_id)
        )
        conn.commit()
        conn.close()
        logger.info(f'Research Decision {decision_id} ({subject}): EXTENDED 4 weeks')
        return jsonify({'ok': True, 'decision_id': decision_id, 'status': 'EXTENDED'})
    except Exception as e:
        logger.error(f'research_decision_extend: {e}')
        return jsonify({'ok': False, 'error': str(e)}), 500


# ─── Phase 3 Discovery Engine endpoints ──────────────────────────────────────

@app.route('/api/research/hypotheses', methods=['GET'])
def research_hypotheses():
    """Return hypothesis_log results. Optional ?status=SIGNIFICANT filter."""
    try:
        conn = _db.connect()
        status_filter = request.args.get('status', None)
        if status_filter:
            rows = conn.execute(
                "SELECT id, hypothesis_id, description, category, instrument, "
                "       lookback_days, signals_generated, win_rate, sharpe, avg_r, "
                "       information_coefficient, p_value, status, run_date, created_at "
                "FROM hypothesis_log WHERE status = ? "
                "ORDER BY COALESCE(ABS(information_coefficient), 0) DESC LIMIT 200",
                (status_filter.upper(),)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, hypothesis_id, description, category, instrument, "
                "       lookback_days, signals_generated, win_rate, sharpe, avg_r, "
                "       information_coefficient, p_value, status, run_date, created_at "
                "FROM hypothesis_log "
                "ORDER BY run_date DESC, COALESCE(ABS(information_coefficient), 0) DESC LIMIT 200"
            ).fetchall()
        conn.close()
        cols = ['id', 'hypothesis_id', 'description', 'category', 'instrument',
                'lookback_days', 'signals_generated', 'win_rate', 'sharpe', 'avg_r',
                'information_coefficient', 'p_value', 'status', 'run_date', 'created_at']
        hypotheses = [dict(zip(cols, r)) for r in rows]

        # Summary counts
        all_rows = _db.connect()
        try:
            counts = all_rows.execute(
                "SELECT status, COUNT(*) FROM hypothesis_log GROUP BY status"
            ).fetchall()
            summary = {r[0]: r[1] for r in counts}
            in_shadow = all_rows.execute(
                "SELECT COUNT(*) FROM shadow_lab WHERE status='ACTIVE'"
            ).fetchone()
            patterns_active = all_rows.execute(
                "SELECT COUNT(*) FROM pattern_library WHERE status='ACTIVE'"
            ).fetchone()
        finally:
            all_rows.close()

        return jsonify({
            'ok': True,
            'hypotheses': hypotheses,
            'summary': {
                'tested_this_week': sum(summary.values()),
                'significant': summary.get('SIGNIFICANT', 0),
                'testing': summary.get('TESTING', 0),
                'rejected': summary.get('REJECTED', 0),
                'in_shadow_lab': in_shadow[0] if in_shadow else 0,
                'patterns_active': patterns_active[0] if patterns_active else 0,
            }
        })
    except Exception as e:
        logger.error(f'research_hypotheses error: {e}')
        return jsonify({'ok': False, 'error': str(e), 'hypotheses': []})


@app.route('/api/research/combinations', methods=['GET'])
def research_combinations():
    """Return top 10 feature combinations by OOS IC."""
    try:
        conn = _db.connect()
        rows = conn.execute(
            "SELECT id, features, oos_ic, oos_auc, run_date, created_at "
            "FROM feature_combinations "
            "ORDER BY ABS(COALESCE(oos_ic, 0)) DESC LIMIT 10"
        ).fetchall()
        conn.close()
        cols = ['id', 'features', 'oos_ic', 'oos_auc', 'run_date', 'created_at']
        combinations = [dict(zip(cols, r)) for r in rows]
        return jsonify({'ok': True, 'combinations': combinations})
    except Exception as e:
        logger.error(f'research_combinations error: {e}')
        return jsonify({'ok': False, 'error': str(e), 'combinations': []})


@app.route('/api/research/patterns', methods=['GET'])
def research_patterns():
    """Return pattern_library ACTIVE patterns."""
    try:
        conn = _db.connect()
        rows = conn.execute(
            "SELECT id, pattern_id, name, description, discovery_source, instrument, "
            "       signals_observed, win_rate, sharpe, information_coefficient, "
            "       first_observed, last_validated, decay_score, status, created_at "
            "FROM pattern_library WHERE status != 'DEAD' "
            "ORDER BY decay_score DESC, COALESCE(information_coefficient, 0) DESC"
        ).fetchall()
        conn.close()
        cols = ['id', 'pattern_id', 'name', 'description', 'discovery_source', 'instrument',
                'signals_observed', 'win_rate', 'sharpe', 'information_coefficient',
                'first_observed', 'last_validated', 'decay_score', 'status', 'created_at']
        patterns = [dict(zip(cols, r)) for r in rows]
        return jsonify({'ok': True, 'patterns': patterns})
    except Exception as e:
        logger.error(f'research_patterns error: {e}')
        return jsonify({'ok': False, 'error': str(e), 'patterns': []})


@app.route('/api/apex/regime', methods=['GET'])
def apex_regime():
    """Current regime state for MNQ and ES — read-only, observation module."""
    try:
        from regime_engine import get_current_regime, get_regime_history
        result = {}
        for sym in ('MNQ', 'ES'):
            result[sym] = get_current_regime(sym)
        history = {}
        limit = int(request.args.get('history', 0))
        if limit > 0:
            for sym in ('MNQ', 'ES'):
                history[sym] = get_regime_history(sym, limit=min(limit, 500))
        resp = {'ok': True, 'regime': result}
        if history:
            resp['history'] = history
        return jsonify(resp)
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/apex/meridian', methods=['GET'])
def apex_meridian():
    """Meridian L3 — current TRENDING probability and position multiplier for MNQ and ES."""
    try:
        import meridian_l3 as _ml3
        result = {}
        for sym in ('MNQ', 'ES'):
            if sym not in _ml3._l3_cache:
                _ml3.load_model(sym)
            cache = _ml3._l3_cache.get(sym, {})
            prob = _ml3.predict(sym)
            # read latest hurst/conf from regime_log for multiplier calc
            try:
                _rc = _db.connect()
                _rrow = _rc.execute(
                    'SELECT hurst, confidence FROM regime_log '
                    'WHERE symbol=? ORDER BY timestamp DESC LIMIT 1',
                    (sym,)
                ).fetchone()
                _rc.close()
                cur_hurst = float(_rrow[0] or 0) if _rrow else 0.5
                cur_conf  = float(_rrow[1] or 0) if _rrow else 0.5
            except Exception:
                cur_hurst, cur_conf = 0.5, 0.5
            mult, _ = _ml3.get_position_multiplier(sym, cur_hurst, cur_conf)
            result[sym] = {
                'l3_prob':       round(prob, 4),
                'l3_prob_pct':   round(prob * 100, 1),
                'multiplier':    mult,
                'model_auc':     round(cache.get('auc', 0), 4),
                'deploy_mode':   cache.get('mode', 'unknown'),
                'trained_at':    cache.get('trained_at', None),
                'model_loaded':  sym in _ml3._l3_cache,
            }
        return jsonify({'ok': True, 'meridian_l3': result})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/apex/mtf', methods=['GET'])
def apex_mtf():
    """Multi-timeframe EMA20 bias for MNQ and ES.
    Returns BULLISH/BEARISH for 1min/5min/15min/30min/1hr/4hr per symbol.
    30min is resampled from 5min bars; all others queried directly.
    """
    try:
        import pandas as _pd_mtf
        from market_structure import load_bars as _lb_mtf

        # (label, db_timeframe, resample_rule_or_None, bars_to_load)
        _TFS = [
            ('1min',  '1min',   None,    120),
            ('5min',  '5min',   None,    100),
            ('15min', '15min',  None,     80),
            ('30min', '5min',   '30min', 400),   # resample 5min → 30min
            ('1hr',   '1hour',  None,     60),
            ('4hr',   '4hour',  None,     50),
        ]

        def _ema20(closes):
            return closes.ewm(span=20, adjust=False).mean()

        result = {}
        for sym in ('MNQ', 'ES'):
            sym_data = {}
            for label, db_tf, resample, n in _TFS:
                try:
                    df = _lb_mtf(sym, db_tf, limit=n)
                    if resample:
                        df = df[['open', 'high', 'low', 'close', 'volume']].resample(resample).agg(
                            {'open': 'first', 'high': 'max', 'low': 'min',
                             'close': 'last', 'volume': 'sum'}
                        ).dropna(subset=['close'])
                    if len(df) < 5:
                        sym_data[label] = {'bias': 'UNKNOWN', 'error': 'insufficient bars'}
                        continue
                    closes    = df['close'].astype(float)
                    ema_series = _ema20(closes)
                    ema20_val  = float(ema_series.iloc[-1])
                    last_close = float(closes.iloc[-1])
                    bias = 'BULLISH' if last_close > ema20_val else 'BEARISH'
                    sym_data[label] = {
                        'bias':  bias,
                        'close': round(last_close, 2),
                        'ema20': round(ema20_val, 2),
                    }
                except Exception as _tf_e:
                    sym_data[label] = {'bias': 'UNKNOWN', 'error': str(_tf_e)}
            result[sym] = sym_data

        return jsonify({'ok': True, 'mtf': result, 'timeframes': [t[0] for t in _TFS]})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/apex/forecast', methods=['GET'])
def apex_forecast():
    """
    Meridian Direction Model — directional price forecast.
    Returns per symbol, per validated horizon: direction, calibrated probability,
    contributing signal breakdown, and live accuracy stats.
    Horizons that did not pass AUC gate return deployed=false, reason='insufficient_edge'.
    """
    try:
        import meridian_direction as _mdir
        result   = {}
        accuracy = {}
        for sym in ('MNQ', 'ES'):
            result[sym]   = _mdir.predict_all(sym)
            accuracy[sym] = {}
            for horizon in _mdir.HORIZONS:
                acc = _mdir.get_accuracy_stats(sym, horizon, n=50)
                accuracy[sym][horizon] = acc
        return jsonify({
            'ok':       True,
            'ts':       time.time(),
            'forecast': result,
            'accuracy': accuracy,
        })
    except Exception as e:
        logger.warning(f'apex_forecast error: {e}')
        return jsonify({'ok': False, 'error': str(e)})


# ─────────────────────────────────────────────────────────────────────────────
#  MARKET CONTEXT — helpers for /api/apex/context
#  Pure informational reads: no signals, no buy/sell framing.
# ─────────────────────────────────────────────────────────────────────────────

def _ctx_atr_series(highs, lows, closes, period=14):
    """Return list of ATR(period) values, one per bar from index `period` onward."""
    if len(closes) < period + 1:
        return []
    trs = [max(highs[i] - lows[i],
               abs(highs[i] - closes[i-1]),
               abs(lows[i]  - closes[i-1]))
           for i in range(1, len(closes))]
    return [sum(trs[i-period:i]) / period for i in range(period, len(trs) + 1)]


def _ctx_volume_profile(bars):
    """
    Simple volume-at-price profile for a list of OHLCV bars.
    Returns (poc, vah, val): point of control, value-area high/low.
    Value area = 70 % of total volume, expanded outward from POC.
    """
    if len(bars) < 3:
        return None, None, None
    tick = 0.25  # MNQ and ES both have 0.25 min tick
    price_vol: dict = {}
    for b in bars:
        mid = round(round((b['h'] + b['l']) / 2 / tick) * tick, 2)
        price_vol[mid] = price_vol.get(mid, 0) + (b['v'] or 0)
    if not price_vol or sum(price_vol.values()) == 0:
        return None, None, None
    sorted_p = sorted(price_vol)
    poc = max(price_vol, key=lambda p: price_vol[p])
    total_vol = sum(price_vol.values())
    va_vol = price_vol[poc]
    va_set = {poc}
    lo = hi = sorted_p.index(poc)
    while va_vol < total_vol * 0.70:
        nlo = price_vol.get(sorted_p[lo - 1], 0) if lo > 0 else 0
        nhi = price_vol.get(sorted_p[hi + 1], 0) if hi < len(sorted_p) - 1 else 0
        if nlo == 0 and nhi == 0:
            break
        if nhi >= nlo:
            hi += 1; va_set.add(sorted_p[hi]); va_vol += nhi
        else:
            lo -= 1; va_set.add(sorted_p[lo]); va_vol += nlo
        if lo == 0 and hi == len(sorted_p) - 1:
            break
    return round(poc, 2), round(max(va_set), 2), round(min(va_set), 2)


def _ctx_volatility():
    """ATR(14) vs 20-bar rolling average for MNQ/ES on 5min and 1hr, plus VIX."""
    result: dict = {}
    for sym in ('MNQ', 'ES'):
        sym_data: dict = {}
        for tf_label, tf_db, n in [('5min', '5min', 120), ('1hr', '1hour', 80)]:
            bars = get_ohlcv(sym, tf_db, limit=n)
            if len(bars) < 20:
                sym_data[tf_label] = None
                continue
            highs  = [b['h'] for b in bars]
            lows   = [b['l'] for b in bars]
            closes = [b['c'] for b in bars]
            atrs   = _ctx_atr_series(highs, lows, closes, 14)
            if not atrs:
                sym_data[tf_label] = None
                continue
            current = atrs[-1]
            avg20   = sum(atrs[-20:]) / min(20, len(atrs))
            ratio   = round(current / avg20, 2) if avg20 else None
            label   = ('elevated'      if ratio and ratio >= 1.5  else
                       'above average' if ratio and ratio >= 1.2  else
                       'compressed'    if ratio and ratio <= 0.70 else
                       'below average' if ratio and ratio <= 0.85 else
                       'average')
            sym_data[tf_label] = {
                'atr':       round(current, 4),
                'atr_avg20': round(avg20, 4),
                'ratio':     ratio,
                'label':     label,
            }
        result[sym] = sym_data
    # VIX level + 5-day change (from daily OHLCV stored in DB)
    macro = fetch_macro_live()
    vix_bars = get_ohlcv('VIX', '1day', limit=10)
    vix_5d = (round(float(vix_bars[-1]['c']) - float(vix_bars[-6]['c']), 2)
              if len(vix_bars) >= 6 else None)
    result['vix'] = {
        'current':   macro.get('vix'),
        'change_5d': vix_5d,
        'live':      macro.get('live', False),
        'source':    macro.get('source', 'reference'),
    }
    return result


def _ctx_session_structure():
    """Opening range (first 30 min), prior-day H/L/C, and volume profile for today."""
    result: dict = {}
    now_utc    = datetime.now(timezone.utc)
    today_date = now_utc.date()
    for sym in ('MNQ', 'ES'):
        bars = get_ohlcv(sym, '5min', limit=600)
        if not bars:
            result[sym] = None
            continue
        parsed = []
        for b in bars:
            dt_b = datetime.fromtimestamp(b['t'], tz=timezone.utc)
            parsed.append({**b, '_date': dt_b.date(), '_hour': dt_b.hour})
        today   = [b for b in parsed if b['_date'] == today_date and 13 <= b['_hour'] < 20]
        or_bars = today[:6]
        or_high = round(max(b['h'] for b in or_bars), 2) if or_bars else None
        or_low  = round(min(b['l'] for b in or_bars), 2) if or_bars else None
        pdh = pdl = pdc = None
        prior_dates = sorted({b['_date'] for b in parsed if b['_date'] < today_date}, reverse=True)
        if prior_dates:
            pb = [b for b in parsed if b['_date'] == prior_dates[0] and 13 <= b['_hour'] < 20]
            if pb:
                pdh = round(max(b['h'] for b in pb), 2)
                pdl = round(min(b['l'] for b in pb), 2)
                pdc = round(pb[-1]['c'], 2)
        poc, vah, val = (_ctx_volume_profile(today) if len(today) >= 6 else (None, None, None))
        current = round(today[-1]['c'], 2) if today else (round(parsed[-1]['c'], 2) if parsed else None)
        position = []
        if current and pdh and pdl:
            position.append('above prior day high' if current > pdh else
                            'below prior day low'  if current < pdl else
                            'inside prior day range')
        if current and or_high and or_low:
            position.append('above opening range'  if current > or_high else
                            'below opening range'  if current < or_low  else
                            'inside opening range')
        if current and vah and val:
            position.append('above value area' if current > vah else
                            'below value area'  if current < val  else
                            'inside value area')
        result[sym] = {
            'current_price': current,
            'or_high': or_high, 'or_low': or_low, 'or_complete': len(or_bars) >= 6,
            'pdh': pdh, 'pdl': pdl, 'pdc': pdc,
            'poc': poc, 'vah': vah, 'val': val,
            'position': position,
        }
    return result


def _ctx_momentum():
    """RSI(14) on 5min, 15min, 1hr, 4hr for MNQ and ES — descriptive bands only."""
    import pandas as _pd_ctx
    RSI_TFS = [
        ('5min',  '5min',  None,  100),
        ('15min', '15min', None,   80),
        ('1hr',   '1hour', None,   60),
        ('4hr',   '5min',  '4h', 600),
    ]
    result: dict = {}
    for sym in ('MNQ', 'ES'):
        sym_data: dict = {}
        for label, db_tf, resample, n in RSI_TFS:
            bars = get_ohlcv(sym, db_tf, limit=n)
            if not bars:
                sym_data[label] = None
                continue
            if resample:
                df = _pd_ctx.DataFrame(bars)
                df['dt'] = _pd_ctx.to_datetime(df['t'], unit='s', utc=True)
                closes = (df.set_index('dt')['c']
                           .resample(resample).last().dropna()
                           .astype(float).tolist())
            else:
                closes = [float(b['c']) for b in bars]
            if len(closes) < 16:
                sym_data[label] = None
                continue
            rsi = calc_rsi(closes)
            if rsi is None:
                sym_data[label] = None
                continue
            band = ('oversold'       if rsi < 30 else
                    'overbought'     if rsi > 70 else
                    'below midpoint' if rsi < 45 else
                    'above midpoint' if rsi > 55 else
                    'neutral')
            sym_data[label] = {'rsi': rsi, 'band': band}
        result[sym] = sym_data
    return result


def _ctx_correlation():
    """Rolling 20-bar Pearson correlation between MNQ and ES 5min returns."""
    mnq_bars = get_ohlcv('MNQ', '5min', limit=25)
    es_bars  = get_ohlcv('ES',  '5min', limit=25)
    mnq_map  = {b['t']: float(b['c']) for b in mnq_bars}
    es_map   = {b['t']: float(b['c']) for b in es_bars}
    common   = sorted(mnq_map.keys() & es_map.keys())
    corr = None
    note = 'Insufficient data'
    if len(common) >= 10:
        m_px = [mnq_map[t] for t in common]
        e_px = [es_map[t]  for t in common]
        m_r  = [m_px[i]/m_px[i-1]-1 for i in range(1, len(m_px))]
        e_r  = [e_px[i]/e_px[i-1]-1 for i in range(1, len(e_px))]
        n2 = len(m_r)
        mm, me = sum(m_r)/n2, sum(e_r)/n2
        cov = sum((m_r[i]-mm)*(e_r[i]-me) for i in range(n2)) / n2
        sm  = (sum((x-mm)**2 for x in m_r)/n2) ** 0.5
        se  = (sum((x-me)**2 for x in e_r)/n2) ** 0.5
        if sm > 0 and se > 0:
            corr = round(cov / (sm * se), 3)
            note = ('Diverging — well below typical 0.9+ range' if corr < 0.70 else
                    'Below typical range'                        if corr < 0.85 else
                    'Near-perfect correlation'                   if corr > 0.99 else
                    'Normal range')
    macro = fetch_macro_live()
    return {
        'mnq_es_corr': corr,
        'corr_note':   note,
        'n_bars':      max(0, len(common) - 1),
        'vix': {'level': macro.get('vix'), 'live': macro.get('live', False),
                'source': macro.get('source', 'reference')},
        'dxy': {'level': macro.get('dxy'), 'live': macro.get('live', False),
                'source': macro.get('source', 'reference')},
    }


def _ctx_calendar():
    """Today's HIGH-impact economic events via the existing CalendarFilter singleton."""
    from calendar_filter import get_filter
    cf     = get_filter()
    status = cf.get_current_status()
    now    = datetime.now(timezone.utc)
    events = cf.get_upcoming_events(hours=32)   # 2h lookback + 32h forward
    for ev in events:
        dt       = datetime.fromisoformat(ev['utc_time'].replace('Z', '+00:00'))
        diff_min = int((now - dt).total_seconds() / 60)
        ev['has_passed'] = (dt < now)
        ev['relative']   = (f'+{abs(diff_min)} min ago' if dt < now
                             else f'in {abs(diff_min)} min')
    return {
        'blocked': status.get('status') == 'BLACKOUT',
        'reason':  status.get('reason') if status.get('status') == 'BLACKOUT' else None,
        'events':  events,
        'source':  'fmp_api' if not status.get('using_fallback') else 'hardcoded_fallback',
        'last_refresh': status.get('last_refresh'),
    }


@app.route('/api/apex/context', methods=['GET'])
def apex_context():
    """
    Market Context block — five purely informational sections for the Market Intel page.
    No signals, no buy/sell recommendations. Framing is descriptive throughout.

    Sections:
      volatility      — ATR(14) vs 20-bar average on 5min and 1hr; VIX level + 5d change
      session_structure — opening range, prior-day H/L/C, value area (POC/VAH/VAL)
      momentum        — RSI(14) on 5min/15min/1hr/4hr; descriptive bands only
      correlation     — rolling 20-bar MNQ–ES return correlation; DXY/VIX reference
      calendar        — upcoming HIGH-impact US economic events; reuses CalendarFilter
    """
    out = {'ok': True, 'ts': time.time()}
    for key, fn in [
        ('volatility',        _ctx_volatility),
        ('session_structure', _ctx_session_structure),
        ('momentum',          _ctx_momentum),
        ('correlation',       _ctx_correlation),
        ('calendar',          _ctx_calendar),
    ]:
        try:
            out[key] = fn()
        except Exception as _e:
            logger.warning(f'apex_context {key}: {_e}')
            out[key] = {'error': str(_e)}
    return jsonify(out)


@app.route('/api/apex/test_log', methods=['POST'])
def apex_test_log():
    """
    Test endpoint: call log_trade() directly with provided signal dict.
    Lets us verify DB writes work independently of the signal pipeline.
    Immediately closes the trade as 'test' after verifying the insert.

    Pass keep_open=true in the JSON body to leave the trade open (for
    manual-close cycle tests). The caller is responsible for cleanup.
    """
    try:
        from trade_tracker import log_trade
        import db as _tl_db
        data = request.get_json(force=True) or {}
        keep_open = bool(data.get('keep_open', False))
        sig = {
            'symbol':    data.get('symbol', 'MNQ'),
            'direction': data.get('direction', 'short'),
            'setup':     data.get('setup', 'I_mathematical_alpha'),
            'mode':      data.get('mode', 'intraday'),
            'entry':     float(data.get('entry', 0)),
            'stop':      float(data.get('stop', 0)),
            'target':    float(data.get('target', 0)),
            'rr':        float(data.get('rr', 1.5)),
            'session':   data.get('session', 'NY Primary'),
            'quality':   data.get('quality', 'test'),
        }
        trade_id = log_trade(sig)
        if not keep_open:
            # Mark it closed immediately so it doesn't pollute open trades
            conn = _tl_db.connect()
            conn.execute(
                "UPDATE apex_trades SET status='closed', exit_reason='test_endpoint', exit_time=? WHERE id=?",
                (datetime.now(timezone.utc).isoformat(), trade_id)
            )
            conn.commit()
            conn.close()
        return jsonify({'ok': True, 'trade_id': trade_id, 'signal': sig,
                        'kept_open': keep_open})
    except Exception as e:
        logger.error(f'test_log endpoint failed: {e}', exc_info=True)
        return jsonify({'ok': False, 'error': str(e), 'type': type(e).__name__}), 500


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


@app.route('/api/apex/test_telegram', methods=['GET', 'POST'])
def apex_test_telegram_get():
    """
    GET-accessible Telegram test with full diagnostics.
    Shows which env vars are set and whether the message was delivered.
    """
    import os, requests as _req
    from datetime import datetime, timezone
    token_a = bool(os.environ.get('TELEGRAM_TOKEN'))
    token_b = bool(os.environ.get('TELEGRAM_BOT_TOKEN'))
    chat    = bool(os.environ.get('TELEGRAM_CHAT_ID'))
    diag = {
        'TELEGRAM_TOKEN_set':     token_a,
        'TELEGRAM_BOT_TOKEN_set': token_b,
        'TELEGRAM_CHAT_ID_set':   chat,
        'active_var': ('TELEGRAM_TOKEN' if token_a else 'TELEGRAM_BOT_TOKEN' if token_b else 'NONE'),
    }
    try:
        from live_scanner import send_telegram, load_telegram_config
        tok, cid = load_telegram_config()
        diag['resolved_token_len'] = len(tok) if tok else 0
        diag['resolved_chat_id']   = cid[:6] + '…' if cid else ''
        if not tok or not cid:
            return jsonify({'ok': False, 'reason': 'No credentials resolved', 'diag': diag})
        now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        result = send_telegram(
            f'✅ <b>APEX Telegram Test</b>\n'
            f'Wise Meridian Capital — APEX ENGINE v1.1\n'
            f'<i>{now}</i>\n'
            f'If you see this, Telegram alerts are working.'
        )
        diag['send_result'] = result
        return jsonify({'ok': result, 'diag': diag})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e), 'diag': diag})


@app.route('/api/apex/test_setup_i', methods=['GET'])
def apex_test_setup_i():
    """
    End-to-end Setup I verification.
    Runs a mock signal through log_trade → send_telegram → execute_via_tradovate
    and returns each step's result as JSON. Does NOT wait for a real signal.
    The mock trade is immediately closed as test_endpoint so it doesn't pollute open trades.
    """
    steps = {}
    try:
        from setup_i_mathematical import scan_setup_i, format_i_alert
        from trade_tracker import log_trade
        from live_scanner import send_telegram
        from tradovate import TRADOVATE_ENABLED as _i_tv_enabled
        from datetime import datetime, timezone
        import db as _si_db

        # Try a real scan first — if market hours and model ready it may fire [I-1/6],[I-2/6]
        now_utc = datetime.now(timezone.utc)
        real_sig = None
        for _sym in ['MNQ', 'ES']:
            try:
                real_sig = scan_setup_i(_sym, now_utc)
                if real_sig:
                    steps['real_signal'] = {'symbol': _sym, 'direction': real_sig.get('direction')}
                    break
            except Exception as _rse:
                steps[f'real_scan_{_sym}_error'] = str(_rse)

        # Fall back to mock signal if no real one
        sig = real_sig or {
            'symbol':    'MNQ',
            'direction': 'long',
            'setup':     'I_mathematical_alpha',
            'mode':      'intraday',
            'entry':     20000.0,
            'stop':      19950.0,
            'target':    20150.0,
            'rr':        3.0,
            'session':   'NY Primary',
            'quality':   'test',
            'xgb_prob':  0.72,
            'lr_prob':   0.68,
            'hurst':     0.62,
        }
        steps['signal_source'] = 'real' if real_sig else 'mock'
        steps['signal'] = {k: v for k, v in sig.items() if k not in ('raw_data',)}

        # [I-3/6] log_trade
        _i_tid = None
        try:
            _i_tid = log_trade(sig)
            steps['I_3_log_trade'] = {'ok': bool(_i_tid), 'trade_id': _i_tid}
        except Exception as _lte:
            steps['I_3_log_trade'] = {'ok': False, 'error': str(_lte)}

        # [I-4/6] confirm DB row
        if _i_tid:
            try:
                conn = _si_db.connect()
                row = conn.execute('SELECT id, symbol, direction, setup, entry_price, status FROM apex_trades WHERE id=?', (_i_tid,)).fetchone()
                conn.close()
                steps['I_4_db_row'] = dict(zip(['id','symbol','direction','setup','entry_price','status'], row)) if row else {'error': 'row not found'}
            except Exception as _dbe:
                steps['I_4_db_row'] = {'error': str(_dbe)}

        # [I-5/6] send_telegram
        try:
            msg = format_i_alert(sig) if callable(format_i_alert) else f'🧪 Setup I test — {sig["symbol"]} {sig["direction"]}'
            _tg_ok = send_telegram(msg + '\n<i>⚠️ TEST — not a real signal</i>')
            steps['I_5_telegram'] = {'ok': _tg_ok}
        except Exception as _te:
            steps['I_5_telegram'] = {'ok': False, 'error': str(_te)}

        # [I-6/6] execute_via_tradovate
        steps['I_6_tradovate_enabled'] = _i_tv_enabled
        if _i_tid and _i_tv_enabled:
            try:
                _execute_via_tradovate(sig, _i_tid)
                steps['I_6_execute'] = {'ok': True}
            except Exception as _exe:
                steps['I_6_execute'] = {'ok': False, 'error': str(_exe)}
        else:
            steps['I_6_execute'] = {'skipped': True, 'reason': 'tradovate_disabled or no trade_id'}

        # Clean up: close the test trade immediately
        if _i_tid:
            try:
                conn = _si_db.connect()
                conn.execute(
                    "UPDATE apex_trades SET status='closed', exit_reason='test_endpoint', exit_time=? WHERE id=?",
                    (datetime.now(timezone.utc).isoformat(), _i_tid)
                )
                conn.commit()
                conn.close()
                steps['cleanup'] = 'trade closed as test_endpoint'
            except Exception as _ce:
                steps['cleanup'] = f'cleanup failed: {_ce}'

        all_ok = (
            steps.get('I_3_log_trade', {}).get('ok') and
            steps.get('I_5_telegram', {}).get('ok')
        )
        return jsonify({'ok': all_ok, 'steps': steps})

    except Exception as e:
        return jsonify({'ok': False, 'error': str(e), 'steps': steps})


# ─────────────────────────────────────────────────────────────
#  APEX SESSION ALERTS
# ─────────────────────────────────────────────────────────────

APEX_SESSIONS = [
    {'name': 'London',     'syms': 'NQ/ES', 'start': 7,  'end': 11},
    {'name': 'NY Primary', 'syms': 'NQ/ES', 'start': 13, 'end': 19},
    {'name': 'GC Primary', 'syms': 'GC',    'start': 12, 'end': 17},
]
_session_state = {}  # in-memory cache; DB is authoritative across redeploys

def _sess_fired(key: str) -> bool:
    """Return True if this session alert already fired (checks DB to survive redeploys)."""
    if _session_state.get(key):
        return True
    try:
        conn = _db.connect()
        cur = conn.execute('SELECT value FROM paper_account WHERE key=?', ('sess_' + key,))
        row = cur.fetchone()
        conn.close()
        if row:
            _session_state[key] = True
            return True
    except Exception:
        pass
    return False

def _sess_mark(key: str):
    """Mark a session alert as fired in memory and DB."""
    _session_state[key] = True
    try:
        conn = _db.connect()
        _db.kv_upsert(conn, 'sess_' + key, '1')
        conn.commit()
        conn.close()
    except Exception:
        pass

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
            if hour == sess['start'] and not _sess_fired(key_open):
                _sess_mark(key_open)
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
            if hour == sess['end'] and not _sess_fired(key_close):
                _sess_mark(key_close)
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
