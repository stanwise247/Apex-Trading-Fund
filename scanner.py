"""
APEX Real-Time Scanner — scanner.py
=====================================
Monitors live market conditions and scores them against
the validated pattern database from patterns.py.

Runs continuously, checks every 60 seconds, and flags
setups when current conditions match a proven edge.

Run with: python3 scanner.py --symbol NQ
Or import and call scan_now(symbol) from server.py
"""

import sqlite3
import json
import logging
import time
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('APEX.scanner')

DB_PATH = 'apex_market.db'


# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────

def load_recent_bars(symbol, timeframe, limit=250):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        'SELECT ts,open,high,low,close,volume FROM ohlcv '
        'WHERE symbol=? AND timeframe=? ORDER BY ts DESC LIMIT ?',
        conn, params=(symbol, timeframe, limit)
    )
    conn.close()
    if df.empty:
        return None
    df = df.iloc[::-1].reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    df.set_index('datetime', inplace=True)
    df = df.astype({'open': float, 'high': float, 'low': float,
                    'close': float, 'volume': float})
    return df


def add_indicators(df):
    df['ma9']   = df['close'].rolling(9).mean()
    df['ma20']  = df['close'].rolling(20).mean()
    df['ma50']  = df['close'].rolling(50).mean()
    df['ma200'] = df['close'].rolling(200).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()

    delta  = df['close'].diff()
    gain   = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss   = (-delta).clip(lower=0).ewm(com=13, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    ema12        = df['close'].ewm(span=12, adjust=False).mean()
    ema26        = df['close'].ewm(span=26, adjust=False).mean()
    df['macd']   = ema12 - ema26
    df['macd_sig']= df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist']= df['macd'] - df['macd_sig']

    tr = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': (df['high'] - df['close'].shift(1)).abs(),
        'lc': (df['low']  - df['close'].shift(1)).abs(),
    }).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    df['bb_mid']   = df['close'].rolling(20).mean()
    bb_std         = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * bb_std
    df['bb_lower'] = df['bb_mid'] - 2 * bb_std
    df['bb_pct']   = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    df['vol_ma20']  = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma20'].replace(0, np.nan)

    typical      = (df['high'] + df['low'] + df['close']) / 3
    df['vwap']   = (typical * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()

    df['prev_high']  = df['high'].shift(1)
    df['prev_low']   = df['low'].shift(1)
    df['prev_close'] = df['close'].shift(1)
    df['gap_pct']    = (df['open'] - df['prev_close']) / df['prev_close'] * 100
    df['body_pct']   = (df['close'] - df['open']).abs() / df['close'] * 100
    df['is_bull']    = (df['close'] > df['open']).astype(int)
    df['is_bear']    = (df['close'] < df['open']).astype(int)
    df['atr_pct']    = df['atr'] / df['close'] * 100

    return df


# ─────────────────────────────────────────────
#  CONDITION CHECKERS
# ─────────────────────────────────────────────

def check_conditions(df):
    """
    Read the last bar and return a dict of all current conditions.
    This is what the scanner compares against each pattern's requirements.
    """
    if df is None or len(df) < 50:
        return {}

    last  = df.iloc[-1]
    prev  = df.iloc[-2]
    close = float(last['close'])

    def safe(val):
        try:
            v = float(val)
            return None if np.isnan(v) else v
        except Exception:
            return None

    ma20  = safe(last['ma20'])
    ma50  = safe(last['ma50'])
    ma200 = safe(last['ma200'])
    rsi   = safe(last['rsi'])
    macd  = safe(last['macd'])
    macd_sig = safe(last['macd_sig'])
    macd_hist= safe(last['macd_hist'])
    atr   = safe(last['atr'])
    bb_pct= safe(last['bb_pct'])
    vol_r = safe(last['vol_ratio'])
    vwap  = safe(last['vwap'])
    gap   = safe(last['gap_pct'])

    prev_rsi  = safe(prev['rsi'])
    prev_macd = safe(prev['macd'])
    prev_macd_sig = safe(prev['macd_sig'])

    conditions = {
        'price':         round(close, 2),
        'atr':           round(atr, 2) if atr else None,
        'atr_pct':       round(float(last['atr_pct']), 3) if safe(last['atr_pct']) else None,
        'rsi':           round(rsi, 1) if rsi else None,
        'macd':          round(macd, 4) if macd else None,
        'macd_hist':     round(macd_hist, 4) if macd_hist else None,
        'bb_pct':        round(bb_pct, 3) if bb_pct else None,
        'vol_ratio':     round(vol_r, 2) if vol_r else None,
        'gap_pct':       round(gap, 3) if gap else None,
        'above_ma20':    close > ma20 if ma20 else None,
        'above_ma50':    close > ma50 if ma50 else None,
        'above_ma200':   close > ma200 if ma200 else None,
        'above_vwap':    close > vwap if vwap else None,
        'ma20_above_ma50':  ma20 > ma50 if (ma20 and ma50) else None,
        'ma50_above_ma200': ma50 > ma200 if (ma50 and ma200) else None,
        'golden_cross':  (ma20 > ma50 and safe(prev['ma20']) and safe(prev['ma50']) and
                          float(prev['ma20']) <= float(prev['ma50'])) if (ma20 and ma50) else False,
        'death_cross':   (ma20 < ma50 and safe(prev['ma20']) and safe(prev['ma50']) and
                          float(prev['ma20']) >= float(prev['ma50'])) if (ma20 and ma50) else False,
        'rsi_oversold':  rsi < 30 if rsi else False,
        'rsi_overbought':rsi > 70 if rsi else False,
        'rsi_mid_bull':  (38 <= rsi <= 52) if rsi else False,
        'rsi_mid_bear':  (50 <= rsi <= 62) if rsi else False,
        'rsi_rising':    (rsi > prev_rsi) if (rsi and prev_rsi) else None,
        'rsi_falling':   (rsi < prev_rsi) if (rsi and prev_rsi) else None,
        'macd_bull_cross': (macd and macd_sig and macd > macd_sig and
                            prev_macd and prev_macd_sig and prev_macd <= prev_macd_sig),
        'macd_bear_cross': (macd and macd_sig and macd < macd_sig and
                            prev_macd and prev_macd_sig and prev_macd >= prev_macd_sig),
        'bb_lower_touch': bb_pct < 0.1 if bb_pct is not None else False,
        'bb_upper_touch': bb_pct > 0.9 if bb_pct is not None else False,
        'high_volume':    vol_r > 1.5 if vol_r else False,
        'low_volume':     vol_r < 0.6 if vol_r else False,
        'bullish_bar':    bool(last['is_bull']),
        'bearish_bar':    bool(last['is_bear']),
        'above_prev_high': close > float(last['prev_high']) if safe(last['prev_high']) else False,
        'below_prev_low':  close < float(last['prev_low']) if safe(last['prev_low']) else False,
        'gap_up':         gap > 0.3 if gap else False,
        'gap_down':       gap < -0.3 if gap else False,
    }

    # Overall trend score (0-100)
    trend_points = 0
    if conditions.get('above_ma20'):    trend_points += 20
    if conditions.get('above_ma50'):    trend_points += 20
    if conditions.get('above_ma200'):   trend_points += 20
    if conditions.get('ma20_above_ma50'): trend_points += 20
    if conditions.get('ma50_above_ma200'):trend_points += 20
    conditions['trend_score'] = trend_points
    conditions['trend_direction'] = 'bullish' if trend_points >= 60 else 'bearish' if trend_points <= 40 else 'neutral'

    # FVG detection — look for open FVGs in the last 50 bars
    fvgs = detect_open_fvgs(df)
    conditions['open_fvgs'] = fvgs

    return conditions


def detect_open_fvgs(df, lookback=50, min_gap_pct=0.15):
    """Find FVGs formed in the last N bars that haven't been filled yet"""
    if len(df) < 5:
        return []

    recent  = df.tail(lookback)
    highs   = recent['high'].values
    lows    = recent['low'].values
    closes  = recent['close'].values
    current = closes[-1]
    fvgs    = []

    for i in range(1, len(recent) - 1):
        # Bullish FVG — gap up: bar i+1 low is above bar i-1 high
        if lows[i+1] > highs[i-1]:
            top     = lows[i+1]
            bottom  = highs[i-1]
            gap_pct = (top - bottom) / closes[i] * 100
            if gap_pct >= min_gap_pct:
                # Filled when price drops back to the bottom of the gap
                min_since = min(lows[i+1:])
                filled = min_since <= bottom
                if not filled:
                    dist_pct = (bottom - current) / current * 100
                    fvgs.append({
                        'type':      'bullish',
                        'top':       round(top, 2),
                        'bottom':    round(bottom, 2),
                        'midpoint':  round((top+bottom)/2, 2),
                        'gap_pct':   round(gap_pct, 3),
                        'dist_pct':  round(dist_pct, 3),
                        'bars_old':  len(recent) - i,
                        'actionable': current > bottom * 0.998 and current < top * 1.002
                    })
        # Bearish FVG — gap down: bar i+1 high is below bar i-1 low
        if highs[i+1] < lows[i-1]:
            top     = lows[i-1]
            bottom  = highs[i+1]
            gap_pct = (top - bottom) / closes[i] * 100
            if gap_pct >= min_gap_pct:
                # Filled when price rallies back to the top of the gap
                max_since = max(highs[i+1:])
                filled = max_since >= top
                if not filled:
                    dist_pct = (top - current) / current * 100
                    fvgs.append({
                        'type':      'bearish',
                        'top':       round(top, 2),
                        'bottom':    round(bottom, 2),
                        'midpoint':  round((top+bottom)/2, 2),
                        'gap_pct':   round(gap_pct, 3),
                        'dist_pct':  round(dist_pct, 3),
                        'bars_old':  len(recent) - i,
                        'actionable': current > bottom * 0.998 and current < top * 1.002
                    })

    # Sort by proximity to current price
    fvgs.sort(key=lambda x: abs(x['dist_pct']))
    return fvgs[:5]


# ─────────────────────────────────────────────
#  PATTERN MATCHING
# ─────────────────────────────────────────────

def load_validated_patterns(symbol):
    """Load patterns with positive expectancy from the database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        'SELECT * FROM patterns WHERE symbol=? AND active=1 AND expectancy > 0 ORDER BY edge_score DESC',
        (symbol,)
    )
    cols = [d[0] for d in c.description]
    rows = [dict(zip(cols, r)) for r in c.fetchall()]
    conn.close()
    return rows


PATTERN_CONDITIONS = {
    'MA_Uptrend_RSI_Pullback_Long': {
        'direction': 'long',
        'required': ['above_ma50', 'ma50_above_ma200', 'rsi_mid_bull', 'rsi_falling'],
        'description': 'Uptrend pullback to RSI 40-50 — buy the dip in a bull trend',
        'priority': 'A',
    },
    'MA_Downtrend_RSI_Bounce_Short': {
        'direction': 'short',
        'required': ['ma50_above_ma200'],
        'check': lambda c: not c.get('above_ma50') and not c.get('ma50_above_ma200') and c.get('rsi_mid_bear'),
        'description': 'Downtrend bounce to RSI 50-62 — sell the rip in a bear trend',
        'priority': 'A',
    },
    'MA200_Support_Touch_Long': {
        'direction': 'long',
        'required': ['above_ma200', 'bullish_bar'],
        'description': 'Price touched MA200 from above and bounced — major support hold',
        'priority': 'A',
    },
    'RSI_Oversold_Recovery_Uptrend': {
        'direction': 'long',
        'required': ['above_ma50', 'rsi_rising'],
        'check': lambda c: c.get('rsi') and c.get('rsi') > 30,
        'description': 'RSI recovering from oversold in uptrend context',
        'priority': 'B',
    },
    'Bearish_RSI_Divergence': {
        'direction': 'short',
        'required': ['rsi_overbought'],
        'description': 'RSI diverging bearishly from price — momentum fading at highs',
        'priority': 'B',
    },
    'Bullish_RSI_Divergence': {
        'direction': 'long',
        'required': ['rsi_oversold'],
        'description': 'RSI diverging bullishly from price — momentum turning at lows',
        'priority': 'B',
    },
    'MACD_Bull_Cross_Below_Zero': {
        'direction': 'long',
        'required': ['macd_bull_cross'],
        'description': 'MACD crossed bullishly below zero — early trend change signal',
        'priority': 'B',
    },
    'MACD_Bear_Cross_Above_Zero': {
        'direction': 'short',
        'required': ['macd_bear_cross'],
        'description': 'MACD crossed bearishly above zero — momentum rolling over',
        'priority': 'B',
    },
    'High_Volume_Breakout_Long': {
        'direction': 'long',
        'required': ['above_prev_high', 'high_volume', 'bullish_bar'],
        'description': 'High volume breakout above previous high — institutional participation',
        'priority': 'A',
    },
    'High_Volume_Bullish_Reversal': {
        'direction': 'long',
        'required': ['high_volume', 'bullish_bar'],
        'check': lambda c: c.get('rsi') and c.get('rsi') < 40,
        'description': 'High volume bullish reversal bar at oversold levels — climactic selling',
        'priority': 'A',
    },
    'Bollinger_Lower_Band_Bounce': {
        'direction': 'long',
        'required': ['bb_lower_touch', 'bullish_bar'],
        'description': 'Price touched lower Bollinger Band and closed bullish — mean reversion setup',
        'priority': 'B',
    },
    'Bullish_Engulfing_Candle': {
        'direction': 'long',
        'required': ['bullish_bar'],
        'description': 'Bullish engulfing pattern — buyers overwhelmed sellers',
        'priority': 'B',
    },
    'Prev_Day_High_Break_Long': {
        'direction': 'long',
        'required': ['above_prev_high', 'bullish_bar'],
        'description': 'Breaking above previous day high — momentum continuation',
        'priority': 'B',
    },
    'VWAP_Reclaim_Long': {
        'direction': 'long',
        'required': ['above_vwap', 'high_volume'],
        'description': 'Price reclaimed VWAP on volume — institutional buying',
        'priority': 'B',
    },
    'Gap_Down_Recovery_Long': {
        'direction': 'long',
        'required': ['gap_down', 'bullish_bar'],
        'check': lambda c: c.get('rsi') and c.get('rsi') < 45,
        'description': 'Gap down recovering — mean reversion, gap fill likely',
        'priority': 'C',
    },
    'Full_Uptrend_Confluence_Long': {
        'direction': 'long',
        'required': ['above_ma20', 'above_ma50', 'above_ma200', 'ma20_above_ma50', 'ma50_above_ma200'],
        'check': lambda c: c.get('rsi') and 50 < c['rsi'] < 70 and c.get('macd_hist') and c['macd_hist'] > 0,
        'description': 'Full trend alignment — all MAs stacked bullish, RSI and MACD confirm',
        'priority': 'A',
    },
}


def score_setup(conditions, pattern_name, pattern_db_row):
    """
    Score a potential setup on a 0-100 scale.
    Combines: pattern conditions met + edge score from database + confluence factors
    """
    pattern_def = PATTERN_CONDITIONS.get(pattern_name)
    if not pattern_def:
        return 0, []

    score         = 0
    reasons       = []
    max_score     = 100

    # 1. Check required conditions (50 points max)
    required = pattern_def.get('required', [])
    met = 0
    for cond in required:
        if conditions.get(cond):
            met += 1
        else:
            reasons.append('MISSING: ' + cond)

    if required:
        cond_score = (met / len(required)) * 50
        score += cond_score
        if met < len(required):
            return round(score), reasons  # don't score further if conditions not met

    # 2. Custom check function (if defined)
    check_fn = pattern_def.get('check')
    if check_fn:
        try:
            if not check_fn(conditions):
                reasons.append('Custom check failed')
                score *= 0.5
        except Exception:
            pass

    # 3. Historical edge score (30 points max)
    edge = float(pattern_db_row.get('edge_score', 0)) if pattern_db_row else 0
    score += (edge / 100) * 30

    # 4. Confluence bonus (20 points)
    bonus = 0
    rsi   = conditions.get('rsi')
    vol_r = conditions.get('vol_ratio')
    trend = conditions.get('trend_score', 50)

    if pattern_def['direction'] == 'long':
        if trend >= 60:                bonus += 5
        if rsi and 40 <= rsi <= 60:    bonus += 5
        if vol_r and vol_r > 1.2:      bonus += 5
        if conditions.get('above_vwap'): bonus += 5
    else:
        if trend <= 40:                bonus += 5
        if rsi and 40 <= rsi <= 60:    bonus += 5
        if vol_r and vol_r > 1.2:      bonus += 5
        if not conditions.get('above_vwap'): bonus += 5

    score += bonus
    reasons.append(f'Trend score: {trend}/100')
    reasons.append(f'Vol ratio: {vol_r}')
    reasons.append(f'RSI: {rsi}')

    return round(min(score, 100)), reasons


def scan_for_setups(symbol, timeframe='1day'):
    """
    Main scanner function.
    Loads current market data, checks all validated patterns,
    returns scored setups ready to display on the dashboard.
    """
    df = load_recent_bars(symbol, timeframe)
    if df is None or len(df) < 50:
        logger.warning(f'Insufficient data for {symbol} {timeframe}')
        return []

    df = add_indicators(df)
    conditions = check_conditions(df)
    if not conditions:
        return []

    validated = load_validated_patterns(symbol)
    if not validated:
        logger.warning(f'No validated patterns for {symbol}. Run patterns.py first.')
        return []

    setups = []
    for db_pattern in validated:
        name = db_pattern.get('name', '')
        if name not in PATTERN_CONDITIONS:
            continue

        score, reasons = score_setup(conditions, name, db_pattern)
        if score < 40:
            continue

        pattern_def = PATTERN_CONDITIONS[name]
        price       = conditions['price']
        atr         = conditions.get('atr') or price * 0.01

        if pattern_def['direction'] == 'long':
            entry  = price
            stop   = round(price - atr * 1.5, 2)
            tgt1   = round(price + atr * 2.0, 2)
            tgt2   = round(price + atr * 3.5, 2)
        else:
            entry  = price
            stop   = round(price + atr * 1.5, 2)
            tgt1   = round(price - atr * 2.0, 2)
            tgt2   = round(price - atr * 3.5, 2)

        risk   = abs(entry - stop)
        reward = abs(tgt1 - entry)
        rr     = round(reward / risk, 2) if risk > 0 else 0

        # Priority badge
        priority = pattern_def.get('priority', 'C')
        if score >= 75:   priority = 'A'
        elif score >= 55: priority = 'B'
        else:             priority = 'C'

        setups.append({
            'pattern':     name,
            'symbol':      symbol,
            'timeframe':   timeframe,
            'direction':   pattern_def['direction'],
            'score':       score,
            'priority':    priority,
            'description': pattern_def['description'],
            'edge_score':  db_pattern.get('edge_score', 0),
            'win_rate':    db_pattern.get('wins', 0) / max(db_pattern.get('occurrences', 1), 1) * 100,
            'expectancy':  db_pattern.get('expectancy', 0),
            'occurrences': db_pattern.get('occurrences', 0),
            'best_regime': db_pattern.get('best_regime', 'all'),
            'entry':       entry,
            'stop':        stop,
            'target1':     tgt1,
            'target2':     tgt2,
            'rr':          rr,
            'atr':         round(atr, 2),
            'conditions':  {
                'rsi':         conditions.get('rsi'),
                'trend_score': conditions.get('trend_score'),
                'vol_ratio':   conditions.get('vol_ratio'),
                'above_ma200': conditions.get('above_ma200'),
                'trend_dir':   conditions.get('trend_direction'),
                'macd_hist':   conditions.get('macd_hist'),
            },
            'open_fvgs':   conditions.get('open_fvgs', []),
            'reasons':     reasons,
            'ts':          int(__import__('time').time()),
        })

        # Log to scan_log table
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.execute(
                'INSERT INTO scan_log (ts,symbol,pattern_name,score,direction,entry,stop,target1,target2,regime,notes) '
                'VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                (int(__import__('time').time()), symbol, name, score,
                 pattern_def['direction'], entry, stop, tgt1, tgt2,
                 db_pattern.get('best_regime',''), pattern_def['description'])
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug('Scan log insert failed: ' + str(e))

    setups.sort(key=lambda x: x['score'], reverse=True)
    return setups


def get_market_conditions_summary(symbol, timeframe='1day'):
    """
    Return a comprehensive summary of current market conditions
    for display on the dashboard — regardless of whether a setup is active.
    """
    df = load_recent_bars(symbol, timeframe)
    if df is None or len(df) < 50:
        return {'error': 'Insufficient data'}

    df  = add_indicators(df)
    c   = check_conditions(df)
    last = df.iloc[-1]

    return {
        'symbol':     symbol,
        'timeframe':  timeframe,
        'price':      c.get('price'),
        'rsi':        c.get('rsi'),
        'macd_hist':  c.get('macd_hist'),
        'bb_pct':     c.get('bb_pct'),
        'vol_ratio':  c.get('vol_ratio'),
        'atr':        c.get('atr'),
        'atr_pct':    c.get('atr_pct'),
        'trend_score':      c.get('trend_score'),
        'trend_direction':  c.get('trend_direction'),
        'above_ma20':  c.get('above_ma20'),
        'above_ma50':  c.get('above_ma50'),
        'above_ma200': c.get('above_ma200'),
        'above_vwap':  c.get('above_vwap'),
        'golden_cross':c.get('golden_cross'),
        'death_cross': c.get('death_cross'),
        'open_fvgs':   c.get('open_fvgs', []),
        'gap_pct':     c.get('gap_pct'),
        'high_volume': c.get('high_volume'),
        'ts':          int(__import__('time').time()),
    }


# ─────────────────────────────────────────────
#  CONTINUOUS SCANNER
# ─────────────────────────────────────────────

def run_continuous_scanner(symbol='NQ', interval_seconds=60):
    logger.info(f'Starting continuous scanner for {symbol} — checking every {interval_seconds}s')
    while True:
        try:
            for tf in ['1day', '1hour']:
                setups = scan_for_setups(symbol, tf)
                if setups:
                    logger.info(f'\n  *** {len(setups)} SETUPS DETECTED — {symbol} {tf} ***')
                    for s in setups:
                        logger.info(
                            f'  [{s["priority"]}] {s["pattern"]} | '
                            f'Score: {s["score"]}/100 | '
                            f'{s["direction"].upper()} | '
                            f'Entry: {s["entry"]} | '
                            f'Stop: {s["stop"]} | '
                            f'T1: {s["target1"]} | '
                            f'R:R {s["rr"]}'
                        )
                else:
                    logger.info(f'  No setups detected — {symbol} {tf}')
        except Exception as e:
            logger.error(f'Scanner error: {e}')
        time.sleep(interval_seconds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='APEX Real-Time Scanner')
    parser.add_argument('--symbol',   default='NQ', help='Symbol to scan (default: NQ)')
    parser.add_argument('--interval', type=int, default=60, help='Scan interval in seconds')
    parser.add_argument('--once',     action='store_true', help='Run once and exit')
    args = parser.parse_args()

    if args.once:
        for tf in ['1day', '1hour']:
            setups = scan_for_setups(args.symbol, tf)
            print(f'\n{args.symbol} {tf}: {len(setups)} setups')
            for s in setups:
                print(f'  [{s["priority"]}] {s["pattern"]} | Score: {s["score"]} | {s["direction"].upper()}')
        summary = get_market_conditions_summary(args.symbol)
        print(f'\nConditions: RSI={summary.get("rsi")} | Trend={summary.get("trend_direction")} | Vol={summary.get("vol_ratio")}')
        print(f'FVGs: {len(summary.get("open_fvgs", []))} open')
    else:
        run_continuous_scanner(args.symbol, args.interval)
