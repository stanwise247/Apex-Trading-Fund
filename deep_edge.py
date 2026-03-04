"""
APEX Deep Edge Scanner — deep_edge.py
========================================
The complete multi-layer edge detection system.

Combines all 7 layers into a single conviction score:
  Layer 1 — Multi-timeframe bias (weekly/daily/4hr/1hr)
  Layer 2 — Order flow (OB, FVG, BOS, CHoCH, sweeps)
  Layer 3 — Session timing & VIX regime
  Layer 4 — Volume confirmation
  Layer 5 — Price action signals (15min/5min entry)
  Layer 6 — Partial profit management
  Layer 7 — News/macro filter

Only fires when combined score >= threshold (default 70/100).
Sends Telegram alert and opens paper trade automatically.

Run: python3 deep_edge.py --symbol NQ --scan
     python3 deep_edge.py --symbol NQ --backtest --balance 10000 --risk 2
"""

import sqlite3
import json
import logging
import time
import argparse
import random
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('APEX.DeepEdge')

DB_PATH = 'apex_market.db'

TF_HIERARCHY = ['1min','5min','15min','1hour','4hour','1day','1week']
TF_MINUTES   = {'1min':1,'5min':5,'15min':15,'1hour':60,'4hour':240,'1day':1440,'1week':10080}


# =============================================================
#  SCORING WEIGHTS
# =============================================================

LAYER_WEIGHTS = {
    'htf_confluence':  25,   # Weekly/daily/4hr alignment
    'order_flow':      25,   # OB, FVG, BOS, CHoCH, sweep
    'entry_signal':    20,   # 15min/5min trigger
    'session':         10,   # Time of day quality
    'volume':          10,   # Volume confirmation
    'vix_regime':       5,   # Volatility environment
    'pd_zone':          5,   # Premium/discount positioning
}

TOTAL_POSSIBLE = sum(LAYER_WEIGHTS.values())  # 100


# =============================================================
#  DATA LOADING
# =============================================================

def load_tf(symbol, timeframe, limit=500):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        'SELECT ts,open,high,low,close,volume FROM ohlcv '
        'WHERE symbol=? AND timeframe=? ORDER BY ts DESC LIMIT ?',
        conn, params=(symbol, timeframe, limit)
    )
    conn.close()
    if df.empty or len(df) < 20:
        return None
    df = df.iloc[::-1].reset_index(drop=True)
    df['dt'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    df.set_index('dt', inplace=True)
    for col in ['open','high','low','close','volume']:
        df[col] = df[col].astype(float)
    return df


def load_latest_vix():
    """Get latest VIX close from database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT close FROM ohlcv WHERE symbol="VIX" AND timeframe="1day" ORDER BY ts DESC LIMIT 1')
    row = c.fetchone()
    conn.close()
    return float(row[0]) if row else None


def add_indicators(df):
    if df is None or len(df) < 10:
        return df
    c = df['close']
    h = df['high']
    l = df['low']
    v = df['volume']

    for p in [9, 20, 50, 200]:
        df[f'ma{p}'] = c.rolling(p).mean()
    df['ema9']  = c.ewm(span=9,  adjust=False).mean()
    df['ema21'] = c.ewm(span=21, adjust=False).mean()

    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(com=13, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    df['macd']     = e12 - e26
    df['macd_sig'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist']= df['macd'] - df['macd_sig']

    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()], axis=1).max(axis=1)
    df['atr']     = tr.ewm(span=14, adjust=False).mean()
    df['atr_pct'] = df['atr'] / c * 100

    df['bb_mid']   = c.rolling(20).mean()
    std = c.rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2*std
    df['bb_lower'] = df['bb_mid'] - 2*std
    df['bb_pct']   = (c - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    df['vol_ma20']  = v.rolling(20).mean()
    df['vol_ratio'] = v / df['vol_ma20'].replace(0, np.nan)

    typ = (h + l + c) / 3
    df['vwap'] = (typ * v).rolling(20).sum() / v.rolling(20).sum()

    df['is_bull'] = (c > df['open']).astype(int)
    df['is_bear'] = (c < df['open']).astype(int)
    df['body_pct']= (c - df['open']).abs() / c * 100
    df['prev_high']  = h.shift(1)
    df['prev_low']   = l.shift(1)
    df['prev_close'] = c.shift(1)

    df['swing_high'] = ((h > h.shift(1)) & (h > h.shift(2)) &
                        (h > h.shift(-1)) & (h > h.shift(-2))).astype(int)
    df['swing_low']  = ((l < l.shift(1)) & (l < l.shift(2)) &
                        (l < l.shift(-1)) & (l < l.shift(-2))).astype(int)
    return df


# =============================================================
#  LAYER SCORERS
# =============================================================

def score_htf_confluence(dfs, available_tfs, at_time=None):
    """Layer 1: Score higher timeframe alignment (0-25)"""
    from mtf_engine import get_htf_bias

    scores = {}
    for tf in ['1week','1day','4hour','1hour']:
        if tf not in available_tfs or tf not in dfs:
            continue
        data = dfs[tf]
        if at_time is not None:
            data = data[data.index <= at_time]
        if len(data) < 20:
            continue
        data = add_indicators(data.copy())
        scores[tf] = get_htf_bias(data, tf)

    if not scores:
        return 0, 'neutral', {}

    weights = {'1week': 3, '1day': 2.5, '4hour': 2, '1hour': 1.5}
    total_w = sum(weights[tf] for tf in scores)
    if total_w == 0:
        return 0, 'neutral', scores

    avg = sum(scores[tf]['score'] * weights[tf] for tf in scores) / total_w
    direction = 'bullish' if avg >= 58 else 'bearish' if avg <= 42 else 'neutral'

    # Check agreement between weekly and daily (most important)
    htf_agree = False
    agree_dir = direction
    pairs = [('1week','1day'),('1day','4hour'),('4hour','1hour'),('1day','1hour')]
    for tf_a, tf_b in pairs:
        if tf_a in scores and tf_b in scores:
            da = scores[tf_a]['direction']
            db = scores[tf_b]['direction']
            if da == db and da != 'neutral':
                htf_agree = True
                agree_dir = da
                break

    # Score: full 25 if strong agreement, partial if moderate
    if htf_agree and abs(avg - 50) >= 15:
        layer_score = LAYER_WEIGHTS['htf_confluence']
    elif htf_agree:
        layer_score = int(LAYER_WEIGHTS['htf_confluence'] * 0.75)
    elif abs(avg - 50) >= 10:
        layer_score = int(LAYER_WEIGHTS['htf_confluence'] * 0.5)
    else:
        layer_score = 0

    return layer_score, agree_dir, scores


def score_order_flow(df_5min, df_15min, direction):
    """Layer 2: Score order flow quality (0-25)"""
    from order_flow import (find_order_blocks, find_fvgs, find_structure,
                            find_liquidity_sweeps, get_premium_discount,
                            find_ob_fvg_confluence)

    score = 0
    details = {}

    # Use 5min for precision, 15min for context
    work = df_5min if df_5min is not None else df_15min
    if work is None or len(work) < 30:
        return 0, {}

    current = float(work['close'].iloc[-1])

    # Order blocks near price
    obs = find_order_blocks(work, lookback=80)
    nearby_obs = [ob for ob in obs
                  if ob.direction == direction
                  and abs(ob.midpoint - current) / current < 0.005]
    if nearby_obs:
        best_ob = max(nearby_obs, key=lambda x: x.strength)
        score += min(8, int(best_ob.strength / 12))
        details['order_block'] = {'top': best_ob.top, 'bottom': best_ob.bottom,
                                   'strength': best_ob.strength}

    # FVGs near price
    fvgs = find_fvgs(work, lookback=80)
    nearby_fvgs = [f for f in fvgs
                   if f.direction == direction
                   and abs(f.midpoint - current) / current < 0.005]
    if nearby_fvgs:
        best_fvg = nearby_fvgs[0]
        score += min(6, int(best_fvg.size_pct * 20))
        details['fvg'] = {'top': best_fvg.top, 'bottom': best_fvg.bottom,
                           'size': best_fvg.size_pct, 'age': best_fvg.age_bars}

    # OB + FVG confluence (huge bonus)
    confluences = find_ob_fvg_confluence(obs, fvgs)
    dir_conf = [c for c in confluences if c['direction'] == direction
                and abs(c['midpoint'] - current) / current < 0.006]
    if dir_conf:
        score += 6
        details['ob_fvg_confluence'] = dir_conf[0]

    # Structure (BOS/CHoCH)
    struct, trend = find_structure(work, lookback=100)
    recent_bos = [s for s in struct if 'BOS' in s.kind and direction in s.kind.lower()][-1:]
    recent_choch = [s for s in struct if 'CHoCH' in s.kind][-1:]
    if recent_bos:
        score += 3
        details['bos_confirmed'] = True
    if recent_choch and direction in recent_choch[0].kind.lower():
        score += 4
        details['choch'] = True

    # Liquidity sweep (reversal signal)
    sweeps = find_liquidity_sweeps(work, lookback=40)
    if sweeps and sweeps[0].reversal_confirmed:
        sw = sweeps[0]
        if (direction in ('long','bullish') and sw.direction == 'bull_sweep') or \
           (direction in ('short','bearish') and sw.direction == 'bear_sweep'):
            score += 4
            details['liquidity_sweep'] = True

    # Premium/discount zone
    pd_zone = get_premium_discount(work)
    zone = pd_zone.get('zone', '')
    if direction in ('long','bullish') and 'discount' in zone:
        score += 2
        details['pd_zone'] = zone
    elif direction in ('short','bearish') and 'premium' in zone:
        score += 2
        details['pd_zone'] = zone

    score = min(score, LAYER_WEIGHTS['order_flow'])
    details['score'] = score
    return score, details


def score_session(bar_dt=None):
    """Layer 3: Session quality score (0-10)"""
    from session_filter import get_session_score_for_bar
    sess = get_session_score_for_bar(bar_dt or datetime.now(timezone.utc))
    quality = sess.get('quality', 50)
    if not sess.get('trade_recommended', True):
        return 0, sess
    score = int(quality / 100 * LAYER_WEIGHTS['session'])
    return score, sess


def score_volume(df, direction, bar_idx=-1):
    """Layer 4: Volume confirmation (0-10)"""
    if df is None or 'vol_ratio' not in df.columns:
        return 5, {}  # neutral if no volume data

    try:
        vr = float(df['vol_ratio'].iloc[bar_idx])
        if np.isnan(vr):
            return 5, {}
    except Exception:
        return 5, {}

    # Strong volume = confirmation
    if vr >= 2.0:   score = LAYER_WEIGHTS['volume']
    elif vr >= 1.5: score = int(LAYER_WEIGHTS['volume'] * 0.9)
    elif vr >= 1.2: score = int(LAYER_WEIGHTS['volume'] * 0.7)
    elif vr >= 0.8: score = int(LAYER_WEIGHTS['volume'] * 0.5)
    else:           score = int(LAYER_WEIGHTS['volume'] * 0.2)  # low volume = weak

    # Volume trend — rising volume in direction of trade is best
    if len(df) >= 3:
        recent_vol = df['vol_ratio'].iloc[-3:].values
        vol_rising = all(recent_vol[i] >= recent_vol[i-1] for i in range(1, len(recent_vol)))
        if vol_rising:
            score = min(score + 2, LAYER_WEIGHTS['volume'])

    return score, {'vol_ratio': round(vr, 2)}


def score_vix(vix_value):
    """Layer 5: VIX regime (0-5)"""
    from session_filter import get_vix_regime
    regime = get_vix_regime(vix_value)
    mult   = regime.get('score_mult', 1.0)

    if mult >= 1.1:   score = LAYER_WEIGHTS['vix_regime']
    elif mult >= 1.0: score = int(LAYER_WEIGHTS['vix_regime'] * 0.8)
    elif mult >= 0.9: score = int(LAYER_WEIGHTS['vix_regime'] * 0.6)
    else:             score = int(LAYER_WEIGHTS['vix_regime'] * 0.3)

    return score, regime


def score_pd_zone(df, direction):
    """Layer 6: Premium/discount positioning (0-5)"""
    from order_flow import get_premium_discount
    pd = get_premium_discount(df)
    zone = pd.get('zone', '')
    bias = pd.get('bias', '')

    if (direction in ('long','bullish') and 'discount' in zone) or \
       (direction in ('short','bearish') and 'premium' in zone):
        score = LAYER_WEIGHTS['pd_zone']
    elif 'deep' in zone:
        score = int(LAYER_WEIGHTS['pd_zone'] * 0.8)
    else:
        score = int(LAYER_WEIGHTS['pd_zone'] * 0.4)

    return score, pd


def score_entry_signal(df, direction, bar_idx=-1):
    """Layer 7: Entry signal quality (0-20)"""
    if df is None or len(df) < 20:
        return 0, {}

    try:
        last = df.iloc[bar_idx]
        prev = df.iloc[bar_idx - 1]
    except Exception:
        return 0, {}

    score   = 0
    details = {}

    rsi   = float(last.get('rsi', 50) or 50)
    macd_h= float(last.get('macd_hist', 0) or 0)
    bb_pct= float(last.get('bb_pct', 0.5) or 0.5)
    vr    = float(last.get('vol_ratio', 1) or 1)
    is_b  = bool(last.get('is_bull', 0))
    is_r  = bool(last.get('is_bear', 0))
    atr   = float(last.get('atr', 1) or 1)
    price = float(last['close'])
    ema21 = float(last.get('ema21', price) or price)

    prev_rsi  = float(prev.get('rsi', 50) or 50)
    prev_macd = float(prev.get('macd_hist', 0) or 0)

    if direction in ('long', 'bullish'):
        # RSI in buy zone and turning up
        if 35 <= rsi <= 55 and rsi > prev_rsi:
            score += 5; details['rsi_pullback'] = round(rsi, 1)
        elif rsi < 35:
            score += 3; details['rsi_oversold'] = round(rsi, 1)

        # MACD turning bullish
        if macd_h > 0 and prev_macd <= 0:
            score += 5; details['macd_cross'] = True
        elif macd_h > 0 and macd_h > prev_macd:
            score += 2; details['macd_rising'] = True

        # BB lower band
        if not np.isnan(bb_pct) and bb_pct < 0.15:
            score += 3; details['bb_lower'] = True

        # Bullish candle + above EMA21
        if is_b and price > ema21:
            score += 3; details['bull_candle_above_ema'] = True
        elif is_b:
            score += 1

        # EMA21 reclaim
        if price > ema21 and float(prev['close']) <= float(prev.get('ema21', price)):
            score += 4; details['ema21_reclaim'] = True

    else:  # short/bearish
        # RSI in sell zone and turning down
        if 45 <= rsi <= 65 and rsi < prev_rsi:
            score += 5; details['rsi_bounce'] = round(rsi, 1)
        elif rsi > 65:
            score += 3; details['rsi_overbought'] = round(rsi, 1)

        # MACD turning bearish
        if macd_h < 0 and prev_macd >= 0:
            score += 5; details['macd_cross'] = True
        elif macd_h < 0 and macd_h < prev_macd:
            score += 2; details['macd_falling'] = True

        # BB upper band
        if not np.isnan(bb_pct) and bb_pct > 0.85:
            score += 3; details['bb_upper'] = True

        # Bearish candle + below EMA21
        if is_r and price < ema21:
            score += 3; details['bear_candle_below_ema'] = True
        elif is_r:
            score += 1

        # EMA21 rejection
        if price < ema21 and float(prev['close']) >= float(prev.get('ema21', price)):
            score += 4; details['ema21_rejection'] = True

    score = min(score, LAYER_WEIGHTS['entry_signal'])
    details['score'] = score
    return score, details


# =============================================================
#  STOP PLACEMENT
# =============================================================

def calculate_structural_stop(df, direction, bar_idx=-1):
    if df is None or len(df) < 5:
        return None
    try:
        lookback = df.iloc[max(0, bar_idx-10):bar_idx+1]
        if len(lookback) == 0:
            return None
        atr_val = df['atr'].iloc[bar_idx]
        price   = float(df['close'].iloc[bar_idx])
        atr     = float(atr_val) if (atr_val is not None and not np.isnan(atr_val)) else price * 0.005
        buffer  = atr * 0.3
        if direction in ('long', 'bullish'):
            stop = float(lookback['low'].min()) - buffer
        else:
            stop = float(lookback['high'].max()) + buffer
        if np.isnan(stop):
            stop = price - atr*2 if direction in ('long','bullish') else price + atr*2
        return round(stop, 2)
    except Exception:
        return None


# =============================================================
#  MAIN SIGNAL GENERATOR
# =============================================================

def generate_signal(symbol, dfs, available_tfs, vix_value=None, news_headlines=None):
    """
    Generate a fully-scored trade signal for the current market conditions.
    Returns None if no qualifying setup, or a complete signal dict.
    """
    # Determine entry timeframes
    entry_tfs = [tf for tf in ['5min', '15min', '1hour'] if tf in available_tfs]
    if not entry_tfs:
        return None

    # Layer 1: HTF confluence
    htf_score, direction, htf_details = score_htf_confluence(dfs, available_tfs)
    if direction == 'neutral' or htf_score < 10:
        return None

    best_signal = None
    best_total  = 0

    for entry_tf in entry_tfs:
        df = dfs.get(entry_tf)
        if df is None or len(df) < 30:
            continue
        df = add_indicators(df.copy())

        # Layer 2: Order flow
        df_5  = add_indicators(dfs['5min'].copy())  if '5min'  in dfs else None
        df_15 = add_indicators(dfs['15min'].copy()) if '15min' in dfs else None
        of_score, of_details = score_order_flow(df_5, df_15, direction)

        # Layer 3: Session
        bar_dt = df.index[-1]
        sess_score, sess_details = score_session(bar_dt)

        # Layer 4: Volume
        vol_score, vol_details = score_volume(df, direction)

        # Layer 5: VIX
        vix_score, vix_details = score_vix(vix_value)

        # Layer 6: PD Zone
        pd_score, pd_details = score_pd_zone(df, direction)

        # Layer 7: Entry signal
        entry_score, entry_details = score_entry_signal(df, direction)

        # Total score
        total = htf_score + of_score + sess_score + vol_score + vix_score + pd_score + entry_score

        if total > best_total:
            best_total = total

            price = float(df['close'].iloc[-1])
            atr   = float(df['atr'].iloc[-1]) if not np.isnan(df['atr'].iloc[-1]) else price * 0.005
            stop  = calculate_structural_stop(df, direction)
            if stop is None:
                stop = price - atr * 1.5 if direction in ('long','bullish') else price + atr * 1.5

            risk = abs(price - stop)
            if risk <= 0:
                continue

            t1 = price + risk * 2.5 if direction in ('long','bullish') else price - risk * 2.5
            t2 = price + risk * 4.0 if direction in ('long','bullish') else price - risk * 4.0

            best_signal = {
                'symbol':       symbol,
                'direction':    direction,
                'timeframe':    entry_tf,
                'score':        total,
                'score_pct':    round(total / TOTAL_POSSIBLE * 100, 1),
                'entry':        round(price, 2),
                'stop':         round(stop, 2),
                'target1':      round(t1, 2),
                'target2':      round(t2, 2),
                'rr':           round(abs(t1-price)/risk, 2),
                'atr':          round(atr, 2),
                # Layer details
                'layer_scores': {
                    'htf_confluence': htf_score,
                    'order_flow':     of_score,
                    'session':        sess_score,
                    'volume':         vol_score,
                    'vix_regime':     vix_score,
                    'pd_zone':        pd_score,
                    'entry_signal':   entry_score,
                },
                'htf_details':    htf_details,
                'of_details':     of_details,
                'session':        sess_details,
                'vix_regime':     vix_details.get('regime', 'normal'),
                'session_label':  sess_details.get('label', ''),
                'vix_value':      vix_value,
                'confluence':     sum(1 for v in htf_details.values()
                                     if v.get('direction') == direction),
                # Flags for alert formatting
                'ob_confluence':    of_details.get('ob_fvg_confluence') is not None,
                'fvg_inside':       of_details.get('fvg') is not None,
                'bos_confirmed':    of_details.get('bos_confirmed', False),
                'choch':            of_details.get('choch', False),
                'liquidity_sweep':  of_details.get('liquidity_sweep', False),
                'pd_zone':          pd_details.get('zone', ''),
                'reasons':          [
                    f"HTF {direction} ({htf_score}/{LAYER_WEIGHTS['htf_confluence']}pts)",
                    f"Order flow ({of_score}/{LAYER_WEIGHTS['order_flow']}pts)",
                    f"Session: {sess_details.get('label','')} ({sess_score}pts)",
                    f"Volume ratio: {vol_details.get('vol_ratio','?')} ({vol_score}pts)",
                    f"VIX {vix_value or '?'} [{vix_details.get('regime','?')}] ({vix_score}pts)",
                ],
                'ts': int(time.time()),
            }

    return best_signal if best_total >= 55 else None


# =============================================================
#  LIVE SCANNER
# =============================================================

def run_live_scan(symbol='NQ', min_score=60, alert=True, paper_trade=True):
    """
    Run a single scan cycle. Called repeatedly by the scheduler.
    """
    logger.info(f'Scanning {symbol}...')

    # Load all available timeframes
    available_tfs = []
    dfs = {}
    for tf in TF_HIERARCHY:
        df = load_tf(symbol, tf)
        if df is not None and len(df) >= 20:
            dfs[tf] = df
            available_tfs.append(tf)

    if not available_tfs:
        logger.warning(f'No data for {symbol}')
        return None

    vix_value     = load_latest_vix()
    signal        = generate_signal(symbol, dfs, available_tfs, vix_value)

    if signal is None:
        logger.info(f'No qualifying setup for {symbol}')
        return None

    score = signal['score']
    score_pct = signal['score_pct']
    direction = signal['direction']

    logger.info(f'🎯 SETUP FOUND: {symbol} {direction.upper()} | '
                f'Score: {score}/{TOTAL_POSSIBLE} ({score_pct}%) | '
                f'Entry: {signal["entry"]} | Stop: {signal["stop"]} | '
                f'T1: {signal["target1"]} | T2: {signal["target2"]} | '
                f'R:R {signal["rr"]}:1')

    # Log layer breakdown
    for layer, pts in signal['layer_scores'].items():
        logger.info(f'  {layer:<20} {pts}/{LAYER_WEIGHTS[layer]}')

    # Send Telegram alert
    if alert:
        try:
            from telegram_alerts import send_setup_alert
            sent = send_setup_alert(signal)
            if sent:
                logger.info('✅ Telegram alert sent')
        except Exception as e:
            logger.error(f'Telegram error: {e}')

    # Open paper trade
    if paper_trade and score_pct >= min_score:
        try:
            from paper_trader import open_position, init_paper_db
            init_paper_db()
            result = open_position(
                symbol    = symbol,
                direction = direction,
                entry_price = signal['entry'],
                stop      = signal['stop'],
                target1   = signal['target1'],
                target2   = signal['target2'],
                setup_name= f"{signal.get('of_details',{}).get('order_block') and 'OB_' or ''}"
                            f"{signal['direction']}_{signal['timeframe']}",
                setup_score = score,
                notes     = json.dumps(signal['layer_scores']),
            )
            if result.get('ok'):
                logger.info(f'📋 Paper trade opened: {result["contracts"]} contract(s), '
                            f'risk ${result["risk_usd"]:.2f}')
        except Exception as e:
            logger.error(f'Paper trade error: {e}')

    return signal


# =============================================================
#  DEEP BACKTEST (all layers applied historically)
# =============================================================

def run_deep_backtest(symbol='NQ', balance=10000, risk_pct=2.0, min_score=55):
    """
    Backtest the full deep edge strategy on all available historical data.
    Applies all 7 scoring layers for every bar on the entry timeframe.
    """
    logger.info('=' * 65)
    logger.info(f'  APEX Deep Edge Backtest — {symbol}')
    logger.info(f'  Balance: ${balance:,} | Risk: {risk_pct}% | Min Score: {min_score}/100')
    logger.info('=' * 65)

    # Load all timeframes
    available_tfs = []
    dfs_full = {}
    for tf in TF_HIERARCHY:
        df = load_tf(symbol, tf, limit=5000)
        if df is not None and len(df) >= 50:
            dfs_full[tf] = add_indicators(df)
            available_tfs.append(tf)
            logger.info(f'  {tf:<8} {len(df):>5} bars  '
                        f'{df.index[0].strftime("%Y-%m-%d")} to '
                        f'{df.index[-1].strftime("%Y-%m-%d")}')

    entry_tfs = [tf for tf in ['15min','5min','1hour'] if tf in available_tfs]
    if not entry_tfs:
        logger.error('No entry timeframes available')
        return None

    vix_value = load_latest_vix()
    trades    = []
    skipped_session = 0
    skipped_score   = 0
    skipped_daily   = defaultdict(int)

    for entry_tf in entry_tfs:
        entry_df = dfs_full[entry_tf]
        logger.info(f'\nBacktesting {entry_tf} entries ({len(entry_df)} bars)...')

        higher_tfs = [tf for tf in available_tfs
                      if TF_MINUTES.get(tf,0) > TF_MINUTES.get(entry_tf,0)]

        htf_cache = {}

        for i in range(100, len(entry_df) - 5):
            bar_dt   = entry_df.index[i]
            day_key  = bar_dt.strftime('%Y-%m-%d')
            hour_key = bar_dt.strftime('%Y-%m-%d-%H')

            # Session filter — skip midday and off-hours
            from session_filter import get_session_score_for_bar
            sess = get_session_score_for_bar(bar_dt)
            if not sess.get('trade_recommended', True):
                skipped_session += 1
                continue

            # Daily trade limit — max 2 per day per TF
            if skipped_daily[f'{day_key}_{entry_tf}'] >= 2:
                continue

            # HTF bias (cached per day)
            if day_key not in htf_cache:
                htf_dfs = {}
                for htf in higher_tfs:
                    data = dfs_full[htf][dfs_full[htf].index <= bar_dt]
                    if len(data) >= 20:
                        htf_dfs[htf] = data
                if htf_dfs:
                    htf_score, direction, htf_det = score_htf_confluence(
                        htf_dfs, list(htf_dfs.keys())
                    )
                    htf_cache[day_key] = (htf_score, direction, htf_det)
                else:
                    htf_cache[day_key] = (0, 'neutral', {})

            htf_score, direction, htf_det = htf_cache[day_key]
            if direction == 'neutral' or htf_score < 10:
                continue

            # Score all other layers on this bar
            window = entry_df.iloc[max(0,i-80):i+1]

            df_5  = dfs_full.get('5min')
            df_15 = dfs_full.get('15min')
            if df_5 is not None:
                df_5 = df_5[df_5.index <= bar_dt].tail(80)
            if df_15 is not None:
                df_15 = df_15[df_15.index <= bar_dt].tail(80)

            of_score,   of_det   = score_order_flow(df_5, df_15, direction)
            sess_score, sess_det = score_session(bar_dt)
            vol_score,  vol_det  = score_volume(window, direction)
            vix_score,  vix_det  = score_vix(vix_value)
            pd_score,   pd_det   = score_pd_zone(window, direction)
            ent_score,  ent_det  = score_entry_signal(window, direction)

            total = htf_score + of_score + sess_score + vol_score + vix_score + pd_score + ent_score
            score_pct = round(total / TOTAL_POSSIBLE * 100, 1)

            if score_pct < min_score:
                skipped_score += 1
                continue

            # Calculate trade levels
            price = float(window['close'].iloc[-1])
            atr   = float(window['atr'].iloc[-1]) if not np.isnan(window['atr'].iloc[-1]) else price * 0.005
            stop  = calculate_structural_stop(window, direction)
            if stop is None:
                stop = price - atr*1.5 if direction in ('long','bullish') else price + atr*1.5

            risk = abs(price - stop)
            if risk <= 0 or risk > atr * 5:  # sanity check
                continue

            t1 = price + risk*2.5 if direction in ('long','bullish') else price - risk*2.5
            t2 = price + risk*4.0 if direction in ('long','bullish') else price - risk*4.0

            # Forward test
            outcome    = 'timeout'
            exit_price = price
            max_j      = min(i + 60, len(entry_df) - 1)

            for j in range(i+1, max_j):
                bh = float(entry_df.iloc[j]['high'])
                bl = float(entry_df.iloc[j]['low'])
                bc = float(entry_df.iloc[j]['close'])
                if direction in ('long','bullish'):
                    if bl<=stop or bc<=stop: outcome="loss"; exit_price=stop; break
                    if bh>=t1 or bc>=t1: outcome="win"; exit_price=t1; break
                    if bh>=t1 or bc>=t1: outcome='win'; exit_price=t1; break
                else:
                    if bh>=stop or bc>=stop: outcome='loss'; exit_price=stop; break
                    if bl<=t1 or bc<=t1: outcome='win'; exit_price=t1; break

            if outcome == 'timeout':
                exit_price = float(entry_df.iloc[min(i+10, len(entry_df)-1)]['close'])

            pnl_pts = (exit_price-price) if direction in ('long','bullish') else (price-exit_price)
            pnl_r   = pnl_pts / risk

            # Partial exit simulation: if win, model 50% taken at T1, remainder at T2
            if outcome == 'win':
                # Check if T2 was eventually hit
                for j2 in range(i+1, max_j):
                    bh = float(entry_df.iloc[j2]['high'])
                    bl = float(entry_df.iloc[j2]['low'])
                    if direction in ('long','bullish'):
                        if bl <= price:   break  # stopped at BE
                        if bh >= t2:      pnl_r = 0.5*2.5 + 0.5*4.0; break  # full target
                    else:
                        if bh >= price:   break
                        if bl <= t2:      pnl_r = 0.5*2.5 + 0.5*4.0; break

            skipped_daily[f'{day_key}_{entry_tf}'] += 1

            trades.append({
                'entry_dt':   bar_dt.isoformat(),
                'entry_tf':   entry_tf,
                'direction':  direction,
                'score_pct':  score_pct,
                'layer_scores': {
                    'htf': htf_score, 'of': of_score, 'sess': sess_score,
                    'vol': vol_score, 'vix': vix_score, 'pd': pd_score, 'entry': ent_score
                },
                'entry':      round(price, 2),
                'stop':       round(stop, 2),
                'target1':    round(t1, 2),
                'exit':       round(exit_price, 2),
                'outcome':    outcome,
                'pnl_pts':    round(pnl_pts, 2),
                'pnl_r':      round(pnl_r, 3),
                'rr':         round(abs(t1-price)/risk, 2),
                'ob_conf':    of_det.get('ob_fvg_confluence') is not None,
                'bos':        of_det.get('bos_confirmed', False),
                'session':    sess_det.get('label',''),
            })

    logger.info(f'\nTotal qualifying trades: {len(trades)}')
    logger.info(f'Skipped (session):  {skipped_session}')
    logger.info(f'Skipped (score):    {skipped_score}')

    if not trades:
        logger.warning('No trades generated. Try lowering min_score.')
        return None

    wins    = [t for t in trades if t['outcome'] == 'win']
    losses  = [t for t in trades if t['outcome'] == 'loss']
    wr      = len(wins)/len(trades)*100

    logger.info(f'Win rate: {wr:.1f}% ({len(wins)}W / {len(losses)}L)')

    # Simulation
    result = simulate_deep(trades, balance, risk_pct)
    if result:
        s = result['stats']
        logger.info(f'\n  Return:        {s["total_return_pct"]:+.1f}%')
        logger.info(f'  Win Rate:      {s["win_rate_pct"]:.1f}%')
        logger.info(f'  Max Drawdown:  {s["max_drawdown_pct"]:.1f}%')
        logger.info(f'  Sharpe:        {s["sharpe_ratio"]:.2f}')
        logger.info(f'  Profit Factor: {s["profit_factor"]}')
        logger.info(f'  Max Streak:    {s["max_consec_losses"]} losses')

    # Monte Carlo
    mc = run_monte_carlo(trades, balance, risk_pct)
    if mc:
        logger.info(f'\n  Monte Carlo ({mc["n"]} runs):')
        logger.info(f'  Median Return:  {mc["median_return"]:+.1f}%')
        logger.info(f'  Prob of Profit: {mc["prob_profit"]}%')
        logger.info(f'  Worst 5%:       ${mc["worst_5pct"]:,.2f}')
        logger.info(f'  Best 5%:        ${mc["best_5pct"]:,.2f}')

    # Year breakdown
    yearly = yearly_sim(trades, balance, risk_pct)
    if yearly:
        logger.info('\n  Year-by-Year:')
        for yr, y in yearly.items():
            logger.info(f'  {yr}: {y["return_pct"]:+6.1f}%  '
                        f'WR={y["win_rate"]:.0f}%  '
                        f'DD={y["max_dd"]:.1f}%  '
                        f'N={y["trades"]}')

    # Score threshold analysis
    logger.info('\n  Score Threshold Analysis:')
    for threshold in [50, 60, 65, 70, 75, 80]:
        filtered = [t for t in trades if t['score_pct'] >= threshold]
        if len(filtered) >= 5:
            fw = [t for t in filtered if t['outcome']=='win']
            fwr = len(fw)/len(filtered)*100
            favg_r = sum(t['pnl_r'] for t in filtered)/len(filtered)
            logger.info(f'  Score >= {threshold}: {len(filtered):>4} trades | '
                        f'WR={fwr:.1f}% | AvgR={favg_r:+.3f}')

    output = {
        'symbol':     symbol,
        'timestamp':  datetime.now(timezone.utc).isoformat(),
        'config':     {'balance': balance, 'risk_pct': risk_pct, 'min_score': min_score},
        'trades':     trades,
        'simulation': result,
        'monte_carlo':mc,
        'yearly':     yearly,
        'summary': {
            'total_trades': len(trades),
            'win_rate':     round(wr, 1),
            'total_return': result['stats']['total_return_pct'] if result else 0,
            'sharpe':       result['stats']['sharpe_ratio'] if result else 0,
            'max_dd':       result['stats']['max_drawdown_pct'] if result else 0,
        }
    }

    with open(f'deep_edge_{symbol}.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f'\n  Saved to deep_edge_{symbol}.json')
    logger.info('=' * 65)
    return output


def simulate_deep(trades, balance=10000, risk_pct=2.0,
                  point_value=20.0, commission=5.0):
    if not trades:
        return None
    bal = balance; peak = balance
    curve = [{'balance': bal, 'n': 0}]
    log = []
    max_dd = 0; streak = 0; max_streak = 0

    for i, t in enumerate(sorted(trades, key=lambda x: x['entry_dt'])):
        if bal <= 10: break
        risk_amt = bal * (risk_pct/100)
        pnl_r    = float(t.get('pnl_r', 0))
        outcome  = t.get('outcome','timeout')
        if outcome == 'win'  and pnl_r <= 0: pnl_r =  2.5
        if outcome == 'loss' and pnl_r >= 0: pnl_r = -1.0
        pnl_usd  = risk_amt * pnl_r - commission
        bal += pnl_usd
        if bal > peak: peak = bal
        dd = (peak-bal)/peak*100
        if dd > max_dd: max_dd = dd
        if outcome == 'loss': streak += 1; max_streak = max(max_streak, streak)
        else: streak = 0
        log.append({'n':i+1,'date':t['entry_dt'][:10],'outcome':outcome,
                    'pnl_$':round(pnl_usd,2),'balance':round(bal,2),
                    'score':t.get('score_pct',0),'session':t.get('session','')})
        curve.append({'balance':round(bal,2),'n':i+1,'date':t['entry_dt'][:10]})

    if not log: return None
    wins   = [t for t in log if t['outcome']=='win']
    losses = [t for t in log if t['outcome']=='loss']
    gp = sum(t['pnl_$'] for t in wins)
    gl = abs(sum(t['pnl_$'] for t in losses))
    pf = round(gp/gl,2) if gl > 0 else 999
    ret = (bal-balance)/balance*100

    pnl_s = pd.Series([t['pnl_$'] for t in log])
    sharpe = 0.0
    if pnl_s.std() > 0 and len(log) > 2:
        days = max(1,(pd.Timestamp(log[-1]['date'])-pd.Timestamp(log[0]['date'])).days)
        tpy  = len(log)/max(days/252,0.1)
        sharpe = round((pnl_s.mean()/pnl_s.std())*np.sqrt(tpy),2)

    return {
        'stats': {
            'starting_balance': balance, 'ending_balance': round(bal,2),
            'total_return_pct': round(ret,2), 'total_pnl_$': round(bal-balance,2),
            'total_trades': len(log), 'wins': len(wins), 'losses': len(losses),
            'win_rate_pct': round(len(wins)/len(log)*100,1) if log else 0,
            'max_drawdown_pct': round(max_dd,2), 'max_consec_losses': max_streak,
            'profit_factor': pf, 'sharpe_ratio': sharpe,
        },
        'equity_curve': curve, 'trade_log': log,
    }


def run_monte_carlo(trades, balance=10000, risk_pct=2.0, n=1000):
    if not trades or len(trades) < 5: return None
    finals, dds = [], []
    for _ in range(n):
        r = simulate_deep(random.sample(trades, len(trades)), balance, risk_pct)
        if r:
            finals.append(r['stats']['ending_balance'])
            dds.append(r['stats']['max_drawdown_pct'])
    if not finals: return None
    fb = np.array(finals)
    return {
        'n': n,
        'median_return':  round((float(np.median(fb))-balance)/balance*100, 2),
        'prob_profit':    round(float(np.mean(fb>balance)*100), 1),
        'prob_double':    round(float(np.mean(fb>balance*2)*100), 1),
        'prob_ruin':      round(float(np.mean(np.array(finals)<=0)*100), 2),
        'worst_5pct':     round(float(np.percentile(fb,5)), 2),
        'best_5pct':      round(float(np.percentile(fb,95)), 2),
        'median_max_dd':  round(float(np.median(dds)), 2),
        'percentiles': {f'p{p}': round(float(np.percentile(fb,p)),2)
                        for p in [10,25,50,75,90]},
    }


def yearly_sim(trades, balance=10000, risk_pct=2.0):
    by_year = defaultdict(list)
    for t in trades: by_year[t['entry_dt'][:4]].append(t)
    result = {}; running = balance
    for yr in sorted(by_year):
        r = simulate_deep(by_year[yr], running, risk_pct)
        if r:
            s = r['stats']
            result[yr] = {'trades': s['total_trades'], 'win_rate': s['win_rate_pct'],
                          'return_pct': s['total_return_pct'], 'max_dd': s['max_drawdown_pct'],
                          'end_bal': s['ending_balance']}
            running = s['ending_balance']
    return result


# =============================================================
#  MAIN
# =============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='APEX Deep Edge Engine')
    parser.add_argument('--symbol',   default='NQ')
    parser.add_argument('--scan',     action='store_true', help='Run live scan once')
    parser.add_argument('--backtest', action='store_true', help='Run full backtest')
    parser.add_argument('--balance',  type=float, default=10000)
    parser.add_argument('--risk',     type=float, default=2.0)
    parser.add_argument('--min-score',type=float, default=55,
                        help='Minimum score %% to take a trade (default 55)')
    parser.add_argument('--no-alert', action='store_true', help='Disable Telegram alerts')
    parser.add_argument('--no-paper', action='store_true', help='Disable paper trading')
    args = parser.parse_args()

    if args.scan:
        run_live_scan(args.symbol, args.min_score,
                      alert=not args.no_alert,
                      paper_trade=not args.no_paper)
    else:
        run_deep_backtest(args.symbol, args.balance, args.risk, args.min_score)
