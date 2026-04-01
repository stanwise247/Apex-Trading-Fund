"""
APEX Multi-Timeframe Edge Engine — mtf_engine.py
=================================================
The complete professional trading framework.

Architecture:
  Weekly  → Major trend (bull/bear/neutral)
  Daily   → Intermediate bias + key S/R levels  
  4-hour  → Setup confirmation + entry zone
  1-hour  → Entry trigger + structure
  15-min  → Precise entry + stop placement
  5-min   → Fine-tune entry, manage trade
  1-min   → Execution timing (optional)

A trade is only taken when:
  1. Weekly + Daily agree on direction (minimum)
  2. 4hr or 1hr confirms the setup
  3. 15min or 5min provides the precise entry signal
  4. Entry is at a key level (S/R, FVG, MA, round number)
  5. Stop is structural (below swing low/above swing high)
  6. R:R >= 2:1

Run: python3 mtf_engine.py --symbol NQ
     python3 mtf_engine.py --symbol NQ --backtest --balance 1000 --risk 2
"""

import json
import logging
import argparse
import time
import random
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
import pandas as pd
import db as _db

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('APEX.MTF')

# Timeframe hierarchy — higher index = higher timeframe
TF_HIERARCHY = ['1min', '5min', '15min', '1hour', '4hour', '1day', '1week']
TF_MINUTES   = {'1min': 1, '5min': 5, '15min': 15, '1hour': 60,
                 '4hour': 240, '1day': 1440, '1week': 10080}

# For position sizing — NQ specifics
POINT_VALUE  = 20.0   # $ per point per contract
COMMISSION   = 5.0    # $ per round trip
SLIPPAGE_PTS = {'1min': 1, '5min': 1, '15min': 2, '1hour': 2,
                '4hour': 3, '1day': 5, '1week': 8}


# =============================================================
#  DATA LOADING
# =============================================================

def load_tf(symbol, timeframe, limit=2000):
    conn = _db.connect()
    df = _db.read_sql(
        'SELECT ts,open,high,low,close,volume FROM ohlcv '
        'WHERE symbol=? AND timeframe=? ORDER BY ts DESC LIMIT ?',
        conn, params=(symbol, timeframe, limit)
    )
    conn.close()
    if df.empty:
        return None
    df = df.iloc[::-1].reset_index(drop=True)
    df['dt'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    df.set_index('dt', inplace=True)
    for col in ['open','high','low','close','volume']:
        df[col] = df[col].astype(float)
    return df


def load_all_timeframes(symbol):
    """Load all available timeframes for a symbol"""
    dfs = {}
    available = []
    for tf in TF_HIERARCHY:
        df = load_tf(symbol, tf)
        if df is not None and len(df) >= 20:
            dfs[tf] = df
            available.append(tf)
            logger.info(f'  Loaded {tf}: {len(df)} bars '
                        f'({df.index[0].strftime("%Y-%m-%d")} to {df.index[-1].strftime("%Y-%m-%d")})')
        else:
            logger.warning(f'  {tf}: insufficient data (need backfill)')
    return dfs, available


# =============================================================
#  INDICATORS (applied to any timeframe)
# =============================================================

def add_indicators(df):
    if df is None or len(df) < 10:
        return df

    c = df['close']
    h = df['high']
    l = df['low']
    v = df['volume']

    # MAs
    for p in [9, 20, 50, 200]:
        df[f'ma{p}'] = c.rolling(p).mean()
    df['ema9']  = c.ewm(span=9,  adjust=False).mean()
    df['ema21'] = c.ewm(span=21, adjust=False).mean()
    df['ema50'] = c.ewm(span=50, adjust=False).mean()

    # RSI
    delta  = c.diff()
    gain   = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss   = (-delta).clip(lower=0).ewm(com=13, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # MACD
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    df['macd']     = e12 - e26
    df['macd_sig'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist']= df['macd'] - df['macd_sig']

    # ATR
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df['atr']     = tr.ewm(span=14, adjust=False).mean()
    df['atr_pct'] = df['atr'] / c * 100

    # Bollinger
    df['bb_mid']   = c.rolling(20).mean()
    std            = c.rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2*std
    df['bb_lower'] = df['bb_mid'] - 2*std
    df['bb_pct']   = (c - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

    # Volume
    df['vol_ma20']  = v.rolling(20).mean()
    df['vol_ratio'] = v / df['vol_ma20'].replace(0, np.nan)

    # VWAP (rolling 20)
    typ = (h + l + c) / 3
    df['vwap'] = (typ * v).rolling(20).sum() / v.rolling(20).sum()

    # Trend flags
    df['above_ma20']   = (c > df['ma20']).astype(int)
    df['above_ma50']   = (c > df['ma50']).astype(int)
    df['above_ma200']  = (c > df['ma200']).astype(int)
    df['above_ema21']  = (c > df['ema21']).astype(int)

    # Candle
    df['body']      = (c - df['open']).abs()
    df['body_pct']  = df['body'] / c * 100
    df['is_bull']   = (c > df['open']).astype(int)
    df['is_bear']   = (c < df['open']).astype(int)
    df['upper_wick']= h - pd.concat([df['open'], c], axis=1).max(axis=1)
    df['lower_wick']= pd.concat([df['open'], c], axis=1).min(axis=1) - l

    # Prev bar
    df['prev_high']  = h.shift(1)
    df['prev_low']   = l.shift(1)
    df['prev_close'] = c.shift(1)
    df['gap_pct']    = (df['open'] - df['prev_close']) / df['prev_close'] * 100

    # Swing points (5-bar)
    df['swing_high'] = ((h > h.shift(1)) & (h > h.shift(2)) &
                        (h > h.shift(-1)) & (h > h.shift(-2))).astype(int)
    df['swing_low']  = ((l < l.shift(1)) & (l < l.shift(2)) &
                        (l < l.shift(-1)) & (l < l.shift(-2))).astype(int)

    # Higher highs / lower lows
    df['hh'] = (h > h.shift(1)).astype(int)
    df['hl'] = (l > l.shift(1)).astype(int)
    df['lh'] = (h < h.shift(1)).astype(int)
    df['ll'] = (l < l.shift(1)).astype(int)

    return df


# =============================================================
#  HIGHER TIMEFRAME BIAS
# =============================================================

def get_htf_bias(df, timeframe):
    """
    Analyse a single timeframe and return a structured bias score.
    Returns a dict with direction, strength, key levels, and reasoning.
    """
    if df is None or len(df) < 50:
        return {'direction': 'neutral', 'score': 50, 'available': False}

    last  = df.iloc[-1]
    c     = float(last['close'])
    rsi   = float(last['rsi'])   if not pd.isna(last['rsi'])   else 50
    macd  = float(last['macd'])  if not pd.isna(last['macd'])  else 0
    mh    = float(last['macd_hist']) if not pd.isna(last['macd_hist']) else 0
    ma20  = float(last['ma20'])  if not pd.isna(last['ma20'])  else c
    ma50  = float(last['ma50'])  if not pd.isna(last['ma50'])  else c
    ma200 = float(last['ma200']) if not pd.isna(last['ma200']) else c
    atr   = float(last['atr'])   if not pd.isna(last['atr'])   else c*0.01

    # Score from 0 (max bearish) to 100 (max bullish), 50 = neutral
    score   = 50
    reasons = []

    # MA structure (40 points)
    if c > ma20:  score += 5;  reasons.append('above MA20')
    else:         score -= 5;  reasons.append('below MA20')
    if c > ma50:  score += 8;  reasons.append('above MA50')
    else:         score -= 8;  reasons.append('below MA50')
    if c > ma200: score += 10; reasons.append('above MA200')
    else:         score -= 10; reasons.append('below MA200')
    if ma20 > ma50:  score += 7;  reasons.append('MA20>MA50')
    else:            score -= 7;  reasons.append('MA20<MA50')
    if ma50 > ma200: score += 10; reasons.append('MA50>MA200')
    else:            score -= 10; reasons.append('MA50<MA200')

    # RSI (20 points)
    if rsi > 60:   score += 10; reasons.append(f'RSI bullish ({rsi:.0f})')
    elif rsi > 50: score += 5;  reasons.append(f'RSI neutral+ ({rsi:.0f})')
    elif rsi < 40: score -= 10; reasons.append(f'RSI bearish ({rsi:.0f})')
    elif rsi < 50: score -= 5;  reasons.append(f'RSI neutral- ({rsi:.0f})')

    if rsi > 70: score -= 5; reasons.append('RSI overbought')
    if rsi < 30: score += 5; reasons.append('RSI oversold')

    # MACD (20 points)
    if macd > 0 and mh > 0:   score += 10; reasons.append('MACD bullish')
    elif macd > 0:             score += 5;  reasons.append('MACD above zero')
    elif macd < 0 and mh < 0:  score -= 10; reasons.append('MACD bearish')
    elif macd < 0:             score -= 5;  reasons.append('MACD below zero')
    if mh > 0 and float(df.iloc[-2]['macd_hist']) < 0:
        score += 5; reasons.append('MACD hist just turned bullish')
    if mh < 0 and float(df.iloc[-2]['macd_hist']) > 0:
        score -= 5; reasons.append('MACD hist just turned bearish')

    # HH/HL sequence (bullish) or LH/LL sequence (bearish)
    recent = df.tail(10)
    hh_count = int(recent['hh'].sum()) if 'hh' in recent.columns else 0
    ll_count = int(recent['ll'].sum()) if 'll' in recent.columns else 0
    if hh_count >= 6: score += 8; reasons.append('Strong HH sequence')
    if ll_count >= 6: score -= 8; reasons.append('Strong LL sequence')

    score = max(0, min(100, score))

    # Key levels
    swing_highs = df[df['swing_high'] == 1]['high'].tail(5).tolist()
    swing_lows  = df[df['swing_low']  == 1]['low'].tail(5).tolist()
    nearest_res = min([x for x in swing_highs if x > c], default=c*1.02)
    nearest_sup = max([x for x in swing_lows  if x < c], default=c*0.98)

    # FVGs in recent bars
    open_fvgs = _find_open_fvgs(df, lookback=30)

    return {
        'timeframe':   timeframe,
        'direction':   'bullish' if score >= 60 else 'bearish' if score <= 40 else 'neutral',
        'score':       round(score, 1),
        'price':       round(c, 2),
        'rsi':         round(rsi, 1),
        'ma20':        round(ma20, 2),
        'ma50':        round(ma50, 2),
        'ma200':       round(ma200, 2),
        'atr':         round(atr, 2),
        'macd_hist':   round(mh, 4),
        'reasons':     reasons[:5],
        'nearest_res': round(nearest_res, 2),
        'nearest_sup': round(nearest_sup, 2),
        'open_fvgs':   open_fvgs,
        'available':   True,
    }


def _find_open_fvgs(df, lookback=50, min_gap_pct=0.1):
    """Find unfilled FVGs in recent bars"""
    if len(df) < 5:
        return []
    recent  = df.tail(lookback)
    h = recent['high'].values
    l = recent['low'].values
    c = recent['close'].values
    cur = c[-1]
    fvgs = []
    for i in range(1, len(recent)-1):
        # Bullish FVG
        if l[i-1] > h[i+1]:
            top = l[i-1]; bot = h[i+1]
            gap_pct = (top-bot)/c[i]*100
            if gap_pct >= min_gap_pct:
                min_since = min(l[i+1:])
                if min_since > bot:  # not yet filled
                    fvgs.append({'type':'bullish','top':round(top,2),'bottom':round(bot,2),
                                 'midpoint':round((top+bot)/2,2),'gap_pct':round(gap_pct,3),
                                 'dist_pct':round((bot-cur)/cur*100,3),'bars_old':len(recent)-i})
        # Bearish FVG
        if h[i-1] < l[i+1]:
            bot = h[i-1]; top = l[i+1]
            gap_pct = (top-bot)/c[i]*100
            if gap_pct >= min_gap_pct:
                max_since = max(h[i+1:])
                if max_since < top:
                    fvgs.append({'type':'bearish','top':round(top,2),'bottom':round(bot,2),
                                 'midpoint':round((top+bot)/2,2),'gap_pct':round(gap_pct,3),
                                 'dist_pct':round((top-cur)/cur*100,3),'bars_old':len(recent)-i})
    fvgs.sort(key=lambda x: abs(x['dist_pct']))
    return fvgs[:4]


# =============================================================
#  MULTI-TIMEFRAME CONFLUENCE SCORING
# =============================================================

def get_mtf_confluence(dfs, available_tfs):
    """
    Compute the full multi-timeframe picture.
    Returns an overall bias, confluence score, and per-timeframe analysis.
    """
    tf_biases = {}
    for tf in available_tfs:
        if tf in dfs:
            df = add_indicators(dfs[tf].copy())
            tf_biases[tf] = get_htf_bias(df, tf)

    if not tf_biases:
        return {'direction': 'neutral', 'confluence': 0, 'biases': {}}

    # Weighted confluence score
    # Higher timeframes get more weight
    weights = {
        '1week': 3.0, '1day': 2.5, '4hour': 2.0,
        '1hour': 1.5, '15min': 1.0, '5min': 0.7, '1min': 0.3
    }

    total_weight  = 0
    weighted_score= 0
    bull_count    = 0
    bear_count    = 0
    neutral_count = 0

    for tf, bias in tf_biases.items():
        if not bias.get('available'):
            continue
        w = weights.get(tf, 1.0)
        total_weight   += w
        weighted_score += bias['score'] * w
        if bias['direction'] == 'bullish':   bull_count += 1
        elif bias['direction'] == 'bearish': bear_count += 1
        else:                                neutral_count += 1

    if total_weight == 0:
        return {'direction': 'neutral', 'confluence': 0, 'biases': tf_biases}

    avg_score = weighted_score / total_weight
    n = bull_count + bear_count + neutral_count

    # Confluence: how aligned are the timeframes?
    dominant     = max(bull_count, bear_count)
    confluence   = round((dominant / n) * 100, 1) if n > 0 else 0

    direction    = 'bullish' if avg_score >= 58 else 'bearish' if avg_score <= 42 else 'neutral'

    # HTF agreement — check multiple combinations, require at least 2 TFs to agree
    htf_agree = False
    agree_pairs = [
        ('1week', '1day'), ('1day', '4hour'), ('4hour', '1hour'),
        ('1week', '4hour'), ('1day', '1hour'),
    ]
    for tf_a, tf_b in agree_pairs:
        if tf_a in tf_biases and tf_b in tf_biases:
            da = tf_biases[tf_a]['direction']
            db = tf_biases[tf_b]['direction']
            if da == db and da != 'neutral':
                htf_agree = True
                direction = da  # use the agreed direction
                break
    # Fallback: if we only have daily+weekly and they disagree,
    # use the daily as primary (more current) if score is strong
    if not htf_agree and '1day' in tf_biases:
        day_score = tf_biases['1day']['score']
        if day_score >= 65 or day_score <= 35:
            htf_agree = True
            direction = tf_biases['1day']['direction']

    # Overall trade quality
    if htf_agree and confluence >= 70:
        quality = 'A — High conviction'
    elif htf_agree and confluence >= 55:
        quality = 'B — Good setup'
    elif confluence >= 50:
        quality = 'C — Moderate'
    else:
        quality = 'D — Wait for alignment'

    return {
        'direction':    direction,
        'avg_score':    round(avg_score, 1),
        'confluence':   confluence,
        'bull_tfs':     bull_count,
        'bear_tfs':     bear_count,
        'neutral_tfs':  neutral_count,
        'htf_agree':    htf_agree,
        'quality':      quality,
        'biases':       tf_biases,
    }


# =============================================================
#  ENTRY SIGNAL DETECTION (lower timeframes)
# =============================================================

def find_entry_signals(df, timeframe, direction, htf_bias):
    """
    Given a higher-timeframe directional bias, scan the lower timeframe
    for precise entry signals. Returns a list of signal dicts.
    """
    if df is None or len(df) < 50:
        return []

    df   = add_indicators(df.copy())
    signals = []

    # Use last 100 bars for signal detection
    work = df.tail(100).copy()
    c    = work['close'].values
    h    = work['high'].values
    l    = work['low'].values
    atr  = work['atr'].values
    rsi  = work['rsi'].values
    mh   = work['macd_hist'].values
    vr   = work['vol_ratio'].values
    bp   = work['bb_pct'].values
    ma20 = work['ma20'].values
    ma50 = work['ma50'].values
    ema21= work['ema21'].values
    sw_h = work['swing_high'].values
    sw_l = work['swing_low'].values
    is_b = work['is_bull'].values
    is_r = work['is_bear'].values
    idx  = work.index

    for i in range(5, len(work)-2):
        price = c[i]
        at    = atr[i] if not np.isnan(atr[i]) else price * 0.005
        rs    = rsi[i] if not np.isnan(rsi[i]) else 50

        found = []

        if direction in ('bullish', 'long'):
            # ── SIGNAL 1: RSI pullback to 40-50 zone at MA support ──
            if (38 <= rs <= 52 and
                not np.isnan(rsi[i-1]) and rsi[i-1] < rsi[i] and  # RSI turning up
                not np.isnan(vr[i]) and vr[i] > 0.7 and  # some volume present
                price > ma50[i] * 0.998 if not np.isnan(ma50[i]) else True):
                stop  = min(l[i-2:i+1]) - at * 0.3
                tgt1  = price + (price - stop) * 3.0
                tgt2  = price + (price - stop) * 5.0
                found.append(('RSI_Pullback_MA_Support', stop, tgt1, tgt2, 70))

            # ── SIGNAL 2: Bullish engulfing at swing low ──
            if (i >= 1 and is_b[i] and is_r[i-1] and
                c[i] > c[i-1] and work['open'].values[i] <= c[i-1] and
                rs < 55):
                stop  = l[i] - at * 0.3
                tgt1  = price + (price - stop) * 3.0
                tgt2  = price + (price - stop) * 5.0
                found.append(('Bullish_Engulfing_Swing_Low', stop, tgt1, tgt2, 72))

            # ── SIGNAL 3: FVG fill (bullish FVG from above, price returning) ──
            fvgs = _find_open_fvgs(work.iloc[:i+1], lookback=30)
            for fvg in fvgs:
                if (fvg['type'] == 'bullish' and
                    fvg['bottom'] <= price <= fvg['top'] * 1.001):
                    stop  = fvg['bottom'] - at * 0.5
                    tgt1  = price + (price - stop) * 2.0
                    tgt2  = price + (price - stop) * 4.0
                    found.append(('FVG_Bullish_Fill', stop, tgt1, tgt2, 80))
                    break

            # ── SIGNAL 4: EMA21 reclaim with volume ──
            if (i >= 1 and
                not np.isnan(ema21[i]) and not np.isnan(ema21[i-1]) and
                price > ema21[i] and c[i-1] <= ema21[i-1] and
                not np.isnan(vr[i]) and vr[i] > 1.2):
                stop  = min(l[i-2:i+1]) - at * 0.2
                tgt1  = price + (price - stop) * 3.0
                tgt2  = price + (price - stop) * 4.5
                found.append(('EMA21_Reclaim_Volume', stop, tgt1, tgt2, 74))

            # ── SIGNAL 5: Bollinger lower band bounce ──
            if (not np.isnan(bp[i]) and bp[i] < 0.08 and is_b[i] and
                rs < 45):
                stop  = l[i] - at * 0.5
                tgt1  = price + at * 2.0
                tgt2  = price + at * 3.5
                found.append(('BB_Lower_Bounce', stop, tgt1, tgt2, 66))

            # ── SIGNAL 6: Previous swing high breakout with volume ──
            if (sw_h[i-3] and price > h[i-3] and
                not np.isnan(vr[i]) and vr[i] > 1.3 and
                is_b[i]):
                stop  = h[i-3] - at * 0.5
                tgt1  = price + (price - stop) * 3.0
                tgt2  = price + (price - stop) * 5.0
                found.append(('Swing_High_Breakout', stop, tgt1, tgt2, 76))

            # ── SIGNAL 7: MACD histogram bullish cross at support ──
            if (i >= 1 and
                not np.isnan(mh[i]) and not np.isnan(mh[i-1]) and
                mh[i] > 0 and mh[i-1] < 0 and
                not np.isnan(ma20[i]) and abs(price - ma20[i]) / ma20[i] < 0.005):
                stop  = l[i] - at * 0.5
                tgt1  = price + at * 2.5
                tgt2  = price + at * 4.0
                found.append(('MACD_Cross_At_MA20', stop, tgt1, tgt2, 68))

        else:  # bearish / short
            # ── SIGNAL 1: RSI bounce to 50-62 zone at MA resistance ──
            if (50 <= rs <= 62 and
                not np.isnan(rsi[i-1]) and rsi[i-1] > rsi[i] and  # RSI turning down
                price < ma50[i] * 1.002 if not np.isnan(ma50[i]) else True):
                stop  = max(h[i-2:i+1]) + at * 0.3
                tgt1  = price - (stop - price) * 3.0
                tgt2  = price - (stop - price) * 5.0
                found.append(('RSI_Bounce_MA_Resistance', stop, tgt1, tgt2, 70))

            # ── SIGNAL 2: Bearish engulfing at swing high ──
            if (i >= 1 and is_r[i] and is_b[i-1] and
                c[i] < c[i-1] and work['open'].values[i] >= c[i-1] and
                rs > 45):
                stop  = h[i] + at * 0.3
                tgt1  = price - (stop - price) * 3.0
                tgt2  = price - (stop - price) * 5.0
                found.append(('Bearish_Engulfing_Swing_High', stop, tgt1, tgt2, 72))

            # ── SIGNAL 3: FVG fill (bearish FVG from below, price returning) ──
            fvgs = _find_open_fvgs(work.iloc[:i+1], lookback=30)
            for fvg in fvgs:
                if (fvg['type'] == 'bearish' and
                    fvg['bottom'] * 0.999 <= price <= fvg['top']):
                    stop  = fvg['top'] + at * 0.5
                    tgt1  = price - (stop - price) * 2.0
                    tgt2  = price - (stop - price) * 4.0
                    found.append(('FVG_Bearish_Fill', stop, tgt1, tgt2, 80))
                    break

            # ── SIGNAL 4: EMA21 rejection with volume ──
            if (i >= 1 and
                not np.isnan(ema21[i]) and not np.isnan(ema21[i-1]) and
                price < ema21[i] and c[i-1] >= ema21[i-1] and
                not np.isnan(vr[i]) and vr[i] > 1.2):
                stop  = max(h[i-2:i+1]) + at * 0.2
                tgt1  = price - (stop - price) * 3.0
                tgt2  = price - (stop - price) * 4.5
                found.append(('EMA21_Rejection_Volume', stop, tgt1, tgt2, 74))

            # ── SIGNAL 5: Bollinger upper band rejection ──
            if (not np.isnan(bp[i]) and bp[i] > 0.92 and is_r[i] and
                rs > 55):
                stop  = h[i] + at * 0.5
                tgt1  = price - at * 2.0
                tgt2  = price - at * 3.5
                found.append(('BB_Upper_Rejection', stop, tgt1, tgt2, 66))

            # ── SIGNAL 6: Previous swing low breakdown with volume ──
            if (sw_l[i-3] and price < l[i-3] and
                not np.isnan(vr[i]) and vr[i] > 1.3 and
                is_r[i]):
                stop  = l[i-3] + at * 0.5
                tgt1  = price - (stop - price) * 3.0
                tgt2  = price - (stop - price) * 5.0
                found.append(('Swing_Low_Breakdown', stop, tgt1, tgt2, 76))

            # ── SIGNAL 7: MACD histogram bearish cross at resistance ──
            if (i >= 1 and
                not np.isnan(mh[i]) and not np.isnan(mh[i-1]) and
                mh[i] < 0 and mh[i-1] > 0 and
                not np.isnan(ma20[i]) and abs(price - ma20[i]) / ma20[i] < 0.005):
                stop  = h[i] + at * 0.5
                tgt1  = price - at * 2.5
                tgt2  = price - at * 4.0
                found.append(('MACD_Cross_At_MA20_Short', stop, tgt1, tgt2, 68))

        # Build signal objects from found entries
        for name, stop, tgt1, tgt2, base_score in found:
            risk   = abs(price - stop)
            if risk <= 0:
                continue
            reward = abs(tgt1 - price)
            rr     = reward / risk

            # Minimum R:R filter
            if rr < 2.0:
                continue

            # Confluence bonus from HTF
            htf_score = htf_bias.get('avg_score', 50)
            bonus = int((abs(htf_score - 50) / 50) * 15)
            final_score = min(100, base_score + bonus)

            signals.append({
                'signal':    name,
                'timeframe': timeframe,
                'direction': 'long' if direction in ('bullish','long') else 'short',
                'bar_idx':   i,
                'bar_dt':    idx[i].isoformat(),
                'price':     round(price, 2),
                'stop':      round(stop, 2),
                'target1':   round(tgt1, 2),
                'target2':   round(tgt2, 2),
                'rr':        round(rr, 2),
                'atr':       round(at, 2),
                'rsi':       round(rs, 1),
                'score':     final_score,
            })

    return signals


# =============================================================
#  MTF BACKTEST
# =============================================================

def run_mtf_backtest(dfs, available_tfs, symbol):
    """
    The full multi-timeframe backtest.

    Strategy:
    1. For each bar on the ENTRY timeframe (15min or 5min):
       a. Check higher TF alignment (1h, 4h, 1d)
       b. If aligned, look for entry signals
       c. Backtest the resulting trade using forward bars
    """
    # Determine which TF to use for entries
    entry_tfs = [tf for tf in ['15min', '5min', '1hour'] if tf in available_tfs]
    if not entry_tfs:
        entry_tfs = [tf for tf in ['1hour', '1day'] if tf in available_tfs]

    if not entry_tfs:
        logger.error('No entry timeframes available')
        return []

    logger.info(f'Entry timeframes available: {entry_tfs}')

    all_trades = []

    for entry_tf in entry_tfs:
        logger.info(f'\nRunning MTF backtest on {entry_tf} entries...')
        entry_df = add_indicators(dfs[entry_tf].copy())

        # Higher TF analysis — align bias every N bars
        higher_tfs = [tf for tf in TF_HIERARCHY
                      if tf in available_tfs and
                      TF_MINUTES.get(tf, 0) > TF_MINUTES.get(entry_tf, 0)]

        if not higher_tfs:
            continue

        # Build HTF bias for each point in time
        # For efficiency, compute once per day
        htf_bias_cache = {}

        for i in range(50, len(entry_df)-5):
            bar_dt  = entry_df.index[i]
            day_key = bar_dt.strftime('%Y-%m-%d')

            if day_key not in htf_bias_cache:
                # Get HTF bias up to this point in time
                htf_dfs_at_time = {}
                for htf in higher_tfs:
                    htf_data = dfs[htf][dfs[htf].index <= bar_dt]
                    if len(htf_data) >= 20:
                        htf_dfs_at_time[htf] = htf_data
                if htf_dfs_at_time:
                    htf_bias_cache[day_key] = get_mtf_confluence(
                        {k: add_indicators(v.copy()) for k, v in htf_dfs_at_time.items()},
                        list(htf_dfs_at_time.keys())
                    )
                else:
                    htf_bias_cache[day_key] = None

            mtf = htf_bias_cache.get(day_key)
            if mtf is None:
                continue

            # Only trade when higher TFs agree
            if not mtf.get('htf_agree') or mtf.get('confluence', 0) < 55:
                continue

            direction = mtf['direction']
            if direction == 'neutral':
                continue

            # Look for entry signal on this bar
            window = entry_df.iloc[max(0, i-50):i+1]
            if len(window) < 20:
                continue

            signals = find_entry_signals(window, entry_tf, direction, mtf)
            if not signals:
                continue

            # Take the highest-scored signal only
            sig = max(signals, key=lambda x: x['score'])
            if sig['score'] < 65:
                continue

            # Max 2 trades per day per timeframe to prevent overtrading
            trade_date = entry_df.index[i].strftime('%Y-%m-%d')
            day_trades = sum(1 for t in all_trades
                           if t['entry_dt'][:10] == trade_date
                           and t['entry_tf'] == entry_tf)
            if day_trades >= 2:
                continue

            # Forward-test this trade
            entry      = sig['price']
            stop       = sig['stop']
            tgt1       = sig['target1']
            tgt2       = sig['target2']
            sl_pts     = SLIPPAGE_PTS.get(entry_tf, 2)

            # Apply slippage
            if direction == 'bullish':
                entry += sl_pts
                stop  = stop
                tgt1  = tgt1
            else:
                entry -= sl_pts

            risk   = abs(entry - stop)
            if risk <= 0:
                continue

            # Forward bars to test outcome
            outcome    = 'timeout'
            exit_price = entry
            max_bars   = min(i + 50, len(entry_df) - 1)

            for j in range(i+1, max_bars):
                bh = float(entry_df.iloc[j]['high'])
                bl = float(entry_df.iloc[j]['low'])
                if direction in ('bullish', 'long'):
                    if bl <= stop:
                        outcome    = 'loss'
                        exit_price = stop
                        break
                    if bh >= tgt1:
                        outcome    = 'win'
                        exit_price = tgt1
                        break
                else:
                    if bh >= stop:
                        outcome    = 'loss'
                        exit_price = stop
                        break
                    if bl <= tgt1:
                        outcome    = 'win'
                        exit_price = tgt1
                        break

            if outcome == 'timeout':
                exit_price = float(entry_df.iloc[min(i+5, len(entry_df)-1)]['close'])

            pnl_pts = (exit_price - entry) if direction in ('bullish','long') else (entry - exit_price)
            pnl_r   = pnl_pts / risk

            all_trades.append({
                'entry_dt':   entry_df.index[i].isoformat(),
                'signal':     sig['signal'],
                'entry_tf':   entry_tf,
                'direction':  direction,
                'htf_quality':mtf.get('quality',''),
                'confluence': mtf.get('confluence', 0),
                'htf_score':  mtf.get('avg_score', 50),
                'entry':      round(entry, 2),
                'stop':       round(stop, 2),
                'target1':    round(tgt1, 2),
                'exit':       round(exit_price, 2),
                'outcome':    outcome,
                'pnl_pts':    round(pnl_pts, 2),
                'pnl_r':      round(pnl_r, 3),
                'rr':         round(abs(tgt1-entry)/risk, 2),
                'score':      sig['score'],
                'rsi':        sig.get('rsi'),
            })

    return all_trades


# =============================================================
#  SIMULATION (same engine as backtest.py)
# =============================================================

def simulate(trades, balance=1000, risk_pct=2.0, point_value=POINT_VALUE, commission=COMMISSION):
    if not trades:
        return None

    bal       = balance
    peak      = balance
    curve     = [{'balance': bal, 'trade_num': 0, 'date': 'start'}]
    log       = []
    max_dd    = 0
    max_streak= 0
    streak    = 0

    for i, t in enumerate(sorted(trades, key=lambda x: x['entry_dt'])):
        if bal <= 10:
            break
        risk_amt   = max(bal * (risk_pct / 100), 1.0)
        stop_dist  = abs(t['entry'] - t['stop'])
        if stop_dist <= 0:
            continue
        # Use R-multiple P&L — each trade risks exactly risk_amt dollars
        # pnl_r = +2.0 means won 2R, pnl_r = -1.0 means lost 1R
        pnl_r   = float(t.get('pnl_r', 0))
        outcome = t.get('outcome', 'timeout')
        # Ensure P&L matches outcome direction
        if outcome == 'win' and pnl_r <= 0:
            pnl_r = abs(float(t.get('rr', 2.0)))
        elif outcome == 'loss' and pnl_r >= 0:
            pnl_r = -1.0
        pnl_usd = (risk_amt * pnl_r) - commission

        bal += pnl_usd
        if bal > peak: peak = bal
        dd = (peak - bal) / peak * 100
        if dd > max_dd: max_dd = dd

        if t['outcome'] == 'loss':
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

        log.append({
            'num':       i+1,
            'date':      t['entry_dt'][:10],
            'signal':    t.get('signal',''),
            'tf':        t.get('entry_tf',''),
            'direction': t['direction'],
            'entry':     t['entry'],
            'exit':      t['exit'],
            'pnl_pts':   round(float(t.get('pnl_pts',0)), 2),
            'pnl_$':     round(pnl_usd, 2),
            'balance':   round(bal, 2),
            'outcome':   t['outcome'],
            'score':     t.get('score', 0),
            'confluence':t.get('confluence', 0),
            'dd_pct':    round(dd, 2),
        })
        curve.append({'balance': round(bal,2), 'trade_num': i+1,
                      'date': t['entry_dt'][:10], 'pnl': round(pnl_usd,2)})

    if not log:
        return None

    wins   = [t for t in log if t['outcome'] == 'win']
    losses = [t for t in log if t['outcome'] == 'loss']
    ret    = (bal - balance) / balance * 100
    gp     = sum(t['pnl_$'] for t in wins)
    gl     = abs(sum(t['pnl_$'] for t in losses))

    pnl_s  = pd.Series([t['pnl_$'] for t in log])
    sharpe = 0.0
    if pnl_s.std() > 0 and len(log) > 2:
        days = max(1, (pd.Timestamp(log[-1]['date']) - pd.Timestamp(log[0]['date'])).days)
        tpy  = len(log) / max(days/252, 0.1)
        sharpe = round((pnl_s.mean() / pnl_s.std()) * np.sqrt(tpy), 2)

    return {
        'stats': {
            'starting_balance':  balance,
            'ending_balance':    round(bal, 2),
            'total_return_pct':  round(ret, 2),
            'total_pnl_$':       round(bal-balance, 2),
            'total_trades':      len(log),
            'wins':              len(wins),
            'losses':            len(losses),
            'win_rate_pct':      round(len(wins)/len(log)*100,1) if log else 0,
            'avg_win_$':         round(np.mean([t['pnl_$'] for t in wins]),2) if wins else 0,
            'avg_loss_$':        round(np.mean([t['pnl_$'] for t in losses]),2) if losses else 0,
            'max_drawdown_pct':  round(max_dd, 2),
            'max_consec_losses': max_streak,
            'profit_factor':     round(gp/gl, 2) if gl > 0 else 999,
            'sharpe_ratio':      sharpe,
            'risk_pct_per_trade':risk_pct,
        },
        'equity_curve': curve,
        'trade_log':    log,
    }


def monte_carlo(trades, balance=1000, risk_pct=2.0, n=1000):
    if not trades or len(trades) < 5:
        return None
    finals, dds = [], []
    for _ in range(n):
        shuffled = random.sample(trades, len(trades))
        r = simulate(shuffled, balance, risk_pct)
        if r:
            finals.append(r['stats']['ending_balance'])
            dds.append(r['stats']['max_drawdown_pct'])
    if not finals:
        return None
    fb = np.array(finals)
    return {
        'n': n,
        'median_final':   round(float(np.median(fb)), 2),
        'median_return':  round((float(np.median(fb))-balance)/balance*100, 2),
        'prob_profit':    round(float(np.mean(fb > balance)*100), 1),
        'prob_double':    round(float(np.mean(fb > balance*2)*100), 1),
        'prob_ruin':      round(float(np.mean(np.array(finals) <= 0)*100), 2),
        'worst_5pct':     round(float(np.percentile(fb, 5)), 2),
        'best_5pct':      round(float(np.percentile(fb, 95)), 2),
        'median_max_dd':  round(float(np.median(dds)), 2),
        'percentiles': {
            'p10': round(float(np.percentile(fb,10)),2),
            'p25': round(float(np.percentile(fb,25)),2),
            'p50': round(float(np.percentile(fb,50)),2),
            'p75': round(float(np.percentile(fb,75)),2),
            'p90': round(float(np.percentile(fb,90)),2),
        },
        'sample': sorted(random.sample(list(fb), min(200, len(fb))))
    }


def yearly_breakdown(trades, balance=1000, risk_pct=2.0):
    by_year = defaultdict(list)
    for t in trades:
        by_year[t['entry_dt'][:4]].append(t)
    result = {}
    running = balance
    for year in sorted(by_year):
        r = simulate(by_year[year], running, risk_pct)
        if r:
            s = r['stats']
            result[year] = {
                'trades':    s['total_trades'],
                'win_rate':  s['win_rate_pct'],
                'return_pct':s['total_return_pct'],
                'pnl_$':     s['total_pnl_$'],
                'max_dd':    s['max_drawdown_pct'],
                'end_bal':   s['ending_balance'],
            }
            running = s['ending_balance']
    return result


def compare_risk_levels(trades, balance=1000):
    out = {}
    for risk in [1, 2, 3, 4]:
        r = simulate(trades, balance, risk)
        if r:
            s = r['stats']
            out[f'{risk}pct'] = {
                'risk_pct':   risk,
                'return_pct': s['total_return_pct'],
                'end_bal':    s['ending_balance'],
                'max_dd':     s['max_drawdown_pct'],
                'sharpe':     s['sharpe_ratio'],
                'pf':         s['profit_factor'],
            }
    return out


# =============================================================
#  LIVE SCANNER
# =============================================================

def scan_live(symbol, dfs, available_tfs):
    """
    Scan for live setups right now using multi-timeframe confluence.
    Returns active setups with full context.
    """
    mtf = get_mtf_confluence(
        {k: add_indicators(v.copy()) for k, v in dfs.items()},
        available_tfs
    )

    direction = mtf.get('direction', 'neutral')
    setups    = []

    # Scan entry timeframes for signals
    entry_tfs = [tf for tf in ['5min', '15min', '1hour'] if tf in available_tfs]

    for entry_tf in entry_tfs:
        df     = add_indicators(dfs[entry_tf].copy())
        window = df.tail(60)
        sigs   = find_entry_signals(window, entry_tf, direction, mtf)

        for sig in sigs:
            # Only show signals from the last few bars
            if sig['bar_idx'] < len(window) - 5:
                continue
            if sig['score'] < 65:
                continue

            setups.append({
                **sig,
                'mtf_direction':  direction,
                'mtf_confluence': mtf.get('confluence', 0),
                'mtf_quality':    mtf.get('quality', ''),
                'htf_agree':      mtf.get('htf_agree', False),
                'bull_tfs':       mtf.get('bull_tfs', 0),
                'bear_tfs':       mtf.get('bear_tfs', 0),
            })

    setups.sort(key=lambda x: x['score'], reverse=True)
    return {'mtf': mtf, 'setups': setups}


# =============================================================
#  MAIN
# =============================================================

def run_mtf_engine(symbol='NQ', balance=1000, risk_pct=2.0,
                   do_backtest=True, n_monte_carlo=1000):
    logger.info('=' * 60)
    logger.info(f'  APEX Multi-Timeframe Engine — {symbol}')
    logger.info(f'  Balance: ${balance:,} | Risk: {risk_pct}%/trade')
    logger.info('=' * 60)

    dfs, available = load_all_timeframes(symbol)

    if not dfs:
        logger.error('No data found. Run backfill for all timeframes first.')
        return None

    # MTF analysis
    logger.info('\nComputing multi-timeframe confluence...')
    mtf_dfs = {k: add_indicators(v.copy()) for k, v in dfs.items()}
    mtf     = get_mtf_confluence(mtf_dfs, available)

    logger.info(f'\n  MTF Summary:')
    logger.info(f'  Overall Direction: {mtf["direction"].upper()}')
    logger.info(f'  Avg Score:         {mtf["avg_score"]}/100')
    logger.info(f'  Confluence:        {mtf["confluence"]}%')
    logger.info(f'  Higher TF Agree:   {mtf["htf_agree"]}')
    logger.info(f'  Quality:           {mtf["quality"]}')
    logger.info(f'\n  Per-timeframe breakdown:')

    for tf in TF_HIERARCHY:
        if tf in mtf['biases'] and mtf['biases'][tf].get('available'):
            b = mtf['biases'][tf]
            logger.info(f'    {tf:<8} {b["direction"].upper():<8} score={b["score"]}/100  '
                        f'RSI={b["rsi"]:.0f}  '
                        f'FVGs={len(b.get("open_fvgs",[]))}')

    if not do_backtest:
        # Live scan mode
        scan  = scan_live(symbol, dfs, available)
        output = {'symbol': symbol, 'mtf': mtf, 'live_setups': scan['setups'],
                  'timestamp': datetime.now(timezone.utc).isoformat()}
        with open(f'mtf_live_{symbol}.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)
        return output

    # Backtest
    logger.info('\nRunning MTF backtest...')
    trades = run_mtf_backtest(dfs, available, symbol)
    logger.info(f'Total trades generated: {len(trades)}')

    if not trades:
        logger.warning('No trades generated. Check data availability for entry timeframes.')
        return None

    wins = [t for t in trades if t['outcome'] == 'win']
    losses = [t for t in trades if t['outcome'] == 'loss']
    logger.info(f'Win rate: {len(wins)/len(trades)*100:.1f}%  '
                f'({len(wins)}W / {len(losses)}L / {len(trades)-len(wins)-len(losses)}T)')

    # Simulation
    logger.info(f'\nRunning simulation: ${balance} at {risk_pct}% risk...')
    sim = simulate(trades, balance, risk_pct)

    if sim:
        s = sim['stats']
        logger.info(f'  Return:       {s["total_return_pct"]:+.1f}%  (${s["total_pnl_$"]:+,.2f})')
        logger.info(f'  Win Rate:     {s["win_rate_pct"]:.1f}%')
        logger.info(f'  Max Drawdown: {s["max_drawdown_pct"]:.1f}%')
        logger.info(f'  Sharpe:       {s["sharpe_ratio"]:.2f}')
        logger.info(f'  Profit Factor:{s["profit_factor"]}')
        logger.info(f'  Max Streak:   {s["max_consec_losses"]} losses')

    # Monte Carlo
    logger.info(f'\nRunning Monte Carlo ({n_monte_carlo} simulations)...')
    mc = monte_carlo(trades, balance, risk_pct, n_monte_carlo)
    if mc:
        logger.info(f'  Median Return:  {mc["median_return"]:+.1f}%')
        logger.info(f'  Prob of Profit: {mc["prob_profit"]}%')
        logger.info(f'  Prob of Double: {mc["prob_double"]}%')
        logger.info(f'  Prob of Ruin:   {mc["prob_ruin"]}%')
        logger.info(f'  Worst 5%:       ${mc["worst_5pct"]:,.2f}')
        logger.info(f'  Best 5%:        ${mc["best_5pct"]:,.2f}')

    # Yearly
    yearly = yearly_breakdown(trades, balance, risk_pct)
    if yearly:
        logger.info(f'\n  Year-by-Year:')
        for yr, y in yearly.items():
            logger.info(f'    {yr}: {y["return_pct"]:+6.1f}%  '
                        f'Win={y["win_rate"]:.0f}%  '
                        f'DD={y["max_dd"]:.1f}%  '
                        f'Trades={y["trades"]}')

    # Risk comparison
    risk_comp = compare_risk_levels(trades, balance)
    logger.info(f'\n  Risk Level Comparison:')
    for key, r in risk_comp.items():
        logger.info(f'    {r["risk_pct"]}% risk: Return={r["return_pct"]:+.1f}%  '
                    f'MaxDD={r["max_dd"]:.1f}%  Sharpe={r["sharpe"]:.2f}')

    output = {
        'symbol':     symbol,
        'timestamp':  datetime.now(timezone.utc).isoformat(),
        'config': {'balance': balance, 'risk_pct': risk_pct},
        'available_timeframes': available,
        'mtf_analysis': mtf,
        'trades':     trades,
        'simulation': sim,
        'monte_carlo':mc,
        'yearly':     yearly,
        'risk_comparison': risk_comp,
        'summary': {
            'total_trades': len(trades),
            'win_rate': round(len(wins)/len(trades)*100, 1) if trades else 0,
            'total_return': sim['stats']['total_return_pct'] if sim else 0,
            'max_dd': sim['stats']['max_drawdown_pct'] if sim else 0,
            'sharpe': sim['stats']['sharpe_ratio'] if sim else 0,
        }
    }

    outfile = f'mtf_results_{symbol}.json'
    with open(outfile, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f'\n  Full results saved to: {outfile}')
    logger.info('=' * 60)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='APEX Multi-Timeframe Engine')
    parser.add_argument('--symbol',    default='NQ')
    parser.add_argument('--balance',   type=float, default=1000)
    parser.add_argument('--risk',      type=float, default=2.0)
    parser.add_argument('--mc',        type=int,   default=1000)
    parser.add_argument('--backtest',  action='store_true', help='Run full backtest')
    parser.add_argument('--scan',      action='store_true', help='Scan for live setups only')
    args = parser.parse_args()

    run_mtf_engine(
        symbol       = args.symbol,
        balance      = args.balance,
        risk_pct     = args.risk,
        do_backtest  = args.backtest or not args.scan,
        n_monte_carlo= args.mc,
    )
