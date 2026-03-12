"""
APEX Liquidity Sweep Detection — liquidity_sweep.py
=====================================================
Detects liquidity sweeps (stop hunts) on any timeframe.

A liquidity sweep occurs when price:
  1. Runs through a level where stops are clustered (equal highs/lows, swing points)
  2. Closes back inside the range (the trap is set)
  3. Often accompanied by a volume spike

This is one of the highest-probability reversal signals in futures markets.

Usage:
    from liquidity_sweep import detect_liquidity_sweep
    result = detect_liquidity_sweep(df_5min, df_1hour)
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger('APEX.LiqSweep')


def find_equal_levels(df, lookback=50, tolerance_pct=0.08):
    """
    Find clusters of equal highs or equal lows within tolerance.
    These are where stop orders are clustered.

    Returns list of {'price': float, 'type': 'high'|'low', 'count': int, 'strength': float}
    """
    if df is None or len(df) < lookback:
        return []

    recent = df.tail(lookback)
    atr    = _calc_atr(recent)
    if atr is None or atr == 0:
        return []

    tol = recent['close'].iloc[-1] * (tolerance_pct / 100)
    levels = []

    # Find swing highs
    highs = recent['high'].values
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
           highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            cluster = sum(1 for h in highs if abs(h - highs[i]) <= tol)
            if cluster >= 2:
                strength = min(cluster / 3, 1.0)
                levels.append({
                    'price':    round(highs[i], 2),
                    'type':     'high',
                    'count':    cluster,
                    'strength': strength,
                })

    # Find swing lows
    lows = recent['low'].values
    for i in range(2, len(lows) - 2):
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
           lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            cluster = sum(1 for l in lows if abs(l - lows[i]) <= tol)
            if cluster >= 2:
                strength = min(cluster / 3, 1.0)
                levels.append({
                    'price':    round(lows[i], 2),
                    'type':     'low',
                    'count':    cluster,
                    'strength': strength,
                })

    # Deduplicate — keep strongest level in each price cluster
    deduped = []
    for lv in sorted(levels, key=lambda x: -x['strength']):
        if not any(abs(lv['price'] - d['price']) <= tol * 2 for d in deduped):
            deduped.append(lv)

    return deduped


def detect_liquidity_sweep(df_entry, df_htf=None, lookback=30, atr_multiplier=0.5):
    """
    Detect a liquidity sweep on the most recent bars.

    Args:
        df_entry:       Entry timeframe df (5min or 15min)
        df_htf:         Higher timeframe df (1hour) for confluence
        lookback:       Bars to look back for equal levels
        atr_multiplier: How far price must sweep beyond level

    Returns dict:
        {
            'detected':      bool,
            'direction':     'long' | 'short',
            'swept_level':   float,
            'sweep_size':    float,   # in ATR units
            'volume_spike':  bool,
            'htf_confluence': bool,
            'score_bonus':   int,     # points to add (0-10)
            'reason':        str,
        }
    """
    empty = {'detected': False, 'direction': None, 'swept_level': None,
             'sweep_size': 0, 'volume_spike': False, 'htf_confluence': False,
             'score_bonus': 0, 'reason': 'No sweep detected'}

    if df_entry is None or len(df_entry) < 20:
        return empty

    atr = _calc_atr(df_entry)
    if atr is None or atr == 0:
        return empty

    levels  = find_equal_levels(df_entry, lookback=lookback)
    if not levels:
        return empty

    last    = df_entry.iloc[-1]
    price   = last['close']
    avg_vol = df_entry['volume'].tail(20).mean()

    # Sweep of highs → short setup (price traps bulls above highs then closes back down)
    for lv in sorted([l for l in levels if l['type'] == 'high'], key=lambda x: -x['strength']):
        level_price = lv['price']
        sweep_size  = (last['high'] - level_price) / atr
        if (last['high'] > level_price + atr * atr_multiplier and
                last['close'] < level_price and sweep_size > 0):
            vol_spike = last['volume'] > avg_vol * 1.3
            htf_conf  = _check_htf_confluence(df_htf, 'short', price)
            bonus     = _calc_bonus(lv['strength'], vol_spike, htf_conf, sweep_size)
            return {
                'detected':       True,
                'direction':      'short',
                'swept_level':    level_price,
                'sweep_size':     round(sweep_size, 2),
                'volume_spike':   vol_spike,
                'htf_confluence': htf_conf,
                'score_bonus':    bonus,
                'reason':         f'Swept highs at {level_price:.2f} ({lv["count"]} equal highs, strength={lv["strength"]:.1f})',
            }

    # Sweep of lows → long setup (price traps bears below lows then closes back up)
    for lv in sorted([l for l in levels if l['type'] == 'low'], key=lambda x: -x['strength']):
        level_price = lv['price']
        sweep_size  = (level_price - last['low']) / atr
        if (last['low'] < level_price - atr * atr_multiplier and
                last['close'] > level_price and sweep_size > 0):
            vol_spike = last['volume'] > avg_vol * 1.3
            htf_conf  = _check_htf_confluence(df_htf, 'long', price)
            bonus     = _calc_bonus(lv['strength'], vol_spike, htf_conf, sweep_size)
            return {
                'detected':       True,
                'direction':      'long',
                'swept_level':    level_price,
                'sweep_size':     round(sweep_size, 2),
                'volume_spike':   vol_spike,
                'htf_confluence': htf_conf,
                'score_bonus':    bonus,
                'reason':         f'Swept lows at {level_price:.2f} ({lv["count"]} equal lows, strength={lv["strength"]:.1f})',
            }

    return empty


def _calc_atr(df, period=14):
    """Calculate Average True Range"""
    try:
        high  = df['high']
        low   = df['low']
        close = df['close'].shift(1)
        tr    = pd.concat([
            high - low,
            (high - close).abs(),
            (low  - close).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean().iloc[-1]
    except Exception:
        return None


def _check_htf_confluence(df_htf, direction, price):
    """Check if higher timeframe structure agrees with sweep direction"""
    if df_htf is None or len(df_htf) < 20:
        return False
    try:
        ma20 = df_htf['close'].rolling(20).mean().iloc[-1]
        ma50 = df_htf['close'].rolling(50).mean().iloc[-1]
        if direction == 'long':
            return price > ma20 or ma20 > ma50
        else:
            return price < ma20 or ma20 < ma50
    except Exception:
        return False


def _calc_bonus(strength, vol_spike, htf_conf, sweep_size):
    """Calculate score bonus (0-10 points)"""
    bonus = 0
    bonus += round(strength * 4)     # up to 4 pts — level strength
    if vol_spike:    bonus += 3      # 3 pts — volume confirmation
    if htf_conf:     bonus += 2      # 2 pts — HTF structure agrees
    if sweep_size > 1.5: bonus += 1  # 1 pt  — aggressive sweep
    return min(bonus, 10)


def get_sweep_summary(df_5min, df_1hour=None):
    """
    Convenience wrapper — returns clean summary dict for dashboard/logging.
    """
    result = detect_liquidity_sweep(df_5min, df_htf=df_1hour)
    if not result['detected']:
        return {'active': False}
    return {
        'active':    True,
        'direction': result['direction'],
        'level':     result['swept_level'],
        'sweep_atr': result['sweep_size'],
        'vol_spike': result['volume_spike'],
        'htf_ok':    result['htf_confluence'],
        'bonus_pts': result['score_bonus'],
        'reason':    result['reason'],
    }
