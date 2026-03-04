"""
APEX Order Flow Engine — order_flow.py
========================================
Detects institutional price action concepts:
  - Order Blocks (OB) — origin of impulse moves
  - Break of Structure (BOS) — trend confirmation
  - Change of Character (CHoCH) — trend reversal signal
  - Fair Value Gaps (FVG) — imbalance zones
  - Liquidity Sweeps — stop hunts before reversal
  - Premium / Discount zones — optimal entry areas
  - OB + FVG confluence — highest probability setups
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# =============================================================
#  DATA STRUCTURES
# =============================================================

@dataclass
class OrderBlock:
    direction:    str        # 'bullish' or 'bearish'
    top:          float
    bottom:       float
    midpoint:     float
    origin_idx:   int
    origin_dt:    str
    strength:     int        # 1-100
    tested:       bool = False
    broken:       bool = False
    fvg_inside:   bool = False
    volume_ratio: float = 1.0

@dataclass
class FairValueGap:
    direction:   str         # 'bullish' or 'bearish'
    top:         float
    bottom:      float
    midpoint:    float
    size_pct:    float
    origin_idx:  int
    origin_dt:   str
    filled:      bool = False
    age_bars:    int = 0
    in_ob:       bool = False   # inside an order block

@dataclass
class StructurePoint:
    kind:        str         # 'BOS_bull', 'BOS_bear', 'CHoCH_bull', 'CHoCH_bear'
    price:       float
    bar_idx:     int
    bar_dt:      str
    confirmed:   bool = False
    strength:    int = 50

@dataclass
class LiquiditySweep:
    direction:   str         # 'bull_sweep' (swept lows) or 'bear_sweep' (swept highs)
    swept_level: float
    sweep_low:   float
    sweep_high:  float
    bar_idx:     int
    bar_dt:      str
    reversal_confirmed: bool = False
    reversal_strength:  int = 0


# =============================================================
#  ORDER BLOCK DETECTION
# =============================================================

def find_order_blocks(df, lookback=100, min_impulse_pct=0.3):
    """
    Order Block: The last opposing candle before a significant impulse move.

    Bullish OB: Last bearish candle before a strong bullish impulse
                (price that later breaks a swing high)
    Bearish OB: Last bullish candle before a strong bearish impulse
                (price that later breaks a swing low)

    The logic: institutions place large orders at these levels.
    When price returns to test them, there's often a strong reaction.
    """
    obs = []
    work = df.tail(lookback).copy()
    o = work['open'].values
    h = work['high'].values
    l = work['low'].values
    c = work['close'].values
    v = work['volume'].values if 'volume' in work.columns else np.ones(len(work))
    vol_ma = pd.Series(v).rolling(20).mean().values
    idx = work.index
    n = len(work)

    for i in range(3, n - 3):
        # Measure impulse move over next 3 bars
        forward_move_up   = max(h[i+1:i+4]) - c[i]
        forward_move_down = c[i] - min(l[i+1:i+4])
        bar_range = h[i] - l[i]
        if bar_range <= 0:
            continue
        impulse_pct = max(forward_move_up, forward_move_down) / c[i] * 100

        if impulse_pct < min_impulse_pct:
            continue

        vol_r = v[i] / vol_ma[i] if vol_ma[i] > 0 and not np.isnan(vol_ma[i]) else 1.0

        # Bullish OB — bearish candle followed by bullish impulse
        if c[i] < o[i] and forward_move_up > forward_move_down:
            # Confirm: next bars make a new swing high
            if max(h[i+1:min(i+6, n)]) > max(h[max(0,i-3):i]):
                strength = min(100, int(impulse_pct * 20 + vol_r * 10))
                obs.append(OrderBlock(
                    direction='bullish', top=o[i], bottom=l[i],
                    midpoint=(o[i]+l[i])/2,
                    origin_idx=i, origin_dt=str(idx[i]),
                    strength=strength, volume_ratio=round(vol_r, 2)
                ))

        # Bearish OB — bullish candle followed by bearish impulse
        elif c[i] > o[i] and forward_move_down > forward_move_up:
            if min(l[i+1:min(i+6, n)]) < min(l[max(0,i-3):i]):
                strength = min(100, int(impulse_pct * 20 + vol_r * 10))
                obs.append(OrderBlock(
                    direction='bearish', top=h[i], bottom=c[i],
                    midpoint=(h[i]+c[i])/2,
                    origin_idx=i, origin_dt=str(idx[i]),
                    strength=strength, volume_ratio=round(vol_r, 2)
                ))

    # Filter: remove OBs that have been clearly broken
    current_price = c[-1]
    active_obs = []
    for ob in obs:
        if ob.direction == 'bullish' and current_price > ob.bottom * 0.995:
            ob.tested = current_price <= ob.top * 1.005
            active_obs.append(ob)
        elif ob.direction == 'bearish' and current_price < ob.top * 1.005:
            ob.tested = current_price >= ob.bottom * 0.995
            active_obs.append(ob)

    # Sort by strength
    active_obs.sort(key=lambda x: x.strength, reverse=True)
    return active_obs[:8]


# =============================================================
#  BREAK OF STRUCTURE / CHANGE OF CHARACTER
# =============================================================

def find_structure(df, lookback=150):
    """
    Break of Structure (BOS): Price takes out a previous swing high/low
    in the direction of the current trend. Confirms continuation.

    Change of Character (CHoCH): Price breaks structure in the OPPOSITE
    direction to the recent trend. Early warning of reversal.
    """
    structure_points = []
    work = df.tail(lookback).copy()
    h = work['high'].values
    l = work['low'].values
    c = work['close'].values
    idx = work.index
    n = len(work)

    # Find swing highs and lows
    swing_highs = []
    swing_lows  = []
    for i in range(2, n-2):
        if h[i] > h[i-1] and h[i] > h[i-2] and h[i] > h[i+1] and h[i] > h[i+2]:
            swing_highs.append((i, h[i]))
        if l[i] < l[i-1] and l[i] < l[i-2] and l[i] < l[i+1] and l[i] < l[i+2]:
            swing_lows.append((i, l[i]))

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return structure_points, 'neutral'

    # Determine recent trend from last 3 swing highs/lows
    recent_sh = swing_highs[-3:]
    recent_sl = swing_lows[-3:]

    hh = all(recent_sh[i][1] > recent_sh[i-1][1] for i in range(1, len(recent_sh)))
    hl = all(recent_sl[i][1] > recent_sl[i-1][1] for i in range(1, len(recent_sl)))
    lh = all(recent_sh[i][1] < recent_sh[i-1][1] for i in range(1, len(recent_sh)))
    ll = all(recent_sl[i][1] < recent_sl[i-1][1] for i in range(1, len(recent_sl)))

    if hh and hl:
        trend = 'bullish'
    elif lh and ll:
        trend = 'bearish'
    else:
        trend = 'neutral'

    # Check last 20 bars for structure breaks
    for i in range(max(0, n-20), n):
        bar_close = c[i]

        # Check if this bar breaks above a recent swing high (BOS bull or CHoCH bull)
        for sh_idx, sh_price in swing_highs[-5:]:
            if sh_idx < i and bar_close > sh_price:
                kind = 'BOS_bull' if trend == 'bullish' else 'CHoCH_bull'
                strength = 80 if trend != 'bullish' else 60  # CHoCH is more significant
                structure_points.append(StructurePoint(
                    kind=kind, price=sh_price,
                    bar_idx=i, bar_dt=str(idx[i]),
                    confirmed=True, strength=strength
                ))
                break

        # Check if this bar breaks below a recent swing low
        for sl_idx, sl_price in swing_lows[-5:]:
            if sl_idx < i and bar_close < sl_price:
                kind = 'BOS_bear' if trend == 'bearish' else 'CHoCH_bear'
                strength = 80 if trend != 'bearish' else 60
                structure_points.append(StructurePoint(
                    kind=kind, price=sl_price,
                    bar_idx=i, bar_dt=str(idx[i]),
                    confirmed=True, strength=strength
                ))
                break

    return structure_points, trend


# =============================================================
#  FAIR VALUE GAP DETECTION
# =============================================================

def find_fvgs(df, lookback=100, min_gap_pct=0.05):
    """
    Fair Value Gap (FVG): A 3-candle pattern where the middle candle
    moves so fast that it creates an imbalance (gap) between candle 1
    and candle 3. Price tends to return to fill these gaps.

    Bullish FVG: candle[i-1].low > candle[i+1].high
    Bearish FVG: candle[i-1].high < candle[i+1].low
    """
    fvgs = []
    work = df.tail(lookback).copy()
    h = work['high'].values
    l = work['low'].values
    c = work['close'].values
    idx = work.index
    n = len(work)
    current = c[-1]

    for i in range(1, n-1):
        # Bullish FVG
        if l[i-1] > h[i+1]:
            top = l[i-1]; bot = h[i+1]
            size_pct = (top-bot)/c[i]*100
            if size_pct >= min_gap_pct:
                # Check if still unfilled
                min_since = min(l[i+1:]) if i+1 < n else bot
                filled = min_since <= bot
                age = n - i
                fvgs.append(FairValueGap(
                    direction='bullish', top=round(top,2), bottom=round(bot,2),
                    midpoint=round((top+bot)/2,2), size_pct=round(size_pct,3),
                    origin_idx=i, origin_dt=str(idx[i]),
                    filled=filled, age_bars=age
                ))

        # Bearish FVG
        if h[i-1] < l[i+1]:
            top = l[i+1]; bot = h[i-1]
            size_pct = (top-bot)/c[i]*100
            if size_pct >= min_gap_pct:
                max_since = max(h[i+1:]) if i+1 < n else top
                filled = max_since >= top
                age = n - i
                fvgs.append(FairValueGap(
                    direction='bearish', top=round(top,2), bottom=round(bot,2),
                    midpoint=round((top+bot)/2,2), size_pct=round(size_pct,3),
                    origin_idx=i, origin_dt=str(idx[i]),
                    filled=filled, age_bars=age
                ))

    # Keep only unfilled, return sorted by proximity to current price
    active = [f for f in fvgs if not f.filled]
    active.sort(key=lambda x: abs(x.midpoint - current))
    return active[:6]


# =============================================================
#  LIQUIDITY SWEEP DETECTION
# =============================================================

def find_liquidity_sweeps(df, lookback=60, min_wick_pct=0.15):
    """
    Liquidity Sweep: Price briefly penetrates a swing high/low
    (taking out stop losses) then reverses sharply.

    Signature: Long wick that exceeds a swing level but closes back
    inside the range. Often precedes a strong move in the opposite direction.
    """
    sweeps = []
    work = df.tail(lookback).copy()
    o = work['open'].values
    h = work['high'].values
    l = work['low'].values
    c = work['close'].values
    idx = work.index
    n = len(work)

    # Find recent swing levels
    swing_highs = []
    swing_lows  = []
    for i in range(2, n-2):
        if h[i] > h[i-1] and h[i] > h[i-2] and h[i] > h[i+1] and h[i] > h[i+2]:
            swing_highs.append((i, h[i]))
        if l[i] < l[i-1] and l[i] < l[i-2] and l[i] < l[i+1] and l[i] < l[i+2]:
            swing_lows.append((i, l[i]))

    for i in range(5, n-1):
        bar_range = h[i] - l[i]
        if bar_range <= 0:
            continue

        upper_wick = h[i] - max(o[i], c[i])
        lower_wick = min(o[i], c[i]) - l[i]
        body = abs(c[i] - o[i])
        wick_pct = upper_wick / bar_range

        # Bear sweep: long upper wick sweeping above swing high, closing back below
        for sh_idx, sh_price in swing_highs:
            if sh_idx < i - 2 and h[i] > sh_price and c[i] < sh_price:
                wick_ratio = upper_wick / bar_range if bar_range > 0 else 0
                if wick_ratio >= min_wick_pct:
                    # Check for reversal in next 3 bars
                    reversal = i + 1 < n and c[i+1] < c[i]
                    rev_strength = int(wick_ratio * 100)
                    sweeps.append(LiquiditySweep(
                        direction='bear_sweep',
                        swept_level=sh_price,
                        sweep_low=l[i], sweep_high=h[i],
                        bar_idx=i, bar_dt=str(idx[i]),
                        reversal_confirmed=reversal,
                        reversal_strength=rev_strength
                    ))
                    break

        # Bull sweep: long lower wick sweeping below swing low, closing back above
        for sl_idx, sl_price in swing_lows:
            if sl_idx < i - 2 and l[i] < sl_price and c[i] > sl_price:
                wick_ratio = lower_wick / bar_range if bar_range > 0 else 0
                if wick_ratio >= min_wick_pct:
                    reversal = i + 1 < n and c[i+1] > c[i]
                    rev_strength = int(wick_ratio * 100)
                    sweeps.append(LiquiditySweep(
                        direction='bull_sweep',
                        swept_level=sl_price,
                        sweep_low=l[i], sweep_high=h[i],
                        bar_idx=i, bar_dt=str(idx[i]),
                        reversal_confirmed=reversal,
                        reversal_strength=rev_strength
                    ))
                    break

    # Return most recent sweeps
    return sorted(sweeps, key=lambda x: x.bar_idx, reverse=True)[:4]


# =============================================================
#  PREMIUM / DISCOUNT ZONES
# =============================================================

def get_premium_discount(df, lookback=50):
    """
    Premium/Discount framework:
    - Find the swing high and swing low of the current range
    - Equilibrium = 50% of the range
    - Discount zone = below 50% (good for longs)
    - Premium zone = above 50% (good for shorts)
    - Optimal entry zones = 25% (deep discount) and 75% (deep premium)
    """
    work = df.tail(lookback)
    range_high = float(work['high'].max())
    range_low  = float(work['low'].min())
    current    = float(work['close'].iloc[-1])
    range_size = range_high - range_low

    if range_size <= 0:
        return {}

    equilibrium  = range_low + range_size * 0.5
    discount_50  = range_low + range_size * 0.5
    discount_25  = range_low + range_size * 0.25
    premium_75   = range_low + range_size * 0.75

    position_pct = (current - range_low) / range_size * 100

    if position_pct <= 25:
        zone = 'deep_discount'
        bias = 'strong_long'
    elif position_pct <= 50:
        zone = 'discount'
        bias = 'long_favoured'
    elif position_pct <= 75:
        zone = 'premium'
        bias = 'short_favoured'
    else:
        zone = 'deep_premium'
        bias = 'strong_short'

    return {
        'range_high':    round(range_high, 2),
        'range_low':     round(range_low, 2),
        'equilibrium':   round(equilibrium, 2),
        'discount_25':   round(discount_25, 2),
        'premium_75':    round(premium_75, 2),
        'current':       round(current, 2),
        'position_pct':  round(position_pct, 1),
        'zone':          zone,
        'bias':          bias,
    }


# =============================================================
#  OB + FVG CONFLUENCE
# =============================================================

def find_ob_fvg_confluence(obs, fvgs, tolerance_pct=0.3):
    """
    Find where Order Blocks and FVGs overlap.
    This is the highest probability setup — institutional order flow
    AND price imbalance at the same level.
    """
    confluences = []
    for ob in obs:
        for fvg in fvgs:
            if ob.direction != fvg.direction:
                continue
            # Check if FVG overlaps with OB
            ob_top = ob.top; ob_bot = ob.bottom
            fvg_top = fvg.top; fvg_bot = fvg.bottom
            tol = ob_top * tolerance_pct / 100
            if (fvg_bot <= ob_top + tol and fvg_top >= ob_bot - tol):
                ob.fvg_inside = True
                fvg.in_ob = True
                overlap_top = min(ob_top, fvg_top)
                overlap_bot = max(ob_bot, fvg_bot)
                confluences.append({
                    'direction':   ob.direction,
                    'top':         round(overlap_top, 2),
                    'bottom':      round(overlap_bot, 2),
                    'midpoint':    round((overlap_top+overlap_bot)/2, 2),
                    'ob_strength': ob.strength,
                    'fvg_size':    fvg.size_pct,
                    'fvg_age':     fvg.age_bars,
                    'score':       min(100, ob.strength + int(fvg.size_pct * 10) + 20),
                })

    confluences.sort(key=lambda x: x['score'], reverse=True)
    return confluences


# =============================================================
#  FULL ORDER FLOW ANALYSIS
# =============================================================

def analyse_order_flow(df, timeframe='5min'):
    """
    Run complete order flow analysis on a dataframe.
    Returns a structured dict with all findings.
    """
    if df is None or len(df) < 50:
        return {'error': 'Insufficient data', 'score': 0}

    obs      = find_order_blocks(df)
    fvgs     = find_fvgs(df)
    struct, trend = find_structure(df)
    sweeps   = find_liquidity_sweeps(df)
    pd_zones = get_premium_discount(df)
    confluences = find_ob_fvg_confluence(obs, fvgs)

    current = float(df['close'].iloc[-1])

    # Find nearest actionable levels
    nearest_bull_ob = next((ob for ob in obs if ob.direction=='bullish' and ob.top >= current*0.99), None)
    nearest_bear_ob = next((ob for ob in obs if ob.direction=='bearish' and ob.bottom <= current*1.01), None)
    nearest_bull_fvg = next((f for f in fvgs if f.direction=='bullish' and not f.filled), None)
    nearest_bear_fvg = next((f for f in fvgs if f.direction=='bearish' and not f.filled), None)

    # Recent structure events
    recent_bos   = [s for s in struct if 'BOS' in s.kind][-3:]
    recent_choch = [s for s in struct if 'CHoCH' in s.kind][-2:]
    recent_sweep = sweeps[:2]

    # Overall order flow score
    of_score = 50
    zone = pd_zones.get('zone', '')
    bias = pd_zones.get('bias', '')

    if 'discount' in zone:   of_score += 10
    if 'premium' in zone:    of_score -= 10
    if trend == 'bullish':   of_score += 15
    elif trend == 'bearish': of_score -= 15
    if recent_choch and 'bull' in recent_choch[-1].kind: of_score += 20
    if recent_choch and 'bear' in recent_choch[-1].kind: of_score -= 20
    if confluences: of_score += 15 if confluences[0]['direction'] == 'bullish' else -15
    if recent_sweep and recent_sweep[0].reversal_confirmed:
        if recent_sweep[0].direction == 'bull_sweep': of_score += 10
        else: of_score -= 10

    of_score = max(0, min(100, of_score))

    return {
        'timeframe':      timeframe,
        'current_price':  round(current, 2),
        'trend':          trend,
        'of_score':       round(of_score, 1),
        'pd_zone':        pd_zones,
        'order_blocks':   [{'direction': ob.direction, 'top': ob.top, 'bottom': ob.bottom,
                            'midpoint': ob.midpoint, 'strength': ob.strength,
                            'tested': ob.tested, 'fvg_inside': ob.fvg_inside,
                            'volume_ratio': ob.volume_ratio, 'dt': ob.origin_dt}
                           for ob in obs[:5]],
        'fvgs':           [{'direction': f.direction, 'top': f.top, 'bottom': f.bottom,
                            'midpoint': f.midpoint, 'size_pct': f.size_pct,
                            'age_bars': f.age_bars, 'in_ob': f.in_ob, 'dt': f.origin_dt}
                           for f in fvgs[:5]],
        'confluences':    confluences[:3],
        'structure': {
            'trend':       trend,
            'recent_bos':  [{'kind': s.kind, 'price': s.price, 'dt': s.bar_dt,
                             'strength': s.strength} for s in recent_bos],
            'recent_choch':[{'kind': s.kind, 'price': s.price, 'dt': s.bar_dt,
                             'strength': s.strength} for s in recent_choch],
        },
        'liquidity_sweeps': [{'direction': s.direction, 'swept': s.swept_level,
                               'reversal': s.reversal_confirmed,
                               'strength': s.reversal_strength, 'dt': s.bar_dt}
                             for s in recent_sweep],
        'nearest_levels': {
            'bull_ob':  {'top': nearest_bull_ob.top, 'bottom': nearest_bull_ob.bottom} if nearest_bull_ob else None,
            'bear_ob':  {'top': nearest_bear_ob.top, 'bottom': nearest_bear_ob.bottom} if nearest_bear_ob else None,
            'bull_fvg': {'top': nearest_bull_fvg.top, 'bottom': nearest_bull_fvg.bottom} if nearest_bull_fvg else None,
            'bear_fvg': {'top': nearest_bear_fvg.top, 'bottom': nearest_bear_fvg.bottom} if nearest_bear_fvg else None,
        },
    }
