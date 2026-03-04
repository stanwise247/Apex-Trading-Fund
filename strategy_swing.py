"""
APEX Strategy — Intraday Swing Mode (strategy_swing.py)
========================================================
Mode 2: Intraday Swing (15min / 1hour entries)

Best conditions:
  - London Prime 5-8am ET OR NY Morning 9:30-11:30am ET
  - VIX 12-30
  - Trending or transitioning market
  - Hold time 1-4 hours, can hold overnight

Setup requirements:
  - Strong HTF alignment (4hr + daily + weekly agree)
  - OB + FVG confluence on 15min
  - BOS or CHoCH confirmation
  - VWAP position aligned
  - Previous day level interaction
  - Volume confirmation

Risk profile:
  - Risk: 1.5-2% per trade
  - R:R: minimum 2.5:1, target 3-4:1
  - Partial exit at 1.5R, move stop to BE
  - Hold to target or structure break

Scoring (0-100):
  - HTF bias:              25pts
  - Order flow:            25pts
  - VWAP + Volume:         20pts
  - Market structure:      20pts
  - Entry signal:          10pts
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

from vwap import score_vwap_confluence
from market_structure import score_market_structure, get_market_condition


@dataclass
class SwingSetup:
    direction:      str
    score:          int
    entry:          float
    stop:           float
    target1:        float       # 1.5R partial exit
    target2:        float       # 3R full exit
    target3:        float       # 4R extended target
    risk_pct:       float
    risk_pts:       float
    rr_ratio:       float
    timeframe:      str
    hold_overnight: bool
    reason:         str
    confluences:    List[str]
    warnings:       List[str]
    details:        Dict


# =============================================================
#  MAIN SWING SCANNER
# =============================================================

def scan_swing(
    df_5min:  Optional[pd.DataFrame],
    df_15min: Optional[pd.DataFrame],
    df_1hour: Optional[pd.DataFrame],
    df_4hour: Optional[pd.DataFrame],
    df_1day:  Optional[pd.DataFrame],
    df_1week: Optional[pd.DataFrame],
    symbol:   str = 'NQ',
    min_score: int = 60,
) -> Optional[SwingSetup]:
    """Scan for intraday swing setups"""

    entry_df = df_15min or df_1hour
    if entry_df is None or len(entry_df) < 50:
        return None

    current_price = float(entry_df['close'].iloc[-1])
    atr = _get_atr(entry_df)

    # Require strong HTF alignment for swing trades
    direction, htf_score, htf_det = _get_htf_direction_swing(
        df_1hour, df_4hour, df_1day, df_1week
    )
    if direction == 'neutral' or htf_score < 15:
        return None

    score       = 0
    confluences = []
    warnings    = []
    details     = {}

    # 1. HTF Bias (25pts) — more weight than scalp
    htf_pts = min(25, int(htf_score * 1.25))
    score  += htf_pts
    details['htf'] = htf_det
    if htf_pts >= 20:
        confluences.append(f'Strong multi-TF {direction} confluence')
    elif htf_pts >= 15:
        confluences.append(f'HTF {direction} bias')

    # 2. Order Flow (25pts)
    of_score, of_det = _score_order_flow_swing(df_15min, df_5min, direction)
    of_pts = min(25, of_score)
    score += of_pts
    details['order_flow'] = of_det

    if of_det.get('ob_fvg_confluence'):
        confluences.append('OB + FVG confluence 🎯')
    elif of_det.get('ob_hit'):
        confluences.append('Order block entry')
    if of_det.get('bos'):
        confluences.append('BOS confirmed')
    if of_det.get('choch'):
        confluences.append('CHoCH — trend change')
    if of_det.get('sweep'):
        confluences.append('Liquidity sweep reversal')
    if of_det.get('premium_discount'):
        confluences.append(f'{"Discount" if direction in ("long","bullish") else "Premium"} zone')

    # 3. VWAP + Volume Profile (20pts)
    vwap_score, vwap_det = score_vwap_confluence(df_5min, df_15min, direction)
    vwap_pts = min(20, vwap_score)
    score   += vwap_pts
    details['vwap'] = vwap_det

    if vwap_det.get('vwap_reclaim'):
        confluences.append('VWAP reclaim')
    if vwap_det.get('at_val_support') or vwap_det.get('at_vah_resistance'):
        confluences.append('Volume profile level')
    if vwap_det.get('above_poc' if direction in ('long','bullish') else 'below_poc'):
        confluences.append('POC aligned')
    if vwap_det.get('delta_warning'):
        warnings.append('Delta divergence — reduce size')

    # 4. Market Structure (20pts)
    ms_score, ms_det = score_market_structure(
        df_5min, df_15min, df_1hour, df_1day, df_1week, direction
    )
    ms_pts = min(20, ms_score)
    score += ms_pts
    details['market_structure'] = ms_det

    if ms_det.get('breaking_pdh') or ms_det.get('breaking_pdl'):
        confluences.append('Breaking previous day level 🔥')
    if ms_det.get('at_golden_pocket'):
        confluences.append('Fibonacci golden pocket (61.8%)')
    if ms_det.get('orb_long') or ms_det.get('orb_short'):
        confluences.append('Opening range breakout')
    if ms_det.get('equal_lows_sweep') or ms_det.get('equal_highs_sweep'):
        confluences.append('Equal levels swept')
    if ms_det.get('volatile_warning'):
        warnings.append('High volatility — use wider stop')

    # 5. Entry Signal (10pts)
    entry_score, entry_det = _score_entry_swing(entry_df, direction)
    entry_pts = min(10, entry_score)
    score    += entry_pts
    details['entry'] = entry_det

    if entry_det.get('rejection_candle'):
        confluences.append('Strong rejection candle')
    if entry_det.get('confirmed_close'):
        confluences.append('Bar closed past level')

    # Market condition check
    mc = get_market_condition(df_15min or df_1hour)
    details['market_condition'] = mc.condition
    if mc.condition == 'volatile':
        warnings.append('Volatile conditions — halve position size')
        score = int(score * 0.85)

    if score < min_score:
        return None

    # Calculate trade levels
    stop = _calculate_swing_stop(entry_df, df_15min, direction, atr)
    if stop is None:
        return None

    risk_pts = abs(current_price - stop)
    if risk_pts <= 0 or risk_pts > atr * 5:
        return None

    is_long = direction in ('long', 'bullish')
    t1 = round(current_price + risk_pts * 1.5, 2) if is_long else round(current_price - risk_pts * 1.5, 2)
    t2 = round(current_price + risk_pts * 3.0, 2) if is_long else round(current_price - risk_pts * 3.0, 2)
    t3 = round(current_price + risk_pts * 4.0, 2) if is_long else round(current_price - risk_pts * 4.0, 2)

    # Dynamic risk based on score and conditions
    if score >= 85 and mc.condition not in ('volatile',):
        risk_pct = 2.0
    elif score >= 75:
        risk_pct = 1.5
    else:
        risk_pct = 1.0

    if mc.condition == 'volatile':
        risk_pct = risk_pct * 0.5

    reason = f"{direction.upper()} Swing — {', '.join(confluences[:3])}"

    # Should we hold overnight?
    hold_overnight = htf_score >= 20 and score >= 75

    return SwingSetup(
        direction      = direction,
        score          = score,
        entry          = round(current_price, 2),
        stop           = round(stop, 2),
        target1        = t1,
        target2        = t2,
        target3        = t3,
        risk_pct       = risk_pct,
        risk_pts       = round(risk_pts, 2),
        rr_ratio       = 3.0,
        timeframe      = '15min',
        hold_overnight = hold_overnight,
        reason         = reason,
        confluences    = confluences,
        warnings       = warnings,
        details        = details,
    )


# =============================================================
#  SCORING HELPERS
# =============================================================

def _get_htf_direction_swing(df_1h, df_4h, df_1d, df_1w) -> Tuple[str, int, Dict]:
    """HTF direction for swing — requires more alignment than scalp"""
    votes  = []
    score  = 0
    detail = {}

    weights = [('1hour', df_1h, 5), ('4hour', df_4h, 8),
               ('1day', df_1d, 9), ('1week', df_1w, 8)]

    for tf_name, df, weight in weights:
        if df is None or len(df) < 20:
            continue
        c   = df['close']
        e20 = float(c.ewm(span=20, adjust=False).mean().iloc[-1])
        e50 = float(c.ewm(span=50, adjust=False).mean().iloc[-1])
        cur = float(c.iloc[-1])

        if cur > e20 > e50:
            votes.append('bullish')
            score += weight
            detail[tf_name] = 'bullish'
        elif cur < e20 < e50:
            votes.append('bearish')
            score += weight
            detail[tf_name] = 'bearish'
        else:
            detail[tf_name] = 'neutral'

    if not votes:
        return 'neutral', 0, detail

    bull = votes.count('bullish')
    bear = votes.count('bearish')

    # Swing requires stronger consensus than scalp
    if bull >= 3 and bull > bear:
        return 'long', score, detail
    elif bear >= 3 and bear > bull:
        return 'short', score, detail
    elif bull == 2 and bull > bear:
        return 'long', int(score * 0.7), detail
    elif bear == 2 and bear > bull:
        return 'short', int(score * 0.7), detail

    return 'neutral', 0, detail


def _score_order_flow_swing(df_15min, df_5min, direction) -> Tuple[int, Dict]:
    """Order flow scoring for swing — heavier weight on confluence"""
    score   = 0
    detail  = {}
    is_long = direction in ('long', 'bullish')

    try:
        from order_flow import (find_order_blocks, find_fair_value_gaps,
                                find_liquidity_sweeps, detect_bos_choch,
                                find_premium_discount)

        df = df_15min
        if df is None or len(df) < 20:
            return 0, {}

        price = float(df['close'].iloc[-1])
        atr   = _get_atr(df)

        # Order blocks
        obs = find_order_blocks(df)
        ob_hit = False
        for ob in obs[-8:]:
            if ob.direction == ('bullish' if is_long else 'bearish'):
                if ob.bottom <= price <= ob.top:
                    ob_hit  = True
                    score  += 8
                    detail['ob_hit']      = True
                    detail['ob_strength'] = ob.strength
                    break

        # FVGs
        fvgs   = find_fair_value_gaps(df)
        fvg_hit = False
        for fvg in fvgs[-8:]:
            if fvg.direction == ('bullish' if is_long else 'bearish'):
                if fvg.bottom <= price <= fvg.top:
                    fvg_hit = True
                    score  += 7
                    detail['fvg_fill'] = True
                    break

        # OB + FVG confluence = premium setup
        if ob_hit and fvg_hit:
            score += 8
            detail['ob_fvg_confluence'] = True

        # Liquidity sweeps
        sweeps = find_liquidity_sweeps(df)
        for sw in sweeps[-5:]:
            if is_long and sw.direction == 'bull_sweep' and sw.reversal_confirmed:
                score += 6
                detail['sweep'] = True
                break
            elif not is_long and sw.direction == 'bear_sweep' and sw.reversal_confirmed:
                score += 6
                detail['sweep'] = True
                break

        # BOS / CHoCH
        structure = detect_bos_choch(df)
        for s in structure[-5:]:
            if is_long and 'bull' in s.kind:
                if 'BOS' in s.kind and s.confirmed:
                    score += 5
                    detail['bos'] = True
                elif 'CHoCH' in s.kind:
                    score += 4
                    detail['choch'] = True
                break
            elif not is_long and 'bear' in s.kind:
                if 'BOS' in s.kind and s.confirmed:
                    score += 5
                    detail['bos'] = True
                elif 'CHoCH' in s.kind:
                    score += 4
                    detail['choch'] = True
                break

        # Premium / discount zone
        try:
            pd_result = find_premium_discount(df)
            if pd_result:
                zone = pd_result.get('zone', '')
                if is_long and zone == 'discount':
                    score += 4
                    detail['premium_discount'] = True
                elif not is_long and zone == 'premium':
                    score += 4
                    detail['premium_discount'] = True
        except Exception:
            pass

    except Exception as e:
        detail['error'] = str(e)

    return min(25, score), detail


def _score_entry_swing(df, direction) -> Tuple[int, Dict]:
    """Entry candle quality for swing"""
    score   = 0
    detail  = {}
    is_long = direction in ('long', 'bullish')

    try:
        if df is None or len(df) < 3:
            return 0, {}

        bar = df.iloc[-1]
        o, h, l, c = float(bar['open']), float(bar['high']), float(bar['low']), float(bar['close'])
        bar_range = h - l if h > l else 0.0001
        body      = abs(c - o)
        upper_wk  = h - max(c, o)
        lower_wk  = min(c, o) - l

        # Strong rejection candle
        if is_long and lower_wk > body * 1.5:
            score += 6
            detail['rejection_candle'] = True
        elif not is_long and upper_wk > body * 1.5:
            score += 6
            detail['rejection_candle'] = True

        # Closed in correct direction (confirmation)
        if is_long and c > o:
            score += 4
            detail['confirmed_close'] = True
        elif not is_long and c < o:
            score += 4
            detail['confirmed_close'] = True

    except Exception:
        pass

    return min(10, score), detail


def _calculate_swing_stop(df, df_15min, direction, atr) -> Optional[float]:
    """Calculate swing stop — wider than scalp, below/above structure"""
    is_long = direction in ('long', 'bullish')
    try:
        ref_df = df_15min or df
        if ref_df is None or len(ref_df) < 10:
            return None

        price  = float(ref_df['close'].iloc[-1])
        recent = ref_df.tail(10)

        if is_long:
            swing_low = float(recent['low'].min())
            stop      = swing_low - atr * 0.2
        else:
            swing_high = float(recent['high'].max())
            stop       = swing_high + atr * 0.2

        return round(stop, 2)
    except Exception:
        return None


def _get_atr(df, period=14) -> float:
    try:
        h, l, c = df['high'], df['low'], df['close']
        tr  = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        atr = float(tr.ewm(span=period, adjust=False).mean().iloc[-1])
        return atr if atr > 0 else float(c.iloc[-1]) * 0.005
    except Exception:
        return float(df['close'].iloc[-1]) * 0.005
