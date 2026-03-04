"""
APEX Strategy — Scalp Mode (strategy_scalp.py)
===============================================
Mode 1: Intraday Scalp (1min / 5min entries)

Best conditions:
  - London Prime 5-8am ET (10am-1pm UK)
  - NY Open 9:30-10am ET (2:30-3pm UK)
  - VIX 15-30 (enough volatility but not chaotic)
  - Trending market condition

Setup requirements:
  - HTF bias aligned (1hr + 4hr agree)
  - VWAP reclaim or rejection on 5min
  - Order block or FVG on 1min/5min
  - Volume spike confirmation
  - Liquidity sweep into setup

Risk profile:
  - Risk: 0.5-1% per trade
  - R:R: minimum 2:1, target 3:1
  - Hold time: 5-30 minutes
  - Max 3 scalps per session window

Scoring (0-100):
  - HTF bias:          20pts
  - VWAP confluence:   20pts
  - Order flow:        20pts
  - Volume:            15pts
  - Entry precision:   15pts
  - Session quality:   10pts
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

from vwap import get_vwap_result, calculate_volume_delta, score_vwap_confluence
from market_structure import get_market_condition, get_previous_day_levels


@dataclass
class ScalpSetup:
    direction:      str         # 'long' or 'short'
    score:          int         # 0-100
    entry:          float
    stop:           float
    target1:        float       # 2R
    target2:        float       # 3R
    risk_pct:       float       # Suggested risk %
    risk_pts:       float       # Points of risk
    rr_ratio:       float
    timeframe:      str
    reason:         str         # Setup description
    confluences:    List[str]   # What triggered
    warnings:       List[str]   # What to watch for
    details:        Dict


# =============================================================
#  MAIN SCALP SCANNER
# =============================================================

def scan_scalp(
    df_1min:  Optional[pd.DataFrame],
    df_5min:  Optional[pd.DataFrame],
    df_15min: Optional[pd.DataFrame],
    df_1hour: Optional[pd.DataFrame],
    df_4hour: Optional[pd.DataFrame],
    df_1day:  Optional[pd.DataFrame],
    symbol:   str = 'NQ',
    min_score: int = 60,
) -> Optional[ScalpSetup]:
    """
    Scan for scalp setups across timeframes.
    Returns best setup if score >= min_score, else None.
    """
    # Use 5min as primary entry timeframe
    entry_df = df_5min
    if entry_df is None or len(entry_df) < 50:
        return None

    current_price = float(entry_df['close'].iloc[-1])
    atr = _get_atr(entry_df)

    # Get HTF direction
    direction, htf_score, htf_details = _get_htf_direction(df_1hour, df_4hour, df_1day)
    if direction == 'neutral' or htf_score < 10:
        return None

    # Run all scoring layers
    score       = 0
    confluences = []
    warnings    = []
    details     = {}

    # 1. HTF Bias (20pts)
    htf_pts = min(20, htf_score)
    score  += htf_pts
    details['htf_score'] = htf_score
    details['htf_details'] = htf_details
    if htf_pts >= 15:
        confluences.append(f'Strong HTF {direction} bias')
    elif htf_pts >= 10:
        confluences.append(f'HTF {direction} bias')

    # 2. VWAP Confluence (20pts)
    vwap_score, vwap_det = score_vwap_confluence(df_5min, df_15min, direction)
    vwap_pts = min(20, vwap_score)
    score   += vwap_pts
    details['vwap'] = vwap_det
    if vwap_det.get('vwap_reclaim'):
        confluences.append('VWAP reclaim ✅')
    elif vwap_det.get('vwap_rejection'):
        confluences.append('VWAP rejection ✅')
    elif vwap_det.get('vwap_aligned'):
        confluences.append(f'Price {vwap_det.get("vwap_position","?")} VWAP')
    if vwap_det.get('delta_warning'):
        warnings.append('Volume delta divergence')

    # 3. Order Flow (20pts)
    of_score, of_det = _score_order_flow_scalp(df_5min, df_1min, direction)
    of_pts = min(20, of_score)
    score += of_pts
    details['order_flow'] = of_det
    if of_det.get('ob_hit'):
        confluences.append('Order block entry')
    if of_det.get('fvg_fill'):
        confluences.append('FVG fill')
    if of_det.get('sweep'):
        confluences.append('Liquidity sweep reversal')
    if of_det.get('bos'):
        confluences.append('BOS confirmed')

    # 4. Volume (15pts)
    vol_score, vol_det = _score_volume_scalp(entry_df, direction)
    vol_pts = min(15, vol_score)
    score  += vol_pts
    details['volume'] = vol_det
    if vol_det.get('spike'):
        confluences.append('Volume spike')
    if vol_det.get('absorption'):
        confluences.append('Absorption detected')

    # 5. Entry Precision (15pts)
    entry_score, entry_det = _score_entry_precision(entry_df, df_1min, direction, atr)
    entry_pts = min(15, entry_score)
    score    += entry_pts
    details['entry'] = entry_det
    if entry_det.get('pinbar'):
        confluences.append('Pin bar / rejection candle')
    if entry_det.get('engulf'):
        confluences.append('Engulfing candle')
    if entry_det.get('inside_break'):
        confluences.append('Inside bar breakout')

    # 6. Session Quality (10pts)
    sess_score = _score_session_scalp()
    score     += min(10, sess_score)

    # Check minimum score
    if score < min_score:
        return None

    # Calculate trade levels
    stop, stop_det = _calculate_scalp_stop(entry_df, df_1min, direction, atr)
    if stop is None:
        return None

    risk_pts = abs(current_price - stop)
    if risk_pts <= 0 or risk_pts > atr * 3:
        return None

    is_long  = direction in ('long', 'bullish')
    target1  = round(current_price + risk_pts * 2.0, 2) if is_long else round(current_price - risk_pts * 2.0, 2)
    target2  = round(current_price + risk_pts * 3.0, 2) if is_long else round(current_price - risk_pts * 3.0, 2)

    # Dynamic risk based on score
    if score >= 80:
        risk_pct = 1.0
    elif score >= 70:
        risk_pct = 0.75
    else:
        risk_pct = 0.5

    # Build reason string
    reason = f"{direction.upper()} Scalp — {', '.join(confluences[:3])}"

    return ScalpSetup(
        direction   = direction,
        score       = score,
        entry       = round(current_price, 2),
        stop        = round(stop, 2),
        target1     = target1,
        target2     = target2,
        risk_pct    = risk_pct,
        risk_pts    = round(risk_pts, 2),
        rr_ratio    = 3.0,
        timeframe   = '5min',
        reason      = reason,
        confluences = confluences,
        warnings    = warnings,
        details     = details,
    )


# =============================================================
#  SCORING HELPERS
# =============================================================

def _get_htf_direction(df_1h, df_4h, df_1d) -> Tuple[str, int, Dict]:
    """Get HTF bias from 1hr, 4hr, daily"""
    votes  = []
    score  = 0
    detail = {}

    for tf_name, df, weight in [('1hour', df_1h, 8), ('4hour', df_4h, 7), ('1day', df_1d, 5)]:
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

    if bull > bear:
        return 'long', score, detail
    elif bear > bull:
        return 'short', score, detail
    return 'neutral', 0, detail


def _score_order_flow_scalp(df_5min, df_1min, direction) -> Tuple[int, Dict]:
    """Score order flow for scalp setups"""
    score  = 0
    detail = {}
    is_long = direction in ('long', 'bullish')

    try:
        from order_flow import find_order_blocks, find_fair_value_gaps, find_liquidity_sweeps, detect_bos_choch

        df = df_5min
        if df is None or len(df) < 20:
            return 0, {}

        price = float(df['close'].iloc[-1])
        atr   = _get_atr(df)

        # Order blocks
        obs = find_order_blocks(df)
        for ob in obs[-5:]:
            if ob.direction == ('bullish' if is_long else 'bearish'):
                if ob.bottom <= price <= ob.top:
                    score += 8
                    detail['ob_hit'] = True
                    detail['ob_strength'] = ob.strength
                    break

        # FVGs
        fvgs = find_fair_value_gaps(df)
        for fvg in fvgs[-5:]:
            if fvg.direction == ('bullish' if is_long else 'bearish'):
                if fvg.bottom <= price <= fvg.top:
                    score += 6
                    detail['fvg_fill'] = True
                    break

        # Liquidity sweeps
        sweeps = find_liquidity_sweeps(df)
        for sw in sweeps[-3:]:
            if is_long and sw.direction == 'bull_sweep':
                score += 6
                detail['sweep'] = True
                break
            elif not is_long and sw.direction == 'bear_sweep':
                score += 6
                detail['sweep'] = True
                break

        # BOS
        structure = detect_bos_choch(df)
        for s in structure[-3:]:
            if is_long and 'bull' in s.kind and s.confirmed:
                score += 4
                detail['bos'] = True
                break
            elif not is_long and 'bear' in s.kind and s.confirmed:
                score += 4
                detail['bos'] = True
                break

    except Exception as e:
        detail['error'] = str(e)

    return min(20, score), detail


def _score_volume_scalp(df, direction) -> Tuple[int, Dict]:
    """Score volume for scalp confirmation"""
    score  = 0
    detail = {}

    try:
        if df is None or len(df) < 10:
            return 0, {}

        vol     = df['volume']
        vol_ma  = vol.rolling(20).mean()
        cur_vol = float(vol.iloc[-1])
        avg_vol = float(vol_ma.iloc[-1]) if not np.isnan(vol_ma.iloc[-1]) else cur_vol

        vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1
        detail['vol_ratio'] = round(vol_ratio, 2)

        if vol_ratio >= 2.0:
            score += 10
            detail['spike'] = True
        elif vol_ratio >= 1.5:
            score += 6
        elif vol_ratio >= 1.2:
            score += 3

        # Volume delta
        vd = calculate_volume_delta(df)
        if vd:
            is_long = direction in ('long', 'bullish')
            if is_long and vd.buying_pressure > 65:
                score += 5
                detail['strong_buying'] = True
            elif not is_long and vd.selling_pressure > 65:
                score += 5
                detail['strong_selling'] = True
            if vd.absorption:
                score += 3
                detail['absorption'] = True
            detail['buying_pct'] = vd.buying_pressure

    except Exception:
        pass

    return min(15, score), detail


def _score_entry_precision(df, df_1min, direction, atr) -> Tuple[int, Dict]:
    """Score entry candle quality"""
    score  = 0
    detail = {}

    try:
        if df is None or len(df) < 3:
            return 0, {}

        is_long = direction in ('long', 'bullish')
        bar = df.iloc[-1]
        o, h, l, c = float(bar['open']), float(bar['high']), float(bar['low']), float(bar['close'])
        bar_range = h - l if h > l else 0.0001

        # Pin bar / hammer
        body     = abs(c - o)
        upper_wk = h - max(c, o)
        lower_wk = min(c, o) - l

        if is_long and lower_wk > body * 2 and lower_wk > upper_wk * 2:
            score += 8
            detail['pinbar'] = True

        if not is_long and upper_wk > body * 2 and upper_wk > lower_wk * 2:
            score += 8
            detail['pinbar'] = True

        # Engulfing
        prev = df.iloc[-2]
        po, pc = float(prev['open']), float(prev['close'])
        if is_long and c > po and o < pc and (c - o) > (po - pc) * 1.2:
            score += 7
            detail['engulf'] = True
        if not is_long and c < po and o > pc and (o - c) > (pc - po) * 1.2:
            score += 7
            detail['engulf'] = True

        # Tight stop distance (smaller risk = better entry)
        if atr > 0:
            body_atr = body / atr
            if body_atr < 0.3:
                score += 5
                detail['tight_entry'] = True
            elif body_atr < 0.5:
                score += 3

    except Exception:
        pass

    return min(15, score), detail


def _score_session_scalp() -> int:
    """Score current session quality for scalping"""
    try:
        from strategy_config import is_tradeable_session
        from datetime import datetime, timezone
        from zoneinfo import ZoneInfo
        dt_ny = datetime.now(timezone.utc).astimezone(ZoneInfo('America/New_York'))
        tradeable, sess_name, quality = is_tradeable_session(dt_ny)
        if 'NY Open' in sess_name:
            return 10  # Best scalp session
        elif 'London' in sess_name:
            return 8
        return int(quality / 10)
    except Exception:
        return 5


def _calculate_scalp_stop(df, df_1min, direction, atr) -> Tuple[Optional[float], Dict]:
    """Calculate tight stop loss for scalp"""
    detail  = {}
    is_long = direction in ('long', 'bullish')

    try:
        if df is None or len(df) < 5:
            return None, {}

        price  = float(df['close'].iloc[-1])
        recent = df.tail(5)

        if is_long:
            # Stop below recent swing low or 1x ATR
            swing_low = float(recent['low'].min())
            atr_stop  = price - atr * 1.0
            stop      = max(swing_low - atr * 0.1, atr_stop)
            detail['method'] = 'swing_low'
        else:
            swing_high = float(recent['high'].max())
            atr_stop   = price + atr * 1.0
            stop       = min(swing_high + atr * 0.1, atr_stop)
            detail['method'] = 'swing_high'

        detail['stop'] = round(stop, 2)
        return round(stop, 2), detail

    except Exception:
        return None, {}


def _get_atr(df, period=14) -> float:
    """Calculate ATR"""
    try:
        h, l, c = df['high'], df['low'], df['close']
        tr  = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        atr = float(tr.ewm(span=period, adjust=False).mean().iloc[-1])
        return atr if atr > 0 else float(c.iloc[-1]) * 0.005
    except Exception:
        return float(df['close'].iloc[-1]) * 0.005
