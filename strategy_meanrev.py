"""
APEX Strategy — Mean Reversion Mode (strategy_meanrev.py)
==========================================================
Mode 3: Mean Reversion (5min / 15min entries)

Best conditions:
  - Ranging market (ADX < 20)
  - VIX below 20 (calm, range-bound)
  - Price extended beyond 2σ from VWAP
  - Midday or low-volume sessions

Setup requirements:
  - Price at 2σ+ deviation from VWAP
  - RSI extreme (>70 short, <30 long)
  - Volume divergence (big move on low volume)
  - At key support/resistance level
  - No HTF trend fighting against you

Risk profile:
  - Risk: 0.75-1% per trade (mean rev = lower conviction)
  - R:R: minimum 1.5:1 (tighter targets)
  - Hold time: 15-90 minutes
  - Tight stop — invalidated quickly if wrong

Scoring (0-100):
  - VWAP extension:      30pts
  - RSI extreme:         20pts
  - Volume divergence:   20pts
  - Level confluence:    20pts
  - Market condition:    10pts
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

from vwap import get_vwap_result, calculate_volume_delta


@dataclass
class MeanRevSetup:
    direction:   str
    score:       int
    entry:       float
    stop:        float
    target1:     float       # VWAP (mean)
    target2:     float       # Opposite band
    risk_pct:    float
    risk_pts:    float
    rr_ratio:    float
    timeframe:   str
    reason:      str
    confluences: List[str]
    warnings:    List[str]
    details:     Dict


def scan_meanrev(
    df_5min:  Optional[pd.DataFrame],
    df_15min: Optional[pd.DataFrame],
    df_1hour: Optional[pd.DataFrame],
    symbol:   str = 'NQ',
    min_score: int = 65,
) -> Optional[MeanRevSetup]:
    """Scan for mean reversion setups"""

    df = df_5min or df_15min
    if df is None or len(df) < 50:
        return None

    # Mean reversion only works in ranging markets
    from market_structure import get_market_condition
    mc = get_market_condition(df)
    if mc.condition in ('trending_up', 'trending_down') and mc.trend_strength > 35:
        return None  # Don't fade strong trends

    current_price = float(df['close'].iloc[-1])
    atr = _get_atr(df)

    score       = 0
    confluences = []
    warnings    = []
    details     = {}
    direction   = None

    # 1. VWAP Extension (30pts) — core signal
    vwap_result = get_vwap_result(df)
    if vwap_result is None:
        return None

    details['vwap'] = {
        'value':     vwap_result.vwap,
        'deviation': vwap_result.deviation,
        'position':  vwap_result.position,
    }

    dev = vwap_result.deviation
    if dev <= -2.0:
        direction = 'long'   # Extremely below VWAP — buy the dip
        if dev <= -3.0:
            score += 30
            confluences.append(f'Price at -3σ VWAP extreme 🔥')
        else:
            score += 20
            confluences.append(f'Price at -2σ VWAP ({dev:.1f}σ)')
    elif dev >= 2.0:
        direction = 'short'  # Extremely above VWAP — sell the rip
        if dev >= 3.0:
            score += 30
            confluences.append(f'Price at +3σ VWAP extreme 🔥')
        else:
            score += 20
            confluences.append(f'Price at +2σ VWAP ({dev:.1f}σ)')
    else:
        return None  # Not extended enough for mean reversion

    is_long = direction == 'long'

    # 2. RSI Extreme (20pts)
    rsi_score, rsi_val = _score_rsi(df, is_long)
    score += min(20, rsi_score)
    details['rsi'] = rsi_val
    if rsi_score >= 15:
        confluences.append(f'RSI extreme ({rsi_val:.0f})')

    # 3. Volume Divergence (20pts)
    vd_score, vd_det = _score_volume_divergence(df, is_long)
    score += min(20, vd_score)
    details['volume_delta'] = vd_det
    if vd_det.get('divergence'):
        confluences.append('Volume divergence — weak move')
    if vd_det.get('low_volume_extreme'):
        confluences.append('Low volume at extreme')

    # 4. Level Confluence (20pts)
    level_score, level_det = _score_level_confluence(df, df_15min, current_price, is_long)
    score += min(20, level_score)
    details['levels'] = level_det
    if level_det.get('at_support') or level_det.get('at_resistance'):
        confluences.append('Key S/R level')
    if level_det.get('bb_extreme'):
        confluences.append('Bollinger band extreme')

    # 5. Market Condition (10pts)
    if mc.condition == 'ranging':
        score += 10
        confluences.append('Ranging market ✅')
    elif mc.condition == 'transitioning':
        score += 5
    elif mc.condition == 'volatile':
        warnings.append('Volatile market — avoid mean reversion')
        return None

    details['market_condition'] = mc.condition

    if score < min_score:
        return None

    # Trade levels
    stop, target1, target2 = _calculate_meanrev_levels(
        current_price, vwap_result, is_long, atr
    )

    risk_pts = abs(current_price - stop)
    if risk_pts <= 0:
        return None

    rr = abs(target1 - current_price) / risk_pts

    # Mean reversion gets lower risk
    risk_pct = 0.75 if score >= 80 else 0.5

    reason = f"{direction.upper()} Mean Reversion — {', '.join(confluences[:2])}"

    return MeanRevSetup(
        direction   = direction,
        score       = score,
        entry       = round(current_price, 2),
        stop        = round(stop, 2),
        target1     = round(target1, 2),
        target2     = round(target2, 2),
        risk_pct    = risk_pct,
        risk_pts    = round(risk_pts, 2),
        rr_ratio    = round(rr, 2),
        timeframe   = '5min',
        reason      = reason,
        confluences = confluences,
        warnings    = warnings,
        details     = details,
    )


def _score_rsi(df, is_long) -> Tuple[int, float]:
    try:
        c     = df['close']
        delta = c.diff()
        gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        loss  = (-delta).clip(lower=0).ewm(com=13, adjust=False).mean()
        rsi   = float((100 - 100 / (1 + gain / loss.replace(0, np.nan))).iloc[-1])

        if is_long:
            if rsi < 20: return 20, rsi
            if rsi < 30: return 15, rsi
            if rsi < 40: return 8,  rsi
        else:
            if rsi > 80: return 20, rsi
            if rsi > 70: return 15, rsi
            if rsi > 60: return 8,  rsi
        return 0, rsi
    except Exception:
        return 0, 50.0


def _score_volume_divergence(df, is_long) -> Tuple[int, Dict]:
    score  = 0
    detail = {}
    try:
        vd = calculate_volume_delta(df)
        if vd:
            detail['buying_pct']  = vd.buying_pressure
            detail['divergence']  = vd.delta_divergence

            # Divergence: price extended but volume not confirming
            if vd.delta_divergence:
                score += 15
                detail['divergence'] = True

            # Low volume at extreme = exhaustion
            vol     = df['volume']
            vol_ma  = vol.rolling(20).mean()
            cur_vol = float(vol.iloc[-1])
            avg_vol = float(vol_ma.iloc[-1]) if not np.isnan(vol_ma.iloc[-1]) else cur_vol
            if avg_vol > 0 and cur_vol < avg_vol * 0.7:
                score += 10
                detail['low_volume_extreme'] = True

            # Selling pressure at top / buying at bottom
            if not is_long and vd.selling_pressure > 60:
                score += 5
            elif is_long and vd.buying_pressure > 60:
                score += 5

    except Exception:
        pass
    return min(20, score), detail


def _score_level_confluence(df, df_15min, price, is_long) -> Tuple[int, Dict]:
    score  = 0
    detail = {}
    try:
        # Bollinger bands
        c      = df['close']
        bb_mid = c.rolling(20).mean()
        bb_std = c.rolling(20).std()
        bb_up  = float((bb_mid + 2*bb_std).iloc[-1])
        bb_dn  = float((bb_mid - 2*bb_std).iloc[-1])

        if is_long and price <= bb_dn * 1.001:
            score += 10
            detail['bb_extreme'] = True
        elif not is_long and price >= bb_up * 0.999:
            score += 10
            detail['bb_extreme'] = True

        # Simple support/resistance from recent swing levels
        ref = df_15min or df
        if ref is not None and len(ref) >= 20:
            recent_highs = ref['high'].tail(50)
            recent_lows  = ref['low'].tail(50)

            # Find clustering levels
            resistance = float(recent_highs.quantile(0.9))
            support    = float(recent_lows.quantile(0.1))

            if is_long and abs(price - support) / price < 0.002:
                score += 10
                detail['at_support'] = round(support, 2)
            elif not is_long and abs(price - resistance) / price < 0.002:
                score += 10
                detail['at_resistance'] = round(resistance, 2)

    except Exception:
        pass
    return min(20, score), detail


def _calculate_meanrev_levels(price, vwap_result, is_long, atr):
    """Mean reversion targets: VWAP = T1, opposite band = T2"""
    vwap  = vwap_result.vwap
    band1 = vwap_result.upper_1 if not is_long else vwap_result.lower_1

    if is_long:
        stop    = price - atr * 0.8
        target1 = vwap                           # mean revert to VWAP
        target2 = vwap_result.upper_1            # reach upper band
    else:
        stop    = price + atr * 0.8
        target1 = vwap
        target2 = vwap_result.lower_1

    return stop, target1, target2


def _get_atr(df, period=14) -> float:
    try:
        h, l, c = df['high'], df['low'], df['close']
        tr  = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        atr = float(tr.ewm(span=period, adjust=False).mean().iloc[-1])
        return atr if atr > 0 else float(c.iloc[-1]) * 0.005
    except Exception:
        return float(df['close'].iloc[-1]) * 0.005
