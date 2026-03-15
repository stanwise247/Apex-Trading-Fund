"""
APEX Strategy — Scalp Mode (strategy_scalp.py)
===============================================
Rewritten with properly calibrated scoring that works correctly
in both live and backtesting contexts.

Scoring (0-100):
  HTF Bias:         25pts  — 1h/4h/1d trend alignment
  VWAP Confluence:  20pts  — price position relative to VWAP
  Momentum:         20pts  — EMA structure + RSI
  Volume:           15pts  — volume ratio vs 20-bar average
  Entry Precision:  15pts  — pin bar, engulfing, rejection candle
  Liquidity Sweep:   5pts  — bonus for sweep confirmation

Score thresholds (calibrated to 0-100 range):
  60+ = signal (low quality)
  70+ = signal (standard)
  75+ = signal (high quality)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field


@dataclass
class ScalpSetup:
    direction:      str
    score:          int
    entry:          float
    stop:           float
    target1:        float
    target2:        float
    risk_pct:       float
    risk_pts:       float
    rr_ratio:       float
    timeframe:      str
    reason:         str
    confluences:    List[str]
    warnings:       List[str]
    details:        Dict


# =============================================================
#  INDICATORS  (self-contained, no external deps)
# =============================================================

def _get_atr(df, period=14) -> float:
    try:
        h, l, c = df['high'], df['low'], df['close']
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        atr = float(tr.ewm(span=period, adjust=False).mean().iloc[-1])
        return atr if atr > 0 else float(c.iloc[-1]) * 0.005
    except Exception:
        return float(df['close'].iloc[-1]) * 0.005


def _calc_vwap(df) -> Optional[float]:
    """Calculate VWAP for the current day's bars."""
    try:
        df = df.copy()
        if 'dt' not in df.columns:
            return None
        df['date'] = df['dt'].apply(lambda x: x.date() if hasattr(x, 'date') else None)
        today = df['date'].iloc[-1]
        today_df = df[df['date'] == today]
        if len(today_df) < 2:
            today_df = df.tail(50)
        tp = (today_df['high'] + today_df['low'] + today_df['close']) / 3
        v  = today_df['volume']
        vwap = (tp * v).sum() / v.sum() if v.sum() > 0 else float(df['close'].iloc[-1])
        return float(vwap)
    except Exception:
        return None


def _calc_emas(df) -> Dict:
    c = df['close'].astype(float)
    return {
        'ema9':  float(c.ewm(span=9,  adjust=False).mean().iloc[-1]),
        'ema20': float(c.ewm(span=20, adjust=False).mean().iloc[-1]),
        'ema50': float(c.ewm(span=50, adjust=False).mean().iloc[-1]),
        'close': float(c.iloc[-1]),
    }


def _calc_rsi(df, period=14) -> float:
    try:
        c = df['close'].astype(float)
        delta = c.diff()
        gain  = delta.clip(lower=0).ewm(com=period-1, adjust=False).mean()
        loss  = (-delta).clip(lower=0).ewm(com=period-1, adjust=False).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = float((100 - 100 / (1 + rs)).iloc[-1])
        return rsi if not np.isnan(rsi) else 50.0
    except Exception:
        return 50.0


def _vol_ratio(df) -> float:
    try:
        v    = df['volume'].astype(float)
        cur  = float(v.iloc[-1])
        avg  = float(v.rolling(20).mean().iloc[-1])
        return cur / avg if avg > 0 else 1.0
    except Exception:
        return 1.0


def _detect_sweep(df, direction) -> Tuple[bool, float]:
    """
    Detect liquidity sweep — long wick beyond recent high/low with close back inside.
    Returns (sweep_detected, reversal_strength 0-1)
    """
    try:
        if len(df) < 10:
            return False, 0.0

        recent = df.tail(20)
        bar    = df.iloc[-1]
        o, h, l, c = float(bar['open']), float(bar['high']), float(bar['low']), float(bar['close'])
        bar_range = h - l if h > l else 0.0001

        if direction == 'long':
            # Bull sweep — price swept below recent lows then recovered
            recent_low = float(recent['low'].iloc[:-1].min())
            if l < recent_low and c > recent_low:
                wick_pct   = (recent_low - l) / bar_range
                close_pct  = (c - l) / bar_range
                if wick_pct > 0.2 and close_pct > 0.5:
                    strength = min(1.0, wick_pct * close_pct * 3)
                    return True, strength
        else:
            # Bear sweep — price swept above recent highs then fell back
            recent_high = float(recent['high'].iloc[:-1].max())
            if h > recent_high and c < recent_high:
                wick_pct   = (h - recent_high) / bar_range
                close_pct  = (h - c) / bar_range
                if wick_pct > 0.2 and close_pct > 0.5:
                    strength = min(1.0, wick_pct * close_pct * 3)
                    return True, strength

        return False, 0.0
    except Exception:
        return False, 0.0


def _detect_ob(df, direction) -> bool:
    """Detect order block — last opposing candle before a strong move."""
    try:
        if len(df) < 5:
            return False
        recent = df.tail(10)
        close  = float(df['close'].iloc[-1])
        atr    = _get_atr(df)

        for i in range(len(recent)-3, 0, -1):
            bar = recent.iloc[i]
            o, h, l, c = float(bar['open']), float(bar['high']), float(bar['low']), float(bar['close'])
            body = abs(c - o)
            if body < atr * 0.1:
                continue
            if direction == 'long' and c < o:  # bearish OB for long entry
                if l <= close <= h:
                    return True
            elif direction == 'short' and c > o:  # bullish OB for short entry
                if l <= close <= h:
                    return True
        return False
    except Exception:
        return False


# =============================================================
#  SCORING COMPONENTS
# =============================================================

def _score_htf(df_1h, df_4h, df_1d) -> Tuple[str, int, Dict]:
    """
    HTF bias scoring — 25pts max.
    Checks EMA alignment across 1h, 4h, 1d timeframes.
    """
    bull_votes = 0
    bear_votes = 0
    detail     = {}
    weights    = {'1hour': 10, '4hour': 9, '1day': 6}

    for name, df, w in [('1hour', df_1h, 10), ('4hour', df_4h, 9), ('1day', df_1d, 6)]:
        if df is None or len(df) < 30:
            continue
        emas = _calc_emas(df)
        c    = emas['close']
        e20  = emas['ema20']
        e50  = emas['ema50']

        if c > e20 > e50:
            bull_votes += w
            detail[name] = 'bullish'
        elif c < e20 < e50:
            bear_votes += w
            detail[name] = 'bearish'
        else:
            detail[name] = 'neutral'

    if bull_votes > bear_votes and bull_votes >= 6:
        return 'long',  min(25, bull_votes), detail
    elif bear_votes > bull_votes and bear_votes >= 6:
        return 'short', min(25, bear_votes), detail

    return 'neutral', 0, detail


def _score_vwap(df_5min, direction) -> Tuple[int, Dict]:
    """
    VWAP confluence — 20pts max.
    Self-contained, doesn't rely on external vwap module.
    """
    score  = 0
    detail = {}

    if df_5min is None or len(df_5min) < 20:
        return 0, {}

    try:
        close = float(df_5min['close'].iloc[-1])
        vwap  = _calc_vwap(df_5min)

        if vwap is None or vwap <= 0:
            # Fallback: use EMA20 as proxy
            vwap = float(df_5min['close'].ewm(span=20, adjust=False).mean().iloc[-1])

        detail['vwap'] = round(vwap, 2)
        deviation_pct  = (close - vwap) / vwap * 100
        detail['deviation_pct'] = round(deviation_pct, 3)

        # Position relative to VWAP (8pts)
        if direction == 'long':
            if deviation_pct > 0.05:
                score += 8
                detail['position'] = 'above_vwap'
            elif abs(deviation_pct) <= 0.05:
                score += 5
                detail['position'] = 'at_vwap'
            elif deviation_pct < -0.3:
                score += 3
                detail['position'] = 'extended_below'  # mean reversion potential
        else:
            if deviation_pct < -0.05:
                score += 8
                detail['position'] = 'below_vwap'
            elif abs(deviation_pct) <= 0.05:
                score += 5
                detail['position'] = 'at_vwap'
            elif deviation_pct > 0.3:
                score += 3
                detail['position'] = 'extended_above'

        # VWAP reclaim/rejection (7pts) — did price just cross VWAP?
        if len(df_5min) >= 3:
            prev_close = float(df_5min['close'].iloc[-2])
            prev_dev   = (prev_close - vwap) / vwap * 100

            if direction == 'long' and prev_dev < 0 and deviation_pct > 0:
                score += 7
                detail['vwap_reclaim'] = True
            elif direction == 'short' and prev_dev > 0 and deviation_pct < 0:
                score += 7
                detail['vwap_rejection'] = True

        # VWAP slope (5pts) — is VWAP trending in our direction?
        if len(df_5min) >= 10:
            vwap_vals = []
            for j in range(5, 0, -1):
                sub = df_5min.tail(j + 10).head(10 + j)
                v   = _calc_vwap(sub)
                if v: vwap_vals.append(v)

            if len(vwap_vals) >= 2:
                slope = vwap_vals[-1] - vwap_vals[0]
                if direction == 'long'  and slope > 0: score += 5; detail['vwap_rising'] = True
                elif direction == 'short' and slope < 0: score += 5; detail['vwap_falling'] = True

    except Exception as e:
        detail['error'] = str(e)

    return min(20, score), detail


def _score_momentum(df_5min, direction) -> Tuple[int, Dict]:
    """
    Momentum scoring — 20pts max.
    Uses EMA structure + RSI. Self-contained.
    """
    score  = 0
    detail = {}

    if df_5min is None or len(df_5min) < 30:
        return 0, {}

    try:
        emas = _calc_emas(df_5min)
        rsi  = _calc_rsi(df_5min)
        atr  = _get_atr(df_5min)
        c    = emas['close']
        e9   = emas['ema9']
        e20  = emas['ema20']
        e50  = emas['ema50']

        detail['rsi']  = round(rsi, 1)
        detail['ema9'] = round(e9, 2)
        detail['ema20']= round(e20, 2)

        # EMA structure (10pts)
        if direction == 'long':
            if c > e9 > e20:
                score += 6
                detail['ema_aligned'] = True
            elif c > e20:
                score += 3
            if e9 > e20 > e50:
                score += 4
                detail['ema_stacked'] = True
            elif e9 > e20:
                score += 2
        else:
            if c < e9 < e20:
                score += 6
                detail['ema_aligned'] = True
            elif c < e20:
                score += 3
            if e9 < e20 < e50:
                score += 4
                detail['ema_stacked'] = True
            elif e9 < e20:
                score += 2

        # RSI (10pts)
        detail['rsi_zone'] = 'neutral'
        if direction == 'long':
            if 45 <= rsi <= 65:
                score += 8
                detail['rsi_zone'] = 'bullish_momentum'
            elif 35 <= rsi < 45:
                score += 6
                detail['rsi_zone'] = 'pullback'
            elif 25 <= rsi < 35:
                score += 10
                detail['rsi_zone'] = 'oversold_bounce'
            elif rsi > 70:
                score += 2
                detail['rsi_zone'] = 'overbought_warning'
        else:
            if 35 <= rsi <= 55:
                score += 8
                detail['rsi_zone'] = 'bearish_momentum'
            elif 55 < rsi <= 65:
                score += 6
                detail['rsi_zone'] = 'pullback'
            elif 65 < rsi <= 75:
                score += 10
                detail['rsi_zone'] = 'overbought_reversal'
            elif rsi < 30:
                score += 2
                detail['rsi_zone'] = 'oversold_warning'

    except Exception as e:
        detail['error'] = str(e)

    return min(20, score), detail


def _score_volume(df_5min, direction) -> Tuple[int, Dict]:
    """
    Volume scoring — 15pts max.
    Uses relative volume ratio with calibrated thresholds.
    Volume ratio 0.8-1.5x is normal range — score accordingly.
    """
    score  = 0
    detail = {}

    if df_5min is None or len(df_5min) < 20:
        return 5, {}  # neutral baseline

    try:
        v       = df_5min['volume'].astype(float)
        cur_vol = float(v.iloc[-1])
        avg_vol = float(v.rolling(20).mean().iloc[-1])
        ratio   = cur_vol / avg_vol if avg_vol > 0 else 1.0
        detail['vol_ratio'] = round(ratio, 2)
        detail['cur_vol']   = int(cur_vol)

        # Calibrated thresholds — normal market volume is 0.7-1.3x
        if ratio >= 2.0:
            score += 15
            detail['vol_spike'] = True
        elif ratio >= 1.5:
            score += 12
            detail['vol_elevated'] = True
        elif ratio >= 1.2:
            score += 9
        elif ratio >= 1.0:
            score += 7
        elif ratio >= 0.8:
            score += 5
        elif ratio >= 0.6:
            score += 3
        else:
            score += 1  # very low volume — weak signal

        # Consecutive volume build (bonus 3pts) — increasing vol over last 3 bars
        if len(v) >= 4:
            last3 = [float(v.iloc[-i]) for i in range(3, 0, -1)]
            if last3[2] > last3[1] > last3[0]:
                score += 3
                detail['vol_building'] = True

        # Volume vs recent session high (bonus)
        if len(v) >= 50:
            session_max = float(v.tail(50).max())
            if cur_vol >= session_max * 0.9:
                score += 2
                detail['session_vol_high'] = True

    except Exception as e:
        detail['error'] = str(e)

    return min(15, score), detail


def _score_entry(df_5min, direction, atr) -> Tuple[int, Dict]:
    """
    Entry precision scoring — 15pts max.
    Pin bars, engulfing, inside bar breaks, rejection wicks.
    """
    score  = 0
    detail = {}

    if df_5min is None or len(df_5min) < 3:
        return 0, {}

    try:
        bar  = df_5min.iloc[-1]
        prev = df_5min.iloc[-2]
        o, h, l, c   = float(bar['open']),  float(bar['high']),  float(bar['low']),  float(bar['close'])
        po, ph, pl, pc = float(prev['open']), float(prev['high']), float(prev['low']), float(prev['close'])

        bar_range  = h - l if h > l else 0.0001
        body       = abs(c - o)
        upper_wick = h - max(c, o)
        lower_wick = min(c, o) - l
        body_pct   = body / bar_range

        # 1. Pin bar / hammer (8pts)
        if direction == 'long':
            # Bullish pin — long lower wick, small body near top
            if lower_wick > body * 1.8 and lower_wick > upper_wick * 1.5 and (c > o or body_pct < 0.35):
                score += 8
                detail['pinbar'] = True
                detail['lower_wick_ratio'] = round(lower_wick / bar_range, 2)
        else:
            # Bearish pin — long upper wick, small body near bottom
            if upper_wick > body * 1.8 and upper_wick > lower_wick * 1.5 and (c < o or body_pct < 0.35):
                score += 8
                detail['pinbar'] = True
                detail['upper_wick_ratio'] = round(upper_wick / bar_range, 2)

        # 2. Engulfing (7pts)
        prev_body = abs(pc - po)
        if direction == 'long' and c > o:  # bullish candle
            if c > po and o < pc and body > prev_body * 1.1:
                score += 7
                detail['engulfing'] = True
        elif direction == 'short' and c < o:  # bearish candle
            if c < po and o > pc and body > prev_body * 1.1:
                score += 7
                detail['engulfing'] = True

        # 3. Close position quality (4pts)
        close_pos = (c - l) / bar_range if bar_range > 0 else 0.5
        if direction == 'long'  and close_pos > 0.65:
            score += 4
            detail['strong_close'] = True
        elif direction == 'short' and close_pos < 0.35:
            score += 4
            detail['strong_close'] = True
        elif direction == 'long'  and close_pos > 0.5:
            score += 2
        elif direction == 'short' and close_pos < 0.5:
            score += 2

        # 4. Tight body = precision entry (2pts)
        if atr > 0 and body / atr < 0.25:
            score += 2
            detail['tight_body'] = True

    except Exception as e:
        detail['error'] = str(e)

    return min(15, score), detail


def _score_sweep_bonus(df_5min, direction) -> Tuple[int, Dict]:
    """
    Liquidity sweep bonus — 5pts max.
    """
    swept, strength = _detect_sweep(df_5min, direction)
    if swept:
        pts = min(5, int(strength * 6) + 2)
        return pts, {'sweep': True, 'strength': round(strength, 2)}
    return 0, {'sweep': False}


def _score_ob_bonus(df_5min, direction) -> int:
    """Order block bonus — 5pts max."""
    return 5 if _detect_ob(df_5min, direction) else 0


# =============================================================
#  STOP CALCULATION
# =============================================================

def _calculate_stop(df_5min, direction, atr) -> Optional[float]:
    """Calculate stop loss using recent swing structure."""
    try:
        if df_5min is None or len(df_5min) < 5:
            return None

        price  = float(df_5min['close'].iloc[-1])
        recent = df_5min.tail(10)

        if direction == 'long':
            swing_low = float(recent['low'].min())
            stop = swing_low - atr * 0.15
            # Cap at 2x ATR
            stop = max(stop, price - atr * 2.0)
        else:
            swing_high = float(recent['high'].max())
            stop = swing_high + atr * 0.15
            stop = min(stop, price + atr * 2.0)

        return round(stop, 2)
    except Exception:
        return None


# =============================================================
#  MAIN SCANNER
# =============================================================

def scan_scalp(
    df_1min:   Optional[pd.DataFrame],
    df_5min:   Optional[pd.DataFrame],
    df_15min:  Optional[pd.DataFrame],
    df_1hour:  Optional[pd.DataFrame],
    df_4hour:  Optional[pd.DataFrame],
    df_1day:   Optional[pd.DataFrame],
    symbol:    str = 'NQ',
    min_score: int = 65,
) -> Optional[ScalpSetup]:
    """
    Scan for scalp setups. Returns ScalpSetup if score >= min_score, else None.
    Fully self-contained — works in both live and backtest contexts.
    """
    entry_df = df_5min
    if entry_df is None or len(entry_df) < 50:
        return None

    atr = _get_atr(entry_df)
    if atr <= 0:
        return None

    # 1. HTF Direction (25pts)
    direction, htf_pts, htf_detail = _score_htf(df_1hour, df_4hour, df_1day)
    if direction == 'neutral' or htf_pts < 6:
        return None

    score       = htf_pts
    confluences = []
    warnings    = []
    details     = {'htf': htf_detail, 'htf_pts': htf_pts}

    if htf_pts >= 20:
        confluences.append(f'Strong HTF {direction} ({htf_pts}pts)')
    else:
        confluences.append(f'HTF {direction} ({htf_pts}pts)')

    # 2. VWAP Confluence (20pts)
    vwap_pts, vwap_detail = _score_vwap(entry_df, direction)
    score   += vwap_pts
    details['vwap'] = vwap_detail
    details['vwap_pts'] = vwap_pts
    if vwap_detail.get('vwap_reclaim'):
        confluences.append('VWAP reclaim ✅')
    elif vwap_detail.get('vwap_rejection'):
        confluences.append('VWAP rejection ✅')
    elif vwap_detail.get('position') == 'above_vwap' and direction == 'long':
        confluences.append('Above VWAP')
    elif vwap_detail.get('position') == 'below_vwap' and direction == 'short':
        confluences.append('Below VWAP')

    # 3. Momentum (20pts)
    mom_pts, mom_detail = _score_momentum(entry_df, direction)
    score   += mom_pts
    details['momentum'] = mom_detail
    details['mom_pts']  = mom_pts
    if mom_detail.get('ema_stacked'):
        confluences.append('EMA stacked')
    elif mom_detail.get('ema_aligned'):
        confluences.append('EMA aligned')
    rsi_zone = mom_detail.get('rsi_zone', '')
    if 'oversold' in rsi_zone or 'overbought' in rsi_zone:
        confluences.append(f'RSI: {rsi_zone}')

    # 4. Volume (15pts)
    vol_pts, vol_detail = _score_volume(entry_df, direction)
    score   += vol_pts
    details['volume'] = vol_detail
    details['vol_pts'] = vol_pts
    if vol_detail.get('vol_spike'):
        confluences.append('Volume spike 🔥')
    elif vol_detail.get('vol_elevated'):
        confluences.append('Volume elevated')
    elif vol_detail.get('vol_building'):
        confluences.append('Volume building')

    # 5. Entry Precision (15pts)
    entry_pts, entry_detail = _score_entry(entry_df, direction, atr)
    score    += entry_pts
    details['entry'] = entry_detail
    details['entry_pts'] = entry_pts
    if entry_detail.get('pinbar'):
        confluences.append('Pin bar ✅')
    elif entry_detail.get('engulfing'):
        confluences.append('Engulfing candle ✅')
    if entry_detail.get('strong_close'):
        confluences.append('Strong close')

    # 6. Liquidity Sweep Bonus (5pts)
    sweep_pts, sweep_detail = _score_sweep_bonus(entry_df, direction)
    score    += sweep_pts
    details['sweep'] = sweep_detail
    if sweep_detail.get('sweep'):
        confluences.append(f'Liquidity sweep ⚡ ({sweep_pts}pts)')

    # 7. Order Block Bonus (5pts — on top of 100)
    ob_pts = _score_ob_bonus(entry_df, direction)
    score  += ob_pts
    details['ob_pts'] = ob_pts
    if ob_pts > 0:
        confluences.append('Order block entry')

    details['total_score'] = score

    if score < min_score:
        return None

    # Calculate levels
    current_price = float(entry_df['close'].iloc[-1])
    stop = _calculate_stop(entry_df, direction, atr)
    if stop is None:
        return None

    risk_pts = abs(current_price - stop)
    if risk_pts <= 0 or risk_pts > atr * 3:
        return None

    is_long = direction == 'long'
    target1 = round(current_price + risk_pts * 2.0, 2) if is_long else round(current_price - risk_pts * 2.0, 2)
    target2 = round(current_price + risk_pts * 3.0, 2) if is_long else round(current_price - risk_pts * 3.0, 2)

    risk_pct = 1.5 if score >= 75 else 1.0 if score >= 65 else 0.75

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
        rr_ratio    = 2.5,
        timeframe   = '5min',
        reason      = reason,
        confluences = confluences,
        warnings    = warnings,
        details     = details,
    )
