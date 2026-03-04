"""
APEX Market Structure Engine — market_structure.py
====================================================
Detects key price levels and market context:

  - Opening Range (first 15min high/low)
  - Previous Day High/Low/Close (PDH/PDL/PDC)
  - Previous Week High/Low
  - Equal Highs/Lows (liquidity pools)
  - Fibonacci retracement levels
  - Swing High/Low mapping
  - Trend strength and direction
  - Market condition (trending/ranging/volatile)

These levels act as magnets and barriers for price.
Institutional traders defend and target these levels.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo('America/New_York')


# =============================================================
#  DATA STRUCTURES
# =============================================================

@dataclass
class OpeningRange:
    high:           float       # OR high (first 15min)
    low:            float       # OR low
    midpoint:       float       # OR midpoint
    range_size:     float       # High - Low in points
    breakout_up:    bool        # Price broke above OR high
    breakout_down:  bool        # Price broke below OR low
    inside:         bool        # Price still inside OR
    bars_since_open:int         # Bars since market open


@dataclass  
class PreviousDayLevels:
    high:           float       # Previous day high
    low:            float       # Previous day low
    close:          float       # Previous day close
    open:           float       # Previous day open
    range_size:     float       # PDH - PDL
    above_pdh:      bool        # Current price above PDH
    below_pdl:      bool        # Current price below PDL
    in_range:       bool        # Price between PDL and PDH
    above_pdc:      bool        # Current price above PDC


@dataclass
class SwingLevel:
    price:          float
    kind:           str         # 'high' or 'low'
    bar_idx:        int
    bar_dt:         str
    strength:       int         # 1-3 (how many bars on each side)
    tested:         int         # Times price returned to level
    broken:         bool


@dataclass
class EqualLevels:
    price:          float       # Average price of equal highs/lows
    kind:           str         # 'equal_highs' or 'equal_lows'
    count:          int         # Number of equal levels
    tolerance_pct:  float       # How close they are


@dataclass
class FibLevels:
    swing_high:     float
    swing_low:      float
    direction:      str         # 'bullish' or 'bearish' (which way swing went)
    fib_236:        float
    fib_382:        float
    fib_500:        float
    fib_618:        float
    fib_786:        float
    fib_1000:       float       # Full retrace
    nearest_level:  str         # Which fib level price is closest to
    nearest_price:  float
    at_key_level:   bool        # Price within 0.2% of a key fib


@dataclass
class MarketCondition:
    condition:      str         # 'trending_up', 'trending_down', 'ranging', 'volatile'
    trend_strength: int         # 0-100
    atr_percentile: int         # Where current ATR sits vs history (0-100)
    range_bound:    bool        # True if price in tight range
    momentum:       str         # 'accelerating', 'decelerating', 'flat'
    recommended_strategy: str   # Which strategy type to use


# =============================================================
#  OPENING RANGE
# =============================================================

def get_opening_range(df: pd.DataFrame, range_minutes: int = 15) -> Optional[OpeningRange]:
    """
    Opening Range: High and low of first N minutes after market open.
    NQ futures: 9:30am ET open.
    
    Classic ORB (Opening Range Breakout) strategy:
    - Break above OR high = long signal
    - Break below OR low = short signal
    - Higher probability when aligned with HTF trend
    """
    if df is None or len(df) < 5:
        return None

    try:
        now_ny = df.index[-1].astimezone(NY_TZ) if df.index[-1].tzinfo else None
        if now_ny is None:
            return None

        # Find today's open (9:30am ET)
        today = now_ny.date()
        market_open = datetime(today.year, today.month, today.day, 9, 30, 
                               tzinfo=NY_TZ)
        range_end   = market_open + timedelta(minutes=range_minutes)

        # Get bars in opening range
        or_bars = df[(df.index >= pd.Timestamp(market_open)) & 
                     (df.index <  pd.Timestamp(range_end))]

        if len(or_bars) == 0:
            return None

        or_high = float(or_bars['high'].max())
        or_low  = float(or_bars['low'].min())
        or_mid  = (or_high + or_low) / 2

        current_price = float(df['close'].iloc[-1])
        
        # Count bars since open
        bars_since = len(df[df.index >= pd.Timestamp(market_open)])

        return OpeningRange(
            high            = round(or_high, 2),
            low             = round(or_low, 2),
            midpoint        = round(or_mid, 2),
            range_size      = round(or_high - or_low, 2),
            breakout_up     = current_price > or_high,
            breakout_down   = current_price < or_low,
            inside          = or_low <= current_price <= or_high,
            bars_since_open = bars_since,
        )

    except Exception:
        return None


# =============================================================
#  PREVIOUS DAY LEVELS
# =============================================================

def get_previous_day_levels(df_daily: pd.DataFrame) -> Optional[PreviousDayLevels]:
    """
    Previous Day High/Low/Close — key institutional reference levels.
    
    PDH and PDL are the most watched levels by institutional traders.
    Breaking PDH = bullish momentum confirmation.
    Breaking PDL = bearish momentum confirmation.
    """
    if df_daily is None or len(df_daily) < 2:
        return None

    try:
        # Get last two complete daily bars
        prev_day  = df_daily.iloc[-2]
        pdh       = float(prev_day['high'])
        pdl       = float(prev_day['low'])
        pdc       = float(prev_day['close'])
        pdo       = float(prev_day['open'])

        # Current price from most recent bar
        current_price = float(df_daily.iloc[-1]['close'])

        return PreviousDayLevels(
            high        = round(pdh, 2),
            low         = round(pdl, 2),
            close       = round(pdc, 2),
            open        = round(pdo, 2),
            range_size  = round(pdh - pdl, 2),
            above_pdh   = current_price > pdh,
            below_pdl   = current_price < pdl,
            in_range    = pdl <= current_price <= pdh,
            above_pdc   = current_price > pdc,
        )

    except Exception:
        return None


def get_previous_week_levels(df_weekly: pd.DataFrame) -> Optional[Dict]:
    """Previous Week High/Low — longer-term reference levels"""
    if df_weekly is None or len(df_weekly) < 2:
        return None

    try:
        prev_week     = df_weekly.iloc[-2]
        current_price = float(df_weekly.iloc[-1]['close'])
        pwh = float(prev_week['high'])
        pwl = float(prev_week['low'])

        return {
            'high':       round(pwh, 2),
            'low':        round(pwl, 2),
            'above_pwh':  current_price > pwh,
            'below_pwl':  current_price < pwl,
            'in_range':   pwl <= current_price <= pwh,
        }
    except Exception:
        return None


# =============================================================
#  SWING HIGH/LOW DETECTION
# =============================================================

def find_swing_highs_lows(df: pd.DataFrame, strength: int = 3) -> Tuple[List[SwingLevel], List[SwingLevel]]:
    """
    Find swing highs and lows with given strength (bars on each side).
    
    Swing High: bar[i] high > bar[i-n] high AND bar[i] high > bar[i+n] high
    Swing Low:  bar[i] low  < bar[i-n] low  AND bar[i] low  < bar[i+n] low
    """
    if df is None or len(df) < strength * 2 + 1:
        return [], []

    highs = []
    lows  = []

    for i in range(strength, len(df) - strength):
        bar_high = float(df['high'].iloc[i])
        bar_low  = float(df['low'].iloc[i])
        bar_dt   = str(df.index[i])

        # Check if swing high
        is_sh = all(bar_high >= float(df['high'].iloc[i-j]) for j in range(1, strength+1)) and \
                all(bar_high >= float(df['high'].iloc[i+j]) for j in range(1, strength+1))

        # Check if swing low  
        is_sl = all(bar_low  <= float(df['low'].iloc[i-j])  for j in range(1, strength+1)) and \
                all(bar_low  <= float(df['low'].iloc[i+j])  for j in range(1, strength+1))

        if is_sh:
            highs.append(SwingLevel(
                price    = round(bar_high, 2),
                kind     = 'high',
                bar_idx  = i,
                bar_dt   = bar_dt,
                strength = strength,
                tested   = 0,
                broken   = False,
            ))

        if is_sl:
            lows.append(SwingLevel(
                price    = round(bar_low, 2),
                kind     = 'low',
                bar_idx  = i,
                bar_dt   = bar_dt,
                strength = strength,
                tested   = 0,
                broken   = False,
            ))

    return highs, lows


# =============================================================
#  EQUAL HIGHS / EQUAL LOWS (Liquidity Pools)
# =============================================================

def find_equal_levels(df: pd.DataFrame, tolerance_pct: float = 0.1,
                      lookback: int = 100) -> List[EqualLevels]:
    """
    Equal Highs/Lows: Multiple swing highs/lows at similar price levels.
    
    These are liquidity pools — stops cluster here.
    Price will often sweep these levels before reversing.
    Key ICT concept: "Sell-side liquidity" (equal lows) and 
    "Buy-side liquidity" (equal highs).
    """
    if df is None or len(df) < 20:
        return []

    window = df.tail(lookback)
    highs, lows = find_swing_highs_lows(window, strength=2)

    results = []

    # Check equal highs
    if len(highs) >= 2:
        for i, h1 in enumerate(highs):
            cluster = [h1.price]
            for h2 in highs[i+1:]:
                if abs(h2.price - h1.price) / h1.price * 100 <= tolerance_pct:
                    cluster.append(h2.price)
            if len(cluster) >= 2:
                avg_price = sum(cluster) / len(cluster)
                results.append(EqualLevels(
                    price         = round(avg_price, 2),
                    kind          = 'equal_highs',
                    count         = len(cluster),
                    tolerance_pct = tolerance_pct,
                ))

    # Check equal lows
    if len(lows) >= 2:
        for i, l1 in enumerate(lows):
            cluster = [l1.price]
            for l2 in lows[i+1:]:
                if abs(l2.price - l1.price) / l1.price * 100 <= tolerance_pct:
                    cluster.append(l2.price)
            if len(cluster) >= 2:
                avg_price = sum(cluster) / len(cluster)
                results.append(EqualLevels(
                    price         = round(avg_price, 2),
                    kind          = 'equal_lows',
                    count         = len(cluster),
                    tolerance_pct = tolerance_pct,
                ))

    return results


# =============================================================
#  FIBONACCI LEVELS
# =============================================================

def get_fibonacci_levels(df: pd.DataFrame, lookback: int = 100) -> Optional[FibLevels]:
    """
    Fibonacci retracement levels from most significant recent swing.
    
    Key levels: 23.6%, 38.2%, 50%, 61.8% (golden ratio), 78.6%
    61.8% is the most respected level — "the golden pocket"
    Combined with OB/FVG = extremely high probability entry
    """
    if df is None or len(df) < 20:
        return None

    try:
        window = df.tail(lookback)
        highs, lows = find_swing_highs_lows(window, strength=3)

        if not highs or not lows:
            return None

        # Most recent swing high and low
        last_high = max(highs, key=lambda x: x.bar_idx)
        last_low  = max(lows,  key=lambda x: x.bar_idx)

        swing_high = last_high.price
        swing_low  = last_low.price
        rng        = swing_high - swing_low

        if rng <= 0:
            return None

        # Determine direction (which came last)
        if last_high.bar_idx > last_low.bar_idx:
            direction = 'bearish'  # High came after low — potential reversal down
            # Fib retracement from high back down
            fib_236 = round(swing_high - 0.236 * rng, 2)
            fib_382 = round(swing_high - 0.382 * rng, 2)
            fib_500 = round(swing_high - 0.500 * rng, 2)
            fib_618 = round(swing_high - 0.618 * rng, 2)
            fib_786 = round(swing_high - 0.786 * rng, 2)
            fib_100 = round(swing_low, 2)
        else:
            direction = 'bullish'  # Low came after high — potential reversal up
            # Fib retracement from low back up
            fib_236 = round(swing_low + 0.236 * rng, 2)
            fib_382 = round(swing_low + 0.382 * rng, 2)
            fib_500 = round(swing_low + 0.500 * rng, 2)
            fib_618 = round(swing_low + 0.618 * rng, 2)
            fib_786 = round(swing_low + 0.786 * rng, 2)
            fib_100 = round(swing_high, 2)

        current_price = float(df['close'].iloc[-1])
        key_fibs = {
            '23.6%': fib_236,
            '38.2%': fib_382,
            '50.0%': fib_500,
            '61.8%': fib_618,
            '78.6%': fib_786,
        }

        # Find nearest fib level
        nearest_name  = min(key_fibs, key=lambda k: abs(key_fibs[k] - current_price))
        nearest_price = key_fibs[nearest_name]
        dist_pct      = abs(current_price - nearest_price) / current_price * 100
        at_key_level  = dist_pct < 0.2

        return FibLevels(
            swing_high    = round(swing_high, 2),
            swing_low     = round(swing_low, 2),
            direction     = direction,
            fib_236       = fib_236,
            fib_382       = fib_382,
            fib_500       = fib_500,
            fib_618       = fib_618,
            fib_786       = fib_786,
            fib_1000      = fib_100,
            nearest_level = nearest_name,
            nearest_price = nearest_price,
            at_key_level  = at_key_level,
        )

    except Exception:
        return None


# =============================================================
#  MARKET CONDITION DETECTION
# =============================================================

def get_market_condition(df: pd.DataFrame) -> MarketCondition:
    """
    Classify current market condition.
    This determines which strategy to use.
    
    Trending up:   Use momentum/BOS strategies
    Trending down: Use short/BOS strategies  
    Ranging:       Use mean reversion / OB bounce strategies
    Volatile:      Reduce size, widen stops, or avoid
    """
    if df is None or len(df) < 50:
        return MarketCondition('unknown', 0, 50, False, 'flat', 'intraday_swing')

    try:
        close  = df['close']
        high   = df['high']
        low    = df['low']

        # --- Trend detection using EMAs ---
        ema20  = close.ewm(span=20,  adjust=False).mean()
        ema50  = close.ewm(span=50,  adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()

        current = float(close.iloc[-1])
        e20     = float(ema20.iloc[-1])
        e50     = float(ema50.iloc[-1])
        e200    = float(ema200.iloc[-1])

        # Trend alignment
        bull_aligned = current > e20 > e50  # strong bull
        bear_aligned = current < e20 < e50  # strong bear

        # Trend strength: ADX approximation using directional movement
        tr    = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
        atr14 = tr.ewm(span=14, adjust=False).mean()

        up_move   = high.diff()
        down_move = -low.diff()
        plus_dm   = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm  = down_move.where((down_move > up_move) & (down_move > 0), 0)

        plus_di   = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / atr14.replace(0, np.nan))
        minus_di  = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / atr14.replace(0, np.nan))
        dx        = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx       = dx.ewm(span=14, adjust=False).mean()

        current_adx = float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 25
        trend_strength = min(100, int(current_adx))

        # ATR percentile vs recent history
        current_atr = float(atr14.iloc[-1])
        atr_history = atr14.tail(100)
        atr_pct     = int(pd.Series(atr_history).rank(pct=True).iloc[-1] * 100)

        # Range detection: price in tight band
        recent_high = float(high.tail(20).max())
        recent_low  = float(low.tail(20).min())
        recent_range_pct = (recent_high - recent_low) / current * 100
        range_bound = recent_range_pct < 0.5  # Less than 0.5% range = tight chop

        # Momentum: is trend accelerating or decelerating?
        roc10 = (close - close.shift(10)) / close.shift(10) * 100
        roc5  = (close - close.shift(5))  / close.shift(5)  * 100
        if abs(float(roc5.iloc[-1])) > abs(float(roc10.iloc[-1])) * 0.6:
            momentum = 'accelerating'
        elif abs(float(roc5.iloc[-1])) < abs(float(roc10.iloc[-1])) * 0.3:
            momentum = 'decelerating'
        else:
            momentum = 'flat'

        # Classify condition
        if atr_pct > 80:
            condition = 'volatile'
            recommended = 'reduce_size'
        elif current_adx > 30 and bull_aligned:
            condition = 'trending_up'
            recommended = 'intraday_swing'
        elif current_adx > 30 and bear_aligned:
            condition = 'trending_down'
            recommended = 'intraday_swing'
        elif range_bound or current_adx < 20:
            condition = 'ranging'
            recommended = 'mean_reversion'
        else:
            condition = 'transitioning'
            recommended = 'intraday_swing'

        return MarketCondition(
            condition           = condition,
            trend_strength      = trend_strength,
            atr_percentile      = atr_pct,
            range_bound         = range_bound,
            momentum            = momentum,
            recommended_strategy= recommended,
        )

    except Exception:
        return MarketCondition('unknown', 0, 50, False, 'flat', 'intraday_swing')


# =============================================================
#  COMBINED MARKET STRUCTURE SCORING
# =============================================================

def score_market_structure(
    df_5min:  pd.DataFrame,
    df_15min: pd.DataFrame,
    df_1hour: pd.DataFrame,
    df_1day:  pd.DataFrame,
    df_1week: pd.DataFrame,
    direction: str
) -> Tuple[int, Dict]:
    """
    Score market structure confluence for a given direction.
    Returns (score_0_to_25, details_dict)
    
    Max 25 points:
    - Previous day levels:    7pts
    - Opening range:          5pts  
    - Fibonacci confluence:   7pts
    - Equal levels (liquidity):3pts
    - Market condition:       3pts
    """
    score   = 0
    details = {}
    is_long = direction in ('long', 'bullish')

    current_df = df_5min or df_15min
    if current_df is None:
        return 0, {}

    current_price = float(current_df['close'].iloc[-1])

    # --- Previous Day Levels (7pts) ---
    if df_1day is not None and len(df_1day) >= 2:
        pdl = get_previous_day_levels(df_1day)
        if pdl:
            details['pdh'] = pdl.high
            details['pdl'] = pdl.low
            details['pdc'] = pdl.close

            if is_long:
                if pdl.above_pdh:
                    score += 7   # Breaking PDH = strong bullish momentum
                    details['breaking_pdh'] = True
                elif pdl.above_pdc and not pdl.above_pdh:
                    score += 4   # Above PDC, approaching PDH
                    details['above_pdc'] = True
                elif abs(current_price - pdl.low) / pdl.range_size < 0.1:
                    score += 5   # Near PDL = discount zone for longs
                    details['at_pdl_support'] = True
            else:
                if pdl.below_pdl:
                    score += 7   # Breaking PDL = strong bearish momentum
                    details['breaking_pdl'] = True
                elif not pdl.above_pdc:
                    score += 4   # Below PDC = bearish
                    details['below_pdc'] = True
                elif abs(current_price - pdl.high) / pdl.range_size < 0.1:
                    score += 5   # Near PDH = premium zone for shorts
                    details['at_pdh_resistance'] = True

    # --- Opening Range (5pts) ---
    if df_5min is not None:
        or_result = get_opening_range(df_5min)
        if or_result:
            details['or_high'] = or_result.high
            details['or_low']  = or_result.low

            if is_long:
                if or_result.breakout_up:
                    score += 5   # ORB to upside
                    details['orb_long'] = True
                elif abs(current_price - or_result.low) / or_result.range_size < 0.1:
                    score += 3   # At OR low = potential long entry
                    details['at_or_low'] = True
            else:
                if or_result.breakout_down:
                    score += 5   # ORB to downside
                    details['orb_short'] = True
                elif abs(current_price - or_result.high) / or_result.range_size < 0.1:
                    score += 3   # At OR high = potential short entry
                    details['at_or_high'] = True

    # --- Fibonacci Levels (7pts) ---
    fib_df = df_15min or df_1hour
    if fib_df is not None and len(fib_df) >= 30:
        fibs = get_fibonacci_levels(fib_df)
        if fibs:
            details['fib_618'] = fibs.fib_618
            details['fib_382'] = fibs.fib_382
            details['fib_nearest'] = fibs.nearest_level

            if fibs.at_key_level:
                # Golden pocket (61.8%) is highest probability
                if '61.8' in fibs.nearest_level:
                    score += 7
                    details['at_golden_pocket'] = True
                elif '38.2' in fibs.nearest_level or '50.0' in fibs.nearest_level:
                    score += 5
                    details['at_key_fib'] = True
                else:
                    score += 3
                    details['at_fib_level'] = True

    # --- Equal Highs/Lows (3pts) ---
    eq_df = df_15min or df_5min
    if eq_df is not None:
        equal_levels = find_equal_levels(eq_df)
        for eq in equal_levels:
            dist_pct = abs(current_price - eq.price) / current_price * 100
            if dist_pct < 0.3:
                if is_long and eq.kind == 'equal_lows':
                    score += 3   # At equal lows = buy-side liquidity grab
                    details['equal_lows_sweep'] = eq.price
                elif not is_long and eq.kind == 'equal_highs':
                    score += 3   # At equal highs = sell-side liquidity grab
                    details['equal_highs_sweep'] = eq.price

    # --- Market Condition (3pts) ---
    mc_df = df_15min or df_5min
    if mc_df is not None and len(mc_df) >= 50:
        mc = get_market_condition(mc_df)
        details['market_condition'] = mc.condition
        details['trend_strength']   = mc.trend_strength

        if is_long and mc.condition == 'trending_up':
            score += 3
            details['trend_aligned'] = True
        elif not is_long and mc.condition == 'trending_down':
            score += 3
            details['trend_aligned'] = True
        elif mc.condition == 'ranging':
            score += 1  # Mean reversion setups work in ranges
        elif mc.condition == 'volatile':
            score -= 2  # Penalise volatile conditions
            details['volatile_warning'] = True

    return max(0, min(25, score)), details


# =============================================================
#  QUICK SUMMARY FOR TELEGRAM
# =============================================================

def get_structure_summary(df_5min, df_1day) -> str:
    """One-line market structure summary for Telegram alerts"""
    lines = []

    if df_1day is not None and len(df_1day) >= 2:
        pdl = get_previous_day_levels(df_1day)
        if pdl:
            price = float(df_5min['close'].iloc[-1]) if df_5min is not None else 0
            if pdl.above_pdh:
                lines.append(f'Above PDH {pdl.high:.0f} 🚀')
            elif pdl.below_pdl:
                lines.append(f'Below PDL {pdl.low:.0f} 📉')
            else:
                lines.append(f'PDH:{pdl.high:.0f} PDL:{pdl.low:.0f}')

    if df_5min is not None:
        or_r = get_opening_range(df_5min)
        if or_r:
            if or_r.breakout_up:
                lines.append(f'ORB UP {or_r.high:.0f} ✅')
            elif or_r.breakout_down:
                lines.append(f'ORB DOWN {or_r.low:.0f} ✅')

    return ' | '.join(lines) if lines else 'Structure: N/A'
