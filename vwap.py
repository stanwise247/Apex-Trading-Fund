"""
APEX VWAP & Volume Profile Engine — vwap.py
=============================================
Calculates:
  - Standard VWAP (session reset daily)
  - Anchored VWAP (from key swing points)
  - VWAP bands (1σ, 2σ, 3σ)
  - Volume Profile (POC, VAH, VAL, HVN, LVN)
  - Volume Delta (buying vs selling pressure)
  - Cumulative Delta Divergence
  - VWAP deviation scoring for setups

Key concepts:
  - Price above VWAP = bullish bias
  - Price below VWAP = bearish bias
  - VWAP reclaim after dip = long setup
  - VWAP rejection after pop = short setup
  - POC = Point of Control (highest volume price)
  - VAH = Value Area High (70% of volume above)
  - VAL = Value Area Low (70% of volume below)
  - HVN = High Volume Node (price magnet)
  - LVN = Low Volume Node (price moves fast through)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple


# =============================================================
#  DATA STRUCTURES
# =============================================================

@dataclass
class VWAPResult:
    vwap:           float
    upper_1:        float       # VWAP + 1σ
    upper_2:        float       # VWAP + 2σ
    upper_3:        float       # VWAP + 3σ
    lower_1:        float       # VWAP - 1σ
    lower_2:        float       # VWAP - 2σ
    lower_3:        float       # VWAP - 3σ
    deviation:      float       # Current price deviation in σ units
    position:       str         # 'above', 'below', 'at'
    reclaim:        bool        # Price just reclaimed VWAP from below
    rejection:      bool        # Price just rejected VWAP from above
    score:          int         # 0-100 score for setup quality


@dataclass
class VolumeProfile:
    poc:            float       # Point of Control
    vah:            float       # Value Area High
    val:            float       # Value Area Low
    value_area_pct: float       # % of volume in value area (target 70%)
    hvn_levels:     List[float] # High Volume Nodes
    lvn_levels:     List[float] # Low Volume Nodes
    profile:        Dict        # Full price->volume map
    current_in_va:  bool        # Current price inside value area
    current_above_poc: bool     # Current price above POC


@dataclass
class VolumeDelta:
    delta:          float       # Net buying - selling volume for period
    cum_delta:      float       # Cumulative delta
    delta_divergence: bool      # Price up but delta down (or vice versa)
    buying_pressure: float      # % of volume that was buying
    selling_pressure: float     # % of volume that was selling
    absorption:     bool        # Large volume with small price move
    score:          int         # 0-100 bullish/bearish score


@dataclass
class AnchoredVWAP:
    anchor_price:   float       # Price at anchor point
    anchor_dt:      str         # DateTime of anchor
    vwap:           float       # Current AVWAP value
    upper_1:        float
    lower_1:        float
    bars_since:     int         # Bars since anchor
    price_above:    bool        # Current price above AVWAP


# =============================================================
#  STANDARD VWAP
# =============================================================

def calculate_vwap(df: pd.DataFrame, session_reset: bool = True) -> pd.Series:
    """
    Calculate VWAP with optional daily session reset.
    
    VWAP = Cumulative(Typical Price × Volume) / Cumulative(Volume)
    Typical Price = (High + Low + Close) / 3
    """
    tp = (df['high'] + df['low'] + df['close']) / 3
    vol = df['volume'].replace(0, np.nan).fillna(1)
    
    if session_reset and df.index.dtype.tz is not None:
        # Reset VWAP at start of each trading day
        dates = df.index.date
        vwap_values = pd.Series(index=df.index, dtype=float)
        
        for date in pd.unique(dates):
            mask = dates == date
            day_tp  = tp[mask]
            day_vol = vol[mask]
            cum_tpv = (day_tp * day_vol).cumsum()
            cum_vol = day_vol.cumsum()
            vwap_values[mask] = cum_tpv / cum_vol
        
        return vwap_values
    else:
        cum_tpv = (tp * vol).cumsum()
        cum_vol = vol.cumsum()
        return cum_tpv / cum_vol


def calculate_vwap_bands(df: pd.DataFrame, vwap: pd.Series) -> Dict[str, pd.Series]:
    """
    Calculate VWAP standard deviation bands.
    Bands show where price is statistically likely to revert to VWAP.
    """
    tp  = (df['high'] + df['low'] + df['close']) / 3
    vol = df['volume'].replace(0, np.nan).fillna(1)

    dates = df.index.date if hasattr(df.index, 'date') else None
    bands = {f'upper_{i}': pd.Series(index=df.index, dtype=float) for i in [1,2,3]}
    bands.update({f'lower_{i}': pd.Series(index=df.index, dtype=float) for i in [1,2,3]})

    def calc_for_segment(mask):
        seg_tp   = tp[mask]
        seg_vol  = vol[mask]
        seg_vwap = vwap[mask]
        cum_vol  = seg_vol.cumsum()
        variance = ((seg_tp - seg_vwap) ** 2 * seg_vol).cumsum() / cum_vol
        std_dev  = np.sqrt(variance.clip(lower=0))
        return std_dev

    if dates is not None:
        for date in pd.unique(dates):
            mask   = dates == date
            std_dev= calc_for_segment(mask)
            for i in [1, 2, 3]:
                bands[f'upper_{i}'][mask] = vwap[mask] + i * std_dev
                bands[f'lower_{i}'][mask] = vwap[mask] - i * std_dev
    else:
        std_dev = calc_for_segment(pd.Series([True]*len(df), index=df.index))
        for i in [1, 2, 3]:
            bands[f'upper_{i}'] = vwap + i * std_dev
            bands[f'lower_{i}'] = vwap - i * std_dev

    return bands


def get_vwap_result(df: pd.DataFrame) -> Optional[VWAPResult]:
    """
    Get current VWAP analysis including position, bands, and setup signals.
    """
    if df is None or len(df) < 20:
        return None

    try:
        vwap   = calculate_vwap(df)
        bands  = calculate_vwap_bands(df, vwap)
        
        current_price = float(df['close'].iloc[-1])
        current_vwap  = float(vwap.iloc[-1])
        
        if current_vwap <= 0:
            return None

        # Standard deviation at current bar
        upper_1 = float(bands['upper_1'].iloc[-1])
        upper_2 = float(bands['upper_2'].iloc[-1])
        upper_3 = float(bands['upper_3'].iloc[-1])
        lower_1 = float(bands['lower_1'].iloc[-1])
        lower_2 = float(bands['lower_2'].iloc[-1])
        lower_3 = float(bands['lower_3'].iloc[-1])

        # How many σ from VWAP?
        band_width = upper_1 - current_vwap
        deviation  = (current_price - current_vwap) / band_width if band_width > 0 else 0

        # Position relative to VWAP
        if current_price > current_vwap * 1.0005:
            position = 'above'
        elif current_price < current_vwap * 0.9995:
            position = 'below'
        else:
            position = 'at'

        # VWAP Reclaim: was below VWAP, now above — bullish
        prev_close = float(df['close'].iloc[-2]) if len(df) > 2 else current_price
        prev_vwap  = float(vwap.iloc[-2]) if len(df) > 2 else current_vwap
        reclaim    = prev_close < prev_vwap and current_price > current_vwap

        # VWAP Rejection: was above VWAP, now below — bearish
        rejection  = prev_close > prev_vwap and current_price < current_vwap

        # Score: 0-100 for setup quality
        score = 50  # neutral
        if position == 'above':
            score += 10
            if reclaim:         score += 20  # fresh reclaim = strong bull signal
            if deviation > 2:   score -= 20  # too extended above VWAP
            if deviation < 0.5: score += 10  # close to VWAP = good entry
        elif position == 'below':
            score -= 10
            if rejection:       score -= 20  # fresh rejection = strong bear signal
            if deviation < -2:  score -= 20  # too extended below VWAP
            if deviation > -0.5:score -= 10  # just below VWAP

        score = max(0, min(100, score))

        return VWAPResult(
            vwap=round(current_vwap, 2),
            upper_1=round(upper_1, 2), upper_2=round(upper_2, 2), upper_3=round(upper_3, 2),
            lower_1=round(lower_1, 2), lower_2=round(lower_2, 2), lower_3=round(lower_3, 2),
            deviation=round(deviation, 2),
            position=position,
            reclaim=reclaim,
            rejection=rejection,
            score=score,
        )

    except Exception as e:
        return None


# =============================================================
#  ANCHORED VWAP
# =============================================================

def calculate_anchored_vwap(df: pd.DataFrame, anchor_idx: int) -> Optional[AnchoredVWAP]:
    """
    Anchored VWAP from a specific bar (swing high, swing low, key event).
    More relevant than session VWAP for swing trades.
    """
    if df is None or anchor_idx >= len(df) or anchor_idx < 0:
        return None

    try:
        segment = df.iloc[anchor_idx:]
        if len(segment) < 2:
            return None

        tp  = (segment['high'] + segment['low'] + segment['close']) / 3
        vol = segment['volume'].replace(0, np.nan).fillna(1)

        cum_tpv = (tp * vol).cumsum()
        cum_vol = vol.cumsum()
        avwap   = cum_tpv / cum_vol

        current_avwap = float(avwap.iloc[-1])
        current_price = float(segment['close'].iloc[-1])

        # Simple 1σ band
        variance = ((tp - avwap) ** 2 * vol).cumsum() / cum_vol
        std_dev  = float(np.sqrt(max(0, variance.iloc[-1])))

        return AnchoredVWAP(
            anchor_price = round(float(df['close'].iloc[anchor_idx]), 2),
            anchor_dt    = str(df.index[anchor_idx]),
            vwap         = round(current_avwap, 2),
            upper_1      = round(current_avwap + std_dev, 2),
            lower_1      = round(current_avwap - std_dev, 2),
            bars_since   = len(segment) - 1,
            price_above  = current_price > current_avwap,
        )

    except Exception:
        return None


def find_key_anchors(df: pd.DataFrame, lookback: int = 100) -> Dict[str, int]:
    """
    Find key anchor points for AVWAP:
    - Most recent swing high
    - Most recent swing low
    - Start of current trend
    """
    if df is None or len(df) < 20:
        return {}

    window = df.tail(lookback)
    anchors = {}

    # Swing high: highest high in lookback
    swing_high_idx = window['high'].idxmax()
    anchors['swing_high'] = df.index.get_loc(swing_high_idx)

    # Swing low: lowest low in lookback
    swing_low_idx = window['low'].idxmin()
    anchors['swing_low'] = df.index.get_loc(swing_low_idx)

    # Recent momentum anchor: 20 bars ago
    anchors['momentum'] = max(0, len(df) - 20)

    return anchors


# =============================================================
#  VOLUME PROFILE
# =============================================================

def calculate_volume_profile(
    df: pd.DataFrame,
    n_bins: int = 100,
    value_area_pct: float = 0.70
) -> Optional[VolumeProfile]:
    """
    Volume Profile: Shows volume distribution by price level.
    
    POC = price level with most volume traded
    Value Area = price range containing 70% of volume
    HVN = High Volume Nodes (support/resistance)
    LVN = Low Volume Nodes (fast-move zones)
    """
    if df is None or len(df) < 20:
        return None

    try:
        price_min = float(df['low'].min())
        price_max = float(df['high'].max())
        if price_max <= price_min:
            return None

        # Create price bins
        bins     = np.linspace(price_min, price_max, n_bins + 1)
        bin_size = (price_max - price_min) / n_bins
        profile  = np.zeros(n_bins)

        # Distribute volume across price range for each bar
        for _, row in df.iterrows():
            bar_low  = float(row['low'])
            bar_high = float(row['high'])
            bar_vol  = float(row['volume']) if row['volume'] > 0 else 1

            # Find bins this bar touches
            low_bin  = max(0, int((bar_low  - price_min) / bin_size))
            high_bin = min(n_bins - 1, int((bar_high - price_min) / bin_size))

            if high_bin == low_bin:
                profile[low_bin] += bar_vol
            else:
                n_touched = high_bin - low_bin + 1
                vol_per_bin = bar_vol / n_touched
                for b in range(low_bin, high_bin + 1):
                    profile[b] += vol_per_bin

        total_vol = profile.sum()
        if total_vol <= 0:
            return None

        # POC = bin with highest volume
        poc_bin   = int(np.argmax(profile))
        poc_price = round(price_min + (poc_bin + 0.5) * bin_size, 2)

        # Value Area: expand from POC until 70% of volume captured
        target_vol = total_vol * value_area_pct
        va_vol     = profile[poc_bin]
        va_low_bin = poc_bin
        va_high_bin= poc_bin

        while va_vol < target_vol:
            # Expand to whichever side has more volume
            can_go_up   = va_high_bin < n_bins - 1
            can_go_down = va_low_bin  > 0
            if not can_go_up and not can_go_down:
                break
            up_vol   = profile[va_high_bin + 1] if can_go_up   else -1
            down_vol = profile[va_low_bin  - 1] if can_go_down else -1
            if up_vol >= down_vol:
                va_high_bin += 1
                va_vol      += up_vol
            else:
                va_low_bin  -= 1
                va_vol      += down_vol

        vah = round(price_min + (va_high_bin + 1) * bin_size, 2)
        val = round(price_min + va_low_bin * bin_size, 2)

        # HVN: bins with volume > 1.5x average
        avg_vol = total_vol / n_bins
        hvn_levels = []
        lvn_levels = []
        for i, v in enumerate(profile):
            price_level = round(price_min + (i + 0.5) * bin_size, 2)
            if v > avg_vol * 1.5:
                hvn_levels.append(price_level)
            elif v < avg_vol * 0.5 and v > 0:
                lvn_levels.append(price_level)

        # Cluster nearby levels (within 0.1% of price)
        def cluster_levels(levels, threshold_pct=0.2):
            if not levels:
                return []
            levels = sorted(levels)
            clustered = [levels[0]]
            for lv in levels[1:]:
                if lv - clustered[-1] > clustered[-1] * threshold_pct / 100:
                    clustered.append(lv)
                else:
                    clustered[-1] = (clustered[-1] + lv) / 2
            return [round(l, 2) for l in clustered]

        hvn_levels = cluster_levels(hvn_levels)[-10:]  # top 10 HVNs
        lvn_levels = cluster_levels(lvn_levels)[-10:]

        current_price = float(df['close'].iloc[-1])

        return VolumeProfile(
            poc             = poc_price,
            vah             = vah,
            val             = val,
            value_area_pct  = round(va_vol / total_vol * 100, 1),
            hvn_levels      = hvn_levels,
            lvn_levels      = lvn_levels,
            profile         = {round(price_min + (i+0.5)*bin_size, 2): round(v, 0)
                               for i, v in enumerate(profile) if v > 0},
            current_in_va   = val <= current_price <= vah,
            current_above_poc= current_price > poc_price,
        )

    except Exception as e:
        return None


def get_nearest_vp_level(price: float, vp: VolumeProfile, max_dist_pct: float = 0.3) -> Optional[Dict]:
    """
    Find the nearest significant volume profile level to current price.
    Returns level type (POC/VAH/VAL/HVN) and distance.
    """
    if vp is None:
        return None

    levels = [
        ('POC', vp.poc),
        ('VAH', vp.vah),
        ('VAL', vp.val),
    ]
    for hvn in vp.hvn_levels:
        levels.append(('HVN', hvn))
    for lvn in vp.lvn_levels:
        levels.append(('LVN', lvn))

    nearest = None
    min_dist = float('inf')

    for name, level in levels:
        dist_pct = abs(price - level) / price * 100
        if dist_pct < min_dist:
            min_dist  = dist_pct
            nearest   = {'name': name, 'level': level, 'dist_pct': round(dist_pct, 3)}

    if nearest and nearest['dist_pct'] <= max_dist_pct:
        return nearest
    return None


# =============================================================
#  VOLUME DELTA
# =============================================================

def calculate_volume_delta(df: pd.DataFrame) -> Optional[VolumeDelta]:
    """
    Volume Delta: Estimate buying vs selling pressure.
    
    Approximation using bar characteristics:
    - Bull bar (close > open): majority buying
    - Bear bar (close < open): majority selling
    - Upper wick: selling pressure
    - Lower wick: buying pressure
    
    True tick-level delta requires Level 2 data (not available via yfinance).
    This is a reliable approximation used by retail traders.
    """
    if df is None or len(df) < 10:
        return None

    try:
        buy_vol  = []
        sell_vol = []

        for _, row in df.iterrows():
            o, h, l, c = float(row['open']), float(row['high']), float(row['low']), float(row['close'])
            v = float(row['volume']) if row['volume'] > 0 else 0
            bar_range = h - l if h > l else 0.0001

            # Bull bar: more buying
            if c >= o:
                body_pct  = (c - o) / bar_range
                upper_wick= (h - c) / bar_range
                lower_wick= (o - l) / bar_range
                buy_ratio = 0.5 + body_pct * 0.3 + lower_wick * 0.2
            # Bear bar: more selling
            else:
                body_pct  = (o - c) / bar_range
                upper_wick= (h - o) / bar_range
                lower_wick= (c - l) / bar_range
                buy_ratio = 0.5 - body_pct * 0.3 - upper_wick * 0.2

            buy_ratio  = max(0.1, min(0.9, buy_ratio))
            buy_vol.append(v * buy_ratio)
            sell_vol.append(v * (1 - buy_ratio))

        buy_arr  = np.array(buy_vol)
        sell_arr = np.array(sell_vol)

        # Recent (last 10 bars) delta
        recent_buy  = buy_arr[-10:].sum()
        recent_sell = sell_arr[-10:].sum()
        delta       = recent_buy - recent_sell
        total_vol   = recent_buy + recent_sell

        # Cumulative delta
        cum_delta = (buy_arr - sell_arr).sum()

        # Divergence: price trending one way, delta trending other
        price_change = float(df['close'].iloc[-1]) - float(df['close'].iloc[-10])
        delta_divergence = (price_change > 0 and delta < 0) or (price_change < 0 and delta > 0)

        # Absorption: large volume, small price move
        recent_atr = float(df['close'].tail(10).std()) if len(df) >= 10 else 1
        price_move = abs(price_change)
        avg_vol    = buy_arr[-10:].mean() + sell_arr[-10:].mean()
        absorption = avg_vol > (buy_arr.mean() + sell_arr.mean()) * 1.5 and price_move < recent_atr * 0.5

        # Score: 0=bearish, 50=neutral, 100=bullish
        if total_vol > 0:
            buy_pct  = recent_buy / total_vol * 100
            sell_pct = 100 - buy_pct
            score    = int(buy_pct)
            if delta_divergence: score = 100 - score  # flip on divergence
        else:
            buy_pct = sell_pct = 50
            score = 50

        return VolumeDelta(
            delta            = round(delta, 0),
            cum_delta        = round(cum_delta, 0),
            delta_divergence = delta_divergence,
            buying_pressure  = round(buy_pct, 1),
            selling_pressure = round(sell_pct, 1),
            absorption       = absorption,
            score            = max(0, min(100, score)),
        )

    except Exception:
        return None


# =============================================================
#  COMBINED VWAP SCORING FOR DEEP EDGE
# =============================================================

def score_vwap_confluence(df_5min: pd.DataFrame, df_15min: pd.DataFrame,
                           direction: str) -> Tuple[int, Dict]:
    """
    Score VWAP/Volume confluence for a given trade direction.
    Returns (score_0_to_20, details_dict)
    
    Max 20 points:
    - VWAP position alignment:     5pts
    - VWAP reclaim/rejection:      5pts
    - Volume profile confluence:   5pts
    - Volume delta alignment:      5pts
    """
    score   = 0
    details = {}
    is_long = direction in ('long', 'bullish')

    # Use 5min for VWAP (more granular), 15min for volume profile (more meaningful)
    primary_df = df_5min if df_5min is not None and len(df_5min) >= 20 else df_15min
    if primary_df is None or len(primary_df) < 20:
        return 0, {}

    # --- VWAP Position (5pts) ---
    vwap_result = get_vwap_result(primary_df)
    if vwap_result:
        details['vwap'] = vwap_result.vwap
        details['vwap_position'] = vwap_result.position
        details['vwap_deviation'] = vwap_result.deviation

        if is_long and vwap_result.position == 'above':
            score += 5
            details['vwap_aligned'] = True
        elif not is_long and vwap_result.position == 'below':
            score += 5
            details['vwap_aligned'] = True
        elif vwap_result.position == 'at':
            score += 2
        else:
            details['vwap_aligned'] = False

        # --- VWAP Reclaim/Rejection (5pts) ---
        if is_long and vwap_result.reclaim:
            score += 5
            details['vwap_reclaim'] = True
        elif not is_long and vwap_result.rejection:
            score += 5
            details['vwap_rejection'] = True
        # Extended beyond 2σ — mean reversion potential
        elif is_long and vwap_result.deviation < -2:
            score += 2
            details['vwap_extended_below'] = True
        elif not is_long and vwap_result.deviation > 2:
            score += 2
            details['vwap_extended_above'] = True

    # --- Volume Profile (5pts) ---
    vp_df = df_15min if df_15min is not None and len(df_15min) >= 30 else primary_df
    if vp_df is not None and len(vp_df) >= 30:
        vp = calculate_volume_profile(vp_df.tail(200))
        if vp:
            details['poc'] = vp.poc
            details['vah'] = vp.vah
            details['val'] = vp.val

            current_price = float(vp_df['close'].iloc[-1])
            nearest       = get_nearest_vp_level(current_price, vp, max_dist_pct=0.5)

            if nearest:
                details['nearest_vp_level'] = nearest

            # Long: price at/above POC, in value area or just below VAL (discount)
            if is_long:
                if vp.current_above_poc:
                    score += 3
                    details['above_poc'] = True
                if current_price <= vp.val * 1.002:  # at/below VAL = discount entry
                    score += 2
                    details['at_val_support'] = True
                elif vp.current_in_va:
                    score += 1
            # Short: price at/below POC, above VAH (premium)
            else:
                if not vp.current_above_poc:
                    score += 3
                    details['below_poc'] = True
                if current_price >= vp.vah * 0.998:  # at/above VAH = premium entry
                    score += 2
                    details['at_vah_resistance'] = True
                elif vp.current_in_va:
                    score += 1

    # --- Volume Delta (5pts) ---
    if primary_df is not None and len(primary_df) >= 10:
        vd = calculate_volume_delta(primary_df)
        if vd:
            details['buying_pressure'] = vd.buying_pressure
            details['delta_divergence'] = vd.delta_divergence
            details['absorption'] = vd.absorption

            if is_long:
                if vd.buying_pressure > 60:
                    score += 3
                    details['strong_buying'] = True
                elif vd.buying_pressure > 50:
                    score += 1
                if vd.delta_divergence:
                    score -= 2  # price up but selling — warning
                    details['delta_warning'] = True
                if vd.absorption and vd.buying_pressure > 50:
                    score += 2
                    details['buying_absorption'] = True
            else:
                if vd.selling_pressure > 60:
                    score += 3
                    details['strong_selling'] = True
                elif vd.selling_pressure > 50:
                    score += 1
                if vd.delta_divergence:
                    score -= 2
                    details['delta_warning'] = True
                if vd.absorption and vd.selling_pressure > 50:
                    score += 2
                    details['selling_absorption'] = True

    return max(0, min(20, score)), details


# =============================================================
#  QUICK SUMMARY FOR TELEGRAM ALERTS
# =============================================================

def get_vwap_summary(df: pd.DataFrame) -> str:
    """Returns a one-line VWAP summary for Telegram alerts"""
    result = get_vwap_result(df)
    if not result:
        return 'VWAP: N/A'

    pos_emoji = '📈' if result.position == 'above' else '📉' if result.position == 'below' else '➡️'
    signal = ''
    if result.reclaim:   signal = ' ✅ RECLAIM'
    if result.rejection: signal = ' ❌ REJECTION'

    return (f"VWAP: {result.vwap:.0f} | Price {result.position} "
            f"({result.deviation:+.1f}σ){signal} {pos_emoji}")
