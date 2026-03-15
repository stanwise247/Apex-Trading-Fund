"""
APEX Liquidity Engine — liquidity_engine.py
=============================================
Session 2: Order Blocks, FVGs, Liquidity Sweeps, Breaker Blocks

Builds on top of market_structure.py.
All detection uses real OHLCV data from apex_market.db.

Run directly to verify on real data:
  python3 liquidity_engine.py
"""

import sqlite3
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
from zoneinfo import ZoneInfo
from market_structure import load_bars, find_swings, detect_structure, compute_bias

DB_PATH = 'apex_market.db'
NY_TZ   = ZoneInfo('America/New_York')


# ─────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class OrderBlock:
    index:      int
    timestamp:  pd.Timestamp
    kind:       str        # 'bull' or 'bear'
    high:       float
    low:        float
    open:       float
    close:      float
    broken:     bool = False
    tested:     int  = 0

    def __repr__(self):
        ts  = self.timestamp.astimezone(NY_TZ).strftime('%Y-%m-%d %H:%M')
        tag = ' [BROKEN]' if self.broken else ''
        return (f"OB [{self.kind.upper()}] {ts} ET | "
                f"H={self.high:.2f} L={self.low:.2f}{tag}")


@dataclass
class FairValueGap:
    index:      int
    timestamp:  pd.Timestamp
    kind:       str        # 'bull' or 'bear'
    high:       float      # top of gap
    low:        float      # bottom of gap
    size:       float      # gap size in points
    filled:     bool = False

    def __repr__(self):
        ts  = self.timestamp.astimezone(NY_TZ).strftime('%Y-%m-%d %H:%M')
        tag = ' [FILLED]' if self.filled else ''
        return (f"FVG [{self.kind.upper()}] {ts} ET | "
                f"H={self.high:.2f} L={self.low:.2f} size={self.size:.2f}{tag}")


@dataclass
class LiquiditySweep:
    index:      int
    timestamp:  pd.Timestamp
    kind:       str        # 'bull' (swept lows, long setup) or 'bear' (swept highs, short setup)
    swept_level: float
    sweep_low:  float      # actual wick low
    sweep_high: float      # actual wick high
    close:      float      # close back inside = confirmation
    atr_size:   float      # sweep size in ATR units
    volume_spike: bool

    def __repr__(self):
        ts = self.timestamp.astimezone(NY_TZ).strftime('%Y-%m-%d %H:%M')
        vs = ' VOL!' if self.volume_spike else ''
        return (f"SWEEP [{self.kind.upper()}] {ts} ET | "
                f"level={self.swept_level:.2f} atr={self.atr_size:.2f}x{vs}")


@dataclass
class BreakerBlock:
    index:      int
    timestamp:  pd.Timestamp
    kind:       str        # 'bull' (former bear OB now support) or 'bear' (former bull OB now resistance)
    high:       float
    low:        float
    source_ob:  OrderBlock

    def __repr__(self):
        ts = self.timestamp.astimezone(NY_TZ).strftime('%Y-%m-%d %H:%M')
        return (f"BREAKER [{self.kind.upper()}] {ts} ET | "
                f"H={self.high:.2f} L={self.low:.2f}")


# ─────────────────────────────────────────────────────────────
#  ATR HELPER
# ─────────────────────────────────────────────────────────────

def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high  = df['high']
    low   = df['low']
    close = df['close'].shift(1)
    tr    = pd.concat([
        high - low,
        (high - close).abs(),
        (low  - close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ─────────────────────────────────────────────────────────────
#  ORDER BLOCKS
# ─────────────────────────────────────────────────────────────

def find_order_blocks(df: pd.DataFrame, swing_highs, swing_lows, structure_events) -> list[OrderBlock]:
    """
    An Order Block is the last opposing candle before a BOS.

    Bullish OB: last bearish (red) candle before a BOS_BULL or CHOCH_BULL
    Bearish OB: last bullish (green) candle before a BOS_BEAR or CHOCH_BEAR

    These are the candles where institutions placed their orders.
    Price often returns to these zones before continuing.
    """
    obs = []

    for event in structure_events:
        i = event.index

        if 'BULL' in event.event_type:
            # Look back for last bearish candle before this BOS
            for j in range(i - 1, max(0, i - 20), -1):
                candle_open  = df['open'].iloc[j]
                candle_close = df['close'].iloc[j]
                if candle_close < candle_open:  # bearish candle
                    obs.append(OrderBlock(
                        index     = j,
                        timestamp = df.index[j],
                        kind      = 'bull',
                        high      = round(float(df['high'].iloc[j]), 2),
                        low       = round(float(df['low'].iloc[j]),  2),
                        open      = round(float(candle_open), 2),
                        close     = round(float(candle_close), 2),
                    ))
                    break

        elif 'BEAR' in event.event_type:
            # Look back for last bullish candle before this BOS
            for j in range(i - 1, max(0, i - 20), -1):
                candle_open  = df['open'].iloc[j]
                candle_close = df['close'].iloc[j]
                if candle_close > candle_open:  # bullish candle
                    obs.append(OrderBlock(
                        index     = j,
                        timestamp = df.index[j],
                        kind      = 'bear',
                        high      = round(float(df['high'].iloc[j]), 2),
                        low       = round(float(df['low'].iloc[j]),  2),
                        open      = round(float(candle_open), 2),
                        close     = round(float(candle_close), 2),
                    ))
                    break

    # Deduplicate — remove OBs at same index
    seen = set()
    deduped = []
    for ob in obs:
        if ob.index not in seen:
            seen.add(ob.index)
            deduped.append(ob)

    # Mark broken OBs — price closed through them
    closes = df['close'].values
    for ob in deduped:
        for i in range(ob.index + 1, len(df)):
            if ob.kind == 'bull' and closes[i] < ob.low:
                ob.broken = True
                break
            elif ob.kind == 'bear' and closes[i] > ob.high:
                ob.broken = True
                break

    return sorted(deduped, key=lambda x: x.index)


# ─────────────────────────────────────────────────────────────
#  FAIR VALUE GAPS
# ─────────────────────────────────────────────────────────────

def find_fvgs(df: pd.DataFrame, min_size_atr: float = 0.5) -> list[FairValueGap]:
    """
    A Fair Value Gap (FVG) is a 3-candle imbalance.

    Bullish FVG: candle[i-1].low > candle[i+1].high
      — gap between bottom of candle 1 and top of candle 3
      — price left inefficiency moving up

    Bearish FVG: candle[i-1].high < candle[i+1].low  (wait — let me be precise)
      Bearish FVG: candle[i+1].low > candle[i-1].high
      — Actually: gap[high] = candle[i-1].low, gap[low] = candle[i+1].high
      — For bearish: gap[high] = candle[i+1].low, gap[low] = candle[i-1].high

    Minimum size filter: gap must be at least min_size_atr * ATR to matter.
    """
    atr    = calc_atr(df)
    fvgs   = []

    for i in range(1, len(df) - 1):
        atr_val = atr.iloc[i]
        if pd.isna(atr_val) or atr_val == 0:
            continue

        low_prev  = df['low'].iloc[i-1]
        high_prev = df['high'].iloc[i-1]
        low_next  = df['low'].iloc[i+1]
        high_next = df['high'].iloc[i+1]

        # Bullish FVG: gap between candle[i-1] low and candle[i+1] high
        # candle[i-1].low > candle[i+1].high — price gapped up
        if low_prev > high_next:
            gap_high = round(float(low_prev), 2)
            gap_low  = round(float(high_next), 2)
            size     = round(gap_high - gap_low, 2)
            if size >= atr_val * min_size_atr:
                fvgs.append(FairValueGap(
                    index     = i,
                    timestamp = df.index[i],
                    kind      = 'bull',
                    high      = gap_high,
                    low       = gap_low,
                    size      = size,
                ))

        # Bearish FVG: candle[i+1].low > candle[i-1].high — price gapped down
        if high_prev < low_next:
            gap_high = round(float(low_next), 2)
            gap_low  = round(float(high_prev), 2)
            size     = round(gap_high - gap_low, 2)
            if size >= atr_val * min_size_atr:
                fvgs.append(FairValueGap(
                    index     = i,
                    timestamp = df.index[i],
                    kind      = 'bear',
                    high      = gap_high,
                    low       = gap_low,
                    size      = size,
                ))

    # Mark filled FVGs — price traded back through them
    for fvg in fvgs:
        for i in range(fvg.index + 1, len(df)):
            if fvg.kind == "bull" and df["high"].iloc[i] >= (fvg.high + fvg.low) / 2:
                fvg.filled = True
                break
            elif fvg.kind == "bear" and df["low"].iloc[i] <= (fvg.high + fvg.low) / 2:
                fvg.filled = True
                break

    return fvgs


# ─────────────────────────────────────────────────────────────
#  LIQUIDITY SWEEPS
# ─────────────────────────────────────────────────────────────

def find_sweeps(df: pd.DataFrame, swing_highs, swing_lows) -> list[LiquiditySweep]:
    """
    A liquidity sweep occurs when price wicks through a swing point
    then closes back inside — trapping traders who entered the breakout.

    Bull sweep: wick below swing low, close back above it = long setup
    Bear sweep: wick above swing high, close back below it = short setup

    Minimum sweep size: 0.3 ATR beyond the level.
    """
    atr     = calc_atr(df)
    sweeps  = []
    avg_vol = df['volume'].rolling(20).mean()

    # Index swing points for fast lookup
    sh_prices = {sp.index: sp.price for sp in swing_highs}
    sl_prices = {sp.index: sp.price for sp in swing_lows}

    # Check each bar for sweeps of prior swing points
    for i in range(20, len(df)):
        atr_val = atr.iloc[i]
        if pd.isna(atr_val) or atr_val == 0:
            continue

        bar_high  = df['high'].iloc[i]
        bar_low   = df['low'].iloc[i]
        bar_close = df['close'].iloc[i]
        vol       = df['volume'].iloc[i]
        avg_v     = avg_vol.iloc[i]
        vol_spike = not pd.isna(avg_v) and vol > avg_v * 1.3

        # Check sweep of swing highs (bear sweep — short setup)
        for sh in swing_highs:
            if sh.index >= i:
                continue
            level = sh.price
            if (bar_high > level + atr_val * 1.0 and
                    bar_close < level):
                sweep_size = round((bar_high - level) / atr_val, 2)
                sweeps.append(LiquiditySweep(
                    index        = i,
                    timestamp    = df.index[i],
                    kind         = 'bear',
                    swept_level  = level,
                    sweep_low    = round(float(bar_low),   2),
                    sweep_high   = round(float(bar_high),  2),
                    close        = round(float(bar_close), 2),
                    atr_size     = sweep_size,
                    volume_spike = vol_spike,
                ))
                break

        # Check sweep of swing lows (bull sweep — long setup)
        for sl in swing_lows:
            if sl.index >= i:
                continue
            level = sl.price
            if (bar_low < level - atr_val * 1.0 and
                    bar_close > level):
                sweep_size = round((level - bar_low) / atr_val, 2)
                sweeps.append(LiquiditySweep(
                    index        = i,
                    timestamp    = df.index[i],
                    kind         = 'bull',
                    swept_level  = level,
                    sweep_low    = round(float(bar_low),   2),
                    sweep_high   = round(float(bar_high),  2),
                    close        = round(float(bar_close), 2),
                    atr_size     = sweep_size,
                    volume_spike = vol_spike,
                ))
                break

    # Deduplicate — one sweep per bar AND per level
    seen_bars   = set()
    seen_levels = set()
    deduped = []
    for s in sweeps:
        level_key = round(s.swept_level, 0)
        if s.index not in seen_bars and level_key not in seen_levels:
            seen_bars.add(s.index)
            seen_levels.add(level_key)
            deduped.append(s)

    return sorted(deduped, key=lambda x: x.index)


# ─────────────────────────────────────────────────────────────
#  BREAKER BLOCKS
# ─────────────────────────────────────────────────────────────

def find_breakers(order_blocks: list[OrderBlock]) -> list[BreakerBlock]:
    """
    A Breaker Block is a broken Order Block that has flipped polarity.

    Broken bull OB → becomes resistance (bear breaker)
    Broken bear OB → becomes support (bull breaker)

    These are high-probability areas for price to react on retest.
    """
    breakers = []
    for ob in order_blocks:
        if ob.broken:
            kind = 'bear' if ob.kind == 'bull' else 'bull'
            breakers.append(BreakerBlock(
                index     = ob.index,
                timestamp = ob.timestamp,
                kind      = kind,
                high      = ob.high,
                low       = ob.low,
                source_ob = ob,
            ))
    return breakers


# ─────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────

def analyse_liquidity(symbol: str, timeframe: str = '1hour') -> dict:
    """
    Main entry point. Returns all liquidity levels for a symbol.
    """
    from market_structure import INSTRUMENT_SETTINGS, DEFAULT_SETTINGS
    cfg = INSTRUMENT_SETTINGS.get(symbol, DEFAULT_SETTINGS)

    df = load_bars(symbol, timeframe, limit=500)
    swing_highs, swing_lows = find_swings(df, lookback=5)
    events, _   = detect_structure(df, swing_highs, swing_lows)

    obs      = find_order_blocks(df, swing_highs, swing_lows, events)
    fvgs     = find_fvgs(df)
    sweeps   = find_sweeps(df, swing_highs, swing_lows)
    breakers = find_breakers(obs)

    # Active only (not broken/filled)
    active_obs      = [o for o in obs      if not o.broken]
    active_fvgs     = [f for f in fvgs     if not f.filled]
    active_breakers = breakers  # breakers are always relevant

    return {
        'symbol':          symbol,
        'timeframe':       timeframe,
        'order_blocks':    obs,
        'active_obs':      active_obs,
        'fvgs':            fvgs,
        'active_fvgs':     active_fvgs,
        'sweeps':          sweeps,
        'breakers':        breakers,
        'active_breakers': active_breakers,
    }


def nearest_ob(obs: list[OrderBlock], price: float, n: int = 3) -> list[OrderBlock]:
    """Return N active order blocks closest to current price."""
    active = [o for o in obs if not o.broken]
    return sorted(active, key=lambda o: abs((o.high + o.low) / 2 - price))[:n]


def nearest_fvg(fvgs: list[FairValueGap], price: float, n: int = 3) -> list[FairValueGap]:
    """Return N unfilled FVGs closest to current price."""
    active = [f for f in fvgs if not f.filled]
    return sorted(active, key=lambda f: abs((f.high + f.low) / 2 - price))[:n]


# ─────────────────────────────────────────────────────────────
#  VERIFY — run directly to check output on real data
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    for symbol in ('NQ', 'ES', 'GC'):
        print(f"\n{'='*60}")
        print(f"  {symbol} — 1hour liquidity")
        print(f"{'='*60}")

        df = load_bars(symbol, '1hour', limit=500)
        atr_val = round(float(calc_atr(df).iloc[-1]), 2)
        price   = round(float(df['close'].iloc[-1]), 2)
        print(f"\n  Last close : {price}  |  ATR(14) = {atr_val}")

        sh, sl  = find_swings(df, lookback=5)
        events, _ = detect_structure(df, sh, sl)

        obs      = find_order_blocks(df, sh, sl, events)
        fvgs     = find_fvgs(df)
        sweeps   = find_sweeps(df, sh, sl)
        breakers = find_breakers(obs)

        active_obs  = [o for o in obs  if not o.broken]
        active_fvgs = [f for f in fvgs if not f.filled]

        print(f"\n  Order Blocks : {len(obs)} total | {len(active_obs)} active")
        print(f"  Nearest 3 to price:")
        for o in nearest_ob(obs, price, 3):
            print(f"    {o}")

        print(f"\n  Fair Value Gaps : {len(fvgs)} total | {len(active_fvgs)} active")
        print(f"  Nearest 3 to price:")
        for f in nearest_fvg(fvgs, price, 3):
            print(f"    {f}")

        print(f"\n  Liquidity Sweeps : {len(sweeps)} total | last 3:")
        for s in sweeps[-3:]:
            print(f"    {s}")

        print(f"\n  Breaker Blocks : {len(breakers)} total | last 3:")
        for b in breakers[-3:]:
            print(f"    {b}")

    print(f"\n{'='*60}")
    print("  Done. Verify the output before Session 3.")
    print(f"{'='*60}\n")
