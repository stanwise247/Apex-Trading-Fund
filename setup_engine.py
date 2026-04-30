"""
APEX Setup Engine — setup_engine.py
=====================================
Session 3: Full gate system across all timeframes.

Gates:
  1. HTF Bias      (4hour)
  2. Structure     (1hour BOS/CHoCH)
  3. POI           (1hour OB/Sweep/Breaker)
  4. Session       (UTC time window)
  5. Entry Trigger (15min)
  6. Confirmation  (5min displacement)

Run directly to verify on real data:
  python3 setup_engine.py

LOG_TRADE CONTRACT (enforced in server.py):
  Required signal keys: symbol, direction, setup, mode, entry, stop, target, rr, session, quality
  Order:  log_trade(sig) FIRST — then send_telegram()
  On failure: raises RuntimeError (trade_tracker.py) → caught as CRITICAL with exc_info=True
  setup names: 'A_sweep_ob', 'B_choch_breaker', 'C_bos_ob'
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
from zoneinfo import ZoneInfo
from datetime import datetime, timezone

from market_structure import (
    load_bars, find_swings, detect_structure,
    compute_bias, get_prev_day, get_opening_ranges
)
from liquidity_engine import (
    find_order_blocks, find_fvgs, find_sweeps,
    find_breakers, calc_atr
)

UTC   = ZoneInfo('UTC')
NY_TZ = ZoneInfo('America/New_York')


def _load_htf(symbol: str, tf: str, limit_htf: int) -> pd.DataFrame:
    """Load 5min bars and resample to tf in-memory — bypasses broken HTF DB rows."""
    mult = 48 if tf == '4hour' else 12   # 5min bars per HTF bar
    df5  = load_bars(symbol, '5min', limit=limit_htf * mult)
    rule = '4h' if tf == '4hour' else '1h'
    return df5.resample(rule).agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    ).dropna()

# ─────────────────────────────────────────────────────────────
#  SESSION WINDOWS (UTC hours)
# ─────────────────────────────────────────────────────────────

SESSION_WINDOWS = {
    'MNQ': [
        {'name': 'London Primary',  'start':  7, 'end': 11, 'quality': 'primary'},
        {'name': 'NY Secondary',    'start': 13, 'end': 20, 'quality': 'secondary'},
    ],
    'NQ': [  # kept for legacy data reference
        {'name': 'London Primary',  'start':  7, 'end': 11, 'quality': 'primary'},
        {'name': 'NY Secondary',    'start': 13, 'end': 20, 'quality': 'secondary'},
    ],
    'ES': [
        {'name': 'London Primary',  'start':  7, 'end': 11, 'quality': 'primary'},
        {'name': 'NY Secondary',    'start': 13, 'end': 19, 'quality': 'secondary'},
    ],
    'GC': [
        {'name': 'NY Primary',      'start': 12, 'end': 17, 'quality': 'primary'},
        {'name': 'Asia Optional',   'start':  0, 'end':  2, 'quality': 'secondary'},
    ],
}

# ─────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class GateResult:
    passed:  bool
    gate:    int
    name:    str
    detail:  str

    def __repr__(self):
        icon = '✅' if self.passed else '❌'
        return f"{icon} Gate {self.gate} [{self.name}]: {self.detail}"


@dataclass
class SetupResult:
    symbol:    str
    direction: str       # 'long' or 'short'
    setup:     str       # 'A_sweep_ob' | 'B_choch_breaker' | 'C_bos_ob'
    valid:     bool
    gates:     list
    entry:     Optional[float]
    stop:      Optional[float]
    target:    Optional[float]
    rr:        Optional[float]
    session:   str
    quality:   str       # 'primary' | 'secondary'
    timestamp: Optional[pd.Timestamp]

    def __repr__(self):
        if not self.valid:
            failed = [g for g in self.gates if not g.passed]
            return (f"❌ {self.symbol} {self.direction.upper()} [{self.setup}] "
                    f"FAILED at {failed[0]}" if failed else f"❌ {self.symbol} NO SETUP")
        return (f"✅ {self.symbol} {self.direction.upper()} [{self.setup}] "
                f"entry={self.entry:.2f} stop={self.stop:.2f} "
                f"target={self.target:.2f} RR={self.rr:.1f}x "
                f"[{self.quality.upper()}]")


# ─────────────────────────────────────────────────────────────
#  GATE 1 — HTF BIAS (4hour)
# ─────────────────────────────────────────────────────────────

def gate1_htf_bias(symbol: str, direction: str) -> GateResult:
    df = _load_htf(symbol, '4hour', 500)
    if len(df) < 21:
        return GateResult(False, 1, 'HTF Bias',
                          f'4hour bias=NEUTRAL — insufficient bars ({len(df)})')
    ema20      = df['close'].ewm(span=20, adjust=False).mean()
    last_close = float(df['close'].iloc[-1])
    last_ema   = float(ema20.iloc[-1])
    if last_close > last_ema * 1.001:
        bias = 'bullish'
    elif last_close < last_ema * 0.999:
        bias = 'bearish'
    else:
        bias = 'neutral'

    if bias == 'neutral':
        return GateResult(False, 1, 'HTF Bias',
                          f'4hour bias=NEUTRAL (close={last_close:.2f} EMA20={last_ema:.2f})')

    if direction == 'long' and bias != 'bullish':
        return GateResult(False, 1, 'HTF Bias',
                          f'4hour bias={bias.upper()} — no long')

    if direction == 'short' and bias != 'bearish':
        return GateResult(False, 1, 'HTF Bias',
                          f'4hour bias={bias.upper()} — no short')

    return GateResult(True, 1, 'HTF Bias',
                      f'4hour bias={bias.upper()} (close={last_close:.2f} EMA20={last_ema:.2f})')


# ─────────────────────────────────────────────────────────────
#  GATE 2 — STRUCTURE CONFIRMATION (1hour)
# ─────────────────────────────────────────────────────────────

def gate2_structure(symbol: str, direction: str) -> GateResult:
    df = _load_htf(symbol, '1hour', 500)
    sh, sl    = find_swings(df, lookback=5)
    events, _ = detect_structure(df, sh, sl)

    if not events:
        return GateResult(False, 2, 'Structure',
                          'No structure events on 1hour')

    last = events[-1]

    if direction == 'long' and 'BULL' not in last.event_type:
        return GateResult(False, 2, 'Structure',
                          f'Last event={last.event_type} — need BULL event for long')

    if direction == 'short' and 'BEAR' not in last.event_type:
        return GateResult(False, 2, 'Structure',
                          f'Last event={last.event_type} — need BEAR event for short')

    return GateResult(True, 2, 'Structure',
                      f'Last event={last.event_type} @ '
                      f'{last.timestamp.astimezone(NY_TZ).strftime("%m-%d %H:%M")} ET')


# ─────────────────────────────────────────────────────────────
#  GATE 3 — POI (1hour)
# ─────────────────────────────────────────────────────────────

def gate3_poi(symbol: str, direction: str) -> tuple[GateResult, str, object]:
    """
    Returns (GateResult, setup_type, poi_object)
    setup_type: 'A_sweep_ob' | 'B_choch_breaker' | 'C_bos_ob'
    """
    df = _load_htf(symbol, '1hour', 500)
    sh, sl    = find_swings(df, lookback=5)
    events, _ = detect_structure(df, sh, sl)
    obs       = find_order_blocks(df, sh, sl, events)
    breakers  = find_breakers(obs)
    atr_series= calc_atr(df)

    price     = float(df['close'].iloc[-1])
    atr_val   = float(atr_series.iloc[-1])

    # Setup A — delegate to gate3_poi_setup_a to avoid duplicating logic
    g3a, stype_a, poi_a = gate3_poi_setup_a(symbol, direction)
    if g3a.passed:
        return g3a, stype_a, poi_a

    # Setup B — CHoCH + Breaker
    choch_events = [e for e in events
                    if 'CHOCH' in e.event_type
                    and e.index >= len(df) - 48]  # last 48 bars
    for evt in reversed(choch_events):
        if direction == 'long'  and 'BULL' not in evt.event_type: continue
        if direction == 'short' and 'BEAR' not in evt.event_type: continue
        nearby_breakers = [b for b in breakers
                           if b.kind == ('bull' if direction == 'long' else 'bear')
                           and abs((b.high + b.low) / 2 - price) < atr_val * 2]
        if nearby_breakers:
            br = nearby_breakers[0]
            return (
                GateResult(True, 3, 'POI',
                           f'Setup B: CHoCH@{evt.level:.2f} + breaker [{br.low:.2f}-{br.high:.2f}]'),
                'B_choch_breaker', br
            )

    # Setup C — BOS + OB pullback
    bos_events = [e for e in events
                  if 'BOS' in e.event_type
                  and e.index >= len(df) - 48]
    for evt in reversed(bos_events):
        if direction == 'long'  and 'BULL' not in evt.event_type: continue
        if direction == 'short' and 'BEAR' not in evt.event_type: continue
        nearby_obs = [o for o in obs
                      if not o.broken
                      and o.kind == ('bull' if direction == 'long' else 'bear')
                      and abs((o.high + o.low) / 2 - price) < atr_val * 1.5]
        if nearby_obs:
            ob = nearby_obs[0]
            return (
                GateResult(True, 3, 'POI',
                           f'Setup C: BOS@{evt.level:.2f} + OB [{ob.low:.2f}-{ob.high:.2f}]'),
                'C_bos_ob', ob
            )

    return (
        GateResult(False, 3, 'POI',
                   f'No valid POI within range | price={price:.2f} atr={atr_val:.2f}'),
        'none', None
    )

# ─────────────────────────────────────────────────────────────
#  GATE 3 — POI SETUP A (Sweep + OB)
# ─────────────────────────────────────────────────────────────

def gate3_poi_setup_a(symbol: str, direction: str) -> tuple:
    """
    Setup A: Liquidity sweep detected within last 24 bars
    + active OB within 1 ATR of current price.
    Returns (GateResult, setup_type, poi_object)
    """
    df = _load_htf(symbol, '1hour', 500)
    sh, sl    = find_swings(df, lookback=5)
    events, _ = detect_structure(df, sh, sl)
    obs       = find_order_blocks(df, sh, sl, events)
    sweeps    = find_sweeps(df, sh, sl)
    atr_series= calc_atr(df)

    price   = float(df['close'].iloc[-1])
    atr_val = float(atr_series.iloc[-1])

    # Recent sweeps — last 24 bars
    recent_sweeps = [s for s in sweeps if s.index >= len(df) - 24]

    for sweep in reversed(recent_sweeps):
        if direction == 'long'  and sweep.kind != 'bull': continue
        if direction == 'short' and sweep.kind != 'bear': continue

        # Find active OB in same direction within 1 ATR
        target_kind = 'bull' if direction == 'long' else 'bear'
        nearby_obs = [o for o in obs
                      if not o.broken
                      and o.kind == target_kind
                      and abs((o.high + o.low) / 2 - price) < atr_val]

        if nearby_obs:
            ob = min(nearby_obs, key=lambda o: abs((o.high + o.low) / 2 - price))
            return (
                GateResult(True, 3, 'POI',
                           f'Setup A: sweep@{sweep.swept_level:.2f} '
                           f'atr={sweep.atr_size:.1f}x + OB [{ob.low:.2f}-{ob.high:.2f}]'),
                'A_sweep_ob', ob
            )

    return (
        GateResult(False, 3, 'POI',
                   f'No sweep+OB confluence | price={price:.2f} atr={atr_val:.2f}'),
        'none', None
    )



# ─────────────────────────────────────────────────────────────
#  GATE 4 — SESSION WINDOW (UTC)
# ─────────────────────────────────────────────────────────────

def gate4_session(symbol: str, dt: datetime = None) -> tuple[GateResult, str]:
    if dt is None:
        dt = datetime.now(timezone.utc)
    hour = dt.hour

    windows = SESSION_WINDOWS.get(symbol, SESSION_WINDOWS['NQ'])
    for w in windows:
        if w['start'] <= hour < w['end']:
            return (
                GateResult(True, 4, 'Session',
                           f"{w['name']} ({w['start']}:00-{w['end']}:00 UTC) [{w['quality'].upper()}]"),
                w['quality']
            )

    return (
        GateResult(False, 4, 'Session',
                   f'Hour {hour:02d}:00 UTC outside all trading windows'),
        'none'
    )


# ─────────────────────────────────────────────────────────────
#  GATE 5 — ENTRY TRIGGER (15min)
# ─────────────────────────────────────────────────────────────

def gate5_entry_trigger(symbol: str, direction: str, poi) -> GateResult:
    df  = load_bars(symbol, '15min', limit=100)
    # ATR from 1hour to match backtest (backtester.py passes atr_1h to check_entry_trigger_15m)
    df_1h = _load_htf(symbol, '1hour', 50)
    atr   = calc_atr(df_1h).iloc[-1]

    if poi is None:
        return GateResult(False, 5, 'Entry Trigger', 'No POI passed from Gate 3')

    price     = float(df['close'].iloc[-1])
    prev_high = float(df['high'].iloc[-2])
    prev_low  = float(df['low'].iloc[-2])
    poi_mid   = (poi.high + poi.low) / 2

    # Price must be at or inside the POI zone
    at_poi = poi.low - float(atr) * 0.5 <= price <= poi.high + float(atr) * 0.5

    if not at_poi:
        return GateResult(False, 5, 'Entry Trigger',
                          f'Price {price:.2f} not at POI [{poi.low:.2f}-{poi.high:.2f}]')

    if direction == 'long':
        # 15min close above prior 15min high
        current_close = float(df['close'].iloc[-1])
        if current_close > prev_high:
            return GateResult(True, 5, 'Entry Trigger',
                              f'15min close {current_close:.2f} > prior high {prev_high:.2f}')
        return GateResult(False, 5, 'Entry Trigger',
                          f'15min close {current_close:.2f} not above prior high {prev_high:.2f}')

    else:  # short
        current_close = float(df['close'].iloc[-1])
        if current_close < prev_low:
            return GateResult(True, 5, 'Entry Trigger',
                              f'15min close {current_close:.2f} < prior low {prev_low:.2f}')
        return GateResult(False, 5, 'Entry Trigger',
                          f'15min close {current_close:.2f} not below prior low {prev_low:.2f}')


# ─────────────────────────────────────────────────────────────
#  GATE 6 — CONFIRMATION (5min displacement)
# ─────────────────────────────────────────────────────────────

def gate6_confirmation(symbol: str, direction: str) -> GateResult:
    df  = load_bars(symbol, '5min', limit=50)
    atr = calc_atr(df)

    last        = df.iloc[-1]
    atr_val     = float(atr.iloc[-1])
    body        = abs(float(last['close']) - float(last['open']))
    body_ratio  = body / atr_val if atr_val > 0 else 0

    is_bull_candle = float(last['close']) > float(last['open'])
    is_bear_candle = float(last['close']) < float(last['open'])

    # Check not entering into opposing FVG
    fvgs = find_fvgs(df, min_size_atr=0.3)
    price = float(last['close'])
    opposing_fvg = False
    for fvg in fvgs:
        if not fvg.filled:
            if direction == 'long'  and fvg.kind == 'bear' and fvg.low <= price <= fvg.high:
                opposing_fvg = True
            if direction == 'short' and fvg.kind == 'bull' and fvg.low <= price <= fvg.high:
                opposing_fvg = True

    if opposing_fvg:
        return GateResult(False, 6, 'Confirmation',
                          f'Price inside opposing 5min FVG')

    if body_ratio < 0.6:
        return GateResult(False, 6, 'Confirmation',
                          f'5min body ratio {body_ratio:.2f} < 0.60 — weak candle')

    if direction == 'long' and not is_bull_candle:
        return GateResult(False, 6, 'Confirmation',
                          f'5min candle is bearish — no long confirmation')

    if direction == 'short' and not is_bear_candle:
        return GateResult(False, 6, 'Confirmation',
                          f'5min candle is bullish — no short confirmation')

    return GateResult(True, 6, 'Confirmation',
                      f'5min displacement body={body_ratio:.2f}x ATR in {direction} direction')


# ─────────────────────────────────────────────────────────────
#  ENTRY / STOP / TARGET CALCULATOR
# ─────────────────────────────────────────────────────────────

def calc_trade_levels(symbol: str, direction: str, poi, mode: str = 'swing',
                      current_price: float = None) -> tuple:
    """
    Returns (entry, stop, target, rr)
    Entry: current_price (actual fill) if provided, else POI edge
    Stop:  beyond POI by 0.8 ATR
    Target: RR based on mode (scalp=2.5, swing=4.0)
    """
    df      = _load_htf(symbol, '1hour', 50)
    atr_val = float(calc_atr(df).iloc[-1])
    rr_map  = {'scalp': 2.5, 'swing': 4.0}
    rr      = rr_map.get(mode, 3.0)

    if poi is None:
        return None, None, None, None

    if direction == 'long':
        entry  = round(current_price if current_price is not None else poi.high, 2)
        stop   = round(poi.low - atr_val * 0.8, 2)
        risk   = entry - stop
        target = round(entry + risk * rr, 2)
    else:
        entry  = round(current_price if current_price is not None else poi.low, 2)
        stop   = round(poi.high + atr_val * 0.8, 2)
        risk   = stop - entry
        target = round(entry - risk * rr, 2)

    actual_rr = round(abs(target - entry) / abs(stop - entry), 1) if abs(stop - entry) > 0 else 0
    return entry, stop, target, actual_rr


# ─────────────────────────────────────────────────────────────
#  MAIN — run all gates for a symbol/direction
# ─────────────────────────────────────────────────────────────

def check_setup(symbol: str, direction: str, mode: str = 'swing',
                dt: datetime = None) -> SetupResult:
    gates    = []
    poi      = None
    setup_type = 'none'

    # Gate 1
    g1 = gate1_htf_bias(symbol, direction)
    gates.append(g1)
    if not g1.passed:
        return SetupResult(symbol, direction, 'none', False, gates,
                           None, None, None, None, '', '', None)

    # Gate 2
    g2 = gate2_structure(symbol, direction)
    gates.append(g2)
    if not g2.passed:
        return SetupResult(symbol, direction, 'none', False, gates,
                           None, None, None, None, '', '', None)

    # Gate 3
    g3, setup_type, poi = gate3_poi(symbol, direction)
    gates.append(g3)
    if not g3.passed:
        return SetupResult(symbol, direction, setup_type, False, gates,
                           None, None, None, None, '', '', None)

    # Gate 4
    g4, quality = gate4_session(symbol, dt)
    gates.append(g4)
    if not g4.passed:
        return SetupResult(symbol, direction, setup_type, False, gates,
                           None, None, None, None, '', quality, None)

    # Gate 5
    g5 = gate5_entry_trigger(symbol, direction, poi)
    gates.append(g5)
    if not g5.passed:
        return SetupResult(symbol, direction, setup_type, False, gates,
                           None, None, None, None, g4.detail, quality, None)

    # Gate 6
    g6 = gate6_confirmation(symbol, direction)
    gates.append(g6)
    if not g6.passed:
        return SetupResult(symbol, direction, setup_type, False, gates,
                           None, None, None, None, g4.detail, quality, None)

    # All gates passed — calculate levels
    df_5m = load_bars(symbol, '5min', limit=2)
    ts    = df_5m.index[-1]
    current_price = float(df_5m['close'].iloc[-1])
    entry, stop, target, rr = calc_trade_levels(symbol, direction, poi, mode, current_price)

    return SetupResult(
        symbol    = symbol,
        direction = direction,
        setup     = setup_type,
        valid     = True,
        gates     = gates,
        entry     = entry,
        stop      = stop,
        target    = target,
        rr        = rr,
        session   = g4.detail,
        quality   = quality,
        timestamp = ts,
    )



def check_setup_a(symbol: str, direction: str, mode: str = 'swing',
                  dt: datetime = None) -> SetupResult:
    """Run all 6 gates for Setup A — Sweep + OB."""
    gates      = []
    poi        = None
    setup_type = 'none'

    g1 = gate1_htf_bias(symbol, direction)
    gates.append(g1)
    if not g1.passed:
        return SetupResult(symbol, direction, 'none', False, gates,
                           None, None, None, None, '', '', None)

    g2 = gate2_structure(symbol, direction)
    gates.append(g2)
    if not g2.passed:
        return SetupResult(symbol, direction, 'none', False, gates,
                           None, None, None, None, '', '', None)

    g3, setup_type, poi = gate3_poi_setup_a(symbol, direction)
    gates.append(g3)
    if not g3.passed:
        return SetupResult(symbol, direction, setup_type, False, gates,
                           None, None, None, None, '', '', None)

    g4, quality = gate4_session(symbol, dt)
    gates.append(g4)
    if not g4.passed:
        return SetupResult(symbol, direction, setup_type, False, gates,
                           None, None, None, None, '', quality, None)

    g5 = gate5_entry_trigger(symbol, direction, poi)
    gates.append(g5)
    if not g5.passed:
        return SetupResult(symbol, direction, setup_type, False, gates,
                           None, None, None, None, g4.detail, quality, None)

    g6 = gate6_confirmation(symbol, direction)
    gates.append(g6)
    if not g6.passed:
        return SetupResult(symbol, direction, setup_type, False, gates,
                           None, None, None, None, g4.detail, quality, None)

    df_5m = load_bars(symbol, '5min', limit=2)
    ts    = df_5m.index[-1]
    current_price = float(df_5m['close'].iloc[-1])
    entry, stop, target, rr = calc_trade_levels(symbol, direction, poi, mode, current_price)

    return SetupResult(
        symbol    = symbol,
        direction = direction,
        setup     = setup_type,
        valid     = True,
        gates     = gates,
        entry     = entry,
        stop      = stop,
        target    = target,
        rr        = rr,
        session   = g4.detail,
        quality   = quality,
        timestamp = ts,
    )


def gate3_poi_setup_c(symbol, direction):
    df = _load_htf(symbol, '1hour', 500)
    sh, sl    = find_swings(df, lookback=5)
    events, _ = detect_structure(df, sh, sl)
    obs       = find_order_blocks(df, sh, sl, events)
    atr_series= calc_atr(df)
    price   = float(df['close'].iloc[-1])
    atr_val = float(atr_series.iloc[-1])
    bos_events = [e for e in events if 'BOS' in e.event_type and e.index >= len(df) - 48]
    for evt in reversed(bos_events):
        if direction == 'long'  and 'BULL' not in evt.event_type: continue
        if direction == 'short' and 'BEAR' not in evt.event_type: continue
        target_kind = 'bull' if direction == 'long' else 'bear'
        nearby_obs = [o for o in obs if not o.broken and o.kind == target_kind
                      and abs((o.high + o.low) / 2 - price) < atr_val * 1.5]
        if nearby_obs:
            ob = min(nearby_obs, key=lambda o: abs((o.high + o.low) / 2 - price))
            return (GateResult(True, 3, 'POI',
                    f'Setup C: BOS@{evt.level:.2f} + OB [{ob.low:.2f}-{ob.high:.2f}]'),
                    'C_bos_ob', ob)
    return (GateResult(False, 3, 'POI',
            f'No BOS+OB pullback | price={price:.2f} atr={atr_val:.2f}'),
            'none', None)


def check_setup_c(symbol, direction, mode='swing', dt=None):
    gates = []
    # Block 07:00–09:59 UTC — 68% loss rate in this window, negative expectancy in backtest
    _hour = (dt or datetime.now(timezone.utc)).hour
    if 7 <= _hour <= 9:
        gates.append(GateResult(False, 0, 'TimeFilter', 'Setup C blocked 07-09 UTC (low-edge window)'))
        return SetupResult(symbol, direction, 'none', False, gates, None, None, None, None, '', '', None)
    g1 = gate1_htf_bias(symbol, direction)
    gates.append(g1)
    if not g1.passed:
        return SetupResult(symbol, direction, 'none', False, gates, None, None, None, None, '', '', None)
    g2 = gate2_structure(symbol, direction)
    gates.append(g2)
    if not g2.passed:
        return SetupResult(symbol, direction, 'none', False, gates, None, None, None, None, '', '', None)
    # Check event type directly from events list — do not parse g2.detail string
    _df_1h = _load_htf(symbol, '1hour', 500)
    _sh, _sl = find_swings(_df_1h, lookback=5)
    _events, _ = detect_structure(_df_1h, _sh, _sl)
    _last_event = _events[-1] if _events else None
    if _last_event and 'CHOCH' in _last_event.event_type:
        gates.append(GateResult(False, 2, 'Structure', 'Setup C requires BOS not CHoCH'))
        return SetupResult(symbol, direction, 'none', False, gates, None, None, None, None, '', '', None)
    g3, setup_type, poi = gate3_poi_setup_c(symbol, direction)
    gates.append(g3)
    if not g3.passed:
        return SetupResult(symbol, direction, setup_type, False, gates, None, None, None, None, '', '', None)
    g4, quality = gate4_session(symbol, dt)
    gates.append(g4)
    if not g4.passed:
        return SetupResult(symbol, direction, setup_type, False, gates, None, None, None, None, '', quality, None)
    g5 = gate5_entry_trigger(symbol, direction, poi)
    gates.append(g5)
    if not g5.passed:
        return SetupResult(symbol, direction, setup_type, False, gates, None, None, None, None, g4.detail, quality, None)
    g6 = gate6_confirmation(symbol, direction)
    gates.append(g6)
    if not g6.passed:
        return SetupResult(symbol, direction, setup_type, False, gates, None, None, None, None, g4.detail, quality, None)
    df_5m = load_bars(symbol, '5min', limit=2)
    ts    = df_5m.index[-1]
    current_price = float(df_5m['close'].iloc[-1])
    entry, stop, target, rr = calc_trade_levels(symbol, direction, poi, mode, current_price)
    return SetupResult(symbol=symbol, direction=direction, setup=setup_type, valid=True,
                       gates=gates, entry=entry, stop=stop, target=target, rr=rr,
                       session=g4.detail, quality=quality, timestamp=ts)

# ─────────────────────────────────────────────────────────────
#  SETUP E — EMA50 PULLBACK  (canonical logic lives in setup_e.py)
# ─────────────────────────────────────────────────────────────

def check_setup_e(symbol: str, direction: str,
                  dt: datetime = None) -> 'SetupResult':
    """
    Thin wrapper — delegates to setup_e.py (canonical implementation).
    Adapts ESetupResult → SetupResult for compatibility with scan_all / server.py.
    """
    from setup_e import check_setup_e as _check_e
    e = _check_e(symbol, direction, dt)
    # Convert EGateResult list → GateResult list
    gates = [GateResult(g.passed, g.gate, g.name, g.detail) for g in e.gates]
    return SetupResult(
        symbol    = e.symbol,
        direction = e.direction,
        setup     = e.setup,
        valid     = e.valid,
        gates     = gates,
        entry     = e.entry,
        stop      = e.stop,
        target    = e.target,
        rr        = e.rr,
        session   = 'NY Primary (13-18 UTC)',
        quality   = 'primary',
        timestamp = e.timestamp,
    )


def scan_all(dt: datetime = None) -> list[SetupResult]:
    """Scan all instruments and directions for all setups."""
    results   = []
    triggered = set()

    for symbol in ('MNQ', 'ES', 'GC'):
        for direction in ('long', 'short'):
            key = (symbol, direction)
            for mode in ('swing',):
                r = check_setup(symbol, direction, mode, dt)
                if r.valid and key not in triggered:
                    results.append(r); triggered.add(key); continue
                r = check_setup_a(symbol, direction, mode, dt)
                if r.valid and key not in triggered:
                    results.append(r); triggered.add(key); continue
                r = check_setup_c(symbol, direction, mode, dt)
                if r.valid and key not in triggered:
                    results.append(r); triggered.add(key)

    # Setup E — EMA50 Pullback (MNQ primary; independent of swing setups)
    for direction in ('long', 'short'):
        try:
            r = check_setup_e('MNQ', direction, dt)
            if r.valid:
                results.append(r)
        except Exception:
            pass

    return results


# ─────────────────────────────────────────────────────────────
#  VERIFY — run directly
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"  APEX Setup Engine — Gate Check")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"{'='*60}")

    for symbol in ('MNQ', 'ES', 'GC'):
        print(f"\n  {symbol}")
        print(f"  {'-'*40}")
        for direction in ('long', 'short'):
            result = check_setup(symbol, direction, mode='swing')
            print(f"\n  {direction.upper()}:")
            for g in result.gates:
                print(f"    {g}")
            if result.valid:
                print(f"\n  >>> {result}")

    print(f"\n{'='*60}")
    print(f"  Valid setups right now:")
    valid = scan_all()
    if valid:
        for r in valid:
            print(f"  {r}")
    else:
        print(f"  None — market conditions don't meet all gates currently")
    print(f"{'='*60}\n")
