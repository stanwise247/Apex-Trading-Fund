"""
APEX Backtester — backtester.py
=================================
Session 4: Bar-by-bar replay, Setup B only (CHoCH + Breaker).

Rules:
  - Signal:  5min displacement candle closes (Gate 6)
  - Entry:   Next 5min bar open (no lookahead)
  - Stop:    Beyond POI by 0.5x 1hour ATR
  - Target:  4.0R swing / 2.5R scalp
  - Exit:    Stop hit | Target hit | Session end | Max 100 bars
  - One trade at a time per instrument
  - All HTF gates checked using only data visible at signal bar

Run:
  python3 backtester.py --symbol NQ --mode swing
  python3 backtester.py --symbol all --mode swing
"""

import sqlite3
import argparse
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from zoneinfo import ZoneInfo
from datetime import datetime, timezone

DB_PATH = 'apex_market.db'
NY_TZ   = ZoneInfo('America/New_York')
UTC     = ZoneInfo('UTC')

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────

SESSION_WINDOWS = {
    'NQ': [{'start': 13, 'end': 19, 'quality': 'primary'},
           {'start':  7, 'end': 11, 'quality': 'secondary'}],
    'ES': [{'start': 13, 'end': 19, 'quality': 'primary'},
           {'start':  7, 'end': 11, 'quality': 'secondary'}],
    'GC': [{'start': 12, 'end': 17, 'quality': 'primary'},
           {'start':  0, 'end':  2, 'quality': 'secondary'}],
    'CL': [{'start': 14, 'end': 20, 'quality': 'primary'},
           {'start':  7, 'end': 11, 'quality': 'secondary'}],
}

RR_MAP = {'swing': 4.0, 'scalp': 2.5}

# Must match live_scanner.py:DAY_FILTERS exactly so backtest trades the same days
DAY_FILTERS = {
    'NQ': [0, 2, 3, 4],   # Mon Wed Thu Fri — no Tue
    'ES': [0, 1, 3],       # Mon Tue Thu     — no Wed Fri
    'GC': [0, 1, 2, 3],   # Mon-Thu          — no Fri
}

SESSION_END_UTC = {
    'NQ': 19, 'ES': 19, 'GC': 17, 'CL': 20
}

# ─────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_all_bars(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load all bars for a symbol/timeframe."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        'SELECT ts, open, high, low, close, volume FROM ohlcv '
        'WHERE symbol=? AND timeframe=? ORDER BY ts ASC',
        conn, params=(symbol, timeframe)
    )
    conn.close()
    if df.empty:
        raise ValueError(f"No data for {symbol} {timeframe}")
    df['datetime'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    df.set_index('datetime', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df


# ─────────────────────────────────────────────────────────────
#  INDICATOR HELPERS (no external imports — self contained)
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


def find_swings(df: pd.DataFrame, lookback: int = 5):
    highs, lows = [], []
    n = lookback
    h = df['high'].values
    l = df['low'].values
    for i in range(n, len(df) - n):
        is_sh = all(h[i] > h[i-j] for j in range(1, n+1)) and \
                all(h[i] > h[i+j] for j in range(1, n+1))
        is_sl = all(l[i] < l[i-j] for j in range(1, n+1)) and \
                all(l[i] < l[i+j] for j in range(1, n+1))
        if is_sh:
            highs.append((i, round(float(h[i]), 2)))
        if is_sl:
            lows.append((i, round(float(l[i]), 2)))
    return highs, lows


def detect_structure(closes, swing_highs, swing_lows):
    """Returns list of (index, type, level) events."""
    events = []
    all_sh = sorted(swing_highs, key=lambda s: s[0])
    all_sl = sorted(swing_lows,  key=lambda s: s[0])
    sh_ptr = sl_ptr = 0
    pending_sh = pending_sl = None
    broken_sh  = broken_sl  = set()
    bias = 'neutral'

    for i in range(len(closes)):
        c = closes[i]
        while sh_ptr < len(all_sh) and all_sh[sh_ptr][0] < i:
            pending_sh = all_sh[sh_ptr]
            sh_ptr += 1
        while sl_ptr < len(all_sl) and all_sl[sl_ptr][0] < i:
            pending_sl = all_sl[sl_ptr]
            sl_ptr += 1

        if pending_sh and pending_sh[0] not in broken_sh and c > pending_sh[1]:
            etype = 'BOS_BULL' if bias in ('bullish', 'neutral') else 'CHOCH_BULL'
            events.append((i, etype, pending_sh[1]))
            broken_sh.add(pending_sh[0])
            bias = 'bullish'

        if pending_sl and pending_sl[0] not in broken_sl and c < pending_sl[1]:
            etype = 'BOS_BEAR' if bias in ('bearish', 'neutral') else 'CHOCH_BEAR'
            events.append((i, etype, pending_sl[1]))
            broken_sl.add(pending_sl[0])
            bias = 'bearish'

    return events, bias


def find_order_blocks(df_slice, events):
    """Find order blocks from structure events."""
    obs = []
    opens  = df_slice['open'].values
    closes = df_slice['close'].values
    highs  = df_slice['high'].values
    lows   = df_slice['low'].values

    for idx, etype, level in events:
        if 'BULL' in etype:
            for j in range(idx - 1, max(0, idx - 20), -1):
                if closes[j] < opens[j]:  # bearish candle
                    obs.append({
                        'idx': j, 'kind': 'bull',
                        'high': round(float(highs[j]), 2),
                        'low':  round(float(lows[j]),  2),
                        'broken': False
                    })
                    break
        elif 'BEAR' in etype:
            for j in range(idx - 1, max(0, idx - 20), -1):
                if closes[j] > opens[j]:  # bullish candle
                    obs.append({
                        'idx': j, 'kind': 'bear',
                        'high': round(float(highs[j]), 2),
                        'low':  round(float(lows[j]),  2),
                        'broken': False
                    })
                    break

    # Deduplicate
    seen = set()
    deduped = []
    for ob in obs:
        if ob['idx'] not in seen:
            seen.add(ob['idx'])
            deduped.append(ob)

    # Mark broken
    for ob in deduped:
        for i in range(ob['idx'] + 1, len(closes)):
            if ob['kind'] == 'bull' and closes[i] < ob['low']:
                ob['broken'] = True
                break
            elif ob['kind'] == 'bear' and closes[i] > ob['high']:
                ob['broken'] = True
                break

    return deduped


def find_breakers(obs):
    return [
        {'idx': ob['idx'], 'kind': 'bear' if ob['kind'] == 'bull' else 'bull',
         'high': ob['high'], 'low': ob['low']}
        for ob in obs if ob['broken']
    ]


def find_sweeps(df_slice, swing_highs, swing_lows):
    atr    = calc_atr(df_slice).values
    sweeps = []
    highs  = df_slice['high'].values
    lows   = df_slice['low'].values
    closes = df_slice['close'].values
    vols   = df_slice['volume'].values
    avg_vol= pd.Series(vols).rolling(20).mean().values
    seen_levels = set()

    for i in range(20, len(df_slice)):
        atr_val = atr[i]
        if np.isnan(atr_val) or atr_val == 0:
            continue

        for sh_idx, sh_price in swing_highs:
            if sh_idx >= i:
                continue
            level_key = round(sh_price, 0)
            if level_key in seen_levels:
                continue
            if (highs[i] > sh_price + atr_val * 1.0 and closes[i] < sh_price):
                vol_spike = not np.isnan(avg_vol[i]) and vols[i] > avg_vol[i] * 1.3
                sweeps.append({'idx': i, 'kind': 'bear', 'level': sh_price,
                               'atr_size': (highs[i] - sh_price) / atr_val,
                               'vol_spike': vol_spike})
                seen_levels.add(level_key)
                break

        for sl_idx, sl_price in swing_lows:
            if sl_idx >= i:
                continue
            level_key = round(sl_price, 0)
            if level_key in seen_levels:
                continue
            if (lows[i] < sl_price - atr_val * 1.0 and closes[i] > sl_price):
                vol_spike = not np.isnan(avg_vol[i]) and vols[i] > avg_vol[i] * 1.3
                sweeps.append({'idx': i, 'kind': 'bull', 'level': sl_price,
                               'atr_size': (sl_price - lows[i]) / atr_val,
                               'vol_spike': vol_spike})
                seen_levels.add(level_key)
                break

    return sweeps


# ─────────────────────────────────────────────────────────────
#  GATE CHECKS (historical — no lookahead)
# ─────────────────────────────────────────────────────────────

def check_htf_bias(df_4h_slice, direction: str) -> bool:
    """Gate 1: 4hour bias matches direction."""
    if len(df_4h_slice) < 30:
        return False
    sh, sl    = find_swings(df_4h_slice, lookback=5)
    events, _ = detect_structure(df_4h_slice['close'].values, sh, sl)
    if not events:
        return False
    window = events[-10:]
    bull   = sum(1 for e in window if 'BULL' in e[1])
    bear   = sum(1 for e in window if 'BEAR' in e[1])
    if bull == bear:
        return False
    bias = 'bullish' if bull > bear else 'bearish'
    return (direction == 'long' and bias == 'bullish') or \
           (direction == 'short' and bias == 'bearish')


def check_structure(df_1h_slice, direction: str):
    """Gate 2: last 1hour event matches direction. Returns (bool, events)."""
    if len(df_1h_slice) < 20:
        return False, []
    sh, sl    = find_swings(df_1h_slice, lookback=5)
    events, _ = detect_structure(df_1h_slice['close'].values, sh, sl)
    if not events:
        return False, []
    last = events[-1]
    if direction == 'long'  and 'BULL' not in last[1]: return False, events
    if direction == 'short' and 'BEAR' not in last[1]: return False, events
    return True, events


def check_poi_setup_b(df_1h_slice, direction: str, price: float, atr_val: float):
    """
    Gate 3 Setup B: CHoCH + Breaker within 2 ATR of current price.
    Returns (bool, poi_dict or None)
    """
    if len(df_1h_slice) < 20:
        return False, None

    sh, sl    = find_swings(df_1h_slice, lookback=5)
    events, _ = detect_structure(df_1h_slice['close'].values, sh, sl)
    obs       = find_order_blocks(df_1h_slice, events)
    breakers  = find_breakers(obs)

    # Find recent CHoCH in our direction
    choch_events = [(i, t, l) for i, t, l in events
                    if 'CHOCH' in t and i >= len(df_1h_slice) - 48]

    for evt in reversed(choch_events):
        if direction == 'long'  and 'BULL' not in evt[1]: continue
        if direction == 'short' and 'BEAR' not in evt[1]: continue

        target_kind = 'bull' if direction == 'long' else 'bear'
        nearby = [b for b in breakers
                  if b['kind'] == target_kind
                  and abs((b['high'] + b['low']) / 2 - price) < atr_val * 2]

        if nearby:
            # Pick closest breaker
            poi = min(nearby, key=lambda b: abs((b['high'] + b['low']) / 2 - price))
            return True, poi

    return False, None



def check_poi_setup_a(df_1h_slice, direction: str, price: float, atr_val: float):
    """
    Gate 3 Setup A: Sweep + OB within 1 ATR of current price.
    Returns (bool, poi_dict or None)
    """
    if len(df_1h_slice) < 20:
        return False, None

    sh, sl    = find_swings(df_1h_slice, lookback=5)
    events, _ = detect_structure(df_1h_slice['close'].values, sh, sl)
    obs       = find_order_blocks(df_1h_slice, events)
    sweeps    = find_sweeps(df_1h_slice, sh, sl)

    recent_sweeps = [s for s in sweeps if s['idx'] >= len(df_1h_slice) - 24]

    for sweep in reversed(recent_sweeps):
        if direction == 'long'  and sweep['kind'] != 'bull': continue
        if direction == 'short' and sweep['kind'] != 'bear': continue

        target_kind = 'bull' if direction == 'long' else 'bear'
        nearby = [o for o in obs
                  if not o['broken']
                  and o['kind'] == target_kind
                  and abs((o['high'] + o['low']) / 2 - price) < atr_val]

        if nearby:
            poi = min(nearby, key=lambda o: abs((o['high'] + o['low']) / 2 - price))
            return True, poi

    return False, None



def check_poi_setup_c(df_1h_slice, direction, price, atr_val):
    """Gate 3 Setup C: BOS + OB pullback within 1.5 ATR."""
    if len(df_1h_slice) < 20:
        return False, None
    sh, sl    = find_swings(df_1h_slice, lookback=5)
    events, _ = detect_structure(df_1h_slice['close'].values, sh, sl)
    obs       = find_order_blocks(df_1h_slice, events)
    bos_events = [(i,t,l) for i,t,l in events
                  if 'BOS' in t and i >= len(df_1h_slice) - 48]
    for evt in reversed(bos_events):
        if direction == 'long'  and 'BULL' not in evt[1]: continue
        if direction == 'short' and 'BEAR' not in evt[1]: continue
        target_kind = 'bull' if direction == 'long' else 'bear'
        nearby = [o for o in obs
                  if not o['broken']
                  and o['kind'] == target_kind
                  and abs((o['high'] + o['low']) / 2 - price) < atr_val * 1.5]
        if nearby:
            poi = min(nearby, key=lambda o: abs((o['high'] + o['low']) / 2 - price))
            return True, poi
    return False, None


def check_session(symbol: str, dt) -> tuple[bool, str]:
    """Gate 4: is current time in a valid session window?"""
    hour = dt.hour
    for w in SESSION_WINDOWS.get(symbol, []):
        if w['start'] <= hour < w['end']:
            return True, w['quality']
    return False, 'none'


def check_entry_trigger_15m(df_15m_slice, direction: str, poi: dict, atr_val: float) -> bool:
    """Gate 5: price at POI on 15min."""
    if len(df_15m_slice) < 3 or poi is None:
        return False
    price     = float(df_15m_slice['close'].iloc[-1])
    prev_high = float(df_15m_slice['high'].iloc[-2])
    prev_low  = float(df_15m_slice['low'].iloc[-2])
    at_poi    = poi['low'] - atr_val * 0.5 <= price <= poi['high'] + atr_val * 0.5
    if not at_poi:
        return False
    if direction == 'long'  and price > prev_high: return True
    if direction == 'short' and price < prev_low:  return True
    return False


def check_confirmation_5m(df_5m_slice, direction: str) -> bool:
    """Gate 6: 5min displacement candle."""
    if len(df_5m_slice) < 15:
        return False
    last    = df_5m_slice.iloc[-1]
    atr_val = calc_atr(df_5m_slice).iloc[-1]
    if np.isnan(atr_val) or atr_val == 0:
        return False
    body       = abs(float(last['close']) - float(last['open']))
    body_ratio = body / atr_val
    if body_ratio < 0.6:
        return False
    if direction == 'long'  and float(last['close']) <= float(last['open']): return False
    if direction == 'short' and float(last['close']) >= float(last['open']): return False
    return True


# ─────────────────────────────────────────────────────────────
#  TRADE DATA CLASS
# ─────────────────────────────────────────────────────────────

@dataclass
class Trade:
    symbol:      str
    direction:   str
    setup:       str
    mode:        str
    entry_time:  pd.Timestamp
    entry_price: float
    stop:        float
    target:      float
    rr_planned:  float
    exit_time:   Optional[pd.Timestamp] = None
    exit_price:  Optional[float]        = None
    exit_reason: str                    = ''
    pnl_r:       Optional[float]        = None
    winner:      Optional[bool]         = None
    session:     str                    = ''
    quality:     str                    = ''
    bars_held:   int                    = 0

    def close(self, exit_time, exit_price, reason, bars_held):
        self.exit_time   = exit_time
        self.exit_price  = exit_price
        self.exit_reason = reason
        self.bars_held   = bars_held
        risk = abs(self.entry_price - self.stop)
        if risk > 0:
            if self.direction == 'long':
                self.pnl_r = round((exit_price - self.entry_price) / risk, 3)
            else:
                self.pnl_r = round((self.entry_price - exit_price) / risk, 3)
        self.winner = self.pnl_r > 0 if self.pnl_r is not None else False

    def to_dict(self):
        return {
            'symbol':      self.symbol,
            'direction':   self.direction,
            'setup':       self.setup,
            'mode':        self.mode,
            'entry_time':  str(self.entry_time),
            'entry_price': self.entry_price,
            'stop':        self.stop,
            'target':      self.target,
            'rr_planned':  self.rr_planned,
            'exit_time':   str(self.exit_time),
            'exit_price':  self.exit_price,
            'exit_reason': self.exit_reason,
            'pnl_r':       self.pnl_r,
            'winner':      self.winner,
            'session':     self.session,
            'quality':     self.quality,
            'bars_held':   self.bars_held,
        }


# ─────────────────────────────────────────────────────────────
#  BACKTESTER
# ─────────────────────────────────────────────────────────────

def run_backtest(symbol: str, mode: str = 'swing',
                 setup: str = 'B',
                 start_date: str = '2024-06-01',
                 end_date:   str = '2026-03-13') -> list[Trade]:
    """
    Bar by bar backtest of Setup B (CHoCH + Breaker).
    Entry on next 5min bar open after all 6 gates pass.
    """
    print(f"\n  Loading data for {symbol}...")

    df_5m  = load_all_bars(symbol, '5min')
    df_15m = load_all_bars(symbol, '15min')
    df_1h  = load_all_bars(symbol, '1hour')
    df_4h  = load_all_bars(symbol, '4hour')

    # Filter date range
    start = pd.Timestamp(start_date, tz='UTC')
    end   = pd.Timestamp(end_date,   tz='UTC')
    df_5m = df_5m[(df_5m.index >= start) & (df_5m.index <= end)]

    print(f"  5min bars in range: {len(df_5m)} "
          f"({df_5m.index[0].date()} to {df_5m.index[-1].date()})")

    trades      = []
    active_trade: Optional[Trade] = None
    last_trade_bar: int = -10  # cooldown tracker
    used_pois: set = set()        # track POIs already traded
    rr          = RR_MAP[mode]
    max_bars    = 100 if mode == 'swing' else 30
    sess_end_h  = SESSION_END_UTC[symbol]

    # Warmup — need enough bars for indicators
    WARMUP_5M  = 200
    WARMUP_1H  = 100
    WARMUP_4H  = 50

    skipped_warmup = 0

    for i in range(WARMUP_5M, len(df_5m)):
        bar       = df_5m.iloc[i]
        bar_time  = df_5m.index[i]
        bar_open  = float(bar['open'])
        bar_high  = float(bar['high'])
        bar_low   = float(bar['low'])
        bar_close = float(bar['close'])

        # ── Manage active trade ──────────────────────────────
        if active_trade is not None:
            bars_held = i - active_trade._entry_bar
            t = active_trade

            # Check stop
            if t.direction == 'long'  and bar_low  <= t.stop:
                t.close(bar_time, t.stop, 'stop', bars_held)
                trades.append(t)
                active_trade = None
                last_trade_bar = i
                continue

            if t.direction == 'short' and bar_high >= t.stop:
                t.close(bar_time, t.stop, 'stop', bars_held)
                trades.append(t)
                active_trade = None
                last_trade_bar = i
                continue

            # Check target
            if t.direction == 'long'  and bar_high >= t.target:
                t.close(bar_time, t.target, 'target', bars_held)
                trades.append(t)
                active_trade = None
                last_trade_bar = i
                continue

            if t.direction == 'short' and bar_low  <= t.target:
                t.close(bar_time, t.target, 'target', bars_held)
                trades.append(t)
                active_trade = None
                last_trade_bar = i
                continue

            # Session end exit
            if bar_time.hour >= sess_end_h:
                t.close(bar_time, bar_close, 'session_end', bars_held)
                trades.append(t)
                active_trade = None
                last_trade_bar = i
                continue

            # Max bars exit
            if bars_held >= max_bars:
                t.close(bar_time, bar_close, 'max_bars', bars_held)
                trades.append(t)
                active_trade = None
                last_trade_bar = i
                continue

            continue  # still in trade, skip signal check

        # ── Signal check ─────────────────────────────────────
        # Day filter — match live_scanner.py so backtest uses identical trading days
        allowed_days = DAY_FILTERS.get(symbol, [0, 1, 2, 3, 4])
        if bar_time.weekday() not in allowed_days:
            continue

        # Reset used POIs at start of each new day
        if i > 0:
            prev_date = df_5m.index[i-1].date()
            curr_date = bar_time.date()
            if curr_date != prev_date:
                used_pois.clear()

        # Gate 4 first — cheapest check
        in_session, quality = check_session(symbol, bar_time)
        if not in_session:
            continue

        # Get HTF slices up to this bar (no lookahead)
        df_4h_slice = df_4h[df_4h.index <= bar_time].tail(200)
        df_1h_slice = df_1h[df_1h.index <= bar_time].tail(200)
        df_15m_slice= df_15m[df_15m.index <= bar_time].tail(100)
        df_5m_slice = df_5m.iloc[max(0, i-50):i+1]

        if len(df_4h_slice) < WARMUP_4H or len(df_1h_slice) < WARMUP_1H:
            skipped_warmup += 1
            continue

        # Get 1hour ATR
        atr_1h = float(calc_atr(df_1h_slice).iloc[-1])
        if np.isnan(atr_1h) or atr_1h == 0:
            continue

        price = bar_close

        # Try both directions
        for direction in ('long', 'short'):

            # Gate 1 — HTF bias
            if not check_htf_bias(df_4h_slice, direction):
                continue

            # Gate 2 — 1hour structure
            struct_ok, events_1h = check_structure(df_1h_slice, direction)
            if not struct_ok:
                continue

            # Gate 3 — Setup A, B or C
            if setup == 'A':
                poi_ok, poi = check_poi_setup_a(df_1h_slice, direction, price, atr_1h)
            elif setup == 'C':
                # Block 07:00–09:59 UTC — data-validated low-edge window
                if 7 <= bar_time.hour <= 9:
                    continue
                # Setup C requires BOS not CHoCH
                last_event = events_1h[-1][1] if events_1h else ''
                if 'CHOCH' in last_event:
                    continue
                poi_ok, poi = check_poi_setup_c(df_1h_slice, direction, price, atr_1h)
            else:
                poi_ok, poi = check_poi_setup_b(df_1h_slice, direction, price, atr_1h)
            if not poi_ok or poi is None:
                continue

            # Gate 5 — 15min entry trigger at POI
            if not check_entry_trigger_15m(df_15m_slice, direction, poi, atr_1h):
                continue

            # Gate 6 — 5min displacement (signal bar)
            if not check_confirmation_5m(df_5m_slice, direction):
                continue

            # ── All gates passed — prepare trade ─────────────
            # Entry is NEXT bar open
            if i + 1 >= len(df_5m):
                continue

            entry_bar   = df_5m.iloc[i + 1]
            entry_price = float(entry_bar['open'])
            entry_time  = df_5m.index[i + 1]

            if direction == 'long':
                stop   = round(poi['low'] - atr_1h * 0.5, 2)
                risk   = entry_price - stop
                target = round(entry_price + risk * rr, 2)
            else:
                stop   = round(poi['high'] + atr_1h * 0.5, 2)
                risk   = stop - entry_price
                target = round(entry_price - risk * rr, 2)

            if risk <= 0:
                continue

            # Mark this POI as used
            poi_key = (round(poi['high'], 0), round(poi['low'], 0), direction)
            if poi_key in used_pois:
                continue
            used_pois.add(poi_key)

            trade = Trade(
                symbol      = symbol,
                direction   = direction,
                setup       = 'A_sweep_ob' if setup == 'A' else ('C_bos_ob' if setup == 'C' else 'B_choch_breaker'),
                mode        = mode,
                entry_time  = entry_time,
                entry_price = entry_price,
                stop        = stop,
                target      = target,
                rr_planned  = rr,
                session     = quality,
                quality     = quality,
            )
            trade._entry_bar = i + 1
            active_trade = trade
            break  # one direction per bar

    # Close any open trade at end of data
    if active_trade is not None:
        last_bar   = df_5m.iloc[-1]
        bars_held  = len(df_5m) - active_trade._entry_bar
        active_trade.close(df_5m.index[-1], float(last_bar['close']), 'end_of_data', bars_held)
        trades.append(active_trade)

    if skipped_warmup > 0:
        print(f"  Skipped {skipped_warmup} bars during warmup period")

    return trades


# ─────────────────────────────────────────────────────────────
#  RESULTS ANALYSIS
# ─────────────────────────────────────────────────────────────

def analyse_results(trades: list[Trade], symbol: str, mode: str):
    if not trades:
        print(f"\n  No trades found for {symbol} {mode}")
        return {}

    closed = [t for t in trades if t.pnl_r is not None]
    if not closed:
        print(f"\n  No closed trades for {symbol} {mode}")
        return {}

    pnl_r     = [t.pnl_r for t in closed]
    winners   = [t for t in closed if t.winner]
    losers    = [t for t in closed if not t.winner]

    total_r   = round(sum(pnl_r), 2)
    win_rate  = round(len(winners) / len(closed) * 100, 1)
    avg_win   = round(np.mean([t.pnl_r for t in winners]), 2) if winners else 0
    avg_loss  = round(np.mean([t.pnl_r for t in losers]),  2) if losers  else 0
    expectancy= round(np.mean(pnl_r), 3)

    # Max drawdown
    equity    = np.cumsum(pnl_r)
    peak      = np.maximum.accumulate(equity)
    drawdown  = equity - peak
    max_dd    = round(float(np.min(drawdown)), 2)

    # Sharpe (annualised, assuming ~252 trading days, ~6 trades per day potential)
    if len(pnl_r) > 1 and np.std(pnl_r) > 0:
        sharpe = round(np.mean(pnl_r) / np.std(pnl_r) * np.sqrt(252), 2)
    else:
        sharpe = 0.0

    # Breakdown by session quality
    primary   = [t for t in closed if t.quality == 'primary']
    secondary = [t for t in closed if t.quality == 'secondary']

    print(f"\n{'='*60}")
    print(f"  {symbol} — Setup B — {mode.upper()}")
    print(f"{'='*60}")
    print(f"  Trades     : {len(closed)}")
    print(f"  Win Rate   : {win_rate}%")
    print(f"  Total R    : {total_r:+.2f}R")
    print(f"  Expectancy : {expectancy:+.3f}R per trade")
    print(f"  Avg Win    : {avg_win:+.2f}R")
    print(f"  Avg Loss   : {avg_loss:+.2f}R")
    print(f"  Max DD     : {max_dd:.2f}R")
    print(f"  Sharpe     : {sharpe}")
    print(f"\n  By session:")
    if primary:
        p_wr = round(sum(1 for t in primary if t.winner) / len(primary) * 100, 1)
        p_r  = round(sum(t.pnl_r for t in primary), 2)
        print(f"    Primary   : {len(primary)} trades | WR={p_wr}% | {p_r:+.2f}R")
    if secondary:
        s_wr = round(sum(1 for t in secondary if t.winner) / len(secondary) * 100, 1)
        s_r  = round(sum(t.pnl_r for t in secondary), 2)
        print(f"    Secondary : {len(secondary)} trades | WR={s_wr}% | {s_r:+.2f}R")

    print(f"\n  Exit reasons:")
    from collections import Counter
    reasons = Counter(t.exit_reason for t in closed)
    for reason, count in reasons.most_common():
        print(f"    {reason:15s} : {count}")

    print(f"\n  Last 10 trades:")
    for t in closed[-10:]:
        ts  = t.entry_time.astimezone(NY_TZ).strftime('%Y-%m-%d %H:%M')
        tag = '✅' if t.winner else '❌'
        print(f"    {tag} {t.direction.upper():5s} {ts} | "
              f"entry={t.entry_price:.2f} stop={t.stop:.2f} "
              f"target={t.target:.2f} | {t.pnl_r:+.2f}R [{t.exit_reason}]")

    return {
        'symbol':     symbol,
        'mode':       mode,
        'trades':     len(closed),
        'win_rate':   win_rate,
        'total_r':    total_r,
        'expectancy': expectancy,
        'max_dd':     max_dd,
        'sharpe':     sharpe,
        'trade_log':  [t.to_dict() for t in closed],
    }


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='NQ',
                        help='NQ | ES | GC | all')
    parser.add_argument('--mode',   default='swing',
                        help='swing | scalp')
    parser.add_argument('--setup',  default='B',
                        help='A | B')
    parser.add_argument('--start',  default='2024-06-01')
    parser.add_argument('--end',    default='2026-03-13')
    args = parser.parse_args()

    symbols = ['NQ', 'ES', 'GC'] if args.symbol == 'all' else [args.symbol]
    all_results = {}

    for sym in symbols:
        print(f"\n{'='*60}")
        print(f"  Backtesting {sym} Setup B — {args.mode.upper()}")
        print(f"  {args.start} to {args.end}")
        print(f"{'='*60}")

        trades  = run_backtest(sym, args.mode, args.setup, args.start, args.end)
        results = analyse_results(trades, sym, args.mode)
        all_results[sym] = results

        # Save trade log
        if results.get('trade_log'):
            fname = f'backtest_{args.setup}_{sym}_{args.mode}.json'
            with open(fname, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n  Saved to {fname}")

    if len(symbols) > 1:
        print(f"\n{'='*60}")
        print(f"  SUMMARY — Setup B {args.mode.upper()}")
        print(f"{'='*60}")
        for sym, r in all_results.items():
            if r:
                print(f"  {sym}: {r.get('trades',0)} trades | "
                      f"WR={r.get('win_rate',0)}% | "
                      f"{r.get('total_r',0):+.2f}R | "
                      f"Sharpe={r.get('sharpe',0)}")
