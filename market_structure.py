"""
APEX Market Structure Engine — market_structure.py
====================================================
Session 1 rebuild — clean, verified, real data.
v2 — instrument-specific swing lookbacks.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
from zoneinfo import ZoneInfo
import db as _db

NY_TZ = ZoneInfo('America/New_York')

# Instrument-specific settings
# Lookback = bars each side required to confirm a swing
# Higher = fewer but more significant swings
INSTRUMENT_SETTINGS = {
    'NQ': {'lookback': 5, 'limit': 500, 'structure_tf': '1hour'},
    'ES': {'lookback': 5, 'limit': 500, 'structure_tf': '1hour'},
    'GC': {'lookback': 5, 'limit': 500, 'structure_tf': '1hour'},
    'CL': {'lookback':  7, 'limit': 2000},
}
DEFAULT_SETTINGS = {'lookback': 8, 'limit': 2000}


def load_bars(symbol: str, timeframe: str, limit: int = 2000) -> pd.DataFrame:
    conn = _db.connect()
    df = _db.read_sql(
        'SELECT ts, open, high, low, close, volume FROM ohlcv '
        'WHERE symbol=? AND timeframe=? ORDER BY ts DESC LIMIT ?',
        conn, params=(symbol, timeframe, limit)
    )
    conn.close()
    if df.empty:
        raise ValueError(f"No data found for {symbol} {timeframe}")
    df = df.iloc[::-1].reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    df.set_index('datetime', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df


@dataclass
class SwingPoint:
    index:     int
    timestamp: pd.Timestamp
    price:     float
    kind:      str
    broken:    bool = False

    def __repr__(self):
        tag = ' [BROKEN]' if self.broken else ''
        k   = 'SH' if self.kind == 'high' else 'SL'
        ts  = self.timestamp.astimezone(NY_TZ).strftime('%Y-%m-%d %H:%M')
        return f"{k} {self.price:.2f} @ {ts} ET{tag}"


@dataclass
class StructureEvent:
    index:      int
    timestamp:  pd.Timestamp
    event_type: str
    level:      float
    close:      float

    def __repr__(self):
        ts = self.timestamp.astimezone(NY_TZ).strftime('%Y-%m-%d %H:%M')
        return f"{self.event_type} | level {self.level:.2f} -> close {self.close:.2f} @ {ts} ET"


@dataclass
class OpeningRange:
    session:    str
    date:       str
    high:       float
    low:        float
    midpoint:   float
    range_size: float

    def __repr__(self):
        return (f"OR [{self.session}] {self.date} | "
                f"H={self.high:.2f}  L={self.low:.2f}  "
                f"mid={self.midpoint:.2f}  size={self.range_size:.2f}")


@dataclass
class PrevDayLevels:
    high:  float
    low:   float
    close: float
    date:  str

    def __repr__(self):
        return f"PDH={self.high:.2f}  PDL={self.low:.2f}  PDC={self.close:.2f}  ({self.date})"


def find_swings(df: pd.DataFrame, lookback: int = 8):
    highs, lows = [], []
    n = lookback
    for i in range(n, len(df) - n):
        h  = df['high'].iloc[i]
        l  = df['low'].iloc[i]
        ts = df.index[i]
        is_sh = all(h > df['high'].iloc[i-j] for j in range(1, n+1)) and \
                all(h > df['high'].iloc[i+j] for j in range(1, n+1))
        is_sl = all(l < df['low'].iloc[i-j] for j in range(1, n+1)) and \
                all(l < df['low'].iloc[i+j] for j in range(1, n+1))
        if is_sh:
            highs.append(SwingPoint(i, ts, round(float(h), 2), 'high'))
        if is_sl:
            lows.append(SwingPoint(i, ts, round(float(l), 2), 'low'))
    highs = _deduplicate(highs, 'high', lookback * 2)
    lows  = _deduplicate(lows,  'low',  lookback * 2)
    return highs, lows


def _deduplicate(swings, kind, cluster_bars):
    if not swings:
        return []
    result  = []
    cluster = [swings[0]]
    for sp in swings[1:]:
        if sp.index - cluster[-1].index <= cluster_bars:
            cluster.append(sp)
        else:
            best = max(cluster, key=lambda s: s.price) if kind == 'high' \
                   else min(cluster, key=lambda s: s.price)
            result.append(best)
            cluster = [sp]
    best = max(cluster, key=lambda s: s.price) if kind == 'high' \
           else min(cluster, key=lambda s: s.price)
    result.append(best)
    return result


def detect_structure(df: pd.DataFrame, swing_highs, swing_lows):
    closes = df['close'].values
    times  = df.index
    events = []
    all_sh = sorted(swing_highs, key=lambda s: s.index)
    all_sl = sorted(swing_lows,  key=lambda s: s.index)
    sh_ptr = sl_ptr = 0
    pending_sh = pending_sl = None
    bias = 'neutral'
    for i in range(len(closes)):
        close = closes[i]
        ts    = times[i]
        while sh_ptr < len(all_sh) and all_sh[sh_ptr].index < i:
            pending_sh = all_sh[sh_ptr]
            sh_ptr += 1
        while sl_ptr < len(all_sl) and all_sl[sl_ptr].index < i:
            pending_sl = all_sl[sl_ptr]
            sl_ptr += 1
        if pending_sh and not pending_sh.broken and close > pending_sh.price:
            etype = 'BOS_BULL' if bias in ('bullish', 'neutral') else 'CHOCH_BULL'
            events.append(StructureEvent(i, ts, etype, pending_sh.price, round(float(close), 2)))
            pending_sh.broken = True
            bias = 'bullish'
        if pending_sl and not pending_sl.broken and close < pending_sl.price:
            etype = 'BOS_BEAR' if bias in ('bearish', 'neutral') else 'CHOCH_BEAR'
            events.append(StructureEvent(i, ts, etype, pending_sl.price, round(float(close), 2)))
            pending_sl.broken = True
            bias = 'bearish'
    return events, bias


SESSIONS = {
    'London':  ('07:00', '09:30'),
    'NY Open': ('13:30', '14:30'),
}

def get_opening_ranges(df: pd.DataFrame):
    ranges = []
    dates  = df.index.normalize().unique()
    for date in dates:
        day_df = df[df.index.normalize() == date]
        for session, (start, end) in SESSIONS.items():
            mask = (
                (day_df.index.time >= pd.Timestamp(start).time()) &
                (day_df.index.time <= pd.Timestamp(end).time())
            )
            s_df = day_df[mask]
            if len(s_df) < 1:
                continue
            h   = round(float(s_df['high'].max()), 2)
            l   = round(float(s_df['low'].min()),  2)
            mid = round((h + l) / 2, 2)
            ranges.append(OpeningRange(session, str(date.date()), h, l, mid, round(h - l, 2)))
    return ranges


def get_prev_day(symbol: str) -> Optional[PrevDayLevels]:
    df = load_bars(symbol, '1day', limit=5)
    if len(df) < 2:
        return None
    prev = df.iloc[-2]
    return PrevDayLevels(
        high  = round(float(prev['high']),  2),
        low   = round(float(prev['low']),   2),
        close = round(float(prev['close']), 2),
        date  = str(df.index[-2].date()),
    )


def compute_bias(events):
    if not events:
        return 'neutral', 0.0
    window   = events[-10:]
    bull     = sum(1 for e in window if 'BULL' in e.event_type)
    bear     = sum(1 for e in window if 'BEAR' in e.event_type)
    last     = events[-1].event_type
    if bull > bear:
        bias     = 'bullish'
        strength = round(min(1.0, bull / len(window)), 2)
    elif bear > bull:
        bias     = 'bearish'
        strength = round(min(1.0, bear / len(window)), 2)
    else:
        bias, strength = 'neutral', 0.0
    if 'CHOCH' in last:
        strength = min(strength, 0.5)
    return bias, strength


def analyse(symbol: str, timeframe: str = '15min') -> dict:
    cfg                      = INSTRUMENT_SETTINGS.get(symbol, DEFAULT_SETTINGS)
    df                       = load_bars(symbol, timeframe, limit=cfg['limit'])
    swing_highs, swing_lows  = find_swings(df, lookback=5)
    events, _                = detect_structure(df, swing_highs, swing_lows)
    bias, strength           = compute_bias(events)
    prev_day                 = get_prev_day(symbol)
    opening_ranges           = get_opening_ranges(df) if timeframe in ('5min', '15min') else []
    return {
        'symbol':         symbol,
        'timeframe':      timeframe,
        'bias':           bias,
        'bias_strength':  strength,
        'swing_highs':    swing_highs,
        'swing_lows':     swing_lows,
        'last_sh':        swing_highs[-1] if swing_highs else None,
        'last_sl':        swing_lows[-1]  if swing_lows  else None,
        'events':         events,
        'last_event':     events[-1]      if events      else None,
        'prev_day':       prev_day,
        'opening_ranges': opening_ranges,
    }


if __name__ == '__main__':
    for symbol in ('NQ', 'ES', 'GC'):
        cfg = INSTRUMENT_SETTINGS.get(symbol, DEFAULT_SETTINGS)
        print(f"\n{'='*60}")
        print(f"  {symbol} — 1hour structure")
        print(f"{'='*60}")
        df = load_bars(symbol, '1hour', limit=500)
        print(f"\n  Bars loaded : {len(df)}")
        print(f"  From        : {df.index[0].astimezone(NY_TZ).strftime('%Y-%m-%d %H:%M')} ET")
        print(f"  To          : {df.index[-1].astimezone(NY_TZ).strftime('%Y-%m-%d %H:%M')} ET")
        print(f"  Last close  : {df['close'].iloc[-1]:.2f}")
        sh, sl = find_swings(df, lookback=5)
        print(f"\n  Swing Highs ({len(sh)}) — last 5:")
        for s in sh[-5:]: print(f"    {s}")
        print(f"\n  Swing Lows ({len(sl)}) — last 5:")
        for s in sl[-5:]: print(f"    {s}")
        events, _ = detect_structure(df, sh, sl)
        print(f"\n  Structure Events ({len(events)}) — last 5:")
        for e in events[-5:]: print(f"    {e}")
        bias, strength = compute_bias(events)
        print(f"\n  Bias : {bias.upper()}  |  strength = {strength}")
        pd_levels = get_prev_day(symbol)
        if pd_levels:
            print(f"\n  Prev Day : {pd_levels}")
        ranges = get_opening_ranges(df)
        print(f"\n  Opening Ranges — last 3:")
        for r in ranges[-3:]: print(f"    {r}")
    print(f"\n{'='*60}")
    print("  Done. Verify the output before Session 2.")
    print(f"{'='*60}\n")
