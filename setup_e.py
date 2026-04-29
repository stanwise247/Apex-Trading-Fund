"""
APEX Setup E — EMA50 Pullback
==============================
5min bar retraces to EMA50 in a 4-hour trend, then closes back in
the trend direction — a momentum resumption entry.

Walk-forward validation (NQ, Sep 2024 – Mar 2026):
  IS  Sep24–Jun25:  n=201  WR=59.7%  EV=+0.528R  Sharpe=6.32  ✅
  OOS Jul25–Mar26:  n=205  WR=52.7%  EV=+0.425R  Sharpe=5.17  ✅
  All 9 OOS months profitable.

Entry rules (all must be true):
  1. NY session — 13:00–18:59 UTC, weekday only
  2. 4h EMA20 bias — bullish (close > EMA20 × 1.001) or bearish
  3. Current 5min bar close within 0.30 × ATR of EMA50
  4. Current bar closes in bias direction (bull bar for long, bear for short)
  5. Prior bar was contra-direction (confirms actual pullback)

Exit:
  Stop:   ±1.5 × ATR from entry
  Target: ±3.75 × ATR from entry  (2.5:1 RR)
  Max hold: end of session (19:00 UTC)

Min gap: 12 bars (60 min) between trades.
"""

import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo
import db as _db

logger = logging.getLogger('APEX.SetupE')

NY_TZ = ZoneInfo('America/New_York')

# ─────────────────────────────────────────────────────────────
#  PARAMETERS
# ─────────────────────────────────────────────────────────────

EMA_PERIOD    = 50
EMA_PROXIMITY = 0.30   # within 0.30 ATR of EMA50
ATR_PERIOD    = 14
STOP_ATR      = 1.50
TARGET_ATR    = 3.75  # 1.5 stop × 2.5 RR = 3.75 ATR target
SESSION_START = 13     # UTC inclusive
SESSION_END   = 19     # UTC exclusive (last entry at 18:xx)
BIAS_EMA      = 20     # 4hour EMA period
BIAS_THRESH   = 0.001  # ±0.1% neutral zone
MIN_GAP_BARS  = 12     # 60 min minimum between trades


# ─────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class EGateResult:
    passed:  bool
    gate:    int
    name:    str
    detail:  str

    def __repr__(self):
        icon = '✅' if self.passed else '❌'
        return f"{icon} Gate {self.gate} [{self.name}]: {self.detail}"


@dataclass
class ESetupResult:
    symbol:    str
    direction: str       # 'long' | 'short'
    setup:     str       # always 'E_ema50_pullback'
    valid:     bool
    gates:     list
    entry:     Optional[float]
    stop:      Optional[float]
    target:    Optional[float]
    rr:        Optional[float]
    atr:       Optional[float]
    ema50:     Optional[float]
    timestamp: Optional[pd.Timestamp]

    def __repr__(self):
        if not self.valid:
            failed = [g for g in self.gates if not g.passed]
            first  = failed[0] if failed else None
            return (f"❌ NQ {self.direction.upper()} [E] blocked: {first.name}" if first
                    else f"❌ NQ {self.direction.upper()} [E] no setup")
        return (f"✅ NQ {self.direction.upper()} [Setup E] "
                f"entry={self.entry:.2f} stop={self.stop:.2f} "
                f"target={self.target:.2f} RR={self.rr:.1f}x "
                f"EMA50={self.ema50:.2f} ATR={self.atr:.2f}")


# ─────────────────────────────────────────────────────────────
#  INTERNAL DATA LOADER
# ─────────────────────────────────────────────────────────────

def _load_bars(symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
    conn = _db.connect()
    df = _db.read_sql(
        'SELECT ts,open,high,low,close,volume FROM ohlcv '
        'WHERE symbol=? AND timeframe=? ORDER BY ts DESC LIMIT ?',
        conn, params=(symbol, timeframe, limit)
    )
    conn.close()
    df = df.iloc[::-1].reset_index(drop=True)
    df['dt'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    df.set_index('dt', inplace=True)
    return df[['open','high','low','close','volume']].astype(float)


# ─────────────────────────────────────────────────────────────
#  GATE FUNCTIONS
# ─────────────────────────────────────────────────────────────

def gate1_session(dt: datetime) -> EGateResult:
    if dt.weekday() >= 5:
        return EGateResult(False, 1, 'Session', 'Weekend — no trading')
    if not (SESSION_START <= dt.hour < SESSION_END):
        return EGateResult(False, 1, 'Session',
                           f'Hour {dt.hour:02d}:00 UTC outside 13-18 UTC window')
    return EGateResult(True, 1, 'Session',
                       f'NY session {dt.hour:02d}:{dt.minute:02d} UTC')


def gate2_htf_bias(symbol: str, direction: str) -> tuple:
    """Returns (EGateResult, bias_str). Resamples 5min bars in-memory — no broken HTF DB rows."""
    df_5m = _load_bars(symbol, '5min', limit=5000)  # 5000 bars = ~104 4h bars; needed for EMA20 convergence
    if df_5m.empty:
        return (EGateResult(False, 2, 'HTF Bias', 'No 5min data'), 'neutral')
    df_4h = df_5m['close'].resample('4h').last().dropna().to_frame('close')
    if len(df_4h) < BIAS_EMA:
        return (EGateResult(False, 2, 'HTF Bias',
                            f'Insufficient 4h bars for EMA{BIAS_EMA} ({len(df_4h)})'), 'neutral')
    ema20       = df_4h['close'].ewm(span=BIAS_EMA, adjust=False).mean()
    last_close  = float(df_4h['close'].iloc[-1])
    last_ema    = float(ema20.iloc[-1])

    if last_close > last_ema * (1 + BIAS_THRESH):
        bias = 'bullish'
    elif last_close < last_ema * (1 - BIAS_THRESH):
        bias = 'bearish'
    else:
        bias = 'neutral'

    if bias == 'neutral':
        return (EGateResult(False, 2, 'HTF Bias', '4h EMA20 neutral — no edge'), bias)
    if direction == 'long' and bias != 'bullish':
        return (EGateResult(False, 2, 'HTF Bias',
                            f'4h bias={bias.upper()} conflicts with LONG'), bias)
    if direction == 'short' and bias != 'bearish':
        return (EGateResult(False, 2, 'HTF Bias',
                            f'4h bias={bias.upper()} conflicts with SHORT'), bias)

    return (EGateResult(True, 2, 'HTF Bias',
                        f'4h EMA20 {bias.upper()} | close={last_close:.2f} ema={last_ema:.2f}'), bias)


def gate3_ema50_proximity(symbol: str) -> tuple:
    """Returns (EGateResult, atr_val, ema50_val, price)"""
    df = _load_bars(symbol, '5min', limit=100)
    df['ema50'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    high = df['high']; low = df['low']; prev = df['close'].shift(1)
    tr = pd.concat([high-low, (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/ATR_PERIOD, adjust=False).mean()

    cur     = df.iloc[-1]
    atr_val = float(cur['atr'])
    ema50   = float(cur['ema50'])
    price   = float(cur['close'])
    dist    = abs(price - ema50)

    if pd.isna(atr_val) or atr_val == 0:
        return (EGateResult(False, 3, 'EMA50 Proximity', 'ATR=0'), 0, 0, price)

    if dist > EMA_PROXIMITY * atr_val:
        return (EGateResult(False, 3, 'EMA50 Proximity',
                            f'Price {price:.2f} is {dist/atr_val:.2f} ATR from EMA50 {ema50:.2f} '
                            f'(max {EMA_PROXIMITY})'),
                atr_val, ema50, price)

    return (EGateResult(True, 3, 'EMA50 Proximity',
                        f'Price {price:.2f} within {dist/atr_val:.2f} ATR of EMA50 {ema50:.2f}'),
            atr_val, ema50, price)


def gate4_bar_direction(symbol: str, direction: str) -> EGateResult:
    df    = _load_bars(symbol, '5min', limit=5)
    cur   = df.iloc[-1]
    close = float(cur['close'])
    open_ = float(cur['open'])

    bull_bar = close > open_
    bear_bar = close < open_

    if direction == 'long' and not bull_bar:
        return EGateResult(False, 4, 'Bar Direction',
                           f'5min bar bearish (o={open_:.2f} c={close:.2f}) — need bullish for long')
    if direction == 'short' and not bear_bar:
        return EGateResult(False, 4, 'Bar Direction',
                           f'5min bar bullish (o={open_:.2f} c={close:.2f}) — need bearish for short')
    return EGateResult(True, 4, 'Bar Direction',
                       f'5min close={close:.2f} open={open_:.2f} '
                       f'→ {"bullish" if bull_bar else "bearish"} confirms {direction.upper()}')


def gate5_pullback_bar(symbol: str, direction: str) -> EGateResult:
    df       = _load_bars(symbol, '5min', limit=5)
    prev_bar = df.iloc[-2]
    p_close  = float(prev_bar['close'])
    p_open   = float(prev_bar['open'])

    prev_bull = p_close > p_open
    prev_bear = p_close < p_open

    if direction == 'long' and not prev_bear:
        return EGateResult(False, 5, 'Pullback Bar',
                           f'Prior bar not bearish — no pullback into EMA50 confirmed')
    if direction == 'short' and not prev_bull:
        return EGateResult(False, 5, 'Pullback Bar',
                           f'Prior bar not bullish — no pullback into EMA50 confirmed')
    return EGateResult(True, 5, 'Pullback Bar',
                       f'Prior bar contra-direction (o={p_open:.2f} c={p_close:.2f}) — pullback confirmed')


# ─────────────────────────────────────────────────────────────
#  MAIN CHECKER
# ─────────────────────────────────────────────────────────────

def check_setup_e(symbol: str = 'MNQ', direction: str = 'long',
                  dt: datetime = None) -> ESetupResult:
    """
    Run all 5 gates for Setup E.
    Returns ESetupResult — check .valid to see if trade is active.
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    gates = []

    # Gate 1 — Session
    g1 = gate1_session(dt)
    gates.append(g1)
    if not g1.passed:
        return ESetupResult(symbol, direction, 'E_ema50_pullback', False, gates,
                            None, None, None, None, None, None, None)

    # Gate 2 — HTF Bias
    g2, bias = gate2_htf_bias(symbol, direction)
    gates.append(g2)
    if not g2.passed:
        return ESetupResult(symbol, direction, 'E_ema50_pullback', False, gates,
                            None, None, None, None, None, None, None)

    # Gate 3 — EMA50 Proximity
    g3, atr_val, ema50, price = gate3_ema50_proximity(symbol)
    gates.append(g3)
    if not g3.passed:
        return ESetupResult(symbol, direction, 'E_ema50_pullback', False, gates,
                            None, None, None, None, atr_val, ema50, None)

    # Gate 4 — Bar direction
    g4 = gate4_bar_direction(symbol, direction)
    gates.append(g4)
    if not g4.passed:
        return ESetupResult(symbol, direction, 'E_ema50_pullback', False, gates,
                            None, None, None, None, atr_val, ema50, None)

    # Gate 5 — Pullback bar
    g5 = gate5_pullback_bar(symbol, direction)
    gates.append(g5)
    if not g5.passed:
        return ESetupResult(symbol, direction, 'E_ema50_pullback', False, gates,
                            None, None, None, None, atr_val, ema50, None)

    # All gates passed — compute levels
    entry  = round(price, 2)
    if direction == 'long':
        stop   = round(price - STOP_ATR   * atr_val, 2)
        target = round(price + TARGET_ATR * atr_val, 2)
    else:
        stop   = round(price + STOP_ATR   * atr_val, 2)
        target = round(price - TARGET_ATR * atr_val, 2)

    risk = abs(entry - stop)
    rr   = round(abs(target - entry) / risk, 1) if risk > 0 else 0.0

    # Timestamp from last 5min bar
    df_ts = _load_bars(symbol, '5min', limit=2)
    ts    = df_ts.index[-1]

    return ESetupResult(
        symbol    = symbol,
        direction = direction,
        setup     = 'E_ema50_pullback',
        valid     = True,
        gates     = gates,
        entry     = entry,
        stop      = stop,
        target    = target,
        rr        = rr,
        atr       = round(atr_val, 2),
        ema50     = round(ema50, 2),
        timestamp = ts,
    )


def scan_setup_e(dt: datetime = None) -> list:
    """
    Scan both directions for Setup E on MNQ.
    Returns list of valid ESetupResult objects.
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    results = []
    for direction in ('long', 'short'):
        try:
            r = check_setup_e('MNQ', direction, dt)
            if r.valid:
                results.append(r)
        except Exception as e:
            logger.debug(f'scan_setup_e error ({direction}): {e}')
    return results


def format_alert(result: ESetupResult) -> str:
    """Format a Telegram alert message for Setup E."""
    now_ny  = datetime.now(timezone.utc).astimezone(NY_TZ)
    dir_emo = '🟢' if result.direction == 'long' else '🔴'
    return (
        f"{dir_emo} <b>APEX SIGNAL — {result.symbol}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Direction:</b> {result.direction.upper()}\n"
        f"<b>Setup:</b>     Setup E — EMA50 Pullback\n"
        f"<b>Session:</b>   ⭐ PRIMARY (NY)\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Entry:</b>     {result.entry:.2f}\n"
        f"<b>Stop:</b>      {result.stop:.2f} "
        f"({abs(round(result.entry - result.stop, 2))} pts)\n"
        f"<b>Target:</b>    {result.target:.2f} "
        f"({abs(round(result.entry - result.target, 2))} pts)\n"
        f"<b>R:R:</b>       {result.rr:.1f}x\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>EMA50:</b>     {result.ema50:.2f}\n"
        f"<b>ATR:</b>       {result.atr:.2f}\n"
        f"<i>{now_ny.strftime('%Y-%m-%d %H:%M')} ET</i>\n"
        f"<i>Gates: all 5 passed ✅</i>"
    )


# ─────────────────────────────────────────────────────────────
#  CLI — verify gate state right now
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from datetime import timezone
    now = datetime.now(timezone.utc)
    print(f"\n{'='*60}")
    print(f"  APEX Setup E — Gate Check")
    print(f"  {now.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"{'='*60}\n")

    for direction in ('long', 'short'):
        print(f"  MNQ {direction.upper()}")
        print(f"  {'-'*40}")
        r = check_setup_e('MNQ', direction, now)
        for g in r.gates:
            print(f"    {g}")
        if r.valid:
            print(f"\n  >>> {r}")
        else:
            failed = [g for g in r.gates if not g.passed]
            if failed:
                print(f"  Blocked at: {failed[0].name} — {failed[0].detail}")
        print()

    print(f"{'='*60}")
    results = scan_setup_e(now)
    if results:
        print(f"  ACTIVE SIGNALS ({len(results)}):")
        for r in results:
            print(f"  {r}")
            print()
            print(format_alert(r))
    else:
        print(f"  No Setup E signals — gates not aligned")
    print(f"{'='*60}\n")
