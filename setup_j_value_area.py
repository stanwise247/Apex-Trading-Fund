"""
APEX Setup J — Value Area Continuation
=======================================
Trade the retest of the previous NY session's Value Area High (VAH)
or Value Area Low (VAL) with a confirming close.

INSTRUMENT CONFIG:
  ES  = LIVE execution on MESM6 (Micro E-mini S&P, $5/pt)
  MNQ = PAPER ONLY (initial live observation period)

BACKTEST RESULTS (365 days, May 2025 – May 2026):
  ES:  Sharpe 8.34 | WR 54.7% | PF 3.02 | MaxDD 4R | +97R total | 11/11 months profitable
  MNQ: Sharpe 6.53 | WR 49.1% | PF 2.41 | MaxDD 11R | +82R total | 10/11 months profitable

PARAMETERS:
  Session:  13:00–19:00 UTC weekdays, NO Monday signals
  VA calc:  70/30 volume percentile on previous NY session 5min bars
  Entry:    bullish close above VAH (long) or bearish close below VAL (short)
  Stop:     0.5 × ATR14 beyond VAH/VAL
  Target:   2.5R
  HTF gate: 4h EMA20 — bullish for longs, bearish for shorts
  Dedup:    max 1 signal per symbol per session day
"""

import logging
import math
import pickle
from datetime import datetime, timezone, date
from typing import Optional

import db as _db

logger = logging.getLogger('APEX.SetupJ')

# ─────────────────────────────────────────────────────────────
#  PARAMETERS
# ─────────────────────────────────────────────────────────────

SESSION_START = 13      # UTC inclusive
SESSION_END   = 19      # UTC exclusive
STOP_ATR_MULT = 0.5     # stop = 0.5 × ATR14 beyond VAH/VAL
TARGET_RR     = 2.5     # target = entry + 2.5 × risk
VA_PCT        = 0.70    # value area covers 70% of previous session volume
BIAS_EMA      = 20      # 4h EMA period for HTF bias
BIAS_THRESH   = 0.001   # ±0.1% neutral zone

# ─────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────

def _load_bars(symbol: str, timeframe: str, limit: int = 2000):
    import pandas as pd
    conn = _db.connect()
    try:
        rows = conn.execute(
            'SELECT ts, open, high, low, close, volume FROM ohlcv '
            'WHERE symbol=? AND timeframe=? ORDER BY ts DESC LIMIT ?',
            (symbol, timeframe, limit)
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return pd.DataFrame(columns=['ts','open','high','low','close','volume'])
    df = pd.DataFrame(rows, columns=['ts','open','high','low','close','volume'])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.iloc[::-1].reset_index(drop=True)
    df['dt']      = pd.to_datetime(df['ts'], unit='s', utc=True)
    df['hour']    = df['dt'].dt.hour
    df['weekday'] = df['dt'].dt.weekday
    df['date']    = df['dt'].dt.date
    return df


def _atr14_j(df, n: int = 14):
    import pandas as pd
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift(1)).abs()
    lc = (df['low']  - df['close'].shift(1)).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(n).mean()


# ─────────────────────────────────────────────────────────────
#  VALUE AREA CALCULATION
# ─────────────────────────────────────────────────────────────

def _calc_value_area(session_bars, va_pct: float = VA_PCT):
    """
    Compute VAH and VAL from session bars using volume-weighted percentile.
    Returns (vah, val) or (None, None) on failure.
    """
    import numpy as np
    if len(session_bars) < 10:
        return None, None
    prices = session_bars['close'].values
    vols   = session_bars['volume'].values
    total  = vols.sum()
    if total <= 0:
        return None, None
    # Sort by price, compute cumulative volume percentile
    order = prices.argsort()
    sorted_prices = prices[order]
    sorted_vols   = vols[order]
    cum_vol       = sorted_vols.cumsum()
    p30_idx = int(np.searchsorted(cum_vol, total * 0.30))
    p70_idx = int(np.searchsorted(cum_vol, total * 0.70))
    p30_idx = min(p30_idx, len(sorted_prices) - 1)
    p70_idx = min(p70_idx, len(sorted_prices) - 1)
    val = float(sorted_prices[p30_idx])
    vah = float(sorted_prices[p70_idx])
    if vah <= val:
        return None, None
    return vah, val


# ─────────────────────────────────────────────────────────────
#  HTF BIAS
# ─────────────────────────────────────────────────────────────

def _get_htf_bias(symbol: str) -> str:
    """4h EMA20 bias — identical logic to fvg_engine.get_htf_bias()."""
    try:
        df_5m = _load_bars(symbol, '5min', limit=5000)
        if df_5m.empty or len(df_5m) < 20:
            return 'neutral'
        df_4h = df_5m.set_index('dt')[['open','high','low','close','volume']].resample('4h').agg(
            {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}
        ).dropna()
        if len(df_4h) < BIAS_EMA + 1:
            return 'neutral'
        ema = df_4h['close'].ewm(span=BIAS_EMA, adjust=False).mean()
        last_close = float(df_4h['close'].iloc[-1])
        last_ema   = float(ema.iloc[-1])
        if last_close > last_ema * (1 + BIAS_THRESH):
            return 'bullish'
        if last_close < last_ema * (1 - BIAS_THRESH):
            return 'bearish'
        return 'neutral'
    except Exception as e:
        logger.debug(f'Setup J HTF bias error {symbol}: {e}')
        return 'neutral'


# ─────────────────────────────────────────────────────────────
#  DEDUP
# ─────────────────────────────────────────────────────────────

_j_dedup_sent: dict = {}   # key → date string; max 1 signal per symbol per session day


# ─────────────────────────────────────────────────────────────
#  MAIN SCAN FUNCTION
# ─────────────────────────────────────────────────────────────

def scan_setup_j(symbol: str, dt: datetime = None) -> Optional[dict]:
    """
    Check for a Value Area Continuation signal on symbol.
    Returns signal dict or None.

    Entry conditions:
    - Weekday Tuesday–Friday only (no Monday signals)
    - NY session 13:00–19:00 UTC
    - Previous session VAH/VAL computed from previous day 13–19 UTC 5min bars
    - Long:  current 5min bar low touched VAH, close > VAH, bullish close, bias bullish/neutral
    - Short: current 5min bar high touched VAL, close < VAL, bearish close, bias bearish/neutral
    - Stop: 0.5 × ATR14 beyond VAH/VAL
    - Target: 2.5R
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    # ── Gate: weekday (Tue-Fri only, no Monday) ─────────────
    if dt.weekday() == 0:   # Monday
        return None
    if dt.weekday() >= 5:   # Weekend
        return None

    # ── Gate: session ────────────────────────────────────────
    if not (SESSION_START <= dt.hour < SESSION_END):
        return None

    # ── Gate: per-session dedup ──────────────────────────────
    today_str = dt.strftime('%Y-%m-%d')
    dedup_key = f'{symbol}_J_{today_str}'
    if dedup_key in _j_dedup_sent:
        return None

    # ── Load 5min bars — need ~200 bars for prev session + ATR ──
    df5 = _load_bars(symbol, '5min', limit=200)
    if df5.empty or len(df5) < 50:
        logger.debug(f'Setup J {symbol}: insufficient 5min bars')
        return None

    # ── Compute ATR14 ─────────────────────────────────────────
    atr_series = _atr14_j(df5)
    atr_val    = float(atr_series.iloc[-1] if not atr_series.iloc[-1] != atr_series.iloc[-1] else 0)
    if atr_val <= 0 or math.isnan(atr_val):
        return None

    # ── Previous session bars (yesterday 13–19 UTC) ───────────
    today_date = dt.date()
    prev_bars  = df5[
        (df5['weekday'] < 5) &
        (df5['hour'] >= 13) & (df5['hour'] < 19) &
        (df5['date'] < today_date)
    ].tail(78)  # up to 78 bars = full 6h session at 5min granularity

    if len(prev_bars) < 20:
        logger.debug(f'Setup J {symbol}: insufficient previous session bars ({len(prev_bars)})')
        return None

    vah, val = _calc_value_area(prev_bars, VA_PCT)
    if vah is None:
        logger.debug(f'Setup J {symbol}: could not compute value area')
        return None

    # ── HTF bias ────────────────────────────────────────────
    bias = _get_htf_bias(symbol)

    # ── Current bar OHLC ─────────────────────────────────────
    bar       = df5.iloc[-1]
    bar_close = float(bar['close'])
    bar_open  = float(bar['open'])
    bar_low   = float(bar['low'])
    bar_high  = float(bar['high'])

    direction = None
    entry = stop = target = None

    # Long: bar touched VAH from above, closed back above VAH (bullish)
    if (bar_low <= vah * 1.001
            and bar_close > vah
            and bar_close > bar_open
            and bias != 'bearish'):          # allow bullish or neutral
        direction = 'long'
        entry     = bar_close
        stop      = vah - STOP_ATR_MULT * atr_val
        risk      = entry - stop
        target    = entry + TARGET_RR * risk

    # Short: bar touched VAL from below, closed back below VAL (bearish)
    elif (bar_high >= val * 0.999
              and bar_close < val
              and bar_close < bar_open
              and bias != 'bullish'):        # allow bearish or neutral
        direction = 'short'
        entry     = bar_close
        stop      = val + STOP_ATR_MULT * atr_val
        risk      = stop - entry
        target    = entry - TARGET_RR * risk

    if direction is None:
        return None

    if abs(entry - stop) <= 0:
        return None

    rr = round(abs(target - entry) / abs(entry - stop), 2)

    # Mark dedup before returning
    _j_dedup_sent[dedup_key] = today_str

    logger.info(
        f'[J-1/6] Signal generated: {symbol} {direction} '
        f'entry={entry:.2f} stop={stop:.2f} target={target:.2f} '
        f'VAH={vah:.2f} VAL={val:.2f} ATR={atr_val:.2f} bias={bias}'
    )
    logger.info(f'[J-2/6] Dedup check: key={dedup_key} (passed)')

    return {
        'symbol':    symbol,
        'direction': direction,
        'setup':     'J_value_area_cont',
        'mode':      'intraday',
        'entry':     round(float(entry), 2),
        'stop':      round(float(stop), 2),
        'target':    round(float(target), 2),
        'rr':        rr,
        'session':   'NY Primary',
        'quality':   'primary',
        'vah':       round(float(vah), 2),
        'val':       round(float(val), 2),
        'atr':       round(float(atr_val), 2),
        'htf_bias':  bias,
        'timestamp': dt.isoformat(),
    }


# ─────────────────────────────────────────────────────────────
#  ALERT FORMATTER
# ─────────────────────────────────────────────────────────────

def format_j_alert(signal: dict) -> str:
    from zoneinfo import ZoneInfo
    NY    = ZoneInfo('America/New_York')
    now   = datetime.now(timezone.utc).astimezone(NY).strftime('%Y-%m-%d %H:%M')
    sym   = signal['symbol']
    dir_  = signal['direction'].upper()
    sep   = chr(9473) * 20
    emoji = '📈' if dir_ == 'LONG' else '📉'
    entry = signal['entry']
    stop  = signal['stop']
    tgt   = signal['target']
    vah   = signal.get('vah', '—')
    val   = signal.get('val', '—')
    bias  = signal.get('htf_bias', '—')
    rr    = signal.get('rr', 2.5)
    stop_pts = abs(entry - stop)

    parts = [
        f'{emoji} <b>APEX SETUP J — {sym}</b>',
        sep,
        f'<b>Strategy:</b>   J — Value Area Continuation',
        f'<b>Direction:</b>  {dir_}',
        f'<b>HTF Bias:</b>   {bias.upper()}',
        sep,
        f'<b>VAH:</b>        {vah:.2f}',
        f'<b>VAL:</b>        {val:.2f}',
        sep,
        f'<b>Entry:</b>      {entry:.2f}',
        f'<b>Stop:</b>       {stop:.2f}  ({stop_pts:.1f} pts)',
        f'<b>Target:</b>     {tgt:.2f}',
        f'<b>R:R:</b>        {rr}×',
        sep,
        f'<i>{now} ET | Setup J {sym} {dir_} | VA Cont | entry={entry:.2f} stop={stop:.2f} tgt={tgt:.2f}</i>',
    ]
    return '\n'.join(parts)
