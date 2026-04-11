"""
APEX Setup H — VWAP 2-Standard Deviation Band Reversal
=======================================================
ES primary (live signals) | NQ secondary (paper tracking only)

Edge: OOS Sharpe 6.03 on ES | WR 76.5% | n=268 trades (5.5-month OOS)
Signal: price reverts from 2σ session VWAP band with 1h SMA20 HTF confirmation bar
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import db as _db

logger = logging.getLogger('APEX.SetupH')

NY_TZ             = ZoneInfo('America/New_York')
SESSION_START_UTC = 13   # 13:00 UTC = NY open
SESSION_END_UTC   = 19   # 19:00 UTC = session end


# ─────────────────────────────────────────────────────────────
#  DATA LOADER
# ─────────────────────────────────────────────────────────────

def _load_bars(symbol: str, limit: int = 500) -> pd.DataFrame:
    """Load 5min bars from DB. Returns DataFrame with ts, dt, OHLCV columns."""
    conn = _db.connect()
    cur  = conn.cursor()
    cur.execute(
        'SELECT ts, open, high, low, close, volume FROM ohlcv '
        'WHERE symbol=? AND timeframe=? ORDER BY ts DESC LIMIT ?',
        (symbol, '5min', limit)
    )
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df = df.iloc[::-1].reset_index(drop=True)
    df['dt'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    df[['open', 'high', 'low', 'close', 'volume']] = (
        df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    )
    return df


def _calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _calc_htf_bias(df_5m: pd.DataFrame) -> str:
    """
    1h SMA20 bias computed from 5min bars resampled in-memory.
    Returns 'bullish', 'bearish', or 'neutral'.
    """
    if df_5m.empty or len(df_5m) < 25:
        return 'neutral'
    df_1h = (
        df_5m.set_index('dt')['close']
        .resample('1h').last()
        .dropna()
        .to_frame('close')
    )
    if len(df_1h) < 20:
        return 'neutral'
    sma20 = df_1h['close'].rolling(20).mean()
    last_sma = sma20.iloc[-1]
    if pd.isna(last_sma):
        return 'neutral'
    last_close = float(df_1h['close'].iloc[-1])
    last_sma   = float(last_sma)
    if last_close > last_sma * 1.001:
        return 'bullish'
    if last_close < last_sma * 0.999:
        return 'bearish'
    return 'neutral'


# ─────────────────────────────────────────────────────────────
#  PAPER LOG TABLE
# ─────────────────────────────────────────────────────────────

def init_vwap_paper_log():
    """Create vwap_paper_log table if it doesn't exist (NQ paper tracking)."""
    conn = _db.connect()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS vwap_paper_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol      TEXT NOT NULL,
            direction   TEXT NOT NULL,
            entry_price REAL,
            stop        REAL,
            target      REAL,
            rr          REAL,
            vwap        REAL,
            upper_band  REAL,
            lower_band  REAL,
            htf_bias    TEXT,
            logged_at   TEXT
        )
    ''')
    conn.commit()
    conn.close()


def log_h_paper(signal: dict):
    """Log a Setup H paper signal (NQ) to vwap_paper_log."""
    try:
        init_vwap_paper_log()
        conn = _db.connect()
        now  = datetime.now(timezone.utc).isoformat()
        conn.execute(
            'INSERT INTO vwap_paper_log '
            '(symbol, direction, entry_price, stop, target, rr, vwap, '
            ' upper_band, lower_band, htf_bias, logged_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (
                signal.get('symbol'), signal.get('direction'),
                signal.get('entry'),  signal.get('stop'),
                signal.get('target'), signal.get('rr'),
                signal.get('vwap'),   signal.get('upper_band'),
                signal.get('lower_band'), signal.get('htf_bias'), now,
            )
        )
        conn.commit()
        conn.close()
        logger.info(f'Setup H NQ paper signal logged to vwap_paper_log')
    except Exception as e:
        logger.error(f'log_h_paper failed: {e}', exc_info=True)


# ─────────────────────────────────────────────────────────────
#  MAIN SCAN
# ─────────────────────────────────────────────────────────────

def scan_setup_h(symbol: str = 'ES', dt: datetime = None,
                 paper_only: bool = False) -> dict | None:
    """
    Scan for VWAP 2σ band reversal signal.
    Returns signal dict or None if no edge.
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    # Weekend guard
    if dt.weekday() >= 5:
        return None

    # Session gate: 13:00–19:00 UTC, skip first 5 bars (before 13:25)
    if not (SESSION_START_UTC <= dt.hour < SESSION_END_UTC):
        return None
    if dt.hour == SESSION_START_UTC and dt.minute < 25:
        return None

    # Load bars
    df = _load_bars(symbol, limit=500)
    if df.empty or len(df) < 30:
        return None

    # Tag session bars for today
    today      = dt.date()
    df['date'] = df['dt'].dt.date
    sess_mask  = (df['date'] == today) & (df['dt'].dt.hour >= SESSION_START_UTC)
    sess_idx   = df.index[sess_mask].tolist()

    if len(sess_idx) < 3:
        return None   # fewer than 3 session bars — bands too narrow

    # ── Session VWAP + expanding ±2σ bands ────────────────
    df['tp']    = (df['high'] + df['low'] + df['close']) / 3.0
    sess_df     = df.loc[sess_idx].copy().reset_index(drop=True)
    cum_tpv     = (sess_df['tp'] * sess_df['volume']).cumsum()
    cum_vol     = sess_df['volume'].cumsum().replace(0, np.nan)
    vwap_s      = cum_tpv / cum_vol
    std_s       = sess_df['tp'].expanding().std().fillna(0)
    upper_s     = vwap_s + 2 * std_s
    lower_s     = vwap_s - 2 * std_s

    curr_i      = len(sess_df) - 1
    prev_i      = curr_i - 1

    curr_close  = float(sess_df['close'].iloc[curr_i])
    prev_close  = float(sess_df['close'].iloc[prev_i])
    curr_upper  = float(upper_s.iloc[curr_i])
    curr_lower  = float(lower_s.iloc[curr_i])
    prev_upper  = float(upper_s.iloc[prev_i])
    prev_lower  = float(lower_s.iloc[prev_i])
    curr_vwap   = float(vwap_s.iloc[curr_i])
    band_width  = curr_upper - curr_lower

    # Skip if bands are degenerate (std near zero — first few session bars)
    if band_width < 0.01:
        return None

    # ── 1h SMA20 HTF bias ─────────────────────────────────
    htf_bias = _calc_htf_bias(df)

    # ── ATR14 ─────────────────────────────────────────────
    atr14 = float(_calc_atr(df, 14).iloc[-1])
    if pd.isna(atr14) or atr14 <= 0:
        return None

    # ── Signal detection ──────────────────────────────────
    direction = None

    # SHORT: prev closed above upper band → curr closes back below → 1h bias bearish
    if prev_close > prev_upper and curr_close < curr_upper and htf_bias == 'bearish':
        direction = 'short'

    # LONG: prev closed below lower band → curr closes back above → 1h bias bullish
    elif prev_close < prev_lower and curr_close > curr_lower and htf_bias == 'bullish':
        direction = 'long'

    if direction is None:
        logger.debug(
            f'Setup H {symbol}: no signal | close={curr_close:.2f} '
            f'upper={curr_upper:.2f} lower={curr_lower:.2f} '
            f'vwap={curr_vwap:.2f} htf={htf_bias}'
        )
        return None

    # ── Levels ────────────────────────────────────────────
    entry = round(curr_close, 2)
    stop  = round(
        entry - 1.5 * atr14 if direction == 'long' else entry + 1.5 * atr14, 2
    )

    # Target = session VWAP. If VWAP already past entry (crossed), use 2×ATR.
    if direction == 'long' and curr_vwap > entry:
        target = round(entry + 2.0 * atr14, 2)
    elif direction == 'short' and curr_vwap < entry:
        target = round(entry - 2.0 * atr14, 2)
    else:
        target = round(curr_vwap, 2)

    rr = round(abs(target - entry) / abs(entry - stop), 2) if abs(entry - stop) > 0 else 0

    logger.info(
        f'Setup H {symbol} SIGNAL: {direction.upper()} @ {entry:.2f} | '
        f'vwap={curr_vwap:.2f} upper={curr_upper:.2f} lower={curr_lower:.2f} | '
        f'stop={stop:.2f} target={target:.2f} htf={htf_bias}'
    )

    return {
        'symbol':     symbol,
        'direction':  direction,
        'setup':      'H_vwap_rev',
        'mode':       'intraday',
        'entry':      entry,
        'stop':       stop,
        'target':     target,
        'rr':         rr,
        'session':    'NY Primary (13-19 UTC)',
        'quality':    'primary',
        'paper_only': paper_only,
        'vwap':       round(curr_vwap, 2),
        'upper_band': round(curr_upper, 2),
        'lower_band': round(curr_lower, 2),
        'band_width': round(band_width, 2),
        'htf_bias':   htf_bias,
        'atr':        round(atr14, 2),
        'timestamp':  dt.isoformat(),
    }


# ─────────────────────────────────────────────────────────────
#  DASHBOARD STATE (always callable, no session gate)
# ─────────────────────────────────────────────────────────────

def get_h_state(symbol: str = 'ES') -> dict:
    """
    Return current VWAP band levels for dashboard display.
    Safe to call at any time — no session gate.
    """
    try:
        dt  = datetime.now(timezone.utc)
        df  = _load_bars(symbol, limit=500)
        if df.empty:
            return {'symbol': symbol, 'ok': False, 'error': 'no data'}

        today      = dt.date()
        df['date'] = df['dt'].dt.date
        df['tp']   = (df['high'] + df['low'] + df['close']) / 3.0
        last_close = float(df['close'].iloc[-1])
        atr14      = float(_calc_atr(df, 14).iloc[-1])
        htf_bias   = _calc_htf_bias(df)

        sess_mask = (df['date'] == today) & (df['dt'].dt.hour >= SESSION_START_UTC)
        sess_idx  = df.index[sess_mask].tolist()
        in_session = (dt.weekday() < 5) and (SESSION_START_UTC <= dt.hour < SESSION_END_UTC)

        if len(sess_idx) < 2:
            return {
                'symbol':       symbol,
                'ok':           True,
                'in_session':   in_session,
                'price':        last_close,
                'htf_bias':     htf_bias,
                'vwap':         None,
                'upper_band':   None,
                'lower_band':   None,
                'atr':          round(atr14, 2) if not pd.isna(atr14) else None,
                'signal_state': 'PRE_SESSION',
            }

        sess_df = df.loc[sess_idx].copy().reset_index(drop=True)
        cum_tpv = (sess_df['tp'] * sess_df['volume']).cumsum()
        cum_vol = sess_df['volume'].cumsum().replace(0, np.nan)
        vwap_s  = cum_tpv / cum_vol
        std_s   = sess_df['tp'].expanding().std().fillna(0)
        upper   = float((vwap_s + 2 * std_s).iloc[-1])
        lower   = float((vwap_s - 2 * std_s).iloc[-1])
        curr_v  = float(vwap_s.iloc[-1])

        dist_upper = round((upper - last_close) / atr14, 2) if atr14 > 0 else None
        dist_lower = round((last_close - lower) / atr14, 2) if atr14 > 0 else None

        if last_close > upper:
            state = 'ABOVE_UPPER'
        elif last_close < lower:
            state = 'BELOW_LOWER'
        elif in_session:
            state = 'WATCHING'
        else:
            state = 'NO_EDGE'

        return {
            'symbol':           symbol,
            'ok':               True,
            'in_session':       in_session,
            'price':            last_close,
            'vwap':             round(curr_v, 2),
            'upper_band':       round(upper, 2),
            'lower_band':       round(lower, 2),
            'dist_upper_atr':   dist_upper,
            'dist_lower_atr':   dist_lower,
            'htf_bias':         htf_bias,
            'atr':              round(atr14, 2) if not pd.isna(atr14) else None,
            'signal_state':     state,
            'session_bars':     len(sess_idx),
        }
    except Exception as e:
        logger.warning(f'get_h_state {symbol} error: {e}')
        return {'symbol': symbol, 'ok': False, 'error': str(e)}


# ─────────────────────────────────────────────────────────────
#  ALERT FORMATTER
# ─────────────────────────────────────────────────────────────

def format_h_alert(signal: dict) -> str:
    """Format Setup H Telegram alert."""
    now   = datetime.now(timezone.utc).astimezone(NY_TZ).strftime('%Y-%m-%d %H:%M')
    sym   = signal['symbol']
    dir_  = signal['direction'].upper()
    sep   = chr(9473) * 20
    emoji = '\U0001F4C8' if dir_ == 'LONG' else '\U0001F4C9'

    entry      = signal['entry']
    stop       = signal['stop']
    target     = signal['target']
    vwap       = signal['vwap']
    upper      = signal['upper_band']
    lower      = signal['lower_band']
    bias       = signal['htf_bias'].upper()
    atr        = signal['atr']
    stop_pts   = abs(round(entry - stop,   2))
    target_pts = abs(round(entry - target, 2))
    band_hit   = f'+2\u03c3 ({upper:.2f})' if dir_ == 'SHORT' else f'-2\u03c3 ({lower:.2f})'

    parts = [
        f'{emoji} <b>APEX H \u2014 VWAP Reversal \u2014 {sym}</b>',
        sep,
        f'<b>Setup:</b>     H \u2014 VWAP 2\u03c3 Band Reversal',
        f'<b>Direction:</b> {dir_}',
        f'<b>Edge:</b>      OOS Sharpe 6.03 | 76.5% WR | 268 trades',
        sep,
        f'<b>Entry:</b>     {entry:.2f}',
        f'<b>Stop:</b>      {stop:.2f}  ({stop_pts} pts | 1.5\u00d7 ATR)',
        f'<b>Target:</b>    {target:.2f}  ({target_pts} pts | VWAP)',
        f'<b>R:R:</b>       {signal["rr"]}x',
        sep,
        f'<b>Session VWAP:</b> {vwap:.2f}',
        f'<b>Band hit:</b>    {band_hit}',
        f'<b>ATR14:</b>       {atr:.2f}',
        f'<b>1h Bias:</b>     {bias}',
        sep,
        f'<i>{now} ET</i>',
        f'<i>ES VWAP reversal from 2\u03c3 band \u2014 NY session</i>',
    ]
    return '\n'.join(parts)
