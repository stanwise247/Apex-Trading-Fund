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
        logger.info(f'Setup J {symbol}: Monday — skip (Tue-Fri only)')
        return None
    if dt.weekday() >= 5:   # Weekend
        return None

    # ── Gate: session ────────────────────────────────────────
    if not (SESSION_START <= dt.hour < SESSION_END):
        logger.info(f'Setup J {symbol}: outside session ({dt.hour:02d}:00 UTC, need {SESSION_START}-{SESSION_END})')
        return None

    # ── Gate: per-session dedup ──────────────────────────────
    today_str = dt.strftime('%Y-%m-%d')
    dedup_key = f'{symbol}_J_{today_str}'
    if dedup_key in _j_dedup_sent:
        logger.info(f'Setup J {symbol}: session dedup — already fired today ({today_str})')
        return None

    # ── Load 5min bars — need ~300+ bars: ~216 overnight + 78 prev session ──
    df5 = _load_bars(symbol, '5min', limit=500)
    if df5.empty or len(df5) < 50:
        logger.info(f'Setup J {symbol}: insufficient 5min bars ({len(df5)}) — skip')
        return None

    # ── Compute ATR14 ─────────────────────────────────────────
    atr_series = _atr14_j(df5)
    atr_val    = float(atr_series.iloc[-1] if not atr_series.iloc[-1] != atr_series.iloc[-1] else 0)
    if atr_val <= 0 or math.isnan(atr_val):
        logger.info(f'Setup J {symbol}: ATR14 invalid ({atr_val}) — skip')
        return None

    # ── Previous session bars (yesterday 13–19 UTC) ───────────
    today_date = dt.date()
    prev_bars  = df5[
        (df5['weekday'] < 5) &
        (df5['hour'] >= 13) & (df5['hour'] < 19) &
        (df5['date'] < today_date)
    ].tail(78)  # up to 78 bars = full 6h session at 5min granularity

    if len(prev_bars) < 20:
        logger.info(f'Setup J {symbol}: insufficient prev session bars ({len(prev_bars)}) — skip')
        return None

    vah, val = _calc_value_area(prev_bars, VA_PCT)
    if vah is None:
        logger.warning(
            f'Setup J {symbol}: VAH/VAL failed — '
            f'prev_bars={len(prev_bars)} (need ≥10), today={today_date}'
        )
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
        logger.info(
            f'Setup J {symbol}: no entry conditions met — '
            f'close={bar_close:.2f} VAH={vah:.2f} VAL={val:.2f} '
            f'bias={bias} low={bar_low:.2f} high={bar_high:.2f}'
        )
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
#  DASHBOARD STATE — gate-by-gate breakdown for /api/apex/scan
# ─────────────────────────────────────────────────────────────

def get_setup_j_state(symbol: str) -> dict:
    """
    Return current gate states for Setup J on symbol.
    Used by /api/apex/scan to populate the dashboard Setups tab card.
    Always returns a dict — never raises.
    """
    result = {
        'symbol':          symbol,
        'ok':              False,
        'vah':             None,
        'val':             None,
        'htf_bias':        'unknown',
        'in_session':      False,
        'signal_state_label': 'DEVELOPING',
        'readiness_score': 0,
        'gates':           [],
        'next_gate_description': '—',
    }
    try:
        now = datetime.now(timezone.utc)
        # Gate 1: session (Tue-Fri, 13-19 UTC)
        in_session = (now.weekday() not in (0, 5, 6)) and (SESSION_START <= now.hour < SESSION_END)
        no_monday  = now.weekday() != 0
        g_session = {
            'gate': 1, 'name': 'Session (Tue–Fri 13–19 UTC)',
            'passed': in_session,
            'detail': (
                f'{now.strftime("%A")} {now.hour:02d}:{now.minute:02d} UTC '
                f'{"✓ in session" if in_session else "— off session"}'
            ),
        }
        result['in_session'] = in_session

        # Load 5min bars — 500 bars covers ~41h (overnight + full prev session)
        df5 = _load_bars(symbol, '5min', limit=500)
        if df5.empty or len(df5) < 50:
            result['gates'] = [g_session]
            result['next_gate_description'] = 'Session: ' + g_session['detail']
            return result

        atr_val = float(_atr14_j(df5).iloc[-1] or 0)

        # Gate 2: previous session VAH/VAL computed
        today_date = now.date()
        prev_bars  = df5[
            (df5['weekday'] < 5) &
            (df5['hour'] >= 13) & (df5['hour'] < 19) &
            (df5['date'] < today_date)
        ].tail(78)
        vah, val = _calc_value_area(prev_bars, VA_PCT) if len(prev_bars) >= 10 else (None, None)
        va_ok = vah is not None and val is not None
        if not va_ok and in_session:
            logger.warning(
                f'Setup J {symbol}: VAH/VAL None during session — '
                f'prev_bars={len(prev_bars)}, today={now.date()}, '
                f'df5_rows={len(df5)}'
            )
        g_va = {
            'gate': 2, 'name': 'Previous Session VA Computed',
            'passed': va_ok,
            'detail': (
                f'VAH={vah:.2f} VAL={val:.2f}' if va_ok
                else f'Insufficient prev session bars ({len(prev_bars)})'
            ),
        }
        if va_ok:
            result['vah'] = round(vah, 2)
            result['val'] = round(val, 2)

        # Gate 3: HTF 4h bias
        bias = _get_htf_bias(symbol)
        bias_ok = bias in ('bullish', 'bearish')
        g_bias = {
            'gate': 3, 'name': '4h EMA20 Bias',
            'passed': bias_ok,
            'detail': f'HTF bias={bias.upper()} (need bullish or bearish, not neutral)',
        }
        result['htf_bias'] = bias

        # Gate 4: price near VAH or VAL (within 1 ATR)
        bar = df5.iloc[-1]
        bc  = float(bar['close']); bh = float(bar['high']); bl = float(bar['low'])
        near_vah = va_ok and abs(bc - vah) <= atr_val
        near_val = va_ok and abs(bc - val) <= atr_val
        near_va  = near_vah or near_val
        g_near = {
            'gate': 4, 'name': 'Price Near VAH/VAL (≤1 ATR)',
            'passed': near_va,
            'detail': (
                f'close={bc:.2f} '
                + (f'near VAH={vah:.2f} ✓' if near_vah else (f'near VAL={val:.2f} ✓' if near_val else 'not at VA level'))
            ) if va_ok else 'VA not computed',
        }

        # Gate 5: confirming close (close above VAH for long, below VAL for short)
        conf_long  = va_ok and bl <= vah * 1.001 and bc > vah and bc > float(bar['open'])
        conf_short = va_ok and bh >= val * 0.999 and bc < val and bc < float(bar['open'])
        g_conf = {
            'gate': 5, 'name': 'Confirming Close at VA Level',
            'passed': conf_long or conf_short,
            'detail': (
                'Long confirm: close above VAH ✓' if conf_long
                else 'Short confirm: close below VAL ✓' if conf_short
                else f'No confirming close (bar close={bc:.2f})'
            ),
        }

        # Gate 6: stop cap (0.5 × ATR14 must not exceed symbol max)
        _j_max_stop = {'MNQ': 60, 'ES': 15}.get(symbol, 9999)
        _j_stop_dist = round(STOP_ATR_MULT * atr_val, 1)
        stop_cap_ok = 0 < _j_stop_dist <= _j_max_stop
        g_stopcap = {
            'gate': 6, 'name': 'Stop Cap',
            'passed': stop_cap_ok,
            'detail': f'ATR14={atr_val:.1f} stop={_j_stop_dist}pts cap={_j_max_stop}pts',
        }

        # Gate 7: calendar blackout
        try:
            from calendar_filter import get_filter as _gcf_j
            _j_blocked, _j_cal_rsn = _gcf_j().is_blocked(symbol, now)
            cal_ok = not _j_blocked
            cal_detail = _j_cal_rsn if _j_blocked else 'No blackout'
        except Exception:
            cal_ok = True
            cal_detail = 'Calendar unavailable'
        g_cal = {
            'gate': 7, 'name': 'Calendar',
            'passed': cal_ok,
            'detail': cal_detail,
        }

        # Gate 8: portfolio heat
        try:
            _conn_h = _db.connect()
            _j_heat = _conn_h.execute(
                "SELECT COUNT(*) FROM apex_trades WHERE status='open' AND quality != 'test'"
            ).fetchone()[0]
            _conn_h.close()
        except Exception:
            _j_heat = 0
        _j_max_heat = 3
        heat_ok = _j_heat < _j_max_heat
        g_heat = {
            'gate': 8, 'name': 'Portfolio Heat',
            'passed': heat_ok,
            'detail': f'{_j_heat}/{_j_max_heat} open trades',
        }

        gates = [g_session, g_va, g_bias, g_near, g_conf, g_stopcap, g_cal, g_heat]
        passed = sum(1 for g in gates if g['passed'])
        total  = len(gates)
        score  = round(passed / total * 100)

        if not in_session:
            label = 'OFF SESSION'
        elif not va_ok:
            label = 'DEVELOPING'
        elif passed == total:
            label = 'SIGNAL READY'
        elif passed >= 3:
            label = 'SCANNING'
        else:
            label = 'DEVELOPING'

        first_failed = next((f"{g['name']}: {g['detail']}" for g in gates if not g['passed']), 'All gates passed')

        result.update({
            'ok':              va_ok,
            'signal_state_label': label,
            'readiness_score': score,
            'gates':           gates,
            'next_gate_description': first_failed if label != 'SIGNAL READY' else 'All gates passed — signal ready',
        })

    except Exception as e:
        logger.debug(f'Setup J state error {symbol}: {e}')

    return result


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
