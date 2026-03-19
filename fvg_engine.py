"""
APEX FVG Engine — fvg_engine.py
================================
Fair Value Gap detection and scanning.
Setup D — 15min FVG + 4hour bias + 1min entry trigger.
"""

import sqlite3
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

logger  = logging.getLogger('APEX.FVG')
DB_PATH = 'apex_market.db'

FVG_PARAMS = {
    'min_fvg_atr':        0.3,
    'target_rr':          2.0,
    'max_trades_session': 3,
    'vol_multiplier':     1.0,
    'session_windows': {
        'NQ': [{'start': 13, 'end': 19}],
        # ES disabled — last 3 months negative, monitoring
        # GC disabled — failed walk-forward validation
    },
    'fvg_lookback_bars':  96,   # look back 96 x 15min = 24 hours
}


def load_bars(symbol, timeframe, limit=500):
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query(
        'SELECT ts,open,high,low,close,volume FROM ohlcv '
        'WHERE symbol=? AND timeframe=? ORDER BY ts DESC LIMIT ?',
        conn, params=(symbol, timeframe, limit)
    )
    conn.close()
    df = df.iloc[::-1].reset_index(drop=True)
    df['dt'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    df.set_index('dt', inplace=True)
    return df


def calc_atr(df, period=14):
    high  = df['high']
    low   = df['low']
    close = df['close'].shift(1)
    tr    = pd.concat([high-low,
                       (high-close).abs(),
                       (low-close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def detect_fvgs(df, atr, min_atr_mult=0.3, lookback=96):
    """Detect recent FVGs on a dataframe."""
    fvgs = []
    highs  = df['high'].values
    lows   = df['low'].values
    times  = df.index
    n      = len(df)
    start  = max(1, n - lookback)

    for i in range(start, n-1):
        atr_val = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else 0
        if atr_val == 0:
            continue

        # Bullish FVG — gap up
        if lows[i+1] > highs[i-1]:
            size = lows[i+1] - highs[i-1]
            if size >= min_atr_mult * atr_val:
                fvgs.append({
                    'type':     'bullish',
                    'top':      float(lows[i+1]),
                    'bottom':   float(highs[i-1]),
                    'mid':      float((lows[i+1] + highs[i-1]) / 2),
                    'size':     float(size),
                    'atr':      float(atr_val),
                    'formed_at':times[i+1],
                    'filled':   False,
                })

        # Bearish FVG — gap down
        if highs[i+1] < lows[i-1]:
            size = float(lows[i-1] - highs[i+1])
            if size >= min_atr_mult * atr_val:
                fvgs.append({
                    'type':     'bearish',
                    'top':      float(lows[i-1]),
                    'bottom':   float(highs[i+1]),
                    'mid':      float((lows[i-1] + highs[i+1]) / 2),
                    'size':     float(size),
                    'atr':      float(atr_val),
                    'formed_at':times[i+1],
                    'filled':   False,
                })

    return fvgs


def get_htf_bias(symbol):
    """Get 4hour bias using EMA."""
    try:
        df = load_bars(symbol, '4hour', limit=50)
        if len(df) < 20:
            return 'neutral'
        ema        = df['close'].ewm(span=20).mean()
        last_close = float(df['close'].iloc[-1])
        last_ema   = float(ema.iloc[-1])
        if last_close > last_ema * 1.001:
            return 'bullish'
        elif last_close < last_ema * 0.999:
            return 'bearish'
        return 'neutral'
    except Exception as e:
        logger.error(f'HTF bias error {symbol}: {e}')
        return 'neutral'


def get_session_trade_count(symbol, session_start_hour):
    """Count FVG trades already taken in current session."""
    try:
        conn = sqlite3.connect(DB_PATH)
        now  = datetime.now(timezone.utc)
        session_start = now.replace(
            hour=session_start_hour, minute=0, second=0, microsecond=0
        )
        if now.hour < session_start_hour:
            return 0
        start_ts = int(session_start.timestamp())
        result   = conn.execute(
            'SELECT COUNT(*) FROM apex_trades WHERE symbol=? AND setup LIKE ? AND entry_time>=?',
            (symbol, 'FVG%', session_start.isoformat())
        ).fetchone()[0]
        conn.close()
        return result
    except Exception:
        return 0


def scan_fvg(symbol, dt=None):
    """
    Scan for FVG signals on a symbol.
    Returns list of signal dicts (can be multiple per scan).
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    signals = []
    params  = FVG_PARAMS

    # Check session
    hour = dt.hour
    sessions = params['session_windows'].get(symbol, [])
    in_session = any(s['start'] <= hour < s['end'] for s in sessions)
    if not in_session:
        return signals

    # Check session trade count
    sess_start = next((s['start'] for s in sessions if s['start'] <= hour < s['end']), None)
    if sess_start is None:
        return signals

    trade_count = get_session_trade_count(symbol, sess_start)
    if trade_count >= params['max_trades_session']:
        return signals

    # Get HTF bias
    bias = get_htf_bias(symbol)
    if bias == 'neutral':
        return signals

    # Load 15min bars for FVG detection
    df_15m  = load_bars(symbol, '15min', limit=200)
    atr_15m = calc_atr(df_15m, 14)
    fvgs    = detect_fvgs(df_15m, atr_15m,
                          params['min_fvg_atr'],
                          params['fvg_lookback_bars'])

    if not fvgs:
        return signals

    # Load 1min bars for entry trigger
    df_1m  = load_bars(symbol, '1min', limit=50)
    atr_1m = calc_atr(df_1m, 14)

    if df_1m.empty:
        return signals

    # Current bar
    last_bar   = df_1m.iloc[-1]
    last_close = float(last_bar['close'])
    last_open  = float(last_bar['open'])
    last_high  = float(last_bar['high'])
    last_low   = float(last_bar['low'])
    last_vol   = float(last_bar['volume'])
    last_ts    = int(last_bar['ts'])

    # Volume baseline
    vol_baseline = float(df_1m['volume'].tail(20).mean())

    # Check each FVG
    for fvg in fvgs:
        # Only use FVGs formed before current bar
        if fvg['formed_at'] >= df_1m.index[-1]:
            continue

        direction = None

        # Bullish FVG — price returns to fill, go long
        if (fvg['type'] == 'bullish'
                and bias == 'bullish'
                and last_low  <= fvg['top']
                and last_close >= fvg['mid']
                and last_close > last_open):
            direction = 'long'

        # Bearish FVG — price returns to fill, go short
        elif (fvg['type'] == 'bearish'
                and bias == 'bearish'
                and last_high >= fvg['bottom']
                and last_close <= fvg['mid']
                and last_close < last_open):
            direction = 'short'

        if direction is None:
            continue

        # Volume filter
        if vol_baseline > 0 and last_vol < params['vol_multiplier'] * vol_baseline:
            continue

        # Calculate levels
        atr_val = float(atr_1m.iloc[-1]) if not pd.isna(atr_1m.iloc[-1]) else fvg['atr'] / 15

        if direction == 'long':
            entry  = last_close
            stop   = fvg['bottom'] - 0.1 * atr_val
            target = entry + params['target_rr'] * (entry - stop)
        else:
            entry  = last_close
            stop   = fvg['top'] + 0.1 * atr_val
            target = entry - params['target_rr'] * (stop - entry)

        risk = abs(entry - stop)
        if risk == 0:
            continue

        rr = round(abs(target - entry) / risk, 2)

        signals.append({
            'symbol':    symbol,
            'direction': direction,
            'setup':     f'FVG_{fvg["type"]}',
            'mode':      'scalp',
            'entry':     round(entry, 2),
            'stop':      round(stop, 2),
            'target':    round(target, 2),
            'rr':        rr,
            'fvg_top':   round(fvg['top'], 2),
            'fvg_bottom':round(fvg['bottom'], 2),
            'fvg_mid':   round(fvg['mid'], 2),
            'bias':      bias,
            'quality':   'primary',
            'session':   f'Session {sess_start:02d}:00-{sessions[0]["end"]:02d}:00 UTC',
            'timestamp': df_1m.index[-1],
        })

        # Max 1 signal per scan to avoid flooding
        break

    return signals


def format_fvg_alert(signal):
    """Format FVG signal for Telegram."""
    from zoneinfo import ZoneInfo
    NY  = ZoneInfo('America/New_York')
    now = datetime.now(timezone.utc).astimezone(NY).strftime('%Y-%m-%d %H:%M')
    sym = signal['symbol']
    dir_= signal['direction'].upper()
    sep = chr(9473) * 20

    emoji = chr(128200) if dir_ == 'LONG' else chr(128201)
    bias  = signal['bias'].upper()

    parts = [
        f'{emoji} <b>APEX FVG — {sym}</b>',
        sep,
        f'<b>Direction:</b> {dir_}',
        f'<b>Setup:</b>     FVG Fill ({signal["fvg_top"]:.2f}-{signal["fvg_bottom"]:.2f})',
        f'<b>Bias:</b>      {bias} (4hour)',
        sep,
        f'<b>Entry:</b>     {signal["entry"]:.2f}',
        f'<b>Stop:</b>      {signal["stop"]:.2f}',
        f'<b>Target:</b>    {signal["target"]:.2f}',
        f'<b>R:R:</b>       {signal["rr"]}x',
        f'<b>Hold:</b>      ~17 min avg',
        sep,
        f'<i>{now} ET</i>',
    ]
    return chr(10).join(parts)


if __name__ == '__main__':
    import os
    # API key loaded from environment variable — never hardcode keys
    print('Testing FVG scanner...')
    for sym in ('NQ', 'ES', 'GC'):
        signals = scan_fvg(sym)
        print(f'{sym}: {len(signals)} signals')
        for s in signals:
            print(f'  {s["direction"].upper()} | entry={s["entry"]} stop={s["stop"]} target={s["target"]}')
