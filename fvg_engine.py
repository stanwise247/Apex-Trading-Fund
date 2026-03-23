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
    'min_fvg_atr':        0.7,   # raised from 0.3 — sub-0.7 ATR FVGs have 44% WR, no edge
    'target_rr':          2.0,
    'max_trades_session': 3,
    'vol_multiplier':     1.0,
    'min_score':          60,   # validated: Score 60 = Sharpe 6.46 OOS
    'session_windows': {
        'NQ': [{'start': 14, 'end': 18}],  # was 13-19: block 13:00 (bad first candle) and 18:00 (negative EV)
        # ES disabled — last 3 months negative, monitoring
        # GC disabled — failed walk-forward validation
    },
    'fvg_lookback_bars':  96,
}


def load_bars(symbol, timeframe, limit=500):
    conn = sqlite3.connect(DB_PATH, timeout=30)
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


# Track FVG zones already alerted — persisted in DB so alerts survive process restarts

def _init_fvg_alerted_table():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS fvg_alerted_zones (
            symbol     TEXT NOT NULL,
            formed_at  TEXT NOT NULL,
            alerted_at TEXT NOT NULL,
            PRIMARY KEY (symbol, formed_at)
        )
    ''')
    conn.commit()
    conn.close()

def _is_fvg_already_alerted(symbol, fvg_formed_at):
    try:
        _init_fvg_alerted_table()
        conn = sqlite3.connect(DB_PATH, timeout=30)
        row  = conn.execute(
            'SELECT 1 FROM fvg_alerted_zones WHERE symbol=? AND formed_at=?',
            (symbol, str(fvg_formed_at))
        ).fetchone()
        conn.close()
        return row is not None
    except Exception as e:
        # Fail safe: if we can't confirm the zone hasn't been alerted, assume it has
        # to prevent duplicate entry alerts under DB contention
        logger.warning(f'_is_fvg_already_alerted DB error (skipping alert): {e}')
        return True

def _mark_fvg_alerted(symbol, fvg_formed_at):
    try:
        _init_fvg_alerted_table()
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.execute(
            'INSERT OR IGNORE INTO fvg_alerted_zones (symbol, formed_at, alerted_at) VALUES (?, ?, ?)',
            (symbol, str(fvg_formed_at), datetime.now(timezone.utc).isoformat())
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f'_mark_fvg_alerted DB error: {e}')


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
        conn = sqlite3.connect(DB_PATH, timeout=30)
        now  = datetime.now(timezone.utc)
        session_start = now.replace(
            hour=session_start_hour, minute=0, second=0, microsecond=0
        )
        if now.hour < session_start_hour:
            return 0
        result   = conn.execute(
            'SELECT COUNT(*) FROM apex_trades WHERE symbol=? AND setup LIKE ? AND entry_time>=?',
            (symbol, 'FVG%', session_start.isoformat())
        ).fetchone()[0]
        conn.close()
        return result
    except Exception:
        return 0


def score_fvg(fvg, df_15m, current_bar_time, vol_baseline):
    """Score FVG quality 0-100. Min 60 required to trade."""
    score = 0

    # 1. SIZE (0-30 pts)
    size_ratio = fvg['size'] / fvg['atr'] if fvg['atr'] > 0 else 0
    if size_ratio >= 1.0:   score += 30
    elif size_ratio >= 0.7: score += 20
    elif size_ratio >= 0.4: score += 10
    else:                   score += 5

    # 2. FRESHNESS (0-25 pts)
    try:
        if fvg['formed_at'] in df_15m.index:
            formed_pos  = df_15m.index.get_loc(fvg['formed_at'])
            current_pos = df_15m.index.searchsorted(current_bar_time, side='right') - 1
            age_bars    = max(0, current_pos - formed_pos)
        else:
            age_bars = 99
        if age_bars <= 4:    score += 25
        elif age_bars <= 8:  score += 18
        elif age_bars <= 16: score += 10
        elif age_bars <= 32: score += 5
    except Exception:
        score += 10

    # 3. CLEAN (0-25 pts) — not heavily penetrated
    try:
        since = df_15m[df_15m.index >= fvg['formed_at']]
        if fvg['type'] == 'bullish':
            pen = (fvg['top'] - float(since['low'].min())) / fvg['size'] if fvg['size'] > 0 else 1
        else:
            pen = (float(since['high'].max()) - fvg['bottom']) / fvg['size'] if fvg['size'] > 0 else 1
        if pen <= 0.25:   score += 25
        elif pen <= 0.50: score += 15
        elif pen <= 0.75: score += 5
    except Exception:
        score += 10

    # 4. VOLUME (0-20 pts)
    try:
        if fvg['formed_at'] in df_15m.index and vol_baseline > 0:
            vol_ratio = float(df_15m.loc[fvg['formed_at'], 'volume']) / vol_baseline
            if vol_ratio >= 2.0:   score += 20
            elif vol_ratio >= 1.5: score += 15
            elif vol_ratio >= 1.0: score += 8
            else:                  score += 3
    except Exception:
        score += 8

    return score


def scan_fvg(symbol, dt=None):
    """
    Scan for FVG signals on a symbol.
    Returns list of signal dicts (can be multiple per scan).
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    signals = []
    params  = FVG_PARAMS

    # Check weekend
    if dt.weekday() >= 5:
        logger.debug(f'FVG scan skipped — weekend')
        return signals

    # Check session
    hour = dt.hour
    sessions = params['session_windows'].get(symbol, [])
    in_session = any(s['start'] <= hour < s['end'] for s in sessions)
    if not in_session:
        return signals

    # Check session trade count
    sess_match = next((s for s in sessions if s['start'] <= hour < s['end']), None)
    if sess_match is None:
        return signals
    sess_start = sess_match['start']
    sess_end   = sess_match['end']

    trade_count = get_session_trade_count(symbol, sess_start)
    if trade_count >= params['max_trades_session']:
        return signals

    # Get HTF bias
    bias = get_htf_bias(symbol)
    if bias == 'neutral':
        return signals

    # Load 15min bars for FVG detection
    df_15m  = load_bars(symbol, '15min', limit=200)

    # Staleness check — 15min data must be within 30 minutes of now
    if not df_15m.empty:
        last_15m_time = df_15m.index[-1]
        if last_15m_time.tzinfo is None:
            last_15m_time = last_15m_time.tz_localize('UTC')
        age_15m = (dt.replace(tzinfo=timezone.utc) - last_15m_time).total_seconds() / 60
        if age_15m > 30:
            logger.warning(f'FVG scan skipped — 15min data {round(age_15m)}min stale (last bar {last_15m_time.strftime("%H:%M")} UTC)')
            return signals

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

    # Staleness check — last bar must be within 15 minutes of now
    last_bar_time = df_1m.index[-1]
    if last_bar_time.tzinfo is None:
        last_bar_time = last_bar_time.tz_localize('UTC')
    age_minutes = (dt.replace(tzinfo=timezone.utc) - last_bar_time).total_seconds() / 60
    if age_minutes > 15:
        logger.warning(f'FVG scan skipped — data {round(age_minutes)}min stale (last bar {last_bar_time.strftime("%H:%M")} UTC)')
        return signals

    # Current bar
    last_bar   = df_1m.iloc[-1]
    last_close = float(last_bar['close'])
    last_open  = float(last_bar['open'])
    last_high  = float(last_bar['high'])
    last_low   = float(last_bar['low'])
    last_vol   = float(last_bar['volume'])
    last_ts    = int(last_bar['ts'])

    # Volume baseline — use 15min bars for a more stable baseline
    vol_baseline = float(df_15m['volume'].tail(20).mean())

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

        # Duplicate prevention
        if _is_fvg_already_alerted(symbol, fvg['formed_at']):
            continue

        # Quality score filter
        vol_baseline_15m = float(df_15m['volume'].tail(20).mean())
        fvg_score = score_fvg(fvg, df_15m, df_1m.index[-1], vol_baseline_15m)
        min_score = params.get('min_score', 60)
        if fvg_score < min_score:
            continue

        # Volume filter
        if vol_baseline > 0 and last_vol < params['vol_multiplier'] * vol_baseline:
            continue

        _mark_fvg_alerted(symbol, fvg['formed_at'])

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
            'session':   f'Session {sess_start:02d}:00-{sess_end:02d}:00 UTC',
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

    entry  = signal["entry"]
    stop   = signal["stop"]
    target = signal["target"]
    stop_pts   = abs(round(entry - stop, 2))
    target_pts = abs(round(entry - target, 2))

    parts = [
        f'{emoji} <b>APEX FVG — {sym}</b>',
        sep,
        f'<b>Direction:</b> {dir_}',
        f'<b>Setup:</b>     FVG Fill ({signal["fvg_top"]:.2f}-{signal["fvg_bottom"]:.2f})',
        f'<b>Bias:</b>      {bias} (4hour)',
        sep,
        f'<b>Entry:</b>     {entry:.2f}',
        f'<b>Stop:</b>      {stop:.2f} ({stop_pts} pts)',
        f'<b>Target:</b>    {target:.2f} ({target_pts} pts)',
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
