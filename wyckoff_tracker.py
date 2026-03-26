"""
APEX Setup G — Wyckoff Upthrust Tracker
========================================
Logging only — no alerts. Confirm at n=30 with OOS Sharpe >= 4.0.
OOS backtest: Sharpe=4.14, n=24 (needs 6 more months).
"""

import sqlite3
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone

logger  = logging.getLogger('APEX.Wyckoff')
DB_PATH = 'apex_market.db'


def _init_wyckoff_table():
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS wyckoff_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT NOT NULL,
                date        TEXT NOT NULL,
                entry_price REAL,
                stop        REAL,
                target      REAL,
                detected_at TEXT NOT NULL,
                resolved    INTEGER DEFAULT 0,
                pnl_r       REAL
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_wyckoff_sym_date ON wyckoff_log (symbol, date)')
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f'wyckoff table init failed: {e}')


def _load_bars(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    df = pd.read_sql_query(
        'SELECT ts, open, high, low, close, volume FROM ohlcv '
        'WHERE symbol=? AND timeframe=? ORDER BY ts DESC LIMIT ?',
        conn, params=(symbol, timeframe, limit)
    )
    conn.close()
    if df.empty:
        return df
    df = df.iloc[::-1].reset_index(drop=True)
    df['dt'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    return df


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low  = df['low']
    prev = df['close'].shift(1)
    tr   = pd.concat([high-low, (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _get_4h_bias(symbol: str) -> str:
    """Returns 'bearish', 'bullish', or 'neutral' based on 4h SMA20."""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        rows = conn.execute(
            'SELECT close FROM ohlcv WHERE symbol=? AND timeframe=? ORDER BY ts DESC LIMIT 25',
            (symbol, '4hour')
        ).fetchall()
        conn.close()
        if len(rows) < 20:
            return 'neutral'
        closes = [r[0] for r in reversed(rows)]
        sma20 = np.mean(closes[-20:])
        last  = closes[-1]
        if last < sma20 * 0.999:
            return 'bearish'
        elif last > sma20 * 1.001:
            return 'bullish'
        return 'neutral'
    except Exception:
        return 'neutral'


def detect_upthrust(symbol: str, df_15m: pd.DataFrame) -> list:
    """
    Detect Wyckoff Upthrust on 15min bars.
    - Price breaks above 20-bar swing high
    - Closes back below swing high within 2 bars
    - Volume at breakout bar > 20-bar average
    - 4h bias must be bearish (filter)
    Returns list of event dicts.
    """
    if df_15m.empty or len(df_15m) < 25:
        return []

    events = []
    atr14  = _atr(df_15m, 14)
    vol_ma = df_15m['volume'].rolling(20).mean()

    for i in range(22, len(df_15m) - 2):
        swing_high = df_15m['high'].iloc[i-20:i].max()
        # Bar i breaks above swing high
        if df_15m['high'].iloc[i] <= swing_high:
            continue
        if df_15m['close'].iloc[i] <= swing_high:
            continue  # Already closed back below — not a clean upthrust yet
        if df_15m['volume'].iloc[i] <= vol_ma.iloc[i]:
            continue  # Volume not elevated

        # Check: does it close back below swing_high within next 2 bars?
        closed_back = False
        close_bar   = None
        for j in [i+1, i+2]:
            if j >= len(df_15m):
                break
            if df_15m['close'].iloc[j] < swing_high:
                closed_back = True
                close_bar   = j
                break

        if not closed_back or close_bar is None:
            continue

        # Entry at close of the bar that closed back below
        entry_price = float(df_15m['close'].iloc[close_bar])
        atr_val     = float(atr14.iloc[close_bar])
        if pd.isna(atr_val) or atr_val <= 0:
            continue

        stop   = round(float(df_15m['high'].iloc[i]) + 0.1 * atr_val, 2)
        target = round(entry_price - 2.0 * atr_val, 2)
        ts     = int(df_15m['ts'].iloc[close_bar])
        dt_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

        events.append({
            'symbol':      symbol,
            'date':        datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d'),
            'entry_price': entry_price,
            'stop':        stop,
            'target':      target,
            'detected_at': datetime.now(timezone.utc).isoformat(),
            'bar_i':       i,
            'close_bar':   close_bar,
        })

    return events


def _already_logged(symbol: str, date: str) -> bool:
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        row  = conn.execute(
            'SELECT id FROM wyckoff_log WHERE symbol=? AND date=?',
            (symbol, date)
        ).fetchone()
        conn.close()
        return row is not None
    except Exception:
        return True  # fail-safe: assume already logged


def _log_event(event: dict):
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.execute(
            'INSERT INTO wyckoff_log (symbol, date, entry_price, stop, target, detected_at) '
            'VALUES (?,?,?,?,?,?)',
            (event['symbol'], event['date'], event['entry_price'],
             event['stop'], event['target'], event['detected_at'])
        )
        conn.commit()
        conn.close()
        logger.info(
            f'Wyckoff upthrust logged: {event["symbol"]} {event["date"]} '
            f'entry={event["entry_price"]} stop={event["stop"]} target={event["target"]}'
        )
    except Exception as e:
        logger.warning(f'wyckoff log insert failed: {e}')


def scan_and_log_wyckoff(symbols: list = None):
    """Scan for upthrusts and log new ones. Called from background_scheduler."""
    _init_wyckoff_table()
    if symbols is None:
        symbols = ['NQ', 'ES', 'GC']

    for sym in symbols:
        try:
            bias = _get_4h_bias(sym)
            if bias != 'bearish':
                continue  # Only log when 4h bearish

            df_15m = _load_bars(sym, '15min', limit=100)
            if df_15m.empty:
                continue

            # Session filter: 13:00-19:00 UTC
            df_15m = df_15m[df_15m['dt'].dt.hour.between(13, 18)].copy().reset_index(drop=True)
            if df_15m.empty:
                continue

            events = detect_upthrust(sym, df_15m)
            for ev in events:
                if not _already_logged(sym, ev['date']):
                    _log_event(ev)
        except Exception as e:
            logger.debug(f'Wyckoff scan error {sym}: {e}')


def get_wyckoff_stats() -> dict:
    """Return stats for dashboard."""
    _init_wyckoff_table()
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        rows = conn.execute(
            'SELECT symbol, date, entry_price, stop, target, detected_at, resolved, pnl_r '
            'FROM wyckoff_log ORDER BY detected_at DESC'
        ).fetchall()
        conn.close()

        total   = len(rows)
        resolved = [r for r in rows if r[6] != 0]
        wins     = [r for r in resolved if r[7] is not None and r[7] > 0]
        wr       = round(len(wins) / len(resolved) * 100, 1) if resolved else None

        events = [
            {'symbol': r[0], 'date': r[1], 'entry_price': r[2], 'stop': r[3],
             'target': r[4], 'detected_at': r[5], 'resolved': r[6], 'pnl_r': r[7]}
            for r in rows
        ]
        return {
            'total':          total,
            'target':         30,
            'progress_pct':   round(min(total / 30 * 100, 100), 1),
            'resolved':       len(resolved),
            'win_rate':       wr,
            'events':         events[:10],  # last 10 for display
            'last_detected':  rows[0][5] if rows else None,
        }
    except Exception as e:
        return {'total': 0, 'target': 30, 'progress_pct': 0, 'error': str(e)}


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    scan_and_log_wyckoff()
    stats = get_wyckoff_stats()
    print(f'Wyckoff log: {stats["total"]} events logged ({stats["progress_pct"]:.0f}% to confirmation)')
