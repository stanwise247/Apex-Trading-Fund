"""
APEX Live Data Feed — data_feed.py
=====================================
Uses Databento Python client for reliable data fetching.
Two modes:
  1. refresh_all() — fetch recent bars every 5 minutes (historical API)
  2. start_live_feed() — stream real-time bars as they close (live API)

Instruments: NQ, ES, GC
"""

import logging
import json
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import databento as db
from db import connect as _db_connect, IS_POSTGRES, read_sql as _db_read_sql, upsert_ohlcv as _db_upsert

logger = logging.getLogger('APEX.DataFeed')

INSTRUMENTS = {
    'NQ': 'NQ.c.0',
    'ES': 'ES.c.0',
    'GC': 'GC.c.0',
}

# Session hours for staleness checks (UTC)
_SESSION_CHECK_START = 13   # 9 AM ET
_SESSION_CHECK_END   = 20   # 4 PM ET (1 hour past close for lingering checks)
_MAX_STALE_MINUTES   = 10   # alert threshold during session
_ALERT_THROTTLE_SEC  = 1800 # max one alert per 30 min per symbol/tf

# Module-level throttle state for stale alerts
_stale_alerted: dict = {}


def get_api_key() -> str:
    key = os.environ.get('DATABENTO_API_KEY', '')
    if not key:
        try:
            with open('config.json') as f:
                cfg = json.load(f)
            key = cfg.get('DATABENTO_API_KEY', cfg.get('databento_api_key', ''))
        except Exception:
            pass
    return key


def store_bars(symbol: str, timeframe: str, bars: list) -> int:
    """Store bars in DB. Returns number of new/updated bars inserted."""
    if not bars:
        return 0
    conn = _db_connect()
    count_before = conn.execute(
        'SELECT COUNT(*) FROM ohlcv WHERE symbol=? AND timeframe=?',
        (symbol, timeframe)
    ).fetchone()[0]
    for bar in bars:
        ts, o, h, l, close, vol = bar
        try:
            # upsert_ohlcv intentionally overwrites stale bars with fresh Databento data.
            # server.py:store_ohlcv uses INSERT OR IGNORE — two deliberate strategies.
            _db_upsert(conn, symbol, timeframe, ts, o, h, l, close, vol)
        except Exception as e:
            logger.warning(f'Insert error {symbol} {timeframe}: {e}')
    conn.commit()
    count_after = conn.execute(
        'SELECT COUNT(*) FROM ohlcv WHERE symbol=? AND timeframe=?',
        (symbol, timeframe)
    ).fetchone()[0]
    conn.close()
    return count_after - count_before


def fetch_recent_bars(symbol: str, timeframe: str,
                      lookback_hours: int = 48) -> list:
    """
    Fetch recent bars using Databento Python client.
    Automatically handles schema and price conversion.

    end = now - 2min: avoids the slow get_dataset_range() metadata call that
    returned a cached, lagging avail_end (root cause of 15-20 min staleness).
    Databento returns bars up to whatever is available — no pre-query needed.
    Retries once on failure before logging WARNING.
    """
    key = get_api_key()
    if not key:
        logger.error('No Databento API key')
        return []

    db_symbol = INSTRUMENTS.get(symbol)
    if not db_symbol:
        return []

    # HTF bars must come from build_htf_from_5min() — not the SDK.
    # The ohlcv-1h Databento schema returns prices ×1000 too large via the
    # Python SDK (SDK divides by 1e9 but raw integers are price×1e12).
    if timeframe in ('1hour', '4hour'):
        logger.debug(f'fetch_recent_bars: skipping {timeframe} — use build_htf_from_5min()')
        return []

    # Map timeframe to schema and aggregation
    schema_map = {
        '1min':  ('ohlcv-1m', 1),
        '5min':  ('ohlcv-1m', 5),
        '15min': ('ohlcv-1m', 15),
        '1day':  ('ohlcv-1d', 1),
    }
    schema, agg = schema_map.get(timeframe, ('ohlcv-1m', 5))

    # end = now - 2min: Databento returns up to what's available.
    # No get_dataset_range() call — that was slow (extra round-trip) and
    # its cached avail_end lagged 15-20 min, making all fetched bars stale.
    end   = datetime.now(timezone.utc) - timedelta(minutes=2)
    start = end - timedelta(hours=lookback_hours)

    last_exc = None
    for attempt in range(2):
        try:
            client = db.Historical(key)
            data   = client.timeseries.get_range(
                dataset   = 'GLBX.MDP3',
                symbols   = [db_symbol],
                schema    = schema,
                start     = start,
                end       = end,
                stype_in  = 'continuous',
            )

            df = data.to_df()
            if df.empty:
                return []

            # Databento Python client already converts prices — no division needed
            # Index is ts_event as DatetimeIndex
            df.index = pd.to_datetime(df.index, utc=True)

            # Aggregate if needed (e.g. 1min -> 5min)
            if agg > 1:
                df = df.resample(f"{agg}min").agg({
                    "open":   "first",
                    "high":   "max",
                    "low":    "min",
                    "close":  "last",
                    "volume": "sum",
                }).dropna()

            # Convert to list of tuples
            result = []
            for ts, row in df.iterrows():
                ts_unix = int(ts.timestamp())
                result.append((
                    ts_unix,
                    round(float(row["open"]),  2),
                    round(float(row["high"]),  2),
                    round(float(row["low"]),   2),
                    round(float(row["close"]), 2),
                    float(row.get("volume", 0)),
                ))

            return result

        except Exception as e:
            last_exc = e
            if attempt == 0:
                logger.warning(
                    f'DataFeed fetch {symbol} {timeframe} attempt 1 failed: {e} — retrying'
                )
                time.sleep(1)

    logger.warning(f'DataFeed fetch {symbol} {timeframe} failed after retry: {last_exc}')
    return []


def fetch_bars_range(symbol: str, timeframe: str,
                     start: datetime, end: datetime) -> list:
    """Fetch bars for an explicit date range. Used for chunked historical backfill."""
    key = get_api_key()
    if not key:
        logger.error('No Databento API key')
        return []
    db_symbol = INSTRUMENTS.get(symbol)
    if not db_symbol:
        return []
    if timeframe in ('1hour', '4hour'):
        return []
    schema_map = {
        '1min':  ('ohlcv-1m', 1),
        '5min':  ('ohlcv-1m', 5),
        '15min': ('ohlcv-1m', 15),
        '1day':  ('ohlcv-1d', 1),
    }
    schema, agg = schema_map.get(timeframe, ('ohlcv-1m', 5))
    try:
        client = db.Historical(key)
        data   = client.timeseries.get_range(
            dataset  = 'GLBX.MDP3',
            symbols  = [db_symbol],
            schema   = schema,
            start    = start,
            end      = end,
            stype_in = 'continuous',
        )
        df = data.to_df()
        if df.empty:
            return []
        df.index = pd.to_datetime(df.index, utc=True)
        if agg > 1:
            df = df.resample(f"{agg}min").agg({
                "open":   "first",
                "high":   "max",
                "low":    "min",
                "close":  "last",
                "volume": "sum",
            }).dropna()
        result = []
        for ts, row in df.iterrows():
            result.append((
                int(ts.timestamp()),
                round(float(row["open"]),  2),
                round(float(row["high"]),  2),
                round(float(row["low"]),   2),
                round(float(row["close"]), 2),
                float(row.get("volume", 0)),
            ))
        return result
    except Exception as e:
        logger.error(f'fetch_bars_range error {symbol} {timeframe}: {e}')
        return []


def build_htf_from_5min(symbol: str) -> dict:
    """Build 1hour and 4hour bars by aggregating 5min bars already in DB."""
    conn = _db_connect()
    df   = _db_read_sql(
        'SELECT ts, open, high, low, close, volume FROM ohlcv '
        'WHERE symbol=? AND timeframe=? ORDER BY ts ASC',
        conn, params=(symbol, '5min')
    )
    conn.close()

    if df.empty:
        logger.warning(f'build_htf_from_5min: no 5min bars for {symbol}')
        return {}

    logger.info(f'build_htf_from_5min: {symbol} — {len(df)} 5min bars loaded')
    df['dt'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    df.set_index('dt', inplace=True)

    results = {}
    for tf, rule in [('1hour', '1h'), ('4hour', '4h')]:
        try:
            agg = df.resample(rule).agg({
                'open':   'first',
                'high':   'max',
                'low':    'min',
                'close':  'last',
                'volume': 'sum',
            }).dropna()
            # Use .asi8 (nanoseconds since epoch) — works on tz-aware indexes in all pandas versions.
            # astype('int64') raises TypeError on tz-aware DatetimeIndex in pandas 2.x.
            agg['ts'] = agg.index.asi8 // 1_000_000_000

            bars = [
                (int(row['ts']), round(float(row['open']),2),
                 round(float(row['high']),2), round(float(row['low']),2),
                 round(float(row['close']),2), float(row['volume']))
                for _, row in agg.iterrows()
            ]
            logger.info(f'build_htf_from_5min: {symbol} {tf} — {len(bars)} aggregated bars')
            inserted = store_bars(symbol, tf, bars)
            results[tf] = inserted
            logger.info(f'DataFeed: {symbol} {tf} built — {inserted} new bars ({len(bars)} total)')
        except Exception as e:
            logger.error(f'HTF build error {symbol} {tf}: {e}', exc_info=True)
            results[tf] = 0

    return results


def refresh_symbol(symbol: str, timeframes: list = None,
                   lookback_hours: int = 48) -> dict:
    """Refresh one symbol across timeframes. Each timeframe is isolated —
    a failure on 15min does not block 5min or other timeframes."""
    if timeframes is None:
        timeframes = ['5min', '15min']
    results = {}
    for tf in timeframes:
        try:
            bars     = fetch_recent_bars(symbol, tf, lookback_hours)
            inserted = store_bars(symbol, tf, bars)
            results[tf] = inserted
            if inserted > 0:
                logger.info(f'DataFeed: {symbol} {tf} +{inserted} new bars')
            else:
                logger.debug(f'DataFeed: {symbol} {tf} up to date')
        except Exception as e:
            logger.warning(f'DataFeed: {symbol} {tf} refresh failed — {e}')
            results[tf] = 0
    return results


def refresh_all(include_htf: bool = False,
                include_daily: bool = False) -> dict:
    """
    Refresh all instruments.
    Call every 5 minutes from scheduler.
    include_htf=True every 30 min — rebuilds 1hour/4hour from 5min.
    """
    tfs = ['5min', '15min']
    if include_daily:
        tfs.append('1day')

    results = {}
    for symbol in INSTRUMENTS:
        results[symbol] = refresh_symbol(symbol, tfs)
        if include_htf:
            htf = build_htf_from_5min(symbol)
            results[symbol].update(htf)

    return results


def get_latest_bar(symbol: str, timeframe: str = '5min') -> dict:
    """Return the most recent bar for a symbol."""
    try:
        conn = _db_connect()
        df   = _db_read_sql(
            'SELECT ts, open, high, low, close, volume FROM ohlcv '
            'WHERE symbol=? AND timeframe=? ORDER BY ts DESC LIMIT 1',
            conn, params=(symbol, timeframe)
        )
        conn.close()
        if df.empty:
            return {}
        row = df.iloc[0]
        return {
            'symbol':    symbol,
            'timeframe': timeframe,
            'time':      str(pd.to_datetime(row['ts'], unit='s', utc=True)),
            'open':      round(float(row['open']),  2),
            'high':      round(float(row['high']),  2),
            'low':       round(float(row['low']),   2),
            'close':     round(float(row['close']), 2),
            'volume':    int(row['volume']),
        }
    except Exception as e:
        logger.error(f'get_latest_bar error: {e}')
        return {}


def check_data_freshness():
    """
    During session hours (13:00-20:00 UTC), check critical timeframes for staleness.
    Sends a Telegram alert (throttled to once per 30 min per symbol/tf) if any
    critical timeframe (1min, 5min, 15min) has a last bar > 10 minutes old.
    """
    now_utc = datetime.now(timezone.utc)
    if not (_SESSION_CHECK_START <= now_utc.hour < _SESSION_CHECK_END):
        return

    CRITICAL_TFS = ['1min', '5min', '15min']

    try:
        conn = _db_connect()
        for symbol in INSTRUMENTS:
            for tf in CRITICAL_TFS:
                row = conn.execute(
                    'SELECT MAX(ts) FROM ohlcv WHERE symbol=? AND timeframe=?',
                    (symbol, tf)
                ).fetchone()
                if not row or not row[0]:
                    continue
                age_min = (now_utc.timestamp() - float(row[0])) / 60
                if age_min > _MAX_STALE_MINUTES:
                    alert_key  = f'{symbol}_{tf}'
                    last_alert = _stale_alerted.get(alert_key, 0)
                    if now_utc.timestamp() - last_alert > _ALERT_THROTTLE_SEC:
                        _stale_alerted[alert_key] = now_utc.timestamp()
                        msg = (
                            f'⚠️ APEX DATA ALERT — {symbol} {tf} last bar '
                            f'{age_min:.0f}min ago — data feed issue'
                        )
                        logger.warning(msg)
                        try:
                            from live_scanner import send_telegram
                            send_telegram(msg)
                        except Exception as te:
                            logger.error(f'Failed to send stale data alert: {te}')
        conn.close()
    except Exception as e:
        logger.error(f'check_data_freshness error: {e}')


if __name__ == '__main__':
    import os
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    os.environ['DATABENTO_API_KEY'] = 'db-ac68nRrSbbULVrwXF7rbDETdqB439'

    print('Testing APEX data feed...\n')

    for sym in ('NQ', 'ES', 'GC'):
        print(f'{sym} 5min (last 2 hours):')
        bars = fetch_recent_bars(sym, '5min', lookback_hours=2)
        if bars:
            last = pd.to_datetime(bars[-1][0], unit='s', utc=True)
            print(f'  Bars:       {len(bars)}')
            print(f'  Last bar:   {last}')
            print(f'  Last close: {bars[-1][4]:.2f}')
        else:
            print(f'  No bars returned')
        print()

    print('Full refresh (5min + 15min + HTF)...')
    results = refresh_all(include_htf=True)
    for sym, tfs in results.items():
        for tf, count in tfs.items():
            if count > 0:
                print(f'  {sym} {tf}: +{count} new bars')

    print('\nLatest bars in DB:')
    conn = _db_connect()
    for sym in ('NQ', 'ES', 'GC'):
        for tf in ('5min', '15min', '1hour', '4hour'):
            df = _db_read_sql(
                'SELECT MAX(ts) FROM ohlcv WHERE symbol=? AND timeframe=?',
                conn, params=(sym, tf)
            )
            ts = df.iloc[0,0]
            if ts:
                dt = pd.to_datetime(ts, unit='s', utc=True)
                print(f'  {sym} {tf}: {dt}')
    conn.close()
