"""
APEX Live Data Feed — data_feed.py
=====================================
Two-mode data feed:

  SESSION (13:00-19:00 UTC):
    LiveBarFeed — background thread streaming ohlcv-1m from Databento Live API.
    Bars stored to DB as each 1-minute bar closes. Zero archive lag.
    HTF (5min/15min/1h/4h) built from 1min by polling every 5 min.

  OFF-SESSION / BACKFILL:
    Historical API — fetches recent bars for 1min/5min/15min.
    Used on startup (backfill) and when live feed is not running.

Instruments: NQ, ES, GC (GLBX.MDP3 continuous front-month)
"""

import logging
import json
import os
import time
import threading
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



# ─────────────────────────────────────────────────────────────
#  LIVE BAR FEED (Databento Live API — zero archive lag)
# ─────────────────────────────────────────────────────────────

class LiveBarFeed:
    """
    Background thread that streams ohlcv-1m bars from Databento Live API.
    Each bar is stored to DB the instant it closes. Auto-reconnects on disconnect.

    Price note: OHLCVMsg.open/high/low/close are fixed-point integers (× 1e9).
                record.pretty_open/high/low/close return the float price.
    Symbol note: Live API delivers SymbolMappingMsg (instrument_id → db_symbol)
                 before bars arrive. We map instrument_id → apex symbol ('NQ' etc.).
    """

    _RECONNECT_DELAY_S = 5

    def __init__(self):
        self._thread         = None
        self._stop_evt       = threading.Event()
        self._live_conn      = None
        self._lock           = threading.Lock()
        self._sym_map        = {}   # instrument_id (int) → 'NQ'/'ES'/'GC'
        self._bar_count             = 0
        self._last_bar_time         = 0.0  # epoch seconds of most recent stored bar (any symbol)
        self._last_bar_time_by_sym  = {}   # symbol → epoch seconds
        self._bar_count_by_sym      = {}   # symbol → lifetime bars stored
        self._window_start_time     = 0.0  # epoch for current 30-min rate window
        self._window_bars_by_sym    = {}   # symbol → bars in current 30-min window
        self._records_seen          = 0    # records processed since last connect

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def bar_count(self) -> int:
        return self._bar_count

    @property
    def last_bar_time(self) -> float:
        return self._last_bar_time

    def seconds_since_last_bar(self) -> float:
        if self._last_bar_time == 0.0:
            return float('inf')
        return time.time() - self._last_bar_time

    def seconds_since_last_bar_for(self, symbol: str) -> float:
        t = self._last_bar_time_by_sym.get(symbol, 0.0)
        if t == 0.0:
            return float('inf')
        return time.time() - t

    def start(self):
        with self._lock:
            if self.is_running:
                return
            self._stop_evt.clear()
            self._thread = threading.Thread(
                target=self._run, name='LiveBarFeed', daemon=True
            )
            self._thread.start()
        logger.info('LiveBarFeed: started — streaming ohlcv-1m for NQ/ES/GC')

    def stop(self):
        self._stop_evt.set()
        with self._lock:
            if self._live_conn:
                try:
                    self._live_conn.stop()
                except Exception:
                    pass
                self._live_conn = None
        logger.info('LiveBarFeed: stopped')

    def _run(self):
        """Outer loop — reconnects on any error."""
        while not self._stop_evt.is_set():
            try:
                self._connect_and_stream()
            except Exception as e:
                if not self._stop_evt.is_set():
                    logger.error(
                        f'LiveBarFeed: stream error: {e} — '
                        f'reconnecting in {self._RECONNECT_DELAY_S}s'
                    )
                    time.sleep(self._RECONNECT_DELAY_S)

    def _connect_and_stream(self):
        key = get_api_key()
        if not key:
            logger.error('LiveBarFeed: DATABENTO_API_KEY not set — feed cannot connect')
            raise RuntimeError('No Databento API key configured')

        live = db.Live(key=key)
        self._sym_map      = {}   # reset on each (re)connect
        self._records_seen = 0
        self._window_start_time  = time.time()
        self._window_bars_by_sym = {}

        # Subscribe all instruments — log each attempt individually so Railway logs
        # show exactly which symbols were attempted and whether any threw on subscribe()
        subscribed = []
        for apex_sym, db_sym in INSTRUMENTS.items():
            try:
                logger.info(f'LiveBarFeed: subscribing {apex_sym} ({db_sym}) on GLBX.MDP3 ohlcv-1m ...')
                live.subscribe(
                    dataset  = 'GLBX.MDP3',
                    schema   = 'ohlcv-1m',
                    symbols  = [db_sym],
                    stype_in = 'continuous',
                )
                subscribed.append(apex_sym)
                logger.info(f'LiveBarFeed: subscribe() returned for {apex_sym} ({db_sym}) — awaiting SymbolMappingMsg')
            except Exception as sub_err:
                logger.error(f'LiveBarFeed: subscribe() FAILED for {apex_sym} ({db_sym}): {sub_err}')

        with self._lock:
            self._live_conn = live

        logger.info(
            f'LiveBarFeed: connected to GLBX.MDP3 ohlcv-1m — '
            f'subscribed={subscribed}  waiting for SymbolMappingMsg per symbol'
        )

        _sym_map_checked = False   # warn once after first batch of records if any symbol unmapped

        for record in live:
            if self._stop_evt.is_set():
                break

            self._records_seen += 1
            rtype = type(record).__name__

            # After 20 records, warn once about any instrument still not mapped
            if not _sym_map_checked and self._records_seen >= 20:
                _sym_map_checked = True
                mapped_syms   = set(self._sym_map.values())
                expected_syms = set(INSTRUMENTS.keys())
                missing = expected_syms - mapped_syms
                if missing:
                    logger.warning(
                        f'LiveBarFeed: {self._records_seen} records processed but '
                        f'NO SymbolMappingMsg received for: {missing} — '
                        f'these symbols will not stream bars. '
                        f'sym_map so far: {self._sym_map}  subscribed: {subscribed}'
                    )
                else:
                    logger.info(f'LiveBarFeed: all symbols mapped after {self._records_seen} records — {self._sym_map}')

            # 30-minute rolling rate window: log bars per symbol and reset
            if self._window_start_time and time.time() - self._window_start_time >= 1800:
                elapsed = round(time.time() - self._window_start_time)
                parts = [f'{s}={self._window_bars_by_sym.get(s, 0)}' for s in INSTRUMENTS]
                logger.info(
                    f'LiveBarFeed: 30-min rate window ({elapsed}s) — '
                    f'bars per symbol: {", ".join(parts)}  '
                    f'sym_map: {self._sym_map}'
                )
                self._window_start_time  = time.time()
                self._window_bars_by_sym = {}

            if rtype == 'SymbolMappingMsg':
                # stype_in_symbol = 'NQ.c.0' — map back to apex symbol
                in_sym = (getattr(record, 'stype_in_symbol', None) or
                          getattr(record, 'input_symbol', None) or '')
                matched = next(
                    (apex for apex, db_s in INSTRUMENTS.items() if db_s == in_sym),
                    None
                )
                if matched:
                    self._sym_map[record.instrument_id] = matched
                    logger.info(
                        f'LiveBarFeed: SymbolMappingMsg  '
                        f'instrument_id={record.instrument_id} → {matched} ({in_sym})  '
                        f'sym_map now: {self._sym_map}'
                    )
                else:
                    logger.warning(
                        f'LiveBarFeed: SymbolMappingMsg unrecognised '
                        f'stype_in_symbol={in_sym!r} — '
                        f'known: {list(INSTRUMENTS.values())}'
                    )

            elif rtype == 'OHLCVMsg':
                apex_sym = self._sym_map.get(record.instrument_id)
                if apex_sym:
                    self._store_bar(apex_sym, record)
                else:
                    logger.warning(
                        f'LiveBarFeed: OHLCVMsg instrument_id={record.instrument_id} '
                        f'not in sym_map {self._sym_map} — bar skipped '
                        f'(expected SymbolMappingMsg first; subscribed={subscribed})'
                    )

            elif rtype == 'ErrorMsg':
                err = getattr(record, 'err', str(record))
                logger.error(f'LiveBarFeed: Databento server error: {err}')
                break   # trigger reconnect

            # SystemMsg = keepalive — ignore

        with self._lock:
            self._live_conn = None

    def _store_bar(self, symbol: str, record):
        """Persist a live ohlcv-1m bar to DB immediately on bar close."""
        try:
            # ts_event: nanoseconds since Unix epoch
            ts_unix = record.ts_event // 1_000_000_000
            o = float(record.pretty_open)
            h = float(record.pretty_high)
            l = float(record.pretty_low)
            c = float(record.pretty_close)
            v = float(record.volume)

            if c <= 0 or ts_unix <= 0:
                logger.warning(
                    f'LiveBarFeed: invalid bar {symbol} ts={ts_unix} close={c} — skipped'
                )
                return

            conn = _db_connect()
            _db_upsert(conn, symbol, '1min', ts_unix, o, h, l, c, v)
            conn.commit()
            conn.close()

            self._bar_count                      += 1
            self._last_bar_time                   = time.time()
            self._last_bar_time_by_sym[symbol]    = self._last_bar_time
            self._bar_count_by_sym[symbol]        = self._bar_count_by_sym.get(symbol, 0) + 1
            self._window_bars_by_sym[symbol]      = self._window_bars_by_sym.get(symbol, 0) + 1
            ts_str = datetime.fromtimestamp(ts_unix, tz=timezone.utc).strftime('%H:%M')
            logger.info(
                f'LiveBarFeed: {symbol} 1min {ts_str}UTC  '
                f'O={o:.2f} H={h:.2f} L={l:.2f} C={c:.2f}  vol={v:.0f}  '
                f'(session: {symbol}={self._bar_count_by_sym[symbol]} total={self._bar_count})'
            )
        except Exception as e:
            logger.error(f'LiveBarFeed: _store_bar error {symbol}: {e}', exc_info=True)


# Module-level singleton — created once, managed by start/stop functions
_live_feed: 'LiveBarFeed | None' = None


def start_live_feed() -> bool:
    """
    Start the live bar feed background thread.
    Returns True if newly started, False if already running.
    """
    global _live_feed
    if _live_feed is None:
        _live_feed = LiveBarFeed()
    if not _live_feed.is_running:
        _live_feed.start()
        return True
    return False


def stop_live_feed():
    """Stop the live bar feed background thread."""
    global _live_feed
    if _live_feed and _live_feed.is_running:
        _live_feed.stop()


def is_live_feed_running() -> bool:
    """True if the live bar feed thread is alive."""
    return _live_feed is not None and _live_feed.is_running


def get_live_feed_stats() -> dict:
    """Return live feed status for monitoring/dashboard."""
    if _live_feed is None:
        return {
            'running': False, 'bar_count': 0, 'seconds_since_last_bar': None,
            'nq_feed_seconds_ago': None, 'es_feed_seconds_ago': None, 'gc_feed_seconds_ago': None,
            'nq_feed_status': 'red', 'es_feed_status': 'red', 'gc_feed_status': 'red',
        }
    secs = _live_feed.seconds_since_last_bar()

    def _sym_secs(sym):
        s = _live_feed.seconds_since_last_bar_for(sym)
        return None if s == float('inf') else round(s)

    def _sym_status(s):
        if s is None:    return 'red'
        if s < 60:       return 'green'
        if s < 300:      return 'amber'
        return 'red'

    nq_s = _sym_secs('NQ')
    es_s = _sym_secs('ES')
    gc_s = _sym_secs('GC')

    return {
        'running':                _live_feed.is_running,
        'bar_count':              _live_feed.bar_count,
        'last_bar_time':          _live_feed.last_bar_time or None,
        'seconds_since_last_bar': None if secs == float('inf') else round(secs),
        'nq_feed_seconds_ago':    nq_s,
        'es_feed_seconds_ago':    es_s,
        'gc_feed_seconds_ago':    gc_s,
        'nq_feed_status':         _sym_status(nq_s),
        'es_feed_status':         _sym_status(es_s),
        'gc_feed_status':         _sym_status(gc_s),
        # Short aliases used by dashboard health strip
        'NQ_secs':                nq_s,
        'ES_secs':                es_s,
        'GC_secs':                gc_s,
        # Per-symbol session bar counts
        'bar_count_by_sym':       dict(_live_feed._bar_count_by_sym),
        'sym_map':                dict(_live_feed._sym_map),
        'records_seen':           _live_feed._records_seen,
    }


def restart_live_feed() -> bool:
    """Force-stop and restart the live feed. Returns True if restarted."""
    global _live_feed
    if _live_feed and _live_feed.is_running:
        _live_feed.stop()
    _live_feed = LiveBarFeed()
    _live_feed.start()
    return True


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
    """Store bars in DB. Returns number of new bars inserted.

    Uses a single transaction — no per-bar exception swallowing.
    Per-bar exception catch in psycopg2 leaves the transaction in an ABORTED
    state; subsequent inserts AND commit() all fail with InFailedSqlTransaction.
    """
    if not bars:
        return 0
    conn = _db_connect()
    try:
        count_before = conn.execute(
            'SELECT COUNT(*) FROM ohlcv WHERE symbol=? AND timeframe=?',
            (symbol, timeframe)
        ).fetchone()[0]
        for bar in bars:
            ts, o, h, l, close, vol = bar
            # upsert_ohlcv intentionally overwrites stale bars with fresh Databento data.
            # server.py:store_ohlcv uses INSERT OR IGNORE — two deliberate strategies.
            _db_upsert(conn, symbol, timeframe, ts, o, h, l, close, vol)
        conn.commit()
        count_after = conn.execute(
            'SELECT COUNT(*) FROM ohlcv WHERE symbol=? AND timeframe=?',
            (symbol, timeframe)
        ).fetchone()[0]
        return count_after - count_before
    except Exception as e:
        logger.error(f'store_bars FAILED {symbol} {timeframe}: {e}', exc_info=True)
        try:
            conn.rollback()
        except Exception:
            pass
        return 0
    finally:
        conn.close()


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

    # Try progressively earlier end times. Databento's archive has a variable
    # ingestion lag — if end = now-2min returns empty (data not indexed yet),
    # fall back to now-10min, then now-15min. Each attempt logs the end used
    # so Railway logs show exactly which window succeeded or failed.
    now_utc       = datetime.now(timezone.utc)
    end_offsets   = [2, 10, 15]   # minutes before now to try as end

    for offset_idx, offset_min in enumerate(end_offsets):
        end   = now_utc - timedelta(minutes=offset_min)
        start = end - timedelta(hours=lookback_hours)
        end_label = end.strftime('%H:%M:%S') + 'UTC'
        is_last = offset_idx == len(end_offsets) - 1

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
                if not is_last:
                    logger.warning(
                        f'DataFeed {symbol} {timeframe}: empty response '
                        f'(end={end_label}) — retrying with earlier end'
                    )
                    continue
                logger.warning(
                    f'DataFeed {symbol} {timeframe}: empty after all end '
                    f'candidates {end_offsets}min — Databento may be behind'
                )
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

            if offset_min > 2:
                logger.info(
                    f'DataFeed {symbol} {timeframe}: fetched {len(result)} bars '
                    f'using end={end_label} (offset {offset_min}min)'
                )
            return result

        except Exception as e:
            if not is_last:
                logger.warning(
                    f'DataFeed {symbol} {timeframe}: fetch error '
                    f'(end={end_label}): {e} — retrying with earlier end'
                )
                time.sleep(1)
            else:
                logger.warning(
                    f'DataFeed {symbol} {timeframe}: failed after all attempts: {e}'
                )
                return []

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
        err_str = str(e)
        if '422' in err_str or 'data_end_after_available_end' in err_str:
            logger.warning(f'fetch_bars_range {symbol} {timeframe}: Databento 422 — requested end beyond available data: {e}')
        else:
            logger.error(f'fetch_bars_range error {symbol} {timeframe}: {e}')
        return []


def build_intraday_from_1min(symbol: str) -> dict:
    """
    Build 5min and 15min bars by resampling live 1min bars already in DB.

    Replaces historical API fetches for 5min/15min entirely. Since LiveBarFeed
    stores 1min bars in real-time, derived bars are always current with zero delay.

    Lookbacks:
      5min  — last 500 1min bars (~8h of intraday data, ~40 5min bars)
      15min — last 1000 1min bars (~16h, ~65 15min bars)
    """
    conn = _db_connect()
    df = _db_read_sql(
        'SELECT ts, open, high, low, close, volume FROM ohlcv '
        'WHERE symbol=? AND timeframe=? ORDER BY ts DESC LIMIT 1000',
        conn, params=(symbol, '1min')
    )
    conn.close()

    if df.empty:
        logger.warning(f'build_intraday_from_1min: no 1min bars for {symbol} — skipping')
        return {}

    # Sort ascending for correct resample direction
    df = df.sort_values('ts')
    df['dt'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    df.set_index('dt', inplace=True)

    logger.info(f'build_intraday_from_1min: {symbol} — {len(df)} 1min bars loaded')

    results = {}
    # (output_tf, resample_rule, lookback_bars)
    for tf, rule, lookback in [('5min', '5min', 500), ('15min', '15min', 1000)]:
        try:
            src = df.tail(lookback)
            agg = src.resample(rule).agg({
                'open':   'first',
                'high':   'max',
                'low':    'min',
                'close':  'last',
                'volume': 'sum',
            }).dropna()
            # .timestamp() is resolution-independent — works whether pandas stores
            # datetime64[ns] or datetime64[s] (pandas 2.1+ changed unit='s' to [s]).
            # .asi8 // 1e9 broke in pandas 2.1+: asi8 returns seconds (not ns) for
            # [s]-resolution indexes, so integer division always yields 1.
            agg['ts'] = agg.index.map(lambda x: int(x.timestamp()))

            bars = [
                (int(row['ts']), round(float(row['open']), 2),
                 round(float(row['high']), 2), round(float(row['low']), 2),
                 round(float(row['close']), 2), float(row['volume']))
                for _, row in agg.iterrows()
            ]

            # Only write bars that are new or the most-recent bar (which may
            # still be forming — re-upsert to keep its OHLCV current).
            # Avoids upserting hundreds of already-stored historical bars on
            # every cycle, which made store_bars() always return 0 new.
            conn_ts = _db_connect()
            last_row = conn_ts.execute(
                'SELECT MAX(ts) FROM ohlcv WHERE symbol=? AND timeframe=?',
                (symbol, tf)
            ).fetchone()
            conn_ts.close()
            last_stored_ts = int(last_row[0]) if last_row and last_row[0] else 0

            # ts >= last_stored_ts: re-upsert the last bar (forming) + new bars
            bars_to_write = [b for b in bars if b[0] >= last_stored_ts]
            new_count = sum(1 for b in bars_to_write if b[0] > last_stored_ts)

            if not bars_to_write:
                logger.debug(
                    f'build_intraday_from_1min: {symbol} {tf} — up to date'
                )
                results[tf] = 0
                continue

            logger.info(
                f'build_intraday_from_1min: {symbol} {tf} — '
                f'writing {len(bars_to_write)} bars ({new_count} new) '
                f'last_stored_ts={last_stored_ts}'
            )

            # Use a clean transaction: no per-bar exception swallowing.
            # Per-bar exception catch in psycopg2 leaves the transaction in an
            # ABORTED state, causing all subsequent inserts AND the final
            # commit() to fail with InFailedSqlTransaction. The result: zero
            # bars written despite no visible error.
            conn_w = _db_connect()
            try:
                for b in bars_to_write:
                    _db_upsert(conn_w, symbol, tf, b[0], b[1], b[2], b[3], b[4], b[5])
                conn_w.commit()
            except Exception as write_err:
                logger.error(
                    f'build_intraday_from_1min: write FAILED {symbol} {tf}: {write_err}',
                    exc_info=True
                )
                try:
                    conn_w.rollback()
                except Exception:
                    pass
                results[tf] = 0
                continue
            finally:
                conn_w.close()

            results[tf] = new_count
            logger.info(
                f'build_intraday_from_1min: {symbol} {tf} — '
                f'committed {len(bars_to_write)} bars ({new_count} new)'
            )
        except Exception as e:
            logger.error(
                f'build_intraday_from_1min error {symbol} {tf}: {e}', exc_info=True
            )
            results[tf] = 0

    return results


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
            # .timestamp() is resolution-independent — works whether pandas stores
            # datetime64[ns] or datetime64[s] (pandas 2.1+ changed unit='s' to [s]).
            # .asi8 // 1e9 broke in pandas 2.1+: asi8 returns seconds for [s]-resolution
            # indexes, so integer division always yielded 1 instead of the real Unix ts.
            agg['ts'] = agg.index.map(lambda x: int(x.timestamp()))

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
    Refresh all instruments. Called every 5 minutes from scheduler.

    Data sources per timeframe:
      1min  — LiveBarFeed streams in real-time (session hours).
              Historical API used as fallback when live feed is not running.
      5min  — Resampled from 1min DB rows (build_intraday_from_1min).
              No historical API call — always derived from live 1min data.
      15min — Same as 5min.
      1h/4h — Resampled from 5min DB rows (build_htf_from_5min), every 30 min.
      1day  — Historical API, when include_daily=True.
    """
    results = {}

    for symbol in INSTRUMENTS:
        sym_results = {}

        # 1min: historical backfill only when live feed is not streaming
        if not is_live_feed_running():
            try:
                bars = fetch_recent_bars(symbol, '1min', lookback_hours=4)
                sym_results['1min'] = store_bars(symbol, '1min', bars)
                if sym_results['1min'] > 0:
                    logger.info(f'DataFeed: {symbol} 1min +{sym_results["1min"]} bars (historical backfill)')
            except Exception as e:
                logger.warning(f'DataFeed: {symbol} 1min historical fetch failed — {e}')
                sym_results['1min'] = 0

        # 5min + 15min: always from 1min resample — zero API delay
        intraday = build_intraday_from_1min(symbol)
        sym_results.update(intraday)

        # 1h + 4h: from 5min resample, every 30 min
        if include_htf:
            htf = build_htf_from_5min(symbol)
            sym_results.update(htf)

        # 1day: historical API, once per day
        if include_daily:
            try:
                bars = fetch_recent_bars(symbol, '1day', lookback_hours=48)
                sym_results['1day'] = store_bars(symbol, '1day', bars)
            except Exception as e:
                logger.warning(f'DataFeed: {symbol} 1day fetch failed — {e}')
                sym_results['1day'] = 0

        results[symbol] = sym_results

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
