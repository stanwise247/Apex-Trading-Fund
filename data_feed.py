"""
APEX Live Data Feed — data_feed.py
=====================================
Polygon.io ("Massive" — same company, same API, same key; api.polygon.io
still works) REST feed, replacing the prior Databento Live API integration.

  SESSION (13:00-20:00 UTC weekdays):
    PolygonPoller — background thread polling Polygon/Massive futures
    aggregates every 60s for 1min/5min/15min/1hour/4hour bars, written
    directly to ohlcv. Off-session it sleeps and checks periodically
    rather than busy-polling.

  BACKFILL / ON-DEMAND REFRESH:
    Same REST aggregates endpoint with a wider lookback window. Used on
    startup and by the manual refresh endpoints in server.py.

Instruments: MNQ, ES, GC (front-month single-leg contracts).

IMPORTANT — contract rollover: front-month tickers below are hardcoded and
must be updated by hand when each contract rolls (no auto-roll logic was
requested). GC settles soonest (2026-08-27) — check it first.

Endpoint note: Polygon's *futures* API is a dedicated namespace
(/futures/v1/...), completely separate from the stocks-style
/v2/aggs/ticker/... endpoints. Futures tickers here have NO "F:" prefix
(e.g. "ESU6", not "F:ESU6") — confirmed empirically against the live API,
which 404s on a prefixed ticker. Verify against
https://massive.com/docs/rest/futures/overview before assuming otherwise.
"""

import logging
import json
import os
import time
import threading
import pandas as pd
import requests
from datetime import datetime, timezone

from db import connect as _db_connect, IS_POSTGRES, read_sql as _db_read_sql, upsert_ohlcv as _db_upsert

logger = logging.getLogger('APEX.DataFeed')

# Front-month contract tickers, verified empirically (highest session volume
# among candidate months) against the real Polygon/Massive API on 2026-07-26.
INSTRUMENTS = {
    'MNQ': 'MNQU6',  # Sep 2026, settles 2026-09-18
    'ES':  'ESU6',   # Sep 2026, settles 2026-09-18
    'GC':  'GCQ6',   # Aug 2026, settles 2026-08-27 — rolls soonest of the three
}

POLYGON_BASE = 'https://api.polygon.io'

# Polygon/Massive futures aggregates accept these resolution strings
# directly — confirmed 1:1 with our internal ohlcv.timeframe naming, no
# translation needed (unlike the old Databento path, which needed manual
# 1min-bar resampling for 1hour/4hour due to an SDK pricing bug).
TIMEFRAMES = ('1min', '5min', '15min', '1hour', '4hour')

SESSION_START_UTC = 13
SESSION_END_UTC   = 20


def get_api_key() -> str:
    key = os.environ.get('POLYGON_API_KEY', '')
    if not key:
        try:
            with open('config.json') as f:
                cfg = json.load(f)
            key = cfg.get('POLYGON_API_KEY', cfg.get('polygon_api_key', ''))
        except Exception:
            pass
    return key


def _fetch_aggs(ticker: str, resolution: str, limit: int = 500,
                 start_ns: int = None, end_ns: int = None, sort: str = 'window_start.asc') -> list:
    """
    Fetch futures aggregate bars from Polygon/Massive.
    Returns list of (ts_unix, open, high, low, close, volume) tuples, ALWAYS
    oldest-first regardless of which `sort` was used to fetch them (see below).
    Never raises — logs and returns [] on any error, matching the rest of
    this module's fault-isolation style (one bad request never blocks others).

    IMPORTANT: with no start_ns/end_ns bounds, sort='window_start.asc' (the
    default, correct for the bounded historical-range callers below) returns
    the OLDEST `limit` bars in Polygon's entire history for that ticker, not
    the most recent ones — this was the root cause of the live poller
    silently freezing on ancient data despite "succeeding" on every call
    (non-empty response, no exception, health counters kept incrementing).
    Callers that want "the latest N bars" with no explicit date range MUST
    pass sort='window_start.desc' — see PolygonPoller._poll_once().
    """
    key = get_api_key()
    if not key:
        logger.error('No Polygon API key (POLYGON_API_KEY)')
        return []
    params = {'apiKey': key, 'resolution': resolution, 'limit': limit, 'sort': sort}
    if start_ns is not None:
        params['window_start.gte'] = start_ns
    if end_ns is not None:
        params['window_start.lte'] = end_ns
    try:
        r = requests.get(f'{POLYGON_BASE}/futures/v1/aggs/{ticker}', params=params, timeout=15)
        if r.status_code != 200:
            logger.warning(f'Polygon aggs {ticker} {resolution}: HTTP {r.status_code} — {r.text[:200]}')
            return []
        results = r.json().get('results', [])
        bars = []
        for row in results:
            ts_unix = int(row['window_start'] // 1_000_000_000)
            bars.append((
                ts_unix,
                round(float(row['open']), 2), round(float(row['high']), 2),
                round(float(row['low']), 2), round(float(row['close']), 2),
                float(row.get('volume', 0)),
            ))
        if sort == 'window_start.desc':
            bars.reverse()   # normalise to oldest-first regardless of fetch order
        return bars
    except Exception as e:
        logger.warning(f'Polygon aggs {ticker} {resolution} error: {e}')
        return []


def get_latest_price(symbol: str):
    """Current price via the Polygon/Massive futures snapshot endpoint. None on any failure."""
    ticker = INSTRUMENTS.get(symbol)
    key = get_api_key()
    if not ticker or not key:
        return None
    try:
        r = requests.get(f'{POLYGON_BASE}/futures/v1/snapshot',
                          params={'apiKey': key, 'ticker': ticker}, timeout=15)
        if r.status_code != 200:
            return None
        results = r.json().get('results', [])
        if not results:
            return None
        res   = results[0]
        price = (res.get('last_trade') or {}).get('price')
        if price is None:
            price = (res.get('last_minute') or {}).get('close')
        if price is None:
            price = (res.get('session') or {}).get('close')
        return float(price) if price is not None else None
    except Exception as e:
        logger.warning(f'Polygon snapshot {symbol} error: {e}')
        return None


# ─────────────────────────────────────────────────────────────
#  LIVE POLLER (Polygon/Massive REST — replaces Databento Live streaming)
# ─────────────────────────────────────────────────────────────

class PolygonPoller:
    """
    Background thread that polls Polygon/Massive futures aggregates every
    60s during session hours (13:00-20:00 UTC weekdays) for all 5
    timeframes, writing fresh bars directly to ohlcv. Off-session it sleeps
    in longer increments rather than busy-polling a closed market.
    """

    _POLL_INTERVAL_S    = 60
    _OFFSESSION_CHECK_S = 300

    def __init__(self):
        self._thread              = None
        self._stop_evt             = threading.Event()
        self._lock                 = threading.Lock()
        self._bar_count            = 0
        self._last_bar_time        = 0.0
        self._last_bar_time_by_sym = {}
        self._bar_count_by_sym     = {}
        self._poll_cycles          = 0

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_actively_polling(self) -> bool:
        """
        Thread alive AND currently within the 13:00-20:00 UTC weekday poll
        window. Distinct from is_running: server.py's scheduler starts/stops
        this thread on its own broader 07:00-21:00 UTC/any-day gate, so the
        thread can be alive but dormant (weekends, or 07:00-13:00 UTC
        weekdays) — callers deciding whether 1min data is actively being
        covered right now (e.g. refresh_all()'s historical-fallback branch)
        need this, not just thread-alive.
        """
        return self.is_running and self._in_session()

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
            self._thread = threading.Thread(target=self._run, name='PolygonPoller', daemon=True)
            self._thread.start()
        logger.info('PolygonPoller: started — polling Polygon/Massive for MNQ/ES/GC')

    def stop(self):
        self._stop_evt.set()
        logger.info('PolygonPoller: stopped')

    @staticmethod
    def _in_session() -> bool:
        now = datetime.now(timezone.utc)
        return now.weekday() < 5 and SESSION_START_UTC <= now.hour < SESSION_END_UTC

    def _run(self):
        while not self._stop_evt.is_set():
            try:
                if self._in_session():
                    self._poll_once()
                    self._stop_evt.wait(self._POLL_INTERVAL_S)
                else:
                    self._stop_evt.wait(self._OFFSESSION_CHECK_S)
            except Exception as e:
                logger.error(f'PolygonPoller: cycle error: {e} — retrying in {self._POLL_INTERVAL_S}s')
                self._stop_evt.wait(self._POLL_INTERVAL_S)

    def _poll_once(self):
        self._poll_cycles += 1
        for symbol, ticker in INSTRUMENTS.items():
            for tf in TIMEFRAMES:
                try:
                    # sort='window_start.desc' is required here — with no
                    # start_ns/end_ns bounds, the default ascending sort
                    # returns the OLDEST bars in Polygon's entire history for
                    # the ticker, not the latest (see _fetch_aggs docstring).
                    # This was the actual bug: every poll "succeeded" (non-
                    # empty response, no exception) but silently re-fetched
                    # the same year-old bars every cycle, so ohlcv's max(ts)
                    # never advanced past whatever the last real write was.
                    bars = _fetch_aggs(ticker, tf, limit=5, sort='window_start.desc')
                    if not bars:
                        logger.warning(f'PolygonPoller: {symbol} {tf} — empty response from Polygon')
                        continue

                    latest_ts = bars[-1][0]
                    staleness_s = time.time() - latest_ts
                    if tf == '1min' and staleness_s > 1800:
                        # >30 min stale on the finest timeframe during an
                        # active poll cycle means something is wrong upstream
                        # (Polygon delay, wrong ticker, etc) — surfaced at
                        # WARNING (not DEBUG) so it actually reaches Railway
                        # logs instead of looking like a normal quiet cycle.
                        logger.warning(
                            f'PolygonPoller: {symbol} {tf} latest bar is {staleness_s/60:.0f}min stale '
                            f'(ts={latest_ts}) — Polygon may be delayed or ticker may be wrong'
                        )

                    inserted = store_bars(symbol, tf, bars)
                    if tf == '1min' and inserted > 0:
                        # Only count real, newly-written bars toward the
                        # health-check freshness stats — previously this
                        # incremented whenever the HTTP call merely returned
                        # *something*, which stayed "green" even while the
                        # sort bug above caused zero real writes for hours.
                        self._bar_count += 1
                        self._last_bar_time = time.time()
                        self._last_bar_time_by_sym[symbol] = self._last_bar_time
                        self._bar_count_by_sym[symbol] = self._bar_count_by_sym.get(symbol, 0) + 1
                    if inserted > 0:
                        logger.info(f'PolygonPoller: {symbol} {tf} +{inserted} new bars (latest ts={latest_ts})')
                except Exception as e:
                    logger.warning(f'PolygonPoller: {symbol} {tf} poll error: {e}')


# Module-level singleton — created once, managed by start/stop functions.
# Function names/signatures below are unchanged from the Databento-era API
# so server.py's scheduler and /api/apex/health need no changes at all.
_live_feed: 'PolygonPoller | None' = None


def start_live_feed() -> bool:
    """Start the live poller thread. Returns True if newly started, False if already running."""
    global _live_feed
    if _live_feed is None:
        _live_feed = PolygonPoller()
    if not _live_feed.is_running:
        _live_feed.start()
        return True
    return False


def stop_live_feed():
    """Stop the live poller thread."""
    global _live_feed
    if _live_feed and _live_feed.is_running:
        _live_feed.stop()


def is_live_feed_running() -> bool:
    """True if the live poller thread is alive."""
    return _live_feed is not None and _live_feed.is_running


def get_live_feed_stats() -> dict:
    """Return live feed status for monitoring/dashboard. Same key shape as the Databento-era version."""
    if _live_feed is None:
        return {
            'running': False, 'bar_count': 0, 'seconds_since_last_bar': None,
            'nq_feed_seconds_ago': None, 'es_feed_seconds_ago': None,
            'gc_feed_seconds_ago': None, 'mnq_feed_seconds_ago': None,
            'nq_feed_status': 'red', 'es_feed_status': 'red',
            'gc_feed_status': 'red', 'mnq_feed_status': 'red',
        }
    secs = _live_feed.seconds_since_last_bar()

    def _sym_secs(sym):
        s = _live_feed.seconds_since_last_bar_for(sym)
        return None if s == float('inf') else round(s)

    def _sym_status(s):
        if s is None: return 'red'
        if s < 90:    return 'green'   # poll interval is 60s — allow one miss before amber
        if s < 300:   return 'amber'
        return 'red'

    # NQ was already excluded from the live feed before this migration
    # (MNQ replaced it) — kept as a key for dashboard shape compatibility.
    nq_s  = None
    es_s  = _sym_secs('ES')
    gc_s  = _sym_secs('GC')
    mnq_s = _sym_secs('MNQ')

    return {
        'running':                _live_feed.is_running,
        'bar_count':              _live_feed._bar_count,
        'last_bar_time':          _live_feed._last_bar_time or None,
        'seconds_since_last_bar': None if secs == float('inf') else round(secs),
        'nq_feed_seconds_ago':    nq_s,
        'es_feed_seconds_ago':    es_s,
        'gc_feed_seconds_ago':    gc_s,
        'mnq_feed_seconds_ago':   mnq_s,
        'nq_feed_status':         _sym_status(nq_s),
        'es_feed_status':         _sym_status(es_s),
        'gc_feed_status':         _sym_status(gc_s),
        'mnq_feed_status':        _sym_status(mnq_s),
        'NQ_secs':                nq_s,
        'ES_secs':                es_s,
        'GC_secs':                gc_s,
        'MNQ_secs':               mnq_s,
        'bar_count_by_sym':       dict(_live_feed._bar_count_by_sym),
        'sym_map':                dict(INSTRUMENTS),
        'records_seen':           _live_feed._poll_cycles,
    }


def restart_live_feed() -> bool:
    """Force-stop and restart the live poller. Returns True if restarted."""
    global _live_feed
    if _live_feed and _live_feed.is_running:
        _live_feed.stop()
    _live_feed = PolygonPoller()
    _live_feed.start()
    return True


# ─────────────────────────────────────────────────────────────
#  DB WRITES / HISTORICAL REFRESH
#  (unchanged from the Databento-era version — pure DB + pandas, no
#  feed-provider dependency)
# ─────────────────────────────────────────────────────────────

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


def fetch_recent_bars(symbol: str, timeframe: str, lookback_hours: int = 48) -> list:
    """
    Fetch recent bars from Polygon/Massive for a lookback window ending now.
    Same signature/return shape as the Databento-era version.
    """
    ticker = INSTRUMENTS.get(symbol)
    if not ticker or timeframe not in TIMEFRAMES:
        return []
    now_ns   = int(time.time() * 1_000_000_000)
    start_ns = now_ns - int(lookback_hours * 3600 * 1_000_000_000)
    return _fetch_aggs(ticker, timeframe, limit=5000, start_ns=start_ns, end_ns=now_ns)


def fetch_bars_range(symbol: str, timeframe: str,
                     start: datetime, end: datetime) -> list:
    """Fetch bars for an explicit date range. Used for chunked historical backfill."""
    ticker = INSTRUMENTS.get(symbol)
    if not ticker or timeframe not in TIMEFRAMES:
        return []
    start_ns = int(start.timestamp() * 1_000_000_000)
    end_ns   = int(end.timestamp() * 1_000_000_000)
    return _fetch_aggs(ticker, timeframe, limit=5000, start_ns=start_ns, end_ns=end_ns)


def build_intraday_from_1min(symbol: str) -> dict:
    """
    Build 5min and 15min bars by resampling live 1min bars already in DB.

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
            agg['ts'] = agg.index.map(lambda x: int(x.timestamp()))

            bars = [
                (int(row['ts']), round(float(row['open']), 2),
                 round(float(row['high']), 2), round(float(row['low']), 2),
                 round(float(row['close']), 2), float(row['volume']))
                for _, row in agg.iterrows()
            ]

            conn_ts = _db_connect()
            last_row = conn_ts.execute(
                'SELECT MAX(ts) FROM ohlcv WHERE symbol=? AND timeframe=?',
                (symbol, tf)
            ).fetchone()
            conn_ts.close()
            last_stored_ts = int(last_row[0]) if last_row and last_row[0] else 0

            bars_to_write = [b for b in bars if b[0] >= last_stored_ts]
            new_count = sum(1 for b in bars_to_write if b[0] > last_stored_ts)

            if not bars_to_write:
                logger.debug(f'build_intraday_from_1min: {symbol} {tf} — up to date')
                results[tf] = 0
                continue

            logger.info(
                f'build_intraday_from_1min: {symbol} {tf} — '
                f'writing {len(bars_to_write)} bars ({new_count} new) '
                f'last_stored_ts={last_stored_ts}'
            )

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
            logger.error(f'build_intraday_from_1min error {symbol} {tf}: {e}', exc_info=True)
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
            agg['ts'] = agg.index.map(lambda x: int(x.timestamp()))

            bars = [
                (int(row['ts']), round(float(row['open']), 2),
                 round(float(row['high']), 2), round(float(row['low']), 2),
                 round(float(row['close']), 2), float(row['volume']))
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
      1min  — PolygonPoller streams every 60s (session hours).
              Historical REST fetch used as fallback when poller isn't running.
      5min  — Resampled from 1min DB rows (build_intraday_from_1min).
      15min — Same as 5min.
      1h/4h — Resampled from 5min DB rows (build_htf_from_5min), every 30 min.
      1day  — Not fetched here (Polygon futures aggs above cap out around
              4hour for this feed; daily bars are a pre-existing separate
              concern, unchanged by this migration).
    """
    results = {}

    # Distinct from the public is_live_feed_running() (thread-alive, used by
    # server.py's own start/stop/watchdog gate on its broader 07:00-21:00 UTC
    # any-day window) — this checks whether the poller is actually inside its
    # own 13:00-20:00 UTC weekday window right now, so 1min historical
    # fallback correctly kicks in during the gap hours where the thread is
    # alive but dormant (e.g. 07:00-13:00 UTC weekdays, or weekends).
    _actively_polling = _live_feed is not None and _live_feed.is_actively_polling

    for symbol in INSTRUMENTS:
        sym_results = {}

        if not _actively_polling:
            try:
                bars = fetch_recent_bars(symbol, '1min', lookback_hours=4)
                sym_results['1min'] = store_bars(symbol, '1min', bars)
                if sym_results['1min'] > 0:
                    logger.info(f'DataFeed: {symbol} 1min +{sym_results["1min"]} bars (historical backfill)')
            except Exception as e:
                logger.warning(f'DataFeed: {symbol} 1min historical fetch failed — {e}')
                sym_results['1min'] = 0

        intraday = build_intraday_from_1min(symbol)
        sym_results.update(intraday)

        if include_htf:
            htf = build_htf_from_5min(symbol)
            sym_results.update(htf)

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
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')

    print('Testing APEX data feed (Polygon/Massive)...\n')
    print('POLYGON_API_KEY configured:', bool(get_api_key()))

    print('\n=== Current prices (live snapshot) ===')
    for sym in ('ES', 'MNQ', 'GC'):
        price = get_latest_price(sym)
        print(f'  {sym} ({INSTRUMENTS[sym]}): {price}')

    for sym in ('ES', 'MNQ', 'GC'):
        print(f'\n{sym} 5min (last 2 hours):')
        bars = fetch_recent_bars(sym, '5min', lookback_hours=2)
        if bars:
            last = pd.to_datetime(bars[-1][0], unit='s', utc=True)
            print(f'  Bars:       {len(bars)}')
            print(f'  Last bar:   {last}')
            print(f'  Last close: {bars[-1][4]:.2f}')
        else:
            print(f'  No bars returned')

    print('\nFull refresh (5min + 15min + HTF)...')
    results = refresh_all(include_htf=True)
    for sym, tfs in results.items():
        for tf, count in tfs.items():
            if count > 0:
                print(f'  {sym} {tf}: +{count} new bars')

    print('\nLatest bars in DB:')
    conn = _db_connect()
    for sym in ('ES', 'MNQ', 'GC'):
        for tf in ('5min', '15min', '1hour', '4hour'):
            df = _db_read_sql(
                'SELECT MAX(ts) FROM ohlcv WHERE symbol=? AND timeframe=?',
                conn, params=(sym, tf)
            )
            ts = df.iloc[0, 0]
            if ts:
                dt = pd.to_datetime(ts, unit='s', utc=True)
                print(f'  {sym} {tf}: {dt}')
