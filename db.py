"""
db.py — Database abstraction layer
====================================
Routes to PostgreSQL (Railway) when DATABASE_URL is set, else SQLite (local dev).

The cursor wrapper auto-translates SQLite dialect → PostgreSQL:
  - ? placeholders          → %s
  - INSERT OR IGNORE        → ON CONFLICT DO NOTHING
  - INTEGER PRIMARY KEY
    AUTOINCREMENT           → SERIAL PRIMARY KEY
  - PRAGMA ...              → silent no-op

Two INSERT patterns that need explicit handling:
  - insert_returning_id()   replaces cursor.lastrowid
  - upsert_ohlcv()          replaces INSERT OR REPLACE INTO ohlcv
  - kv_upsert()             replaces INSERT OR REPLACE INTO paper_account

Other helpers:
  - read_sql()              pandas read_sql_query compatible with both backends
  - init_schema()           creates all tables at startup (PostgreSQL only;
                            SQLite tables are created by existing init functions)
"""

import os
import re
import logging
import warnings
warnings.filterwarnings('ignore', message='.*SQLAlchemy.*')

logger = logging.getLogger('APEX.DB')

DATABASE_URL = os.environ.get('DATABASE_URL', '').strip()
IS_POSTGRES  = bool(DATABASE_URL)

# ─── SQL translation ─────────────────────────────────────────

_PH        = re.compile(r'\?')
_OR_IGNORE = re.compile(r'(?i)\bINSERT\s+OR\s+IGNORE\s+INTO\b')
_AUTOINC   = re.compile(r'(?i)\bINTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT\b')
_PRAGMA    = re.compile(r'(?i)^\s*PRAGMA\b')
_CHANGES   = re.compile(r'(?i)\bSELECT\s+changes\(\)')


def _pg(sql: str):
    """Translate SQLite SQL → PostgreSQL. Returns None for PRAGMA (skip)."""
    if _PRAGMA.match(sql):
        return None
    sql = _PH.sub('%s', sql)
    sql = _AUTOINC.sub('SERIAL PRIMARY KEY', sql)
    sql = _CHANGES.sub('SELECT 0', sql)
    if _OR_IGNORE.search(sql):
        sql = _OR_IGNORE.sub('INSERT INTO', sql)
        sql = sql.rstrip().rstrip(';') + ' ON CONFLICT DO NOTHING'
    return sql


# ─── PostgreSQL cursor wrapper ───────────────────────────────

class _PgCursor:
    def __init__(self, cur):
        self._c = cur

    @property
    def description(self):
        return self._c.description

    @property
    def rowcount(self):
        return self._c.rowcount

    @property
    def lastrowid(self):
        # Use insert_returning_id() for INSERT id retrieval
        return None

    def execute(self, sql, params=None):
        pg_sql = _pg(sql)
        if pg_sql is None:
            return self   # PRAGMA — no-op
        self._c.execute(pg_sql, params if params is not None else ())
        return self

    def executemany(self, sql, params_list):
        pg_sql = _pg(sql)
        if pg_sql is None:
            return self
        self._c.executemany(pg_sql, params_list)
        return self

    def fetchone(self):
        return self._c.fetchone()

    def fetchall(self):
        return self._c.fetchall()

    def __iter__(self):
        return iter(self._c)


# ─── PostgreSQL connection wrapper ───────────────────────────

class _PgConn:
    def __init__(self, raw):
        self._conn = raw   # raw psycopg2 connection

    def cursor(self) -> _PgCursor:
        return _PgCursor(self._conn.cursor())

    def execute(self, sql, params=None) -> _PgCursor:
        cur = self.cursor()
        cur.execute(sql, params)
        return cur

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        if exc_type:
            self._conn.rollback()
        else:
            self._conn.commit()
        self._conn.close()


# ─── Connection factory ──────────────────────────────────────

def connect():
    """Return a database connection (PostgreSQL or SQLite)."""
    if IS_POSTGRES:
        import psycopg2
        raw = psycopg2.connect(DATABASE_URL)
        raw.autocommit = False
        return _PgConn(raw)
    import sqlite3
    return sqlite3.connect('apex_market.db', timeout=30)


# ─── Pandas helper ───────────────────────────────────────────

def read_sql(sql, conn, params=None, **kwargs):
    """pandas read_sql_query wrapper — handles ? vs %s and connection unwrapping."""
    import pandas as pd
    if IS_POSTGRES:
        sql = _PH.sub('%s', sql)
        raw = conn._conn if isinstance(conn, _PgConn) else conn
        return pd.read_sql_query(sql, raw, params=params, **kwargs)
    return pd.read_sql_query(sql, conn, params=params, **kwargs)


# ─── Insert helpers ──────────────────────────────────────────

def insert_returning_id(conn, sql, params=None):
    """Execute INSERT and return the new row id. Replaces cursor.lastrowid."""
    if IS_POSTGRES:
        pg_sql = _pg(sql)
        if pg_sql is None:
            return None
        pg_sql = pg_sql.rstrip().rstrip(';') + ' RETURNING id'
        raw = conn._conn if isinstance(conn, _PgConn) else conn
        cur = raw.cursor()
        cur.execute(pg_sql, params if params is not None else ())
        row = cur.fetchone()
        return row[0] if row else None
    import sqlite3
    c = conn.cursor()
    c.execute(sql, params if params is not None else ())
    return c.lastrowid


def upsert_ohlcv(conn, symbol, timeframe, ts, o, h, l, close, vol):
    """INSERT OR REPLACE for ohlcv. PostgreSQL uses ON CONFLICT UPDATE."""
    o, h, l, close, vol = (
        round(float(o), 2), round(float(h), 2), round(float(l), 2),
        round(float(close), 2), float(vol)
    )
    if IS_POSTGRES:
        conn.execute(
            'INSERT INTO ohlcv (symbol,timeframe,ts,open,high,low,close,volume) '
            'VALUES (?,?,?,?,?,?,?,?) '
            'ON CONFLICT (symbol,timeframe,ts) DO UPDATE SET '
            'open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, '
            'close=EXCLUDED.close, volume=EXCLUDED.volume',
            (symbol, timeframe, int(ts), o, h, l, close, vol)
        )
    else:
        conn.execute(
            'INSERT OR REPLACE INTO ohlcv (symbol,timeframe,ts,open,high,low,close,volume) '
            'VALUES (?,?,?,?,?,?,?,?)',
            (symbol, timeframe, int(ts), o, h, l, close, vol)
        )


def kv_upsert(conn, key, value):
    """Upsert a key-value pair in paper_account. Replaces INSERT OR REPLACE."""
    if IS_POSTGRES:
        conn.execute(
            'INSERT INTO paper_account (key,value) VALUES (?,?) '
            'ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value',
            (key, str(value))
        )
    else:
        conn.execute(
            'INSERT OR REPLACE INTO paper_account (key,value) VALUES (?,?)',
            (key, str(value))
        )


# ─── Schema initialisation (PostgreSQL only) ─────────────────

_PG_DDL = [
    """CREATE TABLE IF NOT EXISTS ohlcv (
        id        SERIAL PRIMARY KEY,
        symbol    TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        ts        BIGINT NOT NULL,
        open      REAL, high REAL, low REAL, close REAL, volume REAL,
        UNIQUE(symbol, timeframe, ts)
    )""",
    """CREATE TABLE IF NOT EXISTS apex_trades (
        id              SERIAL PRIMARY KEY,
        symbol          TEXT NOT NULL,
        direction       TEXT NOT NULL,
        setup           TEXT NOT NULL,
        mode            TEXT DEFAULT 'swing',
        entry_price     REAL,
        stop            REAL,
        target          REAL,
        rr_planned      REAL,
        session         TEXT,
        quality         TEXT,
        entry_time      TEXT,
        exit_price      REAL,
        exit_time       TEXT,
        exit_reason     TEXT,
        pnl_r           REAL,
        status          TEXT DEFAULT 'open',
        bars_held       INTEGER DEFAULT 0,
        notes           TEXT,
        broker_order_id TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS macro_log (
        id     SERIAL PRIMARY KEY,
        ts     BIGINT NOT NULL,
        date   TEXT NOT NULL,
        vix    REAL, tnx REAL, irx REAL, dxy REAL, gold REAL, oil REAL,
        regime TEXT, notes TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS patterns (
        id           SERIAL PRIMARY KEY,
        name         TEXT UNIQUE,
        symbol       TEXT,
        conditions   TEXT,
        occurrences  INTEGER DEFAULT 0,
        wins         INTEGER DEFAULT 0,
        losses       INTEGER DEFAULT 0,
        avg_rr       REAL,
        expectancy   REAL,
        best_regime  TEXT,
        edge_score   REAL DEFAULT 0,
        last_updated BIGINT,
        active       INTEGER DEFAULT 1
    )""",
    """CREATE TABLE IF NOT EXISTS scan_log (
        id           SERIAL PRIMARY KEY,
        ts           BIGINT NOT NULL,
        symbol       TEXT,
        pattern_name TEXT,
        score        REAL,
        direction    TEXT,
        entry        REAL,
        stop         REAL,
        target1      REAL,
        target2      REAL,
        outcome      TEXT DEFAULT 'pending',
        outcome_rr   REAL,
        regime       TEXT,
        notes        TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS fvg_alerted_zones (
        symbol     TEXT NOT NULL,
        formed_at  TEXT NOT NULL,
        alerted_at TEXT NOT NULL,
        PRIMARY KEY (symbol, formed_at)
    )""",
    """CREATE TABLE IF NOT EXISTS swing_alerted_signals (
        id         SERIAL PRIMARY KEY,
        symbol     TEXT NOT NULL,
        direction  TEXT NOT NULL,
        setup      TEXT NOT NULL,
        date       TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS wyckoff_log (
        id          SERIAL PRIMARY KEY,
        symbol      TEXT NOT NULL,
        date        TEXT NOT NULL,
        entry_price REAL,
        stop        REAL,
        target      REAL,
        detected_at TEXT NOT NULL,
        resolved    INTEGER DEFAULT 0,
        pnl_r       REAL
    )""",
    """CREATE TABLE IF NOT EXISTS paper_account (
        id    SERIAL PRIMARY KEY,
        key   TEXT UNIQUE,
        value TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS paper_positions (
        id                  SERIAL PRIMARY KEY,
        symbol              TEXT,
        direction           TEXT,
        entry_price         REAL,
        current_price       REAL,
        stop                REAL,
        target1             REAL,
        target2             REAL,
        contracts           INTEGER DEFAULT 1,
        contracts_remaining INTEGER DEFAULT 1,
        status              TEXT DEFAULT 'open',
        setup_name          TEXT,
        setup_score         INTEGER,
        entry_ts            BIGINT,
        exit_ts             BIGINT,
        exit_price          REAL,
        pnl_pts             REAL DEFAULT 0,
        pnl_usd             REAL DEFAULT 0,
        be_stop_moved       INTEGER DEFAULT 0,
        partial_taken       INTEGER DEFAULT 0,
        notes               TEXT,
        alert_id            TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS paper_trades (
        id           SERIAL PRIMARY KEY,
        position_id  INTEGER,
        symbol       TEXT,
        direction    TEXT,
        entry_price  REAL,
        exit_price   REAL,
        contracts    INTEGER,
        pnl_pts      REAL,
        pnl_usd      REAL,
        outcome      TEXT,
        setup_name   TEXT,
        setup_score  INTEGER,
        entry_ts     BIGINT,
        exit_ts      BIGINT,
        hold_minutes INTEGER,
        partial      INTEGER DEFAULT 0,
        notes        TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS regime_log (
        id         SERIAL PRIMARY KEY,
        symbol     VARCHAR NOT NULL,
        regime     VARCHAR NOT NULL,
        hurst      FLOAT,
        autocorr   FLOAT,
        vol_ratio  FLOAT,
        confidence FLOAT,
        timestamp  TIMESTAMPTZ NOT NULL,
        reason     TEXT
    )""",
    # Indexes
    'CREATE INDEX IF NOT EXISTS idx_ohlcv_sym_tf_ts ON ohlcv (symbol, timeframe, ts)',
    'CREATE INDEX IF NOT EXISTS idx_apex_trades_status ON apex_trades (status)',
    'CREATE INDEX IF NOT EXISTS idx_wyckoff_sym_date ON wyckoff_log (symbol, date)',
    'CREATE INDEX IF NOT EXISTS idx_swing_alert ON swing_alerted_signals (symbol, direction, setup, date)',
    'CREATE INDEX IF NOT EXISTS idx_regime_log_sym_ts ON regime_log (symbol, timestamp)',
]

# Separate migration steps run after DDL — dedup then add unique index.
# ON CONFLICT (symbol,timeframe,ts) requires a UNIQUE INDEX to exist.
# CREATE TABLE IF NOT EXISTS won't add the constraint to a pre-existing table.
_PG_MIGRATIONS = [
    # Remove any duplicate ohlcv rows (keep lowest id) before creating unique index
    'DELETE FROM ohlcv WHERE id NOT IN '
    '(SELECT MIN(id) FROM ohlcv GROUP BY symbol, timeframe, ts)',
    # Unique index — required for ON CONFLICT (symbol,timeframe,ts) to work
    'CREATE UNIQUE INDEX IF NOT EXISTS idx_ohlcv_unique '
    'ON ohlcv (symbol, timeframe, ts)',
]


def init_schema():
    """Create all tables in PostgreSQL at startup. No-op for SQLite."""
    if not IS_POSTGRES:
        return
    try:
        import psycopg2
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = True
        cur = conn.cursor()
        for ddl in _PG_DDL:
            cur.execute(ddl)
        for sql in _PG_MIGRATIONS:
            try:
                cur.execute(sql)
            except Exception as mig_err:
                logger.warning(f'Migration step skipped ({mig_err}): {sql[:60]}')
        conn.close()
        logger.info('PostgreSQL schema initialised (migrations applied)')
    except Exception as e:
        logger.error(f'init_schema failed: {e}')
        raise
