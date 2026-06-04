"""
WISE MERIDIAN CAPITAL — Research Division
==========================================

PURPOSE
Autonomous research and strategy health monitoring system.
Completely isolated from live trading — reads existing data, writes only to research tables.
Never modifies apex_trades, never touches signal generation, never affects order execution.

MANDATE 1: DEFEND
Monitor the health of every live trading strategy.
Detect edge degradation before it becomes a trading problem.
Alert the portfolio manager when action is required.

MANDATE 2: DISCOVER (Phase 2 — coming June 2026)
Hypothesis engine, combination explorer, pattern library.
Shadow lab for validating new strategies before promotion.

SCHEDULING
02:00 UTC daily    — run_daily_backtest()   — full OHLCV backtest for all 7 setups
06:00 UTC Monday   — run_weekly_health_check() + score_shadow_lab() + generate_weekly_telegram_report()
On startup         — seed_shadow_lab_candidates() — idempotent, safe, never crashes server

DATABASE TABLES (research schema — never touches trading tables)
strategy_health_log   weekly health scores per setup (backtest + live dual score)
backtest_results      daily backtest results per setup (7 OHLCV engines)
shadow_lab            candidate strategies being paper validated (8-week programme)
research_decisions    pending human-approval decisions (promotion reviews)

CONNECTION CONTRACT
Every public function opens its own fresh connection, commits or rolls back explicitly,
and closes in a finally block. No shared connections across function boundaries.
A failure in one function can never leave another function's connection in an aborted state.

HEALTH SCORING
Dual scoring: 60% backtest (180-day OHLCV replay) + 40% live performance (apex_trades).
Falls back to 100% backtest if fewer than 5 live closed trades exist.
Alert levels: HEALTHY (>75), WATCH (50-75), ALERT (<50), INSUFFICIENT_DATA

BACKTEST ENGINES
Each setup has a dedicated function that replays its logic on historical OHLCV data.
Adding a new setup:
  1. Implement backtest_setup_X(conn, lookback_days=180) -> Optional[BacktestResult]
  2. Load OHLCV via _load_ohlcv_df(conn, symbol, timeframe, lookback_days)
  3. Detect signals with vectorized pandas logic
  4. Call _sim_outcome() per signal to get pnl_r
  5. Call _backtest_stats() then _edge_score() for 0-100 rating
  6. Register in BACKTEST_FUNCS dict at bottom of file

API ENDPOINTS (all read-only except manual triggers)
GET  /api/research/health       latest health per setup + 4-week trend
GET  /api/research/shadow       active shadow lab candidates
GET  /api/research/decisions    pending human decisions
GET  /api/research/backtest     latest backtest results per setup
POST /api/research/run_check    manual health check trigger (testing)
POST /api/research/run_backtest manual backtest trigger (testing)
POST /api/research/send_report  manual Telegram report trigger (testing)
"""

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, date, timedelta, timezone
from typing import Optional

import db as _db

logger = logging.getLogger('APEX.Research')


# ── Benchmarks (live trading targets) ───────────────────────────────────────

BENCHMARKS = {
    'D': {'sharpe': 10.60, 'wr': 0.61},
    'B': {'sharpe':  7.50, 'wr': 0.52},
    'E': {'sharpe':  5.80, 'wr': 0.49},
    'H': {'sharpe':  6.00, 'wr': 0.53},
    'I': {'sharpe':  8.40, 'wr': 0.67},
    'C': {'sharpe':  9.42, 'wr': 0.55},
    'A': {'sharpe': 12.81, 'wr': 0.60},
}

SETUP_NAMES = {
    'A': 'Sweep + OB',
    'B': 'ChoCh + OB',
    'C': 'BOS + OB',
    'D': 'FVG Fill',
    'E': 'EMA50 Pullback',
    'H': 'VWAP Reversion',
    'I': 'Mathematical Alpha',
}


# ── BacktestResult ───────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    setup_id: str
    lookback_days: int
    total_signals: int
    win_rate: float
    sharpe: Optional[float]
    avg_r: float
    expectancy: float
    max_drawdown: float
    profit_factor: float
    benchmark_sharpe: float
    benchmark_win_rate: float
    sharpe_vs_benchmark: float
    wr_vs_benchmark: float
    edge_score: int
    run_date: date
    bars_analysed: int


# In-memory dedup for Gap ORB shadow scan — resets on process restart (Railway redeploy)
_gap_orb_fired: dict = {}   # key: 'YYYY-MM-DD', value: True when signal already fired today
_shadow_k_es_fired: dict = {}   # Setup K ES dedup — one signal per day
_shadow_k_mnq_fired: dict = {}  # Setup K MNQ dedup — one signal per day


# ═══════════════════════════════════════════════════════════════════════════
#  CONNECTION HELPER
# ═══════════════════════════════════════════════════════════════════════════

def _conn():
    """Open a fresh DB connection. Always close in a finally block."""
    return _db.connect()


def _rollback(conn):
    """Rollback a connection without raising. Safe to call in any except block."""
    try:
        conn.rollback()
    except Exception:
        pass


def _close(conn):
    """Close a connection without raising. Safe to call in any finally block."""
    try:
        conn.close()
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  SCHEMA HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_research_schema():
    """
    Ensure all research tables and columns exist.
    Each DDL runs in its own transaction so one failure cannot contaminate another.
    Safe to call on every startup — all statements use IF NOT EXISTS or catch-and-rollback.
    """
    # Each operation gets a completely isolated connection so failures don't contaminate
    ddl_ops = []

    # PostgreSQL: use ADD COLUMN IF NOT EXISTS (pg >= 9.6) to avoid any exception
    if _db.IS_POSTGRES:
        ddl_ops = [
            ("ALTER TABLE strategy_health_log ADD COLUMN IF NOT EXISTS backtest_score INT",
             "backtest_score column"),
            ("ALTER TABLE strategy_health_log ADD COLUMN IF NOT EXISTS live_score INT",
             "live_score column"),
            # Ensure backtest_results.id has an auto-increment sequence.
            # Earlier deploy may have created it with INTEGER PRIMARY KEY (no SERIAL).
            ("CREATE SEQUENCE IF NOT EXISTS backtest_results_id_seq",
             "backtest_results id sequence"),
            ("ALTER TABLE backtest_results ALTER COLUMN id "
             "SET DEFAULT nextval('backtest_results_id_seq')",
             "backtest_results id default"),
            ("CREATE INDEX IF NOT EXISTS idx_backtest_setup_date "
             "ON backtest_results (setup_id, run_date)",
             "backtest index"),
            # Phase 3 tables
            ("""CREATE TABLE IF NOT EXISTS hypothesis_log (
                id SERIAL PRIMARY KEY,
                hypothesis_id TEXT UNIQUE NOT NULL,
                description TEXT,
                category TEXT,
                instrument TEXT,
                lookback_days INTEGER,
                signals_generated INTEGER,
                win_rate REAL,
                sharpe REAL,
                avg_r REAL,
                information_coefficient REAL,
                p_value REAL,
                status TEXT DEFAULT 'TESTING',
                run_date TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )""", "hypothesis_log table"),
            ("""CREATE TABLE IF NOT EXISTS feature_combinations (
                id SERIAL PRIMARY KEY,
                features TEXT NOT NULL,
                oos_ic REAL,
                oos_auc REAL,
                run_date TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )""", "feature_combinations table"),
            ("""CREATE TABLE IF NOT EXISTS pattern_library (
                id SERIAL PRIMARY KEY,
                pattern_id TEXT UNIQUE NOT NULL,
                name TEXT,
                description TEXT,
                discovery_source TEXT,
                instrument TEXT,
                signals_observed INTEGER DEFAULT 0,
                win_rate REAL,
                sharpe REAL,
                information_coefficient REAL,
                first_observed TEXT,
                last_validated TEXT,
                decay_score REAL DEFAULT 1.0,
                status TEXT DEFAULT 'ACTIVE',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )""", "pattern_library table"),
            # Extend research_decisions with reason column
            ("ALTER TABLE research_decisions ADD COLUMN IF NOT EXISTS reason TEXT",
             "research_decisions reason column"),
        ]
    else:
        # SQLite: CREATE core research tables first, then migrate columns
        ddl_ops = [
            ("""CREATE TABLE IF NOT EXISTS strategy_health_log (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                setup_id            TEXT,
                week_start          TEXT,
                sharpe_30d          REAL,
                sharpe_benchmark    REAL,
                win_rate            REAL,
                win_rate_benchmark  REAL,
                signal_count_week   INTEGER,
                expectancy          REAL,
                health_score        INTEGER,
                alert_level         TEXT,
                backtest_score      INTEGER,
                live_score          INTEGER,
                notes               TEXT,
                created_at          TEXT DEFAULT CURRENT_TIMESTAMP
            )""", "strategy_health_log table"),
            ("""CREATE TABLE IF NOT EXISTS shadow_lab (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name           TEXT,
                description             TEXT,
                entered_date            TEXT,
                week_number             INTEGER,
                total_weeks             INTEGER DEFAULT 8,
                paper_sharpe            REAL,
                paper_win_rate          REAL,
                paper_total_r           REAL,
                paper_signal_count      INTEGER,
                backtest_sharpe         REAL,
                backtest_win_rate       REAL,
                status                  TEXT,
                promotion_eligible_date TEXT,
                notes                   TEXT,
                created_at              TEXT DEFAULT CURRENT_TIMESTAMP
            )""", "shadow_lab table"),
            ("""CREATE TABLE IF NOT EXISTS research_decisions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_type   TEXT,
                subject         TEXT,
                recommendation  TEXT,
                supporting_data TEXT,
                status          TEXT,
                decided_at      TEXT,
                outcome         TEXT,
                reason          TEXT,
                created_at      TEXT DEFAULT CURRENT_TIMESTAMP
            )""", "research_decisions table"),
            ("""CREATE TABLE IF NOT EXISTS backtest_results (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                setup_id            TEXT,
                lookback_days       INTEGER,
                run_date            TEXT,
                total_signals       INTEGER,
                win_rate            REAL,
                sharpe              REAL,
                avg_r               REAL,
                expectancy          REAL,
                max_drawdown        REAL,
                profit_factor       REAL,
                benchmark_sharpe    REAL,
                benchmark_win_rate  REAL,
                sharpe_vs_benchmark REAL,
                wr_vs_benchmark     REAL,
                edge_score          INTEGER,
                bars_analysed       INTEGER,
                created_at          TEXT DEFAULT CURRENT_TIMESTAMP
            )""", "backtest_results table"),
            # SQLite: no IF NOT EXISTS for ALTER TABLE — use try/except with rollback
            ("ALTER TABLE strategy_health_log ADD COLUMN backtest_score INTEGER",
             "backtest_score column"),
            ("ALTER TABLE strategy_health_log ADD COLUMN live_score INTEGER",
             "live_score column"),
            # Phase 3 tables (SQLite supports CREATE TABLE IF NOT EXISTS)
            ("""CREATE TABLE IF NOT EXISTS hypothesis_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hypothesis_id TEXT UNIQUE NOT NULL,
                description TEXT,
                category TEXT,
                instrument TEXT,
                lookback_days INTEGER,
                signals_generated INTEGER,
                win_rate REAL,
                sharpe REAL,
                avg_r REAL,
                information_coefficient REAL,
                p_value REAL,
                status TEXT DEFAULT 'TESTING',
                run_date TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )""", "hypothesis_log table"),
            ("""CREATE TABLE IF NOT EXISTS feature_combinations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                features TEXT NOT NULL,
                oos_ic REAL,
                oos_auc REAL,
                run_date TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )""", "feature_combinations table"),
            ("""CREATE TABLE IF NOT EXISTS pattern_library (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id TEXT UNIQUE NOT NULL,
                name TEXT,
                description TEXT,
                discovery_source TEXT,
                instrument TEXT,
                signals_observed INTEGER DEFAULT 0,
                win_rate REAL,
                sharpe REAL,
                information_coefficient REAL,
                first_observed TEXT,
                last_validated TEXT,
                decay_score REAL DEFAULT 1.0,
                status TEXT DEFAULT 'ACTIVE',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )""", "pattern_library table"),
            ("ALTER TABLE research_decisions ADD COLUMN reason TEXT",
             "research_decisions reason column"),
        ]

    for sql, label in ddl_ops:
        c = _conn()
        try:
            c.execute(sql)
            c.commit()
            logger.debug(f'Research schema: {label} OK')
        except Exception as e:
            _rollback(c)
            # "already exists" is expected on every restart after first deploy — not an error
            if 'already exist' in str(e).lower() or 'duplicate column' in str(e).lower():
                logger.debug(f'Research schema: {label} already exists (OK)')
            else:
                logger.warning(f'Research schema: {label} skipped — {e}')
        finally:
            _close(c)


# ═══════════════════════════════════════════════════════════════════════════
#  RESEARCH STATE — DB-backed dedup for scheduled jobs
# ═══════════════════════════════════════════════════════════════════════════

def _health_check_ran_recently() -> bool:
    """Return True if run_weekly_health_check ran within the last 24 hours.

    Queries the research_state table so the guard survives server restarts.
    Returns False (allow run) on any DB error so a broken state table never
    permanently blocks the health check.
    """
    c = _conn()
    try:
        row = c.execute(
            "SELECT value FROM research_state WHERE key = 'last_health_check_run'"
        ).fetchone()
        if not row:
            return False
        last_run = datetime.fromisoformat(row[0])
        if last_run.tzinfo is None:
            last_run = last_run.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - last_run) < timedelta(hours=24)
    except Exception as e:
        logger.warning(f'Research state read failed: {e}')
        return False
    finally:
        _close(c)


def _mark_health_check_ran() -> None:
    """Persist the current UTC time as the last successful health check timestamp."""
    c = _conn()
    try:
        now_iso = datetime.now(timezone.utc).isoformat()
        if _db.IS_POSTGRES:
            c.execute(
                "INSERT INTO research_state (key, value) VALUES ('last_health_check_run', ?) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                (now_iso,)
            )
        else:
            c.execute(
                "INSERT OR REPLACE INTO research_state (key, value) "
                "VALUES ('last_health_check_run', ?)",
                (now_iso,)
            )
        c.commit()
    except Exception as e:
        logger.warning(f'Research state write failed: {e}')
        _rollback(c)
    finally:
        _close(c)


# ═══════════════════════════════════════════════════════════════════════════
#  BACKTEST HELPERS (read-only, accept conn param for isolation)
# ═══════════════════════════════════════════════════════════════════════════

def _load_ohlcv_df(conn, symbol: str, timeframe: str, lookback_days: int):
    """Load OHLCV bars as a pandas DataFrame. Returns empty DataFrame on any failure."""
    try:
        import pandas as pd
        cutoff = int((datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp())
        rows = conn.execute(
            "SELECT ts, open, high, low, close, volume FROM ohlcv "
            "WHERE symbol=? AND timeframe=? AND ts>? ORDER BY ts ASC",
            (symbol, timeframe, max(cutoff, 1))
        ).fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df['dt']      = pd.to_datetime(df['ts'], unit='s', utc=True)
        df['hour']    = df['dt'].dt.hour
        df['weekday'] = df['dt'].dt.weekday
        df['date']    = df['dt'].dt.date
        return df.reset_index(drop=True)
    except Exception as e:
        logger.warning(f'_load_ohlcv_df {symbol} {timeframe}: {e}')
        try:
            import pandas as pd
            return pd.DataFrame()
        except ImportError:
            return []


def _atr14(df, n: int = 14):
    import pandas as pd
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift(1)).abs()
    lc = (df['low']  - df['close'].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _ema(series, n: int):
    return series.ewm(span=n, adjust=False).mean()


def _sim_outcome(df, start_idx: int, direction: str,
                 entry: float, stop: float, target: float,
                 max_bars: int = 100) -> float:
    risk = abs(entry - stop)
    if risk <= 0:
        return 0.0
    rr   = abs(target - entry) / risk
    end  = min(start_idx + max_bars + 1, len(df))
    highs = df['high'].values
    lows  = df['low'].values
    for i in range(start_idx + 1, end):
        if direction == 'long':
            if highs[i] >= target: return rr
            if lows[i]  <= stop:   return -1.0
        else:
            if lows[i]  <= target: return rr
            if highs[i] >= stop:   return -1.0
    return 0.0


def _py(v):
    """Convert numpy scalar to Python native type. psycopg2 cannot adapt numpy types."""
    if v is None:
        return None
    try:
        # numpy scalars have an item() method
        return v.item()
    except AttributeError:
        return float(v) if isinstance(v, (int, float)) else v


def _backtest_stats(pnl_list: list, benchmark_sharpe: float, benchmark_wr: float,
                    bars_analysed: int, setup_id: str, lookback_days: int) -> BacktestResult:
    # Convert all pnl values to Python floats — they may be numpy.float64 from df.values
    pnl_list = [float(p) for p in pnl_list]
    n = len(pnl_list)
    if n == 0:
        return BacktestResult(
            setup_id=setup_id, lookback_days=int(lookback_days),
            total_signals=0, win_rate=0.0, sharpe=None,
            avg_r=0.0, expectancy=0.0, max_drawdown=0.0,
            profit_factor=0.0, benchmark_sharpe=float(benchmark_sharpe),
            benchmark_win_rate=float(benchmark_wr),
            sharpe_vs_benchmark=0.0, wr_vs_benchmark=0.0,
            edge_score=0, run_date=date.today(), bars_analysed=int(bars_analysed),
        )
    wins       = [p for p in pnl_list if p > 0]
    losses     = [p for p in pnl_list if p < 0]
    win_rate   = float(len(wins)) / n
    avg_r      = sum(pnl_list) / n
    mean       = avg_r
    var        = sum((v - mean) ** 2 for v in pnl_list) / (n - 1) if n > 1 else 0
    std        = math.sqrt(float(var)) if var > 0 else 0
    sharpe     = float(mean / std * math.sqrt(252)) if std > 0 else None
    gross_win  = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = float(
        gross_win / gross_loss if gross_loss > 0
        else (999.0 if gross_win > 0 else 0.0)
    )
    # Drawdown: measure peak-to-trough in absolute R, then normalise by 100R
    # (100R = 1% risk per trade on a standard account — keeps DD bounded and
    # mathematically consistent with 1R max loss per trade).
    cum = peak = max_dd_abs = 0.0
    for p in pnl_list:
        cum += p
        if cum > peak: peak = cum
        dd_abs = peak - cum
        if dd_abs > max_dd_abs: max_dd_abs = dd_abs
    max_dd = float(max_dd_abs / 100.0)

    sharpe_vs_bm = float(((sharpe / benchmark_sharpe) - 1) * 100
                         if (sharpe and benchmark_sharpe) else -100.0)
    wr_vs_bm     = float((win_rate - benchmark_wr) * 100)

    es = _edge_score(win_rate, sharpe, n, profit_factor, max_dd,
                     float(benchmark_sharpe), float(benchmark_wr))
    return BacktestResult(
        setup_id=setup_id, lookback_days=int(lookback_days),
        total_signals=n, win_rate=win_rate, sharpe=sharpe,
        avg_r=avg_r, expectancy=avg_r, max_drawdown=max_dd,
        profit_factor=profit_factor,
        benchmark_sharpe=float(benchmark_sharpe),
        benchmark_win_rate=float(benchmark_wr),
        sharpe_vs_benchmark=round(sharpe_vs_bm, 1),
        wr_vs_benchmark=round(wr_vs_bm, 1),
        edge_score=es, run_date=date.today(), bars_analysed=int(bars_analysed),
    )


def _edge_score(win_rate: float, sharpe: Optional[float], total_signals: int,
                profit_factor: float, max_drawdown: float,
                benchmark_sharpe: float, benchmark_wr: float) -> int:
    """
    Edge score 0-100 using practical backtest thresholds.
    Profit factor is the primary signal (most reliable in simplified backtests).
    Sharpe compared to 30% of live benchmark (floor 1.0) to account for simplified logic.
    """
    score = 50
    if profit_factor >= 1.5:    score += 20
    elif profit_factor >= 1.2:  score += 10
    elif profit_factor >= 1.0:  score +=  0
    elif profit_factor >= 0.85: score -= 10
    else:                       score -= 20
    bt_wr   = max(0.48, min(benchmark_wr * 0.85, 0.55))
    wr_diff = win_rate - bt_wr
    if wr_diff >= -0.03:    score += 15
    elif wr_diff >= -0.08:  score +=  0
    elif wr_diff >= -0.13:  score -= 10
    else:                   score -= 20
    bt_sharpe = max(1.0, benchmark_sharpe * 0.30)
    if sharpe is not None:
        ratio = sharpe / bt_sharpe
        if ratio >= 0.9:    score += 15
        elif ratio >= 0.5:  score +=  0
        elif ratio >= 0.0:  score -= 10
        else:               score -= 20
    else:
        score -= 10
    if total_signals >= 30:   pass
    elif total_signals >= 10: score -= 10
    else:                     score -= 20
    if max_drawdown > 0.35:   score -= 15
    elif max_drawdown > 0.25: score -=  5
    return max(0, min(100, score))


# ═══════════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINES
# ═══════════════════════════════════════════════════════════════════════════

def _fvg_score_backtest(fvg: dict, df15_pre_entry, vol_baseline: float) -> int:
    """
    Score an FVG for historical backtesting.

    Mirrors score_fvg() from fvg_engine.py but uses PRE-ENTRY state:
    - df15_pre_entry: 15min bars UP TO (not including) the trigger bar period
    - vol_baseline: rolling 20-bar volume mean at trigger time (not end-of-dataset)

    This is critical: score_fvg called AFTER the entry bar is in the slice will
    see 50-75% CLEAN penetration (the entry bar itself touched the FVG), giving
    only +5 instead of +25 pts. Pre-entry slice fixes this completely.
    """
    score = 0

    # 1. SIZE (0-30)
    size_ratio = fvg['size'] / fvg['atr'] if fvg['atr'] > 0 else 0
    if size_ratio >= 1.0:   score += 30
    elif size_ratio >= 0.7: score += 20
    elif size_ratio >= 0.4: score += 10
    else:                   score += 5

    # 2. FRESHNESS (0-25): bars from formation to last pre-entry bar
    try:
        if fvg['formed_at'] in df15_pre_entry.index:
            formed_pos  = df15_pre_entry.index.get_loc(fvg['formed_at'])
            current_pos = len(df15_pre_entry) - 1
            age_bars    = max(0, current_pos - formed_pos)
        else:
            age_bars = 99
        if age_bars <= 4:    score += 25
        elif age_bars <= 8:  score += 18
        elif age_bars <= 16: score += 10
        elif age_bars <= 32: score += 5
    except Exception:
        score += 10

    # 3. CLEAN (0-25): penetration in bars from formation to pre-entry (no entry bar)
    try:
        since = df15_pre_entry[df15_pre_entry.index >= fvg['formed_at']]
        if len(since) > 0 and fvg['size'] > 0:
            if fvg['type'] == 'bullish':
                pen = max(0.0, (fvg['top'] - float(since['low'].min())) / fvg['size'])
            else:
                pen = max(0.0, (float(since['high'].max()) - fvg['bottom']) / fvg['size'])
            if pen <= 0.25:   score += 25
            elif pen <= 0.50: score += 15
            elif pen <= 0.75: score += 5
        else:
            score += 25   # just formed — fully clean
    except Exception:
        score += 10

    # 4. VOLUME (0-20): formation-bar volume vs rolling baseline at trigger time
    try:
        if fvg['formed_at'] in df15_pre_entry.index and vol_baseline > 0:
            vol_ratio = float(df15_pre_entry.loc[fvg['formed_at'], 'volume']) / vol_baseline
            if vol_ratio >= 2.0:   score += 20
            elif vol_ratio >= 1.5: score += 15
            elif vol_ratio >= 1.0: score += 8
            else:                  score += 3
        else:
            score += 8
    except Exception:
        score += 8

    return score


def backtest_setup_d(conn, lookback_days: int = 180) -> Optional[BacktestResult]:
    """
    FVG Fill — efficient backtest that matches live scan_setup_d logic.

    Key design decisions vs the broken prior version:
    - Iterates over 15MIN bars (not 5min) as the outer loop.  Detects FVGs
      once per qualifying 15min bar using the last LOOKBACK bars — O(n_15min)
      instead of the prior O(n_5min × n_15min) which was 40x slower.
    - Scores FVGs using _fvg_score_backtest() with PRE-ENTRY state:
      df15 slice ends at the current 15min bar (BEFORE the 5min trigger bar).
      The prior code sliced up to the 5min trigger bar, so the entry bar itself
      was in the CLEAN calculation — showing 50-75% penetration and scoring ~48
      instead of 70+.  This produced 0 passing signals.
    - Uses rolling 20-bar volume baseline (not end-of-dataset fixed baseline).
    - Finds 5min entry trigger in the next 30 bars after each qualifying FVG.
    """
    try:
        import pandas as pd
        import numpy as np
        from fvg_engine import detect_fvgs, SETUP_D_PARAMS

        df15 = _load_ohlcv_df(conn, 'MNQ', '15min', lookback_days)
        if len(df15) < 50:
            return None
        df5  = _load_ohlcv_df(conn, 'MNQ', '5min',  lookback_days)
        if len(df5)  < 50:
            return None

        # ── 4h EMA20 bias (identical to fvg_engine.get_htf_bias) ──────────
        df5_idx = df5.set_index('dt')
        df4h = df5_idx[['open', 'high', 'low', 'close', 'volume']].resample('4h').agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        ).dropna()
        if len(df4h) < 21:
            return None
        df4h['ema20'] = df4h['close'].ewm(span=20, adjust=False).mean()
        _bias_raw = (
            (df4h['close'] > df4h['ema20'] * 1.001).astype(int)
            - (df4h['close'] < df4h['ema20'] * 0.999).astype(int)
        )
        df15['bias4h'] = _bias_raw.reindex(df15.set_index('dt').index, method='ffill').fillna(0).values
        df5['bias4h']  = _bias_raw.reindex(df5_idx.index,              method='ffill').fillna(0).values

        atr5 = _atr14(df5)

        # Precompute rolling 20-bar volume baseline on 15min
        vol_roll = df15['volume'].rolling(20, min_periods=5).mean().values

        # Numpy arrays for fast access
        ts15   = df15['ts'].values;    hrs15  = df15['hour'].values
        wday15 = df15['weekday'].values; bias15 = df15['bias4h'].values
        bars15 = len(df15)

        ts5   = df5['ts'].values;   hrs5  = df5['hour'].values
        c5    = df5['close'].values; o5   = df5['open'].values
        h5    = df5['high'].values;  l5   = df5['low'].values
        atr5v = atr5.values;         bars5 = len(df5)

        LOOKBACK  = SETUP_D_PARAMS['fvg_lookback_bars']   # 96 15min bars = 24h
        MIN_SCORE = SETUP_D_PARAMS['min_score']            # 70
        STOP_ATR  = SETUP_D_PARAMS.get('stop_atr', 1.0)   # 1.0
        TARGET_RR = SETUP_D_PARAMS['target_rr']            # 2.5

        pnl        = []

        for i15 in range(LOOKBACK, bars15 - 1):
            if wday15[i15] >= 5:                    continue
            if not (13 <= hrs15[i15] < 19):         continue
            b = int(bias15[i15])
            if b == 0:                              continue
            bias_str = 'bullish' if b > 0 else 'bearish'

            # ── Detect FVGs on the window ENDING at this 15min bar ─────────
            # This is the PRE-ENTRY state for any 5min bar that follows.
            start15  = max(0, i15 - LOOKBACK + 1)
            df15_win = df15.iloc[start15: i15 + 1].set_index('dt')
            atr15_win = _atr14(df15.iloc[start15: i15 + 1])
            atr15_win.index = df15_win.index

            fvgs = detect_fvgs(
                df15_win, atr15_win,
                min_atr_mult=SETUP_D_PARAMS['min_fvg_atr'],
                lookback=LOOKBACK,
            )
            if not fvgs:
                continue

            # Rolling vol baseline at this point in time
            vb = float(vol_roll[i15]) if not math.isnan(vol_roll[i15]) else 1.0

            current_ts = int(ts15[i15])
            # Start searching for 5min entry AFTER this 15min bar closes
            j_start = int(np.searchsorted(ts5, current_ts, side='right'))

            for fvg in fvgs:
                # Direction filter
                if fvg['type'] == 'bullish' and bias_str != 'bullish': continue
                if fvg['type'] == 'bearish' and bias_str != 'bearish': continue

                # Quality gate — pre-entry state, rolling vol baseline
                sc = _fvg_score_backtest(fvg, df15_win, vb)
                if sc < MIN_SCORE:
                    continue

                direction = 'long' if fvg['type'] == 'bullish' else 'short'

                # ── Search 5min bars for entry trigger (next 30 bars) ──────
                for j in range(j_start, min(j_start + 30, bars5)):
                    if not (13 <= hrs5[j] < 19): break
                    bc  = float(c5[j]); bo = float(o5[j])
                    bh  = float(h5[j]); bl = float(l5[j])
                    av5 = atr5v[j]
                    if not av5 or math.isnan(av5): continue

                    triggered = False
                    if direction == 'long' and bl <= fvg['top'] and bc >= fvg['mid'] and bc > bo:
                        triggered = True
                    elif direction == 'short' and bh >= fvg['bottom'] and bc <= fvg['mid'] and bc < bo:
                        triggered = True

                    if not triggered:
                        continue

                    if direction == 'long':
                        entry  = bc
                        stop   = fvg['bottom'] - STOP_ATR * av5
                        target = entry + TARGET_RR * (entry - stop)
                    else:
                        entry  = bc
                        stop   = fvg['top'] + STOP_ATR * av5
                        target = entry - TARGET_RR * (stop - entry)

                    if abs(entry - stop) > 0:
                        pnl.append(_sim_outcome(df5, j, direction, entry, stop, target))
                    break   # one trade per FVG
                break       # one FVG per 15min bar

        b = BENCHMARKS['D']
        return _backtest_stats(pnl, b['sharpe'], b['wr'], len(df15), 'D', lookback_days)
    except Exception as e:
        logger.error(f'backtest_setup_d: {e}', exc_info=True)
        return None


def backtest_setup_e(conn, lookback_days: int = 180) -> Optional[BacktestResult]:
    """EMA50 Pullback — MNQ 5min, session 13-18, ATR > 25."""
    try:
        df = _load_ohlcv_df(conn, 'MNQ', '5min', lookback_days)
        if len(df) < 60: return None
        atr = _atr14(df); ema50 = _ema(df['close'], 50)
        closes = df['close'].values; hours = df['hour'].values; bars = len(df)
        pnl = []
        for i in range(60, bars - 1):
            if not (13 <= hours[i] < 18): continue
            atr_val = atr.iloc[i]; ema_val = ema50.iloc[i]; ema_prev = ema50.iloc[i - 1]
            if not atr_val or math.isnan(atr_val) or atr_val < 25: continue
            if ema_val > ema_prev and 0 <= closes[i] - ema_val <= 0.5 * atr_val:
                e = closes[i]
                pnl.append(_sim_outcome(df, i, 'long', e, ema_val - 1.5 * atr_val, e + 3.75 * atr_val))
            elif ema_val < ema_prev and 0 <= ema_val - closes[i] <= 0.5 * atr_val:
                e = closes[i]
                pnl.append(_sim_outcome(df, i, 'short', e, ema_val + 1.5 * atr_val, e - 3.75 * atr_val))
        b = BENCHMARKS['E']
        return _backtest_stats(pnl, b['sharpe'], b['wr'], len(df), 'E', lookback_days)
    except Exception as e:
        logger.error(f'backtest_setup_e: {e}'); return None


def _backtest_choch_ob(conn, setup_id: str, symbol: str,
                       session_start: int, session_end: int,
                       stop_mult: float, rr: float, lookback_days: int) -> Optional[BacktestResult]:
    """Shared engine for B/A/C: swing-break + last-candle OB entry + 4h bias filter."""
    try:
        import pandas as pd
        df5 = _load_ohlcv_df(conn, symbol, '5min', lookback_days)
        if len(df5) < 80: return None
        df5_idx = df5.set_index('dt')

        # 4h EMA20 bias — same logic as fvg_engine.get_htf_bias
        df4h = df5_idx[['open', 'high', 'low', 'close', 'volume']].resample('4h').agg(
            {'open': 'first', 'high': 'max', 'low': 'min',
             'close': 'last',  'volume': 'sum'}
        ).dropna()
        if len(df4h) >= 21:
            df4h['ema20'] = df4h['close'].ewm(span=20, adjust=False).mean()
            _bias_raw = (df4h['close'] > df4h['ema20'] * 1.001).astype(int) - \
                        (df4h['close'] < df4h['ema20'] * 0.999).astype(int)
        else:
            _bias_raw = None

        df15 = df5_idx[['open', 'high', 'low', 'close', 'volume']].resample('15min').agg(
            {'open': 'first', 'high': 'max', 'low': 'min',
             'close': 'last',  'volume': 'sum'}
        ).dropna()
        df15['hour'] = df15.index.hour
        if _bias_raw is not None:
            df15['bias4h'] = _bias_raw.reindex(
                df15.index, method='ffill').fillna(0)
        else:
            df15['bias4h'] = 0
        df15 = df15.reset_index(drop=True)

        atr    = _atr14(df15)
        highs  = df15['high'].values;  lows   = df15['low'].values
        closes = df15['close'].values; opens  = df15['open'].values
        hours  = df15['hour'].values;  bars   = len(df15); sw = 20
        bias4h = df15['bias4h'].values
        pnl    = []
        for i in range(sw + 2, bars - 1):
            if not (session_start <= hours[i] < session_end): continue
            atr_val = atr.iloc[i]
            if not atr_val or math.isnan(atr_val): continue
            b = int(bias4h[i])
            sh = max(highs[i - sw: i - 1]); sl = min(lows[i - sw: i - 1])
            if closes[i] > sh and closes[i - 1] <= sh and b >= 0:
                for k in range(i - 1, max(i - 15, 0), -1):
                    if closes[k] < opens[k]:
                        e = highs[k]; s = lows[k] - stop_mult * atr_val
                        pnl.append(_sim_outcome(
                            df15, i, 'long', e, s, e + rr * abs(e - s))); break
            elif closes[i] < sl and closes[i - 1] >= sl and b <= 0:
                for k in range(i - 1, max(i - 15, 0), -1):
                    if closes[k] > opens[k]:
                        e = lows[k]; s = highs[k] + stop_mult * atr_val
                        pnl.append(_sim_outcome(
                            df15, i, 'short', e, s, e - rr * abs(e - s))); break
        b = BENCHMARKS[setup_id]
        return _backtest_stats(pnl, b['sharpe'], b['wr'], len(df15), setup_id, lookback_days)
    except Exception as e:
        logger.error(f'backtest_{setup_id.lower()}: {e}'); return None


def backtest_setup_b(conn, lookback_days: int = 180) -> Optional[BacktestResult]:
    return _backtest_choch_ob(conn, 'B', 'MNQ', 7, 11, 0.8, 4.0, lookback_days)

def backtest_setup_a(conn, lookback_days: int = 180) -> Optional[BacktestResult]:
    return _backtest_choch_ob(conn, 'A', 'MNQ', 7, 19, 0.8, 4.0, lookback_days)

def backtest_setup_c(conn, lookback_days: int = 180) -> Optional[BacktestResult]:
    return _backtest_choch_ob(conn, 'C', 'ES',  7, 11, 0.8, 4.0, lookback_days)


def backtest_setup_h(conn, lookback_days: int = 180) -> Optional[BacktestResult]:
    """
    VWAP 2σ Reversion — ES 5min, session 13-19.

    Conditions relaxed vs prior version:
    - Removed secondary 'vw < closes[i] - 0.5*atr' guard (redundant when
      close is already > vwap + 2*vstd; was preventing valid entries).
    - vstd minimum lowered from 0.5 to relative threshold (0.0001*close)
      so ES at ~6500 with vstd ~25 always passes.
    - Added minimum ATR filter (> 5 pts) to skip near-flat bars.
    """
    try:
        import pandas as pd, numpy as np
        df = _load_ohlcv_df(conn, 'ES', '5min', lookback_days)
        if len(df) < 50: return None
        atr  = _atr14(df)
        df['tp']       = (df['high'] + df['low'] + df['close']) / 3
        df['tp_vol']   = df['tp'] * df['volume'].clip(lower=0)
        df['cum_tpvol']= df.groupby('date')['tp_vol'].cumsum()
        df['cum_vol']  = df.groupby('date')['volume'].transform(lambda x: x.clip(lower=0).cumsum())
        df['vwap']     = df['cum_tpvol'] / df['cum_vol'].replace(0, float('nan'))
        df['dev2']     = ((df['tp'] - df['vwap']) ** 2) * df['volume'].clip(lower=0)
        df['cum_dev2'] = df.groupby('date')['dev2'].cumsum()
        with np.errstate(invalid='ignore'):
            df['vstd'] = (df['cum_dev2'] / df['cum_vol'].replace(0, float('nan'))).pow(0.5)
        highs  = df['high'].values;  lows   = df['low'].values
        closes = df['close'].values; hours  = df['hour'].values
        vwap_v = df['vwap'].values;  vstd_v = df['vstd'].values
        bars   = len(df); pnl = []
        for i in range(20, bars - 1):
            if not (13 <= hours[i] < 19): continue
            atr_val = atr.iloc[i]; vw = vwap_v[i]; vs = vstd_v[i]
            cl = closes[i]
            if not vw or math.isnan(vw): continue
            if not vs or math.isnan(vs) or vs < 0.0001 * cl: continue
            if not atr_val or math.isnan(atr_val) or atr_val < 5: continue
            upper = vw + 2.0 * vs; lower = vw - 2.0 * vs
            # Short: close above upper band (price extended above VWAP)
            if cl > upper:
                pnl.append(_sim_outcome(df, i, 'short', cl, cl + 1.5 * atr_val, vw))
            # Long: close below lower band (price extended below VWAP)
            elif cl < lower:
                pnl.append(_sim_outcome(df, i, 'long',  cl, cl - 1.5 * atr_val, vw))
        b = BENCHMARKS['H']
        return _backtest_stats(pnl, b['sharpe'], b['wr'], len(df), 'H', lookback_days)
    except Exception as e:
        logger.error(f'backtest_setup_h: {e}'); return None


def backtest_setup_i(conn, lookback_days: int = 180) -> Optional[BacktestResult]:
    """
    Mathematical Alpha — MNQ 5min, Hurst + momentum, Tue-Thu, 13-19 UTC.

    Prefers stored XGB predictions from regime_log (hurst, autocorr, vol_ratio).
    Falls back to mathematical calculation when regime_log has insufficient data.
    Threshold: hurst > 0.55 AND autocorr > 0.05 (long) or < -0.05 (short).
    """
    try:
        import numpy as np

        cutoff_ts = int(
            (datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp()
        )

        # ── Attempt 1: use stored regime_log predictions ─────────────────
        # regime_log stores timestamp as TIMESTAMPTZ (PG) or TEXT (SQLite).
        # Use strftime() which works on both backends.
        # On failure we MUST rollback so the PG connection isn't left in aborted state.
        regime_rows = []
        try:
            cutoff_iso = datetime.fromtimestamp(cutoff_ts, tz=timezone.utc).isoformat()
            # Use strftime to get unix epoch — works on both SQLite and PostgreSQL
            regime_rows = conn.execute(
                "SELECT CAST(strftime('%s', timestamp) AS INTEGER), hurst, autocorr, vol_ratio "
                "FROM regime_log WHERE symbol='MNQ' AND timestamp > ? ORDER BY timestamp ASC",
                (cutoff_iso,)
            ).fetchall()
        except Exception as _rl_err:
            logger.debug(f'backtest_setup_i: regime_log path A failed — {_rl_err}')
            try:
                conn.rollback()   # ← critical: clear PG aborted-transaction state
            except Exception:
                pass
            # Retry with PostgreSQL-native EXTRACT — handles TIMESTAMPTZ correctly
            try:
                cutoff_iso = datetime.fromtimestamp(cutoff_ts, tz=timezone.utc).isoformat()
                regime_rows = conn.execute(
                    "SELECT EXTRACT(EPOCH FROM timestamp)::BIGINT, hurst, autocorr, vol_ratio "
                    "FROM regime_log WHERE symbol='MNQ' AND timestamp > ? ORDER BY timestamp ASC",
                    (cutoff_iso,)
                ).fetchall()
            except Exception as _rl_err2:
                logger.debug(f'backtest_setup_i: regime_log path B failed — {_rl_err2}')
                try:
                    conn.rollback()
                except Exception:
                    pass

        if len(regime_rows) >= 10:
            # ── Path A: regime_log has data — simulate signals from stored values ──
            df5 = _load_ohlcv_df(conn, 'MNQ', '5min', lookback_days)
            if len(df5) < 50:
                return None
            atr    = _atr14(df5)
            ts5    = df5['ts'].values
            closes = df5['close'].values
            hours  = df5['hour'].values
            wdays  = df5['weekday'].values
            atr5v  = atr.values
            bars5  = len(df5)
            pnl    = []

            for row in regime_rows:
                row_ts, hurst, autocorr, vol_ratio = row
                if hurst is None or autocorr is None:
                    continue
                if hurst < 0.55:
                    continue
                # Find matching 5min bar
                bar_idx = int(np.searchsorted(ts5, row_ts, side='right')) - 1
                if bar_idx < 1 or bar_idx >= bars5 - 1:
                    continue
                if wdays[bar_idx] not in (1, 2, 3):
                    continue
                if not (13 <= hours[bar_idx] < 19):
                    continue
                atr_val = float(atr5v[bar_idx])
                if not atr_val or math.isnan(atr_val):
                    continue
                e = float(closes[bar_idx])
                if autocorr > 0.05:
                    # momentum long
                    pnl.append(_sim_outcome(df5, bar_idx, 'long',
                                            e, e - atr_val, e + 3.0 * atr_val))
                elif autocorr < -0.05:
                    # momentum short
                    pnl.append(_sim_outcome(df5, bar_idx, 'short',
                                            e, e + atr_val, e - 3.0 * atr_val))

            b = BENCHMARKS['I']
            return _backtest_stats(pnl, b['sharpe'], b['wr'], bars5, 'I', lookback_days)

        # ── Path B: fall back to mathematical calculation ─────────────────
        logger.debug(
            f'backtest_setup_i: regime_log has {len(regime_rows)} rows '
            f'(< 10) — falling back to mathematical Hurst calculation'
        )
        df = _load_ohlcv_df(conn, 'MNQ', '5min', lookback_days)
        if len(df) < 150:
            # MNQ not available locally — try NQ (same price, different multiplier)
            df = _load_ohlcv_df(conn, 'NQ', '5min', lookback_days)
        if len(df) < 150:
            return None
        atr    = _atr14(df)
        closes = df['close'].values
        hours  = df['hour'].values
        wdays  = df['weekday'].values
        bars   = len(df)
        hw     = 100; aw = 50; pnl = []

        def _hurst(x):
            n = len(x)
            if n < 20:
                return 0.5
            m = x.mean(); dv = np.cumsum(x - m)
            r = dv.max() - dv.min(); s = x.std()
            if s == 0:
                return 0.5
            rs = r / s
            return math.log(rs) / math.log(n) if rs > 0 else 0.5

        # Tighter thresholds for Path B to avoid over-generation.
        # The live ML model uses XGB > 0.58 + LR + HTF bias — this mathematical
        # proxy uses stricter Hurst/autocorr/momentum thresholds to approximate
        # the same selectivity (~3-5 signals/week, not 3-5/day).
        HURST_MIN    = 0.60   # live uses XGB > 0.58; Hurst 0.60 is similarly selective
        AUTOCORR_MIN = 0.10   # stronger autocorr required
        MOM_MIN_ATR  = 1.0    # momentum must be > 1× ATR (not just 0.5×)
        MAX_PER_DAY  = 2      # cap at 2 signals per day (live has 2-cap per session)
        day_counts   = {}

        for i in range(hw + aw, bars - 1):
            if wdays[i] not in (1, 2, 3):
                continue
            if not (13 <= hours[i] < 19):
                continue
            atr_val = atr.iloc[i]
            if not atr_val or math.isnan(atr_val):
                continue
            # Per-day signal cap
            day_key = str(df['date'].iloc[i]) if 'date' in df.columns else str(i // 78)
            if day_counts.get(day_key, 0) >= MAX_PER_DAY:
                continue
            log_ret  = np.diff(np.log(closes[i - hw: i + 1]))
            hurst    = _hurst(log_ret)
            if hurst < HURST_MIN:
                continue
            ret50    = np.diff(closes[i - aw: i + 1])
            autocorr = float(np.corrcoef(ret50[:-1], ret50[1:])[0, 1]) if len(ret50) > 5 else 0.0
            if math.isnan(autocorr):
                autocorr = 0.0
            mom = closes[i] - closes[i - 20]
            e   = closes[i]
            if autocorr > AUTOCORR_MIN and mom > MOM_MIN_ATR * atr_val:
                pnl.append(_sim_outcome(df, i, 'long',  e, e - atr_val, e + 3.0 * atr_val))
                day_counts[day_key] = day_counts.get(day_key, 0) + 1
            elif autocorr < -AUTOCORR_MIN and mom < -MOM_MIN_ATR * atr_val:
                pnl.append(_sim_outcome(df, i, 'short', e, e + atr_val, e - 3.0 * atr_val))
                day_counts[day_key] = day_counts.get(day_key, 0) + 1

        b = BENCHMARKS['I']
        return _backtest_stats(pnl, b['sharpe'], b['wr'], len(df), 'I', lookback_days)
    except Exception as e:
        logger.error(f'backtest_setup_i: {e}', exc_info=True)
        return None


BACKTEST_FUNCS = {
    'A': backtest_setup_a, 'B': backtest_setup_b, 'C': backtest_setup_c,
    'D': backtest_setup_d, 'E': backtest_setup_e,
    'H': backtest_setup_h, 'I': backtest_setup_i,
}


# ═══════════════════════════════════════════════════════════════════════════
#  DAILY BACKTEST — each setup uses a fresh connection
# ═══════════════════════════════════════════════════════════════════════════

def run_daily_backtest() -> dict:
    """
    Run all 7 setup backtests and write results to backtest_results.
    Each setup opens and closes its own DB connection.
    Also checks for 3 consecutive days of edge_score < 50 and fires Telegram alert.
    Returns dict with BacktestResult per setup (or None) and '_write_errors' key.
    """
    today   = date.today()
    results = {}
    _write_errors = {}

    for sid, bt_func in BACKTEST_FUNCS.items():
        bt = None
        bt_conn = _conn()
        try:
            bt = bt_func(bt_conn, lookback_days=180)
        except Exception as e:
            logger.error(f'Research Backtest {sid} compute failed: {e}', exc_info=True)
            _rollback(bt_conn)
        finally:
            _close(bt_conn)

        if bt is None:
            logger.warning(f'Research Backtest {sid}: no result (insufficient data or error)')
            results[sid] = None
            continue

        # Write in its own isolated connection
        w_conn = _conn()
        try:
            # All values are Python natives (guaranteed by _backtest_stats + _py())
            w_conn.execute(
                "INSERT INTO backtest_results "
                "(setup_id, lookback_days, run_date, total_signals, win_rate, sharpe, "
                " avg_r, expectancy, max_drawdown, profit_factor, "
                " benchmark_sharpe, benchmark_win_rate, "
                " sharpe_vs_benchmark, wr_vs_benchmark, edge_score, bars_analysed) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    str(bt.setup_id),   int(bt.lookback_days),  str(today.isoformat()),
                    int(bt.total_signals),
                    float(bt.win_rate)  if bt.win_rate  is not None else 0.0,
                    float(bt.sharpe)    if bt.sharpe    is not None else None,
                    float(bt.avg_r),    float(bt.expectancy),
                    float(bt.max_drawdown), float(bt.profit_factor),
                    float(bt.benchmark_sharpe), float(bt.benchmark_win_rate),
                    float(bt.sharpe_vs_benchmark), float(bt.wr_vs_benchmark),
                    int(bt.edge_score), int(bt.bars_analysed),
                )
            )
            w_conn.commit()
            # Verify write succeeded (within same connection before close)
            verify = w_conn.execute(
                "SELECT COUNT(*) FROM backtest_results WHERE setup_id=? AND run_date=?",
                (bt.setup_id, today.isoformat())
            ).fetchone()
            rows_after = verify[0] if verify else 0
            results[sid] = bt
            logger.info(
                f'Research Backtest {sid}: WRITTEN (rows_after_commit={rows_after}) '
                f'signals={bt.total_signals} edge={bt.edge_score} '
                f'sharpe={bt.sharpe} wr={bt.win_rate:.3f} bars={bt.bars_analysed}'
            )
            if rows_after == 0:
                _write_errors[sid] = f'COMMIT_SUCCEEDED_BUT_NO_ROWS (rows_after_commit=0)'
        except Exception as e:
            _rollback(w_conn)
            err_msg = f'{type(e).__name__}: {e}'
            logger.error(f'Research Backtest {sid} write failed: {err_msg}')
            _write_errors[sid] = err_msg
            results[sid] = bt  # return in-memory result even when write fails
        finally:
            _close(w_conn)

    results['_write_errors'] = _write_errors
    _check_edge_degradation()
    return results


def _check_edge_degradation():
    """Log degradation warning if any setup has edge_score < 50 for 3 consecutive days.
    Dashboard Research Health tab is the destination — no Telegram from research functions."""
    c = _conn()
    try:
        for sid in BACKTEST_FUNCS:
            rows = c.execute(
                "SELECT edge_score, sharpe, run_date FROM backtest_results "
                "WHERE setup_id=? ORDER BY run_date DESC LIMIT 3",
                (sid,)
            ).fetchall()
            if len(rows) < 3:
                continue
            scores = [r[0] for r in rows]
            if all(s is not None and s < 50 for s in scores):
                latest_sharpe = rows[0][1] or 0
                bench_s = BENCHMARKS.get(sid, {}).get('sharpe', 0)
                pct = ((latest_sharpe / bench_s) - 1) * 100 if bench_s else 0
                logger.info(
                    f'Research: edge degradation — Setup {sid} ({SETUP_NAMES.get(sid, sid)}) '
                    f'edge degrading; Backtest Sharpe: {latest_sharpe:.2f} (benchmark {bench_s}) '
                    f'— {pct:.0f}% {"above" if pct >= 0 else "below"} benchmark; '
                    f'3 consecutive days below threshold; Action required: review parameters '
                    f'— dashboard only'
                )
    except Exception as e:
        logger.error(f'_check_edge_degradation: {e}')
    finally:
        _close(c)


# ═══════════════════════════════════════════════════════════════════════════
#  HEALTH SCORING (dual: backtest 60% + live 40%)
# ═══════════════════════════════════════════════════════════════════════════

def _sharpe(pnl_r_list):
    n = len(pnl_r_list)
    if n < 5: return None
    mean = sum(pnl_r_list) / n
    var  = sum((v - mean) ** 2 for v in pnl_r_list) / (n - 1) if n > 1 else 0
    std  = math.sqrt(var) if var > 0 else 0
    return (mean / std * math.sqrt(252)) if std > 0 else None


def _calc_live_score(pnl_list: list, bench: dict) -> int:
    wins     = sum(1 for v in pnl_list if v > 0)
    win_rate = wins / len(pnl_list)
    exp      = sum(pnl_list) / len(pnl_list)
    sharpe   = _sharpe(pnl_list)
    score    = 100
    if sharpe is not None and bench['sharpe'] > 0:
        ratio = sharpe / bench['sharpe']
        if ratio < 0.5:    score -= 20
        elif ratio > 1.1:  score += 5
    if (win_rate - bench['wr']) < -0.10: score -= 15
    if exp < 0: score -= 10
    return max(0, min(100, score))


def calculate_health_score(setup_id: str) -> dict:
    """
    Dual-scored health check: 60% backtest (reads backtest_results) + 40% live (reads apex_trades).
    Falls back to 100% backtest score if fewer than 5 live closed trades exist.
    Opens its own connection; always closes it.
    """
    sid   = setup_id.upper()
    bench = BENCHMARKS.get(sid)
    if not bench:
        return {'setup_id': sid, 'health_score': None, 'alert_level': 'INSUFFICIENT_DATA'}

    c = _conn()
    try:
        now_utc = datetime.now(timezone.utc)
        cut_28d = (now_utc - timedelta(days=28)).isoformat()

        rows_20 = c.execute(
            "SELECT pnl_r FROM apex_trades "
            "WHERE UPPER(SUBSTR(setup, 1, 1)) = ? "
            "  AND status = 'closed' AND pnl_r IS NOT NULL "
            "ORDER BY exit_time DESC LIMIT 20",
            (sid,)
        ).fetchall()

        bt_row = c.execute(
            "SELECT edge_score, sharpe, win_rate, bars_analysed, total_signals, "
            "       sharpe_vs_benchmark, wr_vs_benchmark "
            "FROM backtest_results WHERE setup_id=? ORDER BY id DESC LIMIT 1",
            (sid,)
        ).fetchone()

        backtest_score   = bt_row[0] if bt_row else None
        bt_sharpe        = bt_row[1] if bt_row else None
        bt_win_rate      = bt_row[2] if bt_row else None
        bars_analysed    = bt_row[3] if bt_row else 0
        bt_total_signals = bt_row[4] if bt_row else 0

        live_count  = len(rows_20)
        live_score  = None
        sharpe_30d  = None
        win_rate_l  = None
        expectancy  = None
        signal_count_week = None

        if live_count >= 5:
            pnl_l      = [r[0] for r in rows_20]
            live_score = _calc_live_score(pnl_l, bench)
            wins       = sum(1 for v in pnl_l if v > 0)
            win_rate_l = wins / len(pnl_l)
            expectancy = sum(pnl_l) / len(pnl_l)
            sharpe_30d = _sharpe(pnl_l)
            count_28d  = c.execute(
                "SELECT COUNT(*) FROM apex_trades "
                "WHERE UPPER(SUBSTR(setup, 1, 1)) = ? "
                "  AND status = 'closed' AND exit_time >= ?",
                (sid, cut_28d)
            ).fetchone()[0]
            signal_count_week = count_28d / 4.0

        if backtest_score is not None and live_score is not None:
            health_score = int(round(0.6 * backtest_score + 0.4 * live_score))
            score_basis  = 'dual'
        elif backtest_score is not None:
            health_score = backtest_score
            score_basis  = 'backtest_only'
        elif live_score is not None:
            health_score = live_score
            score_basis  = 'live_only'
        else:
            return {
                'setup_id': sid, 'health_score': None, 'alert_level': 'INSUFFICIENT_DATA',
                'sharpe_30d': None, 'sharpe_benchmark': bench['sharpe'],
                'win_rate': None, 'win_rate_benchmark': bench['wr'],
                'signal_count_week': None, 'expectancy': None,
                'backtest_score': None, 'live_score': None,
                'bars_analysed': 0, 'live_trade_count': live_count,
            }

        alert_level = ('HEALTHY' if health_score > 75 else
                       'WATCH'   if health_score >= 50 else 'ALERT')

        return {
            'setup_id':          sid,
            'sharpe_30d':        round(sharpe_30d, 3) if sharpe_30d else None,
            'sharpe_benchmark':  bench['sharpe'],
            'win_rate':          round(win_rate_l, 4)  if win_rate_l  is not None else None,
            'win_rate_benchmark':bench['wr'],
            'signal_count_week': round(signal_count_week, 2) if signal_count_week else None,
            'expectancy':        round(expectancy, 4)  if expectancy  is not None else None,
            'health_score':      health_score,
            'alert_level':       alert_level,
            'backtest_score':    backtest_score,
            'live_score':        live_score,
            'bars_analysed':     bars_analysed,
            'live_trade_count':  live_count,
            'bt_sharpe':         round(bt_sharpe, 3)   if bt_sharpe   else None,
            'bt_win_rate':       round(bt_win_rate, 4) if bt_win_rate  else None,
            'bt_total_signals':  bt_total_signals,
            'score_basis':       score_basis,
        }
    except Exception as e:
        logger.error(f'calculate_health_score {sid}: {e}', exc_info=True)
        return {
            'setup_id':            sid,
            'health_score':        None,
            'alert_level':         'INSUFFICIENT_DATA',
            'backtest_score':      None,
            'live_score':          None,
            'sharpe_benchmark':    None,
            'win_rate_benchmark':  None,
            'win_rate':            None,
            'signal_count_week':   None,
            'expectancy':          None,
            'bars_analysed':       0,
            'live_trade_count':    0,
            'bt_sharpe':           None,
            'bt_win_rate':         None,
            'bt_total_signals':    0,
            'score_basis':         'error',
        }
    finally:
        _close(c)


# ═══════════════════════════════════════════════════════════════════════════
#  WEEKLY HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════

def run_weekly_health_check() -> dict:
    """
    Score all 7 setups and write results to strategy_health_log.
    Each setup uses calculate_health_score() which manages its own connection.
    The write to strategy_health_log uses a separate fresh connection per setup.
    """
    setups = ['A', 'B', 'C', 'D', 'E', 'H', 'I']
    today  = date.today()
    monday = today - timedelta(days=today.weekday())
    results = {}

    for sid in setups:
        try:
            metrics = calculate_health_score(sid)
        except Exception as e:
            logger.error(f'Research Health {sid} calculate failed: {e}', exc_info=True)
            results[sid] = {'error': str(e)}
            continue

        c = _conn()
        try:
            prev = c.execute(
                "SELECT health_score FROM strategy_health_log "
                "WHERE setup_id = ? AND health_score IS NOT NULL "
                "ORDER BY week_start DESC LIMIT 3",
                (sid,)
            ).fetchall()
            notes = None
            if len(prev) >= 2:
                scores = [r[0] for r in prev]
                if all(scores[i] > scores[i + 1] for i in range(len(scores) - 1)):
                    notes = f'Declining {len(scores) + 1} consecutive weeks'

            # Try with new dual-score columns first, fall back if they don't exist yet
            try:
                c.execute(
                    "INSERT INTO strategy_health_log "
                    "(setup_id, week_start, sharpe_30d, sharpe_benchmark, win_rate, "
                    " win_rate_benchmark, signal_count_week, expectancy, health_score, "
                    " alert_level, notes, backtest_score, live_score) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        sid, monday.isoformat(),
                        metrics.get('sharpe_30d'),      metrics.get('sharpe_benchmark'),
                        metrics.get('win_rate'),         metrics.get('win_rate_benchmark'),
                        metrics.get('signal_count_week'),metrics.get('expectancy'),
                        metrics.get('health_score'),     metrics.get('alert_level'),
                        notes, metrics.get('backtest_score'), metrics.get('live_score'),
                    )
                )
            except Exception as col_err:
                _rollback(c)
                c.execute(
                    "INSERT INTO strategy_health_log "
                    "(setup_id, week_start, sharpe_30d, sharpe_benchmark, win_rate, "
                    " win_rate_benchmark, signal_count_week, expectancy, health_score, "
                    " alert_level, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        sid, monday.isoformat(),
                        metrics.get('sharpe_30d'),      metrics.get('sharpe_benchmark'),
                        metrics.get('win_rate'),         metrics.get('win_rate_benchmark'),
                        metrics.get('signal_count_week'),metrics.get('expectancy'),
                        metrics.get('health_score'),     metrics.get('alert_level'), notes,
                    )
                )
                logger.warning(f'Research Health {sid}: wrote without dual columns ({col_err})')

            c.commit()
            results[sid] = metrics
            logger.info(
                f'Research Health {sid}: score={metrics.get("health_score")} '
                f'alert={metrics.get("alert_level")} basis={metrics.get("score_basis")}'
            )
        except Exception as e:
            _rollback(c)
            logger.error(f'Research Health {sid} write failed: {e}', exc_info=True)
            results[sid] = {'error': str(e)}
        finally:
            _close(c)

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  SHADOW LAB
# ═══════════════════════════════════════════════════════════════════════════

def seed_shadow_lab_candidates() -> None:
    """
    Seed initial shadow lab candidates if the table is empty.
    Idempotent — safe to call on every server startup.
    Never raises — any failure is logged and startup continues.
    Uses fresh connection per operation so failures cannot contaminate other init work.
    """
    # Step 1: ensure schema migrations are applied (separate connections per DDL)
    _ensure_research_schema()

    # Step 2: check if already seeded
    c = _conn()
    try:
        count = c.execute("SELECT COUNT(*) FROM shadow_lab").fetchone()[0]
    except Exception as e:
        _rollback(c)
        logger.error(f'Research Division: shadow lab count check failed — {e}', exc_info=True)
        return
    finally:
        _close(c)

    if count > 0:
        logger.info(f'Research Division: shadow lab already seeded ({count} candidates)')
        _ensure_gap_orb_in_shadow_lab()
        _ensure_k_es_in_shadow_lab()
        _ensure_k_mnq_in_shadow_lab()
        return

    # Step 3: insert candidates
    today = date.today()
    candidates = [
        {
            'name': 'Post-Low-Vol Expansion',
            'desc': ('Trades expansion moves following compressed volatility periods. '
                     'Enters when ATR contracts below 20-day average then expands.'),
            'sharpe': 29.91, 'wr': 0.68,
        },
        {
            'name': 'Monday NY Open Long',
            'desc': ('Exploits Monday NY session upside bias on MNQ. Long only, '
                     'first 30 minutes of NY open, requires bullish HTF bias.'),
            'sharpe': 11.43, 'wr': 0.61,
        },
        {
            'name': 'Value Area Continuation',
            'desc': ('Trades continuation moves from previous session value area high/low. '
                     'Enters on retest of value area boundary with momentum confirmation.'),
            'sharpe': 9.99, 'wr': 0.58,
        },
    ]
    c = _conn()
    try:
        for cand in candidates:
            promo = (today + timedelta(weeks=8)).isoformat()
            c.execute(
                "INSERT INTO shadow_lab "
                "(strategy_name, description, entered_date, week_number, total_weeks, "
                " backtest_sharpe, backtest_win_rate, status, promotion_eligible_date) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (cand['name'], cand['desc'], today.isoformat(), 0, 8,
                 cand['sharpe'], cand['wr'], 'ACTIVE', promo)
            )
        c.commit()
        logger.info(f'Research Division: seeded {len(candidates)} shadow lab candidates')
    except Exception as e:
        _rollback(c)
        logger.error(f'Research Division: shadow lab seed failed — {e}', exc_info=True)
    finally:
        _close(c)

    _ensure_gap_orb_in_shadow_lab()
    _ensure_k_es_in_shadow_lab()
    _ensure_k_mnq_in_shadow_lab()


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 3 — DISCOVERY ENGINE
# ═══════════════════════════════════════════════════════════════════════════

# Thresholds for promoting a hypothesis to SIGNIFICANT status
SIGNIFICANCE_THRESHOLDS = {
    'min_signals': 30,
    'min_ic': 0.05,
    'min_sharpe': 3.0,
    'max_p_value': 0.05,
    'min_win_rate': 0.52,
}


def _p_value_approx(ic: float, n: int) -> float:
    """
    Approximate two-tailed p-value for a Pearson IC using the t-distribution.
    Uses math.erf for the normal CDF approximation (no scipy).
    t = IC * sqrt(n-2) / sqrt(1 - IC^2)
    For large df the t-distribution ≈ normal, so p ≈ 2 * (1 - Phi(|t|)).
    """
    try:
        if n < 4 or abs(ic) >= 1.0:
            return 1.0
        t = ic * math.sqrt(n - 2) / math.sqrt(max(1e-12, 1 - ic ** 2))
        # Normal CDF via erf: Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
        p_one_tail = 0.5 * (1 - math.erf(abs(t) / math.sqrt(2)))
        return min(1.0, 2 * p_one_tail)
    except Exception:
        return 1.0


def _hypothesis_sharpe(forward_returns, signal_mask) -> Optional[float]:
    """Annualised Sharpe for signal-triggered forward returns."""
    try:
        vals = [float(r) for r, m in zip(forward_returns, signal_mask) if m and r is not None]
        if len(vals) < 5:
            return None
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / max(1, n - 1)
        std = math.sqrt(var) if var > 0 else 0
        return float(mean / std * math.sqrt(252)) if std > 0 else None
    except Exception:
        return None


def _ic_from_lists(x_vals, y_vals) -> float:
    """Pearson IC between two equal-length lists. Returns 0.0 on failure."""
    try:
        n = len(x_vals)
        if n < 5:
            return 0.0
        mx = sum(x_vals) / n
        my = sum(y_vals) / n
        cov = sum((a - mx) * (b - my) for a, b in zip(x_vals, y_vals)) / n
        sx = math.sqrt(sum((a - mx) ** 2 for a in x_vals) / n)
        sy = math.sqrt(sum((b - my) ** 2 for b in y_vals) / n)
        if sx * sy < 1e-12:
            return 0.0
        return float(cov / (sx * sy))
    except Exception:
        return 0.0


def _hurst_rs(series_vals: list) -> float:
    """R/S Hurst exponent on a list of prices. Returns 0.5 (random walk) on failure."""
    try:
        n = len(series_vals)
        if n < 20:
            return 0.5
        import math as _m
        mean = sum(series_vals) / n
        deviations = [v - mean for v in series_vals]
        cumdev = []
        cum = 0.0
        for d in deviations:
            cum += d
            cumdev.append(cum)
        r = max(cumdev) - min(cumdev)
        s = _m.sqrt(sum((v - mean) ** 2 for v in series_vals) / n)
        if s == 0 or r <= 0:
            return 0.5
        return float(_m.log(r / s) / _m.log(n))
    except Exception:
        return 0.5


def _write_hypothesis(result: dict) -> None:
    """Write/upsert one hypothesis result to hypothesis_log table."""
    c = _conn()
    try:
        if _db.IS_POSTGRES:
            c.execute(
                """INSERT INTO hypothesis_log
                   (hypothesis_id, description, category, instrument, lookback_days,
                    signals_generated, win_rate, sharpe, avg_r, information_coefficient,
                    p_value, status, run_date)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT (hypothesis_id) DO UPDATE SET
                     description=EXCLUDED.description,
                     signals_generated=EXCLUDED.signals_generated,
                     win_rate=EXCLUDED.win_rate,
                     sharpe=EXCLUDED.sharpe,
                     avg_r=EXCLUDED.avg_r,
                     information_coefficient=EXCLUDED.information_coefficient,
                     p_value=EXCLUDED.p_value,
                     status=EXCLUDED.status,
                     run_date=EXCLUDED.run_date""",
                (result['hypothesis_id'], result['description'], result['category'],
                 result['instrument'], result['lookback_days'], result['signals_generated'],
                 result['win_rate'], result['sharpe'], result['avg_r'],
                 result['information_coefficient'], result['p_value'],
                 result['status'], result['run_date'])
            )
        else:
            c.execute(
                """INSERT OR REPLACE INTO hypothesis_log
                   (hypothesis_id, description, category, instrument, lookback_days,
                    signals_generated, win_rate, sharpe, avg_r, information_coefficient,
                    p_value, status, run_date)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (result['hypothesis_id'], result['description'], result['category'],
                 result['instrument'], result['lookback_days'], result['signals_generated'],
                 result['win_rate'], result['sharpe'], result['avg_r'],
                 result['information_coefficient'], result['p_value'],
                 result['status'], result['run_date'])
            )
        c.commit()
    except Exception as e:
        _rollback(c)
        logger.warning(f'_write_hypothesis {result.get("hypothesis_id")}: {e}')
    finally:
        _close(c)


def _score_hypothesis(signals: int, ic: float, sharpe: Optional[float],
                      p_value: float, win_rate: float) -> str:
    """Return SIGNIFICANT, TESTING, or REJECTED."""
    t = SIGNIFICANCE_THRESHOLDS
    if (signals >= t['min_signals']
            and abs(ic) >= t['min_ic']
            and (sharpe is not None and sharpe >= t['min_sharpe'])
            and p_value <= t['max_p_value']
            and win_rate >= t['min_win_rate']):
        return 'SIGNIFICANT'
    if signals < 5:
        return 'REJECTED'
    return 'TESTING'


def _promote_significant_to_shadow(hypothesis_id: str, description: str,
                                   sharpe: float, win_rate: float) -> None:
    """Auto-create shadow lab entry for a SIGNIFICANT hypothesis."""
    try:
        c = _conn()
        try:
            # Check if already in shadow lab (by name match)
            existing = c.execute(
                "SELECT id FROM shadow_lab WHERE strategy_name = ?",
                (hypothesis_id,)
            ).fetchone()
            if existing:
                _close(c)
                return
            today = date.today()
            promo = (today + timedelta(weeks=8)).isoformat()
            c.execute(
                "INSERT INTO shadow_lab "
                "(strategy_name, description, entered_date, week_number, total_weeks, "
                " backtest_sharpe, backtest_win_rate, status, promotion_eligible_date) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (hypothesis_id, description or '', today.isoformat(), 0, 8,
                 float(sharpe) if sharpe else 0.0,
                 float(win_rate) if win_rate else 0.0,
                 'ACTIVE', promo)
            )
            c.commit()
            logger.info(f'Hypothesis Engine: auto-promoted {hypothesis_id} to shadow lab')
        except Exception as e:
            _rollback(c)
            logger.warning(f'_promote_significant_to_shadow: {e}')
        finally:
            _close(c)
    except Exception as e:
        logger.warning(f'_promote_significant_to_shadow outer: {e}')


# ─── Hypothesis category implementations ────────────────────────────────────

def _test_autocorr_hypotheses(conn) -> list:
    """
    Category 1: Autocorrelation pattern hypotheses (4 variants).
    Tests whether high autocorr + low vol ratio predicts trending next session.
    """
    results = []
    try:
        import pandas as pd
        df = _load_ohlcv_df(conn, 'MNQ', '5min', 180)
        if len(df) < 200:
            return results
        atr = _atr14(df)
        atr20 = atr.rolling(20).mean()

        close = df['close']
        for ac_thresh, vol_thresh, variant in [
            (0.05, 0.7, 'v1'), (0.10, 0.8, 'v2'),
            (0.15, 0.9, 'v3'), (0.10, 0.7, 'v4'),
        ]:
            hyp_id = f'autocorr_{ac_thresh}_{vol_thresh}'
            try:
                # Rolling lag-1 autocorrelation on 20-bar window
                autocorr = close.rolling(20).apply(
                    lambda x: x.autocorr(lag=1) if len(x) > 2 else 0.0, raw=False
                )
                vol_ratio = atr / atr20.replace(0, float('nan'))

                # Signal: autocorr > threshold AND vol_ratio < vol_thresh for 3+ consecutive bars
                ac_cond = (autocorr > ac_thresh).astype(int)
                vr_cond = (vol_ratio < vol_thresh).fillna(False).astype(int)
                both = (ac_cond & vr_cond).astype(int)
                consec = both.rolling(3).sum() >= 3
                signal_idx = consec[consec].index.tolist()

                # Forward return: 12-bar move magnitude relative to ATR
                n_fwd = 12
                fwd_dir = []
                actual_dir = []
                for idx in signal_idx:
                    loc = df.index.get_loc(idx)
                    if loc + n_fwd >= len(df):
                        continue
                    entry_close = float(df['close'].iloc[loc])
                    fwd_close = float(df['close'].iloc[loc + n_fwd])
                    atr_val = float(atr.iloc[loc])
                    if atr_val <= 0:
                        continue
                    move = fwd_close - entry_close
                    # Trending = absolute move > 1 ATR
                    is_trending = abs(move) > atr_val
                    fwd_dir.append(float(move / atr_val))  # normalised return
                    actual_dir.append(1.0 if is_trending else 0.0)

                n = len(fwd_dir)
                if n < 5:
                    results.append({
                        'hypothesis_id': hyp_id,
                        'description': f'Autocorr>{ac_thresh} vol_ratio<{vol_thresh} 3-bar: next session trending',
                        'category': 'autocorrelation',
                        'instrument': 'MNQ',
                        'lookback_days': 180,
                        'signals_generated': n,
                        'win_rate': 0.0, 'sharpe': None, 'avg_r': 0.0,
                        'information_coefficient': 0.0, 'p_value': 1.0,
                        'status': 'REJECTED',
                        'run_date': date.today().isoformat(),
                    })
                    continue

                wins = sum(1 for v in actual_dir if v > 0)
                wr = wins / n
                avg_r = sum(fwd_dir) / n
                ic = _ic_from_lists(fwd_dir, actual_dir)
                p = _p_value_approx(ic, n)
                sharpe = _hypothesis_sharpe(fwd_dir, [True] * n)
                status = _score_hypothesis(n, ic, sharpe, p, wr)

                result = {
                    'hypothesis_id': hyp_id,
                    'description': f'Autocorr>{ac_thresh} + vol_ratio<{vol_thresh} 3-bar: next session trending (variant {variant})',
                    'category': 'autocorrelation',
                    'instrument': 'MNQ',
                    'lookback_days': 180,
                    'signals_generated': n,
                    'win_rate': round(float(wr), 4),
                    'sharpe': round(float(sharpe), 3) if sharpe is not None else None,
                    'avg_r': round(float(avg_r), 4),
                    'information_coefficient': round(float(ic), 4),
                    'p_value': round(float(p), 4),
                    'status': status,
                    'run_date': date.today().isoformat(),
                }
                results.append(result)
                if status == 'SIGNIFICANT':
                    _promote_significant_to_shadow(hyp_id, result['description'],
                                                   sharpe, wr)
            except Exception as e:
                logger.warning(f'autocorr hypothesis {hyp_id}: {e}')
    except Exception as e:
        logger.warning(f'_test_autocorr_hypotheses: {e}')
    return results


def _test_hurst_hypotheses(conn) -> list:
    """Category 2: Hurst exponent crossing hypotheses (3 variants)."""
    results = []
    try:
        import pandas as pd
        df = _load_ohlcv_df(conn, 'MNQ', '5min', 180)
        if len(df) < 200:
            return results

        close_vals = df['close'].values
        n_bars = len(close_vals)
        window = 100
        n_fwd = 10

        for hurst_thresh, variant_num in [(0.55, 1), (0.60, 2), (0.65, 3)]:
            hyp_id = f'hurst_cross_{hurst_thresh}'
            try:
                # Rolling Hurst on 100-bar window
                hurst_vals = []
                for i in range(n_bars):
                    if i < window:
                        hurst_vals.append(0.5)
                    else:
                        hurst_vals.append(_hurst_rs(list(close_vals[i - window:i])))

                fwd_ac_vals = []    # forward 10-bar autocorrelation
                indicator_vals = []  # 1 if hurst crossed above threshold, else 0

                for i in range(window + 1, n_bars - n_fwd):
                    prev_h = hurst_vals[i - 1]
                    curr_h = hurst_vals[i]
                    if curr_h > hurst_thresh and prev_h <= hurst_thresh:
                        # Hurst crossing above threshold
                        fwd_slice = list(close_vals[i:i + n_fwd])
                        if len(fwd_slice) == n_fwd:
                            # Positive autocorrelation in next 10 bars
                            fwd_ac = _hurst_rs(fwd_slice) - 0.5  # normalised
                            fwd_ac_vals.append(float(fwd_ac))
                            indicator_vals.append(float(curr_h))

                n = len(fwd_ac_vals)
                if n < 5:
                    results.append({
                        'hypothesis_id': hyp_id,
                        'description': f'Hurst cross>{hurst_thresh}: next 10 bars pos autocorr',
                        'category': 'hurst',
                        'instrument': 'MNQ',
                        'lookback_days': 180,
                        'signals_generated': n,
                        'win_rate': 0.0, 'sharpe': None, 'avg_r': 0.0,
                        'information_coefficient': 0.0, 'p_value': 1.0,
                        'status': 'REJECTED',
                        'run_date': date.today().isoformat(),
                    })
                    continue

                wins = sum(1 for v in fwd_ac_vals if v > 0)
                wr = wins / n
                avg_r = sum(fwd_ac_vals) / n
                ic = _ic_from_lists(indicator_vals, fwd_ac_vals)
                p = _p_value_approx(ic, n)
                sharpe = _hypothesis_sharpe(fwd_ac_vals, [True] * n)
                status = _score_hypothesis(n, ic, sharpe, p, wr)

                result = {
                    'hypothesis_id': hyp_id,
                    'description': f'Hurst crosses >{hurst_thresh}: next 10 bars positive autocorrelation (variant {variant_num})',
                    'category': 'hurst',
                    'instrument': 'MNQ',
                    'lookback_days': 180,
                    'signals_generated': n,
                    'win_rate': round(float(wr), 4),
                    'sharpe': round(float(sharpe), 3) if sharpe is not None else None,
                    'avg_r': round(float(avg_r), 4),
                    'information_coefficient': round(float(ic), 4),
                    'p_value': round(float(p), 4),
                    'status': status,
                    'run_date': date.today().isoformat(),
                }
                results.append(result)
                if status == 'SIGNIFICANT':
                    _promote_significant_to_shadow(hyp_id, result['description'],
                                                   sharpe, wr)
            except Exception as e:
                logger.warning(f'hurst hypothesis {hyp_id}: {e}')
    except Exception as e:
        logger.warning(f'_test_hurst_hypotheses: {e}')
    return results


def _test_session_timing_hypotheses(conn) -> list:
    """Category 3: Session timing hypotheses (4 variants)."""
    results = []
    try:
        import pandas as pd
        df = _load_ohlcv_df(conn, 'MNQ', '5min', 180)
        if len(df) < 200:
            return results
        atr = _atr14(df)

        hypotheses = [
            ('session_ny_open_15min', 'First 15min NY session (13:00-13:15 UTC) momentum win rate'),
            ('session_monday_bullish', 'Monday sessions: bullish bias vs Tue-Fri average'),
            ('session_post_large_range_reversion', 'After >2% session range: next session mean-reverts'),
            ('session_ny_close_reversion', 'Last 15min NY session (18:45-19:00 UTC): mean-reversion bias'),
        ]

        # Hypothesis A: First 15min NY momentum
        try:
            hyp_id, desc = hypotheses[0]
            signal_rets = []
            for i in range(20, len(df) - 3):
                h = df['hour'].iloc[i]
                m = df['dt'].iloc[i].minute if 'dt' in df.columns else 0
                wday = df['weekday'].iloc[i]
                if wday >= 5 or h != 13 or m > 15:
                    continue
                atr_val = float(atr.iloc[i]) if not math.isnan(float(atr.iloc[i] or 0)) else 0
                if atr_val <= 0:
                    continue
                bar_close = float(df['close'].iloc[i])
                bar_open = float(df['open'].iloc[i])
                direction = 1.0 if bar_close > bar_open else -1.0
                fwd_ret = float(df['close'].iloc[i + 3] - bar_close) * direction / atr_val
                signal_rets.append(fwd_ret)
            n = len(signal_rets)
            wins = sum(1 for v in signal_rets if v > 0)
            wr = wins / n if n > 0 else 0.0
            avg_r = sum(signal_rets) / n if n > 0 else 0.0
            ic = _ic_from_lists(signal_rets, [1.0] * n) if n > 0 else 0.0
            p = _p_value_approx(ic, n)
            sharpe = _hypothesis_sharpe(signal_rets, [True] * n)
            status = _score_hypothesis(n, ic, sharpe, p, wr)
            results.append({
                'hypothesis_id': hyp_id,
                'description': desc,
                'category': 'session_timing',
                'instrument': 'MNQ',
                'lookback_days': 180,
                'signals_generated': n,
                'win_rate': round(float(wr), 4),
                'sharpe': round(float(sharpe), 3) if sharpe is not None else None,
                'avg_r': round(float(avg_r), 4),
                'information_coefficient': round(float(ic), 4),
                'p_value': round(float(p), 4),
                'status': status,
                'run_date': date.today().isoformat(),
            })
        except Exception as e:
            logger.warning(f'session timing hyp A: {e}')

        # Hypothesis B: Monday bullish bias
        try:
            hyp_id, desc = hypotheses[1]
            mon_rets = []
            other_rets = []
            for i in range(20, len(df) - 12):
                h = df['hour'].iloc[i]
                wday = df['weekday'].iloc[i]
                if not (13 <= h < 19):
                    continue
                atr_val = float(atr.iloc[i] or 0)
                if atr_val <= 0:
                    continue
                fwd = (float(df['close'].iloc[i + 12]) - float(df['close'].iloc[i])) / atr_val
                if wday == 0:
                    mon_rets.append(fwd)
                elif 1 <= wday <= 4:
                    other_rets.append(fwd)
            # IC: 1 for Monday, 0 for other days — correlation with fwd return
            all_x = [1.0] * len(mon_rets) + [0.0] * len(other_rets)
            all_y = mon_rets + other_rets
            n = len(all_y)
            wins = sum(1 for v in mon_rets if v > 0)
            wr = wins / len(mon_rets) if mon_rets else 0.0
            avg_r = sum(mon_rets) / len(mon_rets) if mon_rets else 0.0
            ic = _ic_from_lists(all_x, all_y)
            p = _p_value_approx(ic, n)
            sharpe = _hypothesis_sharpe(mon_rets, [True] * len(mon_rets))
            status = _score_hypothesis(len(mon_rets), ic, sharpe, p, wr)
            results.append({
                'hypothesis_id': hyp_id,
                'description': desc,
                'category': 'session_timing',
                'instrument': 'MNQ',
                'lookback_days': 180,
                'signals_generated': len(mon_rets),
                'win_rate': round(float(wr), 4),
                'sharpe': round(float(sharpe), 3) if sharpe is not None else None,
                'avg_r': round(float(avg_r), 4),
                'information_coefficient': round(float(ic), 4),
                'p_value': round(float(p), 4),
                'status': status,
                'run_date': date.today().isoformat(),
            })
        except Exception as e:
            logger.warning(f'session timing hyp B: {e}')

        # Hypothesis C: Post large-range reversion
        try:
            hyp_id, desc = hypotheses[2]
            signal_rets = []
            dates = df['date'].unique() if 'date' in df.columns else []
            for i, d in enumerate(dates[1:], 1):
                prev_day_mask = df['date'] == dates[i - 1]
                curr_day_mask = df['date'] == d
                prev_day = df[prev_day_mask & (df['hour'] >= 13) & (df['hour'] < 19)]
                curr_day = df[curr_day_mask & (df['hour'] >= 13) & (df['hour'] < 19)]
                if len(prev_day) < 5 or len(curr_day) < 5:
                    continue
                prev_open = float(prev_day['close'].iloc[0])
                prev_close = float(prev_day['close'].iloc[-1])
                if prev_open <= 0:
                    continue
                prev_range_pct = abs(prev_close - prev_open) / prev_open
                if prev_range_pct <= 0.02:
                    continue
                # Reversion: expect next day to move opposite direction
                direction = -1.0 if prev_close > prev_open else 1.0
                curr_open = float(curr_day['close'].iloc[0])
                curr_close = float(curr_day['close'].iloc[-1])
                atr_val = float(atr.iloc[curr_day.index[-1]] or 1)
                fwd_ret = (curr_close - curr_open) * direction / atr_val
                signal_rets.append(fwd_ret)
            n = len(signal_rets)
            wins = sum(1 for v in signal_rets if v > 0)
            wr = wins / n if n > 0 else 0.0
            avg_r = sum(signal_rets) / n if n > 0 else 0.0
            ic = _ic_from_lists(signal_rets, [1.0] * n) if n > 0 else 0.0
            p = _p_value_approx(ic, n)
            sharpe = _hypothesis_sharpe(signal_rets, [True] * n)
            status = _score_hypothesis(n, ic, sharpe, p, wr)
            results.append({
                'hypothesis_id': hyp_id,
                'description': desc,
                'category': 'session_timing',
                'instrument': 'MNQ',
                'lookback_days': 180,
                'signals_generated': n,
                'win_rate': round(float(wr), 4),
                'sharpe': round(float(sharpe), 3) if sharpe is not None else None,
                'avg_r': round(float(avg_r), 4),
                'information_coefficient': round(float(ic), 4),
                'p_value': round(float(p), 4),
                'status': status,
                'run_date': date.today().isoformat(),
            })
        except Exception as e:
            logger.warning(f'session timing hyp C: {e}')

        # Hypothesis D: Last 15min NY reversion
        try:
            hyp_id, desc = hypotheses[3]
            signal_rets = []
            for i in range(20, len(df) - 3):
                h = df['hour'].iloc[i]
                m = df['dt'].iloc[i].minute if 'dt' in df.columns else 0
                wday = df['weekday'].iloc[i]
                if wday >= 5 or h != 18 or m < 45:
                    continue
                atr_val = float(atr.iloc[i] or 0)
                if atr_val <= 0:
                    continue
                bar_close = float(df['close'].iloc[i])
                bar_open = float(df['open'].iloc[i])
                # Mean-reversion: trade opposite to bar direction
                direction = -1.0 if bar_close > bar_open else 1.0
                fwd_ret = float(df['close'].iloc[i + 3] - bar_close) * direction / atr_val
                signal_rets.append(fwd_ret)
            n = len(signal_rets)
            wins = sum(1 for v in signal_rets if v > 0)
            wr = wins / n if n > 0 else 0.0
            avg_r = sum(signal_rets) / n if n > 0 else 0.0
            ic = _ic_from_lists(signal_rets, [1.0] * n) if n > 0 else 0.0
            p = _p_value_approx(ic, n)
            sharpe = _hypothesis_sharpe(signal_rets, [True] * n)
            status = _score_hypothesis(n, ic, sharpe, p, wr)
            results.append({
                'hypothesis_id': hyp_id,
                'description': desc,
                'category': 'session_timing',
                'instrument': 'MNQ',
                'lookback_days': 180,
                'signals_generated': n,
                'win_rate': round(float(wr), 4),
                'sharpe': round(float(sharpe), 3) if sharpe is not None else None,
                'avg_r': round(float(avg_r), 4),
                'information_coefficient': round(float(ic), 4),
                'p_value': round(float(p), 4),
                'status': status,
                'run_date': date.today().isoformat(),
            })
        except Exception as e:
            logger.warning(f'session timing hyp D: {e}')

    except Exception as e:
        logger.warning(f'_test_session_timing_hypotheses: {e}')
    return results


def _test_volatility_regime_hypotheses(conn) -> list:
    """Category 4: Volatility regime transition hypotheses (4 variants: N = 5,10,20,40)."""
    results = []
    try:
        import pandas as pd
        df = _load_ohlcv_df(conn, 'MNQ', '5min', 180)
        if len(df) < 200:
            return results
        atr = _atr14(df)
        atr20avg = atr.rolling(20).mean()

        for n_fwd in [5, 10, 20, 40]:
            hyp_id = f'vol_regime_expansion_n{n_fwd}'
            try:
                signal_rets = []
                bars = len(df)
                for i in range(30, bars - n_fwd - 1):
                    atr_val = float(atr.iloc[i] or 0)
                    atr20_val = float(atr20avg.iloc[i] or 0)
                    if atr20_val <= 0 or atr_val <= 0:
                        continue
                    prev_atr = float(atr.iloc[i - 1] or atr_val)
                    if prev_atr <= 0:
                        continue
                    # ATR compression on previous bar: ATR < 0.7 × 20d_avg
                    was_compressed = prev_atr < 0.7 * atr20_val
                    # Expansion trigger: current bar range > 1.5 × ATR
                    bar_range = float(df['high'].iloc[i] - df['low'].iloc[i])
                    is_expanding = bar_range > 1.5 * atr_val
                    if not (was_compressed and is_expanding):
                        continue
                    # Direction from bar
                    bar_close = float(df['close'].iloc[i])
                    bar_open = float(df['open'].iloc[i])
                    direction = 1.0 if bar_close > bar_open else -1.0
                    # Forward return: N bars, direction-normalised
                    fwd_close = float(df['close'].iloc[i + n_fwd])
                    fwd_ret = (fwd_close - bar_close) * direction / (atr_val or 1)
                    signal_rets.append(fwd_ret)

                n = len(signal_rets)
                wins = sum(1 for v in signal_rets if v > 0)
                wr = wins / n if n > 0 else 0.0
                avg_r = sum(signal_rets) / n if n > 0 else 0.0
                ic = _ic_from_lists(signal_rets, [1.0] * n) if n > 0 else 0.0
                p = _p_value_approx(ic, n)
                sharpe = _hypothesis_sharpe(signal_rets, [True] * n)
                status = _score_hypothesis(n, ic, sharpe, p, wr)

                result = {
                    'hypothesis_id': hyp_id,
                    'description': (f'ATR compression<0.7avg then expansion>1.5ATR: '
                                    f'sustained momentum for next {n_fwd} bars'),
                    'category': 'volatility_regime',
                    'instrument': 'MNQ',
                    'lookback_days': 180,
                    'signals_generated': n,
                    'win_rate': round(float(wr), 4),
                    'sharpe': round(float(sharpe), 3) if sharpe is not None else None,
                    'avg_r': round(float(avg_r), 4),
                    'information_coefficient': round(float(ic), 4),
                    'p_value': round(float(p), 4),
                    'status': status,
                    'run_date': date.today().isoformat(),
                }
                results.append(result)
                if status == 'SIGNIFICANT':
                    _promote_significant_to_shadow(hyp_id, result['description'],
                                                   sharpe, wr)
            except Exception as e:
                logger.warning(f'vol regime hypothesis n={n_fwd}: {e}')
    except Exception as e:
        logger.warning(f'_test_volatility_regime_hypotheses: {e}')
    return results


def _test_cross_instrument_hypotheses(conn) -> list:
    """Category 5: Cross-instrument hypotheses (3 variants, MNQ + ES)."""
    results = []
    try:
        import pandas as pd
        df_mnq = _load_ohlcv_df(conn, 'MNQ', '5min', 180)
        df_es = _load_ohlcv_df(conn, 'ES', '5min', 180)
        if len(df_mnq) < 100 or len(df_es) < 100:
            return results

        # Align on timestamps
        mnq_ts = set(df_mnq['ts'].values)
        es_ts = set(df_es['ts'].values)
        common_ts = sorted(mnq_ts & es_ts)
        if len(common_ts) < 100:
            return results

        df_m = df_mnq[df_mnq['ts'].isin(common_ts)].set_index('ts').sort_index()
        df_e = df_es[df_es['ts'].isin(common_ts)].set_index('ts').sort_index()

        # Resample to 1h for correlation analysis
        try:
            df_m_h = df_m[['close']].resample('1h', on=pd.to_datetime(pd.Series(df_m.index), unit='s', utc=True)).last()
            df_e_h = df_e[['close']].resample('1h', on=pd.to_datetime(pd.Series(df_e.index), unit='s', utc=True)).last()
        except Exception:
            # Fallback: use raw 5min aligned data grouped by 12-bar blocks (1h approx)
            df_m_h = df_m['close'].iloc[::12]
            df_e_h = df_e['close'].iloc[::12]

        # Hypothesis 1: MNQ/ES 1h correlation drop < 0.8 predicts vol expansion
        try:
            hyp_id = 'cross_mnq_es_corr_drop_vol'
            corr_window = 12  # 12 1h bars = ~0.5 trading day
            mnq_close = list(df_m['close'].values)
            es_close = list(df_e['close'].values)
            n_bars = min(len(mnq_close), len(es_close))
            signal_rets = []
            atr_mnq = _atr14(df_mnq)
            atr_mnq_vals = list(atr_mnq.values)

            for i in range(corr_window + 1, n_bars - 12):
                m_slice = mnq_close[i - corr_window:i]
                e_slice = es_close[i - corr_window:i]
                corr_val = _ic_from_lists(m_slice, e_slice)
                if corr_val >= 0.8:
                    continue
                # Signal: correlation dropped below 0.8 — expect vol expansion
                atr_i = float(atr_mnq_vals[min(i, len(atr_mnq_vals) - 1)] or 1)
                curr_atr = float(atr_mnq_vals[min(i + 6, len(atr_mnq_vals) - 1)] or atr_i)
                # Vol expansion = ATR 6 bars later > current ATR
                vol_expanded = 1.0 if curr_atr > 1.2 * atr_i else 0.0
                signal_rets.append(vol_expanded)

            n = len(signal_rets)
            wins = sum(1 for v in signal_rets if v > 0)
            wr = wins / n if n > 0 else 0.0
            ic = _ic_from_lists([1.0] * n, signal_rets) if n > 0 else 0.0
            p = _p_value_approx(ic, n)
            sharpe = _hypothesis_sharpe(signal_rets, [True] * n)
            status = _score_hypothesis(n, ic, sharpe, p, wr)
            results.append({
                'hypothesis_id': hyp_id,
                'description': 'MNQ/ES 1h correlation drop <0.8 predicts volatility expansion',
                'category': 'cross_instrument',
                'instrument': 'MNQ/ES',
                'lookback_days': 180,
                'signals_generated': n,
                'win_rate': round(float(wr), 4),
                'sharpe': round(float(sharpe), 3) if sharpe is not None else None,
                'avg_r': round(float(wr - 0.5), 4),
                'information_coefficient': round(float(ic), 4),
                'p_value': round(float(p), 4),
                'status': status,
                'run_date': date.today().isoformat(),
            })
        except Exception as e:
            logger.warning(f'cross instrument hyp 1: {e}')

        # Hypothesis 2: MNQ/ES 4h bias diverge — bearish instrument reverts
        try:
            hyp_id = 'cross_mnq_es_bias_diverge_revert'
            signal_rets = []
            # Compute 4h bias for both (simplified: 12-bar EMA comparison)
            for i in range(24, min(len(mnq_close), len(es_close)) - 10):
                m_ema = sum(mnq_close[i - 12:i]) / 12
                e_ema = sum(es_close[i - 12:i]) / 12
                m_bias = 1 if mnq_close[i] > m_ema else -1
                e_bias = 1 if es_close[i] > e_ema else -1
                if m_bias == e_bias:
                    continue
                # Diverge: different biases — bearish one should revert
                bearish_instrument = 'mnq' if m_bias < 0 else 'es'
                if bearish_instrument == 'mnq':
                    entry = mnq_close[i]
                    fwd = mnq_close[min(i + 5, len(mnq_close) - 1)]
                else:
                    entry = es_close[i]
                    fwd = es_close[min(i + 5, len(es_close) - 1)]
                atr_i = float(atr_mnq_vals[min(i, len(atr_mnq_vals) - 1)] or 1)
                fwd_ret = (fwd - entry) / (atr_i or 1)  # positive = reverted up
                signal_rets.append(fwd_ret)

            n = len(signal_rets)
            wins = sum(1 for v in signal_rets if v > 0)
            wr = wins / n if n > 0 else 0.0
            avg_r = sum(signal_rets) / n if n > 0 else 0.0
            ic = _ic_from_lists(signal_rets, [1.0] * n) if n > 0 else 0.0
            p = _p_value_approx(ic, n)
            sharpe = _hypothesis_sharpe(signal_rets, [True] * n)
            status = _score_hypothesis(n, ic, sharpe, p, wr)
            results.append({
                'hypothesis_id': hyp_id,
                'description': 'MNQ/ES 4h bias diverge: bearish instrument reverts within 5 bars',
                'category': 'cross_instrument',
                'instrument': 'MNQ/ES',
                'lookback_days': 180,
                'signals_generated': n,
                'win_rate': round(float(wr), 4),
                'sharpe': round(float(sharpe), 3) if sharpe is not None else None,
                'avg_r': round(float(avg_r), 4),
                'information_coefficient': round(float(ic), 4),
                'p_value': round(float(p), 4),
                'status': status,
                'run_date': date.today().isoformat(),
            })
        except Exception as e:
            logger.warning(f'cross instrument hyp 2: {e}')

        # Hypothesis 3: Days after MNQ >2% move have higher mean-reversion rate
        try:
            hyp_id = 'cross_mnq_large_move_reversion'
            signal_rets = []
            dates_mnq = df_mnq['date'].unique() if 'date' in df_mnq.columns else []
            for i, d in enumerate(dates_mnq[1:], 1):
                prev_mask = df_mnq['date'] == dates_mnq[i - 1]
                curr_mask = df_mnq['date'] == d
                prev_day = df_mnq[prev_mask & (df_mnq['hour'] >= 13) & (df_mnq['hour'] < 19)]
                curr_day = df_mnq[curr_mask & (df_mnq['hour'] >= 13) & (df_mnq['hour'] < 19)]
                if len(prev_day) < 5 or len(curr_day) < 5:
                    continue
                prev_open_p = float(prev_day['close'].iloc[0])
                prev_close_p = float(prev_day['close'].iloc[-1])
                if prev_open_p <= 0:
                    continue
                prev_move_pct = abs(prev_close_p - prev_open_p) / prev_open_p
                if prev_move_pct <= 0.02:
                    continue
                # Mean-reversion day: expect opposite direction
                direction = -1.0 if prev_close_p > prev_open_p else 1.0
                curr_open_p = float(curr_day['close'].iloc[0])
                curr_close_p = float(curr_day['close'].iloc[-1])
                atr_i = float(atr.iloc[curr_day.index[0]] or 1)
                fwd_ret = (curr_close_p - curr_open_p) * direction / (atr_i or 1)
                signal_rets.append(fwd_ret)

            n = len(signal_rets)
            wins = sum(1 for v in signal_rets if v > 0)
            wr = wins / n if n > 0 else 0.0
            avg_r = sum(signal_rets) / n if n > 0 else 0.0
            ic = _ic_from_lists(signal_rets, [1.0] * n) if n > 0 else 0.0
            p = _p_value_approx(ic, n)
            sharpe = _hypothesis_sharpe(signal_rets, [True] * n)
            status = _score_hypothesis(n, ic, sharpe, p, wr)
            results.append({
                'hypothesis_id': hyp_id,
                'description': 'Days after MNQ >2% move have higher mean-reversion win rate',
                'category': 'cross_instrument',
                'instrument': 'MNQ',
                'lookback_days': 180,
                'signals_generated': n,
                'win_rate': round(float(wr), 4),
                'sharpe': round(float(sharpe), 3) if sharpe is not None else None,
                'avg_r': round(float(avg_r), 4),
                'information_coefficient': round(float(ic), 4),
                'p_value': round(float(p), 4),
                'status': status,
                'run_date': date.today().isoformat(),
            })
        except Exception as e:
            logger.warning(f'cross instrument hyp 3: {e}')

    except Exception as e:
        logger.warning(f'_test_cross_instrument_hypotheses: {e}')
    return results


# ─── Main hypothesis engine ──────────────────────────────────────────────────

def run_hypothesis_engine() -> list:
    """
    Test 20+ market hypotheses against historical OHLCV data.
    Returns list of hypothesis results. Writes to hypothesis_log table.
    Only reads ohlcv table. Completely isolated from live trading.
    Runs nightly at 02:00 UTC via background_scheduler.
    Max 10 minutes (uses vectorized pandas, pure Python fallbacks).
    """
    logger.info('Hypothesis Engine: starting run')
    all_results = []
    conn = _conn()
    try:
        cat_funcs = [
            ('autocorrelation', _test_autocorr_hypotheses),
            ('hurst', _test_hurst_hypotheses),
            ('session_timing', _test_session_timing_hypotheses),
            ('volatility_regime', _test_volatility_regime_hypotheses),
            ('cross_instrument', _test_cross_instrument_hypotheses),
        ]
        for cat_name, cat_func in cat_funcs:
            try:
                cat_results = cat_func(conn)
                for r in cat_results:
                    _write_hypothesis(r)
                    all_results.append(r)
                logger.info(f'Hypothesis Engine: {cat_name} — {len(cat_results)} hypotheses tested')
            except Exception as e:
                logger.error(f'Hypothesis Engine: {cat_name} failed — {e}', exc_info=True)
    except Exception as e:
        logger.error(f'run_hypothesis_engine: {e}', exc_info=True)
    finally:
        _close(conn)

    sig = sum(1 for r in all_results if r.get('status') == 'SIGNIFICANT')
    logger.info(f'Hypothesis Engine: complete — {len(all_results)} tested, {sig} significant')
    return all_results


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 3.2 — COMBINATION EXPLORER
# ═══════════════════════════════════════════════════════════════════════════

def run_combination_explorer() -> list:
    """
    Test 3-feature combinations from available feature set.
    Runs Saturday 03:00 UTC. Uses last 210 days (180 train + 30 OOS).
    Tests all C(8,3)=56 combinations. Fits logistic regression on IS data,
    evaluates on OOS. Records oos_ic and oos_auc.
    """
    logger.info('Combination Explorer: starting run')
    results = []
    conn = _conn()
    try:
        import pandas as pd
        df = _load_ohlcv_df(conn, 'MNQ', '5min', 210)
        if len(df) < 500:
            logger.warning('Combination Explorer: insufficient data')
            _close(conn)
            return results

        atr = _atr14(df)
        atr5ago = atr.shift(5)
        avg_atr20 = atr.rolling(20).mean()

        # Compute all features
        close = df['close']
        opn = df['open']
        high = df['high']
        low = df['low']
        vol = df['volume']
        vol20avg = vol.rolling(20).mean()
        vol20std = vol.rolling(20).std()

        autocorr_1 = close.rolling(20).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 2 else 0.0, raw=False
        ).fillna(0)

        hurst_vals = []
        close_arr = list(close.values)
        for i in range(len(close_arr)):
            if i < 100:
                hurst_vals.append(0.5)
            else:
                hurst_vals.append(_hurst_rs(close_arr[i - 100:i]))
        import pandas as pd
        hurst_s = pd.Series(hurst_vals, index=close.index)

        # Compute session_progress per bar (minutes since 13:00 UTC / 360)
        def _session_prog(dt_val):
            try:
                mins = (dt_val.hour * 60 + dt_val.minute) - (13 * 60)
                return max(0.0, min(1.0, mins / 360.0))
            except Exception:
                return 0.0

        session_prog = df['dt'].apply(_session_prog) if 'dt' in df.columns else pd.Series(
            [0.0] * len(df), index=df.index)

        day_of_week = df['weekday'].astype(float) if 'weekday' in df.columns else pd.Series(
            [0.0] * len(df), index=df.index)

        atr_ratio = (atr / atr5ago.replace(0, float('nan'))).fillna(1.0)
        body_ratio = ((close - opn).abs() / (high - low).replace(0, float('nan'))).fillna(0.5)
        vol_zscore = ((vol - vol20avg) / vol20std.replace(0, float('nan'))).fillna(0.0)
        vol_ratio = (atr / avg_atr20.replace(0, float('nan'))).fillna(1.0)

        feature_frame = pd.DataFrame({
            'autocorr_1': autocorr_1,
            'hurst': hurst_s,
            'vol_ratio': vol_ratio,
            'session_progress': session_prog,
            'day_of_week': day_of_week,
            'atr_ratio': atr_ratio,
            'body_ratio': body_ratio,
            'vol_zscore': vol_zscore,
        }).fillna(0)

        # Forward return: 12-bar (1h approx) normalised by ATR
        fwd_return = ((close.shift(-12) - close) / atr.replace(0, float('nan'))).fillna(0)
        target = (fwd_return > 0).astype(int)  # binary: up or not

        # Split: IS = first 180 days of data, OOS = last 30 days
        n_total = len(df)
        ts_arr = df['ts'].values if 'ts' in df.columns else list(range(n_total))
        cutoff_210 = ts_arr[-1] - 210 * 86400
        cutoff_oos = ts_arr[-1] - 30 * 86400
        is_mask = (df['ts'] >= cutoff_210) & (df['ts'] < cutoff_oos) if 'ts' in df.columns else (
            pd.Series([True] * int(n_total * 0.86) + [False] * (n_total - int(n_total * 0.86)),
                      index=df.index))
        oos_mask = (df['ts'] >= cutoff_oos) if 'ts' in df.columns else ~is_mask

        feat_names = list(feature_frame.columns)
        import itertools
        combos = list(itertools.combinations(feat_names, 3))
        today_iso = date.today().isoformat()

        for combo in combos:
            combo_key = ','.join(combo)
            try:
                X_is = feature_frame[list(combo)][is_mask].values
                y_is = target[is_mask].values
                X_oos = feature_frame[list(combo)][oos_mask].values
                y_oos = target[oos_mask].values

                if len(X_is) < 100 or len(X_oos) < 20:
                    continue

                # Logistic regression (pure numpy, no sklearn required)
                # Standardise features
                means = X_is.mean(axis=0)
                stds = X_is.std(axis=0) + 1e-8
                X_is_n = (X_is - means) / stds
                X_oos_n = (X_oos - means) / stds

                # Gradient descent logistic regression
                import numpy as np
                w = np.zeros(X_is_n.shape[1])
                b = 0.0
                lr = 0.05
                for _ in range(200):
                    z = X_is_n.dot(w) + b
                    z = np.clip(z, -20, 20)
                    prob = 1 / (1 + np.exp(-z))
                    err = prob - y_is.astype(float)
                    w -= lr * X_is_n.T.dot(err) / len(y_is)
                    b -= lr * err.mean()

                # OOS predictions
                z_oos = X_oos_n.dot(w) + b
                z_oos = np.clip(z_oos, -20, 20)
                prob_oos = 1 / (1 + np.exp(-z_oos))

                # OOS IC
                oos_ic = _ic_from_lists(list(prob_oos), list(y_oos.astype(float)))

                # OOS AUC (trapezoidal)
                sorted_pairs = sorted(zip(prob_oos, y_oos), reverse=True)
                tp = fp = 0
                n_pos = y_oos.sum()
                n_neg = len(y_oos) - n_pos
                auc = 0.0
                if n_pos > 0 and n_neg > 0:
                    prev_fp = prev_tp = 0
                    for _, label in sorted_pairs:
                        if label == 1:
                            tp += 1
                        else:
                            fp += 1
                        auc += (fp - prev_fp) * (tp + prev_tp) / 2
                        prev_fp, prev_tp = fp, tp
                    auc = auc / (n_pos * n_neg) if n_pos * n_neg > 0 else 0.5

                results.append({
                    'features': combo_key,
                    'oos_ic': round(float(oos_ic), 4),
                    'oos_auc': round(float(auc), 4),
                    'run_date': today_iso,
                })

                # Write to DB
                w_conn = _conn()
                try:
                    w_conn.execute(
                        "INSERT INTO feature_combinations (features, oos_ic, oos_auc, run_date) "
                        "VALUES (?,?,?,?)",
                        (combo_key, round(float(oos_ic), 4), round(float(auc), 4), today_iso)
                    )
                    w_conn.commit()
                except Exception as e:
                    _rollback(w_conn)
                    logger.debug(f'Combination Explorer write {combo_key}: {e}')
                finally:
                    _close(w_conn)

            except Exception as e:
                logger.debug(f'Combination Explorer combo {combo_key}: {e}')
                continue

    except Exception as e:
        logger.error(f'run_combination_explorer: {e}', exc_info=True)
    finally:
        _close(conn)

    results.sort(key=lambda r: abs(r.get('oos_ic', 0)), reverse=True)
    logger.info(f'Combination Explorer: {len(results)} combos evaluated')
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 3.3 — SHADOW LAB LIVE SCANNING
# ═══════════════════════════════════════════════════════════════════════════

def scan_shadow_post_low_vol(symbol: str, df_5m, df_1h, df_4h) -> Optional[dict]:
    """
    Post-Low-Vol Expansion: enter on ATR expansion after compression.
    ATR compression: current ATR_14 < 0.7 × 20-bar avg ATR
    Expansion trigger: current bar range > 1.5 × ATR_14
    Direction: bullish bar = long, bearish = short
    HTF bias must align (df_4h based)
    """
    try:
        import pandas as pd
        if len(df_5m) < 30 or len(df_4h) < 21:
            return None

        atr = _atr14(df_5m)
        atr20avg = atr.rolling(20).mean()

        # Get last completed bar
        i = len(df_5m) - 1
        atr_val = float(atr.iloc[i] or 0)
        atr20_val = float(atr20avg.iloc[i] or 0)
        if atr_val <= 0 or atr20_val <= 0:
            return None

        prev_atr = float(atr.iloc[i - 1] or atr_val)

        # Compression on previous bar
        was_compressed = prev_atr < 0.7 * atr20_val
        if not was_compressed:
            return None

        # Expansion on current bar
        bar_high = float(df_5m['high'].iloc[i])
        bar_low = float(df_5m['low'].iloc[i])
        bar_close = float(df_5m['close'].iloc[i])
        bar_open = float(df_5m['open'].iloc[i])
        bar_range = bar_high - bar_low
        if bar_range <= 1.5 * atr_val:
            return None

        # Direction
        is_bullish = bar_close > bar_open
        direction = 'long' if is_bullish else 'short'

        # HTF bias from 4h (EMA20 alignment)
        ema20_4h = _ema(df_4h['close'], 20)
        htf_close = float(df_4h['close'].iloc[-1])
        htf_ema = float(ema20_4h.iloc[-1])
        if direction == 'long' and htf_close < htf_ema * 0.999:
            return None
        if direction == 'short' and htf_close > htf_ema * 1.001:
            return None

        entry = bar_close
        if direction == 'long':
            stop = bar_low - 0.5 * atr_val
            target = entry + 2.5 * (entry - stop)
        else:
            stop = bar_high + 0.5 * atr_val
            target = entry - 2.5 * (stop - entry)

        return {
            'symbol': symbol,
            'direction': direction,
            'setup': 'shadow_post_low_vol',
            'entry': round(float(entry), 2),
            'stop': round(float(stop), 2),
            'target': round(float(target), 2),
            'rr': 2.5,
            'atr': round(float(atr_val), 2),
            'quality': 'shadow_lab',
        }
    except Exception as e:
        logger.warning(f'scan_shadow_post_low_vol {symbol}: {e}')
        return None


def scan_shadow_monday_ny_open(symbol: str, df_5m, df_1h, df_4h) -> Optional[dict]:
    """
    Monday NY Open Long: first 30min of Monday NY session.
    Day must be Monday, time 13:00-13:30 UTC.
    Direction: long only.
    HTF 4h EMA20 must be bullish.
    Entry: bar close. Stop: bar_low - 0.5*ATR. Target: 2.5R.
    """
    try:
        import pandas as pd
        if len(df_5m) < 30 or len(df_4h) < 21:
            return None

        last = df_5m.iloc[-1]
        if 'dt' not in df_5m.columns:
            return None

        dt = last['dt']
        if hasattr(dt, 'to_pydatetime'):
            dt = dt.to_pydatetime()

        # Must be Monday 13:00-13:30 UTC
        if dt.weekday() != 0:
            return None
        if not (13 <= dt.hour < 13 or (dt.hour == 13 and dt.minute <= 30)):
            return None
        if not (dt.hour == 13 and 0 <= dt.minute <= 30):
            return None

        # Long only
        bar_close = float(last['close'])
        bar_low = float(last['low'])

        atr = _atr14(df_5m)
        atr_val = float(atr.iloc[-1] or 0)
        if atr_val <= 0:
            return None

        # HTF 4h EMA20 bullish
        ema20_4h = _ema(df_4h['close'], 20)
        htf_close = float(df_4h['close'].iloc[-1])
        htf_ema = float(ema20_4h.iloc[-1])
        if htf_close < htf_ema * 1.001:
            return None

        entry = bar_close
        stop = bar_low - 0.5 * atr_val
        target = entry + 2.5 * (entry - stop)

        return {
            'symbol': symbol,
            'direction': 'long',
            'setup': 'shadow_monday_ny_open',
            'entry': round(float(entry), 2),
            'stop': round(float(stop), 2),
            'target': round(float(target), 2),
            'rr': 2.5,
            'atr': round(float(atr_val), 2),
            'quality': 'shadow_lab',
        }
    except Exception as e:
        logger.warning(f'scan_shadow_monday_ny_open {symbol}: {e}')
        return None


def scan_shadow_value_area(symbol: str, df_5m, df_1h, df_4h) -> Optional[dict]:
    """
    Value Area Continuation: trade from previous session VAH/VAL retest.
    Previous session = yesterday 13:00-19:00 UTC (NY session).
    VAH = 70th percentile volume-weighted price level.
    VAL = 30th percentile.
    Entry: price retests VAH from above (long) or VAL from below (short)
    with confirming close back inside value area.
    Stop: beyond VAH/VAL by 0.5 ATR, Target: 2.5R.
    """
    try:
        import pandas as pd
        if len(df_5m) < 50:
            return None

        now_dt = df_5m['dt'].iloc[-1] if 'dt' in df_5m.columns else None
        if now_dt is None:
            return None
        if hasattr(now_dt, 'to_pydatetime'):
            now_dt = now_dt.to_pydatetime()

        # Must be in current NY session (13:00-19:00 UTC)
        if not (13 <= now_dt.hour < 19):
            return None

        # Get previous session bars (yesterday 13:00-19:00 UTC)
        prev_session_bars = df_5m[
            (df_5m['weekday'] < 5) &
            (df_5m['hour'] >= 13) & (df_5m['hour'] < 19) &
            (df_5m['date'] < now_dt.date())
        ].tail(78)  # ~78 bars in NY session (6h × 12 bars/h + buffer)

        if len(prev_session_bars) < 10:
            return None

        # Volume profile: volume-weighted prices
        ps_vol = prev_session_bars['volume'].values
        ps_close = prev_session_bars['close'].values
        total_vol = ps_vol.sum()
        if total_vol <= 0:
            return None

        # Sort by price to compute percentile-based value area
        price_vol_pairs = sorted(zip(ps_close, ps_vol), key=lambda x: x[0])
        cum_vol = 0
        p30_price = None
        p70_price = None
        for price, vol in price_vol_pairs:
            cum_vol += vol
            pct = cum_vol / total_vol
            if p30_price is None and pct >= 0.30:
                p30_price = price
            if p70_price is None and pct >= 0.70:
                p70_price = price

        if p30_price is None or p70_price is None:
            return None

        val = float(p30_price)  # Value Area Low
        vah = float(p70_price)  # Value Area High

        atr = _atr14(df_5m)
        atr_val = float(atr.iloc[-1] or 0)
        if atr_val <= 0:
            return None

        # Current bar
        bar_close = float(df_5m['close'].iloc[-1])
        bar_open = float(df_5m['open'].iloc[-1])
        bar_low = float(df_5m['low'].iloc[-1])
        bar_high = float(df_5m['high'].iloc[-1])

        direction = None
        entry = stop = target = None

        # Long: price tested VAH from above (bar_low <= VAH) with close back above VAH
        # (bullish close confirms rejection of test)
        if (bar_low <= vah * 1.001 and bar_close > vah and
                bar_close > bar_open):
            direction = 'long'
            entry = bar_close
            stop = vah - 0.5 * atr_val
            target = entry + 2.5 * (entry - stop)

        # Short: price tested VAL from below (bar_high >= VAL) with close back below VAL
        elif (bar_high >= val * 0.999 and bar_close < val and
              bar_close < bar_open):
            direction = 'short'
            entry = bar_close
            stop = val + 0.5 * atr_val
            target = entry - 2.5 * (stop - entry)

        if direction is None:
            return None

        return {
            'symbol': symbol,
            'direction': direction,
            'setup': 'shadow_value_area',
            'entry': round(float(entry), 2),
            'stop': round(float(stop), 2),
            'target': round(float(target), 2),
            'rr': 2.5,
            'atr': round(float(atr_val), 2),
            'vah': round(float(vah), 2),
            'val': round(float(val), 2),
            'quality': 'shadow_lab',
        }
    except Exception as e:
        logger.warning(f'scan_shadow_value_area {symbol}: {e}')
        return None


def _ensure_gap_orb_in_shadow_lab() -> None:
    """
    Idempotent: insert Gap ORB MNQ into shadow_lab if not already present.
    Called on every startup from seed_shadow_lab_candidates() regardless of prior seeding.
    """
    try:
        c = _conn()
        try:
            existing = c.execute(
                "SELECT id FROM shadow_lab WHERE strategy_name = ?",
                ('Gap ORB MNQ',)
            ).fetchone()
            if existing:
                logger.info('Research Division: Gap ORB MNQ already in shadow lab')
                return
            today = date.today()
            promo = (today + timedelta(weeks=8)).isoformat()
            c.execute(
                "INSERT INTO shadow_lab "
                "(strategy_name, description, entered_date, week_number, total_weeks, "
                " backtest_sharpe, backtest_win_rate, status, promotion_eligible_date) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    'Gap ORB MNQ',
                    ('Regime-enhanced Opening Range Breakout on MNQ. '
                     'Fires 14:05-19:00 UTC on gap days (open >0.15% from prior close), '
                     'Tue/Wed/Thu only. Requires TRENDING regime + confidence>=0.50. '
                     'Stop: ORB boundary + 0.5xATR (cap 60pts). Target: 2.0R. '
                     'Promotion: WR>=50%, Sharpe>=3.0, >=15 live signals (8-week window). '
                     'Backtest: Sharpe 4.98, WR 55.2%, 29 trades/year.'),
                    today.isoformat(), 0, 8,
                    4.98, 0.552,
                    'ACTIVE', promo,
                )
            )
            c.commit()
            logger.info('Research Division: Gap ORB MNQ added to shadow lab (8-week programme starts today)')
        except Exception as e:
            _rollback(c)
            logger.error(f'_ensure_gap_orb_in_shadow_lab: insert failed — {e}', exc_info=True)
        finally:
            _close(c)
    except Exception as e:
        logger.error(f'_ensure_gap_orb_in_shadow_lab: connection failed — {e}')


def scan_shadow_gap_orb(symbol: str, df_5m, df_1h, df_4h) -> Optional[dict]:
    """
    Gap ORB MNQ — Opening Range Breakout on gap days.

    Backtest (365 days): Sharpe 4.98 | WR 55.2% | 29 trades | MaxDD 3.9R
    Parameters:
      Symbol:    MNQ only
      Days:      Tuesday / Wednesday / Thursday (Friday 20% WR — excluded)
      Gap:       Day open must gap >0.15% from prior session close (either direction)
      ORB:       High/Low of first 3 x 5min bars (13:00, 13:05, 13:10 UTC)
      Entry:     First bar close above ORB High (long) or below ORB Low (short)
                 after 14:05 UTC blackout clears
      Regime:    TRENDING + confidence >= 0.50 (live regime_log)
      Stop:      ORB boundary + 0.5xATR buffer | hard cap 60pts MNQ
      Target:    2.0R from entry
      Dedup:     Once per day (in-memory _gap_orb_fired)
    """
    try:
        import pandas as pd

        if symbol != 'MNQ':
            return None
        if len(df_5m) < 50:
            return None

        df = df_5m.copy()
        if 'minute' not in df.columns:
            df['minute'] = df['dt'].dt.minute

        now = df['dt'].iloc[-1]
        if hasattr(now, 'to_pydatetime'):
            now = now.to_pydatetime()
        today = now.date()

        # Day filter: Tue / Wed / Thu only
        if now.weekday() not in (1, 2, 3):
            return None

        # Time filter: 14:05-19:00 UTC
        in_window = (now.hour > 14) or (now.hour == 14 and now.minute >= 5)
        if not in_window or now.hour >= 19:
            return None

        # Dedup: once per session day
        today_str = str(today)
        if _gap_orb_fired.get(today_str):
            return None

        # ── ORB: first 3 bars at 13:00-13:10 UTC ─────────────────────────────
        today_bars = df[df['date'] == today]
        orb_bars   = today_bars[
            (today_bars['hour'] == 13) & (today_bars['minute'].isin([0, 5, 10]))
        ]
        if len(orb_bars) < 3:
            return None
        orb_high = float(orb_bars['high'].max())
        orb_low  = float(orb_bars['low'].min())

        # ── Gap: today open vs prior day close ───────────────────────────────
        open_bar = today_bars[(today_bars['hour'] == 13) & (today_bars['minute'] == 0)]
        if len(open_bar) == 0:
            return None
        day_open = float(open_bar.iloc[0]['open'])

        prev_bars = df[df['date'] < today]
        if len(prev_bars) == 0:
            return None
        prev_close = float(prev_bars.iloc[-1]['close'])
        if prev_close <= 0:
            return None

        gap_pct = abs(day_open - prev_close) / prev_close * 100
        if gap_pct <= 0.15:
            return None

        # ── ATR ───────────────────────────────────────────────────────────────
        atr_val = float(_atr14(df).iloc[-1] or 0)
        if atr_val <= 0:
            return None

        # ── Regime gate: live TRENDING + confidence >= 0.50 ──────────────────
        regime   = 'UNKNOWN'
        conf_val = 0.0
        try:
            rc = _conn()
            try:
                rrow = rc.execute(
                    "SELECT regime, confidence FROM regime_log "
                    "WHERE symbol=? ORDER BY timestamp DESC LIMIT 1",
                    (symbol,)
                ).fetchone()
                if rrow:
                    regime   = rrow[0] or 'UNKNOWN'
                    conf_val = float(rrow[1] or 0)
            finally:
                _close(rc)
        except Exception as _re:
            logger.debug(f'scan_shadow_gap_orb: regime lookup failed ({_re}) — blocking')
            return None

        if regime != 'TRENDING' or conf_val < 0.50:
            return None

        # ── Direction: bar close must break ORB level ─────────────────────────
        bar_close = float(df['close'].iloc[-1])
        if bar_close > orb_high:
            direction = 'long'
        elif bar_close < orb_low:
            direction = 'short'
        else:
            return None

        # ── Stop / target ─────────────────────────────────────────────────────
        if direction == 'long':
            stop_dist = max(bar_close - orb_low, 0.5 * atr_val)
            stop      = bar_close - stop_dist
            target    = bar_close + 2.0 * stop_dist
        else:
            stop_dist = max(orb_high - bar_close, 0.5 * atr_val)
            stop      = bar_close + stop_dist
            target    = bar_close - 2.0 * stop_dist

        if stop_dist > 60:   # MNQ hard stop cap
            return None

        # ── Mark fired for today ──────────────────────────────────────────────
        _gap_orb_fired[today_str] = True

        logger.info(
            f'[SHADOW LAB] Gap ORB MNQ {direction.upper()} | '
            f'entry={bar_close:.2f} stop={stop:.2f} target={target:.2f} | '
            f'orb={orb_low:.2f}-{orb_high:.2f} | gap={gap_pct:.3f}% | '
            f'regime={regime} conf={conf_val:.2f}'
        )

        return {
            'symbol':    symbol,
            'direction': direction,
            'setup':     'shadow_gap_orb',
            'entry':     round(bar_close, 2),
            'stop':      round(stop, 2),
            'target':    round(target, 2),
            'rr':        2.0,
            'atr':       round(atr_val, 2),
            'orb_high':  round(orb_high, 2),
            'orb_low':   round(orb_low, 2),
            'gap_pct':   round(gap_pct, 3),
            'regime':    regime,
            'conf':      round(conf_val, 2),
            'quality':   'shadow_lab',
        }
    except Exception as e:
        logger.warning(f'scan_shadow_gap_orb {symbol}: {e}')
        return None


def _ensure_k_es_in_shadow_lab() -> None:
    """Idempotent: insert Setup K ES into shadow_lab if not already present."""
    try:
        c = _conn()
        try:
            existing = c.execute(
                "SELECT id FROM shadow_lab WHERE strategy_name = ?",
                ('Setup K ES',)
            ).fetchone()
            if existing:
                logger.info('Research Division: Setup K ES already in shadow lab')
                return
            today = date.today()
            promo = (today + timedelta(weeks=8)).isoformat()
            c.execute(
                "INSERT INTO shadow_lab "
                "(strategy_name, description, entered_date, week_number, total_weeks, "
                " backtest_sharpe, backtest_win_rate, status, promotion_eligible_date) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    'Setup K ES',
                    ('Meridian Transition Entry on ES. '
                     'Entry: first TRENDING bar after >=3 consecutive CHOPPY bars. '
                     'Session: 14:05-16:00 UTC, Tue/Wed/Thu only. '
                     'Direction: VWAP filter (above=long, below=short). '
                     'Stop: 1.0xATR (cap 15pts). Target: 2.5R. '
                     'Requires confidence>=0.50. '
                     'Promotion: WR>=50%, Sharpe>=3.0, >=15 live signals (8-week window). '
                     'Backtest: Sharpe 6.40, WR 48.9%, 3.9/mo, MaxDD 5.0R, 11/13 pos months.'),
                    today.isoformat(), 0, 8,
                    6.40, 0.489,
                    'ACTIVE', promo,
                )
            )
            c.commit()
            logger.info('Research Division: Setup K ES added to shadow lab (8-week programme starts today)')
        except Exception as e:
            _rollback(c)
            logger.error(f'_ensure_k_es_in_shadow_lab: insert failed — {e}', exc_info=True)
        finally:
            _close(c)
    except Exception as e:
        logger.error(f'_ensure_k_es_in_shadow_lab: connection failed — {e}')


def _ensure_k_mnq_in_shadow_lab() -> None:
    """Idempotent: insert Setup K MNQ into shadow_lab if not already present."""
    try:
        c = _conn()
        try:
            existing = c.execute(
                "SELECT id FROM shadow_lab WHERE strategy_name = ?",
                ('Setup K MNQ',)
            ).fetchone()
            if existing:
                logger.info('Research Division: Setup K MNQ already in shadow lab')
                return
            today = date.today()
            promo = (today + timedelta(weeks=8)).isoformat()
            c.execute(
                "INSERT INTO shadow_lab "
                "(strategy_name, description, entered_date, week_number, total_weeks, "
                " backtest_sharpe, backtest_win_rate, status, promotion_eligible_date) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    'Setup K MNQ',
                    ('Meridian Transition Entry on MNQ. '
                     'Entry: CHOPPY>=4 consecutive bars + autocorr rising >=0.02 (15min window) '
                     '+ Hurst>0.60 + Hurst rising >=0.01 + autocorr>-0.02 (L3 proxy). '
                     'Session: 14:05-16:00 UTC, Tue/Wed only. '
                     'Direction: VWAP filter (above=long, below=short). '
                     'Stop: 1.0xATR (cap 60pts). Target: 2.5R. '
                     'Requires confidence>=0.50. '
                     'Promotion: WR>=50%, Sharpe>=3.0, >=15 live signals (8-week window). '
                     'Backtest: Sharpe 8.35, WR 57.1%, 1.8/mo, MaxDD 2.5R, 8/10 pos months.'),
                    today.isoformat(), 0, 8,
                    8.35, 0.571,
                    'ACTIVE', promo,
                )
            )
            c.commit()
            logger.info('Research Division: Setup K MNQ added to shadow lab (8-week programme starts today)')
        except Exception as e:
            _rollback(c)
            logger.error(f'_ensure_k_mnq_in_shadow_lab: insert failed — {e}', exc_info=True)
        finally:
            _close(c)
    except Exception as e:
        logger.error(f'_ensure_k_mnq_in_shadow_lab: connection failed — {e}')


def _get_regime_window_rd(symbol: str, n: int = 10) -> list:
    """Return last n regime_log rows for symbol as list of dicts (newest last)."""
    try:
        rc = _conn()
        try:
            rows = rc.execute(
                "SELECT regime, hurst, autocorr, confidence "
                "FROM regime_log WHERE symbol=? ORDER BY timestamp DESC LIMIT ?",
                (symbol, n)
            ).fetchall()
        finally:
            _close(rc)
        if not rows:
            return []
        return list(reversed(rows))   # chronological order, newest last
    except Exception as _e:
        logger.debug(f'_get_regime_window_rd {symbol}: {_e}')
        return []


def scan_shadow_k_es(symbol: str, df_5m, df_1h, df_4h) -> Optional[dict]:
    """
    Setup K ES — Meridian Transition Entry.

    Backtest (365 days): Sharpe 6.40 | WR 48.9% | 3.9/mo | MaxDD 5.0R | 11/13 pos months
    Parameters:
      Symbol:    ES only
      Days:      Tuesday / Wednesday / Thursday
      Session:   14:05-16:00 UTC (post NY-open blackout, pre midday decay)
      Entry:     First bar where regime transitions to TRENDING
                 after >= 3 consecutive CHOPPY bars
      Direction: Close > session VWAP → long | Close < VWAP → short
      Regime:    confidence >= 0.50
      Stop:      1.0 × ATR14 | hard cap 15pts ES
      Target:    2.5R
      Dedup:     Once per day (in-memory _shadow_k_es_fired)
    """
    try:
        import pandas as pd

        if symbol != 'ES':
            return None
        if len(df_5m) < 50:
            return None

        df = df_5m.copy()
        if 'minute' not in df.columns:
            df['minute'] = df['dt'].dt.minute

        now = df['dt'].iloc[-1]
        if hasattr(now, 'to_pydatetime'):
            now = now.to_pydatetime()
        today = now.date()

        # Day filter: Tue / Wed / Thu only
        if now.weekday() not in (1, 2, 3):
            return None

        # Time filter: 14:05-16:00 UTC (cut midday 16h+ decay)
        in_window = (now.hour == 14 and now.minute >= 5) or now.hour == 15
        if not in_window:
            return None

        # Dedup: once per session day
        today_str = str(today)
        if _shadow_k_es_fired.get(today_str):
            return None

        # Regime window: need last 10 bars
        rw = _get_regime_window_rd('ES', n=10)
        if len(rw) < 5:
            return None

        latest = rw[-1]
        regime   = latest[0] or 'UNKNOWN'
        conf_val = float(latest[3] or 0)

        # Gate: must be TRENDING now
        if regime != 'TRENDING':
            return None

        # Gate: confidence >= 0.50
        if conf_val < 0.50:
            return None

        # Gate: >= 3 of the 4 preceding bars were CHOPPY
        prior_choppy = sum(1 for row in rw[-5:-1] if row[0] == 'CHOPPY')
        if prior_choppy < 3:
            return None

        # ATR
        atr_val = float(_atr14(df).iloc[-1] or 0)
        if atr_val <= 0 or atr_val > 15:   # ES hard cap 15pts
            return None

        # VWAP direction
        today_bars = df[df['date'] == today]
        session_bars = today_bars[today_bars['hour'] >= 13]
        if len(session_bars) < 5:
            return None
        vol_sum = float(session_bars['volume'].sum())
        if vol_sum == 0:
            return None
        vwap = float((session_bars['close'] * session_bars['volume']).sum() / vol_sum)
        bar_close = float(df['close'].iloc[-1])
        direction = 'long' if bar_close > vwap else 'short'

        # Stop / target
        stop_dist = atr_val
        if direction == 'long':
            stop   = bar_close - stop_dist
            target = bar_close + 2.5 * stop_dist
        else:
            stop   = bar_close + stop_dist
            target = bar_close - 2.5 * stop_dist

        # Mark fired
        _shadow_k_es_fired[today_str] = True

        logger.info(
            f'[SHADOW LAB K-ES] Setup K ES {direction.upper()} | '
            f'entry={bar_close:.2f} stop={stop:.2f} target={target:.2f} | '
            f'prior_choppy={prior_choppy} atr={atr_val:.2f} vwap={vwap:.2f} | '
            f'conf={conf_val:.2f}'
        )

        return {
            'symbol':       symbol,
            'direction':    direction,
            'setup':        'shadow_k_es',
            'entry':        round(bar_close, 2),
            'stop':         round(stop, 2),
            'target':       round(target, 2),
            'rr':           2.5,
            'atr':          round(atr_val, 2),
            'vwap':         round(vwap, 2),
            'prior_choppy': prior_choppy,
            'conf':         round(conf_val, 2),
            'quality':      'shadow_lab',
        }
    except Exception as e:
        logger.warning(f'scan_shadow_k_es {symbol}: {e}')
        return None


def scan_shadow_k_mnq(symbol: str, df_5m, df_1h, df_4h) -> Optional[dict]:
    """
    Setup K MNQ — Meridian Transition Entry.

    Backtest (365 days): Sharpe 8.35 | WR 57.1% | 1.8/mo | MaxDD 2.5R | 8/10 pos months
    Parameters:
      Symbol:    MNQ only
      Days:      Tuesday / Wednesday only (Thursday 0% WR — excluded)
      Session:   14:05-16:00 UTC
      Entry:     CHOPPY >= 4 consecutive bars
                 + autocorr rising >= 0.02 over last 3 bars (15 min)
                 + Hurst > 0.60 + Hurst rising >= 0.01 over 3 bars
                 + autocorr > -0.02 (not deeply negative)
      Direction: Close > session VWAP → long | Close < VWAP → short
      Regime:    confidence >= 0.50
      Stop:      1.0 × ATR14 | hard cap 60pts MNQ
      Target:    2.5R
      Dedup:     Once per day (in-memory _shadow_k_mnq_fired)
    """
    try:
        import pandas as pd

        if symbol != 'MNQ':
            return None
        if len(df_5m) < 50:
            return None

        df = df_5m.copy()
        if 'minute' not in df.columns:
            df['minute'] = df['dt'].dt.minute

        now = df['dt'].iloc[-1]
        if hasattr(now, 'to_pydatetime'):
            now = now.to_pydatetime()
        today = now.date()

        # Day filter: Tue / Wed only (Thu had 0% WR)
        if now.weekday() not in (1, 2):
            return None

        # Time filter: 14:05-16:00 UTC
        in_window = (now.hour == 14 and now.minute >= 5) or now.hour == 15
        if not in_window:
            return None

        # Dedup: once per session day
        today_str = str(today)
        if _shadow_k_mnq_fired.get(today_str):
            return None

        # Regime window: need last 8 bars
        rw = _get_regime_window_rd('MNQ', n=8)
        if len(rw) < 5:
            return None

        latest = rw[-1]
        prev3  = rw[-4] if len(rw) >= 4 else rw[0]

        conf_val = float(latest[3] or 0)
        if conf_val < 0.50:
            return None

        # Gate: >= 4 consecutive CHOPPY bars (trailing)
        consec = 0
        for row in reversed(rw):
            if row[0] == 'CHOPPY':
                consec += 1
            else:
                break
        if consec < 4:
            return None

        # Gate: autocorr rising >= 0.02 over 15 min (3 bars)
        ac_now  = float(latest[2] or 0)
        ac_prev = float(prev3[2] or 0)
        if ac_now - ac_prev < 0.02:
            return None

        # Gate: L3 proxy — Hurst > 0.60, rising >= 0.01, autocorr > -0.02
        h_now  = float(latest[1] or 0)
        h_prev = float(prev3[1] or 0)
        if not (h_now > 0.60 and ac_now > -0.02 and h_now - h_prev >= 0.01):
            return None

        # ATR
        atr_val = float(_atr14(df).iloc[-1] or 0)
        if atr_val <= 0 or atr_val > 60:   # MNQ hard cap 60pts
            return None

        # VWAP direction
        today_bars = df[df['date'] == today]
        session_bars = today_bars[today_bars['hour'] >= 13]
        if len(session_bars) < 5:
            return None
        vol_sum = float(session_bars['volume'].sum())
        if vol_sum == 0:
            return None
        vwap = float((session_bars['close'] * session_bars['volume']).sum() / vol_sum)
        bar_close = float(df['close'].iloc[-1])
        direction = 'long' if bar_close > vwap else 'short'

        # Stop / target
        stop_dist = atr_val
        if direction == 'long':
            stop   = bar_close - stop_dist
            target = bar_close + 2.5 * stop_dist
        else:
            stop   = bar_close + stop_dist
            target = bar_close - 2.5 * stop_dist

        # Mark fired
        _shadow_k_mnq_fired[today_str] = True

        logger.info(
            f'[SHADOW LAB K-MNQ] Setup K MNQ {direction.upper()} | '
            f'entry={bar_close:.2f} stop={stop:.2f} target={target:.2f} | '
            f'consec_choppy={consec} ac_rise={ac_now-ac_prev:.3f} '
            f'hurst={h_now:.3f} h_rise={h_now-h_prev:.3f} | '
            f'atr={atr_val:.2f} vwap={vwap:.2f} conf={conf_val:.2f}'
        )

        return {
            'symbol':         symbol,
            'direction':      direction,
            'setup':          'shadow_k_mnq',
            'entry':          round(bar_close, 2),
            'stop':           round(stop, 2),
            'target':         round(target, 2),
            'rr':             2.5,
            'atr':            round(atr_val, 2),
            'vwap':           round(vwap, 2),
            'consec_choppy':  consec,
            'autocorr':       round(ac_now, 4),
            'ac_rise':        round(ac_now - ac_prev, 4),
            'hurst':          round(h_now, 4),
            'h_rise':         round(h_now - h_prev, 4),
            'conf':           round(conf_val, 2),
            'quality':        'shadow_lab',
        }
    except Exception as e:
        logger.warning(f'scan_shadow_k_mnq {symbol}: {e}')
        return None


# Shadow strategy name → scan function mapping
_SHADOW_SCAN_MAP = {
    'Post-Low-Vol Expansion': scan_shadow_post_low_vol,
    'Monday NY Open Long': scan_shadow_monday_ny_open,
    'Value Area Continuation': scan_shadow_value_area,
    'Gap ORB MNQ': scan_shadow_gap_orb,
    'Setup K ES':  scan_shadow_k_es,
    'Setup K MNQ': scan_shadow_k_mnq,
}


def run_shadow_lab_scans() -> list:
    """
    Run live scan functions for all active shadow lab candidates.
    Loads recent 5min, 1h, 4h bars for each relevant symbol.
    Logs signals to apex_trades with setup='shadow_{strategy_slug}'.
    Returns list of signals generated.
    """
    signals_generated = []
    c = _conn()
    try:
        rows = c.execute(
            "SELECT id, strategy_name, status FROM shadow_lab WHERE status = 'ACTIVE'"
        ).fetchall()
    except Exception as e:
        logger.error(f'run_shadow_lab_scans: shadow_lab query failed — {e}')
        _close(c)
        return signals_generated
    finally:
        _close(c)

    for shadow_id, strategy_name, status in rows:
        scan_func = _SHADOW_SCAN_MAP.get(strategy_name)
        if scan_func is None:
            continue

        for symbol in ('MNQ', 'ES'):
            try:
                conn = _conn()
                try:
                    df_5m = _load_ohlcv_df(conn, symbol, '5min', 5)
                    df_1h_raw = _load_ohlcv_df(conn, symbol, '1h', 30)
                    df_4h_raw = _load_ohlcv_df(conn, symbol, '4h', 60)

                    # Fall back to resampled 4h from 5min if 4h bars unavailable
                    if len(df_4h_raw) < 5 and len(df_5m) >= 50:
                        import pandas as pd
                        df5_idx = df_5m.set_index('dt')
                        df_4h_raw = df5_idx[['open', 'high', 'low', 'close', 'volume']].resample('4h').agg(
                            {'open': 'first', 'high': 'max', 'low': 'min',
                             'close': 'last', 'volume': 'sum'}
                        ).dropna().reset_index()

                    if len(df_5m) < 20:
                        continue
                finally:
                    _close(conn)

                signal = scan_func(symbol, df_5m, df_1h_raw, df_4h_raw)
                if signal is None:
                    continue

                # Log signal to apex_trades
                try:
                    from trade_tracker import log_trade
                    trade_id = log_trade(signal)
                    signals_generated.append({
                        'strategy_name': strategy_name,
                        'symbol': symbol,
                        'trade_id': trade_id,
                        'signal': signal,
                    })
                    logger.info(
                        f'Shadow Lab Scan: {strategy_name} signal on {symbol} '
                        f'{signal["direction"]} entry={signal["entry"]}'
                    )
                except Exception as e:
                    logger.warning(f'Shadow lab scan log_trade failed {strategy_name}: {e}')

            except Exception as e:
                logger.warning(f'run_shadow_lab_scans {strategy_name} {symbol}: {e}')

    return signals_generated


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 3.4 — PATTERN LIBRARY
# ═══════════════════════════════════════════════════════════════════════════

def update_pattern_library() -> dict:
    """
    Sync SIGNIFICANT hypotheses into the pattern_library.
    Re-validate ACTIVE patterns against most recent 30 days.
    Apply progressive decay if IC < 0.03 on revalidation run.
    Mark as DEAD if decay_score < 0.1.
    Returns summary dict.
    """
    today_iso = date.today().isoformat()
    summary = {'added': 0, 'updated': 0, 'fading': 0, 'dead': 0, 'active': 0}

    # Step 1: Promote SIGNIFICANT hypotheses into pattern_library
    c = _conn()
    try:
        sig_rows = c.execute(
            "SELECT hypothesis_id, description, category, instrument, "
            "       signals_generated, win_rate, sharpe, information_coefficient "
            "FROM hypothesis_log WHERE status = 'SIGNIFICANT'"
        ).fetchall()
    except Exception as e:
        logger.warning(f'update_pattern_library: hypothesis fetch failed — {e}')
        sig_rows = []
    finally:
        _close(c)

    for row in sig_rows:
        hyp_id, desc, cat, instrument, signals, wr, sharpe, ic = row
        pattern_id = f'hyp_{hyp_id}'
        c = _conn()
        try:
            existing = c.execute(
                "SELECT id, decay_score FROM pattern_library WHERE pattern_id = ?",
                (pattern_id,)
            ).fetchone()
            if not existing:
                c.execute(
                    """INSERT INTO pattern_library
                       (pattern_id, name, description, discovery_source, instrument,
                        signals_observed, win_rate, sharpe, information_coefficient,
                        first_observed, last_validated, decay_score, status)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (pattern_id, hyp_id, desc or '', cat or 'hypothesis',
                     instrument or 'MNQ', int(signals or 0),
                     float(wr or 0), float(sharpe or 0), float(ic or 0),
                     today_iso, today_iso, 1.0, 'ACTIVE')
                )
                c.commit()
                summary['added'] += 1
        except Exception as e:
            _rollback(c)
            logger.warning(f'update_pattern_library add {pattern_id}: {e}')
        finally:
            _close(c)

    # Step 2: Re-validate ACTIVE patterns against last 30 days
    c = _conn()
    try:
        active_rows = c.execute(
            "SELECT id, pattern_id, name, instrument, decay_score "
            "FROM pattern_library WHERE status = 'ACTIVE'"
        ).fetchall()
    except Exception as e:
        logger.warning(f'update_pattern_library: active fetch failed — {e}')
        active_rows = []
    finally:
        _close(c)

    for pat_id, pattern_id, name, instrument, decay_score in active_rows:
        try:
            # Re-validate: run a quick IC check on last 30 days
            conn = _conn()
            try:
                df = _load_ohlcv_df(conn, instrument or 'MNQ', '5min', 30)
            finally:
                _close(conn)

            if len(df) < 50:
                summary['active'] += 1
                continue

            # Quick re-validation: autocorr IC proxy
            close = df['close']
            atr = _atr14(df)
            autocorr = close.rolling(20).apply(
                lambda x: x.autocorr(lag=1) if len(x) > 2 else 0.0, raw=False
            ).fillna(0)
            fwd = (close.shift(-6) - close) / atr.replace(0, float('nan'))
            fwd = fwd.fillna(0)
            valid_mask = autocorr.notna() & fwd.notna()
            x_vals = list(autocorr[valid_mask].values)
            y_vals = list(fwd[valid_mask].values)
            recent_ic = _ic_from_lists(x_vals, y_vals) if len(x_vals) > 5 else 0.0

            new_decay = float(decay_score or 1.0)
            if abs(recent_ic) < 0.03:
                new_decay *= 0.7
            new_status = 'ACTIVE'
            if new_decay < 0.1:
                new_status = 'DEAD'
                summary['dead'] += 1
            elif new_decay < 0.3:
                summary['fading'] += 1
            else:
                summary['active'] += 1

            # Update
            uc = _conn()
            try:
                uc.execute(
                    "UPDATE pattern_library SET decay_score=?, status=?, last_validated=? WHERE id=?",
                    (round(float(new_decay), 4), new_status, today_iso, pat_id)
                )
                uc.commit()
                summary['updated'] += 1
            except Exception as e:
                _rollback(uc)
                logger.warning(f'update_pattern_library update {pattern_id}: {e}')
            finally:
                _close(uc)

        except Exception as e:
            logger.warning(f'update_pattern_library validate {pattern_id}: {e}')

    logger.info(
        f'Pattern Library: active={summary["active"]} fading={summary["fading"]} '
        f'dead={summary["dead"]} added={summary["added"]}'
    )
    return summary


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 3.5 — ENHANCED PROMOTION WORKFLOW
# ═══════════════════════════════════════════════════════════════════════════

def score_shadow_lab() -> list:
    """
    Advance week numbers for active shadow lab candidates.
    When completed (week >= total_weeks), check promotion criteria:
      - live Sharpe > 50% of backtest benchmark
      - WR > 48%
      - min 20 signals
    Creates PROMOTION_REVIEW or REJECTION decision accordingly.
    Opens its own connection.
    """
    c = _conn()
    updated = []
    try:
        rows = c.execute(
            "SELECT id, strategy_name, week_number, total_weeks, "
            "       paper_sharpe, backtest_sharpe, paper_win_rate, paper_signal_count "
            "FROM shadow_lab WHERE status = 'ACTIVE'"
        ).fetchall()

        for row in rows:
            sid, name, week_num, total_weeks, paper_sharpe, bt_sharpe, paper_wr, signal_count = row
            new_week = (week_num or 0) + 1
            try:
                c.execute("UPDATE shadow_lab SET week_number = ? WHERE id = ?", (new_week, sid))
                if new_week >= total_weeks:
                    # Check promotion criteria
                    ps = float(paper_sharpe or 0)
                    bs = float(bt_sharpe or 1)
                    wr = float(paper_wr or 0)
                    sigs = int(signal_count or 0)

                    sharpe_ok = ps > 0.5 * bs
                    wr_ok = wr > 0.48
                    sigs_ok = sigs >= 20

                    if sharpe_ok and wr_ok and sigs_ok:
                        decision_type = 'PROMOTION_REVIEW'
                        rec = (f'{name} completed {total_weeks} weeks. Criteria met: '
                               f'Sharpe={ps:.2f} ({ps/bs*100:.0f}% of BT), '
                               f'WR={wr:.1%}, Signals={sigs}. Recommend promotion.')
                    else:
                        decision_type = 'REJECTION'
                        reasons = []
                        if not sharpe_ok:
                            reasons.append(f'Sharpe {ps:.2f} < 50% of BT {bs:.2f}')
                        if not wr_ok:
                            reasons.append(f'WR {wr:.1%} < 48% minimum')
                        if not sigs_ok:
                            reasons.append(f'Only {sigs} signals (need 20+)')
                        rec = f'{name}: auto-rejected — {"; ".join(reasons)}'
                        # Auto-set shadow lab status to REJECTED
                        c.execute(
                            "UPDATE shadow_lab SET status = 'REJECTED' WHERE id = ?",
                            (sid,)
                        )

                    c.execute(
                        "INSERT INTO research_decisions "
                        "(decision_type, subject, recommendation, supporting_data, status) "
                        "VALUES (?,?,?,?,?)",
                        (
                            decision_type, name, rec,
                            json.dumps({'week_number': new_week,
                                        'backtest_sharpe': bt_sharpe,
                                        'paper_sharpe': paper_sharpe,
                                        'paper_win_rate': paper_wr,
                                        'paper_signal_count': signal_count,
                                        'criteria_met': decision_type == 'PROMOTION_REVIEW'}),
                            'PENDING' if decision_type == 'PROMOTION_REVIEW' else 'REJECTED',
                        )
                    )
                    logger.info(
                        f'Research Shadow Lab: {name} — {decision_type} created'
                        f' (sharpe_ok={sharpe_ok}, wr_ok={wr_ok}, sigs_ok={sigs_ok})'
                    )
                c.commit()
                updated.append({'id': sid, 'name': name, 'week_number': new_week})
            except Exception as e:
                _rollback(c)
                logger.error(f'Research Shadow Lab: {name} update failed — {e}')
    except Exception as e:
        _rollback(c)
        logger.error(f'score_shadow_lab failed: {e}', exc_info=True)
    finally:
        _close(c)
    return updated


# ═══════════════════════════════════════════════════════════════════════════
#  WEEKLY TELEGRAM REPORT
# ═══════════════════════════════════════════════════════════════════════════

def generate_weekly_telegram_report() -> bool:
    """
    Compose the weekly Research Division report and log it for the dashboard.
    Opens its own connection, closes in finally. Never raises — returns True/False.
    Dashboard Research Health tab is the destination — no Telegram from research functions.
    """
    today    = date.today()
    week_str = today.strftime('%d %b %Y')
    sep      = '━' * 21
    c        = _conn()
    try:
        # MAX(id) guarantees exactly one row per setup — avoids duplicate rows
        # when same date appears multiple times (same-day reruns during testing)
        health_rows = c.execute(
            "SELECT s.setup_id, s.health_score, s.alert_level, "
            "       s.sharpe_30d, s.sharpe_benchmark, s.win_rate, "
            "       s.backtest_score, s.live_score "
            "FROM strategy_health_log s "
            "WHERE s.id IN (SELECT MAX(id) FROM strategy_health_log GROUP BY setup_id) "
            "ORDER BY s.setup_id"
        ).fetchall()

        bt_rows = c.execute(
            "SELECT b.setup_id, b.edge_score, b.sharpe, b.win_rate, b.total_signals, b.run_date "
            "FROM backtest_results b "
            "WHERE b.id IN (SELECT MAX(id) FROM backtest_results GROUP BY setup_id) "
            "ORDER BY b.setup_id"
        ).fetchall()

        shadow_rows = c.execute(
            "SELECT strategy_name, week_number, total_weeks, backtest_sharpe "
            "FROM shadow_lab WHERE status = 'ACTIVE' ORDER BY id"
        ).fetchall()

        dec_rows = c.execute(
            "SELECT subject FROM research_decisions "
            "WHERE status = 'PENDING' ORDER BY created_at DESC LIMIT 5"
        ).fetchall()
    except Exception as e:
        _rollback(c)
        logger.error(f'Research report: DB query failed — {e}', exc_info=True)
        return False
    finally:
        _close(c)

    status_map = {'HEALTHY': '[OK]', 'WATCH': '[WATCH]', 'ALERT': '[ALERT]'}
    health_lines = []
    for row in health_rows:
        sid, score, alert, sharpe, sbm, wr, bts, lv = row
        name   = SETUP_NAMES.get(sid, sid)
        status = status_map.get(alert, '—')
        if score is None:
            health_lines.append(f'  {sid} {name}: no data')
        else:
            parts = [f'{status}  {sid} {name}: {score}/100']
            if bts is not None: parts.append(f'BT={bts}')
            if lv  is not None: parts.append(f'Live={lv}')
            if sharpe and sbm:
                pct = (sharpe / sbm - 1) * 100
                parts.append(f'Sharpe {sharpe:.2f} ({pct:+.0f}%)')
            if wr: parts.append(f'WR {wr*100:.1f}%')
            health_lines.append('  '.join(parts))

    bt_lines = [
        f'  {r[0]}: edge={r[1]}  sharpe={f"{r[2]:.2f}" if r[2] else "—"}  '
        f'WR={f"{r[3]*100:.0f}%" if r[3] else "—"}  signals={r[4]}'
        for r in bt_rows
    ]
    shadow_lines = [
        f'  {r[0]}  Week {r[1] or 0}/{r[2]}  BT Sharpe: {r[3]:.2f}'
        for r in shadow_rows
    ] or ['  No active candidates']
    dec_lines = [f'  > {r[0]}' for r in dec_rows] or ['  None this week']

    msg = (
        'WISE MERIDIAN CAPITAL\n'
        f'Research Division — Week of {week_str}\n'
        f'{sep}\n\n'
        'STRATEGY HEALTH\n' + '\n'.join(health_lines or ['No health data yet']) +
        ('\n\nDAILY BACKTEST\n' + '\n'.join(bt_lines) if bt_lines else '') +
        '\n\nSHADOW LAB\n' + '\n'.join(shadow_lines) +
        '\n\nDECISIONS PENDING\n' + '\n'.join(dec_lines) +
        f'\n\n{sep}\n'
        'Wise Meridian Capital - Research Division'
    )

    logger.info(f'Research: weekly report — dashboard only\n{msg}')
    return True
