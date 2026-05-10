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
        ]
    else:
        # SQLite: no IF NOT EXISTS for ALTER TABLE — use try/except with rollback
        ddl_ops = [
            ("ALTER TABLE strategy_health_log ADD COLUMN backtest_score INTEGER",
             "backtest_score column"),
            ("ALTER TABLE strategy_health_log ADD COLUMN live_score INTEGER",
             "live_score column"),
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


def _backtest_stats(pnl_list: list, benchmark_sharpe: float, benchmark_wr: float,
                    bars_analysed: int, setup_id: str, lookback_days: int) -> BacktestResult:
    n = len(pnl_list)
    if n == 0:
        return BacktestResult(
            setup_id=setup_id, lookback_days=lookback_days,
            total_signals=0, win_rate=0.0, sharpe=None,
            avg_r=0.0, expectancy=0.0, max_drawdown=0.0,
            profit_factor=0.0, benchmark_sharpe=benchmark_sharpe,
            benchmark_win_rate=benchmark_wr,
            sharpe_vs_benchmark=0.0, wr_vs_benchmark=0.0,
            edge_score=0, run_date=date.today(), bars_analysed=bars_analysed,
        )
    wins      = [p for p in pnl_list if p > 0]
    losses    = [p for p in pnl_list if p < 0]
    win_rate  = len(wins) / n
    avg_r     = sum(pnl_list) / n
    mean      = avg_r
    var       = sum((v - mean) ** 2 for v in pnl_list) / (n - 1) if n > 1 else 0
    std       = math.sqrt(var) if var > 0 else 0
    sharpe    = (mean / std * math.sqrt(252)) if std > 0 else None
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_win / gross_loss if gross_loss > 0
                     else (999.0 if gross_win > 0 else 0.0))
    cum = peak = max_dd = 0.0
    for p in pnl_list:
        cum += p
        if cum > peak: peak = cum
        dd = (peak - cum) / max(abs(peak), 1)
        if dd > max_dd: max_dd = dd

    sharpe_vs_bm = ((sharpe / benchmark_sharpe) - 1) * 100 if (sharpe and benchmark_sharpe) else -100.0
    wr_vs_bm     = (win_rate - benchmark_wr) * 100

    es = _edge_score(win_rate, sharpe, n, profit_factor, max_dd, benchmark_sharpe, benchmark_wr)
    return BacktestResult(
        setup_id=setup_id, lookback_days=lookback_days,
        total_signals=n, win_rate=win_rate, sharpe=sharpe,
        avg_r=avg_r, expectancy=avg_r, max_drawdown=max_dd,
        profit_factor=profit_factor, benchmark_sharpe=benchmark_sharpe,
        benchmark_win_rate=benchmark_wr,
        sharpe_vs_benchmark=round(sharpe_vs_bm, 1),
        wr_vs_benchmark=round(wr_vs_bm, 1),
        edge_score=es, run_date=date.today(), bars_analysed=bars_analysed,
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

def backtest_setup_d(conn, lookback_days: int = 180) -> Optional[BacktestResult]:
    """FVG Fill — MNQ 5min. Bullish/bearish gap detection, session 13-19 UTC."""
    try:
        import pandas as pd
        df   = _load_ohlcv_df(conn, 'MNQ', '5min', lookback_days)
        if len(df) < 50: return None
        atr  = _atr14(df)
        highs = df['high'].values; lows = df['low'].values
        hours = df['hour'].values; bars = len(df)
        pnl   = []
        for i in range(2, bars - 1):
            if not (13 <= hours[i] < 19): continue
            atr_val = atr.iloc[i]
            if not atr_val or math.isnan(atr_val) or atr_val < 10: continue
            fvg_top = lows[i]; fvg_bot = highs[i - 2]
            if fvg_bot < fvg_top and (fvg_top - fvg_bot) >= 0.3 * atr_val:
                for j in range(i + 1, min(i + 31, bars)):
                    if lows[j] <= fvg_bot + (fvg_top - fvg_bot) * 0.5:
                        e = fvg_bot + (fvg_top - fvg_bot) * 0.5
                        pnl.append(_sim_outcome(df, j, 'long', e, e - atr_val, e + 2.5 * atr_val)); break
            fvg_top2 = lows[i - 2]; fvg_bot2 = highs[i]
            if fvg_bot2 < fvg_top2 and (fvg_top2 - fvg_bot2) >= 0.3 * atr_val:
                for j in range(i + 1, min(i + 31, bars)):
                    if highs[j] >= fvg_top2 - (fvg_top2 - fvg_bot2) * 0.5:
                        e = fvg_top2 - (fvg_top2 - fvg_bot2) * 0.5
                        pnl.append(_sim_outcome(df, j, 'short', e, e + atr_val, e - 2.5 * atr_val)); break
        b = BENCHMARKS['D']
        return _backtest_stats(pnl, b['sharpe'], b['wr'], len(df), 'D', lookback_days)
    except Exception as e:
        logger.error(f'backtest_setup_d: {e}'); return None


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
    """Shared engine for B/A/C: swing-break + last-candle OB entry."""
    try:
        import pandas as pd
        df5 = _load_ohlcv_df(conn, symbol, '5min', lookback_days)
        if len(df5) < 80: return None
        df5 = df5.set_index('dt')
        df15 = df5[['open','high','low','close','volume']].resample('15min').agg(
            {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}
        ).dropna()
        df15['hour'] = df15.index.hour
        df15 = df15.reset_index(drop=True)
        atr   = _atr14(df15)
        highs  = df15['high'].values;  lows   = df15['low'].values
        closes = df15['close'].values; opens  = df15['open'].values
        hours  = df15['hour'].values;  bars   = len(df15); sw = 20
        pnl    = []
        for i in range(sw + 2, bars - 1):
            if not (session_start <= hours[i] < session_end): continue
            atr_val = atr.iloc[i]
            if not atr_val or math.isnan(atr_val): continue
            sh = max(highs[i - sw: i - 1]); sl = min(lows[i - sw: i - 1])
            if closes[i] > sh and closes[i - 1] <= sh:
                for k in range(i - 1, max(i - 15, 0), -1):
                    if closes[k] < opens[k]:
                        e = highs[k]; s = lows[k] - stop_mult * atr_val
                        pnl.append(_sim_outcome(df15, i, 'long', e, s, e + rr * abs(e - s))); break
            elif closes[i] < sl and closes[i - 1] >= sl:
                for k in range(i - 1, max(i - 15, 0), -1):
                    if closes[k] > opens[k]:
                        e = lows[k]; s = highs[k] + stop_mult * atr_val
                        pnl.append(_sim_outcome(df15, i, 'short', e, s, e - rr * abs(e - s))); break
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
    """VWAP 2σ Reversion — ES 5min, session 13-19."""
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
            if not vw or math.isnan(vw) or not vs or math.isnan(vs) or vs < 0.5: continue
            if not atr_val or math.isnan(atr_val): continue
            upper = vw + 2.0 * vs; lower = vw - 2.0 * vs
            if closes[i] > upper and vw < closes[i] - 0.5 * atr_val:
                pnl.append(_sim_outcome(df, i, 'short', closes[i], closes[i] + 1.5 * atr_val, vw))
            elif closes[i] < lower and vw > closes[i] + 0.5 * atr_val:
                pnl.append(_sim_outcome(df, i, 'long',  closes[i], closes[i] - 1.5 * atr_val, vw))
        b = BENCHMARKS['H']
        return _backtest_stats(pnl, b['sharpe'], b['wr'], len(df), 'H', lookback_days)
    except Exception as e:
        logger.error(f'backtest_setup_h: {e}'); return None


def backtest_setup_i(conn, lookback_days: int = 180) -> Optional[BacktestResult]:
    """Mathematical Alpha — MNQ 5min, Hurst + momentum, Tue-Thu, 13-19 UTC."""
    try:
        import numpy as np
        df = _load_ohlcv_df(conn, 'MNQ', '5min', lookback_days)
        if len(df) < 150: return None
        atr    = _atr14(df)
        closes = df['close'].values; hours = df['hour'].values
        wdays  = df['weekday'].values; bars  = len(df)
        hw     = 100; aw = 50; pnl = []

        def _hurst(x):
            n = len(x)
            if n < 20: return 0.5
            m = x.mean(); dv = np.cumsum(x - m)
            r = dv.max() - dv.min(); s = x.std()
            if s == 0: return 0.5
            rs = r / s
            return math.log(rs) / math.log(n) if rs > 0 else 0.5

        for i in range(hw + aw, bars - 1):
            if wdays[i] not in (1, 2, 3): continue
            if not (13 <= hours[i] < 19): continue
            atr_val = atr.iloc[i]
            if not atr_val or math.isnan(atr_val): continue
            log_ret = np.diff(np.log(closes[i - hw: i + 1]))
            hurst   = _hurst(log_ret)
            if hurst < 0.55: continue
            ret50   = np.diff(closes[i - aw: i + 1])
            autocorr = float(np.corrcoef(ret50[:-1], ret50[1:])[0, 1]) if len(ret50) > 5 else 0.0
            if math.isnan(autocorr): autocorr = 0.0
            mom = closes[i] - closes[i - 20]
            if autocorr > 0.05 and mom > 0.5 * atr_val:
                e = closes[i]
                pnl.append(_sim_outcome(df, i, 'long',  e, e - atr_val, e + 3.0 * atr_val))
            elif autocorr > 0.05 and mom < -0.5 * atr_val:
                e = closes[i]
                pnl.append(_sim_outcome(df, i, 'short', e, e + atr_val, e - 3.0 * atr_val))
        b = BENCHMARKS['I']
        return _backtest_stats(pnl, b['sharpe'], b['wr'], len(df), 'I', lookback_days)
    except Exception as e:
        logger.error(f'backtest_setup_i: {e}'); return None


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
    """
    today   = date.today()
    results = {}

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
            w_conn.execute(
                "INSERT INTO backtest_results "
                "(setup_id, lookback_days, run_date, total_signals, win_rate, sharpe, "
                " avg_r, expectancy, max_drawdown, profit_factor, "
                " benchmark_sharpe, benchmark_win_rate, "
                " sharpe_vs_benchmark, wr_vs_benchmark, edge_score, bars_analysed) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    bt.setup_id, bt.lookback_days, today.isoformat(),
                    bt.total_signals,
                    round(bt.win_rate, 4) if bt.win_rate else 0,
                    round(bt.sharpe, 3)   if bt.sharpe  else None,
                    round(bt.avg_r, 4), round(bt.expectancy, 4),
                    round(bt.max_drawdown, 4), round(bt.profit_factor, 3),
                    bt.benchmark_sharpe, bt.benchmark_win_rate,
                    round(bt.sharpe_vs_benchmark, 1), round(bt.wr_vs_benchmark, 1),
                    bt.edge_score, bt.bars_analysed,
                )
            )
            w_conn.commit()
            results[sid] = bt
            logger.info(
                f'Research Backtest {sid}: signals={bt.total_signals} '
                f'edge={bt.edge_score} sharpe={bt.sharpe} '
                f'wr={bt.win_rate:.3f} bars={bt.bars_analysed}'
            )
        except Exception as e:
            _rollback(w_conn)
            logger.error(f'Research Backtest {sid} write failed: {type(e).__name__}: {e}')
            results[sid] = bt  # return in-memory result even when write fails
        finally:
            _close(w_conn)

    _check_edge_degradation()
    return results


def _check_edge_degradation():
    """Send Telegram alert if any setup has edge_score < 50 for 3 consecutive days."""
    c = _conn()
    try:
        from telegram_alerts import load_telegram_config, send_telegram
        token, chat_id = load_telegram_config()
        if not token or not chat_id:
            return
        for sid in BACKTEST_FUNCS:
            rows = c.execute(
                "SELECT edge_score, sharpe, run_date FROM backtest_results "
                "WHERE setup_id=? ORDER BY run_date DESC LIMIT 3",
                (sid,)
            ).fetchall()
            if len(rows) < 3: continue
            scores = [r[0] for r in rows]
            if all(s is not None and s < 50 for s in scores):
                latest_sharpe = rows[0][1] or 0
                bench_s = BENCHMARKS.get(sid, {}).get('sharpe', 0)
                pct = ((latest_sharpe / bench_s) - 1) * 100 if bench_s else 0
                send_telegram(
                    f'⚠️ <b>WISE MERIDIAN CAPITAL — Research Alert</b>\n\n'
                    f'Setup {sid} ({SETUP_NAMES.get(sid, sid)}) edge degrading\n'
                    f'Backtest Sharpe: {latest_sharpe:.2f} (benchmark {bench_s}) '
                    f'— {pct:.0f}% {"above" if pct >= 0 else "below"} benchmark\n'
                    f'3 consecutive days below threshold\n\n'
                    f'<b>Action required:</b> review parameters',
                    token, chat_id, parse_mode='HTML'
                )
                logger.warning(f'Research: edge degradation alert sent for Setup {sid}')
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
            "FROM backtest_results WHERE setup_id=? ORDER BY run_date DESC LIMIT 1",
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
        return {'setup_id': sid, 'health_score': None, 'alert_level': 'INSUFFICIENT_DATA'}
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


def score_shadow_lab() -> list:
    """Advance week numbers for active shadow lab candidates. Opens its own connection."""
    c = _conn()
    updated = []
    try:
        rows = c.execute(
            "SELECT id, strategy_name, week_number, total_weeks, "
            "       paper_sharpe, backtest_sharpe "
            "FROM shadow_lab WHERE status = 'ACTIVE'"
        ).fetchall()

        for row in rows:
            sid, name, week_num, total_weeks, paper_sharpe, bt_sharpe = row
            new_week = (week_num or 0) + 1
            try:
                c.execute("UPDATE shadow_lab SET week_number = ? WHERE id = ?", (new_week, sid))
                if new_week >= total_weeks:
                    c.execute(
                        "INSERT INTO research_decisions "
                        "(decision_type, subject, recommendation, supporting_data, status) "
                        "VALUES (?,?,?,?,?)",
                        (
                            'PROMOTION_REVIEW', name,
                            f'{name} completed {total_weeks} weeks. Review for promotion.',
                            json.dumps({'week_number': new_week,
                                        'backtest_sharpe': bt_sharpe,
                                        'paper_sharpe': paper_sharpe}),
                            'PENDING',
                        )
                    )
                    logger.info(f'Research Shadow Lab: {name} — promotion review created')
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
    Compose and send the weekly Research Division Telegram report.
    Opens its own connection, closes in finally. Never raises — returns True/False.
    """
    try:
        from telegram_alerts import load_telegram_config, send_telegram
        token, chat_id = load_telegram_config()
        if not token or not chat_id:
            logger.warning('Research report: Telegram not configured')
            return False
    except Exception as e:
        logger.error(f'Research report: Telegram import failed — {e}')
        return False

    today    = date.today()
    week_str = today.strftime('%d %b %Y')
    sep      = '━' * 21
    c        = _conn()
    try:
        health_rows = c.execute(
            "SELECT s.setup_id, s.health_score, s.alert_level, "
            "       s.sharpe_30d, s.sharpe_benchmark, s.win_rate, "
            "       s.backtest_score, s.live_score "
            "FROM strategy_health_log s "
            "INNER JOIN (SELECT setup_id, MAX(week_start) AS latest "
            "            FROM strategy_health_log GROUP BY setup_id) m "
            "ON s.setup_id = m.setup_id AND s.week_start = m.latest "
            "ORDER BY s.setup_id"
        ).fetchall()

        bt_rows = c.execute(
            "SELECT b.setup_id, b.edge_score, b.sharpe, b.win_rate, b.total_signals, b.run_date "
            "FROM backtest_results b "
            "INNER JOIN (SELECT setup_id, MAX(run_date) AS latest "
            "            FROM backtest_results GROUP BY setup_id) m "
            "ON b.setup_id = m.setup_id AND b.run_date = m.latest "
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

    emoji_map = {'HEALTHY': '✅', 'WATCH': '⚠️', 'ALERT': '🔴'}
    health_lines = []
    for row in health_rows:
        sid, score, alert, sharpe, sbm, wr, bts, lv = row
        name  = SETUP_NAMES.get(sid, sid)
        emoji = emoji_map.get(alert, '—')
        if score is None:
            health_lines.append(f'—  {sid} {name}: no data')
        else:
            parts = [f'{emoji}  {sid} {name}: <b>{score}/100</b>']
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
    dec_lines = [f'  ▸ {r[0]}' for r in dec_rows] or ['  None this week']

    msg = (
        '<b>WISE MERIDIAN CAPITAL</b>\n'
        f'Research Division — Week of {week_str}\n'
        f'{sep}\n\n'
        '<b>STRATEGY HEALTH</b>\n' + '\n'.join(health_lines or ['No health data yet']) +
        ('\n\n<b>DAILY BACKTEST</b>\n' + '\n'.join(bt_lines) if bt_lines else '') +
        '\n\n<b>SHADOW LAB</b>\n' + '\n'.join(shadow_lines) +
        '\n\n<b>DECISIONS PENDING</b>\n' + '\n'.join(dec_lines) +
        f'\n\n{sep}\n'
        '<i>Wise Meridian Capital · Research Division</i>'
    )

    try:
        result = send_telegram(msg, token, chat_id, parse_mode='HTML')
        if result:
            logger.info('Research Division: weekly report sent via Telegram')
        return bool(result)
    except Exception as e:
        logger.error(f'Research Division: Telegram send failed — {e}')
        return False
