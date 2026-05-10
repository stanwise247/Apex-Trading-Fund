"""
Wise Meridian Capital — Research Division
==========================================

ISOLATION CONTRACT
  Read-only access to: apex_trades, ohlcv, regime_log
  Writes only to:      strategy_health_log, shadow_lab, research_decisions, backtest_results
  Never imports from:  setup_engine, fvg_engine, setup_*.py, trade_tracker, risk_manager
  Zero impact on live trading signal generation.

FUNCTIONS OVERVIEW
  calculate_health_score(setup_id, conn)
      Dual scoring: 60% backtest (reads backtest_results) + 40% live trades.
      Falls back to 100% backtest if < 5 live trades exist.
      Returns a dict with health_score (0-100), alert_level, and all metrics.

  run_weekly_health_check(conn)
      Scores all 7 setups, writes to strategy_health_log, detects consecutive declines.
      Called every Monday 06:00 UTC by the background scheduler.

  run_daily_backtest(conn)
      Runs all 7 setup backtests on last 180 days of OHLCV data.
      Writes results to backtest_results. Checks for 3-day edge degradation alerts.
      Called every weekday 02:00 UTC.

  seed_shadow_lab_candidates(conn)  — idempotent, runs once at startup.
  score_shadow_lab(conn)            — advances week counters every Monday.
  generate_weekly_telegram_report(conn) — sends full weekly digest.

BACKTEST ENGINE
  Each backtest function reads raw OHLCV bars and replays entry/exit logic:
    backtest_setup_d  — FVG Fill:       MNQ 5min, session 13-19, ATR-based stops
    backtest_setup_e  — EMA50 Pullback: MNQ 5min, session 13-18
    backtest_setup_b  — CHoCH + OB:     MNQ/ES aggregated-15min, session 07-11
    backtest_setup_a  — Sweep + OB:     MNQ 5min, session 07-19
    backtest_setup_c  — BOS + OB:       MNQ 5min, session 07-11
    backtest_setup_h  — VWAP Reversion: ES 5min, session 13-19
    backtest_setup_i  — Math Alpha:     MNQ 5min, Hurst + momentum filter

ADDING A NEW SETUP
  1. Create backtest_setup_X(conn, lookback_days=180) -> Optional[BacktestResult]
  2. Load OHLCV via _load_ohlcv_df(conn, symbol, timeframe, lookback_days)
  3. Detect signals with vectorized pandas logic
  4. Call _sim_outcome() per signal to get pnl_r
  5. Call _backtest_stats() to aggregate, then _edge_score() for 0-100 rating
  6. Add the function to BACKTEST_FUNCS dict in run_daily_backtest

SCHEDULING CADENCE
  02:00 UTC (weekdays)  — run_daily_backtest (heavy: ~30s)
  06:00 UTC (Monday)    — run_weekly_health_check + score_shadow_lab + Telegram report
"""

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, date, timedelta, timezone
from typing import Optional, List

import db as _db

logger = logging.getLogger('APEX.Research')

# ── Benchmarks ──────────────────────────────────────────────────────────────

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
    sharpe_vs_benchmark: float   # % difference (positive = above)
    wr_vs_benchmark: float       # pp difference (positive = above)
    edge_score: int              # 0-100 composite
    run_date: date
    bars_analysed: int


# ═══════════════════════════════════════════════════════════════════════════
#  BACKTEST HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _load_ohlcv_df(conn, symbol: str, timeframe: str, lookback_days: int):
    """Load OHLCV bars as a pandas DataFrame. Returns empty DataFrame on failure."""
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
        df['dt'] = pd.to_datetime(df['ts'], unit='s', utc=True)
        df['hour'] = df['dt'].dt.hour
        df['weekday'] = df['dt'].dt.weekday
        df['date'] = df['dt'].dt.date
        return df.reset_index(drop=True)
    except Exception as e:
        logger.warning(f'_load_ohlcv_df {symbol} {timeframe}: {e}')
        import pandas as pd
        return pd.DataFrame()


def _atr14(df, n: int = 14):
    """ATR14 as pandas Series."""
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
    """
    Forward-simulate from start_idx+1 onward.
    Returns pnl_r: positive (RR ratio) for win, -1.0 for stop, 0 for timeout.
    """
    risk = abs(entry - stop)
    if risk <= 0:
        return 0.0
    rr = abs(target - entry) / risk
    end = min(start_idx + max_bars + 1, len(df))
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
                    bars_analysed: int, setup_id: str,
                    lookback_days: int) -> BacktestResult:
    """Aggregate a list of pnl_r values into a BacktestResult."""
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
    wins   = [p for p in pnl_list if p > 0]
    losses = [p for p in pnl_list if p < 0]
    win_rate = len(wins) / n
    avg_r    = sum(pnl_list) / n

    # Sharpe
    mean = avg_r
    if n > 1:
        var = sum((v - mean) ** 2 for v in pnl_list) / (n - 1)
        std = math.sqrt(var) if var > 0 else 0
        sharpe = (mean / std * math.sqrt(252)) if std > 0 else None
    else:
        sharpe = None

    # Profit factor
    gross_win  = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)

    # Max drawdown (% of peak equity)
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnl_list:
        cum += p
        if cum > peak:
            peak = cum
        dd = (peak - cum) / max(abs(peak), 1)
        if dd > max_dd:
            max_dd = dd

    sharpe_vs_bm = ((sharpe / benchmark_sharpe) - 1) * 100 if (sharpe and benchmark_sharpe) else -100.0
    wr_vs_bm     = (win_rate - benchmark_wr) * 100  # pp

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
    """Compute edge_score 0-100 per spec formula (base 60)."""
    score = 60

    # Sharpe vs benchmark
    if sharpe is not None and benchmark_sharpe > 0:
        ratio = sharpe / benchmark_sharpe
        if ratio >= 0.9:
            score += 20   # within 10%
        else:
            below_pct = (1.0 - ratio) * 100
            if below_pct >= 50:
                score -= 30
            else:
                score -= min(30, int(below_pct / 10) * 10)
    else:
        score -= 10  # no Sharpe data

    # Win rate vs benchmark
    wr_diff = win_rate - benchmark_wr
    if wr_diff >= -0.05:
        score += 15   # within 5pp
    else:
        pp_below = abs(wr_diff) * 100 - 5
        score -= min(30, int(pp_below / 5) * 10)

    # Signal count
    if total_signals < 20:
        score -= 20

    # Profit factor
    if profit_factor > 1.5:
        score += 10
    elif profit_factor < 1.0:
        score -= 10

    # Max drawdown
    if max_drawdown > 0.20:
        score -= 15

    return max(0, min(100, score))


# ═══════════════════════════════════════════════════════════════════════════
#  BACKTEST FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def backtest_setup_d(conn, lookback_days: int = 180) -> Optional[BacktestResult]:
    """FVG Fill — MNQ 5min. Bullish/bearish FVG detection, HTF bias, session 13-19 UTC."""
    try:
        import pandas as pd
        df = _load_ohlcv_df(conn, 'MNQ', '5min', lookback_days)
        if len(df) < 50:
            return None
        # Load 4h for HTF bias (EMA20)
        df4 = _load_ohlcv_df(conn, 'MNQ', '4hour', lookback_days)
        htf_bullish = set()
        if not df4.empty:
            ema20 = _ema(df4['close'], 20)
            bullish_ts = set(df4.loc[df4['close'] > ema20, 'ts'].values)
            bearish_ts = set(df4.loc[df4['close'] <= ema20, 'ts'].values)
            # Propagate: each 5min bar inherits the 4h bias at or before it
            df4_ts = sorted(df4['ts'].values)
            df4_bias = {df4['ts'].iloc[i]: (df4['close'].iloc[i] > ema20.iloc[i])
                        for i in range(len(df4))}
        atr = _atr14(df)
        highs = df['high'].values
        lows  = df['low'].values
        closes = df['close'].values
        hours   = df['hour'].values
        bars    = len(df)
        pnl = []

        for i in range(2, bars - 1):
            if not (13 <= hours[i] < 19):
                continue

            atr_val = atr.iloc[i]
            if not atr_val or math.isnan(atr_val) or atr_val < 10:
                continue

            # Bullish FVG: gap above bar[i-2] → bar[i] low > bar[i-2] high
            fvg_top    = lows[i]
            fvg_bottom = highs[i - 2]
            if fvg_bottom < fvg_top and (fvg_top - fvg_bottom) >= 0.3 * atr_val:
                # Look for fill in next 30 bars
                for j in range(i + 1, min(i + 31, bars)):
                    if lows[j] <= fvg_bottom + (fvg_top - fvg_bottom) * 0.5:
                        entry  = fvg_bottom + (fvg_top - fvg_bottom) * 0.5
                        stop   = entry - 1.0 * atr_val
                        target = entry + 2.5 * atr_val
                        pnl.append(_sim_outcome(df, j, 'long', entry, stop, target))
                        break

            # Bearish FVG: gap below bar[i-2] → bar[i] high < bar[i-2] low
            fvg_top2    = lows[i - 2]
            fvg_bottom2 = highs[i]
            if fvg_bottom2 < fvg_top2 and (fvg_top2 - fvg_bottom2) >= 0.3 * atr_val:
                for j in range(i + 1, min(i + 31, bars)):
                    if highs[j] >= fvg_top2 - (fvg_top2 - fvg_bottom2) * 0.5:
                        entry  = fvg_top2 - (fvg_top2 - fvg_bottom2) * 0.5
                        stop   = entry + 1.0 * atr_val
                        target = entry - 2.5 * atr_val
                        pnl.append(_sim_outcome(df, j, 'short', entry, stop, target))
                        break

        bench = BENCHMARKS['D']
        return _backtest_stats(pnl, bench['sharpe'], bench['wr'], len(df), 'D', lookback_days)
    except Exception as e:
        logger.error(f'backtest_setup_d failed: {e}')
        return None


def backtest_setup_e(conn, lookback_days: int = 180) -> Optional[BacktestResult]:
    """EMA50 Pullback — MNQ 5min. Session 13-18, ATR > 25."""
    try:
        df = _load_ohlcv_df(conn, 'MNQ', '5min', lookback_days)
        if len(df) < 60:
            return None
        atr  = _atr14(df)
        ema50 = _ema(df['close'], 50)
        closes = df['close'].values
        highs  = df['high'].values
        lows   = df['low'].values
        hours  = df['hour'].values
        bars   = len(df)
        pnl = []

        for i in range(60, bars - 1):
            if not (13 <= hours[i] < 18):
                continue
            atr_val  = atr.iloc[i]
            ema_val  = ema50.iloc[i]
            ema_prev = ema50.iloc[i - 1]
            if not atr_val or math.isnan(atr_val) or atr_val < 25:
                continue

            # Long: EMA50 rising, price pulls back to within 0.5×ATR of EMA50
            if ema_val > ema_prev:
                dist = closes[i] - ema_val
                if 0 <= dist <= 0.5 * atr_val:
                    entry  = closes[i]
                    stop   = ema_val - 1.5 * atr_val
                    target = entry + 3.75 * atr_val
                    pnl.append(_sim_outcome(df, i, 'long', entry, stop, target))

            # Short: EMA50 falling, price pulls back up to within 0.5×ATR
            elif ema_val < ema_prev:
                dist = ema_val - closes[i]
                if 0 <= dist <= 0.5 * atr_val:
                    entry  = closes[i]
                    stop   = ema_val + 1.5 * atr_val
                    target = entry - 3.75 * atr_val
                    pnl.append(_sim_outcome(df, i, 'short', entry, stop, target))

        bench = BENCHMARKS['E']
        return _backtest_stats(pnl, bench['sharpe'], bench['wr'], len(df), 'E', lookback_days)
    except Exception as e:
        logger.error(f'backtest_setup_e failed: {e}')
        return None


def _backtest_choch_ob(conn, setup_id: str, symbol: str,
                       session_start: int, session_end: int,
                       stop_mult: float, rr: float,
                       lookback_days: int) -> Optional[BacktestResult]:
    """Shared engine for B/A/C: swing-break + last-candle OB entry."""
    try:
        import pandas as pd
        df5 = _load_ohlcv_df(conn, symbol, '5min', lookback_days)
        if len(df5) < 80:
            return None
        # Aggregate 5min → 15min
        df5 = df5.set_index('dt')
        df15 = df5[['open', 'high', 'low', 'close', 'volume']].resample('15min').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()
        df15['hour'] = df15.index.hour
        df15 = df15.reset_index()
        df15 = df15.rename(columns={'dt': 'dt'})
        df15 = df15.reset_index(drop=True)
        atr  = _atr14(df15)
        highs  = df15['high'].values
        lows   = df15['low'].values
        closes = df15['close'].values
        opens  = df15['open'].values
        hours  = df15['hour'].values
        bars   = len(df15)
        swing_window = 20
        pnl = []

        for i in range(swing_window + 2, bars - 1):
            if not (session_start <= hours[i] < session_end):
                continue
            atr_val = atr.iloc[i]
            if not atr_val or math.isnan(atr_val):
                continue

            # Bullish ChoCH/BOS/Sweep: close breaks above recent swing high
            swing_high = max(highs[i - swing_window: i - 1])
            swing_low  = min(lows[i - swing_window: i - 1])

            if closes[i] > swing_high and closes[i - 1] <= swing_high:
                # OB = last bearish candle before breakout
                for k in range(i - 1, max(i - 15, 0), -1):
                    if closes[k] < opens[k]:  # bearish candle
                        ob_low = lows[k]
                        ob_high = highs[k]
                        entry  = ob_high
                        stop   = ob_low - stop_mult * atr_val
                        target = entry + rr * abs(entry - stop)
                        pnl.append(_sim_outcome(df15, i, 'long', entry, stop, target))
                        break

            elif closes[i] < swing_low and closes[i - 1] >= swing_low:
                for k in range(i - 1, max(i - 15, 0), -1):
                    if closes[k] > opens[k]:  # bullish candle
                        ob_high = highs[k]
                        ob_low  = lows[k]
                        entry  = ob_low
                        stop   = ob_high + stop_mult * atr_val
                        target = entry - rr * abs(entry - stop)
                        pnl.append(_sim_outcome(df15, i, 'short', entry, stop, target))
                        break

        bench = BENCHMARKS[setup_id]
        return _backtest_stats(pnl, bench['sharpe'], bench['wr'], len(df15), setup_id, lookback_days)
    except Exception as e:
        logger.error(f'backtest_{setup_id.lower()} failed: {e}')
        return None


def backtest_setup_b(conn, lookback_days: int = 180) -> Optional[BacktestResult]:
    """CHoCH + OB — MNQ, London session 07-11, RR 4.0, stop 0.8×ATR."""
    return _backtest_choch_ob(conn, 'B', 'MNQ', 7, 11, 0.8, 4.0, lookback_days)


def backtest_setup_a(conn, lookback_days: int = 180) -> Optional[BacktestResult]:
    """Sweep + OB — MNQ, session 07-19, RR 4.0, stop 0.8×ATR."""
    return _backtest_choch_ob(conn, 'A', 'MNQ', 7, 19, 0.8, 4.0, lookback_days)


def backtest_setup_c(conn, lookback_days: int = 180) -> Optional[BacktestResult]:
    """BOS + OB — ES, London session 07-11, RR 4.0, stop 0.8×ATR."""
    return _backtest_choch_ob(conn, 'C', 'ES', 7, 11, 0.8, 4.0, lookback_days)


def backtest_setup_h(conn, lookback_days: int = 180) -> Optional[BacktestResult]:
    """VWAP 2σ Reversion — ES 5min, session 13-19."""
    try:
        import pandas as pd
        import numpy as np
        df = _load_ohlcv_df(conn, 'ES', '5min', lookback_days)
        if len(df) < 50:
            return None
        atr = _atr14(df)
        # Daily VWAP + std bands
        df['tp'] = (df['high'] + df['low'] + df['close']) / 3
        df['tp_vol'] = df['tp'] * df['volume'].clip(lower=0)
        df['cum_tp_vol'] = df.groupby('date')['tp_vol'].cumsum()
        df['cum_vol']    = df.groupby('date')['volume'].transform(
            lambda x: x.clip(lower=0).cumsum()
        )
        df['vwap'] = df['cum_tp_vol'] / df['cum_vol'].replace(0, float('nan'))

        # Rolling std for bands (cumulative within day)
        df['dev2'] = ((df['tp'] - df['vwap']) ** 2) * df['volume'].clip(lower=0)
        df['cum_dev2'] = df.groupby('date')['dev2'].cumsum()
        with np.errstate(invalid='ignore'):
            df['vwap_std'] = (df['cum_dev2'] / df['cum_vol'].replace(0, float('nan'))).pow(0.5)

        highs  = df['high'].values
        lows   = df['low'].values
        closes = df['close'].values
        vwap_v = df['vwap'].values
        vstd_v = df['vwap_std'].values
        hours  = df['hour'].values
        bars   = len(df)
        pnl    = []

        for i in range(20, bars - 1):
            if not (13 <= hours[i] < 19):
                continue
            atr_val = atr.iloc[i]
            vw  = vwap_v[i]
            vs  = vstd_v[i]
            if (not vw or math.isnan(vw) or not vs or math.isnan(vs)
                    or vs < 0.5 or not atr_val or math.isnan(atr_val)):
                continue
            upper = vw + 2.0 * vs
            lower = vw - 2.0 * vs

            # Short: price above upper band
            if closes[i] > upper:
                entry  = closes[i]
                stop   = entry + 1.5 * atr_val
                target = vw  # revert to VWAP
                if target < entry - 0.5 * atr_val:  # reasonable target
                    pnl.append(_sim_outcome(df, i, 'short', entry, stop, target))

            # Long: price below lower band
            elif closes[i] < lower:
                entry  = closes[i]
                stop   = entry - 1.5 * atr_val
                target = vw
                if target > entry + 0.5 * atr_val:
                    pnl.append(_sim_outcome(df, i, 'long', entry, stop, target))

        bench = BENCHMARKS['H']
        return _backtest_stats(pnl, bench['sharpe'], bench['wr'], len(df), 'H', lookback_days)
    except Exception as e:
        logger.error(f'backtest_setup_h failed: {e}')
        return None


def backtest_setup_i(conn, lookback_days: int = 180) -> Optional[BacktestResult]:
    """Mathematical Alpha — MNQ 5min. Rolling Hurst + momentum, session 13-19, Tue-Thu."""
    try:
        import numpy as np
        df = _load_ohlcv_df(conn, 'MNQ', '5min', lookback_days)
        if len(df) < 150:
            return None
        atr    = _atr14(df)
        closes = df['close'].values
        hours  = df['hour'].values
        wdays  = df['weekday'].values
        bars   = len(df)
        pnl    = []
        hurst_window = 100
        autocorr_window = 50

        def _hurst(x):
            """Simplified R/S Hurst exponent on array x."""
            n = len(x)
            if n < 20:
                return 0.5
            mean_x = x.mean()
            dv = np.cumsum(x - mean_x)
            r  = dv.max() - dv.min()
            s  = x.std()
            if s == 0:
                return 0.5
            rs = r / s
            if rs <= 0:
                return 0.5
            return math.log(rs) / math.log(n)

        for i in range(hurst_window + autocorr_window, bars - 1):
            if wdays[i] not in (1, 2, 3):  # Tue-Thu only
                continue
            if not (13 <= hours[i] < 19):
                continue
            atr_val = atr.iloc[i]
            if not atr_val or math.isnan(atr_val):
                continue

            # Hurst exponent on recent 100 bars of log-returns
            log_ret = np.diff(np.log(closes[i - hurst_window: i + 1]))
            hurst   = _hurst(log_ret)

            # Lag-1 autocorrelation on recent 50 bars
            ret50 = np.diff(closes[i - autocorr_window: i + 1])
            if len(ret50) > 5:
                autocorr = float(np.corrcoef(ret50[:-1], ret50[1:])[0, 1])
            else:
                autocorr = 0.0

            if math.isnan(autocorr):
                autocorr = 0.0

            # Signal: trending (Hurst > 0.55) + momentum confirmation
            if hurst < 0.55:
                continue

            momentum = closes[i] - closes[i - 20]

            if autocorr > 0.05 and momentum > 0.5 * atr_val:
                entry  = closes[i]
                stop   = entry - 1.0 * atr_val
                target = entry + 3.0 * atr_val
                pnl.append(_sim_outcome(df, i, 'long', entry, stop, target))

            elif autocorr > 0.05 and momentum < -0.5 * atr_val:
                entry  = closes[i]
                stop   = entry + 1.0 * atr_val
                target = entry - 3.0 * atr_val
                pnl.append(_sim_outcome(df, i, 'short', entry, stop, target))

        bench = BENCHMARKS['I']
        return _backtest_stats(pnl, bench['sharpe'], bench['wr'], len(df), 'I', lookback_days)
    except Exception as e:
        logger.error(f'backtest_setup_i failed: {e}')
        return None


BACKTEST_FUNCS = {
    'A': backtest_setup_a,
    'B': backtest_setup_b,
    'C': backtest_setup_c,
    'D': backtest_setup_d,
    'E': backtest_setup_e,
    'H': backtest_setup_h,
    'I': backtest_setup_i,
}


# ═══════════════════════════════════════════════════════════════════════════
#  DAILY BACKTEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_daily_backtest(conn) -> dict:
    """
    Run all 7 setup backtests and write to backtest_results.
    Also checks for 3 consecutive days of edge_score < 50 and sends Telegram alert.
    """
    today   = date.today()
    results = {}

    for sid, bt_func in BACKTEST_FUNCS.items():
        try:
            bt = bt_func(conn, lookback_days=180)
            if bt is None:
                logger.warning(f'Backtest {sid}: returned None')
                results[sid] = None
                continue

            conn.execute(
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
                    round(bt.sharpe, 3) if bt.sharpe else None,
                    round(bt.avg_r, 4),
                    round(bt.expectancy, 4),
                    round(bt.max_drawdown, 4),
                    round(bt.profit_factor, 3),
                    bt.benchmark_sharpe,
                    bt.benchmark_win_rate,
                    round(bt.sharpe_vs_benchmark, 1),
                    round(bt.wr_vs_benchmark, 1),
                    bt.edge_score,
                    bt.bars_analysed,
                )
            )
            conn.commit()
            results[sid] = bt
            logger.info(
                f'Backtest {sid}: signals={bt.total_signals} '
                f'edge={bt.edge_score} sharpe={bt.sharpe}'
            )
        except Exception as e:
            logger.error(f'run_daily_backtest {sid}: {e}')
            results[sid] = None

    # Check for 3-day edge degradation
    _check_edge_degradation(conn)
    return results


def _check_edge_degradation(conn):
    """Send Telegram alert if any setup has edge_score < 50 for 3 consecutive days."""
    try:
        from telegram_alerts import load_telegram_config, send_telegram
        token, chat_id = load_telegram_config()
        if not token or not chat_id:
            return

        for sid in BACKTEST_FUNCS:
            rows = conn.execute(
                "SELECT edge_score, sharpe, run_date FROM backtest_results "
                "WHERE setup_id=? ORDER BY run_date DESC LIMIT 3",
                (sid,)
            ).fetchall()

            if len(rows) < 3:
                continue

            scores = [r[0] for r in rows]
            if all(s is not None and s < 50 for s in scores):
                latest_sharpe = rows[0][1]
                bench = BENCHMARKS.get(sid, {})
                bench_s = bench.get('sharpe', 0)
                pct = ((latest_sharpe / bench_s) - 1) * 100 if (latest_sharpe and bench_s) else 0
                msg = (
                    f'⚠️ <b>WISE MERIDIAN CAPITAL — Research Alert</b>\n\n'
                    f'Setup {sid} ({SETUP_NAMES.get(sid, sid)}) edge degrading\n'
                    f'Backtest Sharpe: {latest_sharpe:.2f} (benchmark {bench_s}) '
                    f'— {pct:.0f}% {"above" if pct >= 0 else "below"} benchmark\n'
                    f'3 consecutive days below threshold\n\n'
                    f'<b>Action required:</b> review parameters'
                )
                send_telegram(msg, token, chat_id, parse_mode='HTML')
                logger.warning(f'Edge degradation alert sent for Setup {sid}')
    except Exception as e:
        logger.error(f'_check_edge_degradation failed: {e}')


# ═══════════════════════════════════════════════════════════════════════════
#  HEALTH SCORING (dual: backtest 60% + live 40%)
# ═══════════════════════════════════════════════════════════════════════════

def _sharpe(pnl_r_list):
    n = len(pnl_r_list)
    if n < 5:
        return None
    mean = sum(pnl_r_list) / n
    var  = sum((v - mean) ** 2 for v in pnl_r_list) / (n - 1) if n > 1 else 0
    std  = math.sqrt(var) if var > 0 else 0
    if std == 0:
        return None
    return (mean / std) * math.sqrt(252)


def _weekly_metrics(setup_id, conn, n_weeks=4):
    today       = date.today()
    monday_this = today - timedelta(days=today.weekday())
    weeks = []
    for i in range(n_weeks - 1, -1, -1):
        w_start = monday_this - timedelta(weeks=i)
        w_end   = w_start + timedelta(days=7)
        rows = conn.execute(
            "SELECT pnl_r FROM apex_trades "
            "WHERE UPPER(SUBSTR(setup, 1, 1)) = ? "
            "  AND status = 'closed' AND pnl_r IS NOT NULL "
            "  AND exit_time >= ? AND exit_time < ?",
            (setup_id, w_start.isoformat(), w_end.isoformat())
        ).fetchall()
        pnl = [r[0] for r in rows]
        wr  = (sum(1 for v in pnl if v > 0) / len(pnl)) if pnl else None
        exp = (sum(pnl) / len(pnl)) if pnl else None
        weeks.append({'week_start': w_start, 'count': len(pnl), 'wr': wr, 'expectancy': exp})
    return weeks


def _consecutive(bool_list, n=2):
    count = 0
    for v in bool_list:
        if v:
            count += 1
            if count >= n:
                return True
        else:
            count = 0
    return False


def _calc_live_score(pnl_20, bench):
    """Compute live performance score 0-100 from last 20 trades."""
    pnl = [r[0] for r in pnl_20]
    wins     = sum(1 for v in pnl if v > 0)
    win_rate = wins / len(pnl)
    exp      = sum(pnl) / len(pnl)
    sharpe   = _sharpe(pnl)

    score = 100
    if sharpe is not None and bench['sharpe'] > 0:
        ratio = sharpe / bench['sharpe']
        if ratio < 0.5:
            score -= 20
        elif ratio > 1.1:
            score += 5
    wr_diff = win_rate - bench['wr']
    if wr_diff < -0.10:
        score -= 15
    if exp < 0:
        score -= 10
    return max(0, min(100, score))


def calculate_health_score(setup_id, conn):
    """
    Dual-scored health check: 60% backtest + 40% live.
    Reads latest backtest_results row; falls back to 100% backtest if < 5 live trades.
    """
    sid   = setup_id.upper()
    bench = BENCHMARKS.get(sid)
    if not bench:
        return {'setup_id': sid, 'health_score': None, 'alert_level': 'INSUFFICIENT_DATA'}

    now_utc = datetime.now(timezone.utc)
    cut_28d = (now_utc - timedelta(days=28)).isoformat()

    # ── Live trades (last 20) ─────────────────────────────
    rows_20 = conn.execute(
        "SELECT pnl_r FROM apex_trades "
        "WHERE UPPER(SUBSTR(setup, 1, 1)) = ? "
        "  AND status = 'closed' AND pnl_r IS NOT NULL "
        "ORDER BY exit_time DESC LIMIT 20",
        (sid,)
    ).fetchall()

    # ── Latest backtest result ────────────────────────────
    bt_row = conn.execute(
        "SELECT edge_score, sharpe, win_rate, bars_analysed, total_signals, "
        "       sharpe_vs_benchmark, wr_vs_benchmark "
        "FROM backtest_results WHERE setup_id=? ORDER BY run_date DESC LIMIT 1",
        (sid,)
    ).fetchone()

    backtest_score      = bt_row[0] if bt_row else None
    bt_sharpe           = bt_row[1] if bt_row else None
    bt_win_rate         = bt_row[2] if bt_row else None
    bars_analysed       = bt_row[3] if bt_row else 0
    bt_total_signals    = bt_row[4] if bt_row else 0
    bt_sharpe_vs_bm     = bt_row[5] if bt_row else None
    bt_wr_vs_bm         = bt_row[6] if bt_row else None

    # ── Live trade metrics ────────────────────────────────
    live_count  = len(rows_20)
    live_score  = None
    sharpe_30d  = None
    win_rate_l  = None
    expectancy  = None
    signal_count_week = None

    if live_count >= 5:
        live_score = _calc_live_score(rows_20, bench)
        pnl_l      = [r[0] for r in rows_20]
        wins       = sum(1 for v in pnl_l if v > 0)
        win_rate_l = wins / len(pnl_l)
        expectancy = sum(pnl_l) / len(pnl_l)
        sharpe_30d = _sharpe(pnl_l)
        count_28d  = conn.execute(
            "SELECT COUNT(*) FROM apex_trades "
            "WHERE UPPER(SUBSTR(setup, 1, 1)) = ? "
            "  AND status = 'closed' AND exit_time >= ?",
            (sid, cut_28d)
        ).fetchone()[0]
        signal_count_week = count_28d / 4.0

    # ── Dual score ────────────────────────────────────────
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
            'setup_id': sid, 'health_score': None,
            'alert_level': 'INSUFFICIENT_DATA',
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
        'win_rate':          round(win_rate_l, 4) if win_rate_l is not None else None,
        'win_rate_benchmark':bench['wr'],
        'signal_count_week': round(signal_count_week, 2) if signal_count_week else None,
        'expectancy':        round(expectancy, 4) if expectancy is not None else None,
        'health_score':      health_score,
        'alert_level':       alert_level,
        'backtest_score':    backtest_score,
        'live_score':        live_score,
        'bars_analysed':     bars_analysed,
        'live_trade_count':  live_count,
        'bt_sharpe':         round(bt_sharpe, 3) if bt_sharpe else None,
        'bt_win_rate':       round(bt_win_rate, 4) if bt_win_rate else None,
        'bt_total_signals':  bt_total_signals,
        'score_basis':       score_basis,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  WEEKLY HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════

def run_weekly_health_check(conn) -> dict:
    """Run health check for all live setups and write results to strategy_health_log."""
    setups  = ['A', 'B', 'C', 'D', 'E', 'H', 'I']
    today   = date.today()
    monday  = today - timedelta(days=today.weekday())
    results = {}

    for sid in setups:
        try:
            metrics = calculate_health_score(sid, conn)
            prev = conn.execute(
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

            conn.execute(
                "INSERT INTO strategy_health_log "
                "(setup_id, week_start, sharpe_30d, sharpe_benchmark, win_rate, "
                " win_rate_benchmark, signal_count_week, expectancy, health_score, "
                " alert_level, notes, backtest_score, live_score) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    sid, monday.isoformat(),
                    metrics.get('sharpe_30d'),
                    metrics.get('sharpe_benchmark'),
                    metrics.get('win_rate'),
                    metrics.get('win_rate_benchmark'),
                    metrics.get('signal_count_week'),
                    metrics.get('expectancy'),
                    metrics.get('health_score'),
                    metrics.get('alert_level'),
                    notes,
                    metrics.get('backtest_score'),
                    metrics.get('live_score'),
                )
            )
            conn.commit()
            results[sid] = metrics
            logger.info(
                f'Health {sid}: score={metrics.get("health_score")} '
                f'alert={metrics.get("alert_level")} basis={metrics.get("score_basis")}'
            )
        except Exception as e:
            logger.error(f'Health check failed {sid}: {e}')
            results[sid] = {'error': str(e)}

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  SHADOW LAB
# ═══════════════════════════════════════════════════════════════════════════

def seed_shadow_lab_candidates(conn):
    """Seed initial shadow lab candidates if the table is empty."""
    count = conn.execute("SELECT COUNT(*) FROM shadow_lab").fetchone()[0]
    if count > 0:
        logger.info(f'Shadow Lab: already seeded ({count} candidates)')
        return

    today = date.today()
    candidates = [
        {
            'name': 'Post-Low-Vol Expansion',
            'desc': (
                'Trades expansion moves following compressed volatility periods. '
                'Enters when ATR contracts below 20-day average then expands.'
            ),
            'sharpe': 29.91, 'wr': 0.68,
        },
        {
            'name': 'Monday NY Open Long',
            'desc': (
                'Exploits Monday NY session upside bias on MNQ. Long only, '
                'first 30 minutes of NY open, requires bullish HTF bias.'
            ),
            'sharpe': 11.43, 'wr': 0.61,
        },
        {
            'name': 'Value Area Continuation',
            'desc': (
                'Trades continuation moves from previous session value area high/low. '
                'Enters on retest of value area boundary with momentum confirmation.'
            ),
            'sharpe': 9.99, 'wr': 0.58,
        },
    ]

    for c in candidates:
        promo = (today + timedelta(weeks=8)).isoformat()
        conn.execute(
            "INSERT INTO shadow_lab "
            "(strategy_name, description, entered_date, week_number, total_weeks, "
            " backtest_sharpe, backtest_win_rate, status, promotion_eligible_date) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (c['name'], c['desc'], today.isoformat(), 0, 8,
             c['sharpe'], c['wr'], 'ACTIVE', promo)
        )
    conn.commit()
    logger.info(f'Shadow Lab: seeded {len(candidates)} candidates')


def score_shadow_lab(conn):
    """Advance week numbers for active shadow lab candidates."""
    rows = conn.execute(
        "SELECT id, strategy_name, week_number, total_weeks, "
        "       paper_sharpe, backtest_sharpe "
        "FROM shadow_lab WHERE status = 'ACTIVE'"
    ).fetchall()

    updated = []
    for row in rows:
        sid, name, week_num, total_weeks, paper_sharpe, bt_sharpe = row
        new_week = (week_num or 0) + 1
        conn.execute("UPDATE shadow_lab SET week_number = ? WHERE id = ?", (new_week, sid))
        if new_week >= total_weeks:
            conn.execute(
                "INSERT INTO research_decisions "
                "(decision_type, subject, recommendation, supporting_data, status) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    'PROMOTION_REVIEW', name,
                    f'{name} completed {total_weeks} weeks. Review performance for promotion.',
                    json.dumps({'week_number': new_week, 'backtest_sharpe': bt_sharpe,
                                'paper_sharpe': paper_sharpe}),
                    'PENDING',
                )
            )
            logger.info(f'Shadow Lab: {name} — promotion review created')
        conn.commit()
        updated.append({'id': sid, 'name': name, 'week_number': new_week})

    return updated


# ═══════════════════════════════════════════════════════════════════════════
#  WEEKLY TELEGRAM REPORT
# ═══════════════════════════════════════════════════════════════════════════

def generate_weekly_telegram_report(conn) -> bool:
    """Send the weekly Research Division Telegram report including backtest summary."""
    try:
        from telegram_alerts import load_telegram_config, send_telegram
        token, chat_id = load_telegram_config()
        if not token or not chat_id:
            logger.warning('Research report: Telegram not configured')
            return False
    except Exception as e:
        logger.error(f'Research report: Telegram import failed: {e}')
        return False

    today    = date.today()
    week_str = today.strftime('%d %b %Y')
    sep      = '━' * 21

    # Latest health per setup
    health_rows = conn.execute(
        "SELECT s.setup_id, s.health_score, s.alert_level, "
        "       s.sharpe_30d, s.sharpe_benchmark, s.win_rate, "
        "       s.backtest_score, s.live_score "
        "FROM strategy_health_log s "
        "INNER JOIN ("
        "  SELECT setup_id, MAX(week_start) AS latest "
        "  FROM strategy_health_log GROUP BY setup_id"
        ") m ON s.setup_id = m.setup_id AND s.week_start = m.latest "
        "ORDER BY s.setup_id"
    ).fetchall()

    emoji_map = {'HEALTHY': '✅', 'WATCH': '⚠️', 'ALERT': '🔴'}
    health_lines = []
    for row in health_rows:
        sid, score, alert, sharpe, sharpe_bench, wr, bt_score, lv_score = row
        name  = SETUP_NAMES.get(sid, sid)
        emoji = emoji_map.get(alert, '—')
        if score is None:
            health_lines.append(f'—  {sid} {name}: no data')
        else:
            parts = [f'{emoji}  {sid} {name}: <b>{score}/100</b>']
            if bt_score is not None:
                parts.append(f'BT={bt_score}')
            if lv_score is not None:
                parts.append(f'Live={lv_score}')
            if sharpe is not None and sharpe_bench:
                pct = (sharpe / sharpe_bench - 1) * 100
                parts.append(f'Sharpe {sharpe:.2f} ({pct:+.0f}%)')
            if wr is not None:
                parts.append(f'WR {wr*100:.1f}%')
            health_lines.append('  '.join(parts))

    # Latest backtest summary
    bt_rows = conn.execute(
        "SELECT b.setup_id, b.edge_score, b.sharpe, b.win_rate, b.total_signals, b.run_date "
        "FROM backtest_results b "
        "INNER JOIN (SELECT setup_id, MAX(run_date) AS latest "
        "            FROM backtest_results GROUP BY setup_id) m "
        "ON b.setup_id = m.setup_id AND b.run_date = m.latest "
        "ORDER BY b.setup_id"
    ).fetchall()

    bt_lines = []
    for row in bt_rows:
        sid, es, sharpe, wr, signals, rdate = row
        bt_lines.append(
            f'  {sid} ({SETUP_NAMES.get(sid,sid)[:12]}): '
            f'edge={es}  sharpe={sharpe:.2f if sharpe else "—"}  '
            f'WR={wr*100:.0f if wr else "—"}%  signals={signals}'
        )

    if not health_lines:
        health_lines = ['No health data — Monday check pending']

    # Shadow Lab
    shadow_rows = conn.execute(
        "SELECT strategy_name, week_number, total_weeks, backtest_sharpe "
        "FROM shadow_lab WHERE status = 'ACTIVE' ORDER BY id"
    ).fetchall()
    shadow_lines = [
        f'  {r[0]}  Week {r[1] or 0}/{r[2]}  BT Sharpe: {r[3]:.2f}'
        for r in shadow_rows
    ] or ['  No active candidates']

    # Pending decisions
    dec_rows = conn.execute(
        "SELECT subject FROM research_decisions WHERE status='PENDING' ORDER BY created_at DESC LIMIT 5"
    ).fetchall()
    dec_lines = [f'  ▸ {r[0]}' for r in dec_rows] or ['  None this week']

    msg = (
        '<b>WISE MERIDIAN CAPITAL</b>\n'
        f'Research Division — Week of {week_str}\n'
        f'{sep}\n\n'
        '<b>STRATEGY HEALTH</b>\n' + '\n'.join(health_lines) +
        ('\n\n<b>DAILY BACKTEST</b>\n' + '\n'.join(bt_lines) if bt_lines else '') +
        '\n\n<b>SHADOW LAB</b>\n' + '\n'.join(shadow_lines) +
        '\n\n<b>DECISIONS PENDING</b>\n' + '\n'.join(dec_lines) +
        f'\n\n{sep}\n'
        '<i>Wise Meridian Capital · Research Division</i>'
    )

    try:
        result = send_telegram(msg, token, chat_id, parse_mode='HTML')
        if result:
            logger.info('Research: weekly report sent')
        return bool(result)
    except Exception as e:
        logger.error(f'Research: Telegram send failed: {e}')
        return False
