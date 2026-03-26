#!/usr/bin/env python3
"""
APEX Trading System — Quantitative Research Script
Parts 1, 2, 3: Feature correlation + 5 pattern walk-forward + comparison table
"""

import sqlite3
import datetime
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

DB_PATH = '/Users/stanleywise/Desktop/apex/apex_market.db'

# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

def load_ohlcv(symbol, timeframe, start_dt=None, end_dt=None):
    conn = sqlite3.connect(DB_PATH)
    q = "SELECT ts, open, high, low, close, volume FROM ohlcv WHERE symbol=? AND timeframe=?"
    params = [symbol, timeframe]
    if start_dt:
        params.append(int(start_dt.timestamp()))
        q += " AND ts >= ?"
    if end_dt:
        params.append(int(end_dt.timestamp()))
        q += " AND ts <= ?"
    q += " ORDER BY ts"
    df = pd.read_sql_query(q, conn, params=params)
    conn.close()
    df['dt'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    df.set_index('dt', inplace=True)
    df.drop(columns=['ts'], inplace=True)
    return df


def wilder_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def session_vwap(df):
    """Compute rolling session VWAP: cumulative from 13:00 UTC each day."""
    df = df.copy()
    df['date'] = df.index.date
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    # session start = 13:00 UTC
    df['session_start'] = df['date'].astype(str)
    # typical price
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_vol'] = df['tp'] * df['volume']

    vwap_vals = []
    cum_tp_vol = 0.0
    cum_vol = 0.0
    prev_date = None
    for idx, row in df.iterrows():
        h = idx.hour
        m = idx.minute
        d = idx.date()
        # Reset at session start (13:00 UTC)
        if d != prev_date or (h == 13 and m == 0):
            if h == 13 and m == 0:
                cum_tp_vol = row['tp_vol']
                cum_vol = row['volume']
                prev_date = d
            elif d != prev_date:
                # new calendar date but not 13:00 - reset anyway
                cum_tp_vol = row['tp_vol']
                cum_vol = row['volume']
                prev_date = d
        else:
            cum_tp_vol += row['tp_vol']
            cum_vol += row['volume']
        if cum_vol > 0:
            vwap_vals.append(cum_tp_vol / cum_vol)
        else:
            vwap_vals.append(np.nan)
    df['vwap'] = vwap_vals
    return df['vwap']


def compute_session_vwap_fast(df):
    """Vectorized session VWAP reset at 13:00 UTC each day."""
    df = df.copy()
    tp = (df['high'] + df['low'] + df['close']) / 3
    tp_vol = tp * df['volume']

    # Create session key: each session starts at 13:00 UTC
    # A new session is when hour==13 and minute==0, or it's a new day
    dates = df.index.date
    hours = df.index.hour
    minutes = df.index.minute

    # Mark session breaks: 13:00 UTC start
    session_break = pd.Series(False, index=df.index)
    for i in range(len(df)):
        if i == 0:
            session_break.iloc[i] = True
        else:
            if dates[i] != dates[i-1]:
                session_break.iloc[i] = True
            elif hours[i] == 13 and minutes[i] == 0:
                session_break.iloc[i] = True

    session_id = session_break.cumsum()

    cum_tp_vol = tp_vol.groupby(session_id).cumsum()
    cum_vol = df['volume'].groupby(session_id).cumsum()
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    return vwap


def trade_stats(results_r):
    """Compute WR, EV, Sharpe from list/array of R-multiples."""
    r = np.array(results_r)
    if len(r) == 0:
        return 0, 0.0, 0.0, 0.0
    wins = r[r > 0]
    wr = len(wins) / len(r)
    ev = r.mean()
    std = r.std(ddof=1) if len(r) > 1 else 1.0
    sharpe = (ev / std * np.sqrt(252)) if std > 0 else 0.0
    return len(r), wr, ev, sharpe


def simulate_trade(df_5min, entry_idx, direction, atr_val, stop_mult=1.5, target_mult=3.0, max_bars=36):
    """
    Simulate a single trade from entry bar close.
    direction: 1=long, -1=short
    Returns R-multiple result.
    """
    entry_price = df_5min['close'].iloc[entry_idx]
    stop = entry_price - direction * stop_mult * atr_val
    target = entry_price + direction * target_mult * atr_val

    for j in range(1, max_bars + 1):
        i = entry_idx + j
        if i >= len(df_5min):
            break
        bar_high = df_5min['high'].iloc[i]
        bar_low = df_5min['low'].iloc[i]

        if direction == 1:
            if bar_low <= stop:
                return -1.0
            if bar_high >= target:
                return +2.0
        else:
            if bar_high >= stop:
                return -1.0
            if bar_low <= target:
                return +2.0

    # timeout: exit at close
    exit_price = df_5min['close'].iloc[min(entry_idx + max_bars, len(df_5min)-1)]
    pnl = direction * (exit_price - entry_price)
    r_val = pnl / (stop_mult * atr_val) if atr_val > 0 else 0
    return r_val


# ─────────────────────────────────────────────────────────
# PART 1 — Feature Engineering & Correlation Analysis
# ─────────────────────────────────────────────────────────
print("=" * 70)
print("PART 1 — FEATURE ENGINEERING & CORRELATION ANALYSIS (NQ 5min)")
print("=" * 70)

start_dt = datetime.datetime(2024, 9, 1, tzinfo=datetime.timezone.utc)
df5 = load_ohlcv('NQ', '5min', start_dt=start_dt)
print(f"  Loaded {len(df5)} bars of NQ 5min from {df5.index[0]} to {df5.index[-1]}")

# Filter to session hours 13:00-18:59 UTC weekdays
df5_sess = df5[(df5.index.hour >= 13) & (df5.index.hour < 19) &
               (df5.index.weekday < 5)].copy()
print(f"  Session-filtered bars: {len(df5_sess)}")

# Compute ATR on full data first for continuity
atr_full = wilder_atr(df5, 14)
df5_sess['atr'] = atr_full.reindex(df5_sess.index)

# EMA on full data
ema20_full = compute_ema(df5['close'], 20)
ema50_full = compute_ema(df5['close'], 50)
df5_sess['ema20'] = ema20_full.reindex(df5_sess.index)
df5_sess['ema50'] = ema50_full.reindex(df5_sess.index)

# RSI on full data
rsi_full = compute_rsi(df5['close'], 14)
df5_sess['rsi'] = rsi_full.reindex(df5_sess.index)

# VWAP (session-based)
print("  Computing session VWAP...")
vwap_full = compute_session_vwap_fast(df5)
df5_sess['vwap'] = vwap_full.reindex(df5_sess.index)

# Features
df5_sess['atr_safe'] = df5_sess['atr'].replace(0, np.nan)
df5_sess['ema20_dist'] = (df5_sess['close'] - df5_sess['ema20']) / df5_sess['atr_safe']
df5_sess['ema50_dist'] = (df5_sess['close'] - df5_sess['ema50']) / df5_sess['atr_safe']
df5_sess['vwap_dist'] = (df5_sess['close'] - df5_sess['vwap']) / df5_sess['atr_safe']
df5_sess['rsi_14'] = (df5_sess['rsi'] - 50) / 50
df5_sess['atr_ratio'] = df5_sess['atr'] / df5_sess['atr'].rolling(20).mean()
df5_sess['body_ratio'] = (df5_sess['close'] - df5_sess['open']).abs() / df5_sess['atr_safe']
df5_sess['vol_ratio'] = df5_sess['volume'] / df5_sess['volume'].rolling(20).mean()
df5_sess['momentum_3'] = (df5_sess['close'] - df5_sess['close'].shift(3)) / df5_sess['atr_safe']
df5_sess['momentum_10'] = (df5_sess['close'] - df5_sess['close'].shift(10)) / df5_sess['atr_safe']
df5_sess['wick_ratio'] = ((df5_sess['high'] - df5_sess['low']) - (df5_sess['close'] - df5_sess['open']).abs()) / df5_sess['atr_safe']

# Simulate next-bar pnl_r: 2:1 RR, long if rises 2*ATR in 12 bars, short if drops, else 0
print("  Simulating next-bar pnl_r (2:1 RR, 12-bar window)...")
df5_sess = df5_sess.reset_index()

pnl_r = []
atr_arr = df5_sess['atr'].values
close_arr = df5_sess['close'].values
high_arr = df5_sess['high'].values
low_arr = df5_sess['low'].values

for i in range(len(df5_sess) - 12):
    entry = close_arr[i]
    atr_v = atr_arr[i]
    if np.isnan(atr_v) or atr_v == 0:
        pnl_r.append(np.nan)
        continue
    tgt_up = entry + 2 * atr_v
    tgt_dn = entry - 2 * atr_v
    result = 0
    for j in range(1, 13):
        if high_arr[i+j] >= tgt_up:
            result = 1
            break
        if low_arr[i+j] <= tgt_dn:
            result = -1
            break
    pnl_r.append(result)

pnl_r.extend([np.nan] * 12)
df5_sess['pnl_r'] = pnl_r

feature_cols = ['ema20_dist', 'ema50_dist', 'vwap_dist', 'rsi_14',
                'atr_ratio', 'body_ratio', 'vol_ratio',
                'momentum_3', 'momentum_10', 'wick_ratio']

print("\n  Pearson Correlations with next-bar pnl_r:")
print(f"  {'Feature':<18} {'Corr':>10}  {'N':>8}")
print("  " + "-" * 42)

corr_results = []
for feat in feature_cols:
    subset = df5_sess[[feat, 'pnl_r']].dropna()
    subset = subset[subset['pnl_r'] != 0]  # exclude no-signal bars
    if len(subset) < 100:
        continue
    corr = subset[feat].corr(subset['pnl_r'])
    corr_results.append((feat, corr, len(subset)))

corr_results.sort(key=lambda x: abs(x[1]), reverse=True)
for feat, corr, n in corr_results:
    marker = " ◄" if corr_results.index((feat, corr, n)) < 5 else ""
    print(f"  {feat:<18} {corr:>10.4f}  {n:>8,}{marker}")

print(f"\n  Top 5 by |correlation|:")
for feat, corr, n in corr_results[:5]:
    print(f"    {feat:<18} corr={corr:+.4f}  n={n:,}")


# ─────────────────────────────────────────────────────────
# PART 2 — Walk-Forward Backtests
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PART 2 — WALK-FORWARD PATTERN BACKTESTS (NQ 5min)")
print("=" * 70)

# Reload full 5min data with index
df5 = load_ohlcv('NQ', '5min', start_dt=start_dt)

# Compute all indicators on full data
atr_s = wilder_atr(df5, 14)
ema20_s = compute_ema(df5['close'], 20)
ema50_s = compute_ema(df5['close'], 50)
rsi_s = compute_rsi(df5['close'], 14)
vwap_s = compute_session_vwap_fast(df5)
vol_ma20_s = df5['volume'].rolling(20).mean()

df5['atr'] = atr_s
df5['ema20'] = ema20_s
df5['ema50'] = ema50_s
df5['rsi'] = rsi_s
df5['vwap'] = vwap_s
df5['vol_ma20'] = vol_ma20_s

# Load 4h NQ data for bias
df4h = load_ohlcv('NQ', '4hour', start_dt=start_dt)
ema20_4h = compute_ema(df4h['close'], 20)
df4h['ema20'] = ema20_4h

def get_4h_bias(ts):
    """Get 4h EMA20 bias at time ts: +1 if close>ema20, -1 otherwise."""
    # Ensure ts is UTC-aware for comparison with df4h.index
    if ts.tzinfo is None:
        ts_utc = ts.tz_localize('UTC')
    else:
        ts_utc = ts.tz_convert('UTC')
    mask = df4h.index <= ts_utc
    if not mask.any():
        return 0
    row = df4h[mask].iloc[-1]
    if row['close'] > row['ema20']:
        return 1
    else:
        return -1


# Session filter: 13:00-18:59 UTC weekdays
def is_session(ts):
    return ts.hour >= 13 and ts.hour < 19 and ts.weekday() < 5


# Splits
IS_START  = datetime.datetime(2024, 9, 1,  tzinfo=datetime.timezone.utc)
IS_END    = datetime.datetime(2025, 6, 30, 23, 59, tzinfo=datetime.timezone.utc)
OOS_START = datetime.datetime(2025, 7, 1,  tzinfo=datetime.timezone.utc)
OOS_END   = datetime.datetime(2026, 3, 24, tzinfo=datetime.timezone.utc)

STOP_MULT   = 1.5
TARGET_MULT = 3.0
MIN_GAP     = 12  # bars


def format_stats(label, trades):
    if len(trades) == 0:
        return f"  {label}: n=0  (no trades)"
    r = np.array(trades)
    wr = (r > 0).mean()
    ev = r.mean()
    std = r.std(ddof=1) if len(r) > 1 else 1e-9
    sharpe = ev / std * np.sqrt(252)
    return f"  {label}: n={len(r):4d}  WR={wr:.1%}  EV={ev:+.3f}R  Sharpe={sharpe:+.2f}"


def run_backtest_full(trades_with_dt):
    """
    trades_with_dt: list of (dt, R) tuples
    Returns full, IS, OOS trade lists, monthly OOS breakdown.
    """
    def to_utc(dt):
        if dt.tzinfo is None:
            return dt.tz_localize('UTC')
        return dt

    all_r = [r for dt, r in trades_with_dt]
    is_r  = [r for dt, r in trades_with_dt if IS_START <= to_utc(dt) <= IS_END]
    oos_r = [r for dt, r in trades_with_dt if OOS_START <= to_utc(dt) <= OOS_END]

    # Monthly OOS
    monthly = {}
    for dt, r in trades_with_dt:
        dt_utc = to_utc(dt)
        if OOS_START <= dt_utc <= OOS_END:
            key = dt.strftime('%Y-%m')
            if key not in monthly:
                monthly[key] = []
            monthly[key].append(r)

    return all_r, is_r, oos_r, monthly


def oos_gate(oos_r):
    if len(oos_r) == 0:
        return False, "n=0"
    r = np.array(oos_r)
    ev = r.mean()
    std = r.std(ddof=1) if len(r) > 1 else 1e-9
    sharpe = ev / std * np.sqrt(252)
    n = len(r)
    if sharpe >= 4.0 and ev > 0 and n >= 50:
        return True, f"Sharpe={sharpe:+.2f} EV={ev:+.3f} n={n}"
    else:
        return False, f"Sharpe={sharpe:+.2f} EV={ev:+.3f} n={n}"


def print_pattern_results(name, trades_with_dt, total_bars, session_days):
    all_r, is_r, oos_r, monthly = run_backtest_full(trades_with_dt)

    # trades per week
    weeks = session_days / 5.0 if session_days > 0 else 1
    t_per_wk = len(all_r) / weeks if weeks > 0 else 0

    print(f"\n  {'─'*60}")
    print(f"  {name}")
    print(f"  {'─'*60}")

    def stats_line(label, r_list):
        if len(r_list) == 0:
            print(f"  {label:<25} n=   0  WR=  n/a  EV=    n/a  Sharpe=   n/a")
            return
        r = np.array(r_list)
        wr = (r > 0).mean()
        ev = r.mean()
        std = r.std(ddof=1) if len(r) > 1 else 1e-9
        sharpe = ev / std * np.sqrt(252)
        extra = f"  T/wk={t_per_wk:.1f}" if label == "Full dataset" else ""
        print(f"  {label:<25} n={len(r):4d}  WR={wr:.1%}  EV={ev:+.3f}R  Sharpe={sharpe:+.2f}{extra}")

    stats_line("Full dataset", all_r)
    stats_line("IS  (Sep24-Jun25)", is_r)
    stats_line("OOS (Jul25-present)", oos_r)

    passed, gate_str = oos_gate(oos_r)
    gate_label = "PASS ✓" if passed else "FAIL ✗"
    print(f"  OOS Gate: {gate_label}  [{gate_str}]")

    if passed and monthly:
        print(f"\n  OOS Monthly Breakdown:")
        print(f"  {'Month':<10} {'n':>5}  {'WR':>7}  {'EV':>9}  {'Sharpe':>9}")
        for month in sorted(monthly.keys()):
            mr = np.array(monthly[month])
            mwr = (mr > 0).mean()
            mev = mr.mean()
            mstd = mr.std(ddof=1) if len(mr) > 1 else 1e-9
            msharpe = mev / mstd * np.sqrt(252)
            print(f"  {month:<10} {len(mr):>5}  {mwr:>7.1%}  {mev:>+9.3f}R  {msharpe:>+9.2f}")

    return all_r, is_r, oos_r, monthly, passed


# ── Prepare index arrays for fast iteration ──
df5_reset = df5.reset_index()
dt_arr    = df5_reset['dt'].values           # numpy datetime64
close_arr = df5_reset['close'].values
open_arr  = df5_reset['open'].values
high_arr  = df5_reset['high'].values
low_arr   = df5_reset['low'].values
vol_arr   = df5_reset['volume'].values
atr_arr   = df5_reset['atr'].values
rsi_arr   = df5_reset['rsi'].values
vwap_arr  = df5_reset['vwap'].values
vol_ma_arr= df5_reset['vol_ma20'].values
ema20_arr = df5_reset['ema20'].values

N = len(df5_reset)

# Pre-compute session filter mask
from pandas import Timestamp
dt_pd = pd.DatetimeIndex(dt_arr)
sess_mask = (dt_pd.hour >= 13) & (dt_pd.hour < 19) & (dt_pd.weekday < 5)

# Count session days
session_days = len(set(dt_pd[sess_mask].date))
print(f"\n  Session days in dataset: {session_days}")

# ─────────────────────────────────────────────────────────
# PATTERN 1 — VWAP Reversion
# ─────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("PATTERN 1 — VWAP Reversion")

trades_p1 = []
last_trade_bar = -MIN_GAP - 1

for i in range(20, N - int(TARGET_MULT) - 1):
    if not sess_mask[i]:
        continue
    if i - last_trade_bar < MIN_GAP:
        continue

    atr_v = atr_arr[i]
    if np.isnan(atr_v) or atr_v <= 0:
        continue

    vwap_v = vwap_arr[i]
    if np.isnan(vwap_v):
        continue

    rsi_v = rsi_arr[i]
    if np.isnan(rsi_v):
        continue

    close_v = close_arr[i]
    dist = (close_v - vwap_v) / atr_v

    direction = 0
    # Long: price > 1.5 ATR BELOW vwap → wait, spec says:
    # price > 1.5×ATR above VWAP → short (mean reversion down)
    # price < 1.5×ATR below VWAP → long (mean reversion up)
    if dist >= 1.5 and rsi_v > 65:
        direction = -1  # short
    elif dist <= -1.5 and rsi_v < 35:
        direction = 1   # long

    if direction == 0:
        continue

    # Simulate trade
    r_val = simulate_trade(df5_reset, i, direction, atr_v, STOP_MULT, TARGET_MULT)
    ts = Timestamp(dt_arr[i])
    trades_p1.append((ts, r_val))
    last_trade_bar = i

print(f"  Total trades found: {len(trades_p1)}")
results_p1 = print_pattern_results("Pattern 1 — VWAP Reversion", trades_p1, N, session_days)

# ─────────────────────────────────────────────────────────
# PATTERN 2 — Volume Climax Reversal
# ─────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("PATTERN 2 — Volume Climax Reversal")

trades_p2 = []
last_trade_bar = -MIN_GAP - 1

for i in range(20, N - 1):
    if not sess_mask[i]:
        continue
    if i - last_trade_bar < MIN_GAP:
        continue

    atr_v = atr_arr[i]
    if np.isnan(atr_v) or atr_v <= 0:
        continue

    vol_ma = vol_ma_arr[i]
    if np.isnan(vol_ma) or vol_ma <= 0:
        continue

    vol_v = vol_arr[i]
    if vol_v < 3 * vol_ma:
        continue

    bar_range = high_arr[i] - low_arr[i]
    if bar_range <= 0:
        continue

    body = abs(close_arr[i] - open_arr[i])
    upper_wick = high_arr[i] - max(close_arr[i], open_arr[i])
    lower_wick = min(close_arr[i], open_arr[i]) - low_arr[i]

    upper_wick_pct = upper_wick / bar_range
    lower_wick_pct = lower_wick / bar_range

    direction = 0
    if upper_wick_pct > 0.6:
        direction = -1  # bearish wick rejection → short
    elif lower_wick_pct > 0.6:
        direction = 1   # bullish wick rejection → long

    if direction == 0:
        continue

    # 4h bias required
    ts = Timestamp(dt_arr[i])
    bias = get_4h_bias(ts)
    if bias != direction:
        continue

    r_val = simulate_trade(df5_reset, i, direction, atr_v, STOP_MULT, TARGET_MULT)
    trades_p2.append((ts, r_val))
    last_trade_bar = i

print(f"  Total trades found: {len(trades_p2)}")
results_p2 = print_pattern_results("Pattern 2 — Volume Climax Reversal", trades_p2, N, session_days)

# ─────────────────────────────────────────────────────────
# PATTERN 3 — Inside Bar Breakout
# ─────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("PATTERN 3 — Inside Bar Breakout")

trades_p3 = []
last_trade_bar = -MIN_GAP - 1

for i in range(1, N - 1):
    # i is potential breakout bar (bar after inside bar at i-1)
    # Check if bar i-1 is an inside bar relative to i-2
    if i < 2:
        continue
    if not sess_mask[i]:
        continue
    if i - last_trade_bar < MIN_GAP:
        continue

    # Inside bar at i-1: contained within i-2
    if not (high_arr[i-1] < high_arr[i-2] and low_arr[i-1] > low_arr[i-2]):
        continue

    atr_v = atr_arr[i]
    if np.isnan(atr_v) or atr_v <= 0:
        continue

    ib_high = high_arr[i-1]
    ib_low  = low_arr[i-1]

    # Check breakout at bar i
    direction = 0
    if close_arr[i] > ib_high:
        direction = 1   # long breakout
    elif close_arr[i] < ib_low:
        direction = -1  # short breakout

    if direction == 0:
        continue

    # 4h bias
    ts = Timestamp(dt_arr[i])
    bias = get_4h_bias(ts)
    if bias != direction:
        continue

    r_val = simulate_trade(df5_reset, i, direction, atr_v, STOP_MULT, TARGET_MULT)
    trades_p3.append((ts, r_val))
    last_trade_bar = i

print(f"  Total trades found: {len(trades_p3)}")
results_p3 = print_pattern_results("Pattern 3 — Inside Bar Breakout", trades_p3, N, session_days)

# ─────────────────────────────────────────────────────────
# PATTERN 4 — Prior Day High/Low Test
# ─────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("PATTERN 4 — Prior Day High/Low Test")

# Compute prior day high/low from daily data
df1d = load_ohlcv('NQ', '1day', start_dt=start_dt - datetime.timedelta(days=5))
df1d['date'] = df1d.index.date
daily_hl = {row['date']: (row['high'], row['low']) for _, row in df1d.reset_index().iterrows()}

trades_p4 = []
last_trade_bar = -MIN_GAP - 1

for i in range(20, N - 1):
    if not sess_mask[i]:
        continue
    if i - last_trade_bar < MIN_GAP:
        continue

    atr_v = atr_arr[i]
    if np.isnan(atr_v) or atr_v <= 0:
        continue

    ts = Timestamp(dt_arr[i])
    cur_date = ts.date()

    # Find prior trading day
    check_d = cur_date - datetime.timedelta(days=1)
    attempts = 0
    while check_d not in daily_hl and attempts < 5:
        check_d -= datetime.timedelta(days=1)
        attempts += 1

    if check_d not in daily_hl:
        continue

    pd_high, pd_low = daily_hl[check_d]

    # Check proximity and direction
    direction = 0
    # Long: bar low touches within 0.2*ATR of prior day low AND close is bullish
    if (abs(low_arr[i] - pd_low) <= 0.2 * atr_v and
            close_arr[i] > open_arr[i]):
        direction = 1
    # Short: bar high touches within 0.2*ATR of prior day high AND close is bearish
    elif (abs(high_arr[i] - pd_high) <= 0.2 * atr_v and
              close_arr[i] < open_arr[i]):
        direction = -1

    if direction == 0:
        continue

    # 4h bias
    bias = get_4h_bias(ts)
    if bias != direction:
        continue

    r_val = simulate_trade(df5_reset, i, direction, atr_v, STOP_MULT, TARGET_MULT)
    trades_p4.append((ts, r_val))
    last_trade_bar = i

print(f"  Total trades found: {len(trades_p4)}")
results_p4 = print_pattern_results("Pattern 4 — Prior Day High/Low Test", trades_p4, N, session_days)

# ─────────────────────────────────────────────────────────
# PATTERN 5 — Opening Range Breakout (ORB)
# ─────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("PATTERN 5 — Opening Range Breakout (ORB)")

trades_p5 = []
last_trade_bar = -MIN_GAP - 1

# Group by date to build ORB
from collections import defaultdict
date_bars = defaultdict(list)  # date -> list of bar indices

for i in range(N):
    ts = Timestamp(dt_arr[i])
    if ts.weekday() < 5:  # weekdays only
        date_bars[ts.date()].append(i)

for date, bar_indices in sorted(date_bars.items()):
    # Find bars 13:00-13:29 UTC (first 6 × 5min bars of session)
    orb_bars = []
    for idx in bar_indices:
        ts = Timestamp(dt_arr[idx])
        if ts.hour == 13 and ts.minute < 30:
            orb_bars.append(idx)

    if len(orb_bars) < 3:  # need at least 3 bars to establish range
        continue

    orb_high = max(high_arr[b] for b in orb_bars)
    orb_low  = min(low_arr[b]  for b in orb_bars)

    if orb_high <= orb_low:
        continue

    # Find subsequent session bars
    orb_end_bar = orb_bars[-1]
    long_taken  = False
    short_taken = False

    for idx in bar_indices:
        ts = Timestamp(dt_arr[idx])
        if idx <= orb_end_bar:
            continue
        if ts.hour < 13 or ts.hour >= 19:
            continue
        if idx - last_trade_bar < MIN_GAP:
            continue

        atr_v = atr_arr[idx]
        if np.isnan(atr_v) or atr_v <= 0:
            continue

        direction = 0
        if not long_taken and close_arr[idx] > orb_high:
            direction = 1
            long_taken = True
        elif not short_taken and close_arr[idx] < orb_low:
            direction = -1
            short_taken = True

        if direction == 0:
            continue

        r_val = simulate_trade(df5_reset, idx, direction, atr_v, STOP_MULT, TARGET_MULT)
        trades_p5.append((ts, r_val))
        last_trade_bar = idx

print(f"  Total trades found: {len(trades_p5)}")
results_p5 = print_pattern_results("Pattern 5 — Opening Range Breakout", trades_p5, N, session_days)

# ─────────────────────────────────────────────────────────
# PART 3 — Comparison Table
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PART 3 — PATTERN COMPARISON TABLE (ranked by OOS Sharpe)")
print("=" * 70)

def oos_sharpe_ev(oos_r):
    if len(oos_r) == 0:
        return 0.0, 0.0, 0
    r = np.array(oos_r)
    ev = r.mean()
    std = r.std(ddof=1) if len(r) > 1 else 1e-9
    sharpe = ev / std * np.sqrt(252)
    return sharpe, ev, len(r)

all_patterns = [
    ("Pattern 1 — VWAP Reversion",        results_p1),
    ("Pattern 2 — Volume Climax",          results_p2),
    ("Pattern 3 — Inside Bar Breakout",    results_p3),
    ("Pattern 4 — Prior Day H/L Test",     results_p4),
    ("Pattern 5 — ORB",                    results_p5),
    ("Setup E (reference)",                None),   # reference — benchmark
]

# Build table rows
table_rows = []
for name, result in all_patterns:
    if result is None:
        # Setup E reference
        table_rows.append((name, 5.80, 0.120, 87, True, "PASS (reference)"))
        continue
    all_r, is_r, oos_r, monthly, passed = result
    sharpe, ev, n = oos_sharpe_ev(oos_r)
    gate = "PASS ✓" if passed else "FAIL ✗"
    table_rows.append((name, sharpe, ev, n, passed, gate))

# Sort by OOS Sharpe descending
table_rows.sort(key=lambda x: x[1], reverse=True)

print(f"\n  {'#':<3} {'Pattern':<38} {'OOS Sharpe':>11} {'OOS EV':>9} {'OOS n':>7}  {'Gate':<10}")
print("  " + "─" * 82)
for rank, (name, sharpe, ev, n, passed, gate) in enumerate(table_rows, 1):
    print(f"  {rank:<3} {name:<38} {sharpe:>+11.2f} {ev:>+9.3f} {n:>7}  {gate}")

print("\n  OOS Gate criteria: Sharpe ≥ 4.0, EV > 0, n ≥ 50")
print("\n" + "=" * 70)
print("RESEARCH COMPLETE")
print("=" * 70)
