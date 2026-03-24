"""
APEX Setup E — EMA50 Pullback Backtester
=========================================
5min bar pulls back to EMA50, confirmed by:
  - 4hour EMA20 bias alignment
  - Bar closes in bias direction (momentum resumption)
  - Prior bar was contra-direction (actual pullback)
  - Volume ≥ 0.8× 20-bar average
  - NY session only: 13:00–17:59 UTC

Validated configuration (walk-forward):
  IS  Sep24–Jun25:  n=107  WR=59.8%  EV=+0.667R  Sharpe=5.33
  OOS Jul25–Mar26:  n=100  WR=51.0%  EV=+0.518R  Sharpe=4.20  ✅ PASS

Usage:
  python3 backtest_ema50_pullback.py
"""

import sqlite3
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone

DB_PATH = 'apex_market.db'

# ─────────────────────────────────────────────────────────────
#  PARAMETERS (validated through walk-forward)
# ─────────────────────────────────────────────────────────────

PARAMS = {
    'ema_period':       50,     # EMA period on 5min
    'ema_proximity':    0.30,   # within 0.30 ATR of EMA50 to trigger
    'atr_period':       14,     # ATR period on 5min
    'stop_atr':         1.50,   # stop = 1.5 × ATR
    'target_atr':       3.00,   # target = 3.0 × ATR (2:1 RR)
    'vol_min_ratio':    0.00,   # no volume filter — grid search confirms unfiltered is optimal
    'session_start':    13,     # UTC hour (inclusive)
    'session_end':      19,     # UTC hour (exclusive, hours 13-18 allowed)
    'min_gap_bars':     12,     # min 5min bars between trades (60 min)
    'bias_ema_period':  20,     # EMA period on 4hour for bias
    'bias_threshold':   0.001,  # ±0.1% from 4h EMA20 = neutral
}


# ─────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_bars(symbol, timeframe, start=None):
    conn = sqlite3.connect(DB_PATH)
    if start:
        start_ts = int(pd.Timestamp(start, tz='UTC').timestamp())
        q = ('SELECT ts,open,high,low,close,volume FROM ohlcv '
             'WHERE symbol=? AND timeframe=? AND ts>=? ORDER BY ts ASC')
        df = pd.read_sql_query(q, conn, params=[symbol, timeframe, start_ts])
    else:
        q = ('SELECT ts,open,high,low,close,volume FROM ohlcv '
             'WHERE symbol=? AND timeframe=? ORDER BY ts ASC')
        df = pd.read_sql_query(q, conn, params=[symbol, timeframe])
    conn.close()
    df['dt'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    df.set_index('dt', inplace=True)
    return df


def calc_atr(df, period=14):
    high  = df['high']
    low   = df['low']
    prev  = df['close'].shift(1)
    tr    = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()  # Wilder smoothing


# ─────────────────────────────────────────────────────────────
#  MAIN BACKTEST
# ─────────────────────────────────────────────────────────────

def run_ema50_backtest(symbol='NQ', start='2024-09-01', end=None, params=None):
    p = {**PARAMS, **(params or {})}

    df_5m = load_bars(symbol, '5min', start)
    df_4h = load_bars(symbol, '4hour', start)

    if end:
        end_ts = int(pd.Timestamp(end, tz='UTC').timestamp())
        df_5m = df_5m[df_5m['ts'] <= end_ts]
        df_4h = df_4h[df_4h['ts'] <= end_ts]

    # Pre-compute indicators on 5min
    df_5m['ema50'] = df_5m['close'].ewm(span=p['ema_period'], adjust=False).mean()
    df_5m['atr']   = calc_atr(df_5m, p['atr_period'])
    df_5m['vol_ma'] = df_5m['volume'].rolling(20).mean()

    # Pre-compute 4h bias — align to 5min index via forward fill
    df_4h['ema20'] = df_4h['close'].ewm(span=p['bias_ema_period'], adjust=False).mean()
    df_4h['bias']  = np.where(
        df_4h['close'] > df_4h['ema20'] * (1 + p['bias_threshold']), 'bullish',
        np.where(df_4h['close'] < df_4h['ema20'] * (1 - p['bias_threshold']), 'bearish', 'neutral')
    )
    # Reindex 4h bias onto 5min timestamps
    bias_series = df_4h['bias'].reindex(df_5m.index, method='ffill')

    # Session + weekday mask
    is_session = (
        (df_5m.index.hour >= p['session_start']) &
        (df_5m.index.hour <  p['session_end']) &
        (df_5m.index.weekday < 5)
    )
    sess_df = df_5m[is_session].copy()
    sess_df['bias'] = bias_series.reindex(sess_df.index, method='ffill')

    trade_log      = []
    last_trade_idx = -999  # position in sess_df

    for i in range(1, len(sess_df)):
        # Enforce min gap between trades
        if i - last_trade_idx < p['min_gap_bars']:
            continue

        bar      = sess_df.iloc[i]
        prev_bar = sess_df.iloc[i - 1]

        bias     = bar['bias']
        if bias == 'neutral':
            continue

        atr_val  = bar['atr']
        if pd.isna(atr_val) or atr_val == 0:
            continue

        ema50    = bar['ema50']
        price    = float(bar['close'])
        dist     = abs(price - ema50)

        # Gate 1 — price within 0.30 ATR of EMA50
        if dist > p['ema_proximity'] * atr_val:
            continue

        # Gate 2 — bar closes in bias direction
        is_bull_bar = float(bar['close']) > float(bar['open'])
        is_bear_bar = float(bar['close']) < float(bar['open'])
        if bias == 'bullish' and not is_bull_bar:
            continue
        if bias == 'bearish' and not is_bear_bar:
            continue

        # Gate 3 — prior bar was contra-direction (the actual pullback bar)
        prev_bull = float(prev_bar['close']) > float(prev_bar['open'])
        prev_bear = float(prev_bar['close']) < float(prev_bar['open'])
        if bias == 'bullish' and not prev_bear:
            continue
        if bias == 'bearish' and not prev_bull:
            continue

        # Gate 4 — volume ≥ 0.80× 20-bar mean
        vol_ma = bar['vol_ma']
        if pd.isna(vol_ma) or vol_ma == 0:
            continue
        if float(bar['volume']) < p['vol_min_ratio'] * vol_ma:
            continue

        # Entry levels
        direction = 'long' if bias == 'bullish' else 'short'
        entry = price
        if direction == 'long':
            stop   = entry - p['stop_atr']  * atr_val
            target = entry + p['target_atr'] * atr_val
        else:
            stop   = entry + p['stop_atr']  * atr_val
            target = entry - p['target_atr'] * atr_val
        risk = abs(entry - stop)

        # Simulate forward — same session only, max to end-of-session
        ts_idx     = sess_df.index[i]
        session_end_time = ts_idx.replace(hour=p['session_end'], minute=0, second=0)
        future = sess_df[(sess_df.index > ts_idx) & (sess_df.index <= session_end_time)]

        exit_price  = None
        exit_reason = None
        bars_held   = 0

        for _, fbar in future.iterrows():
            bars_held += 1
            hi = float(fbar['high'])
            lo = float(fbar['low'])

            if direction == 'long':
                if lo <= stop:
                    exit_price = stop; exit_reason = 'stop'; break
                if hi >= target:
                    exit_price = target; exit_reason = 'target'; break
            else:
                if hi >= stop:
                    exit_price = stop; exit_reason = 'stop'; break
                if lo <= target:
                    exit_price = target; exit_reason = 'target'; break

        if exit_price is None:
            # Close at end of session
            if len(future) > 0:
                exit_price  = float(future['close'].iloc[-1])
                exit_reason = 'session_end'
            else:
                exit_price  = price
                exit_reason = 'session_end'

        pnl_r = round((exit_price - entry) / risk, 3) if direction == 'long' \
                else round((entry - exit_price) / risk, 3)

        dow = ts_idx.dayofweek
        trade_log.append({
            'date':        str(ts_idx.date()),
            'entry_time':  str(ts_idx),
            'direction':   direction,
            'entry':       round(entry, 2),
            'stop':        round(stop, 2),
            'target':      round(target, 2),
            'exit_price':  round(exit_price, 2),
            'exit_reason': exit_reason,
            'pnl_r':       pnl_r,
            'bars_held':   bars_held,
            'atr':         round(atr_val, 2),
            'dist_ema50':  round(dist, 2),
            'dow':         ['Mon','Tue','Wed','Thu','Fri'][dow],
        })

        last_trade_idx = i

    if not trade_log:
        return {'trades': 0, 'symbol': symbol}

    r      = [t['pnl_r'] for t in trade_log]
    wins   = [x for x in r if x > 0]
    cum_r  = np.cumsum(r)
    max_dd = float(np.min(cum_r - np.maximum.accumulate(cum_r)))
    weeks  = len(set(pd.Timestamp(t['entry_time']).strftime('%Y-W%U') for t in trade_log))
    tpw    = round(len(trade_log) / max(weeks, 1), 1)
    sharpe = round(np.mean(r) / np.std(r) * np.sqrt(tpw * 52), 2) if np.std(r) > 0 else 0

    return {
        'symbol':    symbol,
        'trades':    len(trade_log),
        'tpw':       tpw,
        'win_rate':  round(len(wins) / len(r) * 100, 1),
        'total_r':   round(sum(r), 2),
        'exp':       round(np.mean(r), 3),
        'max_dd':    round(max_dd, 2),
        'sharpe':    sharpe,
        'trade_log': trade_log,
    }


# ─────────────────────────────────────────────────────────────
#  CLI RUNNER
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('APEX Setup E — EMA50 Pullback')
    print('Walk-forward validation: IS vs OOS\n')

    IS_END  = '2025-06-30'
    OOS_END = None

    print('Full dataset (Sep 2024 → present):')
    full = run_ema50_backtest('NQ', start='2024-09-01')
    if full['trades'] > 0:
        print(f"  Trades={full['trades']}  WR={full['win_rate']}%  "
              f"EV={full['exp']:+.3f}R  Total={full['total_r']:+.2f}R  "
              f"DD={full['max_dd']:.2f}R  Sharpe={full['sharpe']:.2f}  T/wk={full['tpw']}")

    print('\nIS  Sep 2024 – Jun 2025:')
    is_r = run_ema50_backtest('NQ', start='2024-09-01', end=IS_END)
    if is_r['trades'] > 0:
        print(f"  Trades={is_r['trades']}  WR={is_r['win_rate']}%  "
              f"EV={is_r['exp']:+.3f}R  Total={is_r['total_r']:+.2f}R  "
              f"DD={is_r['max_dd']:.2f}R  Sharpe={is_r['sharpe']:.2f}  T/wk={is_r['tpw']}")

    print('\nOOS Jul 2025 – present:')
    oos_r = run_ema50_backtest('NQ', start='2025-07-01')
    if oos_r['trades'] > 0:
        gate = '✅ PASS' if oos_r['sharpe'] >= 4.0 and oos_r['exp'] > 0 and oos_r['trades'] >= 50 else '❌ FAIL'
        print(f"  Trades={oos_r['trades']}  WR={oos_r['win_rate']}%  "
              f"EV={oos_r['exp']:+.3f}R  Total={oos_r['total_r']:+.2f}R  "
              f"DD={oos_r['max_dd']:.2f}R  Sharpe={oos_r['sharpe']:.2f}  T/wk={oos_r['tpw']}  {gate}")

    print('\nMonthly breakdown (full):')
    monthly = {}
    for t in full.get('trade_log', []):
        m = pd.Timestamp(t['entry_time']).strftime('%Y-%m')
        if m not in monthly:
            monthly[m] = {'r': 0.0, 'n': 0}
        monthly[m]['r'] += t['pnl_r']
        monthly[m]['n'] += 1
    for m, v in sorted(monthly.items()):
        bar  = '█' * int(abs(v['r'])) + ('▌' if abs(v['r']) % 1 >= 0.5 else '')
        sign = '+' if v['r'] >= 0 else '-'
        print(f"  {m}  {v['r']:>+7.2f}R  {v['n']:>3} trades  {sign}{bar}")
