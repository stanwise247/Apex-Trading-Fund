"""
APEX Setup D — Opening Range Breakout Backtester
=================================================
Independent backtest — does not touch live system.
Range: First 30 minutes of NY session (13:30-14:00 UTC)
Instruments: NQ, ES
"""

import sqlite3
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone, timedelta
from collections import defaultdict

DB_PATH = 'apex_market.db'

# ─────────────────────────────────────────────────────────────
#  PARAMETERS
# ─────────────────────────────────────────────────────────────

PARAMS = {
    'range_start_utc':   13.5,    # 13:30 UTC
    'range_end_utc':     14.0,    # 14:00 UTC
    'entry_deadline_utc':16.5,    # 16:30 UTC — latest entry
    'session_end_utc':   19.0,    # 19:00 UTC — force exit
    'min_range_atr':     0.5,     # range must be >= 0.5x ATR
    'max_range_atr':     2.0,     # range must be <= 2.0x ATR
    'break_body_only':   True,    # close must be beyond range (not just wick)
    'volume_filter':     True,    # break candle volume > 20-bar avg
    'volume_multiplier': 1.0,     # volume must be >= 1.0x average
    'target_multiplier': 1.5,     # target = 1.5x range size
    'stop_midpoint':     True,    # stop at range midpoint
    'max_bars':          78,      # max bars held (6.5 hours of 5min bars)
    'skip_days':         [],      # day filters — test without first
    'gap_filter_atr':    1.5,     # skip if open gaps > 1.5x ATR
    'prior_vol_filter':  3.0,     # skip if prior day range > 3x avg range
}

# ─────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_bars(symbol: str, timeframe: str, start: str = None) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    query = 'SELECT ts, open, high, low, close, volume FROM ohlcv WHERE symbol=? AND timeframe=? ORDER BY ts ASC'
    params = [symbol, timeframe]
    if start:
        start_ts = int(pd.Timestamp(start, tz='UTC').timestamp())
        query = 'SELECT ts, open, high, low, close, volume FROM ohlcv WHERE symbol=? AND timeframe=? AND ts>=? ORDER BY ts ASC'
        params.append(start_ts)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    df['dt'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    df.set_index('dt', inplace=True)
    return df


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high  = df['high']
    low   = df['low']
    close = df['close'].shift(1)
    tr    = pd.concat([high - low,
                       (high - close).abs(),
                       (low  - close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ─────────────────────────────────────────────────────────────
#  CORE BACKTEST
# ─────────────────────────────────────────────────────────────

def run_orb_backtest(symbol: str, start: str = '2024-09-01',
                     end: str = None, params: dict = None) -> dict:

    if params is None:
        params = PARAMS

    print(f'\n{"="*60}')
    print(f'  ORB Backtest — {symbol}')
    print(f'  {start} to {end or "latest"}')
    print(f'{"="*60}')

    # Load 5min bars
    df5 = load_bars(symbol, '5min', start)
    if end:
        end_ts = int(pd.Timestamp(end, tz='UTC').timestamp())
        df5 = df5[df5['ts'] <= end_ts]

    print(f'  5min bars loaded: {len(df5)}')

    # Calculate ATR on 5min
    atr = calc_atr(df5, period=14)

    trade_log  = []
    daily_stats = defaultdict(dict)

    # Group by date
    df5['date'] = df5.index.date
    dates = sorted(df5['date'].unique())

    for date in dates:
        day_df = df5[df5['date'] == date].copy()
        dow    = pd.Timestamp(date).dayofweek  # 0=Mon

        # Skip weekends
        if dow >= 5:
            continue

        # Skip filtered days
        if dow in params.get('skip_days', []):
            continue

        # Get bars in range window (13:30-14:00 UTC)
        range_start = params['range_start_utc']
        range_end   = params['range_end_utc']

        range_bars = day_df[
            (day_df.index.hour + day_df.index.minute/60 >= range_start) &
            (day_df.index.hour + day_df.index.minute/60 <  range_end)
        ]

        if len(range_bars) < 3:
            continue

        # Define opening range
        or_high = float(range_bars['high'].max())
        or_low  = float(range_bars['low'].min())
        or_size = or_high - or_low
        or_mid  = (or_high + or_low) / 2

        # Get ATR at range end
        range_end_idx = range_bars.index[-1]
        atr_val = float(atr.loc[range_end_idx]) if range_end_idx in atr.index else None
        if atr_val is None or atr_val == 0:
            continue

        # Range size filter
        if or_size < params['min_range_atr'] * atr_val:
            continue
        if or_size > params['max_range_atr'] * atr_val:
            continue

        # Gap filter — check open vs prior close
        prior_days = [d for d in dates if d < date]
        if prior_days and params.get('gap_filter_atr', 0) > 0:
            prior_day_df = df5[df5['date'] == prior_days[-1]]
            if len(prior_day_df) > 0:
                prior_close = float(prior_day_df['close'].iloc[-1])
                day_open    = float(day_df['open'].iloc[0])
                gap         = abs(day_open - prior_close)
                if gap > params['gap_filter_atr'] * atr_val:
                    continue

        # Prior day volatility filter
        if prior_days and params.get('prior_vol_filter', 0) > 0:
            prior_day_df = df5[df5['date'] == prior_days[-1]]
            if len(prior_day_df) > 0:
                prior_range = float(prior_day_df['high'].max()) - float(prior_day_df['low'].min())
                # Calculate avg daily range over last 10 days
                recent_days = [d for d in prior_days[-10:]]
                ranges = []
                for rd in recent_days:
                    rd_df = df5[df5['date'] == rd]
                    if len(rd_df) > 0:
                        ranges.append(float(rd_df['high'].max()) - float(rd_df['low'].min()))
                if ranges:
                    avg_range = np.mean(ranges)
                    if avg_range > 0 and prior_range > params['prior_vol_filter'] * avg_range:
                        continue

        # Volume baseline — 20-bar average at range end
        vol_baseline = None
        if params.get('volume_filter'):
            vol_window = day_df[day_df.index <= range_end_idx]['volume'].tail(20)
            if len(vol_window) >= 5:
                vol_baseline = float(vol_window.mean())

        # Get post-range bars for entry scanning
        entry_deadline = params['entry_deadline_utc']
        session_end    = params['session_end_utc']

        post_range = day_df[
            day_df.index.hour + day_df.index.minute/60 >= range_end
        ]

        trade_taken = False

        for i, (idx, bar) in enumerate(post_range.iterrows()):
            bar_hour = idx.hour + idx.minute/60

            if bar_hour > entry_deadline:
                break

            if trade_taken:
                break

            bar_close  = float(bar['close'])
            bar_open   = float(bar['open'])
            bar_volume = float(bar['volume'])

            # Check for breakout
            direction = None

            # Long breakout — close above OR high
            if bar_close > or_high:
                if not params.get('break_body_only') or bar_open < bar_close:
                    direction = 'long'

            # Short breakout — close below OR low
            elif bar_close < or_low:
                if not params.get('break_body_only') or bar_open > bar_close:
                    direction = 'short'

            if direction is None:
                continue

            # Volume filter
            if vol_baseline and params.get('volume_filter'):
                if bar_volume < params['volume_multiplier'] * vol_baseline:
                    continue

            # False break filter — check if opposite break happened in last 3 bars
            false_break = False
            if i >= 3:
                recent = list(post_range.iterrows())[max(0,i-3):i]
                for _, rb in recent:
                    if direction == 'long'  and float(rb['close']) < or_low:
                        false_break = True; break
                    if direction == 'short' and float(rb['close']) > or_high:
                        false_break = True; break
            if false_break:
                continue

            # Calculate trade levels
            entry = bar_close
            if params['stop_midpoint']:
                stop = or_mid
            else:
                if direction == 'long':
                    stop = or_low - 0.1 * atr_val
                else:
                    stop = or_high + 0.1 * atr_val

            risk   = abs(entry - stop)
            if risk == 0:
                continue

            target = entry + params['target_multiplier'] * or_size if direction == 'long' \
                     else entry - params['target_multiplier'] * or_size

            rr_planned = abs(target - entry) / risk

            # Simulate trade
            entry_time = idx
            exit_price = None
            exit_reason= None
            bars_held  = 0

            remaining = post_range[post_range.index > idx]
            for j, (exit_idx, exit_bar) in enumerate(remaining.iterrows()):
                bars_held += 1
                exit_hour  = exit_idx.hour + exit_idx.minute/60
                hi = float(exit_bar['high'])
                lo = float(exit_bar['low'])

                # Stop hit
                if direction == 'long'  and lo <= stop:
                    exit_price  = stop
                    exit_reason = 'stop'
                    break
                if direction == 'short' and hi >= stop:
                    exit_price  = stop
                    exit_reason = 'stop'
                    break

                # Target hit
                if direction == 'long'  and hi >= target:
                    exit_price  = target
                    exit_reason = 'target'
                    break
                if direction == 'short' and lo <= target:
                    exit_price  = target
                    exit_reason = 'target'
                    break

                # Session end
                if exit_hour >= session_end:
                    exit_price  = float(exit_bar['close'])
                    exit_reason = 'session_end'
                    break

                # Max bars
                if bars_held >= params['max_bars']:
                    exit_price  = float(exit_bar['close'])
                    exit_reason = 'max_bars'
                    break

            if exit_price is None:
                exit_price  = float(post_range['close'].iloc[-1])
                exit_reason = 'session_end'

            # Calculate P&L
            if direction == 'long':
                pnl_r = round((exit_price - entry) / risk, 3)
            else:
                pnl_r = round((entry - exit_price) / risk, 3)

            trade_log.append({
                'date':       str(date),
                'symbol':     symbol,
                'direction':  direction,
                'entry_time': str(entry_time),
                'entry':      round(entry, 2),
                'stop':       round(stop, 2),
                'target':     round(target, 2),
                'rr_planned': round(rr_planned, 2),
                'or_high':    round(or_high, 2),
                'or_low':     round(or_low, 2),
                'or_size':    round(or_size, 2),
                'atr':        round(atr_val, 2),
                'exit_price': round(exit_price, 2),
                'exit_reason':exit_reason,
                'bars_held':  bars_held,
                'pnl_r':      pnl_r,
                'dow':        ['Mon','Tue','Wed','Thu','Fri'][dow],
            })
            trade_taken = True

    # ── Results ──────────────────────────────────────────────
    if not trade_log:
        print('  No trades found')
        return {}

    r_vals  = [t['pnl_r'] for t in trade_log]
    wins    = [r for r in r_vals if r > 0]
    losses  = [r for r in r_vals if r <= 0]
    cum_r   = np.cumsum(r_vals)
    max_dd  = float(np.min(cum_r - np.maximum.accumulate(cum_r)))
    sharpe  = round(np.mean(r_vals) / np.std(r_vals) * np.sqrt(252), 2) if np.std(r_vals) > 0 else 0

    print(f'  Trades:      {len(trade_log)}')
    print(f'  Win Rate:    {round(len(wins)/len(r_vals)*100,1)}%')
    print(f'  Total R:     {round(sum(r_vals),2)}R')
    print(f'  Expectancy:  {round(np.mean(r_vals),3)}R')
    print(f'  Avg Win:     +{round(np.mean(wins),2)}R') if wins else None
    print(f'  Avg Loss:    {round(np.mean(losses),2)}R') if losses else None
    print(f'  Max DD:      {round(max_dd,2)}R')
    print(f'  Sharpe:      {sharpe}')

    # By exit reason
    print(f'  Exit reasons:')
    for reason in ('stop','target','session_end','max_bars'):
        n = sum(1 for t in trade_log if t['exit_reason']==reason)
        if n: print(f'    {reason}: {n}')

    # By day of week
    print(f'  Day of week:')
    by_dow = defaultdict(list)
    for t in trade_log:
        by_dow[t['dow']].append(t['pnl_r'])
    for dow in ['Mon','Tue','Wed','Thu','Fri']:
        if dow in by_dow:
            dr = by_dow[dow]
            wr = round(sum(1 for x in dr if x>0)/len(dr)*100,1)
            exp= round(np.mean(dr),3)
            print(f'    {dow}: {len(dr):2d} trades | WR={wr}% | exp={exp:+.3f}R')

    # By direction
    print(f'  By direction:')
    for d in ('long','short'):
        dr = [t['pnl_r'] for t in trade_log if t['direction']==d]
        if dr:
            wr = round(sum(1 for x in dr if x>0)/len(dr)*100,1)
            exp= round(np.mean(dr),3)
            print(f'    {d}: {len(dr):2d} trades | WR={wr}% | exp={exp:+.3f}R')

    # Save results
    fname = f'backtest_ORB_{symbol}.json'
    with open(fname, 'w') as f:
        json.dump({'params': params, 'trade_log': trade_log}, f, indent=2)
    print(f'  Saved to {fname}')

    return {
        'trades':     len(trade_log),
        'win_rate':   round(len(wins)/len(r_vals)*100,1),
        'total_r':    round(sum(r_vals),2),
        'expectancy': round(np.mean(r_vals),3),
        'max_dd':     round(max_dd,2),
        'sharpe':     sharpe,
        'trade_log':  trade_log,
    }


if __name__ == '__main__':
    print('APEX Setup D — Opening Range Breakout')
    print('Local backtest only — not in live system')
    print()
    for sym in ('NQ', 'ES'):
        run_orb_backtest(sym, start='2024-09-01')
