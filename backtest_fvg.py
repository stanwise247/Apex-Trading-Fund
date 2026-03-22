"""
APEX Setup D — Fair Value Gap (FVG) Backtester
===============================================
Multi-timeframe FVG fill strategy.
- 4hour HTF bias filter
- 15min FVG identification  
- 1min entry trigger
Local only — not in live system yet.
"""

import sqlite3
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone
from collections import defaultdict

DB_PATH = 'apex_market.db'

PARAMS = {
    'min_fvg_atr':       0.3,   # minimum FVG size as fraction of ATR
    'entry_tf':          '1min',
    'fvg_tf':            '15min',
    'htf_tf':            '4hour',
    'target_rr':         2.0,
    'max_trades_session':3,      # max trades per session per instrument
    'vol_filter':        True,
    'vol_multiplier':    1.0,
    'session_start_utc': 13,
    'session_end_utc':   19,
    'max_bars_held':     60,     # max 1min bars held (60 min)
    'fvg_fill_pct':      0.5,    # price must reach 50% of FVG to trigger
}

# ─────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_bars(symbol, timeframe, start=None):
    conn = sqlite3.connect(DB_PATH)
    q = 'SELECT ts,open,high,low,close,volume FROM ohlcv WHERE symbol=? AND timeframe=? ORDER BY ts ASC'
    params = [symbol, timeframe]
    if start:
        start_ts = int(pd.Timestamp(start, tz='UTC').timestamp())
        q = 'SELECT ts,open,high,low,close,volume FROM ohlcv WHERE symbol=? AND timeframe=? AND ts>=? ORDER BY ts ASC'
        params.append(start_ts)
    df = pd.read_sql_query(q, conn, params=params)
    conn.close()
    df['dt'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    df.set_index('dt', inplace=True)
    return df


def calc_atr(df, period=14):
    high  = df['high']
    low   = df['low']
    close = df['close'].shift(1)
    tr    = pd.concat([high-low, (high-close).abs(), (low-close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ─────────────────────────────────────────────────────────────
#  FVG DETECTION
# ─────────────────────────────────────────────────────────────

def detect_fvgs(df, atr, min_atr_mult=0.3):
    """
    Detect Fair Value Gaps on a given timeframe.
    Bullish FVG: candle[i-1].high < candle[i+1].low (gap up)
    Bearish FVG: candle[i-1].low  > candle[i+1].high (gap down)
    Returns list of dicts with fvg details.
    """
    fvgs = []
    highs  = df['high'].values
    lows   = df['low'].values
    closes = df['close'].values
    times  = df.index

    for i in range(1, len(df)-1):
        atr_val = float(atr.iloc[i]) if i < len(atr) and not pd.isna(atr.iloc[i]) else 0
        if atr_val == 0:
            continue

        # Bullish FVG — gap up, price left imbalance below
        if lows[i+1] > highs[i-1]:
            size = lows[i+1] - highs[i-1]
            if size >= min_atr_mult * atr_val:
                fvgs.append({
                    'type':      'bullish',
                    'top':       lows[i+1],
                    'bottom':    highs[i-1],
                    'mid':       (lows[i+1] + highs[i-1]) / 2,
                    'size':      size,
                    'atr':       atr_val,
                    'formed_at': times[i+1],
                    'ts':        int(times[i+1].timestamp()),
                    'filled':    False,
                    'candle_ts': int(times[i].timestamp()),
                })

        # Bearish FVG — gap down, price left imbalance above
        if highs[i+1] < lows[i-1]:
            size = lows[i-1] - highs[i+1]
            if size >= min_atr_mult * atr_val:
                fvgs.append({
                    'type':      'bearish',
                    'top':       lows[i-1],
                    'bottom':    highs[i+1],
                    'mid':       (lows[i-1] + highs[i+1]) / 2,
                    'size':      size,
                    'atr':       atr_val,
                    'formed_at': times[i+1],
                    'ts':        int(times[i+1].timestamp()),
                    'filled':    False,
                    'candle_ts': int(times[i].timestamp()),
                })

    return fvgs


# ─────────────────────────────────────────────────────────────
#  HTF BIAS
# ─────────────────────────────────────────────────────────────

def get_htf_bias(df_4h, at_time):
    """Get 4hour bias at a given time."""
    past = df_4h[df_4h.index <= at_time]
    if len(past) < 20:
        return 'neutral'
    
    # Simple bias: compare last close to 20-bar EMA
    ema = past['close'].ewm(span=20).mean()
    last_close = float(past['close'].iloc[-1])
    last_ema   = float(ema.iloc[-1])
    
    if last_close > last_ema * 1.001:
        return 'bullish'
    elif last_close < last_ema * 0.999:
        return 'bearish'
    return 'neutral'


# ─────────────────────────────────────────────────────────────
#  MAIN BACKTEST
# ─────────────────────────────────────────────────────────────

def run_fvg_backtest(symbol, start='2024-09-01', end=None, params=None):
    if params is None:
        params = PARAMS

    print(f'\n{"="*60}')
    print(f'  FVG Backtest — {symbol}')
    print(f'  {start} to {end or "latest"}')
    print(f'{"="*60}')

    # Load all timeframes
    df_1m  = load_bars(symbol, '1min',  start)
    df_15m = load_bars(symbol, '15min', start)
    df_4h  = load_bars(symbol, '4hour', start)

    if end:
        end_ts = int(pd.Timestamp(end, tz='UTC').timestamp())
        df_1m  = df_1m[df_1m['ts']  <= end_ts]
        df_15m = df_15m[df_15m['ts'] <= end_ts]

    print(f'  1min bars:  {len(df_1m):,}')
    print(f'  15min bars: {len(df_15m):,}')
    print(f'  4hour bars: {len(df_4h):,}')

    # Calculate ATR
    atr_15m = calc_atr(df_15m, 14)
    atr_1m  = calc_atr(df_1m,  14)

    # Detect 15min FVGs
    fvgs_15m = detect_fvgs(df_15m, atr_15m, params['min_fvg_atr'])
    print(f'  15min FVGs detected: {len(fvgs_15m)}')
    bull_fvgs = [f for f in fvgs_15m if f['type'] == 'bullish']
    bear_fvgs = [f for f in fvgs_15m if f['type'] == 'bearish']
    print(f'    Bullish: {len(bull_fvgs)} | Bearish: {len(bear_fvgs)}')

    # Group 1min bars by date
    df_1m['date'] = df_1m.index.date
    dates = sorted(df_1m['date'].unique())

    trade_log   = []
    traded_fvgs = set()

    for date in dates:
        dow = pd.Timestamp(date).dayofweek
        if dow >= 5:
            continue

        day_1m = df_1m[df_1m['date'] == date]

        # Get session bars
        sess_start = params['session_start_utc']
        sess_end   = params['session_end_utc']

        sess_bars = day_1m[
            (day_1m.index.hour >= sess_start) &
            (day_1m.index.hour <  sess_end)
        ]

        if len(sess_bars) < 30:
            continue

        trades_today = 0

        # Get active FVGs for this date
        date_ts = int(pd.Timestamp(date, tz='UTC').timestamp())
        active_fvgs = [
            f for f in fvgs_15m
            if f['ts'] <= date_ts + 86400
            and f['ts'] >= date_ts - 86400 * 5  # FVGs up to 5 days old
        ]

        if not active_fvgs:
            continue

        # Get vol baseline
        vol_baseline = float(sess_bars['volume'].mean()) if len(sess_bars) > 0 else 0

        for idx, bar in sess_bars.iterrows():
            if trades_today >= params['max_trades_session']:
                break

            bar_ts    = int(idx.timestamp())
            bar_close = float(bar['close'])
            bar_open  = float(bar['open'])
            bar_high  = float(bar['high'])
            bar_low   = float(bar['low'])
            bar_vol   = float(bar['volume'])

            # Get HTF bias
            bias = get_htf_bias(df_4h, idx)
            if bias == 'neutral':
                continue

            # Check each active FVG
            for fvg in active_fvgs:
                fvg_id = fvg['ts']
                if fvg_id in traded_fvgs:
                    continue
                if fvg['ts'] >= bar_ts:
                    continue

                direction = None

                # Bullish FVG — price returns to fill, we go long
                if (fvg['type'] == 'bullish'
                        and bias == 'bullish'
                        and bar_low  <= fvg['top']
                        and bar_close >= fvg['mid']
                        and bar_close > bar_open):  # bullish 1min candle
                    direction = 'long'

                # Bearish FVG — price returns to fill, we go short
                elif (fvg['type'] == 'bearish'
                        and bias == 'bearish'
                        and bar_high >= fvg['bottom']
                        and bar_close <= fvg['mid']
                        and bar_close < bar_open):  # bearish 1min candle
                    direction = 'short'

                if direction is None:
                    continue

                # Volume filter
                if params['vol_filter'] and vol_baseline > 0:
                    if bar_vol < params['vol_multiplier'] * vol_baseline:
                        continue

                # Calculate trade levels
                entry = bar_close
                atr_val = float(atr_1m.loc[idx]) if idx in atr_1m.index and not pd.isna(atr_1m.loc[idx]) else fvg['atr'] / 15

                if direction == 'long':
                    stop   = fvg['bottom'] - 0.1 * atr_val
                    target = entry + params['target_rr'] * (entry - stop)
                else:
                    stop   = fvg['top'] + 0.1 * atr_val
                    target = entry - params['target_rr'] * (stop - entry)

                risk = abs(entry - stop)
                if risk == 0:
                    continue

                rr_planned = abs(target - entry) / risk

                # Simulate trade on remaining 1min bars
                remaining = sess_bars[sess_bars.index > idx]
                exit_price  = None
                exit_reason = None
                bars_held   = 0

                for exit_idx, exit_bar in remaining.iterrows():
                    bars_held += 1
                    exit_hour = exit_idx.hour
                    hi = float(exit_bar['high'])
                    lo = float(exit_bar['low'])

                    if direction == 'long' and lo <= stop:
                        exit_price = stop; exit_reason = 'stop'; break
                    if direction == 'short' and hi >= stop:
                        exit_price = stop; exit_reason = 'stop'; break
                    if direction == 'long' and hi >= target:
                        exit_price = target; exit_reason = 'target'; break
                    if direction == 'short' and lo <= target:
                        exit_price = target; exit_reason = 'target'; break
                    if exit_hour >= sess_end:
                        exit_price = float(exit_bar['close']); exit_reason = 'session_end'; break
                    if bars_held >= params['max_bars_held']:
                        exit_price = float(exit_bar['close']); exit_reason = 'max_bars'; break

                if exit_price is None:
                    exit_price  = float(sess_bars['close'].iloc[-1])
                    exit_reason = 'session_end'

                pnl_r = round((exit_price - entry) / risk, 3) if direction == 'long' \
                        else round((entry - exit_price) / risk, 3)

                trade_log.append({
                    'date':        str(date),
                    'symbol':      symbol,
                    'direction':   direction,
                    'bias':        bias,
                    'fvg_type':    fvg['type'],
                    'fvg_size_atr':round(fvg['size'] / fvg['atr'], 2),
                    'entry_time':  str(idx),
                    'entry_price': round(entry, 2),
                    'stop':        round(stop, 2),
                    'target':      round(target, 2),
                    'rr_planned':  round(rr_planned, 2),
                    'exit_price':  round(exit_price, 2),
                    'exit_reason': exit_reason,
                    'bars_held':   bars_held,
                    'pnl_r':       pnl_r,
                    'dow':         ['Mon','Tue','Wed','Thu','Fri'][dow],
                })

                traded_fvgs.add(fvg_id)
                trades_today += 1
                break  # one trade per bar check

    # ── Results ──────────────────────────────────────────────
    if not trade_log:
        print('  No trades found')
        return {}

    r_vals  = [t['pnl_r'] for t in trade_log]
    wins    = [r for r in r_vals if r > 0]
    losses  = [r for r in r_vals if r <= 0]
    cum_r   = np.cumsum(r_vals)
    max_dd  = float(np.min(cum_r - np.maximum.accumulate(cum_r)))
    sharpe  = round(np.mean(r_vals)/np.std(r_vals)*np.sqrt(252), 2) if np.std(r_vals) > 0 else 0

    print(f'\n  {"─"*50}')
    print(f'  {symbol} — FVG — Results')
    print(f'  {"─"*50}')
    print(f'  Trades:      {len(trade_log)}')
    print(f'  Win Rate:    {round(len(wins)/len(r_vals)*100,1)}%')
    print(f'  Total R:     {round(sum(r_vals),2)}R')
    print(f'  Expectancy:  {round(np.mean(r_vals),3)}R')
    print(f'  Avg Win:     +{round(np.mean(wins),2)}R') if wins else None
    print(f'  Avg Loss:    {round(np.mean(losses),2)}R') if losses else None
    print(f'  Max DD:      {round(max_dd,2)}R')
    print(f'  Sharpe:      {sharpe}')

    weeks = len(set(pd.Timestamp(t['entry_time']).strftime('%Y-W%U') for t in trade_log))
    print(f'  Avg/week:    {round(len(trade_log)/max(weeks,1),1)} trades')

    print(f'  Exit reasons:')
    for reason in ('stop','target','session_end','max_bars'):
        n = sum(1 for t in trade_log if t['exit_reason']==reason)
        if n: print(f'    {reason}: {n}')

    print(f'  Day of week:')
    by_dow = defaultdict(list)
    for t in trade_log:
        by_dow[t['dow']].append(t['pnl_r'])
    for dow in ['Mon','Tue','Wed','Thu','Fri']:
        if dow in by_dow:
            dr  = by_dow[dow]
            wr  = round(sum(1 for x in dr if x>0)/len(dr)*100,1)
            exp = round(np.mean(dr),3)
            print(f'    {dow}: {len(dr):3d} trades | WR={wr}% | exp={exp:+.3f}R')

    print(f'  By direction:')
    for d in ('long','short'):
        dr = [t['pnl_r'] for t in trade_log if t['direction']==d]
        if dr:
            wr  = round(sum(1 for x in dr if x>0)/len(dr)*100,1)
            exp = round(np.mean(dr),3)
            print(f'    {d}: {len(dr):3d} trades | WR={wr}% | exp={exp:+.3f}R')

    fname = f'backtest_FVG_{symbol}.json'
    with open(fname, 'w') as f:
        json.dump({'params': params, 'trade_log': trade_log}, f, indent=2)
    print(f'\n  Saved to {fname}')

    return {
        'trades':     len(trade_log),
        'win_rate':   round(len(wins)/len(r_vals)*100,1),
        'total_r':    round(sum(r_vals),2),
        'expectancy': round(np.mean(r_vals),3),
        'max_dd':     round(max_dd,2),
        'sharpe':     sharpe,
    }


if __name__ == '__main__':
    print('APEX Setup D — Fair Value Gap')
    print('Local backtest only')
    print()
    for sym in ('NQ', 'ES'):
        run_fvg_backtest(sym, start='2024-09-01')
