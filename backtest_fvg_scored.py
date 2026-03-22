"""
APEX Setup D — FVG with Quality Scoring
========================================
Tests different minimum score thresholds to find optimal filter.
"""

import sqlite3
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone
from collections import defaultdict
from backtest_fvg import load_bars, calc_atr, detect_fvgs, get_htf_bias

DB_PATH = 'apex_market.db'

def score_fvg(fvg, df_15m, atr_15m, current_bar_idx, vol_baseline):
    """
    Score an FVG on quality factors. Max score = 100.
    """
    score = 0
    reasons = []

    # 1. SIZE SCORE (0-30 pts) — larger FVG relative to ATR = better
    size_ratio = fvg['size'] / fvg['atr'] if fvg['atr'] > 0 else 0
    if size_ratio >= 1.0:
        score += 30; reasons.append('large FVG')
    elif size_ratio >= 0.7:
        score += 20; reasons.append('medium FVG')
    elif size_ratio >= 0.4:
        score += 10; reasons.append('small FVG')
    else:
        score += 5

    # 2. FRESHNESS SCORE (0-25 pts) — newer FVGs fill more reliably
    bars_ago = current_bar_idx - df_15m.index.get_loc(fvg['formed_at']) \
               if fvg['formed_at'] in df_15m.index else 99
    if bars_ago <= 4:    # within 1 hour
        score += 25; reasons.append('very fresh')
    elif bars_ago <= 8:  # within 2 hours
        score += 18; reasons.append('fresh')
    elif bars_ago <= 16: # within 4 hours
        score += 10; reasons.append('recent')
    elif bars_ago <= 32: # within 8 hours
        score += 5
    # older than 8 hours = 0

    # 3. CLEAN FILL SCORE (0-25 pts) — FVG not partially filled already
    # Check if price has traded through the FVG zone since formation
    formed_idx = df_15m.index.get_loc(fvg['formed_at']) \
                 if fvg['formed_at'] in df_15m.index else None
    if formed_idx is not None:
        since_formed = df_15m.iloc[formed_idx:current_bar_idx]
        if fvg['type'] == 'bullish':
            min_low = float(since_formed['low'].min()) if len(since_formed) > 0 else fvg['top']
            penetration = (fvg['top'] - min_low) / fvg['size'] if fvg['size'] > 0 else 0
        else:
            max_high = float(since_formed['high'].max()) if len(since_formed) > 0 else fvg['bottom']
            penetration = (max_high - fvg['bottom']) / fvg['size'] if fvg['size'] > 0 else 0

        if penetration <= 0.25:   # barely touched
            score += 25; reasons.append('clean')
        elif penetration <= 0.50: # 50% fill
            score += 15; reasons.append('partial fill')
        elif penetration <= 0.75: # 75% fill
            score += 5
        # >75% filled = 0 points

    # 4. VOLUME SCORE (0-20 pts) — high volume on FVG creation
    if vol_baseline > 0:
        # Get volume of the FVG candle
        if fvg['formed_at'] in df_15m.index:
            fvg_vol = float(df_15m.loc[fvg['formed_at'], 'volume'])
            vol_ratio = fvg_vol / vol_baseline
            if vol_ratio >= 2.0:
                score += 20; reasons.append('high vol')
            elif vol_ratio >= 1.5:
                score += 15; reasons.append('above avg vol')
            elif vol_ratio >= 1.0:
                score += 8
            else:
                score += 3

    return score, reasons


def run_scored_fvg_backtest(symbol, min_score=50, start='2024-09-01',
                             end=None, params=None):
    """Run FVG backtest with quality scoring filter."""

    SESSION_START = 13
    SESSION_END   = 19
    MAX_TRADES    = 3
    TARGET_RR     = 2.0
    MIN_FVG_ATR   = 0.3
    MAX_BARS_HELD = 60

    df_1m  = load_bars(symbol, '1min',  start)
    df_15m = load_bars(symbol, '15min', start)
    df_4h  = load_bars(symbol, '4hour', start)

    if end:
        end_ts = int(pd.Timestamp(end, tz='UTC').timestamp())
        df_1m  = df_1m[df_1m['ts']  <= end_ts]
        df_15m = df_15m[df_15m['ts'] <= end_ts]

    atr_15m = calc_atr(df_15m, 14)
    atr_1m  = calc_atr(df_1m,  14)

    fvgs_15m = detect_fvgs(df_15m, atr_15m, MIN_FVG_ATR)

    df_1m['date'] = df_1m.index.date
    dates = sorted(df_1m['date'].unique())
    trade_log = []

    for date in dates:
        dow = pd.Timestamp(date).dayofweek
        if dow >= 5:
            continue

        day_1m = df_1m[df_1m['date'] == date]
        sess_bars = day_1m[
            (day_1m.index.hour >= SESSION_START) &
            (day_1m.index.hour <  SESSION_END)
        ]
        if len(sess_bars) < 30:
            continue

        date_ts = int(pd.Timestamp(date, tz='UTC').timestamp())
        active_fvgs = [
            f for f in fvgs_15m
            if f['ts'] <= date_ts + 86400
            and f['ts'] >= date_ts - 86400 * 5
        ]
        if not active_fvgs:
            continue

        vol_baseline = float(df_15m['volume'].tail(20).mean())
        trades_today = 0
        traded_fvgs  = set()

        for idx, bar in sess_bars.iterrows():
            if trades_today >= MAX_TRADES:
                break

            bar_close = float(bar['close'])
            bar_open  = float(bar['open'])
            bar_high  = float(bar['high'])
            bar_low   = float(bar['low'])
            bar_vol   = float(bar['volume'])

            # HTF bias
            past_4h = df_4h[df_4h.index <= idx]
            if len(past_4h) < 20:
                continue
            ema = past_4h['close'].ewm(span=20).mean()
            last_close = float(past_4h['close'].iloc[-1])
            last_ema   = float(ema.iloc[-1])
            if last_close > last_ema * 1.001:
                bias = 'bullish'
            elif last_close < last_ema * 0.999:
                bias = 'bearish'
            else:
                continue

            vol_1m_baseline = float(atr_1m.iloc[max(0, df_1m.index.get_loc(idx)-20):df_1m.index.get_loc(idx)].mean()) \
                              if idx in df_1m.index else 0

            for fvg in active_fvgs:
                fvg_id = fvg['ts']
                if fvg_id in traded_fvgs:
                    continue
                if fvg['ts'] >= int(idx.timestamp()):
                    continue

                direction = None
                if fvg['type'] == 'bullish' and bias == 'bullish' \
                        and bar_low <= fvg['top'] and bar_close >= fvg['mid'] \
                        and bar_close > bar_open:
                    direction = 'long'
                elif fvg['type'] == 'bearish' and bias == 'bearish' \
                        and bar_high >= fvg['bottom'] and bar_close <= fvg['mid'] \
                        and bar_close < bar_open:
                    direction = 'short'

                if direction is None:
                    continue

                # Get current 15min bar index
                past_15m = df_15m[df_15m.index <= idx]
                cur_15m_idx = len(past_15m) - 1

                # Score the FVG
                score, reasons = score_fvg(fvg, df_15m, atr_15m,
                                           cur_15m_idx, vol_baseline)

                if score < min_score:
                    continue

                # Calculate levels
                atr_val = float(atr_1m.loc[idx]) if idx in atr_1m.index \
                          and not pd.isna(atr_1m.loc[idx]) else fvg['atr'] / 15

                if direction == 'long':
                    entry  = bar_close
                    stop   = fvg['bottom'] - 0.1 * atr_val
                    target = entry + TARGET_RR * (entry - stop)
                else:
                    entry  = bar_close
                    stop   = fvg['top'] + 0.1 * atr_val
                    target = entry - TARGET_RR * (stop - entry)

                risk = abs(entry - stop)
                if risk == 0:
                    continue

                # Simulate trade
                remaining   = sess_bars[sess_bars.index > idx]
                exit_price  = None
                exit_reason = None
                bars_held   = 0

                for exit_idx, exit_bar in remaining.iterrows():
                    bars_held += 1
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
                    if exit_idx.hour >= SESSION_END:
                        exit_price = float(exit_bar['close']); exit_reason = 'session_end'; break
                    if bars_held >= MAX_BARS_HELD:
                        exit_price = float(exit_bar['close']); exit_reason = 'max_bars'; break

                if exit_price is None:
                    exit_price  = float(sess_bars['close'].iloc[-1])
                    exit_reason = 'session_end'

                pnl_r = round((exit_price - entry) / risk, 3) if direction == 'long' \
                        else round((entry - exit_price) / risk, 3)

                trade_log.append({
                    'date':      str(date),
                    'direction': direction,
                    'entry_time':str(idx),
                    'pnl_r':     pnl_r,
                    'exit_reason':exit_reason,
                    'score':     score,
                    'reasons':   reasons,
                    'dow':       ['Mon','Tue','Wed','Thu','Fri'][dow],
                })

                traded_fvgs.add(fvg_id)
                trades_today += 1
                break

    if not trade_log:
        return {'trades': 0, 'min_score': min_score}

    r      = [t['pnl_r'] for t in trade_log]
    wins   = [x for x in r if x > 0]
    cum_r  = np.cumsum(r)
    max_dd = float(np.min(cum_r - np.maximum.accumulate(cum_r)))
    sharpe = round(np.mean(r)/np.std(r)*np.sqrt(252),2) if np.std(r) > 0 else 0
    weeks  = len(set(pd.Timestamp(t['entry_time']).strftime('%Y-W%U') for t in trade_log))

    return {
        'min_score': min_score,
        'trades':    len(trade_log),
        'tpw':       round(len(trade_log)/max(weeks,1),1),
        'win_rate':  round(len(wins)/len(r)*100,1),
        'total_r':   round(sum(r),2),
        'exp':       round(np.mean(r),3),
        'max_dd':    round(max_dd,2),
        'sharpe':    sharpe,
    }


if __name__ == '__main__':
    print('APEX FVG Quality Scoring — NQ')
    print('Testing minimum score thresholds')
    print()
    print(f'{"Score":>6} {"Trades":>7} {"T/wk":>5} {"WR":>7} {"Exp":>7} {"Total":>8} {"MaxDD":>7} {"Sharpe":>7}')
    print('-' * 65)

    for min_score in (0, 30, 40, 50, 60, 70, 80):
        r = run_scored_fvg_backtest('NQ', min_score=min_score, start='2024-09-01')
        if r['trades'] > 0:
            print(f'{r["min_score"]:>6} {r["trades"]:>7} {r["tpw"]:>5} '
                  f'{r["win_rate"]:>6}% {r["exp"]:>+7.3f}R {r["total_r"]:>+7.2f}R '
                  f'{r["max_dd"]:>7.2f}R {r["sharpe"]:>7.2f}')
        else:
            print(f'{min_score:>6} {"no trades":>30}')
