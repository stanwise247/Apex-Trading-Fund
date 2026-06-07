"""
backtest_setup_d.py  — Task 3 (research only, no code changes)
==============================================================
Setup D backtest using actual fvg_engine.py detect_fvgs() logic on 15min bars.
Three variants: raw / regime-filtered / regime+L3 proxy.
All available data (~21 months).
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import psycopg2
from datetime import datetime, timezone

warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/stanleywise/Desktop/apex')

DATABASE_URL = os.environ['DATABASE_URL']

FVG_MIN_SCORE    = 60
FVG_MIN_ATR_MULT = 0.7      # live setting
TARGET_RR        = 2.0
STOP_ATR_MULT    = 0.5
MAX_BARS_FORWARD = 48       # 12h at 15min bars

SESSION_START_UTC = 14
SESSION_END_UTC   = 20


def get_conn():
    return psycopg2.connect(DATABASE_URL)


def load_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute(
        'SELECT ts, open, high, low, close, volume FROM ohlcv '
        'WHERE symbol=%s AND timeframe=%s AND ts > 1000000 ORDER BY ts ASC',
        (symbol, timeframe)
    )
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    for c in ['ts', 'open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['dt'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    df = df.dropna(subset=['open', 'high', 'low', 'close']).reset_index(drop=True)
    return df


def load_regime(symbol: str) -> pd.DataFrame:
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute(
        'SELECT EXTRACT(EPOCH FROM timestamp)::float AS ts, regime, confidence '
        'FROM regime_log WHERE symbol=%s ORDER BY timestamp ASC',
        (symbol,)
    )
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=['ts', 'regime', 'confidence'])
    df['ts'] = df['ts'].astype(float)
    return df


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, pc = df['high'], df['low'], df['close'].shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def detect_fvgs(df15: pd.DataFrame, atr: pd.Series, min_atr_mult: float = FVG_MIN_ATR_MULT):
    """Mirrors fvg_engine.detect_fvgs() exactly."""
    from fvg_engine import score_fvg as _score_fvg
    highs  = df15['high'].values
    lows   = df15['low'].values
    times  = df15['dt'].values
    n      = len(df15)
    fvgs   = []
    for i in range(1, n - 1):
        atr_val = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else 0
        if atr_val == 0:
            continue
        # Bullish FVG
        if lows[i + 1] > highs[i - 1]:
            size = float(lows[i + 1] - highs[i - 1])
            if size >= min_atr_mult * atr_val:
                fvgs.append({
                    'type':      'bullish',
                    'top':       float(lows[i + 1]),
                    'bottom':    float(highs[i - 1]),
                    'mid':       float((lows[i + 1] + highs[i - 1]) / 2),
                    'size':      size,
                    'atr':       atr_val,
                    'bar_idx':   i + 1,
                    'formed_at': times[i + 1],
                    'formed_ts': float(df15['ts'].iloc[i + 1]),
                })
        # Bearish FVG
        if highs[i + 1] < lows[i - 1]:
            size = float(lows[i - 1] - highs[i + 1])
            if size >= min_atr_mult * atr_val:
                fvgs.append({
                    'type':      'bearish',
                    'top':       float(lows[i - 1]),
                    'bottom':    float(highs[i + 1]),
                    'mid':       float((lows[i - 1] + highs[i + 1]) / 2),
                    'size':      size,
                    'atr':       atr_val,
                    'bar_idx':   i + 1,
                    'formed_at': times[i + 1],
                    'formed_ts': float(df15['ts'].iloc[i + 1]),
                })
    return fvgs


def simulate_trades(df15: pd.DataFrame, df_regime: pd.DataFrame, variant: str) -> pd.DataFrame:
    """
    For each FVG, simulate entry when price retraces into zone.
    variant: 'raw' / 'regime' / 'regime_l3'
    """
    atr = calc_atr(df15, 14)
    fvgs = detect_fvgs(df15, atr)

    closes  = df15['close'].values
    highs   = df15['high'].values
    lows    = df15['low'].values
    ts_vals = df15['ts'].values
    # Convert dt column to UTC-aware pandas Timestamps for consistent tz handling
    dt_series = pd.to_datetime(df15['dt'])
    if dt_series.dt.tz is None:
        dt_series = dt_series.dt.tz_localize('UTC')
    else:
        dt_series = dt_series.dt.tz_convert('UTC')
    dt_vals = dt_series.values  # numpy datetime64[ns, UTC]

    trades = []

    for fvg in fvgs:
        bar_i = fvg['bar_idx']
        if bar_i < 2:
            continue

        # Session filter — only trade during active hours
        dt_formed = pd.Timestamp(fvg['formed_at'])
        if dt_formed.tzinfo is None:
            dt_formed = dt_formed.tz_localize('UTC')
        else:
            dt_formed = dt_formed.tz_convert('UTC')
        if dt_formed.hour < SESSION_START_UTC or dt_formed.hour >= SESSION_END_UTC:
            continue

        # Get regime at time of FVG formation
        regime_label, regime_conf = 'UNKNOWN', 0.5
        if not df_regime.empty:
            idx = np.searchsorted(df_regime['ts'].values, fvg['formed_ts'], side='right') - 1
            if idx >= 0:
                regime_label = df_regime['regime'].iloc[idx]
                regime_conf  = float(df_regime['confidence'].iloc[idx])

        # Regime filter for Setup D (TRENDING + CHOPPY)
        if variant in ('regime', 'regime_l3'):
            if regime_label not in ('TRENDING', 'CHOPPY'):
                continue

        # L3 proxy multiplier
        l3_mult = 1.0
        if variant == 'regime_l3':
            if regime_label == 'TRENDING':
                if regime_conf >= 0.55:
                    l3_mult = 1.5 if regime_conf >= 0.65 else 1.25
                elif regime_conf >= 0.45:
                    l3_mult = 1.0
                else:
                    l3_mult = 0.75
            else:
                l3_mult = 0.75

        # Scan forward for price to enter the FVG zone, then outcome
        atr_at_entry = fvg['atr']
        entered = False
        for j in range(bar_i + 1, min(bar_i + MAX_BARS_FORWARD, len(df15) - 1)):
            bar_hour = pd.Timestamp(dt_vals[j]).hour
            if bar_hour < SESSION_START_UTC or bar_hour >= SESSION_END_UTC:
                continue

            c = closes[j]
            h = highs[j]
            lo = lows[j]

            # Entry: price touches FVG zone
            if fvg['type'] == 'bullish':
                if lo <= fvg['top'] and h >= fvg['bottom']:
                    entry = fvg['mid']
                    direction = 'long'
                    stop   = entry - STOP_ATR_MULT * atr_at_entry
                    target = entry + TARGET_RR * STOP_ATR_MULT * atr_at_entry
                    entered = True
            else:
                if h >= fvg['bottom'] and lo <= fvg['top']:
                    entry = fvg['mid']
                    direction = 'short'
                    stop   = entry + STOP_ATR_MULT * atr_at_entry
                    target = entry - TARGET_RR * STOP_ATR_MULT * atr_at_entry
                    entered = True

            if entered:
                # Simulate outcome on subsequent bars
                outcome = None
                for k in range(j, min(j + MAX_BARS_FORWARD, len(df15))):
                    if direction == 'long':
                        if highs[k] >= target:
                            outcome = 'win'
                            break
                        if lows[k] <= stop:
                            outcome = 'loss'
                            break
                    else:
                        if lows[k] <= target:
                            outcome = 'win'
                            break
                        if highs[k] >= stop:
                            outcome = 'loss'
                            break

                if outcome is None:
                    break

                r_per_contract = TARGET_RR if outcome == 'win' else -1.0
                month_str = pd.Timestamp(dt_vals[j]).strftime('%Y-%m')
                trades.append({
                    'ts':        ts_vals[j],
                    'month':     month_str,
                    'direction': direction,
                    'outcome':   outcome,
                    'r':         r_per_contract * l3_mult,
                    'regime':    regime_label,
                    'l3_mult':   l3_mult,
                })
                break

    return pd.DataFrame(trades)


def calc_metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {'signals': 0, 'wr': 0, 'sharpe': 0, 'maxdd': 0, 'total_r': 0, 'pf': 0}
    n = len(trades)
    wins = (trades['r'] > 0).sum()
    wr = wins / n
    total_r = trades['r'].sum()
    # Sharpe: daily equity curve
    monthly = trades.groupby('month')['r'].sum()
    sharpe = (monthly.mean() / (monthly.std() + 1e-9)) * np.sqrt(12)
    # Max drawdown
    cum = trades['r'].cumsum()
    roll_max = cum.cummax()
    dd = (cum - roll_max)
    maxdd = float(dd.min())
    # Profit factor
    wins_r = trades.loc[trades['r'] > 0, 'r'].sum()
    loss_r = abs(trades.loc[trades['r'] < 0, 'r'].sum())
    pf = wins_r / loss_r if loss_r > 0 else float('inf')
    # Positive months
    pos_months = (monthly > 0).sum()
    total_months = len(monthly)
    return {
        'signals': n,
        'signals_per_month': round(n / max(total_months, 1), 1),
        'wr': round(wr * 100, 1),
        'sharpe': round(float(sharpe), 2),
        'maxdd': round(maxdd, 2),
        'total_r': round(float(total_r), 2),
        'pf': round(float(pf), 2),
        'pos_months': pos_months,
        'total_months': total_months,
    }


def run(symbol: str):
    print(f'\n{"="*60}')
    print(f'SETUP D BACKTEST — {symbol}')
    print(f'{"="*60}')

    df5   = load_ohlcv(symbol, '5min')
    if df5.empty:
        print(f'No 5min data for {symbol}')
        return

    # Resample to 15min
    df5 = df5.set_index('dt')
    df15_raw = df5[['open', 'high', 'low', 'close', 'volume', 'ts']].resample('15min').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
         'volume': 'sum', 'ts': 'first'}
    ).dropna(subset=['open', 'close']).reset_index()
    df15_raw['ts'] = df15_raw['ts'].astype(float)

    print(f'15min bars: {len(df15_raw):,}')
    print(f'Date range: {df15_raw["dt"].iloc[0].date()} → {df15_raw["dt"].iloc[-1].date()}')
    months_total = (df15_raw["dt"].iloc[-1] - df15_raw["dt"].iloc[0]).days / 30.44
    print(f'~{months_total:.1f} months')

    df_regime = load_regime(symbol)
    print(f'Regime rows: {len(df_regime):,}')

    variants = ['raw', 'regime', 'regime_l3']
    print(f'\n{"Variant":<14} {"Signals":<10} {"Sig/Mo":<10} {"WR%":<8} '
          f'{"Sharpe":<10} {"MaxDD":<10} {"TotalR":<10} {"PF":<8} {"+Mo/Tot":<10}')
    print('-' * 90)

    for v in variants:
        trades = simulate_trades(df15_raw.copy(), df_regime, v)
        m = calc_metrics(trades)
        print(
            f'{v:<14} {m["signals"]:<10} {m.get("signals_per_month", 0):<10} '
            f'{m["wr"]:<8} {m["sharpe"]:<10} {m["maxdd"]:<10} '
            f'{m["total_r"]:<10} {m["pf"]:<8} '
            f'{m.get("pos_months", 0)}/{m.get("total_months", 0)}'
        )


if __name__ == '__main__':
    for sym in ('MNQ', 'ES'):
        run(sym)
    print('\nTask 3 complete — research only, no code changes.')
