"""
APEX v2 Optimisation Engine — optimiser_v2.py
===============================================
Tests all three strategy modes with their specific parameters.

Mode 1 — Scalp:     Fast entries, tight stops, 5-30min holds
Mode 2 — Swing:     Structure trades, 1-4hr holds
Mode 3 — MeanRev:   Fade extremes, 15-90min holds

For each mode, tests:
  - Score thresholds
  - R:R ratios
  - Session windows
  - VIX filters
  - Day of week filters
  - Mode-specific parameters

Output:
  - Best params per mode
  - Cross-mode comparison
  - Overall system settings
  - Saved to optimiser_v2_results.json

Run: python3 optimiser_v2.py
"""

import sqlite3
import json
import logging
import itertools
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('APEX.OptV2')

DB_PATH  = 'apex_market.db'
NY_TZ    = ZoneInfo('America/New_York')
SYMBOL   = 'NQ'
BALANCE  = 10000
POINT_VAL= 20.0   # $ per NQ point
COMM     = 5.0    # $ commission per round trip


# =============================================================
#  PARAMETER GRIDS PER MODE
# =============================================================

SCALP_GRID = {
    'min_score':  [55, 60, 65, 70],
    'rr_ratio':   [2.0, 2.5, 3.0],
    'risk_pct':   [0.5, 0.75, 1.0],
    'session':    ['london_only', 'ny_only', 'both'],
    'vix_max':    [25, 30, 35],
    'dow':        ['tue_thu', 'mon_fri_out', 'all'],
    'entry_tf':   ['5min', '1min'],
}

SWING_GRID = {
    'min_score':  [55, 60, 65, 70],
    'rr_ratio':   [2.5, 3.0, 4.0],
    'risk_pct':   [1.0, 1.5, 2.0],
    'session':    ['london_only', 'ny_only', 'both'],
    'vix_max':    [20, 25, 30],
    'dow':        ['tue_thu', 'mon_fri_out', 'all'],
    'htf_strict': [True, False],
}

MEANREV_GRID = {
    'min_score':   [60, 65, 70, 75],
    'rr_ratio':    [1.5, 2.0, 2.5],
    'risk_pct':    [0.5, 0.75, 1.0],
    'vwap_dev':    [2.0, 2.5, 3.0],
    'vix_max':     [18, 20, 25],
    'dow':         ['tue_thu', 'mon_fri_out', 'all'],
    'market_cond': ['ranging_only', 'any'],
}

# Session windows (ET hours)
SESSION_RULES = {
    'london_only': lambda h, m: 5 <= h < 8,
    'ny_only':     lambda h, m: (h == 9 and m >= 30) or (h == 10 and m < 30),
    'both':        lambda h, m: (5 <= h < 8) or (h == 9 and m >= 30) or (h == 10 and m < 30),
}

DOW_RULES = {
    'tue_thu':      lambda d: d in (1, 2, 3),   # Tue=1, Wed=2, Thu=3
    'mon_fri_out':  lambda d: d not in (0, 4),
    'all':          lambda d: True,
}

VIX_RULES = {
    18: lambda v: v is None or v < 18,
    20: lambda v: v is None or v < 20,
    25: lambda v: v is None or v < 25,
    30: lambda v: v is None or v < 30,
    35: lambda v: v is None or v < 35,
}


# =============================================================
#  DATA LOADING
# =============================================================

def load_tf(symbol, timeframe, limit=5000):
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            'SELECT ts,open,high,low,close,volume FROM ohlcv '
            'WHERE symbol=? AND timeframe=? ORDER BY ts ASC LIMIT ?',
            conn, params=(symbol, timeframe, limit)
        )
        conn.close()
        if len(df) < 100:
            return None
        df['dt'] = pd.to_datetime(df['ts'], unit='s', utc=True).dt.tz_convert(NY_TZ)
        for c in ['open','high','low','close','volume']:
            df[c] = df[c].astype(float)
        return df
    except Exception as e:
        logger.error(f'Load {symbol} {timeframe}: {e}')
        return None


def load_vix():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            "SELECT ts, close as vix FROM ohlcv WHERE symbol='VIX' AND timeframe='1day' ORDER BY ts ASC",
            conn
        )
        conn.close()
        df['date'] = pd.to_datetime(df['ts'], unit='s', utc=True).dt.date
        return dict(zip(df['date'], df['vix']))
    except Exception:
        return {}


def add_indicators(df):
    c = df['close']
    h, l = df['high'], df['low']

    # EMAs
    df['ema20']  = c.ewm(span=20,  adjust=False).mean()
    df['ema50']  = c.ewm(span=50,  adjust=False).mean()
    df['ema200'] = c.ewm(span=200, adjust=False).mean()

    # ATR
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.ewm(span=14, adjust=False).mean()

    # RSI
    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(com=13, adjust=False).mean()
    df['rsi'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # VWAP approximation (daily reset)
    df['date']  = df['dt'].dt.date
    df['vwap']  = (df['close'] * df['volume']).groupby(df['date']).cumsum() / \
                   df['volume'].groupby(df['date']).cumsum()
    df['vwap_dev'] = (c - df['vwap']) / df['vwap'].replace(0, np.nan) * 100

    # Volume ratio
    df['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

    # Bollinger bands
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df['bb_upper'] = bb_mid + 2 * bb_std
    df['bb_lower'] = bb_mid - 2 * bb_std
    df['bb_pct']   = (c - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)

    # Trend score
    df['trend_bull'] = ((c > df['ema20']) & (df['ema20'] > df['ema50'])).astype(int)
    df['trend_bear'] = ((c < df['ema20']) & (df['ema20'] < df['ema50'])).astype(int)

    return df.dropna(subset=['ema20','atr','rsi'])


# =============================================================
#  SIGNAL GENERATORS PER MODE
# =============================================================

def generate_scalp_signals(df_5min, df_15min, df_1h, params):
    """Generate scalp trade signals based on params"""
    signals = []
    if df_5min is None or len(df_5min) < 100:
        return signals

    df = add_indicators(df_5min.copy())
    vix_data = load_vix()
    sess_fn  = SESSION_RULES[params['session']]
    dow_fn   = DOW_RULES[params['dow']]
    vix_fn   = VIX_RULES[params['vix_max']]

    for i in range(50, len(df)-1):
        row = df.iloc[i]
        dt  = row['dt']
        h, m, d = dt.hour, dt.minute, dt.weekday()

        if not sess_fn(h, m): continue
        if not dow_fn(d): continue

        vix = vix_data.get(dt.date())
        if not vix_fn(vix): continue

        score = 0
        direction = None

        # HTF bias
        if df_15min is not None and len(df_15min) > 50:
            htf_close = df_15min['close'].iloc[-1] if len(df_15min) > 0 else row['close']
            htf_ema20 = df_15min['close'].ewm(span=20, adjust=False).mean().iloc[-1]
            if htf_close > htf_ema20:
                score += 15
                direction = 'long'
            elif htf_close < htf_ema20:
                score += 15
                direction = 'short'

        if direction is None: continue

        # VWAP position
        if direction == 'long' and row['close'] > row.get('vwap', row['close']):
            score += 15
        elif direction == 'short' and row['close'] < row.get('vwap', row['close']):
            score += 15

        # Volume spike
        if row.get('vol_ratio', 1) >= 1.5:
            score += 10

        # RSI alignment
        rsi = row.get('rsi', 50)
        if direction == 'long' and 40 < rsi < 65:
            score += 10
        elif direction == 'short' and 35 < rsi < 60:
            score += 10

        # Trend alignment
        if direction == 'long' and row.get('trend_bull', 0):
            score += 10
        elif direction == 'short' and row.get('trend_bear', 0):
            score += 10

        # EMA bounce
        atr = row['atr']
        close = row['close']
        ema20 = row['ema20']
        if direction == 'long' and abs(close - ema20) < atr * 0.3:
            score += 10
        elif direction == 'short' and abs(close - ema20) < atr * 0.3:
            score += 10

        if score >= params['min_score']:
            stop = close - atr * 1.0 if direction == 'long' else close + atr * 1.0
            risk = abs(close - stop)
            if risk <= 0: continue
            signals.append({
                'ts':        int(dt.timestamp()),
                'direction': direction,
                'entry':     close,
                'stop':      stop,
                'target1':   close + risk * params['rr_ratio'] if direction == 'long' else close - risk * params['rr_ratio'],
                'target2':   close + risk * params['rr_ratio'] * 1.5 if direction == 'long' else close - risk * params['rr_ratio'] * 1.5,
                'score':     score,
                'atr':       atr,
                'vix':       vix,
                'mode':      'scalp',
            })

    return signals


def generate_swing_signals(df_15min, df_1h, df_4h, df_1d, params):
    """Generate swing trade signals"""
    signals = []
    if df_15min is None or len(df_15min) < 100:
        return signals

    df = add_indicators(df_15min.copy())
    vix_data = load_vix()
    sess_fn  = SESSION_RULES[params['session']]
    dow_fn   = DOW_RULES[params['dow']]
    vix_fn   = VIX_RULES[params['vix_max']]

    for i in range(50, len(df)-1):
        row = df.iloc[i]
        dt  = row['dt']
        h, m, d = dt.hour, dt.minute, dt.weekday()

        if not sess_fn(h, m): continue
        if not dow_fn(d): continue

        vix = vix_data.get(dt.date())
        if not vix_fn(vix): continue

        score = 0
        direction = None
        close = row['close']
        atr   = row['atr']

        # Multi-TF bias (core of swing)
        htf_votes_bull = 0
        htf_votes_bear = 0

        for htf_df in [df_1h, df_4h, df_1d]:
            if htf_df is None or len(htf_df) < 20:
                continue
            htf_c   = htf_df['close']
            htf_e20 = float(htf_c.ewm(span=20, adjust=False).mean().iloc[-1])
            htf_e50 = float(htf_c.ewm(span=50, adjust=False).mean().iloc[-1])
            cur     = float(htf_c.iloc[-1])
            if cur > htf_e20 > htf_e50:
                htf_votes_bull += 1
            elif cur < htf_e20 < htf_e50:
                htf_votes_bear += 1

        strict = params.get('htf_strict', True)
        min_votes = 2 if strict else 1

        if htf_votes_bull >= min_votes:
            direction = 'long'
            score += 20 + htf_votes_bull * 5
        elif htf_votes_bear >= min_votes:
            direction = 'short'
            score += 20 + htf_votes_bear * 5

        if direction is None: continue

        # VWAP alignment
        if direction == 'long' and close > row.get('vwap', close):
            score += 15
        elif direction == 'short' and close < row.get('vwap', close):
            score += 15

        # RSI zone
        rsi = row.get('rsi', 50)
        if direction == 'long' and 35 < rsi < 65:
            score += 10
        elif direction == 'short' and 35 < rsi < 65:
            score += 10

        # At EMA (mean reversion to trend)
        if direction == 'long' and abs(close - row['ema20']) < atr * 0.5:
            score += 10
        elif direction == 'short' and abs(close - row['ema20']) < atr * 0.5:
            score += 10

        # Volume confirmation
        if row.get('vol_ratio', 1) >= 1.2:
            score += 10

        if score >= params['min_score']:
            stop = close - atr * 1.5 if direction == 'long' else close + atr * 1.5
            risk = abs(close - stop)
            if risk <= 0: continue
            signals.append({
                'ts':        int(dt.timestamp()),
                'direction': direction,
                'entry':     close,
                'stop':      stop,
                'target1':   close + risk * params['rr_ratio'] if direction == 'long' else close - risk * params['rr_ratio'],
                'target2':   close + risk * params['rr_ratio'] * 1.5 if direction == 'long' else close - risk * params['rr_ratio'] * 1.5,
                'score':     score,
                'atr':       atr,
                'vix':       vix,
                'mode':      'swing',
            })

    return signals


def generate_meanrev_signals(df_5min, df_15min, params):
    """Generate mean reversion signals"""
    signals = []
    if df_5min is None or len(df_5min) < 100:
        return signals

    df = add_indicators(df_5min.copy())
    vix_data = load_vix()
    dow_fn   = DOW_RULES[params['dow']]
    vix_fn   = VIX_RULES[params['vix_max']]
    dev_thresh = params['vwap_dev']

    for i in range(50, len(df)-1):
        row = df.iloc[i]
        dt  = row['dt']
        h, m, d = dt.hour, dt.minute, dt.weekday()

        # Mean rev works any session but skip overnight
        if not (8 <= h < 16): continue
        if not dow_fn(d): continue

        vix = vix_data.get(dt.date())
        if not vix_fn(vix): continue

        close    = row['close']
        atr      = row['atr']
        vwap_dev = row.get('vwap_dev', 0)
        vwap     = row.get('vwap', close)
        rsi      = row.get('rsi', 50)
        bb_pct   = row.get('bb_pct', 0.5)

        score = 0
        direction = None

        # Core: VWAP deviation
        if vwap_dev <= -dev_thresh:
            direction = 'long'
            score += 30
        elif vwap_dev >= dev_thresh:
            direction = 'short'
            score += 30
        else:
            continue

        # RSI extreme
        if direction == 'long' and rsi < 30:
            score += 20
        elif direction == 'long' and rsi < 40:
            score += 10
        elif direction == 'short' and rsi > 70:
            score += 20
        elif direction == 'short' and rsi > 60:
            score += 10

        # Bollinger band extreme
        if direction == 'long' and bb_pct < 0.05:
            score += 15
        elif direction == 'short' and bb_pct > 0.95:
            score += 15

        # Low volume at extreme (exhaustion)
        if row.get('vol_ratio', 1) < 0.8:
            score += 10

        # Market condition check
        if params['market_cond'] == 'ranging_only':
            # Simple ADX proxy: low trend = ranging
            ema_spread = abs(row['ema20'] - row['ema50']) / atr if atr > 0 else 0
            if ema_spread > 2.0:
                continue  # Trending — skip mean rev

        if score >= params['min_score']:
            stop = close - atr * 0.8 if direction == 'long' else close + atr * 0.8
            risk = abs(close - stop)
            if risk <= 0: continue
            # Target = VWAP (mean)
            target1 = vwap
            target2 = vwap + (vwap - close) * 0.5 if direction == 'long' else vwap - (close - vwap) * 0.5
            signals.append({
                'ts':        int(dt.timestamp()),
                'direction': direction,
                'entry':     close,
                'stop':      stop,
                'target1':   target1,
                'target2':   target2,
                'score':     score,
                'atr':       atr,
                'vix':       vix,
                'mode':      'meanrev',
            })

    return signals


# =============================================================
#  FORWARD TEST — simulate trade outcomes
# =============================================================

def forward_test(signals, df_future, rr_ratio):
    """Test each signal against future bars"""
    results = []
    if not signals or df_future is None:
        return results

    future_arr = df_future[['ts','high','low','close']].values

    for sig in signals:
        entry   = sig['entry']
        stop    = sig['stop']
        target1 = sig['target1']
        is_long = sig['direction'] == 'long'
        risk    = abs(entry - stop)
        if risk <= 0: continue

        outcome  = 'timeout'
        exit_px  = entry
        max_bars = 50  # max hold

        # Find bars after signal
        sig_ts   = sig['ts']
        start_idx= np.searchsorted(future_arr[:, 0], sig_ts)

        for j in range(start_idx, min(start_idx + max_bars, len(future_arr))):
            bar_high = future_arr[j, 1]
            bar_low  = future_arr[j, 2]
            bar_close= future_arr[j, 3]

            if is_long:
                if bar_low <= stop:
                    outcome = 'loss'
                    exit_px = stop
                    break
                if bar_high >= target1:
                    outcome = 'win'
                    exit_px = target1
                    break
            else:
                if bar_high >= stop:
                    outcome = 'loss'
                    exit_px = stop
                    break
                if bar_low <= target1:
                    outcome = 'win'
                    exit_px = target1
                    break

        if outcome == 'timeout':
            exit_px = future_arr[min(start_idx + max_bars - 1, len(future_arr)-1), 3]
            outcome = 'win' if (is_long and exit_px > entry) or (not is_long and exit_px < entry) else 'loss'

        pnl_pts = (exit_px - entry) if is_long else (entry - exit_px)
        results.append({**sig, 'outcome': outcome, 'exit_px': exit_px, 'pnl_pts': pnl_pts})

    return results


def simulate_account(results, balance=10000, risk_pct=2.0):
    """Simulate account equity curve"""
    if not results:
        return {'sharpe': 0, 'total_return': 0, 'win_rate': 0,
                'max_dd': 0, 'expectancy': 0, 'trades': 0,
                'profit_factor': 0, 'trades_per_year': 0}

    equity   = [balance]
    returns  = []
    wins     = 0
    gross_w  = 0
    gross_l  = 0
    peak     = balance
    max_dd   = 0

    for r in results:
        bal   = equity[-1]
        risk_usd = bal * risk_pct / 100
        entry = r['entry']
        stop  = r['stop']
        risk_pts = abs(entry - stop)
        if risk_pts <= 0: continue

        pts_per_dollar = risk_usd / (risk_pts * POINT_VAL)
        pnl = r['pnl_pts'] * POINT_VAL * pts_per_dollar - COMM

        bal += pnl
        equity.append(bal)

        ret = pnl / equity[-2] if equity[-2] > 0 else 0
        returns.append(ret)

        if pnl > 0:
            wins   += 1
            gross_w += pnl
        else:
            gross_l += abs(pnl)

        peak   = max(peak, bal)
        dd     = (peak - bal) / peak * 100
        max_dd = max(max_dd, dd)

    n = len(results)
    if n == 0:
        return {'sharpe': 0, 'total_return': 0, 'win_rate': 0,
                'max_dd': 0, 'expectancy': 0, 'trades': 0,
                'profit_factor': 0, 'trades_per_year': 0}

    total_return = (equity[-1] - balance) / balance * 100
    win_rate     = wins / n * 100
    pf           = gross_w / gross_l if gross_l > 0 else 999

    # Expectancy in R
    avg_win_r  = (gross_w / wins / (balance * risk_pct / 100)) if wins > 0 else 0
    avg_loss_r = (gross_l / (n - wins) / (balance * risk_pct / 100)) if n - wins > 0 else 0
    expectancy = (win_rate/100 * avg_win_r) - ((1 - win_rate/100) * avg_loss_r)

    # Sharpe
    if len(returns) > 1:
        r_arr  = np.array(returns)
        sharpe = (np.mean(r_arr) / np.std(r_arr)) * np.sqrt(252) if np.std(r_arr) > 0 else 0
    else:
        sharpe = 0

    # Annualise trade count
    if results:
        span_days = (results[-1]['ts'] - results[0]['ts']) / 86400
        tpy = n / span_days * 252 if span_days > 0 else n
    else:
        tpy = 0

    return {
        'sharpe':          round(sharpe, 3),
        'total_return':    round(total_return, 2),
        'win_rate':        round(win_rate, 1),
        'max_dd':          round(max_dd, 2),
        'expectancy':      round(expectancy, 4),
        'profit_factor':   round(pf, 2),
        'trades':          n,
        'trades_per_year': round(tpy, 0),
        'final_balance':   round(equity[-1], 2),
    }


# =============================================================
#  MODE OPTIMISERS
# =============================================================

def optimise_mode(mode, param_grid, signal_fn, signal_args, df_entry, symbol='NQ'):
    """Run grid search for one strategy mode"""
    logger.info(f'\n{"="*60}')
    logger.info(f'Optimising {mode.upper()} mode')
    logger.info(f'{"="*60}')

    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))
    total  = len(combos)
    logger.info(f'Testing {total} combinations...')

    results_all = []

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))

        try:
            signals = signal_fn(*signal_args, params)
        except Exception as e:
            continue

        if len(signals) < 5:
            continue

        tested  = forward_test(signals, df_entry, params['rr_ratio'])
        metrics = simulate_account(tested, BALANCE, params['risk_pct'])

        if metrics['trades'] < 5:
            continue

        results_all.append({
            'params':  params,
            'metrics': metrics,
        })

        if (i + 1) % 50 == 0:
            logger.info(f'  Progress: {i+1}/{total}')

    if not results_all:
        logger.warning(f'No results for {mode}')
        return None

    # Sort by Sharpe
    results_all.sort(key=lambda x: x['metrics']['sharpe'], reverse=True)
    best = results_all[0]

    logger.info(f'\n✅ BEST {mode.upper()} PARAMS:')
    logger.info(f"  Sharpe:      {best['metrics']['sharpe']}")
    logger.info(f"  Return:      {best['metrics']['total_return']}%")
    logger.info(f"  Win Rate:    {best['metrics']['win_rate']}%")
    logger.info(f"  Max DD:      {best['metrics']['max_dd']}%")
    logger.info(f"  Expectancy:  {best['metrics']['expectancy']}R")
    logger.info(f"  Trades/yr:   {best['metrics']['trades_per_year']}")
    logger.info(f"  Params:      {best['params']}")

    return {
        'mode':    mode,
        'best':    best,
        'top10':   results_all[:10],
        'total_tested': len(results_all),
    }


# =============================================================
#  MAIN
# =============================================================

def run_v2_optimisation(symbol='NQ', quick=False):
    logger.info('Loading market data...')

    df_1min  = load_tf(symbol, '1min',  limit=2000 if quick else 5000)
    df_5min  = load_tf(symbol, '5min',  limit=3000 if quick else 8000)
    df_15min = load_tf(symbol, '15min', limit=2000 if quick else 5000)
    df_1h    = load_tf(symbol, '1hour', limit=1000 if quick else 3000)
    df_4h    = load_tf(symbol, '4hour', limit=500  if quick else 1500)
    df_1d    = load_tf(symbol, '1day',  limit=200  if quick else 800)

    data_summary = {tf: (df is not None and len(df)) for tf, df in
                    [('1min',df_1min),('5min',df_5min),('15min',df_15min),
                     ('1hour',df_1h),('4hour',df_4h),('1day',df_1d)]}
    logger.info(f'Data loaded: {data_summary}')

    if df_5min is None or df_15min is None:
        logger.error('Insufficient data. Run backfill first.')
        return

    results = {}

    # Reduce grid for quick mode
    scalp_grid = SCALP_GRID.copy()
    swing_grid = SWING_GRID.copy()
    mr_grid    = MEANREV_GRID.copy()

    if quick:
        scalp_grid = {k: v[:2] for k, v in scalp_grid.items()}
        swing_grid = {k: v[:2] for k, v in swing_grid.items()}
        mr_grid    = {k: v[:2] for k, v in mr_grid.items()}

    # ── SCALP ──────────────────────────────────────────────
    scalp_result = optimise_mode(
        mode       = 'scalp',
        param_grid = scalp_grid,
        signal_fn  = generate_scalp_signals,
        signal_args= (df_5min, df_15min, df_1h),
        df_entry   = df_5min,
        symbol     = symbol,
    )
    if scalp_result:
        results['scalp'] = scalp_result

    # ── SWING ──────────────────────────────────────────────
    swing_result = optimise_mode(
        mode       = 'swing',
        param_grid = swing_grid,
        signal_fn  = generate_swing_signals,
        signal_args= (df_15min, df_1h, df_4h, df_1d),
        df_entry   = df_15min,
        symbol     = symbol,
    )
    if swing_result:
        results['swing'] = swing_result

    # ── MEAN REVERSION ─────────────────────────────────────
    mr_result = optimise_mode(
        mode       = 'meanrev',
        param_grid = mr_grid,
        signal_fn  = generate_meanrev_signals,
        signal_args= (df_5min, df_15min),
        df_entry   = df_5min,
        symbol     = symbol,
    )
    if mr_result:
        results['meanrev'] = mr_result

    # ── SUMMARY ────────────────────────────────────────────
    logger.info('\n' + '='*60)
    logger.info('V2 OPTIMISATION COMPLETE')
    logger.info('='*60)

    summary = {}
    for mode, r in results.items():
        if r:
            m = r['best']['metrics']
            p = r['best']['params']
            logger.info(f'\n{mode.upper()}:')
            logger.info(f"  Sharpe {m['sharpe']} | Return {m['total_return']}% | WR {m['win_rate']}% | DD {m['max_dd']}%")
            logger.info(f"  Expectancy {m['expectancy']}R | {m['trades_per_year']} trades/yr")
            summary[mode] = {'metrics': m, 'params': p}

    # Save results
    output = {
        'symbol':     symbol,
        'timestamp':  datetime.now().isoformat(),
        'results':    results,
        'summary':    summary,
    }

    with open(f'optimiser_v2_results_{symbol}.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f'\n✅ Results saved to optimiser_v2_results_{symbol}.json')

    # Generate strategy_config update
    generate_config_update(summary)

    return output


def generate_config_update(summary):
    """Print the strategy_config.py update based on best params"""
    logger.info('\n' + '='*60)
    logger.info('RECOMMENDED strategy_config.py UPDATES:')
    logger.info('='*60)

    for mode, data in summary.items():
        p = data['params']
        m = data['metrics']
        logger.info(f'\n# {mode.upper()} (Sharpe: {m["sharpe"]})')
        logger.info(f'  min_score:  {p.get("min_score", 55)}')
        logger.info(f'  rr_ratio:   {p.get("rr_ratio", 3.0)}')
        logger.info(f'  risk_pct:   {p.get("risk_pct", 1.0)}')
        logger.info(f'  session:    {p.get("session", "both")}')
        logger.info(f'  vix_max:    {p.get("vix_max", 25)}')
        logger.info(f'  dow:        {p.get("dow", "tue_thu")}')


if __name__ == '__main__':
    import sys
    quick = '--quick' in sys.argv
    symbol = 'NQ'
    for arg in sys.argv[1:]:
        if arg.startswith('--symbol='):
            symbol = arg.split('=')[1]

    logger.info(f'APEX v2 Optimiser — {symbol} {"(QUICK MODE)" if quick else ""}')
    run_v2_optimisation(symbol=symbol, quick=quick)
