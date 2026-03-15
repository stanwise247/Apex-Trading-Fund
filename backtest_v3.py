"""
APEX v3 Walk-Forward Backtester — backtest_v3.py
=================================================
Multiprocessing version — uses all available CPU cores.
Finds optimal session windows per instrument by testing
flexible start/end times in 30-min increments.

Run (three terminals simultaneously):
  python3 backtest_v3.py --symbol=NQ
  python3 backtest_v3.py --symbol=ES
  python3 backtest_v3.py --symbol=GC

Results saved to backtest_v3_results_{SYMBOL}.json
"""

import sqlite3
import json
import logging
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional, Dict, List, Tuple
from multiprocessing import Pool, cpu_count

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('APEX.BT3')

DB_PATH   = 'apex_market.db'
NY_TZ     = ZoneInfo('America/New_York')
POINT_VAL = 20.0
COMM      = 5.0
BALANCE   = 10000


# =============================================================
#  SESSION WINDOW GRID
# =============================================================

def build_session_windows(symbol: str) -> List[Dict]:
    if symbol == 'GC':
        candidate_times = [
            (6,0),(7,0),(8,0),(8,30),(9,30),(10,0),(10,30),
            (11,0),(12,0),(13,0),(14,0),(15,0),(16,0),
        ]
    else:
        candidate_times = [
            (4,0),(5,0),(6,0),(7,0),(8,0),(8,30),(9,0),(9,30),
            (10,0),(10,30),(11,0),(12,0),(13,0),(14,0),(15,0),(16,0),
        ]

    windows = []
    for i, (sh, sm) in enumerate(candidate_times):
        for (eh, em) in candidate_times[i+1:]:
            duration = (eh*60+em) - (sh*60+sm)
            if duration < 30 or duration > 360:
                continue
            windows.append({
                'label':   f'{sh:02d}:{sm:02d}-{eh:02d}:{em:02d}ET',
                'start_h': sh, 'start_m': sm,
                'end_h':   eh, 'end_m':   em,
            })
    return windows


DOW_FILTERS = {
    'tue_wed_thu': lambda d: d in (1, 2, 3),
    'mon_fri_out': lambda d: d not in (0, 4),
    'all_week':    lambda d: True,
}


# =============================================================
#  DATA LOADING
# =============================================================

def load_tf(symbol: str, tf: str, limit: int = 10000) -> Optional[pd.DataFrame]:
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            'SELECT ts,open,high,low,close,volume FROM ohlcv '
            'WHERE symbol=? AND timeframe=? ORDER BY ts DESC LIMIT ?',
            conn, params=(symbol, tf, limit)
        )
        conn.close()
        if len(df) < 50:
            return None
        df = df.sort_values('ts').reset_index(drop=True)  # re-sort ASC after DESC fetch
        df['dt'] = pd.to_datetime(df['ts'], unit='s', utc=True).dt.tz_convert(NY_TZ)
        for c in ['open','high','low','close','volume']:
            df[c] = df[c].astype(float)
        return df.reset_index(drop=True)
    except Exception as e:
        logger.error(f'load_tf {symbol} {tf}: {e}')
        return None


def load_vix_by_date() -> Dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            "SELECT ts, close FROM ohlcv WHERE symbol='VIX' AND timeframe='1day' ORDER BY ts ASC",
            conn
        )
        conn.close()
        df['date'] = pd.to_datetime(df['ts'], unit='s', utc=True).dt.date
        return dict(zip(df['date'], df['close'].astype(float)))
    except Exception:
        return {}


# =============================================================
#  INDICATORS
# =============================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df['close'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    v = df['volume'].astype(float)

    df['ema9']   = c.ewm(span=9,   adjust=False).mean()
    df['ema20']  = c.ewm(span=20,  adjust=False).mean()
    df['ema50']  = c.ewm(span=50,  adjust=False).mean()
    df['ema200'] = c.ewm(span=200, adjust=False).mean()

    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df['atr14'] = tr.ewm(span=14, adjust=False).mean()

    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(com=13, adjust=False).mean()
    df['rsi'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    df['date'] = df['dt'].dt.date
    tp = (h + l + c) / 3
    df['vwap'] = (tp * v).groupby(df['date']).cumsum() / \
                  v.groupby(df['date']).cumsum()
    df['vwap_dev'] = (c - df['vwap']) / df['vwap'].replace(0, np.nan) * 100

    df['vol_ma']    = v.rolling(20).mean()
    df['vol_ratio'] = v / df['vol_ma'].replace(0, np.nan)

    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df['bb_upper'] = bb_mid + 2 * bb_std
    df['bb_lower'] = bb_mid - 2 * bb_std
    df['bb_pct']   = (c - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)

    df['above_ema20']  = (c > df['ema20']).astype(int)
    df['above_ema50']  = (c > df['ema50']).astype(int)
    df['above_ema200'] = (c > df['ema200']).astype(int)
    df['trend_score']  = df['above_ema20'] + df['above_ema50'] + df['above_ema200']

    df['hh'] = (h > h.rolling(10).max().shift(1)).astype(int)
    df['ll'] = (l < l.rolling(10).min().shift(1)).astype(int)

    return df.ffill().dropna(subset=['ema20', 'atr14', 'rsi'])


# =============================================================
#  SIGNAL SCORERS
# =============================================================

def score_scalp_bar(row, htf_row, params) -> Tuple[Optional[str], int]:
    score = 0
    direction = None
    close = row['close']
    atr   = row['atr14']
    rsi   = row['rsi']
    vwap  = row['vwap']
    if atr <= 0 or pd.isna(atr): return None, 0

    if htf_row is not None:
        ts = htf_row['trend_score']
        if ts >= 2:   direction = 'long';  score += 20
        elif ts <= 1: direction = 'short'; score += 20
    if direction is None: return None, 0

    vwap_dev = row.get('vwap_dev', 0)
    if direction == 'long':
        if close > vwap and vwap_dev > 0: score += 20
        elif abs(vwap_dev) < 0.1:         score += 10
    else:
        if close < vwap and vwap_dev < 0: score += 20
        elif abs(vwap_dev) < 0.1:         score += 10

    vol_ratio = row.get('vol_ratio', 1)
    if vol_ratio >= 2.0:   score += 20
    elif vol_ratio >= 1.5: score += 12
    elif vol_ratio >= 1.2: score += 6

    dist_ema = abs(close - row['ema20']) / atr
    if dist_ema < 0.3:   score += 15
    elif dist_ema < 0.7: score += 8

    if direction == 'long':
        if 40 <= rsi <= 60:  score += 15
        elif 30 <= rsi < 40: score += 10
        elif rsi < 30:       score += 5
    else:
        if 40 <= rsi <= 60:  score += 15
        elif 60 < rsi <= 70: score += 10
        elif rsi > 70:       score += 5

    bar_range = row['high'] - row['low']
    if bar_range > 0:
        close_pos = (close - row['low']) / bar_range
        if direction == 'long'  and close_pos > 0.6: score += 10
        elif direction == 'short' and close_pos < 0.4: score += 10

    return direction, score


def score_swing_bar(row, htf_rows, params) -> Tuple[Optional[str], int]:
    score = 0
    direction = None
    close = row['close']
    atr   = row['atr14']
    rsi   = row['rsi']
    if atr <= 0 or pd.isna(atr): return None, 0

    bull_votes = sum(1 for r in htf_rows.values() if r is not None and r.get('trend_score', 1) >= 2)
    bear_votes = sum(1 for r in htf_rows.values() if r is not None and r.get('trend_score', 1) <= 1)
    min_votes  = 2 if params.get('htf_strict', True) else 1

    if bull_votes >= min_votes:   direction = 'long';  score += 10 + bull_votes * 7
    elif bear_votes >= min_votes: direction = 'short'; score += 10 + bear_votes * 7
    if direction is None: return None, 0

    vwap_dev = row.get('vwap_dev', 0)
    if direction == 'long'  and close > row['vwap']: score += 20
    elif direction == 'short' and close < row['vwap']: score += 20
    elif abs(vwap_dev) < 0.2: score += 10

    if direction == 'long'  and row.get('hh', 0):         score += 20
    elif direction == 'short' and row.get('ll', 0):        score += 20
    elif direction == 'long'  and row['above_ema50']:      score += 10
    elif direction == 'short' and not row['above_ema50']:  score += 10

    if direction == 'long':
        if 35 < rsi < 60:  score += 15
        elif rsi < 35:     score += 8
    else:
        if 40 < rsi < 65:  score += 15
        elif rsi > 65:     score += 8

    vol_ratio = row.get('vol_ratio', 1)
    if vol_ratio >= 1.5:   score += 15
    elif vol_ratio >= 1.2: score += 8

    return direction, score


# =============================================================
#  TRADE & ACCOUNT SIMULATORS
# =============================================================

def simulate_trade(entry, stop, target, direction, future_bars, max_bars=60):
    is_long = direction == 'long'
    for bar in future_bars[:max_bars]:
        bh, bl = bar[1], bar[2]
        if is_long:
            if bl <= stop:   return 'loss', stop
            if bh >= target: return 'win',  target
        else:
            if bh >= stop:   return 'loss', stop
            if bl <= target: return 'win',  target
    last_close = future_bars[min(max_bars-1, len(future_bars)-1)][3] if len(future_bars) > 0 else entry
    if (is_long and last_close > entry) or (not is_long and last_close < entry):
        return 'timeout_win', last_close
    return 'timeout_loss', last_close


def simulate_account(trades, balance=10000, risk_pct=2.0) -> Dict:
    if not trades:
        return {'sharpe': -99, 'total_return': -99, 'win_rate': 0,
                'max_dd': 0, 'expectancy': 0, 'trades': 0, 'trades_per_year': 0}

    equity = [balance]
    rets   = []
    wins   = 0
    gw     = 0.0
    gl     = 0.0
    peak   = balance
    max_dd = 0.0

    for t in trades:
        bal      = equity[-1]
        risk_pts = abs(t['entry'] - t['stop'])
        if risk_pts <= 0: continue
        size    = (bal * risk_pct / 100) / (risk_pts * POINT_VAL)
        pnl_pts = (t['exit_px'] - t['entry']) if t['direction'] == 'long' else (t['entry'] - t['exit_px'])
        pnl_usd = pnl_pts * POINT_VAL * size - COMM

        bal += pnl_usd
        equity.append(bal)
        rets.append(pnl_usd / equity[-2] if equity[-2] > 0 else 0)

        if pnl_usd > 0: wins += 1; gw += pnl_usd
        else:           gl += abs(pnl_usd)

        peak   = max(peak, bal)
        max_dd = max(max_dd, (peak - bal) / peak * 100)

    n = len(trades)
    if n == 0:
        return {'sharpe': -99, 'total_return': -99, 'win_rate': 0,
                'max_dd': 0, 'expectancy': 0, 'trades': 0, 'trades_per_year': 0}

    wr         = wins / n * 100
    avg_win_r  = (gw / wins)     / (balance * risk_pct / 100) if wins   > 0 else 0
    avg_loss_r = (gl / (n-wins)) / (balance * risk_pct / 100) if n-wins > 0 else 0
    expectancy = (wr/100 * avg_win_r) - ((1 - wr/100) * avg_loss_r)

    r_arr  = np.array(rets)
    sharpe = (np.mean(r_arr) / np.std(r_arr)) * np.sqrt(252 * 390 / max(n, 1)) \
             if len(r_arr) > 1 and np.std(r_arr) > 0 else 0

    span_days = (trades[-1]['ts'] - trades[0]['ts']) / 86400 if len(trades) >= 2 else 0
    tpy       = n / span_days * 252 if span_days > 5 else 0

    return {
        'sharpe':          round(sharpe, 3),
        'total_return':    round((equity[-1] - balance) / balance * 100, 2),
        'win_rate':        round(wr, 1),
        'max_dd':          round(max_dd, 2),
        'expectancy':      round(expectancy, 4),
        'profit_factor':   round(gw / gl, 2) if gl > 0 else 999,
        'trades':          n,
        'trades_per_year': round(tpy, 1),
        'final_balance':   round(equity[-1], 2),
    }


# =============================================================
#  WALK-FORWARD ENGINE
# =============================================================

def walk_forward(mode, params, sess, dow_key, df_entry_ind,
                 htf_snapshots, future_arr, vix_data) -> List[Dict]:
    trades     = []
    start_mins = sess['start_h'] * 60 + sess['start_m']
    end_mins   = sess['end_h']   * 60 + sess['end_m']
    dow_fn     = DOW_FILTERS[dow_key]
    vix_max    = params.get('vix_max', 25)
    min_score  = params.get('min_score', 65)
    rr_ratio   = params.get('rr_ratio', 3.0)
    risk_pct   = params.get('risk_pct', 1.5)
    stop_mult  = params.get('stop_atr', 1.0)
    last_ts    = 0
    warmup     = 50

    # Pre-extract numpy arrays for fast iteration
    ts_arr     = df_entry_ind['ts'].astype(int).values
    hour_arr   = df_entry_ind['dt'].apply(lambda d: d.hour).values
    min_arr    = df_entry_ind['dt'].apply(lambda d: d.minute).values
    dow_arr    = df_entry_ind['dt'].apply(lambda d: d.weekday()).values
    date_arr   = df_entry_ind['dt'].apply(lambda d: d.date()).values
    rows_list  = df_entry_ind.to_dict('records')

    # Pre-sort future_arr ts column index for fast slicing
    future_ts  = future_arr[:, 0]

    for i in range(warmup, len(rows_list) - 1):
        ts     = ts_arr[i]
        t_mins = hour_arr[i] * 60 + min_arr[i]

        if ts - last_ts < 300:                        continue
        if not (start_mins <= t_mins < end_mins):     continue
        if not dow_fn(int(dow_arr[i])):               continue

        vix = vix_data.get(date_arr[i])
        if vix and vix > vix_max: continue

        # O(log n) binary search for HTF snapshot
        htf_rows = {}
        for name, snap in htf_snapshots.items():
            idx = np.searchsorted(snap['ts_arr'], ts, side='right') - 1
            if idx >= 0:
                htf_rows[name] = snap['rows'][idx]

        row     = rows_list[i]
        htf_row = htf_rows.get('1h')

        if mode == 'scalp':
            direction, score = score_scalp_bar(row, htf_row, params)
        elif mode == 'swing':
            direction, score = score_swing_bar(row, htf_rows, params)
        else:
            continue

        if direction is None or score < min_score: continue

        entry    = row['close']
        atr      = row['atr14']
        if atr <= 0: continue

        if direction == 'long':
            stop   = entry - atr * stop_mult
            target = entry + abs(entry - stop) * rr_ratio
        else:
            stop   = entry + atr * stop_mult
            target = entry - abs(entry - stop) * rr_ratio

        if abs(entry - stop) <= 0: continue

        # O(log n) future bars lookup
        fut_idx = np.searchsorted(future_ts, ts, side='right')
        future_from = future_arr[fut_idx:]
        if len(future_from) < 5: continue

        outcome, exit_px = simulate_trade(entry, stop, target, direction, future_from)
        pnl_pts = (exit_px - entry) if direction == 'long' else (entry - exit_px)

        trades.append({
            'ts':        ts,
            'direction': direction,
            'entry':     round(entry, 2),
            'stop':      round(stop, 2),
            'target':    round(target, 2),
            'exit_px':   round(exit_px, 2),
            'outcome':   outcome,
            'pnl_pts':   round(pnl_pts, 2),
            'score':     score,
            'risk_pct':  risk_pct,
        })
        last_ts = ts

    return trades


# =============================================================
#  WORKER (multiprocessing with pool initializer)
#  Data is loaded ONCE per process, not copied per task.
# =============================================================

_worker_state = {}

def pool_initializer(df_entry_ind, htf_snapshots, future_arr, vix_data, min_trades):
    """Runs once per worker process — stores shared data in global."""
    _worker_state['df_entry_ind']  = df_entry_ind
    _worker_state['htf_snapshots'] = htf_snapshots
    _worker_state['future_arr']    = future_arr
    _worker_state['vix_data']      = vix_data
    _worker_state['min_trades']    = min_trades


def worker(args):
    """Only receives lightweight args — reads data from process-local state."""
    mode, params, sess, dow_key = args
    df_entry_ind  = _worker_state['df_entry_ind']
    htf_snapshots = _worker_state['htf_snapshots']
    future_arr    = _worker_state['future_arr']
    vix_data      = _worker_state['vix_data']
    min_trades    = _worker_state['min_trades']

    trades  = walk_forward(mode, params, sess, dow_key, df_entry_ind,
                           htf_snapshots, future_arr, vix_data)
    metrics = simulate_account(trades, BALANCE, params.get('risk_pct', 1.5))

    if metrics['trades'] < min_trades or metrics['sharpe'] <= 0:
        return None

    return {
        'params':  {**params, 'mode': mode, 'session': sess['label'], 'dow': dow_key},
        'session': dict(sess),
        'metrics': metrics,
    }


def build_htf_snapshots(htf_dfs: Dict) -> Dict:
    """Build sorted arrays for O(log n) binary search lookup."""
    snapshots = {}
    for name, df in htf_dfs.items():
        ts_arr = df['ts'].astype(int).values
        rows   = df.to_dict('records')
        snapshots[name] = {'ts_arr': ts_arr, 'rows': rows}
    return snapshots


# =============================================================
#  OPTIMISER
# =============================================================

def optimise(symbol, mode, param_grid, df_entry_ind, htf_snapshots,
             future_arr, vix_data, session_windows, min_trades=10, n_cores=8):

    dow_keys     = list(DOW_FILTERS.keys())
    param_keys   = list(param_grid.keys())
    param_combos = [dict(zip(param_keys, c)) for c in itertools.product(*param_grid.values())]

    # Lightweight work items — no data, just instructions
    work = [
        (mode, params, sess, dow_key)
        for sess in session_windows
        for dow_key in dow_keys
        for params in param_combos
    ]

    total = len(work)
    logger.info(f'\n{"="*60}')
    logger.info(f'Optimising {symbol} {mode.upper()} — {total} combos on {n_cores} cores')
    logger.info(f'{"="*60}')

    all_results = []
    done        = 0

    with Pool(
        processes=n_cores,
        initializer=pool_initializer,
        initargs=(df_entry_ind, htf_snapshots, future_arr, vix_data, min_trades)
    ) as pool:
        for result in pool.imap_unordered(worker, work, chunksize=50):
            done += 1
            if result is not None:
                all_results.append(result)
            if done % 1000 == 0:
                best = max(all_results, key=lambda x: x['metrics']['sharpe'])['metrics']['sharpe'] \
                       if all_results else 0
                logger.info(f'  {done}/{total} — best Sharpe: {best:.3f} ({len(all_results)} valid)')

    if not all_results:
        logger.warning(f'No valid results for {symbol} {mode}')
        return None

    all_results.sort(key=lambda x: x['metrics']['sharpe'], reverse=True)
    best = all_results[0]
    m    = best['metrics']
    p    = best['params']
    s    = best['session']

    logger.info(f'\n✅ BEST {symbol} {mode.upper()}:')
    logger.info(f"  Sharpe:    {m['sharpe']}")
    logger.info(f"  Return:    {m['total_return']}%")
    logger.info(f"  Win Rate:  {m['win_rate']}%")
    logger.info(f"  Max DD:    {m['max_dd']}%")
    logger.info(f"  Trades/yr: {m['trades_per_year']}")
    logger.info(f"  Session:   {s['label']}")
    logger.info(f"  DoW:       {p['dow']}")
    logger.info(f"  Params:    { {k:v for k,v in p.items() if k not in ('mode','session','dow')} }")

    return {
        'mode':         mode,
        'best':         best,
        'top10':        all_results[:10],
        'total_tested': total,
        'valid_combos': len(all_results),
    }


# =============================================================
#  INSTRUMENT PARAM GRIDS
# =============================================================

GRIDS = {
    'NQ': {
        'scalp': {
            'min_score': [65, 70],
            'rr_ratio':  [2.0, 2.5, 3.0],
            'risk_pct':  [1.0, 1.5],
            'vix_max':   [20, 25],
            'stop_atr':  [0.8, 1.0, 1.5],
        },
        'swing': {
            'min_score':  [65, 70],
            'rr_ratio':   [3.0, 4.0],
            'risk_pct':   [1.5, 2.0],
            'vix_max':    [20, 25],
            'stop_atr':   [1.2, 1.5, 2.0],
            'htf_strict': [True, False],
        },
    },
    'ES': {
        'swing': {
            'min_score':  [65, 70],
            'rr_ratio':   [3.0, 4.0],
            'risk_pct':   [1.5, 2.0],
            'vix_max':    [20, 25],
            'stop_atr':   [1.5, 2.0, 2.5],
            'htf_strict': [True, False],
        },
    },
    'GC': {
        'scalp': {
            'min_score': [65, 70],
            'rr_ratio':  [1.5, 2.0, 2.5],
            'risk_pct':  [1.0, 1.5],
            'vix_max':   [20, 25],
            'stop_atr':  [1.0, 1.5, 2.0],
        },
        'swing': {
            'min_score':  [65, 70],
            'rr_ratio':   [3.0, 4.0],
            'risk_pct':   [1.5, 2.0],
            'vix_max':    [20, 25],
            'stop_atr':   [1.5, 2.0, 2.5],
            'htf_strict': [True, False],
        },
    },
}


# =============================================================
#  MAIN
# =============================================================

def run_backtest_v3(symbol: str = 'NQ'):
    logger.info(f'\n{"#"*60}')
    logger.info(f'  APEX v3 Backtest — {symbol}')
    logger.info(f'{"#"*60}')

    if symbol not in GRIDS:
        logger.error(f'Unknown symbol: {symbol}. Use NQ, ES or GC.')
        return

    n_cores = min(cpu_count(), 8)
    logger.info(f'  Using {n_cores} CPU cores')

    logger.info('Loading data...')
    df_5min  = load_tf(symbol, '5min',  10000)
    df_15min = load_tf(symbol, '15min', 10000)
    df_1h    = load_tf(symbol, '1hour', 5000)
    df_4h    = load_tf(symbol, '4hour', 2000)
    df_1d    = load_tf(symbol, '1day',  800)
    vix_data = load_vix_by_date()

    for name, df in [('5min',df_5min),('15min',df_15min),('1h',df_1h),('4h',df_4h)]:
        logger.info(f'  {name}: {len(df) if df is not None else 0} bars')

    if df_5min is None or len(df_5min) < 200:
        logger.error('Need at least 200 5min bars.')
        return

    session_windows = build_session_windows(symbol)
    logger.info(f'  Session windows to test: {len(session_windows)}')

    logger.info('Pre-computing indicators...')
    htf_dfs = {}
    for name, df in [('1h', df_1h), ('4h', df_4h), ('1d', df_1d)]:
        if df is not None and len(df) > 20:
            htf_dfs[name] = add_indicators(df.copy())

    df_5min_ind  = add_indicators(df_5min.copy())
    df_15min_ind = add_indicators(df_15min.copy()) \
                   if df_15min is not None and len(df_15min) > 50 else df_5min_ind

    htf_snapshots = build_htf_snapshots(htf_dfs)
    logger.info(f'  HTF snapshots: {list(htf_snapshots.keys())}')

    future_5min  = df_5min[['ts','high','low','close']].values
    future_15min = df_15min[['ts','high','low','close']].values \
                   if df_15min is not None and len(df_15min) > 50 else future_5min

    results = {}

    for mode, grid in GRIDS[symbol].items():
        entry_ind  = df_5min_ind  if mode == 'scalp' else df_15min_ind
        future_arr = future_5min  if mode == 'scalp' else future_15min

        r = optimise(symbol, mode, grid, entry_ind, htf_snapshots,
                     future_arr, vix_data, session_windows, n_cores=n_cores)
        if r:
            results[mode] = r

    logger.info(f'\n{"="*60}')
    logger.info(f'BACKTEST v3 COMPLETE — {symbol}')
    logger.info(f'{"="*60}')

    summary = {}
    for mode, r in results.items():
        m = r['best']['metrics']
        p = r['best']['params']
        s = r['best']['session']
        logger.info(f'\n{mode.upper()}:')
        logger.info(f"  Sharpe {m['sharpe']} | Return {m['total_return']}% | WR {m['win_rate']}%")
        logger.info(f"  DD {m['max_dd']}% | {m['trades_per_year']} trades/yr")
        logger.info(f"  Session: {s['label']}  DoW: {p['dow']}")
        summary[mode] = {'metrics': m, 'params': p, 'session': s}

    output = {
        'symbol':    symbol,
        'timestamp': datetime.now().isoformat(),
        'results':   results,
        'summary':   summary,
    }

    fname = f'backtest_v3_results_{symbol}.json'
    with open(fname, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f'\n✅ Saved to {fname}')

    logger.info(f'\n{"="*60}')
    logger.info(f'RECOMMENDED PARAMS — {symbol}')
    logger.info(f'{"="*60}')
    for mode, data in summary.items():
        m = data['metrics']
        p = data['params']
        s = data['session']
        logger.info(f'\n# {symbol} {mode.upper()} (Sharpe: {m["sharpe"]})')
        logger.info(f'  session: {s["start_h"]:02d}:{s["start_m"]:02d}–{s["end_h"]:02d}:{s["end_m"]:02d} ET')
        logger.info(f'  dow:     {p["dow"]}')
        for k, v in p.items():
            if k not in ('mode', 'session', 'dow'):
                logger.info(f'  {k}: {v}')

    return output


if __name__ == '__main__':
    import sys
    symbol = next((a.split('=')[1] for a in sys.argv if a.startswith('--symbol=')), 'NQ')
    run_backtest_v3(symbol=symbol)
