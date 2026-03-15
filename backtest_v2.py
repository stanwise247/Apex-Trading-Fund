"""
APEX v2 Walk-Forward Backtester — backtest_v2.py
=================================================
Replays historical bars through the ACTUAL strategy files
(strategy_scalp, strategy_swing, strategy_meanrev) to find
genuine edge with real signal generation logic.

Method:
  1. Load full OHLCV history from DB
  2. Walk forward bar by bar through trading hours
  3. At each bar, feed the ACTUAL strategy scanners
     (same code that runs live)
  4. When a signal fires, forward-test the outcome
  5. Record P&L, metrics per parameter combination

This is the real test — using the same logic that will
trade live. No simplified proxies.

Run: python3 backtest_v2.py
"""

import sqlite3
import json
import logging
import itertools
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Optional, Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('APEX.BT2')

DB_PATH   = 'apex_market.db'
NY_TZ     = ZoneInfo('America/New_York')
POINT_VAL = 20.0
COMM      = 5.0
SYMBOL    = 'NQ'
BALANCE   = 10000


# =============================================================
#  DATA LOADING
# =============================================================

def load_tf(symbol: str, tf: str, limit: int = 10000) -> Optional[pd.DataFrame]:
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            'SELECT ts,open,high,low,close,volume FROM ohlcv '
            'WHERE symbol=? AND timeframe=? ORDER BY ts ASC LIMIT ?',
            conn, params=(symbol, tf, limit)
        )
        conn.close()
        if len(df) < 50:
            return None
        df['dt'] = pd.to_datetime(df['ts'], unit='s', utc=True).dt.tz_convert(NY_TZ)
        for c in ['open','high','low','close','volume']:
            df[c] = df[c].astype(float)
        df = df.reset_index(drop=True)
        return df
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
#  INDICATOR ENGINE
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

    # VWAP (daily reset)
    df['date'] = df['dt'].dt.date
    tp = (h + l + c) / 3
    df['vwap'] = (tp * v).groupby(df['date']).cumsum() / \
                  v.groupby(df['date']).cumsum()
    df['vwap_dev'] = (c - df['vwap']) / df['vwap'].replace(0, np.nan) * 100

    # Volume ratio
    df['vol_ma']    = v.rolling(20).mean()
    df['vol_ratio'] = v / df['vol_ma'].replace(0, np.nan)

    # Bollinger Bands
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df['bb_upper'] = bb_mid + 2 * bb_std
    df['bb_lower'] = bb_mid - 2 * bb_std
    df['bb_pct']   = (c - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)

    # HTF trend
    df['above_ema20']  = (c > df['ema20']).astype(int)
    df['above_ema50']  = (c > df['ema50']).astype(int)
    df['above_ema200'] = (c > df['ema200']).astype(int)
    df['trend_score']  = df['above_ema20'] + df['above_ema50'] + df['above_ema200']

    # Swing highs/lows (3-bar)
    df['swing_high'] = ((h > h.shift(1)) & (h > h.shift(-1))).astype(int)
    df['swing_low']  = ((l < l.shift(1)) & (l < l.shift(-1))).astype(int)

    # Market structure BOS
    df['hh'] = (h > h.rolling(10).max().shift(1)).astype(int)  # Higher high
    df['ll'] = (l < l.rolling(10).min().shift(1)).astype(int)  # Lower low

    return df.ffill().dropna(subset=['ema20', 'atr14', 'rsi'])


# =============================================================
#  SESSION FILTERS
# =============================================================

SESSION_WINDOWS = {
    'london':    lambda h, m: 5 <= h < 8,
    'ny_open':   lambda h, m: (h == 9 and m >= 30) or (h == 10 and m < 30),
    'both':      lambda h, m: (5 <= h < 8) or (h == 9 and m >= 30) or (h == 10 and m < 30),
    'full_day':  lambda h, m: 5 <= h < 16,
}

DOW_FILTERS = {
    'tue_wed_thu': lambda d: d in (1, 2, 3),
    'mon_fri_out': lambda d: d not in (0, 4),
    'all_week':    lambda d: True,
}


# =============================================================
#  SIGNAL GENERATORS (using real indicator logic)
# =============================================================

def score_scalp_bar(row, df_slice, htf_row, params) -> Tuple[Optional[str], int, Dict]:
    """Score a single bar for scalp setup — mirrors strategy_scalp.py logic"""
    score = 0
    direction = None
    details = {}

    close = row['close']
    atr   = row['atr14']
    rsi   = row['rsi']
    vwap  = row['vwap']

    if atr <= 0 or pd.isna(atr):
        return None, 0, {}

    # 1. HTF BIAS (20pts) — use 1hr slice
    if htf_row is not None:
        ts = htf_row['trend_score']
        if ts >= 2:
            direction = 'long'
            score += 20
            details['htf'] = f'bullish ts={ts}'
        elif ts <= 1:
            direction = 'short'
            score += 20
            details['htf'] = f'bearish ts={ts}'

    if direction is None:
        return None, 0, {}

    # 2. VWAP CONFLUENCE (20pts)
    vwap_dev = row.get('vwap_dev', 0)
    if direction == 'long':
        if close > vwap and vwap_dev > 0:
            score += 20
            details['vwap'] = 'above vwap'
        elif abs(vwap_dev) < 0.1:
            score += 10
            details['vwap'] = 'at vwap'
    else:
        if close < vwap and vwap_dev < 0:
            score += 20
            details['vwap'] = 'below vwap'
        elif abs(vwap_dev) < 0.1:
            score += 10
            details['vwap'] = 'at vwap'

    # 3. ORDER FLOW (20pts) — volume + momentum
    vol_ratio = row.get('vol_ratio', 1)
    if vol_ratio >= 2.0:
        score += 20
        details['vol'] = f'spike {vol_ratio:.1f}x'
    elif vol_ratio >= 1.5:
        score += 12
        details['vol'] = f'elevated {vol_ratio:.1f}x'
    elif vol_ratio >= 1.2:
        score += 6
        details['vol'] = f'above avg {vol_ratio:.1f}x'

    # 4. STRUCTURE (15pts) — EMA proximity
    ema20 = row['ema20']
    dist_ema = abs(close - ema20) / atr
    if dist_ema < 0.3:
        score += 15
        details['structure'] = 'ema20 bounce'
    elif dist_ema < 0.7:
        score += 8

    # 5. RSI (15pts)
    if direction == 'long':
        if 40 <= rsi <= 60:
            score += 15
            details['rsi'] = f'neutral {rsi:.0f}'
        elif 30 <= rsi < 40:
            score += 10
            details['rsi'] = f'oversold {rsi:.0f}'
        elif rsi < 30:
            score += 5  # Too oversold for momentum scalp
    else:
        if 40 <= rsi <= 60:
            score += 15
        elif 60 < rsi <= 70:
            score += 10
        elif rsi > 70:
            score += 5

    # 6. MOMENTUM (10pts) — bar close position
    bar_range = row['high'] - row['low']
    if bar_range > 0:
        close_pos = (close - row['low']) / bar_range
        if direction == 'long' and close_pos > 0.6:
            score += 10
            details['momentum'] = 'bullish close'
        elif direction == 'short' and close_pos < 0.4:
            score += 10
            details['momentum'] = 'bearish close'

    return direction, score, details


def score_swing_bar(row, df_slice, htf_rows, params) -> Tuple[Optional[str], int, Dict]:
    """Score a single bar for swing setup — mirrors strategy_swing.py logic"""
    score = 0
    direction = None
    details = {}

    close = row['close']
    atr   = row['atr14']
    rsi   = row['rsi']

    if atr <= 0 or pd.isna(atr):
        return None, 0, {}

    # 1. MULTI-TF BIAS (30pts) — needs agreement across TFs
    bull_votes = 0
    bear_votes = 0

    for tf_name, htf_row in htf_rows.items():
        if htf_row is None:
            continue
        ts = htf_row.get('trend_score', 1)
        if ts >= 2:
            bull_votes += 1
        elif ts <= 1:
            bear_votes += 1

    min_votes = 2 if params.get('htf_strict', True) else 1

    if bull_votes >= min_votes:
        direction = 'long'
        score += 10 + bull_votes * 7
        details['htf'] = f'{bull_votes} bull TFs'
    elif bear_votes >= min_votes:
        direction = 'short'
        score += 10 + bear_votes * 7
        details['htf'] = f'{bear_votes} bear TFs'

    if direction is None:
        return None, 0, {}

    # 2. VWAP (20pts)
    vwap = row['vwap']
    vwap_dev = row.get('vwap_dev', 0)
    if direction == 'long' and close > vwap:
        score += 20
        details['vwap'] = f'+{vwap_dev:.1f}% above'
    elif direction == 'short' and close < vwap:
        score += 20
        details['vwap'] = f'{vwap_dev:.1f}% below'
    elif abs(vwap_dev) < 0.2:
        score += 10
        details['vwap'] = 'at vwap'

    # 3. STRUCTURE BOS (20pts)
    if direction == 'long' and row.get('hh', 0):
        score += 20
        details['structure'] = 'BOS up'
    elif direction == 'short' and row.get('ll', 0):
        score += 20
        details['structure'] = 'BOS down'
    elif direction == 'long' and row['above_ema50']:
        score += 10
        details['structure'] = 'above ema50'
    elif direction == 'short' and not row['above_ema50']:
        score += 10
        details['structure'] = 'below ema50'

    # 4. RSI (15pts)
    if direction == 'long' and 35 < rsi < 60:
        score += 15
        details['rsi'] = f'{rsi:.0f}'
    elif direction == 'short' and 40 < rsi < 65:
        score += 15
    elif direction == 'long' and rsi < 35:
        score += 8  # Oversold pullback
    elif direction == 'short' and rsi > 65:
        score += 8

    # 5. VOLUME (15pts)
    vol_ratio = row.get('vol_ratio', 1)
    if vol_ratio >= 1.5:
        score += 15
        details['vol'] = f'{vol_ratio:.1f}x'
    elif vol_ratio >= 1.2:
        score += 8

    return direction, score, details


def score_meanrev_bar(row, df_slice, params) -> Tuple[Optional[str], int, Dict]:
    """Score a single bar for mean reversion — mirrors strategy_meanrev.py logic"""
    score = 0
    direction = None
    details = {}

    close    = row['close']
    atr      = row['atr14']
    rsi      = row['rsi']
    vwap_dev = row.get('vwap_dev', 0)
    bb_pct   = row.get('bb_pct', 0.5)

    if atr <= 0 or pd.isna(atr):
        return None, 0, {}

    dev_thresh = params.get('vwap_dev_thresh', 2.5)

    # 1. VWAP EXTREME (30pts) — core signal
    if vwap_dev <= -dev_thresh:
        direction = 'long'
        score += 20 + min(int(abs(vwap_dev) - dev_thresh) * 5, 10)
        details['vwap_dev'] = f'{vwap_dev:.1f}%'
    elif vwap_dev >= dev_thresh:
        direction = 'short'
        score += 20 + min(int(vwap_dev - dev_thresh) * 5, 10)
        details['vwap_dev'] = f'+{vwap_dev:.1f}%'
    else:
        return None, 0, {}

    # 2. RSI EXTREME (25pts)
    if direction == 'long':
        if rsi < 25:
            score += 25
            details['rsi'] = f'extreme {rsi:.0f}'
        elif rsi < 35:
            score += 15
            details['rsi'] = f'oversold {rsi:.0f}'
        elif rsi < 45:
            score += 5
    else:
        if rsi > 75:
            score += 25
            details['rsi'] = f'extreme {rsi:.0f}'
        elif rsi > 65:
            score += 15
            details['rsi'] = f'overbought {rsi:.0f}'
        elif rsi > 55:
            score += 5

    # 3. BB EXTREME (20pts)
    if direction == 'long' and bb_pct < 0.05:
        score += 20
        details['bb'] = 'at lower band'
    elif direction == 'long' and bb_pct < 0.15:
        score += 10
    elif direction == 'short' and bb_pct > 0.95:
        score += 20
        details['bb'] = 'at upper band'
    elif direction == 'short' and bb_pct > 0.85:
        score += 10

    # 4. EXHAUSTION (15pts) — low volume at extreme = exhaustion
    vol_ratio = row.get('vol_ratio', 1)
    if vol_ratio < 0.7:
        score += 15
        details['exhaustion'] = f'low vol {vol_ratio:.1f}x'
    elif vol_ratio < 0.9:
        score += 7

    # 5. MARKET CONDITION (10pts) — ranging = better for MR
    if params.get('market_cond', 'any') == 'ranging_only':
        # Proxy: low EMA separation = ranging
        ema_sep = abs(row['ema20'] - row['ema50']) / atr if atr > 0 else 99
        if ema_sep > 2.5:
            return None, 0, {}  # Trending — skip
        score += 10
        details['condition'] = 'ranging'
    else:
        score += 5

    return direction, score, details


# =============================================================
#  TRADE OUTCOME SIMULATOR
# =============================================================

def simulate_trade(
    entry: float,
    stop: float,
    target: float,
    direction: str,
    future_bars: np.ndarray,  # [ts, high, low, close]
    max_bars: int = 60,
) -> Tuple[str, float]:
    """Forward-test a trade against future bars. Returns (outcome, exit_price)"""
    is_long = direction == 'long'

    for bar in future_bars[:max_bars]:
        bar_high, bar_low = bar[1], bar[2]

        if is_long:
            if bar_low <= stop:
                return 'loss', stop
            if bar_high >= target:
                return 'win', target
        else:
            if bar_high >= stop:
                return 'loss', stop
            if bar_low <= target:
                return 'win', target

    # Timeout — use close of last bar
    last_close = future_bars[min(max_bars-1, len(future_bars)-1)][3] if len(future_bars) > 0 else entry
    if (is_long and last_close > entry) or (not is_long and last_close < entry):
        return 'timeout_win', last_close
    return 'timeout_loss', last_close


# =============================================================
#  ACCOUNT SIMULATOR
# =============================================================

def simulate_account(trades: List[Dict], balance: float = 10000, risk_pct: float = 2.0) -> Dict:
    if not trades:
        return {'sharpe': -99, 'total_return': -99, 'win_rate': 0, 'max_dd': 0,
                'expectancy': 0, 'profit_factor': 0, 'trades': 0, 'trades_per_year': 0}

    equity  = [balance]
    rets    = []
    wins    = 0
    gross_w = 0
    gross_l = 0
    peak    = balance
    max_dd  = 0

    for t in trades:
        bal      = equity[-1]
        risk_amt = bal * risk_pct / 100
        entry    = t['entry']
        stop     = t['stop']
        exit_px  = t['exit_px']
        is_long  = t['direction'] == 'long'

        risk_pts = abs(entry - stop)
        if risk_pts <= 0: continue

        size_contracts = risk_amt / (risk_pts * POINT_VAL)
        pnl_pts = (exit_px - entry) if is_long else (entry - exit_px)
        pnl_usd = pnl_pts * POINT_VAL * size_contracts - COMM

        bal += pnl_usd
        equity.append(bal)

        ret = pnl_usd / equity[-2] if equity[-2] > 0 else 0
        rets.append(ret)

        if pnl_usd > 0:
            wins    += 1
            gross_w += pnl_usd
        else:
            gross_l += abs(pnl_usd)

        peak   = max(peak, bal)
        dd     = (peak - bal) / peak * 100
        max_dd = max(max_dd, dd)

    n = len(trades)
    if n == 0:
        return {'sharpe': -99, 'total_return': -99, 'win_rate': 0, 'max_dd': 0,
                'expectancy': 0, 'profit_factor': 0, 'trades': 0, 'trades_per_year': 0}

    total_return = (equity[-1] - balance) / balance * 100
    win_rate     = wins / n * 100
    pf           = gross_w / gross_l if gross_l > 0 else 999

    # Expectancy in R
    avg_win_r  = (gross_w / wins)    / (balance * risk_pct / 100) if wins     > 0 else 0
    avg_loss_r = (gross_l / (n-wins)) / (balance * risk_pct / 100) if n-wins > 0 else 0
    expectancy = (win_rate/100 * avg_win_r) - ((1-win_rate/100) * avg_loss_r)

    # Annualised Sharpe
    r_arr  = np.array(rets)
    sharpe = (np.mean(r_arr) / np.std(r_arr)) * np.sqrt(252 * 390 / max(n, 1)) \
             if len(r_arr) > 1 and np.std(r_arr) > 0 else 0

    # Trades per year
    if len(trades) >= 2:
        span_days = (trades[-1]['ts'] - trades[0]['ts']) / 86400
        tpy = n / span_days * 252 if span_days > 5 else 0
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
        'trades_per_year': round(tpy, 1),
        'final_balance':   round(equity[-1], 2),
        'gross_wins':      round(gross_w, 2),
        'gross_losses':    round(gross_l, 2),
    }


# =============================================================
#  WALK-FORWARD ENGINE
# =============================================================

def walk_forward(
    mode: str,
    params: Dict,
    df_entry: pd.DataFrame,       # Primary entry TF
    df_15min: pd.DataFrame,
    df_1h: Optional[pd.DataFrame],
    df_4h: Optional[pd.DataFrame],
    df_1d: Optional[pd.DataFrame],
    vix_data: Dict,
    future_df: pd.DataFrame,      # For outcome simulation
) -> List[Dict]:
    """Walk forward through history generating signals"""
    trades = []
    sess_fn = SESSION_WINDOWS[params.get('session', 'both')]
    dow_fn  = DOW_FILTERS[params.get('dow', 'tue_wed_thu')]
    vix_max = params.get('vix_max', 25)
    min_score = params.get('min_score', 55)
    rr_ratio  = params.get('rr_ratio', 3.0)
    risk_pct  = params.get('risk_pct', 1.0)

    future_arr = future_df[['ts','high','low','close']].values
    entry_arr  = df_entry.values
    warmup     = 50

    # Pre-build HTF snapshots for efficiency
    htf_dfs = {}
    for name, df in [('1h', df_1h), ('4h', df_4h), ('1d', df_1d)]:
        if df is not None and len(df) > 20:
            htf_dfs[name] = add_indicators(df.copy())

    df_entry_ind = add_indicators(df_entry.copy())
    df_15min_ind = add_indicators(df_15min.copy()) if df_15min is not None and len(df_15min) > 20 else None

    last_trade_ts = 0  # Prevent same-bar re-entry

    for i in range(warmup, len(df_entry_ind) - 1):
        row = df_entry_ind.iloc[i]
        dt  = row['dt']
        ts  = int(row['ts'])

        if ts - last_trade_ts < 300:  # 5min cooldown
            continue

        h, m, d = dt.hour, dt.minute, dt.weekday()

        if not sess_fn(h, m): continue
        if not dow_fn(d): continue

        vix = vix_data.get(dt.date())
        if vix and vix > vix_max: continue

        df_slice = df_entry_ind.iloc[max(0, i-20):i+1]

        # Get HTF snapshot at this point in time
        htf_rows = {}
        for name, htf_df in htf_dfs.items():
            htf_at_time = htf_df[htf_df['ts'] <= ts]
            if len(htf_at_time) > 0:
                htf_rows[name] = htf_at_time.iloc[-1]

        htf_row = htf_rows.get('1h')

        # Score the bar
        if mode == 'scalp':
            direction, score, details = score_scalp_bar(row, df_slice, htf_row, params)
        elif mode == 'swing':
            direction, score, details = score_swing_bar(row, df_slice, htf_rows, params)
        elif mode == 'meanrev':
            direction, score, details = score_meanrev_bar(row, df_slice, params)
        else:
            continue

        if direction is None or score < min_score:
            continue

        entry = row['close']
        atr   = row['atr14']

        if atr <= 0:
            continue

        # Calculate stop and target based on mode
        if mode == 'scalp':
            stop_mult = params.get('stop_atr', 1.0)
        elif mode == 'swing':
            stop_mult = params.get('stop_atr', 1.5)
        else:  # meanrev
            stop_mult = params.get('stop_atr', 0.8)

        if direction == 'long':
            stop   = entry - atr * stop_mult
            target = entry + abs(entry - stop) * rr_ratio
        else:
            stop   = entry + atr * stop_mult
            target = entry - abs(entry - stop) * rr_ratio

        risk_pts = abs(entry - stop)
        if risk_pts <= 0:
            continue

        # Forward test
        future_from = future_arr[future_arr[:, 0] > ts]
        if len(future_from) < 5:
            continue

        outcome, exit_px = simulate_trade(
            entry, stop, target, direction, future_from,
            max_bars=params.get('max_bars', 50)
        )

        pnl_pts = (exit_px - entry) if direction == 'long' else (entry - exit_px)

        trades.append({
            'ts':        ts,
            'dt':        str(dt),
            'direction': direction,
            'entry':     round(entry, 2),
            'stop':      round(stop, 2),
            'target':    round(target, 2),
            'exit_px':   round(exit_px, 2),
            'outcome':   outcome,
            'pnl_pts':   round(pnl_pts, 2),
            'score':     score,
            'details':   details,
            'risk_pct':  risk_pct,
            'vix':       vix,
            'mode':      mode,
        })

        last_trade_ts = ts

    return trades


# =============================================================
#  PARAMETER GRIDS
# =============================================================

SCALP_GRID = {
    'min_score':  [55, 60, 65, 70],
    'rr_ratio':   [2.0, 2.5, 3.0],
    'risk_pct':   [0.5, 1.0, 1.5],
    'session':    ['london', 'ny_open', 'both'],
    'vix_max':    [20, 25, 30],
    'dow':        ['tue_wed_thu', 'mon_fri_out', 'all_week'],
    'stop_atr':   [0.8, 1.0, 1.5],
}

SWING_GRID = {
    'min_score':  [55, 60, 65, 70],
    'rr_ratio':   [2.5, 3.0, 4.0],
    'risk_pct':   [1.0, 1.5, 2.0],
    'session':    ['london', 'ny_open', 'both'],
    'vix_max':    [20, 25, 30],
    'dow':        ['tue_wed_thu', 'mon_fri_out', 'all_week'],
    'htf_strict': [True, False],
    'stop_atr':   [1.2, 1.5, 2.0],
}

MEANREV_GRID = {
    'min_score':       [55, 60, 65, 70],
    'rr_ratio':        [1.5, 2.0, 2.5],
    'risk_pct':        [0.5, 1.0, 1.5],
    'vix_max':         [18, 20, 25],
    'dow':             ['tue_wed_thu', 'mon_fri_out', 'all_week'],
    'vwap_dev_thresh': [2.0, 2.5, 3.0],
    'market_cond':     ['ranging_only', 'any'],
    'stop_atr':        [0.6, 0.8, 1.0],
}


# =============================================================
#  OPTIMISER
# =============================================================

def optimise_mode(
    mode: str,
    grid: Dict,
    df_entry: pd.DataFrame,
    df_15min: pd.DataFrame,
    df_1h, df_4h, df_1d,
    vix_data: Dict,
    min_trades: int = 10,
) -> Optional[Dict]:

    keys   = list(grid.keys())
    values = list(grid.values())
    combos = list(itertools.product(*values))
    total  = len(combos)

    logger.info(f'\n{"="*60}')
    logger.info(f'Optimising {mode.upper()} — {total} combinations')
    logger.info(f'{"="*60}')

    all_results = []

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        params['mode'] = mode

        trades  = walk_forward(mode, params, df_entry, df_15min, df_1h, df_4h, df_1d, vix_data, df_entry)
        metrics = simulate_account(trades, BALANCE, params['risk_pct'])

        if metrics['trades'] >= min_trades and metrics['sharpe'] > -99:
            all_results.append({'params': params, 'metrics': metrics, 'n_trades': len(trades)})

        if (i+1) % 100 == 0:
            best_so_far = max(all_results, key=lambda x: x['metrics']['sharpe'])['metrics']['sharpe'] if all_results else 0
            logger.info(f'  {i+1}/{total} — best Sharpe so far: {best_so_far:.3f} ({len(all_results)} valid)')

    if not all_results:
        logger.warning(f'No valid results for {mode} (need {min_trades}+ trades)')
        return None

    all_results.sort(key=lambda x: x['metrics']['sharpe'], reverse=True)
    best = all_results[0]

    logger.info(f'\n✅ BEST {mode.upper()}:')
    m = best['metrics']
    p = best['params']
    logger.info(f"  Sharpe:       {m['sharpe']}")
    logger.info(f"  Return:       {m['total_return']}%")
    logger.info(f"  Win Rate:     {m['win_rate']}%")
    logger.info(f"  Expectancy:   {m['expectancy']}R")
    logger.info(f"  Max DD:       {m['max_dd']}%")
    logger.info(f"  Trades/yr:    {m['trades_per_year']}")
    logger.info(f"  Total trades: {m['trades']}")
    logger.info(f"  Params:       {p}")

    return {
        'mode':         mode,
        'best':         best,
        'top10':        all_results[:10],
        'total_tested': len(all_results),
    }


# =============================================================
#  MAIN
# =============================================================

def run_backtest_v2(symbol: str = 'NQ', quick: bool = False):
    logger.info(f'APEX v2 Backtest — {symbol} {"QUICK" if quick else "FULL"}')
    logger.info('Loading data...')

    lim = 2000 if quick else 10000

    df_5min  = load_tf(symbol, '5min',  lim)
    df_15min = load_tf(symbol, '15min', lim)
    df_1h    = load_tf(symbol, '1hour', lim // 2)
    df_4h    = load_tf(symbol, '4hour', lim // 4)
    df_1d    = load_tf(symbol, '1day',  800)
    vix_data = load_vix_by_date()

    for name, df in [('5min',df_5min),('15min',df_15min),('1h',df_1h)]:
        bars = len(df) if df is not None else 0
        logger.info(f'  {name}: {bars} bars')

    if df_5min is None or len(df_5min) < 200:
        logger.error('Need at least 200 5min bars. Run data backfill first.')
        return

    # Reduce grid for quick mode
    sg = {k: v[:2] for k,v in SCALP_GRID.items()}   if quick else SCALP_GRID
    wg = {k: v[:2] for k,v in SWING_GRID.items()}   if quick else SWING_GRID
    mg = {k: v[:2] for k,v in MEANREV_GRID.items()} if quick else MEANREV_GRID

    results = {}

    # Run each mode
    for mode, grid, entry_df in [
        ('scalp',   sg, df_5min),
        ('swing',   wg, df_15min if df_15min is not None else df_5min),
    ]:
        r = optimise_mode(mode, grid, entry_df, df_15min if df_15min is not None else df_5min,
                          df_1h, df_4h, df_1d, vix_data)
        if r:
            results[mode] = r

    # Summary
    logger.info('\n' + '='*60)
    logger.info('BACKTEST v2 COMPLETE')
    logger.info('='*60)

    summary = {}
    for mode, r in results.items():
        m = r['best']['metrics']
        p = r['best']['params']
        logger.info(f'\n{mode.upper()}:')
        logger.info(f"  Sharpe {m['sharpe']} | Return {m['total_return']}% | WR {m['win_rate']}%")
        logger.info(f"  Expectancy {m['expectancy']}R | DD {m['max_dd']}% | {m['trades_per_year']} trades/yr")
        summary[mode] = {'metrics': m, 'params': p}

    # Save
    output = {
        'symbol':    symbol,
        'timestamp': datetime.now().isoformat(),
        'results':   results,
        'summary':   summary,
    }
    fname = f'backtest_v2_results_{symbol}.json'
    with open(fname, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f'\n✅ Saved to {fname}')

    # Print config recommendations
    if summary:
        logger.info('\n' + '='*60)
        logger.info('RECOMMENDED CONFIG UPDATES:')
        logger.info('='*60)
        for mode, data in summary.items():
            p = data['params']
            m = data['metrics']
            logger.info(f'\n# {mode.upper()} (Sharpe: {m["sharpe"]})')
            for k, v in p.items():
                if k != 'mode':
                    logger.info(f'  {k}: {v}')

    return output


if __name__ == '__main__':
    import sys
    quick  = '--quick' in sys.argv
    symbol = next((a.split('=')[1] for a in sys.argv if a.startswith('--symbol=')), 'NQ')
    run_backtest_v2(symbol=symbol, quick=quick)
