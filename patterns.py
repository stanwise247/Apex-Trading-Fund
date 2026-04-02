"""
APEX Edge Engine — patterns.py
================================
Historical pattern miner. Tests every combination of conditions against
10 years of OHLCV data and calculates statistical edge for each pattern.

Tests:
  - Support & Resistance levels and retests
  - Fair Value Gaps (FVGs) — fill rate, rejection rate
  - Moving average structures and crossovers
  - RSI levels, divergences, regime filters
  - MACD crossovers and histogram signals
  - Volume confirmation and dry-up signals
  - Session timing effects (open, midday, power hour)
  - Multi-timeframe confluence
  - Market regime filters (VIX, trend, rate environment)
  - Overnight gap fills
  - Round number psychology

Run with: python3 patterns.py --symbol NQ --timeframe 1day
"""

import json
import db as _db
import argparse
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('APEX.patterns')

DB_PATH = 'apex_market.db'

# ─────────────────────────────────────────────
#  DATABASE HELPERS
# ─────────────────────────────────────────────

def load_ohlcv(symbol, timeframe, min_bars=100):
    conn = _db.connect()
    df = _db.read_sql(
        'SELECT ts, open, high, low, close, volume FROM ohlcv '
        'WHERE symbol=? AND timeframe=? ORDER BY ts ASC',
        conn, params=(symbol, timeframe)
    )
    conn.close()
    if df.empty or len(df) < min_bars:
        return None
    df['datetime'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    df.set_index('datetime', inplace=True)
    df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
    logger.info(f'Loaded {len(df)} {timeframe} bars for {symbol} '
                f'({df.index[0].strftime("%Y-%m-%d")} to {df.index[-1].strftime("%Y-%m-%d")})')
    return df


def save_pattern(pattern):
    conn = _db.connect()
    c = conn.cursor()
    # DELETE + INSERT works for both SQLite and PostgreSQL (INSERT OR REPLACE is SQLite-only)
    c.execute('DELETE FROM patterns WHERE name=?', (pattern['name'],))
    c.execute('''INSERT INTO patterns
        (name, symbol, conditions, occurrences, wins, losses,
         avg_rr, expectancy, best_regime, edge_score, last_updated, active)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''', (
        pattern['name'], pattern['symbol'],
        json.dumps(pattern.get('conditions', {})),
        pattern['occurrences'], pattern['wins'], pattern['losses'],
        pattern['avg_rr'], pattern['expectancy'],
        pattern.get('best_regime', ''),
        pattern['edge_score'],
        int(datetime.now().timestamp()), 1
    ))
    conn.commit()
    conn.close()


def save_all_patterns(patterns):
    # Clear existing patterns for this symbol first
    if patterns:
        sym = patterns[0]['symbol']
        conn = _db.connect()
        conn.execute('DELETE FROM patterns WHERE symbol=?', (sym,))
        conn.commit()
        conn.close()
    for p in patterns:
        save_pattern(p)
    logger.info(f'Saved {len(patterns)} patterns to database')


# ─────────────────────────────────────────────
#  INDICATOR CALCULATIONS
# ─────────────────────────────────────────────

def add_indicators(df):
    c = df['close'].values
    h = df['high'].values
    l = df['low'].values
    v = df['volume'].values

    # Moving averages
    df['ma9']   = df['close'].rolling(9).mean()
    df['ma20']  = df['close'].rolling(20).mean()
    df['ma50']  = df['close'].rolling(50).mean()
    df['ma200'] = df['close'].rolling(200).mean()
    df['ema9']  = df['close'].ewm(span=9, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()

    # RSI
    delta = df['close'].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=13, adjust=False).mean()
    avg_l = loss.ewm(com=13, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema12        = df['close'].ewm(span=12, adjust=False).mean()
    ema26        = df['close'].ewm(span=26, adjust=False).mean()
    df['macd']   = ema12 - ema26
    df['macd_sig']= df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_sig']

    # ATR
    tr = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': (df['high'] - df['close'].shift(1)).abs(),
        'lc': (df['low']  - df['close'].shift(1)).abs(),
    }).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'] / df['close'] * 100

    # Bollinger Bands
    df['bb_mid']   = df['close'].rolling(20).mean()
    bb_std         = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * bb_std
    df['bb_lower'] = df['bb_mid'] - 2 * bb_std
    df['bb_pct']   = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # Volume
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma20'].replace(0, np.nan)

    # VWAP (rolling 20-bar approximation)
    typical = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (typical * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()

    # Trend flags
    df['above_ma20']  = (df['close'] > df['ma20']).astype(int)
    df['above_ma50']  = (df['close'] > df['ma50']).astype(int)
    df['above_ma200'] = (df['close'] > df['ma200']).astype(int)
    df['golden_cross'] = ((df['ma20'] > df['ma50']) & (df['ma20'].shift(1) <= df['ma50'].shift(1))).astype(int)
    df['death_cross']  = ((df['ma20'] < df['ma50']) & (df['ma20'].shift(1) >= df['ma50'].shift(1))).astype(int)

    # Swing highs/lows (5-bar)
    df['swing_high'] = ((df['high'] > df['high'].shift(1)) &
                        (df['high'] > df['high'].shift(2)) &
                        (df['high'] > df['high'].shift(-1)) &
                        (df['high'] > df['high'].shift(-2))).astype(int)
    df['swing_low']  = ((df['low'] < df['low'].shift(1)) &
                        (df['low'] < df['low'].shift(2)) &
                        (df['low'] < df['low'].shift(-1)) &
                        (df['low'] < df['low'].shift(-2))).astype(int)

    # Session hour (UTC — NY open = 14:30 UTC)
    df['hour_utc'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek  # 0=Mon, 4=Fri

    # Prev day levels (shifted 1)
    df['prev_high']  = df['high'].shift(1)
    df['prev_low']   = df['low'].shift(1)
    df['prev_close'] = df['close'].shift(1)

    # Gap
    df['gap_pct'] = (df['open'] - df['prev_close']) / df['prev_close'] * 100

    # Candle body / wick
    df['body']      = (df['close'] - df['open']).abs()
    df['body_pct']  = df['body'] / df['close'] * 100
    df['upper_wick'] = df['high'] - df[['open','close']].max(axis=1)
    df['lower_wick'] = df[['open','close']].min(axis=1) - df['low']
    df['is_bull']   = (df['close'] > df['open']).astype(int)
    df['is_bear']   = (df['close'] < df['open']).astype(int)

    # Higher high / lower low
    df['hh'] = (df['high'] > df['high'].shift(1)).astype(int)
    df['ll'] = (df['low']  < df['low'].shift(1)).astype(int)

    return df


# ─────────────────────────────────────────────
#  FAIR VALUE GAPS
# ─────────────────────────────────────────────

def find_fvgs(df, min_gap_pct=0.1):
    """
    Find all Fair Value Gaps in the data.
    Bullish FVG: candle[i-1].low > candle[i+1].high — gap up
    Bearish FVG: candle[i-1].high < candle[i+1].low — gap down
    """
    fvgs = []
    highs  = df['high'].values
    lows   = df['low'].values
    closes = df['close'].values
    idx    = df.index

    for i in range(1, len(df) - 1):
        price = closes[i]
        # Bullish FVG — gap between candle i-1 bottom and candle i+1 top
        if lows[i-1] > highs[i+1]:
            gap_size = lows[i-1] - highs[i+1]
            gap_pct  = gap_size / price * 100
            if gap_pct >= min_gap_pct:
                fvgs.append({
                    'type':       'bullish',
                    'bar_index':  i,
                    'datetime':   idx[i],
                    'top':        lows[i-1],
                    'bottom':     highs[i+1],
                    'midpoint':   (lows[i-1] + highs[i+1]) / 2,
                    'gap_size':   gap_size,
                    'gap_pct':    gap_pct,
                    'price_at_formation': price,
                })
        # Bearish FVG — gap between candle i-1 top and candle i+1 bottom
        if highs[i-1] < lows[i+1]:
            gap_size = lows[i+1] - highs[i-1]
            gap_pct  = gap_size / price * 100
            if gap_pct >= min_gap_pct:
                fvgs.append({
                    'type':       'bearish',
                    'bar_index':  i,
                    'datetime':   idx[i],
                    'top':        lows[i+1],
                    'bottom':     highs[i-1],
                    'midpoint':   (lows[i+1] + highs[i-1]) / 2,
                    'gap_size':   gap_size,
                    'gap_pct':    gap_pct,
                    'price_at_formation': price,
                })
    return fvgs


def analyse_fvg_fills(df, fvgs, max_bars_to_fill=20):
    """
    For each FVG, look forward to see if price fills it and what happens after.
    Returns stats on fill rate, fill timeframe, and post-fill price action.
    """
    closes = df['close'].values
    highs  = df['high'].values
    lows   = df['low'].values

    results = []
    for fvg in fvgs:
        i = fvg['bar_index']
        if i + max_bars_to_fill >= len(df):
            continue

        filled      = False
        bars_to_fill = None
        post_fill_return = None

        for j in range(i+1, min(i+max_bars_to_fill+1, len(df))):
            if fvg['type'] == 'bullish':
                # Price fills bullish FVG if it drops into the gap
                if lows[j] <= fvg['top'] and highs[j] >= fvg['bottom']:
                    filled = True
                    bars_to_fill = j - i
                    # What happens 5 bars after fill?
                    if j + 5 < len(df):
                        post_fill_return = (closes[j+5] - closes[j]) / closes[j] * 100
                    break
            else:
                # Price fills bearish FVG if it rallies into the gap
                if highs[j] >= fvg['bottom'] and lows[j] <= fvg['top']:
                    filled = True
                    bars_to_fill = j - i
                    if j + 5 < len(df):
                        post_fill_return = (closes[j+5] - closes[j]) / closes[j] * 100
                    break

        results.append({
            **fvg,
            'filled':           filled,
            'bars_to_fill':     bars_to_fill,
            'post_fill_return': post_fill_return,
        })
    return results


# ─────────────────────────────────────────────
#  SUPPORT & RESISTANCE DETECTION
# ─────────────────────────────────────────────

def find_sr_levels(df, lookback=100, tolerance_pct=0.3):
    """
    Find significant S/R levels by clustering swing highs and lows.
    Levels with more touches are stronger.
    """
    h = df['high'].values[-lookback:]
    l = df['low'].values[-lookback:]
    c = df['close'].values[-lookback:]

    # Collect all swing points
    candidates = []
    for i in range(2, len(h)-2):
        if h[i] > h[i-1] and h[i] > h[i-2] and h[i] > h[i+1] and h[i] > h[i+2]:
            candidates.append(h[i])
        if l[i] < l[i-1] and l[i] < l[i-2] and l[i] < l[i+1] and l[i] < l[i+2]:
            candidates.append(l[i])

    if not candidates:
        return []

    # Cluster nearby levels
    price_ref = c[-1]
    tol = price_ref * tolerance_pct / 100
    levels = []
    used   = set()

    for i, p1 in enumerate(candidates):
        if i in used:
            continue
        cluster = [p1]
        for j, p2 in enumerate(candidates[i+1:], i+1):
            if j not in used and abs(p1 - p2) < tol:
                cluster.append(p2)
                used.add(j)
        used.add(i)
        level_price = np.mean(cluster)
        touches     = len(cluster)
        dist_pct    = (level_price - price_ref) / price_ref * 100
        level_type  = 'resistance' if level_price > price_ref else 'support'
        levels.append({
            'price':     round(level_price, 2),
            'touches':   touches,
            'type':      level_type,
            'dist_pct':  round(dist_pct, 2),
            'strength':  min(touches * 20, 100),
        })

    # Sort by number of touches
    levels.sort(key=lambda x: x['touches'], reverse=True)
    return levels[:10]


def find_round_numbers(price, range_pct=5):
    """Find psychologically significant round number levels near current price"""
    if price > 10000:
        step = 500
    elif price > 1000:
        step = 100
    elif price > 100:
        step = 10
    else:
        step = 1

    low  = price * (1 - range_pct/100)
    high = price * (1 + range_pct/100)
    levels = []
    n = int(low // step)
    while n * step <= high:
        lp = n * step
        dist = (lp - price) / price * 100
        levels.append({
            'price':   lp,
            'type':    'resistance' if lp > price else 'support',
            'dist_pct': round(dist, 2),
            'touches':  3,
            'strength': 60,
            'note':    'Round number',
        })
        n += 1
    return levels


# ─────────────────────────────────────────────
#  BACKTESTING ENGINE
# ─────────────────────────────────────────────

def backtest_pattern(df, signal_mask, direction='long',
                     atr_stop_mult=1.5, atr_target_mult=3.0,
                     hold_bars=10, label='pattern'):
    """
    Core backtesting function.

    signal_mask: boolean Series, True where pattern fires
    direction:   'long' or 'short'
    atr_stop_mult:   stop = entry +/- ATR * mult
    atr_target_mult: target = entry +/- ATR * mult
    hold_bars:   max bars to hold if neither stop nor target hit

    Returns dict with full statistics.
    """
    trades = []
    signals = df[signal_mask].index

    for sig_dt in signals:
        try:
            i = df.index.get_loc(sig_dt)
        except KeyError:
            continue

        if i + 1 >= len(df) or i + hold_bars >= len(df):
            continue

        entry_bar = df.iloc[i+1]
        entry     = float(entry_bar['open'])
        atr       = float(df.iloc[i]['atr']) if not np.isnan(df.iloc[i]['atr']) else entry * 0.01

        if direction == 'long':
            stop   = entry - atr * atr_stop_mult
            target = entry + atr * atr_target_mult
        else:
            stop   = entry + atr * atr_stop_mult
            target = entry - atr * atr_target_mult

        risk   = abs(entry - stop)
        reward = abs(target - entry)
        rr     = reward / risk if risk > 0 else 0

        outcome    = 'timeout'
        exit_price = float(df.iloc[min(i+1+hold_bars, len(df)-1)]['close'])
        exit_bar   = i + hold_bars

        for j in range(i+1, min(i+1+hold_bars, len(df))):
            bar_h = float(df.iloc[j]['high'])
            bar_l = float(df.iloc[j]['low'])
            if direction == 'long':
                if bar_l <= stop:
                    outcome    = 'loss'
                    exit_price = stop
                    exit_bar   = j
                    break
                if bar_h >= target:
                    outcome    = 'win'
                    exit_price = target
                    exit_bar   = j
                    break
            else:
                if bar_h >= stop:
                    outcome    = 'loss'
                    exit_price = stop
                    exit_bar   = j
                    break
                if bar_l <= target:
                    outcome    = 'win'
                    exit_price = target
                    exit_bar   = j
                    break

        if direction == 'long':
            pnl_pts = exit_price - entry
        else:
            pnl_pts = entry - exit_price

        pnl_r = pnl_pts / risk if risk > 0 else 0

        trades.append({
            'entry_dt':   sig_dt.isoformat(),
            'exit_dt':    df.index[exit_bar].isoformat(),
            'direction':  direction,
            'entry':      round(entry, 2),
            'stop':       round(stop, 2),
            'target':     round(target, 2),
            'exit':       round(exit_price, 2),
            'outcome':    outcome,
            'pnl_pts':    round(pnl_pts, 2),
            'pnl_r':      round(pnl_r, 3),
            'rr':         round(rr, 2),
            'hold_bars':  exit_bar - i,
            'atr':        round(atr, 2),
        })

    if not trades:
        return None

    wins    = [t for t in trades if t['outcome'] == 'win']
    losses  = [t for t in trades if t['outcome'] == 'loss']
    timeouts= [t for t in trades if t['outcome'] == 'timeout']

    win_rate   = len(wins) / len(trades) if trades else 0
    avg_win_r  = np.mean([t['pnl_r'] for t in wins])    if wins    else 0
    avg_loss_r = np.mean([t['pnl_r'] for t in losses])  if losses  else 0
    avg_to_r   = np.mean([t['pnl_r'] for t in timeouts])if timeouts else 0

    expectancy = (win_rate * avg_win_r) + ((1 - win_rate) * avg_loss_r)
    avg_rr     = np.mean([t['rr'] for t in trades]) if trades else 0

    # Edge score: combination of expectancy, win rate, sample size confidence
    sample_conf = min(len(trades) / 50, 1.0)  # full confidence at 50+ trades
    edge_score  = max(0, expectancy * 50 + win_rate * 30 + sample_conf * 20)
    edge_score  = min(edge_score, 100)

    return {
        'label':        label,
        'direction':    direction,
        'occurrences':  len(trades),
        'wins':         len(wins),
        'losses':       len(losses),
        'timeouts':     len(timeouts),
        'win_rate':     round(win_rate * 100, 1),
        'avg_win_r':    round(avg_win_r, 3),
        'avg_loss_r':   round(avg_loss_r, 3),
        'avg_rr':       round(avg_rr, 2),
        'expectancy':   round(expectancy, 3),
        'edge_score':   round(edge_score, 1),
        'trades':       trades,
    }


# ─────────────────────────────────────────────
#  PATTERN DEFINITIONS
# ─────────────────────────────────────────────

def run_all_patterns(df, symbol):
    """Run every pattern test and return results"""
    results = []

    logger.info('Running pattern tests...')

    # ── 1. MA TREND + RSI PULLBACK ──────────────────────────
    # Long: above MA50, MA50 > MA200 (uptrend), RSI pulls back to 40-50
    mask = (
        (df['above_ma50'] == 1) &
        (df['ma50'] > df['ma200']) &
        (df['rsi'] >= 38) & (df['rsi'] <= 52) &
        (df['rsi'].shift(1) > df['rsi'])  # RSI declining into the zone
    )
    r = backtest_pattern(df, mask, 'long', label='MA_Uptrend_RSI_Pullback_Long')
    if r and r['occurrences'] >= 10:
        results.append(r)

    # Short: below MA50, MA50 < MA200 (downtrend), RSI bounces to 50-62
    mask = (
        (df['above_ma50'] == 0) &
        (df['ma50'] < df['ma200']) &
        (df['rsi'] >= 50) & (df['rsi'] <= 62) &
        (df['rsi'].shift(1) < df['rsi'])  # RSI rising into resistance zone
    )
    r = backtest_pattern(df, mask, 'short', label='MA_Downtrend_RSI_Bounce_Short')
    if r and r['occurrences'] >= 10:
        results.append(r)

    # ── 2. MA200 SUPPORT / RESISTANCE ───────────────────────
    # Price touches MA200 from above in uptrend
    ma200_touch_long = (
        (df['low'] <= df['ma200'] * 1.002) &
        (df['close'] > df['ma200']) &
        (df['above_ma200'].shift(5) == 1) &
        (df['vol_ratio'] > 0.8)
    )
    r = backtest_pattern(df, ma200_touch_long, 'long', label='MA200_Support_Touch_Long')
    if r and r['occurrences'] >= 5:
        results.append(r)

    # Price touches MA200 from below in downtrend — rejection short
    ma200_reject_short = (
        (df['high'] >= df['ma200'] * 0.998) &
        (df['close'] < df['ma200']) &
        (df['above_ma200'].shift(5) == 0)
    )
    r = backtest_pattern(df, ma200_reject_short, 'short', label='MA200_Resistance_Rejection_Short')
    if r and r['occurrences'] >= 5:
        results.append(r)

    # ── 3. GOLDEN / DEATH CROSS ─────────────────────────────
    r = backtest_pattern(df, df['golden_cross'] == 1, 'long',
                         hold_bars=20, label='Golden_Cross_Long')
    if r and r['occurrences'] >= 5:
        results.append(r)

    r = backtest_pattern(df, df['death_cross'] == 1, 'short',
                         hold_bars=20, label='Death_Cross_Short')
    if r and r['occurrences'] >= 5:
        results.append(r)

    # ── 4. RSI OVERSOLD / OVERBOUGHT ────────────────────────
    # RSI crosses back above 30 (oversold recovery)
    rsi_oversold = (
        (df['rsi'] > 30) &
        (df['rsi'].shift(1) <= 30) &
        (df['above_ma50'] == 1)  # only in uptrend context
    )
    r = backtest_pattern(df, rsi_oversold, 'long', label='RSI_Oversold_Recovery_Uptrend')
    if r and r['occurrences'] >= 5:
        results.append(r)

    rsi_overbought = (
        (df['rsi'] < 70) &
        (df['rsi'].shift(1) >= 70) &
        (df['above_ma50'] == 0)  # only in downtrend context
    )
    r = backtest_pattern(df, rsi_overbought, 'short', label='RSI_Overbought_Rejection_Downtrend')
    if r and r['occurrences'] >= 5:
        results.append(r)

    # ── 5. RSI DIVERGENCE ───────────────────────────────────
    # Bearish divergence: price makes HH but RSI makes LH (5-bar)
    price_hh  = df['close'] > df['close'].rolling(5).max().shift(1)
    rsi_lh    = df['rsi']   < df['rsi'].rolling(5).max().shift(1)
    bear_div  = price_hh & rsi_lh & (df['rsi'] > 60)
    r = backtest_pattern(df, bear_div, 'short', label='Bearish_RSI_Divergence')
    if r and r['occurrences'] >= 5:
        results.append(r)

    # Bullish divergence: price makes LL but RSI makes HL
    price_ll  = df['close'] < df['close'].rolling(5).min().shift(1)
    rsi_hl    = df['rsi']   > df['rsi'].rolling(5).min().shift(1)
    bull_div  = price_ll & rsi_hl & (df['rsi'] < 40)
    r = backtest_pattern(df, bull_div, 'long', label='Bullish_RSI_Divergence')
    if r and r['occurrences'] >= 5:
        results.append(r)

    # ── 6. MACD CROSSOVERS ──────────────────────────────────
    macd_bull_cross = (
        (df['macd'] > df['macd_sig']) &
        (df['macd'].shift(1) <= df['macd_sig'].shift(1)) &
        (df['macd'] < 0)  # crossing in negative territory = stronger signal
    )
    r = backtest_pattern(df, macd_bull_cross, 'long', label='MACD_Bull_Cross_Below_Zero')
    if r and r['occurrences'] >= 5:
        results.append(r)

    macd_bear_cross = (
        (df['macd'] < df['macd_sig']) &
        (df['macd'].shift(1) >= df['macd_sig'].shift(1)) &
        (df['macd'] > 0)  # crossing in positive territory
    )
    r = backtest_pattern(df, macd_bear_cross, 'short', label='MACD_Bear_Cross_Above_Zero')
    if r and r['occurrences'] >= 5:
        results.append(r)

    # ── 7. BOLLINGER BAND PATTERNS ──────────────────────────
    # BB squeeze breakout: band width narrows then expands with direction
    bb_width        = df['bb_upper'] - df['bb_lower']
    bb_width_ma     = bb_width.rolling(20).mean()
    bb_squeeze      = bb_width < bb_width_ma * 0.7
    bb_bull_breakout= bb_squeeze.shift(1) & (df['close'] > df['bb_upper'])
    r = backtest_pattern(df, bb_bull_breakout, 'long', label='Bollinger_Squeeze_Bull_Breakout')
    if r and r['occurrences'] >= 5:
        results.append(r)

    # BB lower band walk (oversold bounce)
    bb_lower_touch = (df['low'] <= df['bb_lower']) & (df['close'] > df['open'])
    r = backtest_pattern(df, bb_lower_touch, 'long', label='Bollinger_Lower_Band_Bounce')
    if r and r['occurrences'] >= 5:
        results.append(r)

    # ── 8. VOLUME CONFIRMATION ──────────────────────────────
    # High volume breakout above prev 10-bar high
    prev_high_10     = df['high'].rolling(10).max().shift(1)
    vol_breakout_long= (df['close'] > prev_high_10) & (df['vol_ratio'] > 1.5)
    r = backtest_pattern(df, vol_breakout_long, 'long', label='High_Volume_Breakout_Long')
    if r and r['occurrences'] >= 5:
        results.append(r)

    # Low volume at support (consolidation before bounce)
    at_support  = (df['low'] <= df['ma20'] * 1.002) & (df['close'] > df['ma20'] * 0.998)
    low_vol_sup = at_support & (df['vol_ratio'] < 0.6)
    r = backtest_pattern(df, low_vol_sup, 'long', label='Low_Volume_Support_Hold_Long')
    if r and r['occurrences'] >= 5:
        results.append(r)

    # High volume reversal candle (climactic selling into support)
    vol_reversal = (
        (df['vol_ratio'] > 2.0) &
        (df['is_bull'] == 1) &
        (df['body_pct'] > 0.5) &
        (df['rsi'] < 40)
    )
    r = backtest_pattern(df, vol_reversal, 'long', label='High_Volume_Bullish_Reversal')
    if r and r['occurrences'] >= 5:
        results.append(r)

    # ── 9. PREV DAY HIGH/LOW BREAKS ─────────────────────────
    pdh_break = (df['close'] > df['prev_high']) & (df['open'] < df['prev_high'])
    r = backtest_pattern(df, pdh_break, 'long', label='Prev_Day_High_Break_Long')
    if r and r['occurrences'] >= 10:
        results.append(r)

    pdl_break = (df['close'] < df['prev_low']) & (df['open'] > df['prev_low'])
    r = backtest_pattern(df, pdl_break, 'short', label='Prev_Day_Low_Break_Short')
    if r and r['occurrences'] >= 10:
        results.append(r)

    # PDH retest after break (pull back to test broken resistance as support)
    pdh_retest = (
        (df['close'].shift(1) > df['prev_high'].shift(1)) &  # broke yesterday
        (df['low'] <= df['prev_high'] * 1.003) &              # retesting today
        (df['close'] > df['prev_high'])                        # holds above
    )
    r = backtest_pattern(df, pdh_retest, 'long', label='Prev_Day_High_Retest_Long')
    if r and r['occurrences'] >= 5:
        results.append(r)

    # ── 10. GAP ANALYSIS ────────────────────────────────────
    # Gap up and hold (bull continuation)
    gap_up_hold = (df['gap_pct'] > 0.3) & (df['close'] > df['open'])
    r = backtest_pattern(df, gap_up_hold, 'long', label='Gap_Up_Hold_Long')
    if r and r['occurrences'] >= 5:
        results.append(r)

    # Gap up and fade (mean reversion short)
    gap_up_fade = (df['gap_pct'] > 0.3) & (df['close'] < df['open']) & (df['rsi'] > 65)
    r = backtest_pattern(df, gap_up_fade, 'short', label='Gap_Up_Fade_Short')
    if r and r['occurrences'] >= 5:
        results.append(r)

    # Gap down and recover (mean reversion long)
    gap_down_recover = (df['gap_pct'] < -0.3) & (df['close'] > df['open']) & (df['rsi'] < 40)
    r = backtest_pattern(df, gap_down_recover, 'long', label='Gap_Down_Recovery_Long')
    if r and r['occurrences'] >= 5:
        results.append(r)

    # ── 11. MULTI-TIMEFRAME CONFLUENCE ──────────────────────
    # Strong uptrend on daily: price > MA20 > MA50 > MA200
    strong_uptrend = (
        (df['close'] > df['ma20']) &
        (df['ma20'] > df['ma50']) &
        (df['ma50'] > df['ma200']) &
        (df['rsi'] > 50) & (df['rsi'] < 70) &
        (df['macd'] > df['macd_sig'])
    )
    r = backtest_pattern(df, strong_uptrend, 'long', hold_bars=5,
                         label='Full_Uptrend_Confluence_Long')
    if r and r['occurrences'] >= 10:
        results.append(r)

    # Full downtrend alignment
    strong_downtrend = (
        (df['close'] < df['ma20']) &
        (df['ma20'] < df['ma50']) &
        (df['ma50'] < df['ma200']) &
        (df['rsi'] < 50) & (df['rsi'] > 30) &
        (df['macd'] < df['macd_sig'])
    )
    r = backtest_pattern(df, strong_downtrend, 'short', hold_bars=5,
                         label='Full_Downtrend_Confluence_Short')
    if r and r['occurrences'] >= 10:
        results.append(r)

    # ── 12. ENGULFING CANDLES ───────────────────────────────
    # Bullish engulfing at support
    bull_engulf = (
        (df['is_bear'].shift(1) == 1) &
        (df['is_bull'] == 1) &
        (df['open'] <= df['close'].shift(1)) &
        (df['close'] >= df['open'].shift(1)) &
        (df['rsi'] < 50)
    )
    r = backtest_pattern(df, bull_engulf, 'long', label='Bullish_Engulfing_Candle')
    if r and r['occurrences'] >= 5:
        results.append(r)

    # Bearish engulfing at resistance
    bear_engulf = (
        (df['is_bull'].shift(1) == 1) &
        (df['is_bear'] == 1) &
        (df['open'] >= df['close'].shift(1)) &
        (df['close'] <= df['open'].shift(1)) &
        (df['rsi'] > 50)
    )
    r = backtest_pattern(df, bear_engulf, 'short', label='Bearish_Engulfing_Candle')
    if r and r['occurrences'] >= 5:
        results.append(r)

    # ── 13. MONDAY REVERSAL ─────────────────────────────────
    # NQ reverses previous week's direction on Monday
    prev_week_up = df['close'].shift(1) > df['close'].shift(6)
    monday_rev_short = (
        (df['day_of_week'] == 0) &
        prev_week_up &
        (df['rsi'] > 60) &
        (df['gap_pct'] > 0.2)
    )
    r = backtest_pattern(df, monday_rev_short, 'short', hold_bars=3,
                         label='Monday_Reversal_After_Up_Week_Short')
    if r and r['occurrences'] >= 5:
        results.append(r)

    # ── 14. VWAP RELATIONSHIP ───────────────────────────────
    vwap_reclaim = (
        (df['close'] > df['vwap']) &
        (df['close'].shift(1) < df['vwap'].shift(1)) &
        (df['vol_ratio'] > 1.2)
    )
    r = backtest_pattern(df, vwap_reclaim, 'long', hold_bars=5, label='VWAP_Reclaim_Long')
    if r and r['occurrences'] >= 5:
        results.append(r)

    vwap_reject = (
        (df['close'] < df['vwap']) &
        (df['close'].shift(1) > df['vwap'].shift(1)) &
        (df['vol_ratio'] > 1.2)
    )
    r = backtest_pattern(df, vwap_reject, 'short', hold_bars=5, label='VWAP_Rejection_Short')
    if r and r['occurrences'] >= 5:
        results.append(r)

    # ── 15. HIGH VOLATILITY REGIME PATTERNS ─────────────────
    high_atr = df['atr_pct'] > df['atr_pct'].rolling(50).mean() * 1.5
    # In high vol — fade extreme RSI moves
    high_vol_short = high_atr & (df['rsi'] > 72) & (df['vol_ratio'] > 1.5)
    r = backtest_pattern(df, high_vol_short, 'short', label='High_Vol_RSI_Exhaustion_Short')
    if r and r['occurrences'] >= 5:
        results.append(r)

    high_vol_long = high_atr & (df['rsi'] < 28) & (df['vol_ratio'] > 1.5)
    r = backtest_pattern(df, high_vol_long, 'long', label='High_Vol_RSI_Capitulation_Long')
    if r and r['occurrences'] >= 5:
        results.append(r)

    # Sort by edge score
    results.sort(key=lambda x: x['edge_score'], reverse=True)
    logger.info(f'Found {len(results)} patterns with sufficient sample size')
    return results


# ─────────────────────────────────────────────
#  FVG PATTERN ANALYSIS
# ─────────────────────────────────────────────

def analyse_fvg_patterns(df, symbol):
    """Find all FVGs and compute their statistical edge"""
    logger.info('Analysing Fair Value Gaps...')
    fvgs    = find_fvgs(df, min_gap_pct=0.15)
    results = analyse_fvg_fills(df, fvgs)

    if not results:
        return {}

    bull_fvgs = [r for r in results if r['type'] == 'bullish']
    bear_fvgs = [r for r in results if r['type'] == 'bearish']

    def fvg_stats(fvg_list, label):
        if not fvg_list:
            return {}
        filled    = [f for f in fvg_list if f['filled']]
        unfilled  = [f for f in fvg_list if not f['filled']]
        fill_rate = len(filled) / len(fvg_list) * 100

        avg_bars  = np.mean([f['bars_to_fill'] for f in filled]) if filled else None
        post_rets = [f['post_fill_return'] for f in filled if f['post_fill_return'] is not None]
        avg_post  = np.mean(post_rets) if post_rets else None

        return {
            'label':        label,
            'total_fvgs':   len(fvg_list),
            'filled':       len(filled),
            'unfilled':     len(unfilled),
            'fill_rate_pct':round(fill_rate, 1),
            'avg_bars_to_fill': round(avg_bars, 1) if avg_bars else None,
            'avg_post_fill_return_pct': round(avg_post, 3) if avg_post else None,
            'avg_gap_size_pct': round(np.mean([f['gap_pct'] for f in fvg_list]), 3),
        }

    bull_stats = fvg_stats(bull_fvgs, 'Bullish_FVG')
    bear_stats = fvg_stats(bear_fvgs, 'Bearish_FVG')

    logger.info(f'FVGs found: {len(bull_fvgs)} bullish, {len(bear_fvgs)} bearish')
    logger.info(f'Bull FVG fill rate: {bull_stats.get("fill_rate_pct")}% | '
                f'Bear FVG fill rate: {bear_stats.get("fill_rate_pct")}%')

    return {'bullish': bull_stats, 'bearish': bear_stats, 'all_fvgs': results}


# ─────────────────────────────────────────────
#  REGIME BREAKDOWN
# ─────────────────────────────────────────────

def classify_bar_regime(df):
    """Add regime label to each bar based on rolling indicators"""
    df = df.copy()

    # Rolling ATR percentile as vol proxy
    df['vol_pct'] = df['atr_pct'].rolling(50).rank(pct=True)

    def regime_label(row):
        if pd.isna(row['ma50']) or pd.isna(row['vol_pct']):
            return 'unknown'
        trending   = abs(row['close'] - row['ma50']) / row['ma50'] > 0.03
        high_vol   = row['vol_pct'] > 0.7
        bull_trend = row['close'] > row['ma50']
        if trending and bull_trend and not high_vol:
            return 'bull_trending'
        if trending and not bull_trend and not high_vol:
            return 'bear_trending'
        if trending and high_vol and bull_trend:
            return 'bull_volatile'
        if trending and high_vol and not bull_trend:
            return 'bear_volatile'
        if not trending and not high_vol:
            return 'low_vol_range'
        return 'high_vol_range'

    df['regime'] = df.apply(regime_label, axis=1)
    return df


def breakdown_by_regime(pattern_results, df):
    """Split each pattern's performance by market regime"""
    if 'regime' not in df.columns:
        df = classify_bar_regime(df)

    for pattern in pattern_results:
        regime_stats = {}
        trades = pattern.get('trades', [])
        for trade in trades:
            try:
                entry_dt = pd.Timestamp(trade['entry_dt'])
                if entry_dt.tzinfo is None:
                    entry_dt = entry_dt.tz_localize('UTC')
                regime = df.loc[entry_dt, 'regime'] if entry_dt in df.index else 'unknown'
            except Exception:
                regime = 'unknown'
            if regime not in regime_stats:
                regime_stats[regime] = {'wins': 0, 'losses': 0, 'total': 0, 'pnl_r': []}
            regime_stats[regime]['total'] += 1
            regime_stats[regime]['pnl_r'].append(trade['pnl_r'])
            if trade['outcome'] == 'win':
                regime_stats[regime]['wins'] += 1
            elif trade['outcome'] == 'loss':
                regime_stats[regime]['losses'] += 1

        regime_summary = {}
        best_regime, best_exp = None, -999
        for reg, stats in regime_stats.items():
            if stats['total'] < 3:
                continue
            wr  = stats['wins'] / stats['total']
            avg = np.mean(stats['pnl_r'])
            regime_summary[reg] = {
                'occurrences': stats['total'],
                'win_rate':    round(wr * 100, 1),
                'expectancy':  round(avg, 3),
            }
            if avg > best_exp:
                best_exp, best_regime = avg, reg

        pattern['regime_breakdown'] = regime_summary
        pattern['best_regime']      = best_regime or 'all'

    return pattern_results


# ─────────────────────────────────────────────
#  MAIN RUNNER
# ─────────────────────────────────────────────

def run_pattern_engine(symbol='NQ', timeframe='1day'):
    logger.info('=' * 55)
    logger.info(f'  APEX Pattern Engine — {symbol} {timeframe}')
    logger.info('=' * 55)

    df = load_ohlcv(symbol, timeframe)
    if df is None:
        logger.error(f'No data found for {symbol} {timeframe}. Run backfill first.')
        return

    df = add_indicators(df)
    df = classify_bar_regime(df)

    # Run all pattern tests
    patterns   = run_all_patterns(df, symbol)
    patterns   = breakdown_by_regime(patterns, df)

    # Run FVG analysis
    fvg_stats  = analyse_fvg_patterns(df, symbol)

    # Find S/R levels
    sr_levels  = find_sr_levels(df)
    round_nums = find_round_numbers(float(df['close'].iloc[-1]))

    # Print summary
    logger.info('\n' + '='*55)
    logger.info(f'  PATTERN RESULTS — {symbol} {timeframe}')
    logger.info('='*55)
    logger.info(f'  {"Pattern":<45} {"WR%":>6} {"ExpR":>7} {"Score":>7} {"N":>5}')
    logger.info('  ' + '-'*55)
    for p in patterns:
        logger.info(f'  {p["label"]:<45} {p["win_rate"]:>5.1f}% {p["expectancy"]:>7.3f} {p["edge_score"]:>7.1f} {p["occurrences"]:>5}')

    if fvg_stats:
        b = fvg_stats.get('bullish', {})
        bear = fvg_stats.get('bearish', {})
        logger.info(f'\n  FVG Analysis:')
        logger.info(f'  Bullish FVGs: {b.get("total_fvgs",0)} total, {b.get("fill_rate_pct",0)}% fill rate, avg {b.get("avg_bars_to_fill","?")} bars')
        logger.info(f'  Bearish FVGs: {bear.get("total_fvgs",0)} total, {bear.get("fill_rate_pct",0)}% fill rate, avg {bear.get("avg_bars_to_fill","?")} bars')

    # Save to database
    db_patterns = []
    for p in patterns:
        db_patterns.append({
            'name':        p['label'],
            'symbol':      symbol,
            'conditions':  {'timeframe': timeframe, 'direction': p['direction']},
            'occurrences': p['occurrences'],
            'wins':        p['wins'],
            'losses':      p['losses'],
            'avg_rr':      p['avg_rr'],
            'expectancy':  p['expectancy'],
            'best_regime': p.get('best_regime', 'all'),
            'edge_score':  p['edge_score'],
        })
    save_all_patterns(db_patterns)

    # Save full results to JSON for the dashboard and backtest engine
    output = {
        'symbol':    symbol,
        'timeframe': timeframe,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'bars_analysed': len(df),
        'date_range': {
            'from': df.index[0].isoformat(),
            'to':   df.index[-1].isoformat(),
        },
        'patterns':  patterns,
        'fvg_stats': fvg_stats,
        'sr_levels': sr_levels,
        'round_numbers': round_nums,
    }

    outfile = f'edge_results_{symbol}_{timeframe}.json'
    with open(outfile, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f'\n  Full results saved to: {outfile}')
    logger.info(f'  Patterns saved to database: {len(db_patterns)}')
    logger.info('='*55)

    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='APEX Pattern Engine')
    parser.add_argument('--symbol',    default='NQ',   help='Symbol to analyse (default: NQ)')
    parser.add_argument('--timeframe', default='1day', help='Timeframe (default: 1day)')
    parser.add_argument('--all',       action='store_true', help='Run all timeframes')
    args = parser.parse_args()

    if args.all:
        for tf in ['1week', '1day', '1hour']:
            run_pattern_engine(args.symbol, tf)
    else:
        run_pattern_engine(args.symbol, args.timeframe)
