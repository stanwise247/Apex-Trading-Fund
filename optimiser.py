"""
APEX Parameter Optimisation Engine — optimiser.py
===================================================
Systematically tests every variable combination to find
the exact optimal strategy parameters.

Tests:
  - Score thresholds (45, 50, 55, 60, 65, 70)
  - R:R ratios (1.5, 2.0, 2.5, 3.0, 4.0)
  - Risk % per trade (1, 1.5, 2, 2.5, 3)
  - Entry timeframes (5min, 15min, 1hour, combined)
  - Session windows (all, prime only, no midday, power hour)
  - Day of week filters (all days, weekdays, Tue-Thu only)
  - VIX regime filters (all, <25, <20)
  - Layer combinations (which layers add the most edge)
  - HTF confluence requirements (strict vs flexible)

Output:
  - Best parameter combination by Sharpe ratio
  - Best combination by win rate
  - Best combination by max drawdown
  - Layer-by-layer edge contribution
  - Full results saved to optimiser_results.json

Run: python3 optimiser.py --symbol NQ --balance 10000
"""

import sqlite3
import json
import logging
import itertools
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('APEX.Optimiser')

DB_PATH = 'apex_market.db'

# =============================================================
#  PARAMETER GRID — everything we want to test
# =============================================================

PARAM_GRID = {
    'min_score':        [45, 50, 55, 60, 65],
    'rr_ratio':         [2.0, 2.5, 3.0, 4.0],
    'risk_pct':         [1.0, 1.5, 2.0, 2.5, 3.0],
    'entry_tfs':        [['15min'], ['5min'], ['1hour'], ['15min','5min'], ['15min','1hour']],
    'session_filter':   ['all', 'prime_only', 'no_midday', 'power_hour_too', 'london', 'london_ny_open', 'all_sessions'],
    'dow_filter':       ['all', 'mon_fri_out', 'tue_thu_only'],
    'vix_filter':       ['all', 'below_25', 'below_20'],
    'htf_strict':       [True, False],
}

SESSION_RULES = {
    'all':              lambda h, m: True,
    'prime_only':       lambda h, m: (9 < h < 12) or (h == 9 and m >= 45),
    'no_midday':        lambda h, m: not ((h == 11 and m >= 30) or h == 12 or (h == 13 and m < 30)),
    'power_hour_too':   lambda h, m: (9 < h < 12) or (h == 9 and m >= 45) or (15 <= h < 16),
    'london':           lambda h, m: 3 <= h < 8,
    'london_ny_open':   lambda h, m: (3 <= h < 8) or (h == 9 and m >= 30) or (9 < h < 12),
    'london_only':      lambda h, m: 3 <= h < 8,
    'all_sessions':     lambda h, m: (3 <= h < 8) or (h == 9 and m >= 30) or (9 < h < 16),
}

DOW_RULES = {
    'all':          lambda d: True,
    'mon_fri_out':  lambda d: d not in (0, 4),
    'tue_thu_only': lambda d: d in (1, 3),
}

VIX_RULES = {
    'all':      lambda v: True,
    'below_25': lambda v: v is None or v < 25,
    'below_20': lambda v: v is None or v < 20,
}


# =============================================================
#  DATA LOADING
# =============================================================

def load_tf(symbol, timeframe, limit=5000):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        'SELECT ts,open,high,low,close,volume FROM ohlcv '
        'WHERE symbol=? AND timeframe=? ORDER BY ts DESC LIMIT ?',
        conn, params=(symbol, timeframe, limit)
    )
    conn.close()
    if df.empty or len(df) < 50:
        return None
    df = df.iloc[::-1].reset_index(drop=True)
    df['dt'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    df.set_index('dt', inplace=True)
    for col in ['open','high','low','close','volume']:
        df[col] = df[col].astype(float)
    return df


def add_indicators(df):
    if df is None or len(df) < 10:
        return df
    c = df['close']; h = df['high']; l = df['low']; v = df['volume']
    for p in [9,20,50,200]: df[f'ma{p}'] = c.rolling(p).mean()
    df['ema9']  = c.ewm(span=9,  adjust=False).mean()
    df['ema21'] = c.ewm(span=21, adjust=False).mean()
    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(com=13, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    df['macd']      = e12 - e26
    df['macd_sig']  = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_sig']
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    df['atr']     = tr.ewm(span=14, adjust=False).mean()
    df['bb_mid']  = c.rolling(20).mean()
    std = c.rolling(20).std()
    df['bb_upper']= df['bb_mid'] + 2*std
    df['bb_lower']= df['bb_mid'] - 2*std
    df['bb_pct']  = (c - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['vol_ma20']  = v.rolling(20).mean()
    df['vol_ratio'] = v / df['vol_ma20'].replace(0, np.nan)
    df['is_bull'] = (c > df['open']).astype(int)
    df['is_bear'] = (c < df['open']).astype(int)
    df['hh'] = ((h > h.shift(1)) & (h > h.shift(2))).astype(int)
    df['ll'] = ((l < l.shift(1)) & (l < l.shift(2))).astype(int)
    return df


def load_vix_by_date():
    """Load VIX daily closes indexed by date"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT ts, close FROM ohlcv WHERE symbol='VIX' AND timeframe='1day' ORDER BY ts",
        conn
    )
    conn.close()
    if df.empty:
        return {}
    df['date'] = pd.to_datetime(df['ts'], unit='s').dt.strftime('%Y-%m-%d')
    return dict(zip(df['date'], df['close']))


# =============================================================
#  GENERATE BASE TRADES (score all bars once, cache results)
# =============================================================

def generate_base_trades(symbol='NQ'):
    """
    Run the full scoring engine once on all historical data.
    Cache every bar's scores so we can filter them quickly
    without re-running the heavy analysis for each parameter combo.
    """
    logger.info(f'Generating base trade scores for {symbol}...')

    from deep_edge import (score_htf_confluence, score_order_flow, score_session,
                           score_volume, score_vix, score_pd_zone, score_entry_signal,
                           calculate_structural_stop, LAYER_WEIGHTS, TOTAL_POSSIBLE,
                           TF_HIERARCHY, TF_MINUTES)

    # Load all timeframes
    dfs = {}
    available_tfs = []
    for tf in TF_HIERARCHY:
        df = load_tf(symbol, tf)
        if df is not None and len(df) >= 50:
            dfs[tf] = add_indicators(df)
            available_tfs.append(tf)
            logger.info(f'  {tf:<8} {len(dfs[tf]):>5} bars')

    vix_by_date = load_vix_by_date()
    entry_tfs_all = [tf for tf in ['5min','15min','1hour'] if tf in available_tfs]

    all_bars = []
    htf_cache = {}
    daily_counts = defaultdict(int)

    for entry_tf in entry_tfs_all:
        entry_df = dfs[entry_tf]
        higher_tfs = [tf for tf in available_tfs
                      if TF_MINUTES.get(tf,0) > TF_MINUTES.get(entry_tf,0)]

        logger.info(f'Scoring {entry_tf} ({len(entry_df)} bars)...')

        for i in range(100, len(entry_df) - 5):
            bar_dt   = entry_df.index[i]
            day_key  = bar_dt.strftime('%Y-%m-%d')
            dow      = bar_dt.weekday()
            ny_hour  = (bar_dt.hour - 5) % 24
            ny_min   = bar_dt.minute

            # Get VIX for this date
            vix_val = vix_by_date.get(day_key)

            # HTF confluence (cached per day)
            if f'{day_key}_{entry_tf}' not in htf_cache:
                htf_dfs = {}
                for htf in higher_tfs:
                    data = dfs[htf][dfs[htf].index <= bar_dt]
                    if len(data) >= 20:
                        htf_dfs[htf] = data
                if htf_dfs:
                    hs, direction, hd = score_htf_confluence(htf_dfs, list(htf_dfs.keys()))
                else:
                    hs, direction, hd = 0, 'neutral', {}
                htf_cache[f'{day_key}_{entry_tf}'] = (hs, direction, hd)

            htf_score, direction, htf_det = htf_cache[f'{day_key}_{entry_tf}']
            if direction == 'neutral':
                continue

            # Score all layers
            window = entry_df.iloc[max(0,i-80):i+1]
            df_5   = dfs.get('5min')
            df_15  = dfs.get('15min')
            if df_5  is not None: df_5  = df_5[df_5.index   <= bar_dt].tail(80)
            if df_15 is not None: df_15 = df_15[df_15.index <= bar_dt].tail(80)

            of_score,   of_det   = score_order_flow(df_5, df_15, direction)
            sess_score, sess_det = score_session(bar_dt)
            vol_score,  _        = score_volume(window, direction)
            vix_score,  vix_det  = score_vix(vix_val)
            pd_score,   pd_det   = score_pd_zone(window, direction)
            ent_score,  ent_det  = score_entry_signal(window, direction)

            total = htf_score + of_score + sess_score + vol_score + vix_score + pd_score + ent_score
            score_pct = round(total / TOTAL_POSSIBLE * 100, 1)

            # Calculate trade levels
            price = float(window['close'].iloc[-1])
            atr   = float(window['atr'].iloc[-1]) if not np.isnan(window['atr'].iloc[-1]) else price*0.005
            stop  = None
            try:
                from deep_edge import calculate_structural_stop
                stop = calculate_structural_stop(window, direction)
            except Exception:
                pass
            if stop is None or np.isnan(stop):
                stop = price - atr*1.5 if direction in ('long','bullish') else price + atr*1.5

            risk = abs(price - stop)
            if risk <= 0 or risk > atr * 6:
                continue

            all_bars.append({
                'dt':        bar_dt.isoformat(),
                'day':       day_key,
                'dow':       dow,
                'ny_hour':   ny_hour,
                'ny_min':    ny_min,
                'entry_tf':  entry_tf,
                'direction': direction,
                'score_pct': score_pct,
                'price':     round(price, 2),
                'stop':      round(stop, 2),
                'atr':       round(atr, 2),
                'risk':      round(risk, 2),
                'vix':       vix_val,
                # Individual layer scores
                'l_htf':     htf_score,
                'l_of':      of_score,
                'l_sess':    sess_score,
                'l_vol':     vol_score,
                'l_vix':     vix_score,
                'l_pd':      pd_score,
                'l_entry':   ent_score,
                # Flags
                'ob_conf':   of_det.get('ob_fvg_confluence') is not None,
                'bos':       of_det.get('bos_confirmed', False),
                'fvg':       of_det.get('fvg') is not None,
                'sweep':     of_det.get('liquidity_sweep', False),
                'sess_ok':   sess_det.get('trade_recommended', False),
            })

    logger.info(f'Generated {len(all_bars)} scored bars')
    return all_bars, dfs


# =============================================================
#  FORWARD TEST A FILTERED SET OF BARS
# =============================================================

def forward_test_bars(bars, dfs, rr_ratio=2.5):
    """Forward test a filtered set of bars with a given R:R"""
    trades = []
    for b in bars:
        entry_tf = b['entry_tf']
        entry_df = dfs.get(entry_tf)
        if entry_df is None:
            continue

        dt   = pd.Timestamp(b['dt'])
        idx  = entry_df.index.searchsorted(dt)
        if idx >= len(entry_df) - 5:
            continue

        price     = b['price']
        stop      = b['stop']
        risk      = b['risk']
        direction = b['direction']
        t1        = price + risk*rr_ratio if direction in ('long','bullish') else price - risk*rr_ratio
        t2        = price + risk*(rr_ratio+1.5) if direction in ('long','bullish') else price - risk*(rr_ratio+1.5)

        outcome    = 'timeout'
        exit_price = price
        max_j      = min(idx + 60, len(entry_df) - 1)

        for j in range(idx+1, max_j):
            bh = float(entry_df.iloc[j]['high'])
            bl = float(entry_df.iloc[j]['low'])
            bc = float(entry_df.iloc[j]['close'])
            if direction in ('long','bullish'):
                if bl <= stop or bc <= stop: outcome='loss';  exit_price=stop;  break
                if bh >= t1  or bc >= t1:   outcome='win';   exit_price=t1;    break
            else:
                if bh >= stop or bc >= stop: outcome='loss'; exit_price=stop;  break
                if bl <= t1  or bc <= t1:   outcome='win';   exit_price=t1;    break

        pnl_r = 0.0
        if outcome == 'win':
            pnl_r = rr_ratio
            # Check if T2 also hit
            for j2 in range(idx+1, max_j):
                bh = float(entry_df.iloc[j2]['high'])
                bl = float(entry_df.iloc[j2]['low'])
                bc = float(entry_df.iloc[j2]['close'])
                if direction in ('long','bullish'):
                    if bl <= price: break
                    if bh >= t2 or bc >= t2: pnl_r = rr_ratio*0.5 + (rr_ratio+1.5)*0.5; break
                else:
                    if bh >= price: break
                    if bl <= t2 or bc <= t2: pnl_r = rr_ratio*0.5 + (rr_ratio+1.5)*0.5; break
        elif outcome == 'loss':
            pnl_r = -1.0

        trades.append({**b, 'outcome': outcome, 'pnl_r': pnl_r, 'rr': rr_ratio})

    return trades


# =============================================================
#  SIMULATE TRADES
# =============================================================

def simulate(trades, balance=10000, risk_pct=2.0, commission=5.0):
    if not trades:
        return None
    bal = balance; peak = balance
    streak = 0; max_streak = 0; max_dd = 0
    pnls = []

    for t in sorted(trades, key=lambda x: x['dt']):
        if bal <= 0: break
        risk_amt = bal * (risk_pct/100)
        pnl_r    = float(t.get('pnl_r', 0))
        outcome  = t.get('outcome','timeout')
        if outcome == 'win'  and pnl_r <= 0: pnl_r =  t.get('rr', 2.5)
        if outcome == 'loss' and pnl_r >= 0: pnl_r = -1.0
        pnl_usd  = risk_amt * pnl_r - commission
        bal += pnl_usd
        pnls.append(pnl_usd)
        if bal > peak: peak = bal
        dd = (peak-bal)/peak*100
        if dd > max_dd: max_dd = dd
        if outcome == 'loss': streak+=1; max_streak=max(max_streak,streak)
        else: streak = 0

    if not pnls or bal <= 0:
        return None

    wins   = [t for t in trades if t['outcome']=='win']
    losses = [t for t in trades if t['outcome']=='loss']
    wr     = len(wins)/len(trades)*100 if trades else 0
    ret    = (bal-balance)/balance*100
    gp     = sum(p for p in pnls if p>0)
    gl     = abs(sum(p for p in pnls if p<0))
    pf     = round(gp/gl,2) if gl>0 else 999

    s = pd.Series(pnls)
    sharpe = 0.0
    if s.std() > 0 and len(pnls) > 2:
        sharpe = round((s.mean()/s.std())*np.sqrt(252),2)

    expectancy = sum(t['pnl_r'] for t in trades)/len(trades) if trades else 0

    return {
        'n_trades':    len(trades),
        'win_rate':    round(wr,1),
        'return_pct':  round(ret,2),
        'end_balance': round(bal,2),
        'max_dd':      round(max_dd,2),
        'sharpe':      sharpe,
        'profit_factor':pf,
        'max_streak':  max_streak,
        'expectancy':  round(expectancy,3),
    }


# =============================================================
#  MAIN OPTIMISATION LOOP
# =============================================================

def run_optimisation(symbol='NQ', balance=10000, quick=False):
    logger.info('=' * 65)
    logger.info(f'  APEX Parameter Optimisation — {symbol}')
    logger.info(f'  Balance: ${balance:,}')
    logger.info('=' * 65)

    # Generate all scored bars once
    all_bars, dfs = generate_base_trades(symbol)
    if not all_bars:
        logger.error('No bars generated')
        return None

    results = []
    total_combos = 0

    # Reduced grid for quick mode
    score_range  = [50, 55, 60] if quick else PARAM_GRID['min_score']
    rr_range     = [2.5, 3.0]   if quick else PARAM_GRID['rr_ratio']
    risk_range   = [2.0]        if quick else PARAM_GRID['risk_pct']
    sess_range   = ['all', 'no_midday', 'london_ny_open', 'all_sessions'] if quick else PARAM_GRID['session_filter']
    dow_range    = ['all', 'mon_fri_out'] if quick else PARAM_GRID['dow_filter']
    vix_range    = ['all', 'below_25'] if quick else PARAM_GRID['vix_filter']
    tf_range     = [['15min'], ['15min','5min']] if quick else PARAM_GRID['entry_tfs']

    combos = list(itertools.product(
        score_range, rr_range, risk_range,
        sess_range, dow_range, vix_range, tf_range
    ))

    logger.info(f'Testing {len(combos)} parameter combinations...')

    for i, (min_score, rr, risk_pct, sess_f, dow_f, vix_f, entry_tfs) in enumerate(combos):
        if i % 20 == 0:
            logger.info(f'  Progress: {i}/{len(combos)} ({i/len(combos)*100:.0f}%)')

        sess_fn = SESSION_RULES[sess_f]
        dow_fn  = DOW_RULES[dow_f]
        vix_fn  = VIX_RULES[vix_f]

        # Filter bars by this parameter combination
        filtered = [
            b for b in all_bars
            if b['score_pct'] >= min_score
            and b['entry_tf'] in entry_tfs
            and sess_fn(b['ny_hour'], b['ny_min'])
            and dow_fn(b['dow'])
            and vix_fn(b['vix'])
        ]

        if len(filtered) < 10:
            continue

        # Enforce max 2 trades per day
        daily_counts = defaultdict(int)
        capped = []
        for b in sorted(filtered, key=lambda x: x['dt']):
            key = f"{b['day']}_{b['entry_tf']}"
            if daily_counts[key] < 2:
                capped.append(b)
                daily_counts[key] += 1

        if len(capped) < 10:
            continue

        # Forward test
        trades = forward_test_bars(capped, dfs, rr_ratio=rr)
        stats  = simulate(trades, balance, risk_pct)

        if stats is None:
            continue

        results.append({
            'min_score':  min_score,
            'rr_ratio':   rr,
            'risk_pct':   risk_pct,
            'session':    sess_f,
            'dow':        dow_f,
            'vix':        vix_f,
            'entry_tfs':  '+'.join(entry_tfs),
            **stats,
        })

    if not results:
        logger.error('No valid results')
        return None

    df_r = pd.DataFrame(results)

    # Sort by different metrics
    by_sharpe  = df_r.sort_values('sharpe',      ascending=False).head(10)
    by_return  = df_r.sort_values('return_pct',  ascending=False).head(10)
    by_winrate = df_r.sort_values('win_rate',     ascending=False).head(10)
    by_dd      = df_r[df_r['return_pct']>0].sort_values('max_dd', ascending=True).head(10)

    logger.info('\n' + '='*65)
    logger.info('  OPTIMISATION RESULTS')
    logger.info('='*65)

    logger.info('\n  TOP 5 BY SHARPE RATIO:')
    for _, r in by_sharpe.head(5).iterrows():
        logger.info(f"  Score≥{r['min_score']} RR={r['rr_ratio']} Risk={r['risk_pct']}% "
                    f"{r['session']} {r['dow']} VIX:{r['vix']} TF:{r['entry_tfs']} | "
                    f"Sharpe={r['sharpe']:.2f} Ret={r['return_pct']:+.1f}% "
                    f"WR={r['win_rate']:.1f}% DD={r['max_dd']:.1f}% N={r['n_trades']}")

    logger.info('\n  TOP 5 BY RETURN:')
    for _, r in by_return.head(5).iterrows():
        logger.info(f"  Score≥{r['min_score']} RR={r['rr_ratio']} Risk={r['risk_pct']}% "
                    f"{r['session']} {r['dow']} VIX:{r['vix']} TF:{r['entry_tfs']} | "
                    f"Ret={r['return_pct']:+.1f}% Sharpe={r['sharpe']:.2f} "
                    f"WR={r['win_rate']:.1f}% DD={r['max_dd']:.1f}% N={r['n_trades']}")

    logger.info('\n  TOP 5 LOWEST DRAWDOWN (profitable only):')
    for _, r in by_dd.head(5).iterrows():
        logger.info(f"  Score≥{r['min_score']} RR={r['rr_ratio']} Risk={r['risk_pct']}% "
                    f"{r['session']} {r['dow']} VIX:{r['vix']} TF:{r['entry_tfs']} | "
                    f"DD={r['max_dd']:.1f}% Ret={r['return_pct']:+.1f}% "
                    f"Sharpe={r['sharpe']:.2f} N={r['n_trades']}")

    # Best overall
    best = by_sharpe.iloc[0]
    logger.info('\n' + '='*65)
    logger.info('  RECOMMENDED SETTINGS (best Sharpe):')
    logger.info(f"  Min Score:    {best['min_score']}/100")
    logger.info(f"  R:R Ratio:    {best['rr_ratio']}:1")
    logger.info(f"  Risk/Trade:   {best['risk_pct']}%")
    logger.info(f"  Session:      {best['session']}")
    logger.info(f"  Days:         {best['dow']}")
    logger.info(f"  VIX Filter:   {best['vix']}")
    logger.info(f"  Entry TFs:    {best['entry_tfs']}")
    logger.info(f"  --- Performance ---")
    logger.info(f"  Return:       {best['return_pct']:+.1f}%")
    logger.info(f"  Win Rate:     {best['win_rate']:.1f}%")
    logger.info(f"  Sharpe:       {best['sharpe']:.2f}")
    logger.info(f"  Max DD:       {best['max_dd']:.1f}%")
    logger.info(f"  Trades:       {best['n_trades']}")
    logger.info(f"  Expectancy:   {best['expectancy']:+.3f}R")
    logger.info('='*65)

    # Layer contribution analysis
    logger.info('\n  LAYER CONTRIBUTION ANALYSIS:')
    layer_analysis(all_bars, dfs)

    # Day of week analysis
    logger.info('\n  DAY OF WEEK ANALYSIS:')
    dow_analysis(all_bars, dfs)

    # Session time analysis
    logger.info('\n  SESSION TIME ANALYSIS:')
    session_analysis(all_bars, dfs)

    # Save results
    output = {
        'symbol':     symbol,
        'timestamp':  datetime.now(timezone.utc).isoformat(),
        'balance':    balance,
        'n_combos':   len(results),
        'best_sharpe':best.to_dict(),
        'top10_sharpe':  by_sharpe.to_dict('records'),
        'top10_return':  by_return.to_dict('records'),
        'top10_winrate': by_winrate.to_dict('records'),
        'top10_low_dd':  by_dd.to_dict('records'),
        'all_results':   df_r.sort_values('sharpe', ascending=False).to_dict('records'),
    }

    with open(f'optimiser_results_{symbol}.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f'\n  Full results saved to optimiser_results_{symbol}.json')

    return output


# =============================================================
#  LAYER CONTRIBUTION ANALYSIS
# =============================================================

def layer_analysis(all_bars, dfs):
    """Test each layer in isolation to find its individual edge contribution"""
    layers = {
        'HTF only':         lambda b: b['l_htf'] >= 15,
        'Order Flow only':  lambda b: b['l_of']  >= 15,
        'OB+FVG confluence':lambda b: b['ob_conf'],
        'BOS confirmed':    lambda b: b['bos'],
        'FVG present':      lambda b: b['fvg'],
        'Liquidity sweep':  lambda b: b['sweep'],
        'Good session':     lambda b: b['sess_ok'],
        'High volume':      lambda b: b['l_vol'] >= 7,
    }

    for name, fn in layers.items():
        filtered = [b for b in all_bars if fn(b)]
        if len(filtered) < 10:
            continue
        trades = forward_test_bars(filtered[:500], dfs, rr_ratio=2.5)
        if not trades:
            continue
        wins = [t for t in trades if t['outcome']=='win']
        losses = [t for t in trades if t['outcome']=='loss']
        wr = len(wins)/len(trades)*100 if trades else 0
        exp = sum(t['pnl_r'] for t in trades)/len(trades)
        logger.info(f"  {name:<25} N={len(trades):>4} WR={wr:.1f}% Exp={exp:+.3f}R")


# =============================================================
#  DAY OF WEEK ANALYSIS
# =============================================================

def dow_analysis(all_bars, dfs):
    days = ['Monday','Tuesday','Wednesday','Thursday','Friday']
    for dow, name in enumerate(days):
        filtered = [b for b in all_bars if b['dow']==dow and b['score_pct']>=55]
        if len(filtered) < 5:
            continue
        trades = forward_test_bars(filtered[:300], dfs, rr_ratio=2.5)
        if not trades:
            continue
        wins = [t for t in trades if t['outcome']=='win']
        wr   = len(wins)/len(trades)*100
        exp  = sum(t['pnl_r'] for t in trades)/len(trades)
        logger.info(f"  {name:<12} N={len(trades):>4} WR={wr:.1f}% Exp={exp:+.3f}R")


# =============================================================
#  SESSION TIME ANALYSIS
# =============================================================

def session_analysis(all_bars, dfs):
    windows = {
        'London 3:00-5:00':  lambda h,m: 3<=h<5,
        'London 5:00-8:00':  lambda h,m: 5<=h<8,
        'London Close 8-9':  lambda h,m: h==8,
        'Open  9:30-9:45':  lambda h,m: h==9 and 30<=m<45,
        'Prime 9:45-11:30': lambda h,m: (h==9 and m>=45) or (10<=h<11) or (h==11 and m<30),
        'Midday 11:30-1:30':lambda h,m: (h==11 and m>=30) or h==12 or (h==13 and m<30),
        'Arvo  1:30-3:00':  lambda h,m: (h==13 and m>=30) or h==14,
        'Power 3:00-4:00':  lambda h,m: h==15,
    }

    for name, fn in windows.items():
        filtered = [b for b in all_bars if fn(b['ny_hour'], b['ny_min']) and b['score_pct']>=55]
        if len(filtered) < 5:
            continue
        trades = forward_test_bars(filtered[:300], dfs, rr_ratio=2.5)
        if not trades:
            continue
        wins = [t for t in trades if t['outcome']=='win']
        wr   = len(wins)/len(trades)*100
        exp  = sum(t['pnl_r'] for t in trades)/len(trades)
        logger.info(f"  {name:<22} N={len(trades):>4} WR={wr:.1f}% Exp={exp:+.3f}R")


# =============================================================
#  MAIN
# =============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol',  default='NQ')
    parser.add_argument('--balance', type=float, default=10000)
    parser.add_argument('--quick',   action='store_true',
                        help='Quick mode — fewer combinations, faster results')
    args = parser.parse_args()

    run_optimisation(args.symbol, args.balance, args.quick)
