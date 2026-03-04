"""
APEX Backtest Engine — backtest.py
====================================
Takes pattern results from patterns.py and runs full historical simulations.

Features:
  - Full equity curve simulation from any starting balance
  - Variable risk per trade (1-4% of account)
  - Slippage and commission modelling
  - Monte Carlo simulation (1000 runs)
  - Performance statistics (Sharpe, max DD, profit factor etc)
  - Regime breakdown
  - Multi-pattern portfolio simulation
  - Results saved to JSON for the dashboard

Run with: python3 backtest.py --symbol NQ --balance 1000 --risk 2
"""

import json
import sqlite3
import argparse
import logging
import random
from datetime import datetime, timezone
from copy import deepcopy

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('APEX.backtest')

DB_PATH     = 'apex_market.db'
COMMISSION  = 5.0    # $ per round trip per contract
SLIPPAGE    = 2      # NQ points slippage per trade
NQ_TICK     = 5.0    # $ per point for NQ (1 contract = $20/point)
POINT_VALUE = 20.0   # $ per NQ point


# ─────────────────────────────────────────────
#  LOAD PATTERN RESULTS
# ─────────────────────────────────────────────

def load_pattern_results(symbol, timeframe):
    filename = f'edge_results_{symbol}_{timeframe}.json'
    try:
        with open(filename) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f'No pattern results found. Run patterns.py first: python3 patterns.py --symbol {symbol}')
        return None


def load_patterns_from_db(symbol=None):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    if symbol:
        c.execute('SELECT * FROM patterns WHERE active=1 AND symbol=? ORDER BY edge_score DESC', (symbol,))
    else:
        c.execute('SELECT * FROM patterns WHERE active=1 ORDER BY edge_score DESC')
    cols = [d[0] for d in c.description]
    rows = [dict(zip(cols, r)) for r in c.fetchall()]
    conn.close()
    return rows


# ─────────────────────────────────────────────
#  SIMULATION ENGINE
# ─────────────────────────────────────────────

def simulate_strategy(trades, starting_balance=1000, risk_pct=2.0,
                       slippage_pts=SLIPPAGE, commission=COMMISSION,
                       point_value=POINT_VALUE):
    """
    Simulate trading a pattern's historical trades on a real account.

    trades:            list of trade dicts from backtest_pattern()
    starting_balance:  account balance in $
    risk_pct:          % of account to risk per trade
    slippage_pts:      NQ points lost to slippage per trade
    commission:        $ commission per trade round trip
    point_value:       $ value per NQ point (1 contract = $20/point)

    Returns equity_curve, trade_log, statistics
    """
    if not trades:
        return None

    balance        = starting_balance
    peak_balance   = starting_balance
    equity_curve   = [{'balance': balance, 'trade_num': 0, 'date': 'start'}]
    trade_log      = []
    max_drawdown   = 0
    consecutive_losses = 0
    max_consec_losses  = 0
    current_consec     = 0

    for i, trade in enumerate(sorted(trades, key=lambda x: x['entry_dt'])):
        if balance <= 0:
            break

        # Position sizing — risk fixed % of current balance
        risk_amount = balance * (risk_pct / 100)

        # Calculate stop distance in points
        stop_dist = abs(trade['entry'] - trade['stop'])
        if stop_dist <= 0:
            continue

        # How many contracts can we trade given our risk?
        # risk_amount = contracts * stop_dist * point_value
        contracts = max(1, int(risk_amount / (stop_dist * point_value)))
        # Cap at reasonable level
        contracts = min(contracts, 10)

        # Apply slippage to entry and exit
        entry  = trade['entry']
        exit_p = trade['exit']

        if trade['direction'] == 'long':
            entry  += slippage_pts * 0.5   # slippage on entry
            exit_p -= slippage_pts * 0.5   # slippage on exit
            pnl_pts = exit_p - entry
        else:
            entry  -= slippage_pts * 0.5
            exit_p += slippage_pts * 0.5
            pnl_pts = entry - exit_p

        pnl_dollars = pnl_pts * point_value * contracts - commission
        balance    += pnl_dollars

        # Track drawdown
        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / peak_balance * 100
        if dd > max_drawdown:
            max_drawdown = dd

        # Consecutive losses
        if trade['outcome'] == 'loss':
            current_consec += 1
            max_consec_losses = max(max_consec_losses, current_consec)
        else:
            current_consec = 0

        trade_log.append({
            'num':       i + 1,
            'date':      trade['entry_dt'][:10],
            'direction': trade['direction'],
            'entry':     round(entry, 2),
            'exit':      round(exit_p, 2),
            'contracts': contracts,
            'pnl_pts':   round(pnl_pts, 2),
            'pnl_$':     round(pnl_dollars, 2),
            'balance':   round(balance, 2),
            'outcome':   trade['outcome'],
            'dd_pct':    round(dd, 2),
        })

        equity_curve.append({
            'balance':   round(balance, 2),
            'trade_num': i + 1,
            'date':      trade['entry_dt'][:10],
            'pnl':       round(pnl_dollars, 2),
        })

    if not trade_log:
        return None

    wins      = [t for t in trade_log if t['outcome'] == 'win']
    losses    = [t for t in trade_log if t['outcome'] == 'loss']
    total_ret = (balance - starting_balance) / starting_balance * 100

    # Monthly breakdown
    monthly = {}
    for t in trade_log:
        month = t['date'][:7]
        if month not in monthly:
            monthly[month] = {'pnl': 0, 'trades': 0, 'wins': 0}
        monthly[month]['pnl']    += t['pnl_$']
        monthly[month]['trades'] += 1
        if t['outcome'] == 'win':
            monthly[month]['wins'] += 1
    monthly_returns = [
        {'month': m, 'pnl': round(v['pnl'],2), 'trades': v['trades'],
         'win_rate': round(v['wins']/v['trades']*100,1) if v['trades'] else 0}
        for m, v in sorted(monthly.items())
    ]

    # Profit factor
    gross_profit = sum(t['pnl_$'] for t in wins)
    gross_loss   = abs(sum(t['pnl_$'] for t in losses))
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float('inf')

    # Sharpe ratio (annualised, assuming 252 trading days)
    pnl_series = pd.Series([t['pnl_$'] for t in trade_log])
    sharpe = 0.0
    if pnl_series.std() > 0:
        trades_per_year = len(trade_log) / max(1,
            (pd.Timestamp(trade_log[-1]['date']) - pd.Timestamp(trade_log[0]['date'])).days / 365)
        sharpe = round((pnl_series.mean() / pnl_series.std()) * np.sqrt(trades_per_year), 2)

    stats = {
        'starting_balance':   starting_balance,
        'ending_balance':     round(balance, 2),
        'total_return_pct':   round(total_ret, 2),
        'total_pnl_$':        round(balance - starting_balance, 2),
        'risk_pct_per_trade': risk_pct,
        'total_trades':       len(trade_log),
        'wins':               len(wins),
        'losses':             len(losses),
        'win_rate_pct':       round(len(wins)/len(trade_log)*100, 1) if trade_log else 0,
        'avg_win_$':          round(np.mean([t['pnl_$'] for t in wins]), 2) if wins else 0,
        'avg_loss_$':         round(np.mean([t['pnl_$'] for t in losses]), 2) if losses else 0,
        'largest_win_$':      round(max([t['pnl_$'] for t in wins]), 2) if wins else 0,
        'largest_loss_$':     round(min([t['pnl_$'] for t in losses]), 2) if losses else 0,
        'max_drawdown_pct':   round(max_drawdown, 2),
        'max_consec_losses':  max_consec_losses,
        'profit_factor':      profit_factor,
        'sharpe_ratio':       sharpe,
        'gross_profit_$':     round(gross_profit, 2),
        'gross_loss_$':       round(gross_loss, 2),
        'slippage_pts':       slippage_pts,
        'commission_per_trade': commission,
    }

    return {
        'stats':           stats,
        'equity_curve':    equity_curve,
        'trade_log':       trade_log,
        'monthly_returns': monthly_returns,
    }


# ─────────────────────────────────────────────
#  MONTE CARLO SIMULATION
# ─────────────────────────────────────────────

def monte_carlo(trades, starting_balance=1000, risk_pct=2.0, n_simulations=1000):
    """
    Run N simulations with randomised trade ordering.
    Shows range of possible outcomes — not just the historical sequence.
    """
    if not trades or len(trades) < 5:
        return None

    logger.info(f'Running Monte Carlo: {n_simulations} simulations...')

    final_balances = []
    max_drawdowns  = []
    ruin_count     = 0  # simulations where balance hit 0

    for _ in range(n_simulations):
        shuffled   = random.sample(trades, len(trades))
        result     = simulate_strategy(shuffled, starting_balance, risk_pct)
        if result:
            fb = result['stats']['ending_balance']
            dd = result['stats']['max_drawdown_pct']
            final_balances.append(fb)
            max_drawdowns.append(dd)
            if fb <= 0:
                ruin_count += 1

    if not final_balances:
        return None

    fb = np.array(final_balances)
    dd = np.array(max_drawdowns)

    return {
        'n_simulations':      n_simulations,
        'starting_balance':   starting_balance,
        'risk_pct':           risk_pct,
        'median_final':       round(float(np.median(fb)), 2),
        'mean_final':         round(float(np.mean(fb)), 2),
        'best_case_5pct':     round(float(np.percentile(fb, 95)), 2),
        'worst_case_5pct':    round(float(np.percentile(fb, 5)), 2),
        'best_case_1pct':     round(float(np.percentile(fb, 99)), 2),
        'worst_case_1pct':    round(float(np.percentile(fb, 1)), 2),
        'median_return_pct':  round((float(np.median(fb)) - starting_balance) / starting_balance * 100, 2),
        'prob_profit_pct':    round(float(np.mean(fb > starting_balance) * 100), 1),
        'prob_50pct_gain':    round(float(np.mean(fb > starting_balance * 1.5) * 100), 1),
        'prob_double':        round(float(np.mean(fb > starting_balance * 2) * 100), 1),
        'prob_ruin':          round(ruin_count / n_simulations * 100, 2),
        'median_max_dd':      round(float(np.median(dd)), 2),
        'worst_max_dd_95':    round(float(np.percentile(dd, 95)), 2),
        'distribution_percentiles': {
            'p5':  round(float(np.percentile(fb, 5)),  2),
            'p10': round(float(np.percentile(fb, 10)), 2),
            'p25': round(float(np.percentile(fb, 25)), 2),
            'p50': round(float(np.percentile(fb, 50)), 2),
            'p75': round(float(np.percentile(fb, 75)), 2),
            'p90': round(float(np.percentile(fb, 90)), 2),
            'p95': round(float(np.percentile(fb, 95)), 2),
        },
        'final_balances_sample': sorted(random.sample(list(fb), min(100, len(fb))))
    }


# ─────────────────────────────────────────────
#  RISK LEVEL COMPARISON
# ─────────────────────────────────────────────

def compare_risk_levels(trades, starting_balance=1000):
    """Run simulation at 1%, 2%, 3%, 4% risk to show tradeoff"""
    results = {}
    for risk in [1, 2, 3, 4]:
        sim = simulate_strategy(trades, starting_balance, risk)
        if sim:
            s = sim['stats']
            results[str(risk) + 'pct'] = {
                'risk_pct':         risk,
                'ending_balance':   s['ending_balance'],
                'total_return_pct': s['total_return_pct'],
                'max_drawdown_pct': s['max_drawdown_pct'],
                'sharpe_ratio':     s['sharpe_ratio'],
                'profit_factor':    s['profit_factor'],
            }
    return results


# ─────────────────────────────────────────────
#  YEAR-BY-YEAR BREAKDOWN
# ─────────────────────────────────────────────

def yearly_breakdown(trades, starting_balance=1000, risk_pct=2.0):
    """Show performance year by year"""
    if not trades:
        return {}

    sorted_trades = sorted(trades, key=lambda x: x['entry_dt'])
    years = {}
    for t in sorted_trades:
        year = t['entry_dt'][:4]
        if year not in years:
            years[year] = []
        years[year].append(t)

    results = {}
    running_balance = starting_balance

    for year in sorted(years.keys()):
        year_trades = years[year]
        sim = simulate_strategy(year_trades, running_balance, risk_pct)
        if sim:
            s = sim['stats']
            results[year] = {
                'trades':         s['total_trades'],
                'win_rate':       s['win_rate_pct'],
                'return_pct':     s['total_return_pct'],
                'pnl_$':          s['total_pnl_$'],
                'max_dd_pct':     s['max_drawdown_pct'],
                'ending_balance': s['ending_balance'],
            }
            running_balance = s['ending_balance']

    return results


# ─────────────────────────────────────────────
#  PORTFOLIO SIMULATION — top N patterns together
# ─────────────────────────────────────────────

def portfolio_simulation(pattern_results, starting_balance=1000,
                          risk_pct_per_pattern=1.0, top_n=5):
    """
    Simulate trading the top N patterns simultaneously.
    Each pattern gets its own % risk allocation.
    Trades are merged chronologically.
    """
    patterns = sorted(pattern_results, key=lambda x: x['edge_score'], reverse=True)[:top_n]
    all_trades = []
    for p in patterns:
        for t in p.get('trades', []):
            t_copy = dict(t)
            t_copy['pattern'] = p['label']
            all_trades.append(t_copy)

    all_trades.sort(key=lambda x: x['entry_dt'])
    sim = simulate_strategy(all_trades, starting_balance, risk_pct_per_pattern)
    if sim:
        sim['portfolio_patterns'] = [p['label'] for p in patterns]
    return sim


# ─────────────────────────────────────────────
#  MAIN RUNNER
# ─────────────────────────────────────────────

def run_backtest(symbol='NQ', timeframe='1day',
                 starting_balance=1000, risk_pct=2.0,
                 n_monte_carlo=1000):

    logger.info('=' * 60)
    logger.info(f'  APEX Backtest Engine')
    logger.info(f'  Symbol: {symbol} | Timeframe: {timeframe}')
    logger.info(f'  Starting Balance: ${starting_balance:,} | Risk: {risk_pct}%/trade')
    logger.info('=' * 60)

    data = load_pattern_results(symbol, timeframe)
    if not data:
        return None

    patterns     = data.get('patterns', [])
    fvg_stats    = data.get('fvg_stats', {})
    sr_levels    = data.get('sr_levels', [])
    bars_analysed= data.get('bars_analysed', 0)
    date_range   = data.get('date_range', {})

    if not patterns:
        logger.error('No patterns found. Run patterns.py first.')
        return None

    logger.info(f'  Loaded {len(patterns)} patterns | {bars_analysed} bars analysed')
    logger.info(f'  Date range: {date_range.get("from","?")} to {date_range.get("to","?")}')

    all_results = []

    for pattern in patterns:
        trades = pattern.get('trades', [])
        if not trades:
            continue

        logger.info(f'\n  Simulating: {pattern["label"]} ({len(trades)} trades)')

        # Primary simulation
        sim = simulate_strategy(trades, starting_balance, risk_pct)
        if not sim:
            continue

        # Monte Carlo
        mc = monte_carlo(trades, starting_balance, risk_pct, n_monte_carlo)

        # Risk level comparison
        risk_comp = compare_risk_levels(trades, starting_balance)

        # Year by year
        yearly = yearly_breakdown(trades, starting_balance, risk_pct)

        s = sim['stats']
        logger.info(f'    Return: {s["total_return_pct"]:+.1f}% | '
                    f'Win Rate: {s["win_rate_pct"]:.1f}% | '
                    f'Max DD: {s["max_drawdown_pct"]:.1f}% | '
                    f'Sharpe: {s["sharpe_ratio"]:.2f} | '
                    f'PF: {s["profit_factor"]}')

        if mc:
            logger.info(f'    Monte Carlo: Median {mc["median_return_pct"]:+.1f}% | '
                        f'Prob Profit: {mc["prob_profit_pct"]}% | '
                        f'Prob Ruin: {mc["prob_ruin"]}%')

        all_results.append({
            'pattern':       pattern['label'],
            'symbol':        symbol,
            'direction':     pattern['direction'],
            'edge_score':    pattern['edge_score'],
            'win_rate':      pattern['win_rate'],
            'expectancy':    pattern['expectancy'],
            'occurrences':   pattern['occurrences'],
            'best_regime':   pattern.get('best_regime', 'all'),
            'regime_breakdown': pattern.get('regime_breakdown', {}),
            'simulation':    sim,
            'monte_carlo':   mc,
            'risk_comparison': risk_comp,
            'yearly_breakdown': yearly,
        })

    # Portfolio simulation — top 5 patterns together
    logger.info('\n  Running portfolio simulation (top 5 patterns)...')
    portfolio_sim = portfolio_simulation(patterns, starting_balance,
                                          risk_pct_per_pattern=risk_pct/2)

    # Build final output
    output = {
        'symbol':          symbol,
        'timeframe':       timeframe,
        'timestamp':       datetime.now(timezone.utc).isoformat(),
        'config': {
            'starting_balance': starting_balance,
            'risk_pct':         risk_pct,
            'slippage_pts':     SLIPPAGE,
            'commission':       COMMISSION,
            'point_value':      POINT_VALUE,
            'n_monte_carlo':    n_monte_carlo,
        },
        'bars_analysed':   bars_analysed,
        'date_range':      date_range,
        'results':         all_results,
        'fvg_stats':       fvg_stats,
        'sr_levels':       sr_levels,
        'portfolio_sim':   portfolio_sim,
        'summary': {
            'total_patterns_tested':    len(all_results),
            'patterns_positive_return': sum(1 for r in all_results
                                            if r['simulation']['stats']['total_return_pct'] > 0),
            'best_pattern':   max(all_results, key=lambda x: x['simulation']['stats']['total_return_pct'])['pattern'] if all_results else None,
            'best_return':    max(all_results, key=lambda x: x['simulation']['stats']['total_return_pct'])['simulation']['stats']['total_return_pct'] if all_results else 0,
            'best_sharpe':    max(all_results, key=lambda x: x['simulation']['stats']['sharpe_ratio'])['pattern'] if all_results else None,
        }
    }

    outfile = f'backtest_{symbol}_{timeframe}_{starting_balance}.json'
    with open(outfile, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info('\n' + '=' * 60)
    logger.info(f'  BACKTEST COMPLETE')
    logger.info(f'  Patterns tested:      {len(all_results)}')
    logger.info(f'  Profitable patterns:  {output["summary"]["patterns_positive_return"]}')
    logger.info(f'  Best pattern:         {output["summary"]["best_pattern"]}')
    logger.info(f'  Best return:          {output["summary"]["best_return"]:+.1f}%')
    if portfolio_sim:
        ps = portfolio_sim['stats']
        logger.info(f'  Portfolio return:     {ps["total_return_pct"]:+.1f}%')
        logger.info(f'  Portfolio max DD:     {ps["max_drawdown_pct"]:.1f}%')
    logger.info(f'  Full results saved to: {outfile}')
    logger.info('=' * 60)

    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='APEX Backtest Engine')
    parser.add_argument('--symbol',    default='NQ',   help='Symbol (default: NQ)')
    parser.add_argument('--timeframe', default='1day', help='Timeframe (default: 1day)')
    parser.add_argument('--balance',   type=float, default=1000, help='Starting balance $ (default: 1000)')
    parser.add_argument('--risk',      type=float, default=2.0,  help='Risk %% per trade (default: 2)')
    parser.add_argument('--mc',        type=int,   default=1000, help='Monte Carlo simulations (default: 1000)')
    args = parser.parse_args()

    run_backtest(
        symbol           = args.symbol,
        timeframe        = args.timeframe,
        starting_balance = args.balance,
        risk_pct         = args.risk,
        n_monte_carlo    = args.mc,
    )
