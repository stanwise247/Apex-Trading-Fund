"""
APEX Combined Portfolio Analysis — portfolio_analysis.py
=========================================================
Merges all validated strategy backtests, applies per-instrument
day/session filters, and computes compounded equity at 1% and 2% risk
starting from $1,000.

Usage:
  python3 portfolio_analysis.py
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from backtest_fvg_scored import run_scored_fvg_backtest

# ─────────────────────────────────────────────────────────────
#  CONFIG — day filters and session rules per instrument/setup
# ─────────────────────────────────────────────────────────────

SWING_CONFIGS = {
    'B_NQ': {'file': 'backtest_B_NQ_swing.json', 'days': [0,2,3,4], 'session': 'both'},
    'B_ES': {'file': 'backtest_B_ES_swing.json', 'days': [0,1,3],   'session': 'both'},
    'B_GC': {'file': 'backtest_B_GC_swing.json', 'days': [0,1,2,3], 'session': 'both'},
    'A_NQ': {'file': 'backtest_A_NQ_swing.json', 'days': None,       'session': 'primary'},
    'A_ES': {'file': 'backtest_A_ES_swing.json', 'days': None,       'session': 'primary'},
    'C_NQ': {'file': 'backtest_C_NQ_swing.json', 'days': None,       'session': 'primary'},
    'C_ES': {'file': 'backtest_C_ES_swing.json', 'days': [0,1,2,4], 'session': 'primary'},
    'C_GC': {'file': 'backtest_C_GC_swing.json', 'days': [0,1,2,3], 'session': 'primary'},
}

DOW_MAP = {'Mon':0, 'Tue':1, 'Wed':2, 'Thu':3, 'Fri':4, 'Sat':5, 'Sun':6}


def load_swing(key, cfg):
    with open(cfg['file']) as f:
        data = json.load(f)
    tlog = data.get('trade_log', [])
    rows = []
    for t in tlog:
        # Day filter
        entry_dt = pd.Timestamp(t['entry_time'])
        dow = entry_dt.dayofweek
        if cfg['days'] is not None and dow not in cfg['days']:
            continue
        # Session filter
        if cfg['session'] == 'primary' and t.get('quality', '') != 'primary':
            continue
        rows.append({
            'strategy':   key,
            'entry_time': entry_dt,
            'pnl_r':      float(t['pnl_r']),
            'exit_reason':t.get('exit_reason', ''),
            'session':    t.get('quality', ''),
        })
    return rows


def load_fvg(min_score=60):
    result = run_scored_fvg_backtest('NQ', min_score=min_score, start='2024-09-01')
    rows = []
    for t in result.get('trade_log', []):
        rows.append({
            'strategy':   f'FVG_NQ_s{min_score}',
            'entry_time': pd.Timestamp(t['entry_time']),
            'pnl_r':      float(t['pnl_r']),
            'exit_reason':t.get('exit_reason', ''),
            'session':    'primary',
        })
    return rows


def per_instrument_stats(name, trades):
    if not trades:
        return
    pnls = [t['pnl_r'] for t in trades]
    wins = [p for p in pnls if p > 0]
    wr   = len(wins)/len(pnls)*100
    exp  = np.mean(pnls)
    cum  = np.cumsum(pnls)
    dd   = float(np.min(cum - np.maximum.accumulate(cum)))
    sh   = float(np.mean(pnls)/np.std(pnls)*np.sqrt(252)) if np.std(pnls) > 0 else 0
    # trades per week
    dates  = sorted(t['entry_time'] for t in trades)
    weeks  = max((dates[-1] - dates[0]).days / 7, 1)
    tpw    = len(trades) / weeks
    print(f'  {name:<12} {len(trades):>6}  {wr:>6.1f}%  {exp:>+7.3f}R  {sum(pnls):>+8.2f}R  {dd:>7.2f}R  {sh:>6.2f}  {tpw:>5.1f}')


def compound_equity(all_trades_sorted, risk_pct, start=1000.0):
    balance = start
    peak    = start
    max_dd_pct = 0.0
    curve   = [balance]
    for t in all_trades_sorted:
        risk_amt = balance * (risk_pct / 100)
        balance += risk_amt * t['pnl_r']
        if balance > peak:
            peak = balance
        dd_pct = (peak - balance) / peak * 100
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
        curve.append(balance)
    total_ret = (balance - start) / start * 100
    # CAGR
    dates  = [t['entry_time'] for t in all_trades_sorted]
    if dates:
        years = (dates[-1] - dates[0]).days / 365.25
        cagr  = ((balance / start) ** (1/max(years,0.01)) - 1) * 100
    else:
        cagr  = 0
    return balance, total_ret, max_dd_pct, cagr, curve


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

print('Loading swing strategies...')
all_trades = []

print(f'\n  {"Strategy":<12} {"Trades":>6}  {"WR":>7}  {"Exp":>8}  {"Total R":>9}  {"MaxDD":>8}  {"Sharpe":>7}  {"T/wk":>5}')
print('  ' + '─'*72)

for key, cfg in SWING_CONFIGS.items():
    trades = load_swing(key, cfg)
    all_trades.extend(trades)
    per_instrument_stats(key, trades)

print('\nLoading FVG NQ (min_score=60)...')
fvg_trades = load_fvg(min_score=60)
all_trades.extend(fvg_trades)
per_instrument_stats('FVG_NQ_s60', fvg_trades)

# Sort all trades by entry time
all_trades.sort(key=lambda t: t['entry_time'])

# ─── Overall portfolio stats ────────────────────────────────
print(f'\n{"="*72}')
print('  COMBINED PORTFOLIO')
print(f'{"="*72}')
pnls = [t['pnl_r'] for t in all_trades]
wins = [p for p in pnls if p > 0]
wr   = len(wins)/len(pnls)*100
exp  = np.mean(pnls)
cum  = np.cumsum(pnls)
dd_r = float(np.min(cum - np.maximum.accumulate(cum)))
sh   = float(np.mean(pnls)/np.std(pnls)*np.sqrt(252)) if np.std(pnls) > 0 else 0
dates  = sorted(t['entry_time'] for t in all_trades)
weeks  = max((dates[-1] - dates[0]).days / 7, 1)
tpw    = len(all_trades) / weeks

print(f'  Total trades : {len(all_trades)}')
print(f'  Date range   : {dates[0].date()} → {dates[-1].date()}')
print(f'  Win rate     : {wr:.1f}%')
print(f'  Expectancy   : {exp:+.3f}R')
print(f'  Total R      : {sum(pnls):+.2f}R')
print(f'  Max DD (R)   : {dd_r:.2f}R')
print(f'  Sharpe       : {sh:.2f}')
print(f'  Avg T/week   : {tpw:.1f}')

# ─── Compounding at 1% and 2% ───────────────────────────────
print(f'\n{"─"*72}')
print('  COMPOUNDED EQUITY — $1,000 starting balance')
print(f'{"─"*72}')
print(f'  {"Risk":>6}  {"Final $":>10}  {"Total %":>9}  {"CAGR":>8}  {"Max DD%":>9}')
print(f'  {"─"*6}  {"─"*10}  {"─"*9}  {"─"*8}  {"─"*9}')

for risk_pct in (1.0, 2.0):
    final, total_ret, max_dd_pct, cagr, curve = compound_equity(all_trades, risk_pct)
    print(f'  {risk_pct:>5.0f}%  ${final:>9,.0f}  {total_ret:>+8.1f}%  {cagr:>+7.1f}%  {max_dd_pct:>8.1f}%')

# ─── T/week by strategy ─────────────────────────────────────
print(f'\n{"─"*72}')
print('  AVG TRADES PER WEEK BY STRATEGY')
print(f'{"─"*72}')
strategy_groups = {}
for t in all_trades:
    strategy_groups.setdefault(t['strategy'], []).append(t)

for strat, strades in sorted(strategy_groups.items()):
    sdates = sorted(t['entry_time'] for t in strades)
    wks = max((sdates[-1] - sdates[0]).days / 7, 1)
    spnls = [t['pnl_r'] for t in strades]
    swins = [p for p in spnls if p > 0]
    print(f'  {strat:<14}  {len(strades):>5} trades  {len(strades)/wks:>4.1f}/wk  '
          f'WR={len(swins)/len(spnls)*100:.0f}%  EV={np.mean(spnls):+.3f}R')

total_tpw = sum(len(v)/max((sorted(t['entry_time'] for t in v)[-1]-sorted(t['entry_time'] for t in v)[0]).days/7,1)
               for v in strategy_groups.values())
print(f'  {"TOTAL":<14}  {len(all_trades):>5} trades  {total_tpw:>4.1f}/wk')

# ─── Month-by-month returns ─────────────────────────────────
print(f'\n{"─"*72}')
print('  MONTHLY R BREAKDOWN (portfolio, 1% risk)')
print(f'{"─"*72}')
balance = 1000.0
monthly = {}
for t in all_trades:
    month_key = t['entry_time'].strftime('%Y-%m')
    risk_amt  = balance * 0.01
    gain      = risk_amt * t['pnl_r']
    monthly[month_key] = monthly.get(month_key, {'r': 0.0, 'trades': 0, 'pct': 0.0})
    monthly[month_key]['r']      += t['pnl_r']
    monthly[month_key]['trades'] += 1
    monthly[month_key]['pct']    += gain / 1000 * 100  # approx % vs $1k start for display
    balance += gain

for month, m in sorted(monthly.items()):
    bar = '█' * int(abs(m['r'])) + ('▌' if abs(m['r']) % 1 >= 0.5 else '')
    sign = '+' if m['r'] >= 0 else '-'
    print(f'  {month}  {m["r"]:>+7.2f}R  {m["trades"]:>4} trades  {sign}{bar}')

print()
