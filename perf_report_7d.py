import os, psycopg2, psycopg2.extras
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

conn  = psycopg2.connect(os.environ['DATABASE_URL'])
cur   = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

PNL_PER_POINT = {'MNQ': 2.0, 'ES': 50.0, 'GC': 100.0, 'NQ': 20.0}
cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

cur.execute("""
    SELECT id, symbol, setup, direction, entry_price, exit_price,
           stop, target, rr_planned, entry_time, exit_time,
           exit_reason, pnl_r, contracts, notes
    FROM apex_trades
    WHERE status = 'closed'
      AND exit_time >= %s
    ORDER BY entry_time ASC
""", (cutoff,))
trades = cur.fetchall()

if not trades:
    print("No closed trades in the last 7 days.")
    conn.close()
    raise SystemExit

print("\n" + "="*90)
print(f"  APEX PERFORMANCE REPORT  |  Last 7 Days  ({cutoff[:10]} to now)")
print("="*90)

report_rows = []
for t in trades:
    sym        = t['symbol']
    entry_time = t['entry_time']

    cur.execute("""
        SELECT regime, confidence, hurst, autocorr, vol_ratio
        FROM regime_log
        WHERE symbol = %s
          AND timestamp <= %s::timestamptz
        ORDER BY timestamp DESC
        LIMIT 1
    """, (sym, entry_time))
    reg = cur.fetchone()

    regime = reg['regime']     if reg else 'N/A'
    conf   = reg['confidence'] if reg else None
    hurst  = reg['hurst']      if reg else None

    setup_letter = (t['setup'] or '').split('_')[0].upper()
    l3_opted_out = setup_letter in ('D', 'H', 'J')

    if l3_opted_out:
        l3_mult_approx = '1.0x (opt-out)'
        l3_prob_display = 'N/A'
    elif reg and conf is not None:
        l3_prob_display = f'{conf:.2f}'
        if regime == 'TRENDING':
            if conf >= 0.65:       l3_mult_approx = '1.5x'
            elif conf >= 0.55:     l3_mult_approx = '1.25x'
            elif conf >= 0.45:     l3_mult_approx = '1.0x'
            else:                  l3_mult_approx = '0.75x'
        elif regime in ('CHOPPY', 'MEAN_REVERTING'):
            l3_mult_approx = '0.75x'
        else:
            l3_mult_approx = '1.0x'
    else:
        l3_mult_approx = 'N/A'
        l3_prob_display = 'N/A'

    ppl = PNL_PER_POINT.get(sym, 0)
    contracts = t['contracts'] or 1
    if t['exit_price'] and t['entry_price']:
        raw_pts = t['exit_price'] - t['entry_price']
        direction_mult = 1 if t['direction'] == 'long' else -1
        pnl_dollars = round(raw_pts * direction_mult * ppl * contracts, 2)
    else:
        pnl_dollars = None

    pnl_r    = t['pnl_r'] or 0
    win_loss = 'WIN' if pnl_r > 0 else 'LOSS'

    reason = (t['exit_reason'] or '').lower()
    if 'stop' in reason:
        exit_bucket = 'STOPPED OUT'
    elif 'target' in reason:
        exit_bucket = 'HIT TARGET'
    elif 'session' in reason or 'eod' in reason:
        exit_bucket = 'SESSION CLOSE'
    elif 'manual' in reason:
        exit_bucket = 'MANUAL CLOSE'
    else:
        exit_bucket = (t['exit_reason'] or 'OTHER').upper()

    entry_str = str(entry_time)[:19].replace('T', ' ')
    exit_str  = str(t['exit_time'])[:19].replace('T', ' ') if t['exit_time'] else '—'

    report_rows.append({
        'id':          t['id'],
        'setup':       t['setup'],
        'symbol':      sym,
        'direction':   t['direction'],
        'entry':       t['entry_price'],
        'exit':        t['exit_price'],
        'entry_time':  entry_str,
        'exit_time':   exit_str,
        'pnl_r':       round(pnl_r, 3),
        'pnl_dollars': pnl_dollars,
        'win_loss':    win_loss,
        'contracts':   contracts,
        'regime':      regime,
        'conf':        round(conf, 3) if conf is not None else None,
        'hurst':       round(hurst, 3) if hurst is not None else None,
        'l3_prob':     l3_prob_display,
        'l3_mult':     l3_mult_approx,
        'exit_reason': t['exit_reason'] or '—',
        'exit_bucket': exit_bucket,
    })

conn.close()
df = pd.DataFrame(report_rows)

# ── Trade detail ─────────────────────────────────────────────
print("\n" + "─"*90)
print("  TRADE DETAIL")
print("─"*90)
for r in report_rows:
    wl_tag = 'WIN ' if r['win_loss'] == 'WIN' else 'LOSS'
    d_str  = (f"${r['pnl_dollars']:+,.2f}") if r['pnl_dollars'] is not None else '$N/A'
    print(f"""
  ID #{r['id']}  |  {r['symbol']} {r['direction'].upper()}  [{r['setup']}]  →  {wl_tag}
  ├─ Entry:       {r['entry']:.2f}  @  {r['entry_time']} UTC
  ├─ Exit:        {r['exit']:.2f}  @  {r['exit_time']} UTC
  ├─ Contracts:   {r['contracts']}
  ├─ P&L:         {r['pnl_r']:+.3f}R   {d_str}
  ├─ Exit type:   {r['exit_reason']}
  ├─ Regime:      {r['regime']}  (conf={r['conf']}  hurst={r['hurst']})
  └─ L3:          prob={r['l3_prob']}  approx mult={r['l3_mult']}""")

# ── Summary ───────────────────────────────────────────────────
total   = len(df)
wins    = (df['win_loss'] == 'WIN').sum()
wr      = wins / total * 100
total_r = df['pnl_r'].sum()
total_d = df['pnl_dollars'].dropna().sum()
avg_r   = df['pnl_r'].mean()
best_i  = df['pnl_r'].idxmax()
worst_i = df['pnl_r'].idxmin()
best    = df.loc[best_i]
worst   = df.loc[worst_i]

best_d  = (f"${best['pnl_dollars']:+,.2f}") if best['pnl_dollars'] is not None else ''
worst_d = (f"${worst['pnl_dollars']:+,.2f}") if worst['pnl_dollars'] is not None else ''

print("\n" + "="*90)
print("  SUMMARY")
print("="*90)
print(f"""
  Total trades:    {total}
  Wins / Losses:   {wins}W  /  {total - wins}L
  Win rate:        {wr:.1f}%
  Total R:         {total_r:+.2f}R
  Total dollars:   ${total_d:+,.2f}
  Avg R/trade:     {avg_r:+.3f}R

  Best trade:   #{int(best['id'])}  {best['symbol']} {best['direction'].upper()}  [{best['setup']}]  {best['pnl_r']:+.3f}R  {best_d}
  Worst trade:  #{int(worst['id'])}  {worst['symbol']} {worst['direction'].upper()}  [{worst['setup']}]  {worst['pnl_r']:+.3f}R  {worst_d}
""")

# ── By setup ──────────────────────────────────────────────────
print("─"*70)
print("  BY SETUP")
print("─"*70)
by_setup = df.groupby('setup').agg(
    trades   = ('pnl_r', 'count'),
    wins     = ('win_loss', lambda x: (x == 'WIN').sum()),
    total_r  = ('pnl_r', 'sum'),
    avg_r    = ('pnl_r', 'mean'),
    total_d  = ('pnl_dollars', 'sum'),
).sort_values('total_r', ascending=False)
by_setup['wr_pct'] = (by_setup['wins'] / by_setup['trades'] * 100).round(1)

print(f"\n  {'Setup':<22} {'N':<5} {'W/L':<8} {'WR%':<7} {'Total R':<10} {'Avg R':<10} {'Total $'}")
print(f"  {'─'*22} {'─'*5} {'─'*8} {'─'*7} {'─'*10} {'─'*10} {'─'*10}")
for s, row in by_setup.iterrows():
    wl = f"{int(row['wins'])}W/{int(row['trades'] - row['wins'])}L"
    print(f"  {s:<22} {int(row['trades']):<5} {wl:<8} {row['wr_pct']:<7} "
          f"{row['total_r']:+.2f}R{'':>4} {row['avg_r']:+.3f}R{'':>3} ${row['total_d']:+,.2f}")

# ── By symbol ─────────────────────────────────────────────────
print(f"\n{'─'*70}")
print("  BY SYMBOL")
print("─"*70)
by_sym = df.groupby('symbol').agg(
    trades  = ('pnl_r', 'count'),
    wins    = ('win_loss', lambda x: (x == 'WIN').sum()),
    total_r = ('pnl_r', 'sum'),
    avg_r   = ('pnl_r', 'mean'),
    total_d = ('pnl_dollars', 'sum'),
).sort_values('total_r', ascending=False)
by_sym['wr_pct'] = (by_sym['wins'] / by_sym['trades'] * 100).round(1)

print(f"\n  {'Symbol':<10} {'N':<5} {'W/L':<8} {'WR%':<7} {'Total R':<10} {'Avg R':<10} {'Total $'}")
print(f"  {'─'*10} {'─'*5} {'─'*8} {'─'*7} {'─'*10} {'─'*10} {'─'*10}")
for s, row in by_sym.iterrows():
    wl = f"{int(row['wins'])}W/{int(row['trades'] - row['wins'])}L"
    print(f"  {s:<10} {int(row['trades']):<5} {wl:<8} {row['wr_pct']:<7} "
          f"{row['total_r']:+.2f}R{'':>4} {row['avg_r']:+.3f}R{'':>3} ${row['total_d']:+,.2f}")

# ── Exit reason breakdown ─────────────────────────────────────
print(f"\n{'─'*70}")
print("  EXIT REASON BREAKDOWN")
print("─"*70)
by_exit = df.groupby('exit_bucket').agg(
    count   = ('pnl_r', 'count'),
    wins    = ('win_loss', lambda x: (x == 'WIN').sum()),
    total_r = ('pnl_r', 'sum'),
    avg_r   = ('pnl_r', 'mean'),
).sort_values('count', ascending=False)
by_exit['wr_pct'] = (by_exit['wins'] / by_exit['count'] * 100).round(1)

print(f"\n  {'Exit Type':<18} {'Count':<8} {'WR%':<8} {'Total R':<10} {'Avg R'}")
print(f"  {'─'*18} {'─'*8} {'─'*8} {'─'*10} {'─'*8}")
for ex, row in by_exit.iterrows():
    print(f"  {ex:<18} {int(row['count']):<8} {row['wr_pct']:<8} "
          f"{row['total_r']:+.2f}R{'':>4} {row['avg_r']:+.3f}R")

# ── By entry regime ───────────────────────────────────────────
print(f"\n{'─'*70}")
print("  BY ENTRY REGIME")
print("─"*70)
by_regime = df.groupby('regime').agg(
    count   = ('pnl_r', 'count'),
    wins    = ('win_loss', lambda x: (x == 'WIN').sum()),
    total_r = ('pnl_r', 'sum'),
    avg_r   = ('pnl_r', 'mean'),
).sort_values('total_r', ascending=False)
by_regime['wr_pct'] = (by_regime['wins'] / by_regime['count'] * 100).round(1)

print(f"\n  {'Regime':<22} {'Count':<8} {'WR%':<8} {'Total R':<10} {'Avg R'}")
print(f"  {'─'*22} {'─'*8} {'─'*8} {'─'*10} {'─'*8}")
for reg_name, row in by_regime.iterrows():
    print(f"  {reg_name:<22} {int(row['count']):<8} {row['wr_pct']:<8} "
          f"{row['total_r']:+.2f}R{'':>4} {row['avg_r']:+.3f}R")

print("\n" + "="*90 + "\n")
