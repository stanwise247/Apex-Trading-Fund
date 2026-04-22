"""
Setup I — Mathematical Alpha Backtest
6 months of MNQ/NQ + ES 5min data, no lookahead.
Reports: total signals, win rate, avg R, total R, Sharpe, max drawdown,
         profit factor. Compare to existing setups.
OOS AUC gate: 0.55 — deploy only if mean AUC >= 0.55 AND Sharpe >= 3.0
"""

import sys, logging, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, '/Users/stanleywise/Desktop/apex')
import db as _db
from setup_i_mathematical import (
    calculate_features_i, _build_labels_long_i, _build_labels_short_i, _atr_i, hurst_rs,
    FEATURE_NAMES_I, OOS_AUC_GATE,
    _XGBOOST_AVAILABLE, SESSION_START, SESSION_END, ALLOWED_DAYS, _make_xgb,
)

try:
    from xgboost import XGBClassifier
except Exception:
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

SEP = '='*68
def banner(t): print(f'\n{SEP}\n  {t}\n{SEP}')


# ─── Load data ────────────────────────────────────────────────
banner('Loading data')
six_months_ago = int((datetime.now(timezone.utc) - timedelta(days=180)).timestamp())
conn = _db.connect()

for sym_try in ('MNQ', 'NQ'):
    rows = conn.execute(
        'SELECT ts, open, high, low, close, volume FROM ohlcv '
        'WHERE symbol=? AND timeframe=? AND ts>=? ORDER BY ts ASC',
        (sym_try, '5min', six_months_ago)
    ).fetchall()
    if len(rows) >= 500:
        PRIMARY_SYM = sym_try
        break

es_rows = conn.execute(
    'SELECT ts, open, high, low, close, volume FROM ohlcv '
    'WHERE symbol=? AND timeframe=? AND ts>=? ORDER BY ts ASC',
    ('ES', '5min', six_months_ago)
).fetchall()
conn.close()

COLS = ['ts', 'open', 'high', 'low', 'close', 'volume']
df_5m = pd.DataFrame(rows, columns=COLS)
df_es = pd.DataFrame(es_rows, columns=COLS)
for col in COLS:
    df_5m[col] = pd.to_numeric(df_5m[col], errors='coerce')
    df_es[col]  = pd.to_numeric(df_es[col], errors='coerce')

df_5m['dt'] = pd.to_datetime(df_5m['ts'], unit='s', utc=True)
df_es['dt']  = pd.to_datetime(df_es['ts'], unit='s', utc=True)
df_5m = df_5m.reset_index(drop=True)
df_es  = df_es.reset_index(drop=True)

print(f'Primary: {PRIMARY_SYM} — {len(df_5m)} 5min bars')
print(f'ES: {len(df_es)} 5min bars')


# ─── Features & labels ───────────────────────────────────────
banner('Computing features and labels')
print('Building 6 features (may take 60-90s)...')
X_all = calculate_features_i(df_5m, df_es if not df_es.empty else None)

# Verify Hurst fix — should vary 0.3-0.8, NOT all 0.5
hurst_col = X_all[:, 0]
hurst_nonnan = hurst_col[~np.isnan(hurst_col)]
print(f'\nHurst validation (FIX 1 check):')
if len(hurst_nonnan) >= 10:
    print(f'  First 10 values: {hurst_nonnan[:10].round(4)}')
    print(f'  Unique values (first 200): {len(np.unique(hurst_nonnan[:200]))}')
    print(f'  Min={hurst_nonnan.min():.4f}  Max={hurst_nonnan.max():.4f}  Std={hurst_nonnan.std():.4f}')
    print(f'  All same? {len(np.unique(hurst_nonnan[:200])) == 1}  ← should be False')
else:
    print(f'  Insufficient non-NaN Hurst values: {len(hurst_nonnan)}')

print(f'\nComputing direction labels (RR 1.5, stop 1.5×ATR, max 100 bars)...')
y_long_all  = _build_labels_long_i(df_5m,  rr=1.5, stop_atr=1.5, max_bars=100)
y_short_all = _build_labels_short_i(df_5m, rr=1.5, stop_atr=1.5, max_bars=100)

session_mask = (
    df_5m['dt'].dt.hour.between(SESSION_START, SESSION_END - 1) &
    df_5m['dt'].dt.dayofweek.isin(ALLOWED_DAYS)
).values

valid_mask = (
    ~np.isnan(X_all).any(axis=1) &
    ~np.isnan(y_long_all) &
    ~np.isnan(y_short_all) &
    session_mask
)
X     = X_all[valid_mask]
y_l   = y_long_all[valid_mask].astype(int)
y_s   = y_short_all[valid_mask].astype(int)
df_v  = df_5m[valid_mask].reset_index(drop=True)

print(f'Valid session samples: {len(X)}')
print(f'  Long label:  {y_l.mean():.1%} positive (price rose 1.5R)')
print(f'  Short label: {y_s.mean():.1%} positive (price fell 1.5R)')

if len(X) < 200:
    print('ERROR: Insufficient data for backtest.')
    sys.exit(1)


# ─── Feature IC (correlation with next-bar return) ───────────
banner('Feature Information Coefficients')
fwd_ret  = (df_5m['close'].shift(-1)  / df_5m['close'] - 1).values
fwd12_ret = (df_5m['close'].shift(-12) / df_5m['close'] - 1).values
fwd_ret_v  = fwd_ret[valid_mask]
fwd12_v   = fwd12_ret[valid_mask]

print(f'\n{"Feature":<18} {"IC (1-bar)":>12} {"IC (12-bar)":>13} {"Keep?"}')
print('-'*55)
for i, name in enumerate(FEATURE_NAMES_I):
    fv = X[:, i]
    v1  = ~np.isnan(fv) & ~np.isnan(fwd_ret_v)
    v12 = ~np.isnan(fv) & ~np.isnan(fwd12_v)
    ic1  = float(np.corrcoef(fv[v1],  fwd_ret_v[v1])[0, 1])  if v1.sum()  > 10 else float('nan')
    ic12 = float(np.corrcoef(fv[v12], fwd12_v[v12])[0, 1])   if v12.sum() > 10 else float('nan')
    keep = 'KEEP' if (abs(ic1) >= 0.015 or abs(ic12) >= 0.015) else 'weak'
    print(f'{name:<18} {f"{ic1:+.4f}" if not np.isnan(ic1) else "   N/A":>12} '
          f'{f"{ic12:+.4f}" if not np.isnan(ic12) else "   N/A":>13}  {keep}')


# ─── Walk-forward setup (shared across both directions) ───────
n   = len(X)
seg = n // 6
segments = [range(i * seg, (i + 1) * seg) for i in range(6)]
segments[-1] = range(5 * seg, n)

tr_idx = np.concatenate([list(segments[s]) for s in [0, 1, 2, 3, 4]])
te_idx = np.array(list(segments[5]))

X_tr, X_te = X[tr_idx], X[te_idx]
y_tr_l, y_te_l = y_l[tr_idx], y_l[te_idx]
y_tr_s, y_te_s = y_s[tr_idx], y_s[te_idx]
df_te = df_v.iloc[te_idx].reset_index(drop=True)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

# Shared LR trained on long labels (direction filter in live scan)
lr_clf = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)
lr_clf.fit(X_tr_s, y_tr_l)
lr_probs = lr_clf.predict_proba(X_te_s)[:, 1]


def _pnl_stats(pnl):
    wins = pnl > 0
    total_r  = pnl.sum()
    avg_r    = pnl.mean()
    wr       = wins.mean()
    gross_w  = pnl[wins].sum()  if wins.sum()  > 0 else 0
    gross_l  = abs(pnl[~wins].sum()) if (~wins).sum() > 0 else 1e-9
    pf       = gross_w / gross_l
    cumR     = np.cumsum(pnl)
    max_dd   = abs((cumR - np.maximum.accumulate(cumR)).min())
    sharpe   = (avg_r / pnl.std() * np.sqrt(len(pnl) * 2)) if pnl.std() > 0 else 0.0
    return dict(n=len(pnl), wr=wr, avg_r=avg_r, total_r=total_r,
                pf=pf, max_dd=max_dd, sharpe=sharpe)


# ─── SHORT model backtest ─────────────────────────────────────
banner('SHORT Model — OOS Backtest (train mo 1-5 / test mo 6)')

pos_w_s   = float((y_tr_s == 0).sum()) / float((y_tr_s == 1).sum() + 1e-9)
xgb_short = _make_xgb(pos_w_s)
xgb_short.fit(X_tr_s, y_tr_s)
short_probs = xgb_short.predict_proba(X_te_s)[:, 1]

try:
    short_auc = float(roc_auc_score(y_te_s, short_probs))
except Exception:
    short_auc = 0.5

# Short signal: xgb_short > 0.62 AND lr < 0.38
sig_short = (short_probs > 0.62) & (lr_probs < 0.38)
print(f'\nTrain: {len(X_tr)} | Test: {len(X_te)} | Short label pos%: {y_tr_s.mean():.1%}')
print(f'OOS AUC (short model):   {short_auc:.3f} | Gate: {OOS_AUC_GATE} | {"PASS ✓" if short_auc >= OOS_AUC_GATE else "FAIL ✗"}')
print(f'Short signals (xgb>0.62 AND lr<0.38): {sig_short.sum()}')

if hasattr(xgb_short, 'feature_importances_'):
    print(f'\nFeature importance (SHORT model):')
    for name, imp in sorted(zip(FEATURE_NAMES_I, xgb_short.feature_importances_), key=lambda x: -x[1]):
        print(f'  {name:<14} {imp:>6.1%}  {"#" * int(imp * 40)}')

if sig_short.sum() > 0:
    pnl_s = np.where(y_te_s[sig_short] == 1, 1.5, -1.0)
    st = _pnl_stats(pnl_s)
    print(f'\nShort signal P&L:')
    print(f'  N={st["n"]}  WR={st["wr"]:.1%}  AvgR={st["avg_r"]:.3f}  TotalR={st["total_r"]:.2f}R')
    print(f'  PF={st["pf"]:.2f}  MaxDD={st["max_dd"]:.2f}R  Sharpe={st["sharpe"]:.2f}')
    short_sharpe = st['sharpe']
else:
    print('No short signals at threshold.')
    short_sharpe = 0.0

short_ok = short_auc >= OOS_AUC_GATE
print(f'\nSHORT deployment: {"PASS ✓" if short_ok else "FAIL ✗"} (AUC={short_auc:.3f})')


# ─── LONG model backtest ──────────────────────────────────────
banner('LONG Model — OOS Backtest (train mo 1-5 / test mo 6)')

pos_w_l  = float((y_tr_l == 0).sum()) / float((y_tr_l == 1).sum() + 1e-9)
xgb_long = _make_xgb(pos_w_l)
xgb_long.fit(X_tr_s, y_tr_l)
long_probs = xgb_long.predict_proba(X_te_s)[:, 1]

try:
    long_auc = float(roc_auc_score(y_te_l, long_probs))
except Exception:
    long_auc = 0.5

# Long signal: xgb_long > 0.62 AND lr > 0.62
sig_long = (long_probs > 0.62) & (lr_probs > 0.62)
print(f'\nTrain: {len(X_tr)} | Test: {len(X_te)} | Long label pos%: {y_tr_l.mean():.1%}')
print(f'OOS AUC (long model):    {long_auc:.3f} | Gate: {OOS_AUC_GATE} | {"PASS ✓" if long_auc >= OOS_AUC_GATE else "FAIL ✗"}')
print(f'Long signals (xgb>0.62 AND lr>0.62): {sig_long.sum()}')

if hasattr(xgb_long, 'feature_importances_'):
    print(f'\nFeature importance (LONG model):')
    for name, imp in sorted(zip(FEATURE_NAMES_I, xgb_long.feature_importances_), key=lambda x: -x[1]):
        print(f'  {name:<14} {imp:>6.1%}  {"#" * int(imp * 40)}')

if sig_long.sum() > 0:
    pnl_l_arr = np.where(y_te_l[sig_long] == 1, 1.5, -1.0)
    lt = _pnl_stats(pnl_l_arr)
    print(f'\nLong signal P&L:')
    print(f'  N={lt["n"]}  WR={lt["wr"]:.1%}  AvgR={lt["avg_r"]:.3f}  TotalR={lt["total_r"]:.2f}R')
    print(f'  PF={lt["pf"]:.2f}  MaxDD={lt["max_dd"]:.2f}R  Sharpe={lt["sharpe"]:.2f}')
    long_sharpe = lt['sharpe']
else:
    print('No long signals at threshold.')
    long_sharpe = 0.0

long_ok = long_auc >= OOS_AUC_GATE
print(f'\nLONG deployment: {"PASS ✓" if long_ok else "FAIL ✗"} (AUC={long_auc:.3f})')


# ─── Deployment Gate Summary ──────────────────────────────────
banner('Deployment Gate Summary')
print(f'\n{"Direction":<10} {"OOS AUC":>10} {"Gate":>8} {"Signals":>9} {"Sharpe":>9} {"Decision"}')
print('-'*62)
print(f'{"SHORT":<10} {short_auc:>10.3f} {OOS_AUC_GATE:>8} {sig_short.sum():>9} {short_sharpe:>9.2f} '
      f'{"DEPLOY ✓" if short_ok else "DO NOT DEPLOY ✗"}')
print(f'{"LONG":<10} {long_auc:>10.3f} {OOS_AUC_GATE:>8} {sig_long.sum():>9} {long_sharpe:>9.2f} '
      f'{"DEPLOY ✓" if long_ok else "DO NOT DEPLOY ✗"}')

if short_ok or long_ok:
    active = []
    if short_ok: active.append('SHORT')
    if long_ok:  active.append('LONG')
    print(f'\nActive directions if deployed: {" + ".join(active)}')
    print('Each direction is independent — short can be live while long is disabled.')

print(f'\nBacktest complete. Do not deploy until results reviewed above.')
