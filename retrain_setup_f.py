"""
retrain_setup_f.py
==================
Retrain Setup F Random Forest on ALL available 5min data.
5-fold TimeSeriesSplit CV — report OOS AUC per fold and mean.

Promotion logic:
  mean OOS AUC >= 0.56 → save to ml_models + enable strategy_config (live)
  mean OOS AUC 0.54-0.56 → save to ml_models (shadow lab, paper only)
  mean OOS AUC < 0.54 → no save, report only
"""

import os
import sys
import pickle
import logging
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from datetime import datetime, timezone

sys.path.insert(0, '/Users/stanleywise/Desktop/apex')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('RetainSetupF')

DATABASE_URL = os.environ['DATABASE_URL']
AUC_LIVE_GATE   = 0.56
AUC_SHADOW_GATE = 0.54
N_FOLDS = 5


def get_conn():
    return psycopg2.connect(DATABASE_URL)


def load_ohlcv_pg(symbol: str, timeframe: str = '5min') -> pd.DataFrame:
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            'SELECT ts, open, high, low, close, volume FROM ohlcv '
            'WHERE symbol=%s AND timeframe=%s ORDER BY ts ASC',
            (symbol, timeframe)
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    for col in ['ts', 'open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['dt'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    return df.reset_index(drop=True)


def save_model_to_db(model_name: str, model, metadata: dict):
    payload      = {'model': model, 'metadata': metadata}
    model_bytes  = pickle.dumps(payload)
    trained_at   = metadata.get('trained_at', datetime.now(timezone.utc).isoformat())
    oos_auc      = float(metadata.get('mean_oos_auc', 0))
    feature_count = int(metadata.get('n_features', 0))
    n_samples    = int(metadata.get('n_samples', 0))

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute('SELECT id FROM ml_models WHERE model_name=%s', (model_name,))
        row = cur.fetchone()
        if row:
            cur.execute(
                'UPDATE ml_models SET model_bytes=%s, trained_at=%s, oos_auc=%s, '
                'feature_count=%s, training_samples=%s WHERE model_name=%s',
                (psycopg2.Binary(model_bytes), trained_at, oos_auc,
                 feature_count, n_samples, model_name)
            )
            logger.info(f'ml_models: updated "{model_name}" ({len(model_bytes):,} bytes)')
        else:
            cur.execute(
                'INSERT INTO ml_models (model_name, model_bytes, trained_at, oos_auc, '
                'feature_count, training_samples) VALUES (%s, %s, %s, %s, %s, %s)',
                (model_name, psycopg2.Binary(model_bytes), trained_at, oos_auc,
                 feature_count, n_samples)
            )
            logger.info(f'ml_models: inserted "{model_name}" ({len(model_bytes):,} bytes)')
        conn.commit()
    finally:
        conn.close()


def update_strategy_config(live: bool, shadow_only: bool):
    conn = get_conn()
    try:
        cur = conn.cursor()
        if live:
            cur.execute(
                "UPDATE strategy_config SET enabled=TRUE, disabled_reason=NULL, "
                "disabled_at=NULL, enabled_at=NOW(), updated_by='retrain_setup_f', "
                "paper_instruments='' WHERE setup_id='F'"
            )
            logger.info("strategy_config: Setup F enabled=TRUE (live both instruments)")
        elif shadow_only:
            cur.execute(
                "UPDATE strategy_config SET enabled=FALSE, "
                "disabled_reason='shadow lab — AUC 0.54-0.56 awaiting more data', "
                "updated_by='retrain_setup_f' WHERE setup_id='F'"
            )
            logger.info("strategy_config: Setup F shadow lab (kept disabled, model saved to DB)")
        conn.commit()
    finally:
        conn.close()


def retrain(symbol: str):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score
    from setup_f_ml import calculate_features, _build_labels, FEATURE_NAMES, SESSION_WINDOWS

    print(f'\n{"="*60}')
    print(f'SETUP F RETRAIN — {symbol}')
    print(f'{"="*60}')

    df_5m = load_ohlcv_pg(symbol, '5min')
    if df_5m.empty or len(df_5m) < 500:
        print(f'INSUFFICIENT DATA: {len(df_5m)} bars — abort')
        return None, None

    start_hr, end_hr = SESSION_WINDOWS.get(symbol, (13, 19))
    months = (df_5m['dt'].iloc[-1] - df_5m['dt'].iloc[0]).days / 30.44

    print(f'Loaded {len(df_5m):,} 5min bars')
    print(f'Date range: {df_5m["dt"].iloc[0].date()} → {df_5m["dt"].iloc[-1].date()}')
    print(f'~{months:.1f} months of data')

    session_mask = df_5m['dt'].dt.hour.between(start_hr, end_hr - 1).values
    print(f'Session bars (UTC {start_hr}–{end_hr}): {session_mask.sum():,}')

    print('Computing features...')
    X_all = calculate_features(symbol, df_5m)

    print('Computing outcome labels (RR=2.5, stop=1.5×ATR, max 100 bars)...')
    y_all = _build_labels(df_5m, rr=2.5, stop_atr=1.5, max_bars=100)

    valid_mask = (
        ~np.isnan(X_all).any(axis=1) &
        ~np.isnan(y_all) &
        session_mask
    )
    X = X_all[valid_mask]
    y = y_all[valid_mask].astype(int)

    print(f'Training samples: {len(X):,} | Win rate: {y.mean():.1%}')

    if len(X) < 200:
        print(f'INSUFFICIENT SAMPLES: {len(X)} (need 200) — abort')
        return None, None

    # ── 5-fold TimeSeriesSplit CV ─────────────────────────────────
    tscv = TimeSeriesSplit(n_splits=N_FOLDS)
    fold_aucs = []

    print(f'\n5-Fold TimeSeriesSplit CV:')
    print(f'{"Fold":<6} {"Train":<8} {"Test":<8} {"AUC":<8} {"Prec@0.65":<12} {"Signals":<8}')
    print('-' * 52)

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        if len(X_te) < 30 or y_te.sum() < 5:
            print(f'{fold:<6} {len(X_tr):<8} {len(X_te):<8} SKIP (insufficient test set)')
            continue

        clf = RandomForestClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=20,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_te)[:, 1]

        try:
            auc = float(roc_auc_score(y_te, probs))
        except Exception:
            auc = 0.5

        sig_mask = probs > 0.65
        prec = float(y_te[sig_mask].mean()) if sig_mask.sum() > 0 else 0.0
        fold_aucs.append(auc)
        print(f'{fold:<6} {len(X_tr):<8} {len(X_te):<8} {auc:<8.3f} {prec:<12.1%} {sig_mask.sum():<8}')

    if not fold_aucs:
        print('No valid folds — abort')
        return None, None

    mean_auc = float(np.mean(fold_aucs))
    std_auc  = float(np.std(fold_aucs))
    print(f'\nMean OOS AUC: {mean_auc:.3f} ± {std_auc:.3f}')

    # ── Promotion verdict ─────────────────────────────────────────
    if mean_auc >= AUC_LIVE_GATE:
        verdict = 'LIVE'
    elif mean_auc >= AUC_SHADOW_GATE:
        verdict = 'SHADOW LAB'
    else:
        verdict = 'NO DEPLOY'

    print(f'Verdict: {verdict}')

    if mean_auc < AUC_SHADOW_GATE:
        print(f'AUC {mean_auc:.3f} < {AUC_SHADOW_GATE} — model not saved')
        return mean_auc, verdict

    # ── Train final model on all data ─────────────────────────────
    print('\nTraining final model on full dataset...')
    final_clf = RandomForestClassifier(
        n_estimators=300, max_depth=4, min_samples_leaf=20,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    final_clf.fit(X, y)
    train_acc = final_clf.score(X, y)
    print(f'In-sample accuracy: {train_acc:.3f}')

    print('\nFeature importances:')
    for name, imp in sorted(
        zip(FEATURE_NAMES, final_clf.feature_importances_),
        key=lambda x: x[1], reverse=True
    ):
        bar = '█' * int(imp * 40)
        print(f'  {name:<15} {imp:.1%}  {bar}')

    # Save to ml_models
    model_name = f'apex_f_{symbol}'
    metadata = {
        'symbol':       symbol,
        'trained_at':   datetime.now(timezone.utc).isoformat(),
        'mean_oos_auc': mean_auc,
        'std_oos_auc':  std_auc,
        'fold_aucs':    fold_aucs,
        'n_folds':      N_FOLDS,
        'n_samples':    int(len(X)),
        'n_features':   len(FEATURE_NAMES),
        'features':     FEATURE_NAMES,
        'verdict':      verdict,
        'train_acc':    float(train_acc),
    }
    save_model_to_db(model_name, final_clf, metadata)

    # Also save local pkl for fast load
    pkl_path = f'apex_rf_{symbol}.pkl'
    with open(pkl_path, 'wb') as fh:
        pickle.dump({
            'model':      final_clf,
            'trained_at': metadata['trained_at'],
            'symbol':     symbol,
            'accuracy':   train_acc,
            'oos_auc':    mean_auc,
            'n_features': len(FEATURE_NAMES),
            'features':   FEATURE_NAMES,
        }, fh)
    print(f'Local pkl saved: {pkl_path}')

    return mean_auc, verdict


if __name__ == '__main__':
    results = {}
    for sym in ('MNQ', 'ES'):
        auc, verdict = retrain(sym) or (None, None)
        results[sym] = (auc, verdict)

    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    any_live = False
    any_shadow = False
    for sym, (auc, verdict) in results.items():
        auc_str = f'{auc:.3f}' if auc is not None else 'N/A'
        print(f'  {sym}: AUC={auc_str} → {verdict}')
        if verdict == 'LIVE':
            any_live = True
        elif verdict == 'SHADOW LAB':
            any_shadow = True

    if any_live:
        update_strategy_config(live=True, shadow_only=False)
        print('\nSetup F ENABLED for live trading on both instruments.')
        print('NOTE: server.py Setup F scan loop already wired — no server changes needed.')
        print('NOTE: update setup_f_ml.py load_or_train_model() to load from DB on Railway.')
    elif any_shadow:
        update_strategy_config(live=False, shadow_only=True)
        print('\nSetup F in shadow lab — model saved to ml_models, kept disabled in strategy_config.')
    else:
        print('\nSetup F not deployed — AUC below minimum threshold.')
