"""
Meridian Layer 3 — Regime Transition Prediction
Predicts probability of TRENDING regime in next 30 minutes.
Uses GradientBoostingClassifier trained on regime_log history.
OOS AUC: 0.91 | Deploy mode: full (0.75x-1.5x multiplier range)
"""

import pickle
import logging
import numpy as np
from datetime import datetime, timezone

import db as _db

logger = logging.getLogger(__name__)

# symbol → dict with model/scaler/features/auc
_l3_cache: dict = {}

FEATURES = [
    'hurst', 'autocorr', 'vol_ratio', 'confidence',
    'hurst_change_3', 'hurst_change_6',
    'autocorr_change_3', 'autocorr_change_6',
    'vol_change_3', 'hurst_ma6', 'hurst_ma12',
    'autocorr_ma6', 'vol_ma6', 'hurst_vs_ma',
    'consecutive_choppy', 'currently_choppy',
    'currently_trending', 'hour', 'dow', 'session_progress',
    'high_autocorr', 'high_hurst_autocorr', 'autocorr_x_hurst', 'deep_choppy'
]


def load_model(symbol: str = 'MNQ') -> bool:
    model_name = f'meridian_l3_{symbol}'
    try:
        conn = _db.connect()
        row = conn.execute(
            'SELECT model_bytes, oos_auc, trained_at FROM ml_models '
            'WHERE model_name = ?', (model_name,)
        ).fetchone()
        conn.close()
        if not row:
            logger.warning(f'Meridian L3: no model found for {model_name}')
            return False
        data = pickle.loads(bytes(row[0]))
        _l3_cache[symbol] = {
            'model':    data['model'],
            'scaler':   data['scaler'],
            'features': data.get('features', FEATURES),
            'auc':      float(row[1] or 0),
            'mode':     data.get('deploy_mode', 'full'),
            'trained_at': row[2],
        }
        logger.info(
            f'Meridian L3: loaded {model_name} '
            f'(AUC={row[1]:.4f} mode={data.get("deploy_mode","?")} trained={row[2]})'
        )
        return True
    except Exception as e:
        logger.warning(f'Meridian L3: model load failed ({model_name}) — {e}')
        return False


def get_trending_probability(symbol: str = 'MNQ') -> float:
    """
    Returns probability (0.0–1.0) that symbol will be in TRENDING regime
    in 30 minutes. Returns 0.5 (neutral) if model unavailable.
    """
    if symbol not in _l3_cache:
        if not load_model(symbol):
            return 0.5

    try:
        conn = _db.connect()
        rows = conn.execute(
            'SELECT hurst, autocorr, vol_ratio, confidence, '
            'CAST(strftime(\'%s\', timestamp) AS INTEGER) as ts, '
            'CAST(strftime(\'%H\', timestamp) AS INTEGER) as hour, '
            'CAST(strftime(\'%w\', timestamp) AS INTEGER) as dow, '
            'regime '
            'FROM regime_log WHERE symbol = ? '
            'ORDER BY timestamp DESC LIMIT 24',
            (symbol,)
        ).fetchall()
        conn.close()

        if len(rows) < 12:
            return 0.5

        # rows come back newest-first — reverse to chronological
        rows = list(reversed(rows))
        h_vals = [float(r[0] or 0) for r in rows]
        a_vals = [float(r[1] or 0) for r in rows]
        v_vals = [float(r[2] or 0) for r in rows]
        c_vals = [float(r[3] or 0) for r in rows]
        regimes = [r[7] for r in rows]
        latest_hour = float(rows[-1][5] or 0)
        latest_dow  = float(rows[-1][6] or 0)

        def _safe(arr, idx):
            return arr[idx] if 0 <= idx < len(arr) else arr[-1]

        n = len(rows)
        consecutive_choppy = sum(1 for r in regimes[-6:] if r == 'CHOPPY')

        features = {
            'hurst':             h_vals[-1],
            'autocorr':          a_vals[-1],
            'vol_ratio':         v_vals[-1],
            'confidence':        c_vals[-1],
            'hurst_change_3':    h_vals[-1] - _safe(h_vals, n-4),
            'hurst_change_6':    h_vals[-1] - _safe(h_vals, n-7),
            'autocorr_change_3': a_vals[-1] - _safe(a_vals, n-4),
            'autocorr_change_6': a_vals[-1] - _safe(a_vals, n-7),
            'vol_change_3':      v_vals[-1] - _safe(v_vals, n-4),
            'hurst_ma6':         float(np.mean(h_vals[-6:])),
            'hurst_ma12':        float(np.mean(h_vals[-12:])) if n >= 12 else h_vals[-1],
            'autocorr_ma6':      float(np.mean(a_vals[-6:])),
            'vol_ma6':           float(np.mean(v_vals[-6:])),
            'hurst_vs_ma':       h_vals[-1] - (float(np.mean(h_vals[-12:])) if n >= 12 else h_vals[-1]),
            'consecutive_choppy': float(consecutive_choppy),
            'currently_choppy':  1.0 if regimes[-1] == 'CHOPPY' else 0.0,
            'currently_trending': 1.0 if regimes[-1] == 'TRENDING' else 0.0,
            'hour':              latest_hour,
            'dow':               latest_dow,
            'session_progress':  (latest_hour - 13) / 6.0 if 13 <= latest_hour < 19 else 0.5,
            'high_autocorr':     1.0 if a_vals[-1] > 0.10 else 0.0,
            'high_hurst_autocorr': 1.0 if (h_vals[-1] > 0.65 and a_vals[-1] > 0.10) else 0.0,
            'autocorr_x_hurst':  a_vals[-1] * h_vals[-1],
            'deep_choppy':       1.0 if consecutive_choppy >= 5 else 0.0,
        }

        cache = _l3_cache[symbol]
        feat_vec = np.array([features[f] for f in cache['features']], dtype=np.float64).reshape(1, -1)
        feat_scaled = cache['scaler'].transform(feat_vec)
        prob = float(cache['model'].predict_proba(feat_scaled)[0][1])
        return prob

    except Exception as e:
        logger.warning(f'Meridian L3: prediction failed for {symbol} — {e}')
        return 0.5


def get_trending_probability_pg(symbol: str = 'MNQ') -> float:
    """PostgreSQL-aware variant used when IS_POSTGRES=True (Railway)."""
    if symbol not in _l3_cache:
        if not load_model(symbol):
            return 0.5

    try:
        import psycopg2
        import os
        DB_URL = os.environ.get('DATABASE_URL', '').strip()
        pg = psycopg2.connect(DB_URL)
        cur = pg.cursor()
        cur.execute("""
            SELECT hurst::float, autocorr::float, vol_ratio::float, confidence::float,
                   EXTRACT(HOUR FROM timestamp)::int,
                   EXTRACT(DOW FROM timestamp)::int,
                   regime
            FROM regime_log WHERE symbol = %s
            ORDER BY timestamp DESC LIMIT 24
        """, (symbol,))
        rows = cur.fetchall()
        pg.close()

        if len(rows) < 12:
            return 0.5

        rows = list(reversed(rows))
        h_vals = [float(r[0] or 0) for r in rows]
        a_vals = [float(r[1] or 0) for r in rows]
        v_vals = [float(r[2] or 0) for r in rows]
        c_vals = [float(r[3] or 0) for r in rows]
        latest_hour = float(rows[-1][4] or 0)
        latest_dow  = float(rows[-1][5] or 0)
        regimes = [r[6] for r in rows]

        def _safe(arr, idx):
            return arr[idx] if 0 <= idx < len(arr) else arr[-1]

        n = len(rows)
        consecutive_choppy = sum(1 for r in regimes[-6:] if r == 'CHOPPY')

        features = {
            'hurst':             h_vals[-1],
            'autocorr':          a_vals[-1],
            'vol_ratio':         v_vals[-1],
            'confidence':        c_vals[-1],
            'hurst_change_3':    h_vals[-1] - _safe(h_vals, n-4),
            'hurst_change_6':    h_vals[-1] - _safe(h_vals, n-7),
            'autocorr_change_3': a_vals[-1] - _safe(a_vals, n-4),
            'autocorr_change_6': a_vals[-1] - _safe(a_vals, n-7),
            'vol_change_3':      v_vals[-1] - _safe(v_vals, n-4),
            'hurst_ma6':         float(np.mean(h_vals[-6:])),
            'hurst_ma12':        float(np.mean(h_vals[-12:])) if n >= 12 else h_vals[-1],
            'autocorr_ma6':      float(np.mean(a_vals[-6:])),
            'vol_ma6':           float(np.mean(v_vals[-6:])),
            'hurst_vs_ma':       h_vals[-1] - (float(np.mean(h_vals[-12:])) if n >= 12 else h_vals[-1]),
            'consecutive_choppy': float(consecutive_choppy),
            'currently_choppy':  1.0 if regimes[-1] == 'CHOPPY' else 0.0,
            'currently_trending': 1.0 if regimes[-1] == 'TRENDING' else 0.0,
            'hour':              latest_hour,
            'dow':               latest_dow,
            'session_progress':  (latest_hour - 13) / 6.0 if 13 <= latest_hour < 19 else 0.5,
            'high_autocorr':     1.0 if a_vals[-1] > 0.10 else 0.0,
            'high_hurst_autocorr': 1.0 if (h_vals[-1] > 0.65 and a_vals[-1] > 0.10) else 0.0,
            'autocorr_x_hurst':  a_vals[-1] * h_vals[-1],
            'deep_choppy':       1.0 if consecutive_choppy >= 5 else 0.0,
        }

        cache = _l3_cache[symbol]
        feat_vec = np.array([features[f] for f in cache['features']], dtype=np.float64).reshape(1, -1)
        feat_scaled = cache['scaler'].transform(feat_vec)
        prob = float(cache['model'].predict_proba(feat_scaled)[0][1])
        return prob

    except Exception as e:
        logger.warning(f'Meridian L3 (PG): prediction failed for {symbol} — {e}')
        return 0.5


def predict(symbol: str = 'MNQ') -> float:
    """Dispatch to correct DB variant based on IS_POSTGRES flag."""
    try:
        if _db.IS_POSTGRES:
            return get_trending_probability_pg(symbol)
        return get_trending_probability(symbol)
    except Exception as e:
        logger.warning(f'Meridian L3 predict: {e}')
        return 0.5


def get_position_multiplier(symbol: str = 'MNQ',
                            current_hurst: float = 0.5,
                            current_conf: float = 0.5) -> tuple:
    """
    Returns (multiplier, l3_prob) based on L3 prediction.

    Deploy mode: full
      l3_prob >= 0.65 + high quality → 1.5×
      l3_prob >= 0.50 + high quality → 1.25×
      l3_prob < 0.30                 → 0.75×
      otherwise                      → 1.0×

    High quality = Hurst >= 0.70 AND confidence >= 0.67
    """
    cache = _l3_cache.get(symbol, {})
    mode = cache.get('mode', 'full')

    l3_prob = predict(symbol)
    high_quality = current_hurst >= 0.70 and current_conf >= 0.67

    if mode == 'observation':
        return (1.0, l3_prob)

    if mode == 'conservative':
        if l3_prob >= 0.50 and high_quality:
            mult = 1.25
        elif l3_prob < 0.35:
            mult = 0.90
        else:
            mult = 1.0
        return (mult, l3_prob)

    # full mode
    if l3_prob >= 0.65 and high_quality:
        mult = 1.5
    elif l3_prob >= 0.50 and high_quality:
        mult = 1.25
    elif l3_prob < 0.30:
        mult = 0.75
    else:
        mult = 1.0

    return (mult, l3_prob)


def retrain(symbol: str = 'MNQ') -> bool:
    """
    Weekly retrain — pulls latest regime_log data and overwrites ml_models.
    Called by background_scheduler on Sunday 02:00 UTC.
    Returns True on success.
    """
    logger.info(f'Meridian L3: starting weekly retrain for {symbol}')
    try:
        import os
        import psycopg2
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import TimeSeriesSplit
        import pandas as pd

        DB_URL = os.environ.get('DATABASE_URL', '').strip()
        if not DB_URL:
            logger.warning('Meridian L3 retrain: no DATABASE_URL — skipping')
            return False

        pg = psycopg2.connect(DB_URL)
        cur = pg.cursor()
        cur.execute("""
            SELECT regime, hurst::float, autocorr::float, vol_ratio::float,
                   confidence::float,
                   EXTRACT(HOUR FROM timestamp)::int as hour,
                   EXTRACT(DOW FROM timestamp)::int as dow
            FROM regime_log WHERE symbol = %s ORDER BY timestamp ASC
        """, (symbol,))
        rows = cur.fetchall()

        df = pd.DataFrame(rows, columns=[
            'regime','hurst','autocorr','vol_ratio','confidence','hour','dow'
        ])
        for c in ['hurst','autocorr','vol_ratio','confidence','hour','dow']:
            df[c] = df[c].astype(float)

        df['hurst_change_3'] = df['hurst'] - df['hurst'].shift(3)
        df['hurst_change_6'] = df['hurst'] - df['hurst'].shift(6)
        df['autocorr_change_3'] = df['autocorr'] - df['autocorr'].shift(3)
        df['autocorr_change_6'] = df['autocorr'] - df['autocorr'].shift(6)
        df['vol_change_3'] = df['vol_ratio'] - df['vol_ratio'].shift(3)
        df['hurst_ma6'] = df['hurst'].rolling(6).mean()
        df['hurst_ma12'] = df['hurst'].rolling(12).mean()
        df['autocorr_ma6'] = df['autocorr'].rolling(6).mean()
        df['vol_ma6'] = df['vol_ratio'].rolling(6).mean()
        df['hurst_vs_ma'] = df['hurst'] - df['hurst_ma12']
        df['consecutive_choppy'] = (df['regime'] == 'CHOPPY').astype(float).rolling(6).sum()
        df['currently_choppy'] = (df['regime'] == 'CHOPPY').astype(float)
        df['currently_trending'] = (df['regime'] == 'TRENDING').astype(float)
        df['session_progress'] = np.where(
            (df['hour'] >= 13) & (df['hour'] < 19), (df['hour'] - 13) / 6.0, 0.5
        )
        df['high_autocorr'] = (df['autocorr'] > 0.10).astype(float)
        df['high_hurst_autocorr'] = ((df['hurst'] > 0.65) & (df['autocorr'] > 0.10)).astype(float)
        df['autocorr_x_hurst'] = df['autocorr'] * df['hurst']
        df['deep_choppy'] = (df['consecutive_choppy'] >= 5).astype(float)
        df['target'] = (df['regime'].shift(-6) == 'TRENDING').astype(float)
        df = df.dropna()

        X = df[FEATURES].values.astype(np.float64)
        y = df['target'].values.astype(np.int32)

        tscv = TimeSeriesSplit(n_splits=5)
        aucs = []
        for _, (tr, te) in enumerate(tscv.split(X)):
            sc = StandardScaler()
            m = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                learning_rate=0.05, subsample=0.8, random_state=42)
            m.fit(sc.fit_transform(X[tr]), y[tr])
            aucs.append(roc_auc_score(y[te], m.predict_proba(sc.transform(X[te]))[:,1]))
        mean_auc = float(np.mean(aucs))

        mode = 'full' if mean_auc >= 0.56 else ('conservative' if mean_auc >= 0.54 else 'observation')

        scaler_f = StandardScaler()
        X_sc = scaler_f.fit_transform(X)
        model_f = GradientBoostingClassifier(n_estimators=200, max_depth=3,
            learning_rate=0.05, subsample=0.8, random_state=42)
        model_f.fit(X_sc, y)

        trained_at = datetime.now(timezone.utc)
        mb = pickle.dumps({
            'model': model_f, 'scaler': scaler_f, 'features': FEATURES,
            'mean_auc': mean_auc, 'deploy_mode': mode,
            'trained_at': trained_at.isoformat()
        })
        cur.execute("""
            INSERT INTO ml_models
                (model_name, model_bytes, oos_auc, feature_count, training_samples, trained_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (model_name) DO UPDATE SET
                model_bytes = EXCLUDED.model_bytes,
                oos_auc = EXCLUDED.oos_auc,
                feature_count = EXCLUDED.feature_count,
                training_samples = EXCLUDED.training_samples,
                trained_at = EXCLUDED.trained_at
        """, (f'meridian_l3_{symbol}', psycopg2.Binary(mb),
              mean_auc, len(FEATURES), len(X), trained_at.isoformat()))
        pg.commit()
        pg.close()

        # Reload into cache
        _l3_cache.pop(symbol, None)
        load_model(symbol)

        logger.info(f'Meridian L3: retrain complete — {symbol} AUC={mean_auc:.4f} mode={mode} n={len(X):,}')
        return True

    except Exception as e:
        logger.error(f'Meridian L3: retrain failed for {symbol} — {e}', exc_info=True)
        return False
