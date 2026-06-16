"""
meridian_direction.py — Directional Price Prediction
=====================================================
Predicts probability that price will be HIGHER in N minutes.
Separate calibrated models per symbol (MNQ, ES) × horizon (30min, 1hr, 4hr).

Walk-forward validation: TimeSeriesSplit(n_splits=5).
Calibration: CalibratedClassifierCV(method='sigmoid', cv='prefit') on last fold.
AUC gate: 0.54 per horizon — below means 'insufficient_edge'.

Model key: 'meridian_dir_{symbol}_{horizon}' in ml_models table.
Weekly retrain: Sunday 03:00 UTC.
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List

import db as _db

logger = logging.getLogger('APEX.MeridianDir')

SYMBOLS      = ('MNQ', 'ES')
HORIZONS     = ('30min', '1hr', '4hr')
HORIZON_BARS = {'30min': 6, '1hr': 12, '4hr': 48}   # forward steps in 5min units
AUC_GATE     = 0.54

# in-memory cache: sym -> {horizon -> payload dict}
_dir_cache: dict = {}

FEATURE_NAMES = [
    # Regime state (from regime_log)
    'regime_trending', 'regime_choppy', 'regime_mr',
    'regime_conf', 'hurst', 'autocorr', 'vol_ratio',
    # Regime dynamics
    'hurst_chg3', 'autocorr_chg3', 'consec_choppy',
    # MTF EMA20 biases (+1 bullish / -1 bearish)
    'ema_1m', 'ema_5m', 'ema_15m', 'ema_30m', 'ema_1h', 'ema_4h',
    'mtf_bull_count',      # 0–6 count of bullish timeframes
    # Price position
    'vwap_dist_atr',       # (close – session_vwap) / ATR14, clipped ±5
    'pct_range_pdhl',      # (close – PDL) / (PDH – PDL), 0–1
    'above_pdh',           # 1 if close > prior-day high
    'below_pdl',           # 1 if close < prior-day low
    # Momentum
    'ret_5', 'ret_10', 'ret_20',
    # Volatility
    'atr_ratio', 'vol_ratio_20',
    # Time
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'session_progress',
]


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l = df['high'], df['low']
    pc = df['close'].shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def _ema20_bias_series(df_idx: pd.DataFrame, rule: str) -> pd.Series:
    """Resample close to rule, compute EMA20, return +1/-1 indexed at bar close times."""
    s = df_idx['close'].resample(rule).last().dropna()
    if len(s) < 5:
        return pd.Series(dtype=np.float64)
    ema = s.ewm(span=20, adjust=False).mean()
    return pd.Series(np.where(s.values > ema.values, 1.0, -1.0), index=s.index)


def _map_bias_to_5m(bias_s: pd.Series, df5_ts: pd.Series) -> np.ndarray:
    """merge_asof: map resampled bias series back to the 5min bar timeline."""
    if bias_s.empty:
        return np.zeros(len(df5_ts))
    b_df = bias_s.reset_index()
    b_df.columns = ['dt', 'bias']
    b_df['ts'] = (b_df['dt'].astype(np.int64) // 10**9).astype(np.int64)
    ref = pd.DataFrame({'ts': df5_ts.values})
    merged = pd.merge_asof(ref.sort_values('ts'), b_df[['ts', 'bias']].sort_values('ts'),
                           on='ts', direction='backward')
    # merge_asof reorders — restore original order
    merged.index = ref.sort_values('ts').index
    result = merged['bias'].reindex(range(len(df5_ts))).fillna(0.0)
    return result.values


# ─────────────────────────────────────────────────────────────────────────────
#  FULL FEATURE MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df5: pd.DataFrame, regime_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Build full feature matrix from 5min OHLCV + optional regime_log history.

    df5 must have columns: ts (unix int), open, high, low, close, volume, dt (UTC Timestamp).
    regime_df (optional) columns: ts_reg, regime, hurst, autocorr, vol_ratio, confidence.

    Returns DataFrame with FEATURE_NAMES columns, same index as df5.
    Rows needing more history than available will have NaN — caller drops them.
    """
    out = pd.DataFrame(index=df5.index, columns=FEATURE_NAMES, dtype=np.float64)

    # ── 1. Regime features ─────────────────────────────────────────────────
    if regime_df is not None and len(regime_df) >= 5:
        ref_ts = pd.DataFrame({'ts': df5['ts'].values}, index=df5.index)
        reg_sorted = regime_df.sort_values('ts_reg').rename(columns={'ts_reg': 'ts'})
        merged_r = pd.merge_asof(
            ref_ts.reset_index().sort_values('ts'),
            reg_sorted[['ts', 'regime', 'hurst', 'autocorr', 'vol_ratio', 'confidence']],
            on='ts', direction='backward'
        ).set_index('index').reindex(df5.index)

        out['regime_trending'] = (merged_r['regime'] == 'TRENDING').astype(float)
        out['regime_choppy']   = (merged_r['regime'] == 'CHOPPY').astype(float)
        out['regime_mr']       = (merged_r['regime'] == 'MEAN_REVERTING').astype(float)
        out['regime_conf']     = merged_r['confidence'].astype(float)
        out['hurst']           = merged_r['hurst'].astype(float)
        out['autocorr']        = merged_r['autocorr'].astype(float)
        out['vol_ratio']       = merged_r['vol_ratio'].astype(float)
        out['hurst_chg3']      = out['hurst']   - out['hurst'].shift(3)
        out['autocorr_chg3']   = out['autocorr'] - out['autocorr'].shift(3)
        out['consec_choppy']   = out['regime_choppy'].rolling(6, min_periods=1).sum()
    else:
        for col in ['regime_trending', 'regime_choppy', 'regime_mr', 'regime_conf',
                    'hurst', 'autocorr', 'vol_ratio', 'hurst_chg3', 'autocorr_chg3', 'consec_choppy']:
            out[col] = 0.5

    # ── 2. MTF EMA20 biases ────────────────────────────────────────────────
    df_idx = df5.set_index('dt')[['open', 'high', 'low', 'close', 'volume']]

    mtf_rules = [
        ('ema_1m',  '1min'),
        ('ema_5m',  '5min'),
        ('ema_15m', '15min'),
        ('ema_30m', '30min'),
        ('ema_1h',  '1h'),
        ('ema_4h',  '4h'),
    ]
    for col, rule in mtf_rules:
        try:
            bias_s = _ema20_bias_series(df_idx, rule)
            out[col] = _map_bias_to_5m(bias_s, df5['ts'])
        except Exception:
            out[col] = 0.0

    ema_cols = ['ema_1m', 'ema_5m', 'ema_15m', 'ema_30m', 'ema_1h', 'ema_4h']
    out['mtf_bull_count'] = (out[ema_cols] == 1.0).sum(axis=1).astype(float)

    # ── 3. VWAP distance ───────────────────────────────────────────────────
    df5c = df5.copy()
    df5c['date'] = df5c['dt'].dt.date
    df5c['tp']   = (df5c['high'] + df5c['low'] + df5c['close']) / 3.0
    df5c['tpv']  = df5c['tp'] * df5c['volume'].clip(lower=1)
    df5c['cum_tpv'] = df5c.groupby('date')['tpv'].cumsum()
    df5c['cum_vol'] = df5c.groupby('date')['volume'].transform('cumsum').clip(lower=1)
    df5c['vwap']    = df5c['cum_tpv'] / df5c['cum_vol']
    atr14 = _atr(df5, 14)
    out['vwap_dist_atr'] = ((df5c['close'] - df5c['vwap']) / (atr14 + 1e-9)).clip(-5, 5).values

    # ── 4. Prior-day high / low ────────────────────────────────────────────
    daily_hl = df5c.groupby('date').agg(day_h=('high', 'max'), day_l=('low', 'min'))
    daily_hl_shifted = daily_hl.shift(1)   # one prior trading day
    df5c['pdh'] = pd.to_datetime(df5c['date']).dt.date.map(daily_hl_shifted['day_h'])
    df5c['pdl'] = pd.to_datetime(df5c['date']).dt.date.map(daily_hl_shifted['day_l'])
    pdh = df5c['pdh'].values
    pdl = df5c['pdl'].values
    rng = np.where(pdh - pdl > 0, pdh - pdl, np.nan)
    out['pct_range_pdhl'] = np.clip((df5c['close'].values - pdl) / (rng + 1e-9), 0, 1)
    out['above_pdh']      = (df5c['close'].values > pdh).astype(float)
    out['below_pdl']      = (df5c['close'].values < pdl).astype(float)

    # ── 5. Momentum ────────────────────────────────────────────────────────
    lr = np.log(df5['close'] / df5['close'].shift(1))
    out['ret_5']  = lr.rolling(5,  min_periods=5).sum()
    out['ret_10'] = lr.rolling(10, min_periods=10).sum()
    out['ret_20'] = lr.rolling(20, min_periods=20).sum()

    # ── 6. Volatility ──────────────────────────────────────────────────────
    atr5  = _atr(df5,  5)
    atr20 = _atr(df5, 20)
    out['atr_ratio']    = (atr5 / (atr20 + 1e-9)).clip(0, 5)
    vol_ma              = df5['volume'].rolling(20, min_periods=1).mean()
    out['vol_ratio_20'] = (df5['volume'] / (vol_ma + 1e-9)).clip(0, 10)

    # ── 7. Time ────────────────────────────────────────────────────────────
    h = df5['dt'].dt.hour.astype(float)
    d = df5['dt'].dt.dayofweek.astype(float)
    out['hour_sin'] = np.sin(2 * np.pi * h / 24).values
    out['hour_cos'] = np.cos(2 * np.pi * h / 24).values
    out['dow_sin']  = np.sin(2 * np.pi * d / 7).values
    out['dow_cos']  = np.cos(2 * np.pi * d / 7).values
    out['session_progress'] = np.where((h >= 13) & (h < 20), (h - 13) / 7.0, 0.5)

    return out


def build_targets(df5: pd.DataFrame) -> pd.DataFrame:
    """Forward return sign targets (+1 up / 0 down) for each horizon."""
    closes = df5['close'].values
    n = len(closes)
    tgt = pd.DataFrame(index=df5.index)
    for name, bars in HORIZON_BARS.items():
        y = np.full(n, np.nan)
        for i in range(n - bars):
            y[i] = 1.0 if closes[i + bars] > closes[i] else 0.0
        tgt[name] = y
    return tgt


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray,
                       n_bins: int = 5) -> List[dict]:
    """
    Compute calibration table for a binary classifier.
    Returns list of dicts with keys: bin, count, mean_pred, actual_freq, gap.
    """
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        mp = float(y_prob[mask].mean())
        af = float(y_true[mask].mean())
        rows.append({
            'bin':         f'{lo:.2f}–{hi:.2f}',
            'count':       int(mask.sum()),
            'mean_pred':   round(mp, 3),
            'actual_freq': round(af, 3),
            'gap':         round(abs(mp - af), 3),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def _load_ohlcv_pg(symbol: str, pg) -> pd.DataFrame:
    cur = pg.cursor()
    cur.execute("""
        SELECT ts::bigint, open::float, high::float, low::float, close::float, volume::float
        FROM ohlcv WHERE symbol=%s AND timeframe='5min' ORDER BY ts ASC
    """, (symbol,))
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['dt'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    return df


def _load_regime_pg(symbol: str, pg) -> pd.DataFrame:
    cur = pg.cursor()
    cur.execute("""
        SELECT EXTRACT(EPOCH FROM timestamp)::bigint, regime, hurst::float,
               autocorr::float, vol_ratio::float, confidence::float
        FROM regime_log WHERE symbol=%s ORDER BY timestamp ASC
    """, (symbol,))
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=['ts_reg', 'regime', 'hurst', 'autocorr', 'vol_ratio', 'confidence'])


def train(symbol: str) -> Dict[str, Any]:
    """
    Train directional models for all horizons for one symbol.
    Connects directly to Railway PostgreSQL via DATABASE_URL.
    Returns report dict with AUC, calibration, and deployed status per horizon.
    """
    import psycopg2
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.calibration import CalibratedClassifierCV

    DB_URL = os.environ.get('DATABASE_URL', '').strip()
    if not DB_URL:
        logger.warning('meridian_direction: no DATABASE_URL — skipping training')
        return {}

    logger.info(f'meridian_direction: starting training for {symbol}')
    pg = psycopg2.connect(DB_URL)

    df5       = _load_ohlcv_pg(symbol, pg)
    regime_df = _load_regime_pg(symbol, pg)

    if df5.empty or len(df5) < 500:
        pg.close()
        logger.warning(f'meridian_direction: insufficient 5min bars for {symbol}: {len(df5)}')
        return {}

    # Restrict to session hours only (NY primary: 13:00–20:00 UTC, weekdays)
    session_mask = (
        (df5['dt'].dt.dayofweek < 5) &
        (df5['dt'].dt.hour >= 13) &
        (df5['dt'].dt.hour < 20)
    )
    df5_sess = df5[session_mask].copy().reset_index(drop=True)
    logger.info(f'  {symbol}: {len(df5)} total bars → {len(df5_sess)} session bars')

    if len(df5_sess) < 300:
        pg.close()
        logger.warning(f'meridian_direction: too few session bars for {symbol}')
        return {}

    # Build features on session bars (need full df5 for regime merge)
    # We still pass regime_df but limit ohlcv to session
    feat_df = build_features(df5_sess, regime_df if not regime_df.empty else None)
    tgt_df  = build_targets(df5_sess)

    report = {}
    for horizon in HORIZONS:
        bars = HORIZON_BARS[horizon]
        logger.info(f'  {symbol}/{horizon}: training...')

        valid = (
            ~feat_df[FEATURE_NAMES].isna().any(axis=1) &
            ~tgt_df[horizon].isna()
        )
        X = feat_df.loc[valid, FEATURE_NAMES].values.astype(np.float64)
        y = tgt_df.loc[valid, horizon].values.astype(int)

        if len(X) < 200:
            logger.warning(f'  {symbol}/{horizon}: only {len(X)} samples — skipping')
            report[horizon] = {'deployed': False, 'reason': 'insufficient_data', 'n': len(X)}
            continue

        label_rate = float(y.mean())
        logger.info(f'  {symbol}/{horizon}: n={len(X)} label_rate={label_rate:.3f}')

        tscv = TimeSeriesSplit(n_splits=5)
        fold_aucs = []
        for fold_idx, (tr_idx, te_idx) in enumerate(tscv.split(X)):
            scaler_f = StandardScaler()
            clf = GradientBoostingClassifier(
                n_estimators=150, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=42
            )
            Xtr, Xte = scaler_f.fit_transform(X[tr_idx]), scaler_f.transform(X[te_idx])
            clf.fit(Xtr, y[tr_idx])
            try:
                auc = roc_auc_score(y[te_idx], clf.predict_proba(Xte)[:, 1])
            except Exception:
                auc = 0.5
            fold_aucs.append(auc)

        mean_auc = float(np.mean(fold_aucs))
        logger.info(f'  {symbol}/{horizon}: fold AUCs={[round(a,3) for a in fold_aucs]} mean={mean_auc:.4f}')

        if mean_auc < AUC_GATE:
            report[horizon] = {
                'deployed': False, 'reason': 'insufficient_edge',
                'auc': round(mean_auc, 4), 'fold_aucs': [round(a,3) for a in fold_aucs],
                'n': len(X),
            }
            logger.info(f'  {symbol}/{horizon}: AUC={mean_auc:.4f} < {AUC_GATE} — NOT deploying')
            _save_model(pg, symbol, horizon, None, None, mean_auc, fold_aucs,
                        len(X), deployed=False, cal_table=[])
            continue

        # Train final model on all data, calibrate on last 20%
        split = int(len(X) * 0.80)
        scaler_final = StandardScaler()
        X_sc = scaler_final.fit_transform(X)
        clf_base = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
        clf_base.fit(scaler_final.transform(X[:split]), y[:split])
        clf_cal = CalibratedClassifierCV(clf_base, method='sigmoid', cv='prefit')
        clf_cal.fit(scaler_final.transform(X[split:]), y[split:])

        # Calibration table on held-out set
        y_prob_te = clf_cal.predict_proba(scaler_final.transform(X[split:]))[:, 1]
        cal_table = calibration_table(y[split:], y_prob_te, n_bins=5)

        report[horizon] = {
            'deployed': True, 'auc': round(mean_auc, 4),
            'fold_aucs': [round(a, 3) for a in fold_aucs],
            'n': len(X), 'calibration': cal_table,
        }
        logger.info(f'  {symbol}/{horizon}: AUC={mean_auc:.4f} — DEPLOYING')
        _save_model(pg, symbol, horizon, clf_cal, scaler_final, mean_auc, fold_aucs,
                    len(X), deployed=True, cal_table=cal_table)

    pg.commit()
    pg.close()

    _dir_cache.pop(symbol, None)
    load_models(symbol)

    logger.info(f'meridian_direction: training complete for {symbol}: {report}')
    return report


def _save_model(pg, symbol, horizon, model, scaler, auc, fold_aucs, n_samples,
                deployed, cal_table):
    import psycopg2
    model_name = f'meridian_dir_{symbol}_{horizon}'
    payload = {
        'model': model, 'scaler': scaler, 'features': FEATURE_NAMES,
        'auc': auc, 'fold_aucs': fold_aucs, 'deployed': deployed,
        'calibration_table': cal_table, 'trained_at': datetime.now(timezone.utc).isoformat(),
    }
    mb = pickle.dumps(payload)
    cur = pg.cursor()
    cur.execute("""
        INSERT INTO ml_models (model_name, model_bytes, oos_auc, feature_count, training_samples, trained_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (model_name) DO UPDATE SET
            model_bytes=EXCLUDED.model_bytes, oos_auc=EXCLUDED.oos_auc,
            feature_count=EXCLUDED.feature_count, training_samples=EXCLUDED.training_samples,
            trained_at=EXCLUDED.trained_at
    """, (model_name, psycopg2.Binary(mb),
          auc, len(FEATURE_NAMES), n_samples, datetime.now(timezone.utc).isoformat()))


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_models(symbol: str) -> bool:
    """Load all horizon models for a symbol from ml_models table."""
    loaded = 0
    horizons = {}
    for horizon in HORIZONS:
        model_name = f'meridian_dir_{symbol}_{horizon}'
        try:
            conn = _db.connect()
            row = conn.execute(
                'SELECT model_bytes, oos_auc, trained_at FROM ml_models WHERE model_name=?',
                (model_name,)
            ).fetchone()
            conn.close()
            if not row:
                continue
            data = pickle.loads(bytes(row[0]))
            horizons[horizon] = {
                'model':     data.get('model'),
                'scaler':    data.get('scaler'),
                'features':  data.get('features', FEATURE_NAMES),
                'auc':       float(row[1] or 0),
                'deployed':  data.get('deployed', False),
                'cal_table': data.get('calibration_table', []),
                'trained_at': row[2],
            }
            loaded += 1
        except Exception as e:
            logger.debug(f'meridian_direction load {model_name}: {e}')
    if horizons:
        _dir_cache[symbol] = horizons
    logger.info(f'meridian_direction: loaded {loaded}/{len(HORIZONS)} models for {symbol}')
    return loaded > 0


def _ensure_loaded(symbol: str):
    if symbol not in _dir_cache:
        load_models(symbol)


# ─────────────────────────────────────────────────────────────────────────────
#  CURRENT FEATURE VECTOR (for real-time inference)
# ─────────────────────────────────────────────────────────────────────────────

def _load_current_ohlcv(symbol: str, n: int = 400) -> Optional[pd.DataFrame]:
    """Load latest N 5min bars from DB (both PG and SQLite)."""
    try:
        conn = _db.connect()
        cur = conn.execute(
            'SELECT ts, open, high, low, close, volume FROM ohlcv '
            'WHERE symbol=? AND timeframe=? ORDER BY ts DESC LIMIT ?',
            (symbol, '5min', n)
        )
        rows = cur.fetchall()
        conn.close()
    except Exception as e:
        logger.warning(f'meridian_direction _load_current_ohlcv {symbol}: {e}')
        return None
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.iloc[::-1].reset_index(drop=True)
    df['dt'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    return df


def _load_current_regime(symbol: str, n: int = 200) -> Optional[pd.DataFrame]:
    try:
        conn = _db.connect()
        if _db.IS_POSTGRES:
            cur = conn.execute(
                "SELECT EXTRACT(EPOCH FROM timestamp)::bigint, regime, "
                "hurst::float, autocorr::float, vol_ratio::float, confidence::float "
                "FROM regime_log WHERE symbol=? ORDER BY timestamp DESC LIMIT ?",
                (symbol, n)
            )
        else:
            cur = conn.execute(
                "SELECT CAST(strftime('%s', timestamp) AS INTEGER), regime, "
                "hurst, autocorr, vol_ratio, confidence "
                "FROM regime_log WHERE symbol=? ORDER BY timestamp DESC LIMIT ?",
                (symbol, n)
            )
        rows = cur.fetchall()
        conn.close()
    except Exception as e:
        logger.warning(f'meridian_direction _load_current_regime {symbol}: {e}')
        return None
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=['ts_reg', 'regime', 'hurst', 'autocorr', 'vol_ratio', 'confidence'])
    for c in ['ts_reg', 'hurst', 'autocorr', 'vol_ratio', 'confidence']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.iloc[::-1].reset_index(drop=True)


def get_current_features(symbol: str) -> Optional[np.ndarray]:
    """Build and return the most recent feature vector for inference."""
    df5      = _load_current_ohlcv(symbol, n=400)
    reg_df   = _load_current_regime(symbol, n=200)
    if df5 is None or len(df5) < 50:
        return None
    try:
        feat_df = build_features(df5, reg_df)
        last_row = feat_df.iloc[-1][FEATURE_NAMES].values.astype(np.float64)
        if np.isnan(last_row).any():
            # Replace NaN with column medians from recent history
            for i, v in enumerate(last_row):
                if np.isnan(v):
                    col_vals = feat_df[FEATURE_NAMES[i]].dropna()
                    last_row[i] = float(col_vals.median()) if len(col_vals) > 0 else 0.0
        return last_row
    except Exception as e:
        logger.warning(f'meridian_direction get_current_features {symbol}: {e}')
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_all(symbol: str) -> Dict[str, Any]:
    """
    Return directional forecast for all horizons for one symbol.
    Each horizon returns either a deployed prediction or an 'insufficient_edge' marker.

    Returns dict keyed by horizon: '30min', '1hr', '4hr'.
    """
    _ensure_loaded(symbol)
    horizons_cache = _dir_cache.get(symbol, {})

    feat_vec = get_current_features(symbol)
    result = {}

    for horizon in HORIZONS:
        entry = horizons_cache.get(horizon)
        if entry is None:
            result[horizon] = {'deployed': False, 'reason': 'no_model'}
            continue
        if not entry.get('deployed'):
            result[horizon] = {
                'deployed': False,
                'reason':   entry.get('reason', 'insufficient_edge'),
                'auc':      round(entry.get('auc', 0), 4),
            }
            continue
        if feat_vec is None:
            result[horizon] = {'deployed': False, 'reason': 'no_features'}
            continue

        try:
            model  = entry['model']
            scaler = entry['scaler']
            X_sc   = scaler.transform(feat_vec.reshape(1, -1))
            prob   = float(model.predict_proba(X_sc)[0][1])
            direction = 'UP' if prob >= 0.5 else 'DOWN'

            # Build contributing signal breakdown for the 'why' section
            breakdown = _build_breakdown(feat_vec, prob)

            result[horizon] = {
                'deployed':   True,
                'direction':  direction,
                'probability': round(prob, 4),
                'prob_pct':   round(prob * 100, 1),
                'auc':        round(entry.get('auc', 0), 4),
                'trained_at': entry.get('trained_at'),
                'breakdown':  breakdown,
            }
        except Exception as e:
            logger.warning(f'meridian_direction predict {symbol}/{horizon}: {e}')
            result[horizon] = {'deployed': False, 'reason': f'predict_error: {e}'}

    return result


def _build_breakdown(feat_vec: np.ndarray, prob: float) -> Dict[str, Any]:
    """Extract human-readable signal breakdown from feature vector."""
    f = dict(zip(FEATURE_NAMES, feat_vec))

    # Regime
    if f.get('regime_trending', 0) > 0.5:
        regime_label = 'TRENDING'
    elif f.get('regime_choppy', 0) > 0.5:
        regime_label = 'CHOPPY'
    else:
        regime_label = 'MEAN_REV'
    regime_leans = 'UP' if f.get('regime_trending', 0) > 0.5 else 'NEUTRAL'

    # HTF bias (1hr + 4hr)
    htf_score = f.get('ema_1h', 0) + f.get('ema_4h', 0)
    htf_leans = 'UP' if htf_score > 0 else 'DOWN' if htf_score < 0 else 'NEUTRAL'

    # MTF alignment
    bull_count = int(f.get('mtf_bull_count', 0))
    mtf_leans  = 'UP' if bull_count >= 4 else 'DOWN' if bull_count <= 2 else 'NEUTRAL'

    # VWAP position
    vwap_d = f.get('vwap_dist_atr', 0)
    vwap_leans = 'UP' if vwap_d > 0.3 else 'DOWN' if vwap_d < -0.3 else 'NEUTRAL'

    # Momentum (5-bar return)
    ret5 = f.get('ret_5', 0)
    mom_leans = 'UP' if ret5 > 0 else 'DOWN'

    return {
        'regime':       {'label': regime_label,      'leans': regime_leans,
                          'conf': round(f.get('regime_conf', 0), 3)},
        'htf_bias':     {'label': f'1h={_bias_str(f.get("ema_1h",0))} 4h={_bias_str(f.get("ema_4h",0))}',
                          'leans': htf_leans, 'score': round(htf_score, 1)},
        'mtf_alignment':{'label': f'{bull_count}/6 bullish', 'leans': mtf_leans, 'count': bull_count},
        'vwap_pos':     {'label': f'{"ABOVE" if vwap_d > 0 else "BELOW"} VWAP',
                          'leans': vwap_leans, 'dist': round(vwap_d, 2)},
        'momentum':     {'label': f'ret5={round(ret5*100,3)}%', 'leans': mom_leans,
                          'value': round(ret5, 5)},
    }


def _bias_str(v: float) -> str:
    return 'BULL' if v > 0 else 'BEAR' if v < 0 else 'NEUT'


# ─────────────────────────────────────────────────────────────────────────────
#  FORECAST LOG — live accuracy tracker
# ─────────────────────────────────────────────────────────────────────────────

def log_prediction(symbol: str, horizon: str, direction: str, probability: float,
                   price: float):
    """Write a prediction to direction_forecast_log for later resolution."""
    now = datetime.now(timezone.utc)
    bars = HORIZON_BARS.get(horizon, 12)
    resolves_at = now + timedelta(minutes=bars * 5)
    try:
        conn = _db.connect()
        conn.execute(
            'INSERT INTO direction_forecast_log '
            '(symbol, horizon, direction, probability, price_at_log, predicted_at, resolves_at) '
            'VALUES (?,?,?,?,?,?,?)',
            (symbol, horizon, direction, probability, price,
             now.isoformat(), resolves_at.isoformat())
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.debug(f'meridian_direction log_prediction: {e}')


def resolve_predictions():
    """Check unresolved predictions that have passed their resolves_at time and mark correct/incorrect."""
    try:
        conn = _db.connect()
        if _db.IS_POSTGRES:
            rows = conn.execute(
                "SELECT id, symbol, horizon, direction, price_at_log, resolves_at "
                "FROM direction_forecast_log WHERE resolved=FALSE AND resolves_at <= NOW()"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, symbol, horizon, direction, price_at_log, resolves_at "
                "FROM direction_forecast_log WHERE resolved=0 AND resolves_at <= datetime('now')"
            ).fetchall()

        for row in rows:
            row_id, symbol, horizon, direction, price_at_log, resolves_at = row
            # Get the current close price as resolution
            try:
                cur = conn.execute(
                    'SELECT close FROM ohlcv WHERE symbol=? AND timeframe=? ORDER BY ts DESC LIMIT 1',
                    (symbol, '5min')
                )
                r = cur.fetchone()
                if r is None:
                    continue
                current_price = float(r[0])
                correct = (direction == 'UP' and current_price > price_at_log) or \
                          (direction == 'DOWN' and current_price < price_at_log)
                conn.execute(
                    'UPDATE direction_forecast_log SET resolved=?, correct=? WHERE id=?',
                    (True if _db.IS_POSTGRES else 1, correct, row_id)
                )
            except Exception:
                pass
        conn.commit()
        conn.close()
    except Exception as e:
        logger.debug(f'meridian_direction resolve_predictions: {e}')


def get_accuracy_stats(symbol: str, horizon: str, n: int = 50) -> Optional[Dict]:
    """Return rolling accuracy stats for the live tracker."""
    try:
        conn = _db.connect()
        rows = conn.execute(
            'SELECT correct FROM direction_forecast_log '
            'WHERE symbol=? AND horizon=? AND resolved=? '
            'ORDER BY predicted_at DESC LIMIT ?',
            (symbol, horizon, True if _db.IS_POSTGRES else 1, n)
        ).fetchall()
        conn.close()
        if not rows:
            return None
        total   = len(rows)
        correct = sum(1 for r in rows if r[0])
        return {
            'total':    total,
            'correct':  correct,
            'accuracy': round(correct / total, 4) if total > 0 else None,
            'pct':      round(correct / total * 100, 1) if total > 0 else None,
        }
    except Exception as e:
        logger.debug(f'get_accuracy_stats {symbol}/{horizon}: {e}')
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  RETRAIN (weekly cron — Sunday 03:00 UTC)
# ─────────────────────────────────────────────────────────────────────────────

def retrain_all() -> Dict[str, Any]:
    """Weekly retrain for all symbols. Called by background_scheduler."""
    results = {}
    for sym in SYMBOLS:
        try:
            results[sym] = train(sym)
        except Exception as e:
            logger.error(f'meridian_direction retrain_all {sym}: {e}', exc_info=True)
            results[sym] = {'error': str(e)}
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  CLI — run training directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    syms = sys.argv[1:] if len(sys.argv) > 1 else list(SYMBOLS)
    for sym in syms:
        print(f'\n{"="*60}')
        print(f'Training {sym}...')
        report = train(sym)
        for h, r in report.items():
            if r.get('deployed'):
                print(f'  {h}: AUC={r["auc"]:.4f}  DEPLOYED  n={r["n"]}')
                for row in r.get('calibration', []):
                    print(f'    bin {row["bin"]}: n={row["count"]} '
                          f'pred={row["mean_pred"]:.3f} actual={row["actual_freq"]:.3f} '
                          f'gap={row["gap"]:.3f}')
            else:
                print(f'  {h}: NOT deployed — {r.get("reason","?")} AUC={r.get("auc","—")}')
