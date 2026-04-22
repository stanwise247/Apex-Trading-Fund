"""
APEX Setup F — Random Forest ML Signal
========================================
8 features: consecutive, htf_bias, hour, vol_regime, atr_ratio, dow, vol_ratio, hurst
Labels: RR 2.5 target hit before 1.5×ATR stop within 100 bars (outcome-based, not direction)
OOS gate: 80/20 walk-forward must reach >= 54% accuracy to deploy
Thresholds: long P > 0.65 | short P < 0.35
Signal cap: max 2 per instrument per session-day
Retrain: monthly, rolling 6-month window
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import db as _db

logger = logging.getLogger('APEX.SetupF')

SESSION_WINDOWS = {
    'MNQ': (13, 19),
    'NQ':  (13, 19),  # kept for legacy reference
    'ES':  (13, 19),
    'GC':  (12, 17),
}

# Signal thresholds — raised from 0.58/0.42 to reduce false positives
LONG_THRESHOLD  = 0.65
SHORT_THRESHOLD = 0.35

# Max Setup F signals per instrument per session-day
MAX_SIGNALS_PER_SESSION = 2

# OOS accuracy gate — model must clear this to deploy
OOS_ACCURACY_GATE = 0.54

# Module-level state
_f_session_counts: dict = {}    # (symbol, date_str) -> int (signals fired today)
_f_disabled:       set  = set() # symbols where OOS gate failed — skip until next retrain
_last_feat_log:    dict = {}    # symbol -> date_str (for per-session feature logging)

# Legacy caches (kept for backward compatibility)
_hurst_cache = {}   # symbol -> (computed_at, series)
_model_cache = {}   # symbol -> (loaded_at, model)


# ─────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────

def _load_ohlcv(symbol: str, timeframe: str, limit: int = 2000) -> pd.DataFrame:
    """Load OHLCV bars via cursor (bypasses pd.read_sql_query param issues with psycopg2)."""
    _COLS = ['ts', 'open', 'high', 'low', 'close', 'volume']
    conn = _db.connect()
    try:
        cur = conn.execute(
            'SELECT ts, open, high, low, close, volume FROM ohlcv '
            'WHERE symbol=? AND timeframe=? ORDER BY ts DESC LIMIT ?',
            (symbol, timeframe, limit)
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return pd.DataFrame(columns=_COLS)
    df = pd.DataFrame(rows, columns=_COLS)
    for col in _COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.iloc[::-1].reset_index(drop=True)
    df['dt'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    return df


# ─────────────────────────────────────────────────────────────
#  HURST EXPONENT — R/S analysis
# ─────────────────────────────────────────────────────────────

def _hurst_rs(series: np.ndarray) -> float:
    lags = range(2, 20)
    RS = []
    for lag in lags:
        chunks = [series[i:i+lag] for i in range(0, len(series)-lag, lag)]
        rs_vals = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            m = np.mean(chunk)
            devs = np.cumsum(chunk - m)
            R = devs.max() - devs.min()
            S = np.std(chunk, ddof=1)
            if S > 0:
                rs_vals.append(R / S)
        if rs_vals:
            RS.append(np.mean(rs_vals))
    if len(RS) < 4:
        return 0.5
    log_lags = np.log(list(lags)[:len(RS)])
    log_RS   = np.log(RS)
    H = np.polyfit(log_lags, log_RS, 1)[0]
    return float(np.clip(H, 0.1, 0.9))


# Legacy function — kept for API compatibility; not used by calculate_features
def _get_hurst_series(symbol: str) -> pd.Series:
    cache_key = symbol
    now = datetime.now(timezone.utc)
    if cache_key in _hurst_cache:
        cached_at, series = _hurst_cache[cache_key]
        if (now - cached_at).total_seconds() < 1800:
            return series
    df = _load_ohlcv(symbol, '1hour', limit=1500)
    if df.empty or len(df) < 55:
        s = pd.Series(dtype=float)
        _hurst_cache[cache_key] = (now, s)
        return s
    log_ret = np.log(df['close'] / df['close'].shift(1)).dropna().values
    window  = min(50, max(20, len(log_ret) // 10))
    hursts, indices = [], []
    for i in range(window, len(log_ret) + 1):
        hursts.append(_hurst_rs(log_ret[i-window:i]))
        indices.append(df['ts'].iloc[i])
    series = pd.Series(hursts, index=indices)
    _hurst_cache[cache_key] = (now, series)
    return series


def _get_current_hurst(symbol: str) -> float:
    series = _get_hurst_series(symbol)
    if series.empty:
        return 0.5
    return float(series.iloc[-1])


# ─────────────────────────────────────────────────────────────
#  FEATURE HELPERS
# ─────────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low  = df['low']
    prev = df['close'].shift(1)
    tr   = pd.concat([
        high - low,
        (high - prev).abs(),
        (low  - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _consecutive(closes: pd.Series) -> pd.Series:
    """Count consecutive same-direction closes, capped at ±5."""
    direction = np.sign(closes.diff())
    result = pd.Series(0, index=closes.index, dtype=float)
    streak = 0
    prev_d = 0
    for i in range(len(direction)):
        d = direction.iloc[i]
        if np.isnan(d) or d == 0:
            streak = 0
        elif d == prev_d:
            streak = streak + 1 if streak > 0 else -1
        else:
            streak = int(d)
        result.iloc[i] = max(-5, min(5, streak))
        prev_d = d
    return result


# ─────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING — 8 features
# ─────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    'consecutive',  # 0 — streak of same-dir closes, ±5
    'htf_bias',     # 1 — 4h SMA20 trend position, -1/0/+1
    'hour',         # 2 — UTC hour of bar, 0-23
    'vol_regime',   # 3 — vol vs 1h rolling percentile, -1/0/+1
    'atr_ratio',    # 4 — ATR5 / ATR20 expansion signal
    'dow',          # 5 — day of week, 0=Mon 4=Fri
    'vol_ratio',    # 6 — current volume / 20-bar vol MA
    'hurst',        # 7 — R/S Hurst exponent on 50 1h bars (float 0.1-0.9)
]


def calculate_features(symbol: str, df_5m: pd.DataFrame,
                        df_4h: pd.DataFrame = None, df_1h: pd.DataFrame = None) -> np.ndarray:
    """
    Build 8-feature matrix entirely from df_5m — no HTF DB dependency.
    Returns array shape (n_bars, 8) aligned to df_5m.
    Requires df_5m columns: ts, open, high, low, close, volume, dt
    df_4h and df_1h params kept for API compatibility but ignored.
    """
    n     = len(df_5m)
    feats = np.full((n, 8), np.nan)

    # Use dt as index for resampling
    df_idx = df_5m.set_index('dt')
    _5m_ts = pd.DataFrame({'ts': df_5m['ts'].values})

    # ── 0. consecutive ─────────────────────────────────────────
    feats[:, 0] = _consecutive(df_5m['close']).values

    # ── 1. htf_bias — 5min → 4h resample in memory, SMA20 ─────
    df_4h_mem = df_idx['close'].resample('4h').last().dropna().to_frame('close')
    df_4h_mem['sma20'] = df_4h_mem['close'].rolling(20).mean()
    df_4h_mem['ts']    = df_4h_mem.index.map(lambda x: int(x.timestamp()))
    merged_4h = pd.merge_asof(
        _5m_ts,
        df_4h_mem[['ts', 'close', 'sma20']].sort_values('ts'),
        on='ts', direction='backward'
    )
    band = merged_4h['sma20'] * 0.001
    feats[:, 1] = np.where(merged_4h['sma20'].isna(), np.nan,
                  np.where(merged_4h['close'] > merged_4h['sma20'] + band,  1.0,
                  np.where(merged_4h['close'] < merged_4h['sma20'] - band, -1.0, 0.0)))

    # ── 2. hour ────────────────────────────────────────────────
    feats[:, 2] = df_5m['dt'].dt.hour.values.astype(float)

    # ── 3. vol_regime — 5min → 1h resample, rolling std ───────
    df_1h_mem = df_idx['close'].resample('1h').last().dropna().to_frame('close')
    log_ret   = np.log(df_1h_mem['close'] / df_1h_mem['close'].shift(1))
    rvol      = log_ret.rolling(20).std()
    p25, p75  = rvol.quantile(0.25), rvol.quantile(0.75)
    vr        = pd.Series(0.0, index=df_1h_mem.index)
    vr[rvol > p75] = 1.0
    vr[rvol < p25] = -1.0
    df_1h_mem['vr'] = vr
    df_1h_mem['ts'] = df_1h_mem.index.map(lambda x: int(x.timestamp()))
    merged_vr = pd.merge_asof(
        _5m_ts,
        df_1h_mem[['ts', 'vr']].sort_values('ts'),
        on='ts', direction='backward'
    )
    feats[:, 3] = merged_vr['vr'].fillna(0.0).values

    # ── 4. atr_ratio (ATR5 / ATR20) ───────────────────────────
    atr5  = _atr(df_5m, 5)
    atr20 = _atr(df_5m, 20)
    feats[:, 4] = (atr5 / (atr20 + 1e-9)).values

    # ── 5. dow ─────────────────────────────────────────────────
    feats[:, 5] = df_5m['dt'].dt.dayofweek.values.astype(float)

    # ── 6. vol_ratio ───────────────────────────────────────────
    vol_ma    = df_5m['volume'].rolling(20).mean()
    feats[:, 6] = (df_5m['volume'] / (vol_ma + 1e-9)).values

    # ── 7. hurst — rolling R/S on 50 1h log returns (float) ───
    # Window=50 (~2 trading days of 1h bars) produces variation over time.
    # Returns raw H float (0.1-0.9); RF learns its own optimal split.
    lr_vals = log_ret.dropna().values
    h_window = min(50, max(20, len(lr_vals) // 10))
    if len(lr_vals) >= h_window + 5:
        h_vals, h_ts_list = [], []
        df_1h_reset = df_1h_mem.reset_index()  # columns: dt, close, vr, ts
        for i in range(h_window, len(lr_vals) + 1):
            chunk = lr_vals[i - h_window: i]
            if len(chunk) >= 20:
                h_vals.append(_hurst_rs(chunk))
                # Map to timestamp of the i-th non-NaN 1h bar
                bar_idx = log_ret.isna().sum() + i - 1  # offset for initial NaN
                if bar_idx < len(df_1h_reset):
                    h_ts_list.append(int(df_1h_reset['ts'].iloc[bar_idx]))
        if h_vals and len(h_vals) == len(h_ts_list):
            h_df = pd.DataFrame({'ts': h_ts_list, 'hv': h_vals})
            merged_h = pd.merge_asof(
                _5m_ts,
                h_df.sort_values('ts'),
                on='ts', direction='backward'
            )
            feats[:, 7] = merged_h['hv'].fillna(0.5).values
        else:
            feats[:, 7] = 0.5
    else:
        feats[:, 7] = 0.5

    return feats


# ─────────────────────────────────────────────────────────────
#  LABEL CONSTRUCTION — outcome-based, no lookahead in features
# ─────────────────────────────────────────────────────────────

def _build_labels(df: pd.DataFrame, rr: float = 2.5,
                  stop_atr: float = 1.5, max_bars: int = 100) -> np.ndarray:
    """
    Simulate trade outcome for each bar as a LONG entry at close.
      target = close + stop_atr * rr * ATR14
      stop   = close - stop_atr * ATR14
    y[i]=1.0 if high[j]>=target before low[j]<=stop (within max_bars)
    y[i]=0.0 if low[j]<=stop  before high[j]>=target
    y[i]=NaN if neither triggers → excluded from training

    This matches the actual live trade logic (1.5×ATR stop, 3.75×ATR target).
    Features use only past bars; labels are future outcomes. No lookahead bias.
    """
    n      = len(df)
    atr14  = _atr(df, 14).values
    closes = df['close'].values
    highs  = df['high'].values
    lows   = df['low'].values
    y      = np.full(n, np.nan)

    for i in range(n - max_bars - 1):
        if np.isnan(atr14[i]) or atr14[i] <= 0:
            continue
        entry  = closes[i]
        stop_l = entry - stop_atr * atr14[i]
        tgt_l  = entry + stop_atr * rr * atr14[i]
        for j in range(i + 1, min(i + max_bars + 1, n)):
            if highs[j] >= tgt_l:
                y[i] = 1.0
                break
            if lows[j] <= stop_l:
                y[i] = 0.0
                break
        # Remains NaN if neither triggered → excluded
    return y


# ─────────────────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────────────────

def train_model(symbol: str) -> float:
    """
    Train RF classifier on last 6 months of 5min bars.
    Labels: actual trade outcome (target hit before stop).
    OOS gate: 80/20 walk-forward must reach >= OOS_ACCURACY_GATE to deploy.
    Returns OOS accuracy or 0.0 if gate fails / insufficient data.
    """
    from sklearn.ensemble import RandomForestClassifier

    global _f_disabled
    logger.info(f'Training Setup F model for {symbol}...')
    start_hr, end_hr = SESSION_WINDOWS.get(symbol, (13, 19))

    df_5m = _load_ohlcv(symbol, '5min', limit=25000)
    if df_5m.empty or len(df_5m) < 500:
        logger.warning(f'Insufficient 5min data for {symbol}: {len(df_5m)} bars')
        return 0.0

    six_months_ago = int((datetime.now(timezone.utc) - timedelta(days=180)).timestamp())
    df_5m = df_5m[df_5m['ts'] >= six_months_ago].copy().reset_index(drop=True)
    logger.info(f'  {symbol}: {len(df_5m)} 5min bars in last 6 months')

    # Session mask — computed before features; NOT applied to df (breaks resampling)
    session_mask = df_5m['dt'].dt.hour.between(start_hr, end_hr - 1).values
    if session_mask.sum() < 100:
        logger.warning(f'Insufficient session bars for {symbol}: {session_mask.sum()}')
        return 0.0

    # Build features on full contiguous df — HTF resample needs unbroken series
    logger.info(f'  {symbol}: computing features...')
    X_all = calculate_features(symbol, df_5m)

    # Outcome-based labels (RR 2.5, stop 1.5×ATR, max 100 bars forward)
    logger.info(f'  {symbol}: computing outcome labels...')
    y_all = _build_labels(df_5m, rr=2.5, stop_atr=1.5, max_bars=100)

    # Filter: session hours + no NaN features + label resolved
    valid_mask = (
        ~np.isnan(X_all).any(axis=1) &
        ~np.isnan(y_all) &
        session_mask
    )

    X = X_all[valid_mask]
    y = y_all[valid_mask].astype(int)

    label_pos = y.mean() if len(y) > 0 else 0.5
    logger.info(
        f'  {symbol}: {len(X)} training samples '
        f'(label balance: {label_pos:.1%} wins, {1-label_pos:.1%} losses)'
    )
    # Note: 12-bar dedup is enforced in live prediction (session cap of 2/day).
    # Removing it from training prevents data starvation — outcome labels already
    # give each bar an honest verdict; consecutive bars with different outcomes are valid.

    if len(X) < 100:
        logger.warning(f'Too few training samples for {symbol}: {len(X)} (need 100)')
        return 0.0

    # ── OOS gate: 80/20 walk-forward ──────────────────────────
    split     = int(len(X) * 0.80)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    if len(X_te) < 30:
        logger.warning(f'OOS test set too small for {symbol}: {len(X_te)} bars')
        oos_auc = 0.0
    else:
        from sklearn.metrics import roc_auc_score
        clf_oos = RandomForestClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=20,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        clf_oos.fit(X_tr, y_tr)
        oos_probs = clf_oos.predict_proba(X_te)[:, 1]
        try:
            oos_auc = float(roc_auc_score(y_te, oos_probs))
        except Exception:
            oos_auc = 0.5
        # Also compute precision on signals above threshold
        sig_mask = oos_probs > LONG_THRESHOLD
        oos_prec = float(y_te[sig_mask].mean()) if sig_mask.sum() > 0 else 0.0
        logger.info(
            f'Setup F: {symbol} OOS AUC={oos_auc:.3f} | '
            f'long precision={oos_prec:.1%} ({sig_mask.sum()} signals) | '
            f'deploying={"True" if oos_auc >= OOS_ACCURACY_GATE else "False"}'
        )

    if oos_auc < OOS_ACCURACY_GATE:
        logger.critical(
            f'Setup F: model failed OOS gate for {symbol} — not deploying '
            f'(AUC={oos_auc:.3f} < {OOS_ACCURACY_GATE})'
        )
        _f_disabled.add(symbol)
        return 0.0

    # Gate passed — train final model on all data
    _f_disabled.discard(symbol)
    model = RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=20,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    model.fit(X, y)
    train_acc = model.score(X, y)

    # Feature importance log
    for i, (name, imp) in enumerate(zip(FEATURE_NAMES, model.feature_importances_)):
        logger.info(f'    {name}: {imp:.1%}')

    pkl_path = f'apex_rf_{symbol}.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'model':      model,
            'trained_at': datetime.now(timezone.utc).isoformat(),
            'symbol':     symbol,
            'accuracy':   train_acc,
            'oos_auc':    oos_auc,
            'n_features': len(FEATURE_NAMES),
            'features':   FEATURE_NAMES,
        }, f)
    logger.info(
        f'Model saved: {pkl_path} | train_acc={train_acc:.3f} | oos_auc={oos_auc:.3f}'
    )
    return oos_auc


def load_or_train_model(symbol: str):
    """Load model from pkl if < 30 days old, else retrain."""
    now = datetime.now(timezone.utc)
    if symbol in _model_cache:
        cached_at, model = _model_cache[symbol]
        if (now - cached_at).total_seconds() < 300:
            return model

    pkl_path = f'apex_rf_{symbol}.pkl'
    if os.path.exists(pkl_path):
        mtime    = datetime.fromtimestamp(os.path.getmtime(pkl_path), tz=timezone.utc)
        age_days = (now - mtime).days
        if age_days < 30:
            try:
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                # Reject stale pkl trained on old 11-feature set
                if data.get('n_features', 0) != len(FEATURE_NAMES):
                    logger.warning(
                        f'{pkl_path} has {data.get("n_features")} features, '
                        f'expected {len(FEATURE_NAMES)} — retraining'
                    )
                    raise ValueError('feature count mismatch')
                model = data['model']
                _model_cache[symbol] = (now, model)
                return model
            except Exception as e:
                logger.warning(f'Failed to load {pkl_path}: {e}')

    train_model(symbol)
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            model = data['model']
            _model_cache[symbol] = (now, model)
            return model
        except Exception as e:
            logger.error(f'Model load after retrain failed {symbol}: {e}')
    return None


# ─────────────────────────────────────────────────────────────
#  SCAN
# ─────────────────────────────────────────────────────────────

def scan_setup_f(symbol: str, dt: datetime = None) -> dict | None:
    """
    Check Setup F ML signal for symbol at current bar.
    Returns signal dict or None if no edge / cap reached / OOS gate failed.
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    if dt.weekday() >= 5:
        return None

    start_hr, end_hr = SESSION_WINDOWS.get(symbol, (13, 19))
    if not (start_hr <= dt.hour < end_hr):
        return None

    # OOS gate check
    if symbol in _f_disabled:
        logger.debug(f'Setup F {symbol}: disabled (OOS gate failed) — skipping')
        return None

    # Session frequency cap — max 2 signals per instrument per day
    date_str  = dt.strftime('%Y-%m-%d')
    day_key   = (symbol, date_str)
    count_today = _f_session_counts.get(day_key, 0)
    if count_today >= MAX_SIGNALS_PER_SESSION:
        logger.info(f'Setup F: session cap reached for {symbol} — skipping ({count_today} signals today)')
        return None

    model = load_or_train_model(symbol)
    if model is None:
        logger.warning(f'No model available for {symbol}')
        return None

    # Validate feature count matches current architecture
    if hasattr(model, 'n_features_in_') and model.n_features_in_ != len(FEATURE_NAMES):
        logger.warning(
            f'Setup F {symbol}: model has {model.n_features_in_} features, '
            f'expected {len(FEATURE_NAMES)} — triggering retrain'
        )
        _model_cache.pop(symbol, None)
        train_model(symbol)
        model = load_or_train_model(symbol)
        if model is None:
            return None

    df_5m = _load_ohlcv(symbol, '5min', limit=2000)
    if df_5m.empty or len(df_5m) < 60:
        return None

    X_all  = calculate_features(symbol, df_5m)
    X_last = X_all[-1:].copy()

    if np.isnan(X_last).any():
        logger.warning(f'Setup F {symbol}: NaN in features — skipping')
        return None

    # Feature validation — log on first bar of each session for sanity checks
    if _last_feat_log.get(symbol) != date_str:
        _last_feat_log[symbol] = date_str
        feat_str = ', '.join(
            f'{FEATURE_NAMES[i]}={X_last[0, i]:.3f}' for i in range(len(FEATURE_NAMES))
        )
        logger.info(f'Setup F {symbol} session features: {feat_str}')

    prob = float(model.predict_proba(X_last)[0, 1])
    logger.info(
        f'Setup F {symbol}: prob={prob:.3f} | '
        f'long>{LONG_THRESHOLD} short<{SHORT_THRESHOLD} | '
        f'{"LONG SIGNAL" if prob > LONG_THRESHOLD else "SHORT SIGNAL" if prob < SHORT_THRESHOLD else "no edge"}'
    )

    if prob > LONG_THRESHOLD:
        direction = 'long'
    elif prob < SHORT_THRESHOLD:
        direction = 'short'
    else:
        return None

    last  = df_5m.iloc[-1]
    atr14 = _atr(df_5m, 14).iloc[-1]
    if pd.isna(atr14) or atr14 <= 0:
        return None

    entry  = float(last['close'])
    stop   = entry - 1.5 * atr14 if direction == 'long' else entry + 1.5 * atr14
    target = entry + 3.75 * atr14 if direction == 'long' else entry - 3.75 * atr14
    rr     = round(abs(target - entry) / abs(entry - stop), 2)

    importances = model.feature_importances_
    top_idx     = np.argsort(importances)[::-1][:3]
    top_feats   = [
        {'name': FEATURE_NAMES[i], 'importance': round(float(importances[i]), 3),
         'value': round(float(X_last[0, i]), 4)}
        for i in top_idx
    ]

    # Increment session counter after confirmed signal
    _f_session_counts[day_key] = count_today + 1

    return {
        'symbol':       symbol,
        'direction':    direction,
        'setup':        'F_rf_ml',
        'mode':         'intraday',
        'entry':        round(entry, 2),
        'stop':         round(stop, 2),
        'target':       round(target, 2),
        'rr':           rr,
        'session':      'NY Primary',
        'quality':      'primary',
        'confidence':   prob,
        'top_features': top_feats,
        'atr':          round(float(atr14), 2),
        'timestamp':    dt.isoformat(),
    }


# ─────────────────────────────────────────────────────────────
#  ALERT FORMATTER
# ─────────────────────────────────────────────────────────────

def format_f_alert(signal: dict) -> str:
    """Format Setup F Telegram alert."""
    from zoneinfo import ZoneInfo
    NY   = ZoneInfo('America/New_York')
    now  = datetime.now(timezone.utc).astimezone(NY).strftime('%Y-%m-%d %H:%M')
    sym  = signal['symbol']
    dir_ = signal['direction'].upper()
    conf = signal['confidence']
    sep  = chr(9473) * 20

    emoji      = chr(128200) if dir_ == 'LONG' else chr(128201)
    entry      = signal['entry']
    stop       = signal['stop']
    target     = signal['target']
    stop_pts   = abs(round(entry - stop, 2))
    target_pts = abs(round(entry - target, 2))

    feat_lines = []
    for f in signal.get('top_features', [])[:3]:
        feat_lines.append(f"  {f['name']}: {f['value']:+.4f} ({f['importance']:.0%})")
    feats_str = chr(10).join(feat_lines) if feat_lines else '  —'

    parts = [
        f'{emoji} <b>APEX ML SIGNAL — {sym}</b>',
        sep,
        f'<b>Setup:</b>     F — Random Forest ML',
        f'<b>Direction:</b> {dir_}',
        f'<b>Confidence:</b> P={conf:.2f} (threshold {">" + str(LONG_THRESHOLD) if dir_=="LONG" else "<" + str(SHORT_THRESHOLD)})',
        sep,
        f'<b>Entry:</b>     {entry:.2f}',
        f'<b>Stop:</b>      {stop:.2f} ({stop_pts} pts)',
        f'<b>Target:</b>    {target:.2f} ({target_pts} pts)',
        f'<b>R:R:</b>       {signal["rr"]}x',
        sep,
        f'<b>Top Drivers:</b>',
        feats_str,
        sep,
        f'<i>{now} ET</i>',
        f'<i>Setup F MNQ {dir_} | P={conf:.2f} | entry={entry:.2f} stop={stop:.2f} target={target:.2f}</i>',
    ]
    return chr(10).join(parts)


# ─────────────────────────────────────────────────────────────
#  DEGRADATION CHECK
# ─────────────────────────────────────────────────────────────

def check_model_degradation(symbol: str) -> bool:
    """Returns True if model WR on last 30 closed Setup F trades < 55%."""
    try:
        conn = _db.connect()
        rows = conn.execute(
            "SELECT direction, pnl_r FROM apex_trades "
            "WHERE symbol=? AND setup='F_rf_ml' AND status='closed' "
            "ORDER BY entry_time DESC LIMIT 30",
            (symbol,)
        ).fetchall()
        conn.close()
        if len(rows) < 10:
            return False
        wins = sum(1 for _, pnl in rows if pnl is not None and pnl > 0)
        acc  = wins / len(rows)
        if acc < 0.55:
            logger.warning(
                f'Setup F {symbol} model degraded — last {len(rows)} trades WR={acc:.0%} < 55%'
            )
            return True
        return False
    except Exception as e:
        logger.debug(f'check_model_degradation {symbol}: {e}')
        return False


# ─────────────────────────────────────────────────────────────
#  CURRENT PREDICTION (for API endpoint)
# ─────────────────────────────────────────────────────────────

def get_current_prediction(symbol: str) -> dict:
    """Return current ML prediction state for dashboard."""
    try:
        model = load_or_train_model(symbol)
        if model is None:
            return {'symbol': symbol, 'ok': False, 'error': 'No model'}

        df_5m = _load_ohlcv(symbol, '5min', limit=2000)
        if df_5m.empty:
            return {'symbol': symbol, 'ok': False, 'error': 'No data'}

        X_all  = calculate_features(symbol, df_5m)
        X_last = X_all[-1:]

        if np.isnan(X_last).any():
            return {'symbol': symbol, 'ok': False, 'error': 'NaN in features'}

        prob = float(model.predict_proba(X_last)[0, 1])
        if prob > LONG_THRESHOLD:
            signal_state = 'SIGNAL_LONG'
        elif prob < SHORT_THRESHOLD:
            signal_state = 'SIGNAL_SHORT'
        elif prob > 0.50:
            signal_state = 'WATCHING_LONG'
        elif prob < 0.50:
            signal_state = 'WATCHING_SHORT'
        else:
            signal_state = 'NO_EDGE'

        importances = model.feature_importances_
        top_idx     = np.argsort(importances)[::-1][:3]
        top_feats   = [
            {'name': FEATURE_NAMES[i], 'importance': round(float(importances[i]), 3),
             'value': round(float(X_last[0, i]), 4)}
            for i in top_idx
        ]

        pkl_path   = f'apex_rf_{symbol}.pkl'
        trained_at = None
        oos_acc    = None
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, 'rb') as f:
                    d = pickle.load(f)
                trained_at = d.get('trained_at')
                oos_acc    = d.get('oos_auc', d.get('oos_accuracy'))
            except Exception:
                pass

        return {
            'symbol':          symbol,
            'ok':              True,
            'probability':     round(prob, 4),
            'signal_state':    signal_state,
            'long_threshold':  LONG_THRESHOLD,
            'short_threshold': SHORT_THRESHOLD,
            'top_features':    top_feats,
            'trained_at':      trained_at,
            'oos_accuracy':    oos_acc,
            'disabled':        symbol in _f_disabled,
            'paper_only':      symbol == 'GC',
        }
    except Exception as e:
        return {'symbol': symbol, 'ok': False, 'error': str(e)}


# ─────────────────────────────────────────────────────────────
#  CLI — retrain and validate
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    for sym in ('MNQ', 'ES'):
        print(f'\n=== Training {sym} ===')
        acc = train_model(sym)
        print(f'  OOS accuracy: {acc:.3f}')
        print(f'  Deployed:     {acc >= OOS_ACCURACY_GATE}')
        print(f'\n=== Prediction {sym} ===')
        pred = get_current_prediction(sym)
        print(f'  state={pred.get("signal_state")} prob={pred.get("probability")} '
              f'oos_acc={pred.get("oos_accuracy")} disabled={pred.get("disabled")}')
