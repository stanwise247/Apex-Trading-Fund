"""
APEX Setup F — Random Forest ML Signal
========================================
OOS Sharpe: NQ=54.21, ES=63.69, GC=40.40 (paper only)
Features: 11 — dominant: htf_bias (37%), ema50_dist (25%), roc5 (8%)
Retrain: monthly, rolling 6-month window
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import db as _db

logger  = logging.getLogger('APEX.SetupF')

SESSION_WINDOWS = {
    'MNQ': (13, 19),
    'NQ':  (13, 19),  # kept for legacy reference
    'ES':  (13, 19),
    'GC':  (12, 17),
}

# Signal frequency tracking — warns if > 3 F signals fire per session per instrument
_f_session_counts: dict = {}   # (symbol, date_str) -> int

# Module-level caches
_hurst_cache = {}   # symbol -> (computed_at, values_series)
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
#  HURST EXPONENT
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


def _get_hurst_series(symbol: str) -> pd.Series:
    """Rolling 500-bar Hurst on 1h log returns. Cached for 30 minutes."""
    cache_key = symbol
    now = datetime.now(timezone.utc)
    if cache_key in _hurst_cache:
        cached_at, series = _hurst_cache[cache_key]
        if (now - cached_at).total_seconds() < 1800:
            return series

    df = _load_ohlcv(symbol, '1hour', limit=1500)
    if df.empty or len(df) < 510:
        s = pd.Series(dtype=float)
        _hurst_cache[cache_key] = (now, s)
        return s

    log_ret = np.log(df['close'] / df['close'].shift(1)).dropna().values
    window  = 500
    hursts  = []
    indices = []
    for i in range(window, len(log_ret) + 1):
        h = _hurst_rs(log_ret[i-window:i])
        hursts.append(h)
        indices.append(df['ts'].iloc[i])

    series = pd.Series(hursts, index=indices)
    _hurst_cache[cache_key] = (now, series)
    return series


def _get_current_hurst(symbol: str) -> float:
    series = _get_hurst_series(symbol)
    if series.empty:
        return 0.55
    return float(series.iloc[-1])


# ─────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high  = df['high']
    low   = df['low']
    prev  = df['close'].shift(1)
    tr    = pd.concat([
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


def calculate_features(symbol: str, df_5m: pd.DataFrame,
                        df_4h: pd.DataFrame = None, df_1h: pd.DataFrame = None) -> np.ndarray:
    """
    Build feature matrix entirely from df_5m — no HTF DB dependency.
    Returns array shape (n_bars, 11) aligned to df_5m.
    Requires df_5m to have columns: ts, open, high, low, close, volume, dt
    df_4h and df_1h params kept for API compatibility but ignored.
    """
    n = len(df_5m)
    feats = np.full((n, 11), np.nan)

    # df_5m must have a DatetimeIndex for resample — use dt column as index temporarily
    df_idx = df_5m.set_index('dt')

    # ── 1. consecutive ─────────────────────────────────────────
    feats[:, 0] = _consecutive(df_5m['close']).values

    # ── 2. htf_bias — resample 5min → 4h in memory, SMA20, map back ──
    df_4h_mem = df_idx['close'].resample('4h').last().dropna().to_frame('close')
    df_4h_mem['sma20'] = df_4h_mem['close'].rolling(20).mean()
    df_4h_mem['ts']    = df_4h_mem.index.map(lambda x: int(x.timestamp()))
    _5m_ts = pd.DataFrame({'ts': df_5m['ts'].values})
    merged = pd.merge_asof(_5m_ts, df_4h_mem[['ts','close','sma20']].sort_values('ts'),
                            on='ts', direction='backward')
    mid  = merged['sma20'] * 0.001
    feats[:, 1] = np.where(merged['sma20'].isna(), np.nan,
                  np.where(merged['close'] > merged['sma20'] + mid,  1.0,
                  np.where(merged['close'] < merged['sma20'] - mid, -1.0, 0.0)))

    # ── 3. ema50_dist (5min EMA50, ATR14 normalised) ──────────
    ema50   = _ema(df_5m['close'], 50)
    atr14   = _atr(df_5m, 14)
    feats[:, 2] = ((df_5m['close'] - ema50) / (atr14 + 1e-9)).values

    # ── 4. roc5 ────────────────────────────────────────────────
    feats[:, 3] = (df_5m['close'].pct_change(5)).values

    # ── 5. hour ────────────────────────────────────────────────
    feats[:, 4] = df_5m['dt'].dt.hour.values.astype(float)

    # ── 6. vol_regime — resample 5min → 1h in memory, rolling std ──
    df_1h_mem = df_idx['close'].resample('1h').last().dropna().to_frame('close')
    log_ret   = np.log(df_1h_mem['close'] / df_1h_mem['close'].shift(1))
    rvol      = log_ret.rolling(20).std()
    p25, p75  = rvol.quantile(0.25), rvol.quantile(0.75)
    vr        = pd.Series(0.0, index=df_1h_mem.index)
    vr[rvol > p75] = 1.0
    vr[rvol < p25] = -1.0
    df_1h_mem['vr'] = vr
    df_1h_mem['ts'] = df_1h_mem.index.map(lambda x: int(x.timestamp()))
    merged_vr = pd.merge_asof(_5m_ts, df_1h_mem[['ts','vr']].sort_values('ts'),
                               on='ts', direction='backward')
    feats[:, 5] = merged_vr['vr'].fillna(0.0).values

    # ── 7. atr_ratio (ATR5 / ATR20) ───────────────────────────
    atr5  = _atr(df_5m, 5)
    atr20 = _atr(df_5m, 20)
    feats[:, 6] = (atr5 / (atr20 + 1e-9)).values

    # ── 8. efficiency (last bar body/range) ───────────────────
    feats[:, 7] = (
        (df_5m['close'] - df_5m['open']).abs() /
        (df_5m['high'] - df_5m['low'] + 1e-9)
    ).values

    # ── 9. dow ─────────────────────────────────────────────────
    feats[:, 8] = df_5m['dt'].dt.dayofweek.values.astype(float)

    # ── 10. vol_ratio ──────────────────────────────────────────
    vol_ma = df_5m['volume'].rolling(20).mean()
    feats[:, 9] = (df_5m['volume'] / (vol_ma + 1e-9)).values

    # ── 11. hurst_regime — rolling Hurst on 1h log returns from 5min ──
    # Reuse df_1h_mem log returns already computed above
    lr_vals = log_ret.dropna().values
    window  = min(500, len(lr_vals))
    if window >= 50:
        h_vals = []
        for i in range(window, len(lr_vals) + 1):
            h_vals.append(_hurst_rs(lr_vals[i-window:i]))
        # Map each hurst value back to the corresponding 1h bar ts
        valid_1h = df_1h_mem.iloc[log_ret.isna().sum() + window - 1:]
        h_ts = valid_1h.index[:len(h_vals)].map(lambda x: int(x.timestamp())).values
        h_df = pd.DataFrame({'ts': h_ts, 'hv': h_vals})
        merged_h = pd.merge_asof(_5m_ts, h_df.sort_values('ts'),
                                  on='ts', direction='backward')
        feats[:, 10] = np.where(merged_h['hv'].isna(), 0.0,
                                np.where(merged_h['hv'] > 0.6, 1.0, 0.0))
    else:
        feats[:, 10] = 0.0

    return feats


FEATURE_NAMES = [
    'consecutive', 'htf_bias', 'ema50_dist', 'roc5', 'hour',
    'vol_regime', 'atr_ratio', 'efficiency', 'dow', 'vol_ratio', 'hurst_regime'
]


# ─────────────────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────────────────

def train_model(symbol: str) -> float:
    """Train RF classifier on last 6 months of 5min bars. Save to pkl."""
    from sklearn.ensemble import RandomForestClassifier

    logger.info(f'Training Setup F model for {symbol}...')
    start_hr, end_hr = SESSION_WINDOWS.get(symbol, (13, 19))

    # Load 5min bars only — HTF features derived in-memory from these
    df_5m = _load_ohlcv(symbol, '5min', limit=25000)

    if df_5m.empty or len(df_5m) < 500:
        logger.warning(f'Insufficient 5min data for {symbol}: {len(df_5m)} bars')
        return 0.0

    logger.info(f'  {symbol}: {len(df_5m)} 5min bars loaded (HTF computed in-memory)')

    # 6-month window — keep full contiguous df for feature calculation
    six_months_ago = int((datetime.now(timezone.utc) - timedelta(days=180)).timestamp())
    df_5m = df_5m[df_5m['ts'] >= six_months_ago].copy().reset_index(drop=True)

    # Session mask computed here but NOT applied to df yet — applying it before
    # calculate_features() produces non-contiguous bars that break the 4h resample
    # (HTF features become 100% NaN → model trains on near-zero rows → prob=0.500 exactly)
    session_mask = df_5m['dt'].dt.hour.between(start_hr, end_hr - 1).values

    logger.info(f'  {symbol}: {len(df_5m)} bars in last 6 months, {session_mask.sum()} session bars ({start_hr}-{end_hr} UTC)')

    if session_mask.sum() < 100:
        logger.warning(f'Insufficient session bars for {symbol}: {session_mask.sum()} (need 100)')
        return 0.0

    # Build features on full contiguous df — HTF resample needs unbroken 5min series
    X = calculate_features(symbol, df_5m)

    # Target: forward 12-bar return direction
    fwd = df_5m['close'].shift(-12)
    y   = (fwd > df_5m['close']).astype(int).values

    # Drop NaN rows AND restrict to session hours
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y.astype(float))) & session_mask
    X, y = X[valid], y[valid]

    if len(X) < 100:
        nan_counts = np.isnan(calculate_features(symbol, df_5m)).sum(axis=0)
        for i, (name, nans) in enumerate(zip(FEATURE_NAMES, nan_counts)):
            if nans > 0:
                logger.warning(f'  Feature "{name}": {nans}/{len(df_5m)} NaN values')
        logger.warning(f'Too few valid rows for {symbol}: {len(X)}')
        return 0.0

    # Train
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X, y)
    acc = model.score(X, y)

    # Save
    pkl_path = f'apex_rf_{symbol}.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump({'model': model, 'trained_at': datetime.now(timezone.utc).isoformat(),
                     'symbol': symbol, 'accuracy': acc}, f)
    logger.info(f'Model saved to {pkl_path} — train accuracy={acc:.3f}')
    return acc


def load_or_train_model(symbol: str):
    """Load model from pkl if < 30 days old, else retrain."""
    now = datetime.now(timezone.utc)
    # Check in-memory cache (5-minute TTL)
    if symbol in _model_cache:
        cached_at, model = _model_cache[symbol]
        if (now - cached_at).total_seconds() < 300:
            return model

    pkl_path = f'apex_rf_{symbol}.pkl'
    if os.path.exists(pkl_path):
        mtime = datetime.fromtimestamp(os.path.getmtime(pkl_path), tz=timezone.utc)
        age_days = (now - mtime).days
        if age_days < 30:
            try:
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                model = data['model']
                _model_cache[symbol] = (now, model)
                return model
            except Exception as e:
                logger.warning(f'Failed to load {pkl_path}: {e}')

    # Retrain
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
    Returns signal dict or None if no edge.
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    # Weekend check
    if dt.weekday() >= 5:
        return None

    # Session check
    start_hr, end_hr = SESSION_WINDOWS.get(symbol, (13, 19))
    if not (start_hr <= dt.hour < end_hr):
        return None

    # Load model
    model = load_or_train_model(symbol)
    if model is None:
        logger.warning(f'No model available for {symbol}')
        return None

    # Load 5min bars — 2000 bars needed for HTF resample (SMA20 on 4h requires 20 4h bars)
    df_5m = _load_ohlcv(symbol, '5min', limit=2000)

    if df_5m.empty or len(df_5m) < 60:
        return None

    # Build features on full contiguous df — session filter must NOT be applied here
    # (non-contiguous bars break 4h resample → 100% NaN → silent return None)
    X_all = calculate_features(symbol, df_5m)
    X_last = X_all[-1:].copy()

    if np.isnan(X_last).any():
        logger.warning(f'Setup F {symbol}: NaN in features — skipping')
        return None

    # Predict
    prob = float(model.predict_proba(X_last)[0, 1])
    logger.info(f'Setup F {symbol} scan: prob={prob:.3f} threshold=0.58')

    if prob > 0.58:
        direction = 'long'
    elif prob < 0.42:
        direction = 'short'
    else:
        return None  # No edge

    # Price data
    last = df_5m.iloc[-1]
    atr14 = _atr(df_5m, 14).iloc[-1]
    if pd.isna(atr14) or atr14 <= 0:
        return None

    entry  = float(last['close'])
    stop   = entry - 1.5 * atr14 if direction == 'long' else entry + 1.5 * atr14
    target = entry + 3.75 * atr14 if direction == 'long' else entry - 3.75 * atr14  # 1.5 stop × 2.5 RR
    rr     = round(abs(target - entry) / abs(entry - stop), 2)

    # Top features by importance (use model feature importances)
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:3]
    top_feats = [
        {'name': FEATURE_NAMES[i], 'importance': round(float(importances[i]), 3),
         'value': round(float(X_last[0, i]), 4)}
        for i in top_idx
    ]

    # Signal frequency monitoring — warn if > 3 F signals fire per session per instrument
    _date_key = (symbol, dt.strftime('%Y-%m-%d'))
    _f_session_counts[_date_key] = _f_session_counts.get(_date_key, 0) + 1
    if _f_session_counts[_date_key] > 3:
        logger.warning(
            f'Setup F: high signal frequency — {symbol} has fired '
            f'{_f_session_counts[_date_key]} times today — possible over-triggering'
        )

    return {
        'symbol':      symbol,
        'direction':   direction,
        'setup':       'F_rf_ml',
        'mode':        'intraday',
        'entry':       round(entry, 2),
        'stop':        round(stop, 2),
        'target':      round(target, 2),
        'rr':          rr,
        'session':     'NY Primary',
        'quality':     'primary',
        'confidence':  prob,
        'top_features': top_feats,
        'atr':         round(float(atr14), 2),
        'timestamp':   dt.isoformat(),
    }


# ─────────────────────────────────────────────────────────────
#  ALERT FORMATTER
# ─────────────────────────────────────────────────────────────

def format_f_alert(signal: dict) -> str:
    """Format Setup F Telegram alert."""
    from zoneinfo import ZoneInfo
    NY  = ZoneInfo('America/New_York')
    now = datetime.now(timezone.utc).astimezone(NY).strftime('%Y-%m-%d %H:%M')
    sym = signal['symbol']
    dir_= signal['direction'].upper()
    conf = signal['confidence']
    sep  = chr(9473) * 20

    emoji = chr(128200) if dir_ == 'LONG' else chr(128201)
    entry  = signal['entry']
    stop   = signal['stop']
    target = signal['target']
    stop_pts   = abs(round(entry - stop,   2))
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
        f'<b>Confidence:</b> {conf:.0%}',
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
        f'<i>ML model — 63.8% WR OOS | Sharpe 53.39</i>',
    ]
    return chr(10).join(parts)


# ─────────────────────────────────────────────────────────────
#  DEGRADATION CHECK
# ─────────────────────────────────────────────────────────────

def check_model_degradation(symbol: str) -> bool:
    """
    Returns True if model accuracy on last 30 closed Setup F trades < 55%.
    Triggers retrain warning.
    """
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
            return False  # Not enough data to assess
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

        # Need 2000 bars so 4h resample produces 30+ bars for SMA20 (20 × 48 = 960 5min minimum)
        df_5m = _load_ohlcv(symbol, '5min', limit=2000)

        if df_5m.empty:
            return {'symbol': symbol, 'ok': False, 'error': 'No data'}

        X_all  = calculate_features(symbol, df_5m)
        X_last = X_all[-1:]

        if np.isnan(X_last).any():
            return {'symbol': symbol, 'ok': False, 'error': 'NaN in features'}

        prob = float(model.predict_proba(X_last)[0, 1])
        if prob > 0.58:
            signal_state = 'SIGNAL_LONG'
        elif prob < 0.42:
            signal_state = 'SIGNAL_SHORT'
        elif prob > 0.50:
            signal_state = 'WATCHING_LONG'
        elif prob < 0.50:
            signal_state = 'WATCHING_SHORT'
        else:
            signal_state = 'NO_EDGE'

        importances = model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:3]
        top_feats = [
            {'name': FEATURE_NAMES[i], 'importance': round(float(importances[i]), 3),
             'value': round(float(X_last[0, i]), 4)}
            for i in top_idx
        ]

        pkl_path = f'apex_rf_{symbol}.pkl'
        trained_at = None
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, 'rb') as f:
                    d = pickle.load(f)
                trained_at = d.get('trained_at')
            except Exception:
                pass

        return {
            'symbol':         symbol,
            'ok':             True,
            'probability':    round(prob, 4),
            'signal_state':   signal_state,
            'long_threshold': 0.58,
            'short_threshold': 0.42,
            'top_features':   top_feats,
            'trained_at':     trained_at,
            'paper_only':     symbol == 'GC',
        }
    except Exception as e:
        return {'symbol': symbol, 'ok': False, 'error': str(e)}


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    for sym in ('NQ', 'ES', 'GC'):
        print(f'\n=== Training {sym} ===')
        acc = train_model(sym)
        print(f'  Accuracy: {acc:.3f}')
        print(f'\n=== Prediction {sym} ===')
        pred = get_current_prediction(sym)
        print(f'  {pred}')
