"""
APEX Setup I — Mathematical Alpha
===================================
Completely standalone strategy based purely on mathematics.
No price action patterns, candlesticks, order blocks, or existing setup logic.

Two independent mathematical signals must both agree before a trade fires:
  Component 1: Regime Transition Detector (CHOPPY→TRENDING crossover)
  Component 2: ML Ensemble (XGBoost + LogisticRegression on 6 pure-math features)

ISOLATION RULE:
  Do NOT import this module from any existing setup file.
  Only server.py adds the scheduler call and signal wiring.
  Do NOT use .asi8 anywhere.

MODEL FILES: apex_xi_MNQ_short.pkl / apex_xi_MNQ_long.pkl (direction-specific)

LOG_TRADE CONTRACT (enforced in server.py):
  Required signal keys: symbol, direction, setup, mode, entry, stop, target, rr, session, quality
  Order:  log_trade(sig) FIRST — then send_telegram()
  On failure: raises RuntimeError (trade_tracker.py) → caught as CRITICAL with exc_info=True
  setup name: 'I_mathematical_alpha'
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import db as _db

logger = logging.getLogger('APEX.SetupI')

# ─── XGBoost with sklearn fallback ───────────────────────────
try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except Exception:
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
    _XGBOOST_AVAILABLE = False
    logger.warning('xgboost not available — using GradientBoostingClassifier')

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ─── Constants ───────────────────────────────────────────────
SESSION_START = 13
SESSION_END   = 19
ALLOWED_DAYS  = {1, 2, 3}           # Tue / Wed / Thu

XGB_LONG_THRESHOLD  = 0.58   # lowered from 0.62 — market strongly trending, need wider gate
XGB_SHORT_THRESHOLD = 0.42   # raised from 0.38 — symmetric with long threshold
LR_LONG_THRESHOLD   = 0.58
LR_SHORT_THRESHOLD  = 0.42
OOS_AUC_GATE        = 0.54

FEATURE_NAMES_I = [
    'hurst',            # 0  rolling R/S Hurst on 100 close prices
    'autocorr_1',       # 1  lag-1 autocorrelation (50-bar window)
    'autocorr_3',       # 2  lag-3 autocorrelation
    'autocorr_5',       # 3  lag-5 autocorrelation
    'vol_ratio',        # 4  realised-20 / historical-100 vol ratio
    'entropy',          # 5  Shannon entropy of discretised returns (5 bins)
    'htf_momentum',     # 6  (close_1h - ema20_1h) / atr_1h — directional bias
    'vwap_position',    # 7  (close - session_vwap) / atr_5min — intraday bias
    'return_persistence',# 8 fraction of last 10 bars with positive returns
]

# Module-level state
_i_model_cache:    dict = {}   # symbol -> (loaded_at, xgb_short, xgb_long, lr, scaler)
_i_disabled_short: set  = set()   # symbols where short model failed OOS gate
_i_disabled_long:  set  = set()   # symbols where long model failed OOS gate
_i_dedup_sent:     dict = {}   # key -> timestamp
_i_session_counts: dict = {}   # (symbol, date_str) -> int


# ─────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────

def _load_5min(symbol: str, limit: int = 2000) -> pd.DataFrame:
    conn = _db.connect()
    try:
        cur = conn.execute(
            'SELECT ts, open, high, low, close, volume FROM ohlcv '
            'WHERE symbol=? AND timeframe=? ORDER BY ts DESC LIMIT ?',
            (symbol, '5min', limit)
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return pd.DataFrame(columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.iloc[::-1].reset_index(drop=True)
    df['dt'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    return df


# ─────────────────────────────────────────────────────────────
#  HURST R/S — takes close prices directly (not log returns)
# ─────────────────────────────────────────────────────────────

def hurst_rs(prices, min_n: int = 8) -> float:
    """R/S Hurst on price array. Compatible with raw=True rolling.apply."""
    prices = np.array(prices, dtype=float)
    if len(prices) < 32:
        return 0.5
    log_returns = np.diff(np.log(prices))
    ns = [n for n in [8, 16, 32, 64] if n <= len(log_returns)]
    rs_values = []
    for n in ns:
        chunks = [log_returns[i:i+n] for i in range(0, len(log_returns)-n+1, n)]
        rs_chunk = []
        for chunk in chunks:
            mean = np.mean(chunk)
            dev = np.cumsum(chunk - mean)
            r = np.max(dev) - np.min(dev)
            s = np.std(chunk, ddof=1)
            if s > 0:
                rs_chunk.append(r / s)
        if rs_chunk:
            rs_values.append((np.log(n), np.log(np.mean(rs_chunk))))
    if len(rs_values) < 2:
        return 0.5
    x = np.array([v[0] for v in rs_values])
    y = np.array([v[1] for v in rs_values])
    slope = np.polyfit(x, y, 1)[0]
    return float(np.clip(slope, 0.0, 1.0))


def _entropy_i(returns: np.ndarray, n_bins: int = 5) -> float:
    """Shannon entropy of discretised returns."""
    if len(returns) < 5:
        return 0.0
    counts, _ = np.histogram(returns, bins=n_bins)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-np.sum(probs * np.log(probs)))


def _atr_i(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low  = df['low']
    prev = df['close'].shift(1)
    tr   = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ─────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING — 9 features (6 regime + 3 directional)
# ─────────────────────────────────────────────────────────────

def calculate_features_i(df_5m: pd.DataFrame, df_5m_es: pd.DataFrame = None) -> np.ndarray:
    """
    Build 9-feature matrix from 5min bars.
    Features 0-5: pure-math regime measures.
    Features 6-8: directional context (HTF momentum, VWAP position, return persistence).
    All computation in-memory — no DB reads.
    Returns array shape (n, 9).
    """
    n      = len(df_5m)
    out    = np.full((n, 9), np.nan)
    closes = df_5m['close'].values.astype(float)

    lr = pd.Series(
        np.concatenate([[np.nan], np.log(np.maximum(closes[1:], 1e-10) /
                                         np.maximum(closes[:-1], 1e-10))]),
        index=df_5m.index
    )

    # ── Regime features (0-5) ─────────────────────────────────

    # 0. hurst — rolling R/S on 100 close prices
    out[:, 0] = df_5m['close'].rolling(100, min_periods=32).apply(
        hurst_rs, raw=True
    ).values

    # 1-3. autocorr lag 1, 3, 5
    out[:, 1] = lr.rolling(50, min_periods=10).corr(lr.shift(1)).values
    out[:, 2] = lr.rolling(50, min_periods=10).corr(lr.shift(3)).values
    out[:, 3] = lr.rolling(50, min_periods=10).corr(lr.shift(5)).values

    # 4. vol_ratio — realised-20 / historical-100 vol
    std20  = lr.rolling(20,  min_periods=10).std()
    std100 = lr.rolling(100, min_periods=20).std()
    out[:, 4] = (std20 / (std100 + 1e-10)).values

    # 5. entropy — Shannon entropy of discretised returns
    def _ent_window(x):
        clean = x[~np.isnan(x)]
        return _entropy_i(clean) if len(clean) >= 5 else np.nan

    out[:, 5] = lr.rolling(50, min_periods=10).apply(_ent_window, raw=True).values

    # ── Directional features (6-8) ────────────────────────────

    has_dt = 'dt' in df_5m.columns

    # 6. htf_momentum — (close_1h - ema20_1h) / atr_1h
    if has_dt:
        try:
            df_1h = df_5m.set_index('dt')[['high', 'low', 'close']].resample('1h').agg(
                {'high': 'max', 'low': 'min', 'close': 'last'}
            ).dropna()
            if len(df_1h) >= 20:
                prev_c = df_1h['close'].shift(1)
                tr_1h  = pd.concat([
                    df_1h['high'] - df_1h['low'],
                    (df_1h['high'] - prev_c).abs(),
                    (df_1h['low']  - prev_c).abs(),
                ], axis=1).max(axis=1)
                atr_1h   = tr_1h.rolling(14, min_periods=5).mean()
                ema20_1h = df_1h['close'].ewm(span=20, adjust=False).mean()
                htf_mom  = (df_1h['close'] - ema20_1h) / (atr_1h + 1e-10)

                htf_df = pd.DataFrame({'dt': df_1h.index, 'htf_momentum': htf_mom.values}).dropna()
                merged = pd.merge_asof(
                    df_5m[['dt']].reset_index(drop=True),
                    htf_df,
                    on='dt', direction='backward'
                )
                out[:, 6] = merged['htf_momentum'].values
        except Exception:
            pass

    # 7. vwap_position — (close - session_vwap) / atr_5min, 0 outside session
    if has_dt:
        try:
            atr5     = _atr_i(df_5m, 14).values
            hour_arr = df_5m['dt'].dt.hour.values
            date_arr = df_5m['dt'].dt.strftime('%Y-%m-%d').values
            vol_arr  = df_5m['volume'].values.astype(float)
            pv_arr   = closes * vol_arr
            in_sess  = (hour_arr >= SESSION_START) & (hour_arr < SESSION_END)

            vwap_pos = np.zeros(n)
            prev_d, cum_pv, cum_v = '', 0.0, 0.0
            for i in range(n):
                if not in_sess[i]:
                    continue
                d = date_arr[i]
                if d != prev_d:
                    cum_pv, cum_v, prev_d = 0.0, 0.0, d
                cum_pv += pv_arr[i]
                cum_v  += max(vol_arr[i], 0.0)
                if cum_v > 0 and not np.isnan(atr5[i]) and atr5[i] > 0:
                    vwap_pos[i] = (closes[i] - cum_pv / cum_v) / atr5[i]
            out[:, 7] = vwap_pos
        except Exception:
            pass

    # 8. return_persistence — fraction of last 10 bars with positive returns
    out[:, 8] = (lr > 0).rolling(10, min_periods=5).mean().values

    return out


# ─────────────────────────────────────────────────────────────
#  LABEL CONSTRUCTION — direction-specific
# ─────────────────────────────────────────────────────────────

def _build_labels_long_i(df: pd.DataFrame, rr: float = 1.5,
                          stop_atr: float = 1.5, max_bars: int = 100) -> np.ndarray:
    """Label=1 if price rose RR×stop before stop (long wins), else 0, else NaN."""
    n      = len(df)
    atr14  = _atr_i(df, 14).values
    closes = df['close'].values
    highs  = df['high'].values
    lows   = df['low'].values
    y      = np.full(n, np.nan)
    for i in range(n - max_bars - 1):
        if np.isnan(atr14[i]) or atr14[i] <= 0:
            continue
        entry = closes[i]
        stop  = entry - stop_atr * atr14[i]
        tgt   = entry + stop_atr * rr * atr14[i]
        for j in range(i + 1, min(i + max_bars + 1, n)):
            if highs[j] >= tgt:
                y[i] = 1.0
                break
            if lows[j] <= stop:
                y[i] = 0.0
                break
    return y


def _build_labels_short_i(df: pd.DataFrame, rr: float = 1.5,
                           stop_atr: float = 1.5, max_bars: int = 100) -> np.ndarray:
    """Label=1 if price fell RR×stop before stop (short wins), else 0, else NaN."""
    n      = len(df)
    atr14  = _atr_i(df, 14).values
    closes = df['close'].values
    highs  = df['high'].values
    lows   = df['low'].values
    y      = np.full(n, np.nan)
    for i in range(n - max_bars - 1):
        if np.isnan(atr14[i]) or atr14[i] <= 0:
            continue
        entry = closes[i]
        stop  = entry + stop_atr * atr14[i]           # stop above entry
        tgt   = entry - stop_atr * rr * atr14[i]      # target below entry
        for j in range(i + 1, min(i + max_bars + 1, n)):
            if lows[j] <= tgt:
                y[i] = 1.0
                break
            if highs[j] >= stop:
                y[i] = 0.0
                break
    return y


# Backward-compat alias used by backtest script
_build_labels_i = _build_labels_long_i


# ─────────────────────────────────────────────────────────────
#  HTF 4h BIAS (in-memory from 5min resample)
# ─────────────────────────────────────────────────────────────

def _htf_4h_bias(df_5m: pd.DataFrame) -> str:
    """BULLISH / BEARISH / NEUTRAL — same in-memory method as existing setups."""
    try:
        df_idx  = df_5m.set_index('dt')
        df_4h   = df_idx['close'].resample('4h').last().dropna().to_frame('close')
        df_4h['sma20'] = df_4h['close'].rolling(20).mean()
        if df_4h['sma20'].dropna().empty:
            return 'NEUTRAL'
        last = df_4h.dropna().iloc[-1]
        band = last['sma20'] * 0.001
        if last['close'] > last['sma20'] + band:
            return 'BULLISH'
        if last['close'] < last['sma20'] - band:
            return 'BEARISH'
        return 'NEUTRAL'
    except Exception:
        return 'NEUTRAL'


# ─────────────────────────────────────────────────────────────
#  REGIME TRANSITION DETECTOR (Component 1)
# ─────────────────────────────────────────────────────────────

def detect_regime_transition(symbol: str) -> tuple:
    """
    Returns (transition_detected: bool, hint: str | None).
    Fires on either:
      1. CHOPPY→TRENDING crossover: Hurst crosses 0.55 from below, preceded by 3 CHOPPY bars
      2. Sustained TRENDING: regime is TRENDING AND confidence >= 0.67

    Both require autocorr > 0.05 and vol_ratio < 1.3 to confirm trending conditions.
    """
    try:
        conn = _db.connect()
        rows = conn.execute(
            'SELECT regime, hurst, autocorr, vol_ratio, confidence, timestamp '
            'FROM regime_log WHERE symbol=? ORDER BY timestamp DESC LIMIT 5',
            (symbol,)
        ).fetchall()
        conn.close()
    except Exception as e:
        logger.debug(f'SetupI transition check DB error ({symbol}): {e}')
        return False, None

    if not rows:
        return False, None

    latest = rows[0]
    lat_regime, lat_h, lat_ac, lat_vr, lat_conf, _ = latest

    # Must be in TRENDING regime with clean autocorr / vol
    if lat_regime != 'TRENDING':
        return False, None
    if lat_ac is None or lat_ac <= 0.05:
        return False, None
    if lat_vr is None or lat_vr >= 1.3:
        return False, None

    # ── Trigger 1: CHOPPY→TRENDING crossover ─────────────────────
    if (len(rows) >= 4 and lat_h is not None and lat_conf is not None and
            lat_conf >= 0.67):
        prev1, prev2, prev3 = rows[1], rows[2], rows[3]
        prv_h = prev1[1]
        if (prv_h is not None and prv_h < 0.55 and lat_h > 0.55 and
                all(r[0] == 'CHOPPY' for r in [prev1, prev2, prev3])):
            return True, 'choppy_trending_crossover'

    # ── Trigger 2: Sustained TRENDING with high confidence ────────
    if lat_conf is not None and lat_conf >= 0.67:
        return True, 'sustained_trending'

    return False, None


# ─────────────────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────────────────

def _make_xgb(pos_w: float):
    if _XGBOOST_AVAILABLE:
        return XGBClassifier(
            max_depth=4, n_estimators=200, learning_rate=0.05,
            scale_pos_weight=pos_w, min_child_weight=30,
            eval_metric='auc', random_state=42, n_jobs=-1
        )
    return XGBClassifier(
        max_depth=4, n_estimators=200, learning_rate=0.05,
        min_samples_leaf=30, random_state=42
    )


def train_model_i(symbol: str) -> dict:
    """
    Train direction-specific models (short XGB + long XGB + shared LR).
    Each model has its own OOS gate (AUC >= 0.54).
    Returns {'short_auc', 'long_auc', 'short_ok', 'long_ok'}.
    """
    from sklearn.metrics import roc_auc_score

    logger.info(f'Setup I: training direction models for {symbol}...')
    _fail = {'short_auc': 0.0, 'long_auc': 0.0, 'short_ok': False, 'long_ok': False}

    df_5m = _load_5min(symbol, limit=25000)
    if df_5m.empty or len(df_5m) < 500:
        df_5m = _load_5min('NQ', limit=25000)
        if df_5m.empty or len(df_5m) < 500:
            logger.warning(f'Setup I: insufficient 5min data for {symbol}')
            return _fail

    six_months_ago = int((datetime.now(timezone.utc) - timedelta(days=180)).timestamp())
    df_5m = df_5m[df_5m['ts'] >= six_months_ago].copy().reset_index(drop=True)

    session_mask = (
        df_5m['dt'].dt.hour.between(SESSION_START, SESSION_END - 1) &
        df_5m['dt'].dt.dayofweek.isin(ALLOWED_DAYS)
    ).values

    if session_mask.sum() < 100:
        logger.warning(f'Setup I: insufficient session bars for {symbol}: {session_mask.sum()}')
        return _fail

    logger.info(f'  {symbol}: computing 6 features...')
    X_all = calculate_features_i(df_5m)

    logger.info(f'  {symbol}: computing direction labels (RR=1.5)...')
    y_long_all  = _build_labels_long_i(df_5m,  rr=1.5, stop_atr=1.5, max_bars=100)
    y_short_all = _build_labels_short_i(df_5m, rr=1.5, stop_atr=1.5, max_bars=100)

    valid_mask = (
        ~np.isnan(X_all).any(axis=1) &
        ~np.isnan(y_long_all) &
        ~np.isnan(y_short_all) &
        session_mask
    )
    X   = X_all[valid_mask]
    y_l = y_long_all[valid_mask].astype(int)
    y_s = y_short_all[valid_mask].astype(int)

    logger.info(
        f'  {symbol}: {len(X)} samples | '
        f'long_pos={y_l.mean():.1%} | short_pos={y_s.mean():.1%}'
    )

    if len(X) < 100:
        logger.warning(f'Setup I: too few samples for {symbol}: {len(X)}')
        return _fail

    # Single fold: train months 1-5, test month 6
    n   = len(X)
    seg = n // 6
    segments = [range(i * seg, (i + 1) * seg) for i in range(6)]
    segments[-1] = range(5 * seg, n)

    tr_idx = np.concatenate([list(segments[s]) for s in [0, 1, 2, 3, 4]])
    te_idx = np.array(list(segments[5]))

    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr_l, y_te_l = y_l[tr_idx], y_l[te_idx]
    y_tr_s, y_te_s = y_s[tr_idx], y_s[te_idx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Shared LR trained on long labels (used as directional filter in scan)
    lr_clf = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)
    lr_clf.fit(X_tr_s, y_tr_l)

    # SHORT model — label=1 means price fell (short wins)
    pos_w_s = float((y_tr_s == 0).sum()) / float((y_tr_s == 1).sum() + 1e-9)
    xgb_short = _make_xgb(pos_w_s)
    xgb_short.fit(X_tr_s, y_tr_s)

    # LONG model — label=1 means price rose (long wins)
    pos_w_l = float((y_tr_l == 0).sum()) / float((y_tr_l == 1).sum() + 1e-9)
    xgb_long = _make_xgb(pos_w_l)
    xgb_long.fit(X_tr_s, y_tr_l)

    # OOS AUC per direction
    short_probs = xgb_short.predict_proba(X_te_s)[:, 1]
    long_probs  = xgb_long.predict_proba(X_te_s)[:, 1]

    try:
        short_auc = float(roc_auc_score(y_te_s, short_probs))
    except Exception:
        short_auc = 0.5
    try:
        long_auc = float(roc_auc_score(y_te_l, long_probs))
    except Exception:
        long_auc = 0.5

    short_ok = short_auc >= OOS_AUC_GATE
    long_ok  = long_auc  >= OOS_AUC_GATE

    logger.info(
        f'Setup I: {symbol} short AUC={short_auc:.3f} ({"PASS" if short_ok else "FAIL"}) | '
        f'long AUC={long_auc:.3f} ({"PASS" if long_ok else "FAIL"})'
    )
    logger.info(f'Setup I: {symbol} active directions — SHORT={short_ok} LONG={long_ok}')

    # Train final models on ALL data for each direction that passed
    now_str = datetime.now(timezone.utc).isoformat()
    scaler_f = StandardScaler()
    X_sf = scaler_f.fit_transform(X)

    lr_f = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)
    lr_f.fit(X_sf, y_l)

    def _save(direction, xgb_model, oos_auc):
        path = f'apex_xi_{symbol}_{direction}.pkl'
        with open(path, 'wb') as fh:
            pickle.dump({
                'xgb': xgb_model, 'lr': lr_f, 'scaler': scaler_f,
                'direction': direction, 'oos_auc': oos_auc,
                'trained_at': now_str, 'symbol': symbol,
                'n_features': len(FEATURE_NAMES_I),
            }, fh)
        logger.info(f'Setup I: saved {path} (AUC={oos_auc:.3f})')

    if short_ok:
        _i_disabled_short.discard(symbol)
        xgb_s_f = _make_xgb(pos_w_s)
        xgb_s_f.fit(X_sf, y_s)
        _save('short', xgb_s_f, short_auc)
    else:
        _i_disabled_short.add(symbol)
        logger.critical(
            f'Setup I: {symbol} SHORT failed gate (AUC={short_auc:.3f} < {OOS_AUC_GATE})'
        )

    if long_ok:
        _i_disabled_long.discard(symbol)
        xgb_l_f = _make_xgb(pos_w_l)
        xgb_l_f.fit(X_sf, y_l)
        _save('long', xgb_l_f, long_auc)
    else:
        _i_disabled_long.add(symbol)
        logger.critical(
            f'Setup I: {symbol} LONG failed gate (AUC={long_auc:.3f} < {OOS_AUC_GATE})'
        )

    return {'short_auc': short_auc, 'long_auc': long_auc,
            'short_ok': short_ok, 'long_ok': long_ok}


def load_or_train_model_i(symbol: str):
    """Load (xgb_short, xgb_long, lr, scaler) from pkl or retrain both direction models."""
    now = datetime.now(timezone.utc)
    if symbol in _i_model_cache:
        cached_at, xs, xl, lr, sc = _i_model_cache[symbol]
        if (now - cached_at).total_seconds() < 300:
            return xs, xl, lr, sc

    def _load_pkl(path):
        if not os.path.exists(path):
            logger.warning(f'Setup I model not found: {path} — will trigger training')
            return None
        mtime = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
        if (now - mtime).days >= 30:
            logger.warning(f'Setup I model stale (>{(now-mtime).days}d): {path} — will retrain')
            return None
        try:
            with open(path, 'rb') as fh:
                d = pickle.load(fh)
            if d.get('n_features') != len(FEATURE_NAMES_I):
                raise ValueError(
                    f'feature count mismatch: pkl has {d.get("n_features")}, '
                    f'expected {len(FEATURE_NAMES_I)}'
                )
            return d
        except Exception as e:
            logger.error(f'Setup I failed to load {path}: {e}')
            return None

    short_d = _load_pkl(f'apex_xi_{symbol}_short.pkl')
    long_d  = _load_pkl(f'apex_xi_{symbol}_long.pkl')

    if short_d is None and long_d is None:
        logger.warning(f'Setup I {symbol}: no models loaded — triggering train_model_i()')
        try:
            train_model_i(symbol)
        except Exception as _train_err:
            logger.error(f'Setup I {symbol}: train_model_i() failed: {_train_err}', exc_info=True)
        short_d = _load_pkl(f'apex_xi_{symbol}_short.pkl')
        long_d  = _load_pkl(f'apex_xi_{symbol}_long.pkl')

    xgb_short = short_d['xgb'] if short_d else None
    xgb_long  = long_d['xgb']  if long_d  else None
    ref       = short_d or long_d
    lr_clf    = ref['lr']     if ref else None
    scaler    = ref['scaler'] if ref else None

    if xgb_short is None:
        logger.error(f'Setup I {symbol}: xgb_short model unavailable after load/train')
        _i_disabled_short.add(symbol)
    else:
        _i_disabled_short.discard(symbol)
    if xgb_long is None:
        logger.error(f'Setup I {symbol}: xgb_long model unavailable after load/train')
        _i_disabled_long.add(symbol)
    else:
        _i_disabled_long.discard(symbol)

    _i_model_cache[symbol] = (now, xgb_short, xgb_long, lr_clf, scaler)
    return xgb_short, xgb_long, lr_clf, scaler


# ─────────────────────────────────────────────────────────────
#  SCAN — combined signal gate
# ─────────────────────────────────────────────────────────────

def scan_setup_i(symbol: str, dt: datetime = None) :
    """
    Check Setup I signal for symbol using direction-specific models.
    SHORT: xgb_short > 0.62 AND lr < 0.38 AND HTF BEARISH
    LONG:  xgb_long  > 0.62 AND lr > 0.62 AND HTF BULLISH
    Returns signal dict or None.
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    if dt.weekday() >= 5:
        return None
    if dt.weekday() not in ALLOWED_DAYS:
        return None
    _sess_end = 20 if symbol == 'MNQ' else SESSION_END
    _in_sess = (SESSION_START <= dt.hour < _sess_end)
    if symbol == 'MNQ' and dt.hour == 19 and dt.minute >= 30:
        _in_sess = False  # MNQ session ends at 19:30 — liquidity thins after this
    if not _in_sess:
        return None

    date_str = dt.strftime('%Y-%m-%d')

    # Gate: max 2 Setup I signals per instrument per session
    sess_key = f'{symbol}_{date_str}'
    if _i_session_counts.get(sess_key, 0) >= 2:
        logger.debug(f'Setup I {symbol}: session cap reached (2/session)')
        return None

    # Both directions disabled → skip model load
    if symbol in _i_disabled_short and symbol in _i_disabled_long:
        logger.debug(f'Setup I {symbol}: both directions disabled')
        return None

    # ── Component 1: ML dual confirmation is primary gate ─────
    # Transition gate removed — XGB model incorporates regime features (hurst, vol_ratio,
    # autocorr) so hard-gating on regime double-counts it and blocks valid CHOPPY-market alpha.
    # Risk control: XGB > 0.62 AND LR confirmation + HTF bias gate below.
    logger.debug(f'Setup I {symbol}: transition gate removed — ML dual confirmation is primary gate')

    xgb_short, xgb_long, lr_clf, scaler = load_or_train_model_i(symbol)
    if xgb_short is None and xgb_long is None:
        logger.warning(f'Setup I {symbol}: no models available')
        return None

    df_5m = _load_5min(symbol, limit=2000)
    if df_5m.empty or len(df_5m) < 60:
        return None

    X_all  = calculate_features_i(df_5m)
    X_last = X_all[-1:]
    if np.isnan(X_last).any():
        logger.warning(f'Setup I {symbol}: NaN in features — skipping')
        return None

    # Validate feature count against whichever model exists
    ref_model = xgb_short or xgb_long
    if hasattr(ref_model, 'n_features_in_') and ref_model.n_features_in_ != len(FEATURE_NAMES_I):
        logger.warning(f'Setup I {symbol}: feature count mismatch — triggering retrain')
        _i_model_cache.pop(symbol, None)
        train_model_i(symbol)
        return None

    X_last_s = scaler.transform(X_last)
    lr_prob  = float(lr_clf.predict_proba(X_last_s)[0, 1])

    short_prob = float(xgb_short.predict_proba(X_last_s)[0, 1]) if xgb_short else None
    long_prob  = float(xgb_long.predict_proba(X_last_s)[0, 1])  if xgb_long  else None

    logger.info(
        f'Setup I {symbol}: short_xgb={short_prob} long_xgb={long_prob} lr={lr_prob:.3f} | '
        f'short gate: xgb>{XGB_LONG_THRESHOLD} AND lr<{LR_SHORT_THRESHOLD} | '
        f'long gate: xgb>{XGB_LONG_THRESHOLD} AND lr>{LR_LONG_THRESHOLD}'
    )

    # ── Component 2: Direction-specific gate ──────────────────
    if short_prob is not None and short_prob > XGB_LONG_THRESHOLD and lr_prob < LR_SHORT_THRESHOLD:
        direction = 'short'
    elif long_prob is not None and long_prob > XGB_LONG_THRESHOLD and lr_prob > LR_LONG_THRESHOLD:
        direction = 'long'
    else:
        return None

    # ── Gate: HTF 4h bias ─────────────────────────────────────
    bias = _htf_4h_bias(df_5m)
    bias_ok = (direction == 'long'  and bias == 'BULLISH') or \
              (direction == 'short' and bias == 'BEARISH')
    if not bias_ok:
        logger.info(f'Setup I {symbol}: {direction} blocked by 4h bias ({bias})')
        return None

    # ── Gate: Max 1 open Setup I trade per instrument ─────────
    try:
        conn = _db.connect()
        existing = conn.execute(
            "SELECT id FROM apex_trades WHERE symbol=? AND setup='I_mathematical_alpha' AND status='open' LIMIT 1",
            (symbol,)
        ).fetchone()
        conn.close()
        if existing:
            logger.info(f'Setup I {symbol}: open trade exists — skipping')
            return None
    except Exception as e:
        logger.debug(f'Setup I {symbol}: open trade check failed: {e}')

    # ── Gate: dedup ───────────────────────────────────────────
    dedup_key = f'{symbol}_I_{direction}_{date_str}'
    now_ts = datetime.now(timezone.utc).timestamp()
    if dedup_key in _i_dedup_sent and now_ts - _i_dedup_sent[dedup_key] < 86400:
        logger.debug(f'Setup I {symbol}: duplicate suppressed ({dedup_key})')
        return None
    _i_dedup_sent[dedup_key] = now_ts

    # ── Build signal ──────────────────────────────────────────
    last  = df_5m.iloc[-1]
    atr14 = _atr_i(df_5m, 14).iloc[-1]
    if pd.isna(atr14) or atr14 <= 0:
        return None

    entry  = float(last['close'])
    stop   = entry - 1.5 * atr14 if direction == 'long' else entry + 1.5 * atr14
    target = entry + 3.75 * atr14 if direction == 'long' else entry - 3.75 * atr14  # RR=2.5 live
    rr     = round(abs(target - entry) / abs(entry - stop), 2)

    feat_str = ', '.join(
        f'{FEATURE_NAMES_I[i]}={X_last[0, i]:.3f}' for i in range(len(FEATURE_NAMES_I))
    )
    logger.info(f'Setup I {symbol} signal features: {feat_str}')

    # Increment session cap counter before returning
    _i_session_counts[sess_key] = _i_session_counts.get(sess_key, 0) + 1

    active_xgb_prob = short_prob if direction == 'short' else long_prob
    return {
        'symbol':     symbol,
        'direction':  direction,
        'setup':      'I_mathematical_alpha',
        'mode':       'intraday',
        'entry':      round(entry, 2),
        'stop':       round(stop, 2),
        'target':     round(target, 2),
        'rr':         rr,
        'session':    'NY Primary',
        'quality':    'primary',
        'xgb_prob':   round(active_xgb_prob, 4),
        'lr_prob':    round(lr_prob, 4),
        'htf_bias':   bias,
        'hurst':      round(float(X_last[0, 0]), 3),
        'autocorr':   round(float(X_last[0, 1]), 3),
        'atr':        round(float(atr14), 2),
        'timestamp':  dt.isoformat(),
    }


def scan_all_i(dt: datetime = None, symbols: list = None) -> list:
    """Scan all instruments for Setup I signals. Called by server.py scheduler."""
    if symbols is None:
        symbols = ['MNQ', 'ES']
    results = []
    for sym in symbols:
        try:
            sig = scan_setup_i(sym, dt)
            if sig:
                results.append(sig)
        except Exception as e:
            logger.error(f'Setup I scan error {sym}: {e}')
    return results


# ─────────────────────────────────────────────────────────────
#  ALERT FORMATTER
# ─────────────────────────────────────────────────────────────

def format_i_alert(signal: dict) -> str:
    from zoneinfo import ZoneInfo
    NY    = ZoneInfo('America/New_York')
    now   = datetime.now(timezone.utc).astimezone(NY).strftime('%Y-%m-%d %H:%M')
    sym   = signal['symbol']
    dir_  = signal['direction'].upper()
    sep   = chr(9473) * 20
    emoji = '📈' if dir_ == 'LONG' else '📉'
    entry = signal['entry']
    stop  = signal['stop']
    tgt   = signal['target']
    xgb   = signal.get('xgb_prob', 0)
    lr    = signal.get('lr_prob', 0)
    h     = signal.get('hurst', 0)
    ac    = signal.get('autocorr', 0)

    parts = [
        f'{emoji} <b>APEX MATHEMATICAL SIGNAL — {sym}</b>',
        sep,
        f'<b>Setup:</b>      I — Mathematical Alpha',
        f'<b>Direction:</b>  {dir_}',
        f'<b>Regime:</b>     TRENDING (Mathematical)',
        f'<b>XGB:</b>        {xgb:.2f} | <b>LR:</b> {lr:.2f}',
        f'<b>H:</b>          {h:.2f} | <b>Autocorr:</b> {ac:.2f}',
        f'<b>HTF Bias:</b>   {signal.get("htf_bias", "—")}',
        sep,
        f'<b>Entry:</b>      {entry:.2f}',
        f'<b>Stop:</b>       {stop:.2f} ({abs(round(entry - stop, 2))} pts)',
        f'<b>Target:</b>     {tgt:.2f} ({abs(round(entry - tgt, 2))} pts)',
        f'<b>R:R:</b>        {signal["rr"]}x',
        sep,
        f'<i>{now} ET</i>',
        f'<i>Setup I {sym} {dir_} | Math Alpha | XGB={xgb:.2f} LR={lr:.2f} | '
        f'H={h:.2f} autocorr={ac:.2f} | entry={entry:.2f} stop={stop:.2f} target={tgt:.2f}</i>',
    ]
    return '\n'.join(parts)


# ─────────────────────────────────────────────────────────────
#  CURRENT STATE (for dashboard / scan endpoint)
# ─────────────────────────────────────────────────────────────

def get_setup_i_state(symbol: str) -> dict:
    """Return current Setup I state for dashboard display."""
    try:
        transition, _ = detect_regime_transition(symbol)
        xgb_short, xgb_long, lr_clf, scaler = load_or_train_model_i(symbol)

        result = {
            'symbol':         symbol,
            'ok':             (xgb_short is not None or xgb_long is not None),
            'transition':     transition,
            'short_enabled':  xgb_short is not None,
            'long_enabled':   xgb_long  is not None,
            'short_xgb_prob': None,
            'long_xgb_prob':  None,
            'lr_prob':        None,
        }

        if not (xgb_short or xgb_long):
            logger.error(f'Setup I {symbol}: both XGB models are None — probabilities unavailable')
        elif scaler is None:
            logger.error(f'Setup I {symbol}: scaler is None — cannot compute probabilities')
        else:
            df_5m = _load_5min(symbol, limit=600)  # 600 bars = ~50 1h bars; needed for htf_momentum (requires >=20 1h bars)
            if df_5m.empty:
                logger.error(f'Setup I {symbol}: _load_5min returned empty DataFrame — probabilities unavailable')
            else:
                X_all  = calculate_features_i(df_5m)
                X_last = X_all[-1:]
                nan_cols = int(np.isnan(X_last).sum())
                if nan_cols > 0:
                    logger.error(
                        f'Setup I {symbol}: {nan_cols}/{X_last.shape[1]} feature(s) are NaN '
                        f'in last row — probabilities unavailable'
                    )
                else:
                    X_s = scaler.transform(X_last)
                    result['lr_prob'] = round(float(lr_clf.predict_proba(X_s)[0, 1]), 4)
                    if xgb_short:
                        result['short_xgb_prob'] = round(float(xgb_short.predict_proba(X_s)[0, 1]), 4)
                    if xgb_long:
                        result['long_xgb_prob'] = round(float(xgb_long.predict_proba(X_s)[0, 1]), 4)

        for direction in ('short', 'long'):
            pkl = f'apex_xi_{symbol}_{direction}.pkl'
            if os.path.exists(pkl):
                try:
                    with open(pkl, 'rb') as fh:
                        d = pickle.load(fh)
                    result[f'{direction}_trained_at'] = d.get('trained_at')
                    result[f'{direction}_oos_auc']    = d.get('oos_auc')
                except Exception:
                    pass

        return result
    except Exception as e:
        return {'symbol': symbol, 'ok': False, 'error': str(e)}


# ─────────────────────────────────────────────────────────────
#  CLI — retrain and validate
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    for sym in ('MNQ', 'ES'):
        print(f'\n=== Training {sym} ===')
        result = train_model_i(sym)
        print(f'  Short AUC: {result["short_auc"]:.3f} | Deployed: {result["short_ok"]}')
        print(f'  Long  AUC: {result["long_auc"]:.3f} | Deployed: {result["long_ok"]}')
        print(f'  Active directions — SHORT={result["short_ok"]} LONG={result["long_ok"]}')
