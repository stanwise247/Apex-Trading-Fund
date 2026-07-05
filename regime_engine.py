"""
APEX Market Regime Engine — regime_engine.py
=============================================
OBSERVATION ONLY. Zero impact on live signal pipeline.

Do NOT import this module from:
  setup_engine.py, setup_e.py, setup_f_ml.py, fvg_engine.py, setup_h_vwap.py,
  live_scanner.py, or any file that gates real trades.

The engine classifies market regime every 5 minutes for MNQ and ES,
stores results in regime_log, and exposes a read-only API endpoint.
Data accumulates alongside live trades so we can later validate whether
regime predicted trade outcomes.

Three measures:
  1. Hurst Exponent (R/S Analysis) — last 100 5min log returns
  2. LAG-1 Autocorrelation — last 50 5min returns
  3. Volatility Ratio — realised 20-bar vs historical 100-bar vol
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timezone
import db as _db

logger = logging.getLogger('APEX.RegimeEngine')

# Annualisation factor for 5min bars: 252 trading days × 78 bars/day
_ANN = np.sqrt(252 * 78)

# Module-level: track last stored regime per symbol to detect changes
_last_regime: dict = {}   # symbol -> regime str


# ─────────────────────────────────────────────────────────────
#  DATACLASS
# ─────────────────────────────────────────────────────────────

@dataclass
class RegimeState:
    symbol:     str
    regime:     str    # 'TRENDING' / 'MEAN_REVERTING' / 'CHOPPY'
    hurst:      float
    autocorr:   float
    vol_ratio:  float
    confidence: float
    timestamp:  datetime
    reason:     str

    def as_dict(self) -> dict:
        return {
            'symbol':     self.symbol,
            'regime':     self.regime,
            'hurst':      round(self.hurst, 4),
            'autocorr':   round(self.autocorr, 4),
            'vol_ratio':  round(self.vol_ratio, 4),
            'confidence': round(self.confidence, 2),
            'timestamp':  self.timestamp.isoformat(),
            'reason':     self.reason,
        }


# ─────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────

def _load_5min(symbol: str, limit: int = 200) -> pd.DataFrame:
    """Load 5min OHLCV bars from DB, sorted oldest-first."""
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
    return df.iloc[::-1].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
#  MEASURE 1 — HURST EXPONENT (R/S Analysis)
# ─────────────────────────────────────────────────────────────

def _hurst_rs(log_returns: np.ndarray) -> float:
    """
    Rescaled Range (R/S) analysis on log_returns array.
    Sub-periods n in [8, 16, 32, 64].
    Returns H: >0.55=trending, <0.45=mean-reverting, 0.45-0.55=choppy.
    """
    sub_periods = [8, 16, 32, 64]
    rs_vals = []
    valid_n = []

    for n in sub_periods:
        if len(log_returns) < n * 2:
            continue
        chunks = [log_returns[i:i + n] for i in range(0, len(log_returns) - n, n)]
        chunk_rs = []
        for chunk in chunks:
            if len(chunk) < n:
                continue
            m = np.mean(chunk)
            devs = np.cumsum(chunk - m)
            R = devs.max() - devs.min()
            S = np.std(chunk, ddof=1)
            if S > 0:
                chunk_rs.append(R / S)
        if chunk_rs:
            rs_vals.append(np.mean(chunk_rs))
            valid_n.append(n)

    if len(rs_vals) < 3:
        return 0.5  # insufficient data — neutral

    log_n  = np.log(valid_n)
    log_rs = np.log(rs_vals)
    H = float(np.polyfit(log_n, log_rs, 1)[0])
    return float(np.clip(H, 0.1, 0.9))


# ─────────────────────────────────────────────────────────────
#  MEASURE 2 — LAG-1 AUTOCORRELATION
# ─────────────────────────────────────────────────────────────

def _autocorr_lag1(log_returns: np.ndarray) -> float:
    """LAG-1 autocorrelation of last 50 5min returns."""
    series = pd.Series(log_returns[-50:])
    if len(series) < 10:
        return 0.0
    ac = float(series.autocorr(lag=1))
    return ac if not np.isnan(ac) else 0.0


# ─────────────────────────────────────────────────────────────
#  MEASURE 3 — VOLATILITY RATIO
# ─────────────────────────────────────────────────────────────

def _vol_ratio(log_returns: np.ndarray) -> float:
    """
    realised  = std of last 20 returns × sqrt(252×78)
    historical = std of last 100 returns × sqrt(252×78)
    ratio = realised / historical
    >1.3 = elevated (choppy hard override at 1.5)
    <0.7 = compressed (calm/mean-reverting)
    """
    if len(log_returns) < 21:
        return 1.0
    r20  = np.std(log_returns[-20:], ddof=1)
    r100 = np.std(log_returns[-100:] if len(log_returns) >= 100 else log_returns, ddof=1)
    if r100 < 1e-10:
        return 1.0
    return float(r20 / r100)


# ─────────────────────────────────────────────────────────────
#  CLASSIFICATION
# ─────────────────────────────────────────────────────────────

def classify(hurst: float, autocorr: float, vr: float,
             net_dir: float = 0.0) -> tuple[str, float, str]:
    """
    Returns (regime, confidence, reason).

    Four-vote classification:
      if vol_ratio > 1.5 → CHOPPY (hard override — erratic price action)
      trending_score = +1 H>0.55, +1 autocorr>0.05, +1 net_dir>0.60
      mean_rev_score = +1 H<0.45, +1 autocorr<-0.05
      if trending_score >= 2 → TRENDING   (any 2-of-3 directional measures)
      if mean_rev_score >= 2 → MEAN_REVERTING
      else → CHOPPY

    net_dir = |net_move| / session_range over last 100 bars.
    Catches staircase-down/up moves that fool lag-1 autocorr (alternating bars
    suppress autocorr even when price moves strongly in one direction).
    """
    # Hard override
    if vr > 1.5:
        h_choppy  = 1 if 0.45 <= hurst   <= 0.55 else 0
        ac_choppy = 1 if -0.05 <= autocorr <= 0.05 else 0
        vr_choppy = 1
        conf = (h_choppy + ac_choppy + vr_choppy) / 3
        return 'CHOPPY', round(conf, 2), f'vol_override(vr={vr:.2f}>1.5)'

    nd_vote = 1 if net_dir > 0.60 else 0
    trending_score = (1 if hurst > 0.55 else 0) + (1 if autocorr > 0.05 else 0) + nd_vote
    mean_rev_score = (1 if hurst < 0.45 else 0) + (1 if autocorr < -0.05 else 0)

    if trending_score >= 2:
        regime = 'TRENDING'
        h_vote  = 1 if hurst > 0.55 else 0
        ac_vote = 1 if autocorr > 0.05 else 0
        vr_vote = 1 if 0.7 <= vr <= 1.3 else 0
        conf    = (h_vote + ac_vote + nd_vote + vr_vote) / 4
        reason  = (
            f'H={hurst:.3f}>0.55,autocorr={autocorr:.3f}>0.05,net={net_dir:.2f}'
            if not nd_vote
            else f'H={hurst:.3f},autocorr={autocorr:.3f},net_dir={net_dir:.2f}>0.60'
        )
    elif mean_rev_score >= 2:
        regime = 'MEAN_REVERTING'
        h_vote  = 1 if hurst < 0.45 else 0
        ac_vote = 1 if autocorr < -0.05 else 0
        vr_vote = 1 if vr < 0.7 else 0
        conf    = (h_vote + ac_vote + vr_vote) / 3
        reason  = f'H={hurst:.3f}<0.45,autocorr={autocorr:.3f}<-0.05'
    else:
        regime = 'CHOPPY'
        h_vote  = 1 if 0.45 <= hurst  <= 0.55 else 0
        ac_vote = 1 if -0.05 <= autocorr <= 0.05 else 0
        vr_vote = 1 if vr > 1.0 else 0
        conf    = (h_vote + ac_vote + vr_vote) / 3
        reason  = f'mixed:H={hurst:.3f},autocorr={autocorr:.3f},vr={vr:.2f},net={net_dir:.2f}'

    return regime, round(conf, 2), reason


# ─────────────────────────────────────────────────────────────
#  DB — store regime classification
# ─────────────────────────────────────────────────────────────

def _ensure_table():
    """Create regime_log table if it doesn't exist (called once per process)."""
    conn = _db.connect()
    try:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS regime_log (
                id         SERIAL PRIMARY KEY,
                symbol     VARCHAR NOT NULL,
                regime     VARCHAR NOT NULL,
                hurst      FLOAT,
                autocorr   FLOAT,
                vol_ratio  FLOAT,
                confidence FLOAT,
                timestamp  TIMESTAMPTZ NOT NULL,
                reason     TEXT
            )
        ''')
        conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_regime_log_sym_ts '
            'ON regime_log (symbol, timestamp)'
        )
        conn.commit()
    except Exception as e:
        logger.debug(f'regime_log table ensure failed (may already exist): {e}')
    finally:
        conn.close()


_table_ensured = False


def _store(state: RegimeState):
    global _table_ensured
    if not _table_ensured:
        _ensure_table()
        _table_ensured = True

    conn = _db.connect()
    try:
        conn.execute(
            'INSERT INTO regime_log (symbol, regime, hurst, autocorr, vol_ratio, confidence, timestamp, reason) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (state.symbol, state.regime,
             round(state.hurst, 4), round(state.autocorr, 4),
             round(state.vol_ratio, 4), round(state.confidence, 2),
             state.timestamp.isoformat(), state.reason)
        )
        conn.commit()
    except Exception as e:
        logger.warning(f'regime_log insert failed ({state.symbol}): {e}')
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────

def calculate_and_store(symbol: str, df_5m: pd.DataFrame = None) :
    """
    Calculate regime state for symbol and persist to regime_log.
    If df_5m is None, loads last 200 5min bars from DB.
    Returns RegimeState or None if insufficient data.

    # REGIME RESEARCH — observation only, no signal impact
    """
    try:
        if df_5m is None:
            df_5m = _load_5min(symbol, limit=200)

        if df_5m.empty or len(df_5m) < 30:
            logger.debug(f'RegimeEngine: {symbol} — insufficient bars ({len(df_5m)})')
            return None

        closes = df_5m['close'].values.astype(float)
        if np.any(np.isnan(closes)) or np.any(closes <= 0):
            closes = closes[~(np.isnan(closes) | (closes <= 0))]
        if len(closes) < 30:
            return None

        log_ret = np.diff(np.log(closes))
        if len(log_ret) < 20:
            return None

        hurst    = _hurst_rs(log_ret[-100:] if len(log_ret) >= 100 else log_ret)
        autocorr = _autocorr_lag1(log_ret)
        vr       = _vol_ratio(log_ret)

        # Net directional move: |close[-1] - close[-100]| / session high-low range
        # Detects staircase moves that suppress lag-1 autocorr (alternating bar pattern)
        n_nd = min(len(closes), 100)
        nd_slice  = closes[-n_nd:]
        nd_range  = float(nd_slice.max() - nd_slice.min())
        nd_net    = float(abs(closes[-1] - closes[-n_nd]))
        net_dir   = nd_net / nd_range if nd_range > 1e-6 else 0.0

        regime, confidence, reason = classify(hurst, autocorr, vr, net_dir)
        now_utc = datetime.now(timezone.utc)

        state = RegimeState(
            symbol=symbol, regime=regime, hurst=hurst,
            autocorr=autocorr, vol_ratio=vr, confidence=confidence,
            timestamp=now_utc, reason=reason,
        )

        # Always log calculation
        logger.info(
            f'Regime {symbol}: {regime} | '
            f'H={hurst:.2f} | autocorr={autocorr:.2f} | '
            f'vol={vr:.2f} | net_dir={net_dir:.2f} | conf={confidence:.2f}'
        )

        # Log WARNING on regime change
        prev = _last_regime.get(symbol)
        if prev and prev != regime:
            logger.warning(f'Regime {symbol}: CHANGED {prev}→{regime}')
        _last_regime[symbol] = regime

        _store(state)
        return state

    except Exception as e:
        logger.error(f'RegimeEngine calculate_and_store({symbol}): {e}')
        return None


def get_current_regime(symbol: str) -> dict:
    """Return most recent regime state from DB for symbol."""
    global _table_ensured
    if not _table_ensured:
        _ensure_table()
        _table_ensured = True

    try:
        conn = _db.connect()
        row = conn.execute(
            'SELECT symbol, regime, hurst, autocorr, vol_ratio, confidence, timestamp, reason '
            'FROM regime_log WHERE symbol=? ORDER BY timestamp DESC LIMIT 1',
            (symbol,)
        ).fetchone()
        conn.close()
        if row is None:
            return {'symbol': symbol, 'regime': None, 'error': 'no data'}
        return {
            'symbol':     row[0],
            'regime':     row[1],
            'hurst':      row[2],
            'autocorr':   row[3],
            'vol_ratio':  row[4],
            'confidence': row[5],
            'timestamp':  row[6],
            'reason':     row[7],
        }
    except Exception as e:
        return {'symbol': symbol, 'regime': None, 'error': str(e)}


def get_regime_history(symbol: str, limit: int = 100) -> list:
    """Return last N regime classifications for symbol."""
    try:
        conn = _db.connect()
        rows = conn.execute(
            'SELECT symbol, regime, hurst, autocorr, vol_ratio, confidence, timestamp, reason '
            'FROM regime_log WHERE symbol=? ORDER BY timestamp DESC LIMIT ?',
            (symbol, limit)
        ).fetchall()
        conn.close()
        return [
            {'symbol': r[0], 'regime': r[1], 'hurst': r[2], 'autocorr': r[3],
             'vol_ratio': r[4], 'confidence': r[5], 'timestamp': r[6], 'reason': r[7]}
            for r in rows
        ]
    except Exception as e:
        logger.warning(f'get_regime_history({symbol}): {e}')
        return []


# ─────────────────────────────────────────────────────────────
#  CLI — run manually for testing
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    for sym in ('MNQ', 'ES', 'NQ'):
        print(f'\n=== {sym} ===')
        state = calculate_and_store(sym)
        if state:
            print(f'  Regime:     {state.regime}')
            print(f'  Hurst:      {state.hurst:.4f}')
            print(f'  Autocorr:   {state.autocorr:.4f}')
            print(f'  Vol ratio:  {state.vol_ratio:.4f}')
            print(f'  Confidence: {state.confidence:.2f}')
            print(f'  Reason:     {state.reason}')
        else:
            print('  No data / insufficient bars')
