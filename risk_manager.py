"""
APEX Risk Manager — risk_manager.py
=====================================
Five components:
  1. RegimeDetector   — classify NQ 4h market regime every bar
  2. PositionSizer    — size trades with regime + drawdown adjustments
  3. DailyRiskMonitor — track daily P&L limits
  4. DrawdownTracker  — rolling high-watermark drawdown
  5. RiskGate         — combine all checks into a single go/no-go

Run directly to see current state + backtest impact:
  python3 risk_manager.py
"""

import sqlite3
import json
import logging
import time
from datetime import datetime, timezone, date, timedelta
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np

DB_PATH = 'apex_market.db'

logger = logging.getLogger('APEX.Risk')

# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────

POINT_VALUE = {'NQ': 20.0, 'ES': 50.0, 'GC': 100.0, 'CL': 1000.0}

BASE_RISK_PCT    = 1.0    # % of account per trade
CORR_RISK_PCT    = 0.5    # % when correlated instrument already open
MAX_OPEN_TRADES  = 3      # hard cap on concurrent positions

# Daily P&L limits
DAILY_LOSS_LIMIT_R  = -3.0   # stop trading for the day
DAILY_WIN_PROTECT_R =  5.0   # lock in profits, no new trades

# Regime ADX thresholds
ADX_TREND_THRESH  = 25.0
ADX_WEAK_THRESH   = 15.0
ATR_HIGH_VOL_MULT = 1.30   # ATR > 1.3× 20-bar mean = HIGH_VOL
ATR_LOW_VOL_MULT  = 0.70   # ATR < 0.7× 20-bar mean = LOW_VOL

# Drawdown adjustment thresholds
DD_FULL_THRESH    = 5.0    # < 5% DD  → 1.0× risk
DD_REDUCE_THRESH  = 10.0   # 5–10% DD → 0.75×
DD_HALF_THRESH    = 15.0   # 10–15%   → 0.5×
DD_STOP_THRESH    = 15.0   # > 15%    → 0 (no trading)

# Regime risk multipliers
REGIME_MULT = {
    'TRENDING_BULL':    {'long': 1.0, 'short': 0.5, 'any': 1.0},
    'TRENDING_BEAR':    {'long': 0.5, 'short': 1.0, 'any': 1.0},
    'RANGING_HIGH_VOL': {'long': 0.5, 'short': 0.5, 'any': 0.5},
    'RANGING_LOW_VOL':  {'long': 0.75,'short': 0.75,'any': 0.75},
    'NORMAL':           {'long': 1.0, 'short': 1.0, 'any': 1.0},
}


# ─────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────

def _load_bars(symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    df = pd.read_sql_query(
        'SELECT ts,open,high,low,close,volume FROM ohlcv '
        'WHERE symbol=? AND timeframe=? ORDER BY ts DESC LIMIT ?',
        conn, params=(symbol, timeframe, limit)
    )
    conn.close()
    df = df.iloc[::-1].reset_index(drop=True)
    df['dt'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    df.set_index('dt', inplace=True)
    return df[['ts','open','high','low','close','volume']].astype(float)


def _load_bars_from(symbol: str, timeframe: str, start: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    ts0 = int(pd.Timestamp(start, tz='UTC').timestamp())
    df = pd.read_sql_query(
        'SELECT ts,open,high,low,close,volume FROM ohlcv '
        'WHERE symbol=? AND timeframe=? AND ts>=? ORDER BY ts ASC',
        conn, params=(symbol, timeframe, ts0)
    )
    conn.close()
    df['dt'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    df.set_index('dt', inplace=True)
    return df[['ts','open','high','low','close','volume']].astype(float)


# ─────────────────────────────────────────────────────────────
#  INDICATOR HELPERS
# ─────────────────────────────────────────────────────────────

def _wilder(series: pd.Series, period: int) -> pd.Series:
    """Wilder smoothing (equivalent to EMA with alpha=1/period)."""
    return series.ewm(alpha=1/period, adjust=False).mean()


def _calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low']  - df['close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return _wilder(tr, period)


def _calc_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Returns DataFrame with columns: adx, plus_di, minus_di."""
    high  = df['high']
    low   = df['low']
    close = df['close']

    plus_dm  = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    # Where +DM < -DM or < 0, zero it; same for -DM
    plus_dm  = plus_dm.where(plus_dm > minus_dm, 0.0)
    minus_dm = minus_dm.where(minus_dm > plus_dm.where(plus_dm > minus_dm, 0.0), 0.0)
    # Recalculate properly: +DM when up-move > down-move, else 0
    up   = high - high.shift(1)
    down = low.shift(1) - low
    plus_dm2  = pd.Series(np.where((up > down) & (up > 0), up,   0.0), index=df.index)
    minus_dm2 = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr14       = _wilder(tr, period)
    smooth_pdm  = _wilder(plus_dm2,  period)
    smooth_mdm  = _wilder(minus_dm2, period)

    plus_di  = 100 * smooth_pdm  / atr14.replace(0, np.nan)
    minus_di = 100 * smooth_mdm  / atr14.replace(0, np.nan)

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = _wilder(dx, period)

    return pd.DataFrame({'adx': adx, 'plus_di': plus_di, 'minus_di': minus_di}, index=df.index)


# ═════════════════════════════════════════════════════════════
#  1. REGIME DETECTOR
# ═════════════════════════════════════════════════════════════

@dataclass
class Regime:
    label:      str          # TRENDING_BULL / TRENDING_BEAR / RANGING_HIGH_VOL / RANGING_LOW_VOL / NORMAL
    adx:        float
    atr:        float
    atr_ratio:  float        # atr / 20-bar atr mean
    ema20:      float
    ema50:      float
    timestamp:  pd.Timestamp

    def __repr__(self):
        return (f"Regime({self.label} | ADX={self.adx:.1f} "
                f"ATR_ratio={self.atr_ratio:.2f} "
                f"EMA20={'>' if self.ema20>self.ema50 else '<'}EMA50 "
                f"@ {self.timestamp.strftime('%Y-%m-%d %H:%M')} UTC)")


class RegimeDetector:
    """
    Classify NQ 4-hour market regime.
    Cache refreshes on each new 4-hour bar.
    """

    def __init__(self, symbol: str = 'NQ', lookback: int = 100):
        self.symbol    = symbol
        self.lookback  = lookback
        self._cache:   Optional[Regime] = None
        self._cache_ts: float = 0.0
        self._cache_ttl = 3600  # 1 hour

    def _classify(self, adx: float, plus_di: float, minus_di: float,
                  atr_ratio: float, ema20: float, ema50: float) -> str:
        if adx >= ADX_TREND_THRESH:
            return 'TRENDING_BULL' if plus_di > minus_di else 'TRENDING_BEAR'
        if atr_ratio >= ATR_HIGH_VOL_MULT:
            return 'RANGING_HIGH_VOL'
        if adx < ADX_WEAK_THRESH:
            return 'RANGING_LOW_VOL' if atr_ratio < ATR_LOW_VOL_MULT else 'RANGING_LOW_VOL'
        return 'NORMAL'

    def get_regime(self, force: bool = False) -> Regime:
        """Return current regime, using cache if fresh."""
        now = time.time()
        if not force and self._cache and (now - self._cache_ts) < self._cache_ttl:
            return self._cache

        df = _load_bars(self.symbol, '4hour', self.lookback)
        if len(df) < 30:
            return Regime('NORMAL', 0, 0, 1.0, 0, 0, pd.Timestamp.now(tz='UTC'))

        atr_series  = _calc_atr(df, 14)
        adx_df      = _calc_adx(df, 14)
        ema20_s     = df['close'].ewm(span=20, adjust=False).mean()
        ema50_s     = df['close'].ewm(span=50, adjust=False).mean()
        atr_ma20    = atr_series.rolling(20).mean()

        cur_atr     = float(atr_series.iloc[-1])
        cur_atr_ma  = float(atr_ma20.iloc[-1])
        atr_ratio   = cur_atr / cur_atr_ma if cur_atr_ma > 0 else 1.0

        cur_adx     = float(adx_df['adx'].iloc[-1])
        cur_pdi     = float(adx_df['plus_di'].iloc[-1])
        cur_mdi     = float(adx_df['minus_di'].iloc[-1])
        cur_ema20   = float(ema20_s.iloc[-1])
        cur_ema50   = float(ema50_s.iloc[-1])

        label = self._classify(cur_adx, cur_pdi, cur_mdi, atr_ratio, cur_ema20, cur_ema50)

        regime = Regime(
            label      = label,
            adx        = cur_adx,
            atr        = cur_atr,
            atr_ratio  = atr_ratio,
            ema20      = cur_ema20,
            ema50      = cur_ema50,
            timestamp  = df.index[-1],
        )
        self._cache    = regime
        self._cache_ts = now
        return regime

    def get_regime_multiplier(self, direction: str, regime: Regime = None) -> float:
        r = regime or self.get_regime()
        mult_map = REGIME_MULT.get(r.label, REGIME_MULT['NORMAL'])
        return mult_map.get(direction.lower(), mult_map['any'])

    def get_regime_series(self, start: str = '2024-09-01') -> pd.DataFrame:
        """
        Compute regime for every 4-hour bar from start — for backtesting.
        Returns DataFrame indexed by timestamp with columns: label, adx, atr_ratio, ema20, ema50.
        """
        df = _load_bars_from(self.symbol, '4hour', start)
        if len(df) < 30:
            return pd.DataFrame()

        atr_series = _calc_atr(df, 14)
        adx_df     = _calc_adx(df, 14)
        ema20_s    = df['close'].ewm(span=20, adjust=False).mean()
        ema50_s    = df['close'].ewm(span=50, adjust=False).mean()
        atr_ma20   = atr_series.rolling(20).mean()
        atr_ratio  = atr_series / atr_ma20.replace(0, np.nan)

        out = pd.DataFrame({
            'adx':      adx_df['adx'],
            'plus_di':  adx_df['plus_di'],
            'minus_di': adx_df['minus_di'],
            'atr':      atr_series,
            'atr_ratio':atr_ratio,
            'ema20':    ema20_s,
            'ema50':    ema50_s,
        }, index=df.index)

        def classify_row(row):
            if pd.isna(row['adx']): return 'NORMAL'
            return self._classify(row['adx'], row['plus_di'], row['minus_di'],
                                  row['atr_ratio'], row['ema20'], row['ema50'])

        out['label'] = out.apply(classify_row, axis=1)
        return out


# ═════════════════════════════════════════════════════════════
#  2. POSITION SIZER
# ═════════════════════════════════════════════════════════════

@dataclass
class SizeResult:
    contracts:         int
    dollar_risk:       float
    adjusted_risk_pct: float
    regime_mult:       float
    dd_mult:           float
    corr_mult:         float
    blocked:           bool
    block_reason:      str
    regime_label:      str

    def __repr__(self):
        if self.blocked:
            return f"BLOCKED — {self.block_reason}"
        return (f"{self.contracts} contract(s) | ${self.dollar_risk:,.0f} risk | "
                f"{self.adjusted_risk_pct:.2f}% | "
                f"regime={self.regime_mult:.2f}× dd={self.dd_mult:.2f}× corr={self.corr_mult:.2f}×")


class PositionSizer:
    """
    Size each trade accounting for regime, drawdown, and correlation.
    open_trades: list of dicts with keys: symbol, direction, entry, stop
    """

    def __init__(self, regime_detector: RegimeDetector,
                 drawdown_tracker: 'DrawdownTracker'):
        self.regime   = regime_detector
        self.dd       = drawdown_tracker

    def size(self, symbol: str, direction: str,
             account_balance: float, stop_distance_pts: float,
             open_trades: list = None) -> SizeResult:

        open_trades = open_trades or []

        # ── Hard cap: max concurrent trades ──────────────────
        if len(open_trades) >= MAX_OPEN_TRADES:
            return SizeResult(0, 0, 0, 1, 1, 1, True,
                              f'Max {MAX_OPEN_TRADES} concurrent trades reached',
                              'N/A')

        # ── Correlation cap ───────────────────────────────────
        same_sym = [t for t in open_trades if t.get('symbol') == symbol]
        corr_mult = 0.5 if same_sym else 1.0

        # ── Drawdown multiplier ───────────────────────────────
        dd_pct  = self.dd.get_drawdown_pct(account_balance)
        dd_mult = self.dd.get_risk_multiplier(dd_pct)

        if dd_mult == 0:
            return SizeResult(0, 0, 0, 1, 0, corr_mult, True,
                              f'Drawdown {dd_pct:.1f}% exceeds {DD_STOP_THRESH}% limit',
                              'N/A')

        # ── Regime multiplier ─────────────────────────────────
        regime      = self.regime.get_regime()
        regime_mult = self.regime.get_regime_multiplier(direction, regime)

        # ── Final risk % ──────────────────────────────────────
        final_risk_pct = BASE_RISK_PCT * corr_mult * dd_mult * regime_mult
        dollar_risk    = account_balance * (final_risk_pct / 100)

        # ── Contracts ─────────────────────────────────────────
        pv       = POINT_VALUE.get(symbol, 20.0)
        if stop_distance_pts <= 0:
            return SizeResult(0, 0, 0, regime_mult, dd_mult, corr_mult, True,
                              'Stop distance must be > 0', regime.label)

        contracts = max(int(dollar_risk / (stop_distance_pts * pv)), 0)

        return SizeResult(
            contracts         = contracts,
            dollar_risk       = round(dollar_risk, 2),
            adjusted_risk_pct = round(final_risk_pct, 3),
            regime_mult       = regime_mult,
            dd_mult           = dd_mult,
            corr_mult         = corr_mult,
            blocked           = contracts == 0,
            block_reason      = 'Contracts rounded to 0 — account too small for this stop' if contracts == 0 else '',
            regime_label      = regime.label,
        )


# ═════════════════════════════════════════════════════════════
#  3. DAILY RISK MONITOR
# ═════════════════════════════════════════════════════════════

class DailyRiskMonitor:
    """
    Monitor daily P&L against hard limits.
    Reads from apex_trades table; falls back to 0 if table absent.
    """

    def daily_pnl_r(self) -> float:
        """Sum of closed trade R today (UTC date)."""
        today = date.today().isoformat()
        try:
            conn = sqlite3.connect(DB_PATH, timeout=30)
            rows = conn.execute(
                """SELECT pnl_r FROM apex_trades
                   WHERE DATE(entry_time) = ? AND status = 'closed'""",
                (today,)
            ).fetchall()
            conn.close()
            return sum(r[0] for r in rows if r[0] is not None)
        except Exception:
            return 0.0

    def is_daily_limit_hit(self) -> bool:
        """True if daily losses exceed DAILY_LOSS_LIMIT_R."""
        return self.daily_pnl_r() <= DAILY_LOSS_LIMIT_R

    def is_daily_win_protection(self) -> bool:
        """True if daily profit exceeds DAILY_WIN_PROTECT_R — lock in gains."""
        return self.daily_pnl_r() >= DAILY_WIN_PROTECT_R

    def get_status(self) -> dict:
        pnl = self.daily_pnl_r()
        return {
            'daily_pnl_r':      round(pnl, 3),
            'limit_hit':        pnl <= DAILY_LOSS_LIMIT_R,
            'win_protection':   pnl >= DAILY_WIN_PROTECT_R,
            'remaining_loss_r': round(pnl - DAILY_LOSS_LIMIT_R, 2),
            'remaining_gain_r': round(DAILY_WIN_PROTECT_R - pnl, 2),
        }


# ═════════════════════════════════════════════════════════════
#  4. DRAWDOWN TRACKER
# ═════════════════════════════════════════════════════════════

class DrawdownTracker:
    """
    Track account drawdown from high watermark.
    Primary source: apex_trades DB (live system).
    Falls back to a flat equity curve if no trades exist.
    """

    def __init__(self, start_balance: float = 10_000.0, risk_pct: float = 1.0):
        self.start_balance = start_balance
        self.risk_pct      = risk_pct
        self._hwm: float   = start_balance

    def _load_closed_trades(self) -> list:
        try:
            conn = sqlite3.connect(DB_PATH, timeout=30)
            rows = conn.execute(
                """SELECT entry_time, pnl_r FROM apex_trades
                   WHERE status = 'closed' AND pnl_r IS NOT NULL
                   ORDER BY entry_time ASC"""
            ).fetchall()
            conn.close()
            return rows
        except Exception:
            return []

    def get_equity_curve(self) -> tuple:
        """Returns (timestamps, balances, high_watermarks)."""
        trades = self._load_closed_trades()
        bal    = self.start_balance
        peak   = self.start_balance
        bals   = [bal]; peaks = [peak]; ts = ['start']
        for entry_time, pnl_r in trades:
            bal  += bal * (self.risk_pct / 100) * pnl_r
            if bal > peak: peak = bal
            bals.append(bal); peaks.append(peak); ts.append(entry_time)
        self._hwm = peak
        return ts, bals, peaks

    def get_drawdown_pct(self, current_balance: float = None) -> float:
        """Current drawdown % from high watermark."""
        _, bals, peaks = self.get_equity_curve()
        hwm = peaks[-1]
        if current_balance is not None:
            bal = current_balance
        else:
            bal = bals[-1]
        if hwm <= 0:
            return 0.0
        return max((hwm - bal) / hwm * 100, 0.0)

    def get_risk_multiplier(self, dd_pct: float = None) -> float:
        """Risk multiplier based on current drawdown."""
        if dd_pct is None:
            dd_pct = self.get_drawdown_pct()
        if dd_pct >= DD_STOP_THRESH:
            return 0.0
        if dd_pct >= DD_REDUCE_THRESH:
            return 0.5
        if dd_pct >= DD_FULL_THRESH:
            return 0.75
        return 1.0

    def get_status(self, current_balance: float = None) -> dict:
        _, bals, peaks = self.get_equity_curve()
        hwm = peaks[-1]
        bal = current_balance or bals[-1]
        dd  = self.get_drawdown_pct(bal)
        return {
            'current_balance': round(bal, 2),
            'high_watermark':  round(hwm, 2),
            'drawdown_pct':    round(dd, 2),
            'risk_multiplier': self.get_risk_multiplier(dd),
            'status': ('STOP_TRADING' if dd >= DD_STOP_THRESH
                       else 'HALF_SIZE'  if dd >= DD_REDUCE_THRESH
                       else 'THREE_QUARTER' if dd >= DD_FULL_THRESH
                       else 'FULL_SIZE'),
        }


# ═════════════════════════════════════════════════════════════
#  5. RISK GATE  (combines all checks into one call)
# ═════════════════════════════════════════════════════════════

@dataclass
class RiskDecision:
    approved:         bool
    contracts:        int
    dollar_risk:      float
    adjusted_risk_pct:float
    block_reasons:    list
    regime_label:     str
    regime_mult:      float
    dd_pct:           float
    dd_mult:          float
    daily_pnl_r:      float
    notes:            list = field(default_factory=list)

    def __repr__(self):
        if not self.approved:
            return f"❌ BLOCKED: {', '.join(self.block_reasons)}"
        return (f"✅ APPROVED: {self.contracts}ct | "
                f"${self.dollar_risk:,.0f} risk ({self.adjusted_risk_pct:.2f}%) | "
                f"regime={self.regime_label}({self.regime_mult:.2f}×) | "
                f"dd={self.dd_pct:.1f}%({self.dd_mult:.2f}×)")


class RiskGate:
    """Single entry point — returns a RiskDecision for any proposed trade."""

    def __init__(self):
        self.regime  = RegimeDetector()
        self.dd      = DrawdownTracker()
        self.daily   = DailyRiskMonitor()
        self.sizer   = PositionSizer(self.regime, self.dd)

    def evaluate(self, symbol: str, direction: str,
                 account_balance: float, stop_distance_pts: float,
                 open_trades: list = None) -> RiskDecision:
        blocks = []; notes = []

        # Daily limits
        daily_status = self.daily.get_status()
        if daily_status['limit_hit']:
            blocks.append(f"Daily loss limit hit ({daily_status['daily_pnl_r']:.1f}R ≤ {DAILY_LOSS_LIMIT_R}R)")
        if daily_status['win_protection']:
            blocks.append(f"Daily win protection active ({daily_status['daily_pnl_r']:.1f}R ≥ {DAILY_WIN_PROTECT_R}R)")

        # Position size
        size = self.sizer.size(symbol, direction, account_balance,
                               stop_distance_pts, open_trades or [])
        if size.blocked:
            blocks.append(size.block_reason)

        # Regime note (non-blocking)
        regime = self.regime.get_regime()
        if regime.label in ('RANGING_HIGH_VOL',):
            notes.append(f"High volatility regime — size reduced 50%")
        elif regime.label in ('RANGING_LOW_VOL',):
            notes.append(f"Low volatility / ranging — size reduced 25%")
        elif regime.label == 'TRENDING_BEAR' and direction == 'long':
            notes.append(f"Counter-trend trade in BEAR regime — size halved")
        elif regime.label == 'TRENDING_BULL' and direction == 'short':
            notes.append(f"Counter-trend trade in BULL regime — size halved")

        dd_status = self.dd.get_status(account_balance)

        return RiskDecision(
            approved          = len(blocks) == 0 and not size.blocked,
            contracts         = size.contracts,
            dollar_risk       = size.dollar_risk,
            adjusted_risk_pct = size.adjusted_risk_pct,
            block_reasons     = blocks,
            regime_label      = regime.label,
            regime_mult       = size.regime_mult,
            dd_pct            = dd_status['drawdown_pct'],
            dd_mult           = size.dd_mult,
            daily_pnl_r       = daily_status['daily_pnl_r'],
            notes             = notes,
        )


# ═════════════════════════════════════════════════════════════
#  BACKTEST IMPACT SIMULATION
# ═════════════════════════════════════════════════════════════

def run_backtest_impact():
    """
    Retroactively apply regime + drawdown risk adjustments to all
    historical backtest trades and show before/after stats.
    """
    import os

    print(f"\n{'='*72}")
    print(f"  REGIME + DRAWDOWN BACKTEST IMPACT SIMULATION")
    print(f"{'='*72}")

    # ── Load all trades ───────────────────────────────────────
    SWING_CONFIGS = {
        'B_NQ': {'file':'backtest_B_NQ_swing.json','days':[0,2,3,4],'session':'both'},
        'B_ES': {'file':'backtest_B_ES_swing.json','days':[0,1,3],  'session':'both'},
        'B_GC': {'file':'backtest_B_GC_swing.json','days':[0,1,2,3],'session':'both'},
        'A_NQ': {'file':'backtest_A_NQ_swing.json','days':None,      'session':'primary'},
        'A_ES': {'file':'backtest_A_ES_swing.json','days':None,      'session':'primary'},
        'C_NQ': {'file':'backtest_C_NQ_swing.json','days':None,      'session':'primary'},
        'C_ES': {'file':'backtest_C_ES_swing.json','days':[0,1,2,4],'session':'primary'},
        'C_GC': {'file':'backtest_C_GC_swing.json','days':[0,1,2,3],'session':'primary'},
    }

    all_trades = []
    for key, cfg in SWING_CONFIGS.items():
        if not os.path.exists(cfg['file']): continue
        with open(cfg['file']) as f: data = json.load(f)
        for t in data.get('trade_log', []):
            ts = pd.Timestamp(t['entry_time'])
            if cfg['days'] is not None and ts.dayofweek not in cfg['days']: continue
            if cfg['session'] == 'primary' and t.get('quality','') != 'primary': continue
            all_trades.append({'entry_time': ts, 'pnl_r': float(t['pnl_r']),
                               'strategy': key, 'symbol': key.split('_')[1],
                               'direction': t.get('direction','long')})

    if os.path.exists('backtest_FVG_NQ.json'):
        with open('backtest_FVG_NQ.json') as f: fvg_data = json.load(f)
        for t in fvg_data.get('trade_log', []):
            if t.get('score', 100) >= 60:
                all_trades.append({'entry_time': pd.Timestamp(t['entry_time']),
                                   'pnl_r': float(t['pnl_r']), 'strategy': 'FVG_NQ',
                                   'symbol': 'NQ', 'direction': t.get('direction','long')})

    # Setup E inline
    conn = sqlite3.connect(DB_PATH, timeout=30)
    ts0 = int(pd.Timestamp('2024-09-01', tz='UTC').timestamp())
    df5 = pd.read_sql_query('SELECT ts,open,high,low,close FROM ohlcv WHERE symbol=? AND timeframe=? AND ts>=? ORDER BY ts ASC',
                             conn, params=['NQ','5min',ts0])
    df4h = pd.read_sql_query('SELECT ts,open,high,low,close FROM ohlcv WHERE symbol=? AND timeframe=? AND ts>=? ORDER BY ts ASC',
                              conn, params=['NQ','4hour',ts0])
    conn.close()
    for df in [df5, df4h]:
        df['dt'] = pd.to_datetime(df['ts'], unit='s', utc=True); df.set_index('dt', inplace=True)
    df5['ema50'] = df5['close'].ewm(span=50,adjust=False).mean()
    tr5 = pd.concat([df5['high']-df5['low'],(df5['high']-df5['close'].shift(1)).abs(),(df5['low']-df5['close'].shift(1)).abs()],axis=1).max(axis=1)
    df5['atr'] = tr5.ewm(alpha=1/14,adjust=False).mean()
    df4h['ema20'] = df4h['close'].ewm(span=20,adjust=False).mean()
    df4h['bias'] = np.where(df4h['close']>df4h['ema20']*1.001,'bullish',np.where(df4h['close']<df4h['ema20']*0.999,'bearish','neutral'))
    bias_s = df4h['bias'].reindex(df5.index, method='ffill')
    mask = (df5.index.hour>=13)&(df5.index.hour<19)&(df5.index.weekday<5)
    sess = df5[mask].copy(); sess['bias']=bias_s.reindex(sess.index, method='ffill')
    last_e=-999
    for i in range(1,len(sess)):
        if i-last_e<12: continue
        b=sess.iloc[i]; pb=sess.iloc[i-1]
        bias=b['bias']
        if bias=='neutral': continue
        atr=b['atr']
        if pd.isna(atr) or atr==0: continue
        price=float(b['close'])
        if abs(price-float(b['ema50']))>0.30*atr: continue
        bull_bar=float(b['close'])>float(b['open']); bear_bar=float(b['close'])<float(b['open'])
        if bias=='bullish' and not bull_bar: continue
        if bias=='bearish' and not bear_bar: continue
        pb_bear=float(pb['close'])<float(pb['open']); pb_bull=float(pb['close'])>float(pb['open'])
        if bias=='bullish' and not pb_bear: continue
        if bias=='bearish' and not pb_bull: continue
        d='long' if bias=='bullish' else 'short'
        stop=price-1.5*atr if d=='long' else price+1.5*atr
        target=price+3.0*atr if d=='long' else price-3.0*atr; risk=abs(price-stop)
        ti=sess.index[i]; se=ti.replace(hour=19,minute=0,second=0)
        fut=sess[(sess.index>ti)&(sess.index<=se)]
        ep=None
        for _,fb in fut.iterrows():
            hi=float(fb['high']); lo=float(fb['low'])
            if d=='long':
                if lo<=stop: ep=stop;break
                if hi>=target: ep=target;break
            else:
                if hi>=stop: ep=stop;break
                if lo<=target: ep=target;break
        if ep is None: ep=float(fut['close'].iloc[-1]) if len(fut)>0 else price
        pnl=round((ep-price)/risk,3) if d=='long' else round((price-ep)/risk,3)
        all_trades.append({'entry_time':ti,'pnl_r':pnl,'strategy':'E_NQ','symbol':'NQ','direction':d})
        last_e=i

    all_trades.sort(key=lambda x: x['entry_time'])
    print(f"\n  Loaded {len(all_trades)} trades from all strategies")

    # ── Build regime time series ──────────────────────────────
    print("  Computing regime for every 4-hour bar…")
    rd = RegimeDetector()
    regime_series = rd.get_regime_series('2024-06-01')   # extra lookback for ADX warmup
    print(f"  Regime series: {len(regime_series)} 4h bars, "
          f"labels: {regime_series['label'].value_counts().to_dict()}")

    # ── Assign regime to each trade ───────────────────────────
    def get_regime_at(ts):
        past = regime_series[regime_series.index <= ts]
        if past.empty: return 'NORMAL', 1.0
        row = past.iloc[-1]
        label = row['label']
        return label, row

    # ── Simulate with adjustments ─────────────────────────────
    START_BAL = 10_000.0
    RISK = 1.0

    # Stats accumulators
    raw_r       = []   # original pnl_r
    adj_r       = []   # adjusted pnl_r (after regime+dd multiplier)
    regime_info = []   # regime label at each trade
    dd_info     = []   # dd_pct at each trade
    mult_info   = []   # final multiplier applied
    filtered    = []   # trades blocked (multiplier=0)

    bal_raw  = START_BAL; peak_raw  = START_BAL
    bal_adj  = START_BAL; peak_adj  = START_BAL

    # Track drawdown for adjusted simulation
    curves_raw = [START_BAL]; curves_adj = [START_BAL]

    n_regime_reduced = 0
    n_dd_reduced     = 0
    n_blocked        = 0

    label_stats = {}   # {regime_label: {wins, losses, total_r_raw, total_r_adj}}

    for t in all_trades:
        pnl     = t['pnl_r']
        ts      = t['entry_time']
        direction = t.get('direction', 'long')
        raw_r.append(pnl)

        # Regime at trade time
        r_label, r_row = get_regime_at(ts)
        regime_info.append(r_label)

        # Regime multiplier
        mult_map  = REGIME_MULT.get(r_label, REGIME_MULT['NORMAL'])
        r_mult    = mult_map.get(direction.lower(), mult_map['any'])

        # Drawdown of adjusted account
        dd_pct = max((peak_adj - bal_adj)/peak_adj*100, 0) if peak_adj > 0 else 0
        dd_info.append(dd_pct)
        d_mult = (0.0 if dd_pct >= DD_STOP_THRESH else
                  0.5 if dd_pct >= DD_REDUCE_THRESH else
                  0.75 if dd_pct >= DD_FULL_THRESH else 1.0)

        final_mult = r_mult * d_mult

        if final_mult == 0:
            n_blocked += 1
            adj_r.append(0.0)
            filtered.append({'trade': t, 'reason': f'DD={dd_pct:.1f}% blocked' if d_mult==0 else f'regime={r_label} blocked'})
            mult_info.append(0)
        else:
            if r_mult < 1.0: n_regime_reduced += 1
            if d_mult < 1.0: n_dd_reduced += 1
            adj_r.append(pnl * final_mult)
            mult_info.append(final_mult)

        # Update raw equity
        bal_raw  += bal_raw  * (RISK/100) * pnl
        if bal_raw > peak_raw:  peak_raw  = bal_raw
        curves_raw.append(bal_raw)

        # Update adjusted equity
        bal_adj += bal_adj * (RISK/100) * (pnl * final_mult)
        if bal_adj > peak_adj: peak_adj = bal_adj
        curves_adj.append(bal_adj)

        # Label stats
        if r_label not in label_stats:
            label_stats[r_label] = {'n':0,'wins':0,'total_raw':0,'total_adj':0}
        ls = label_stats[r_label]
        ls['n'] += 1
        if pnl > 0: ls['wins'] += 1
        ls['total_raw'] += pnl
        ls['total_adj'] += pnl * final_mult

    raw_r  = np.array(raw_r)
    adj_r  = np.array(adj_r)
    curves_raw = np.array(curves_raw)
    curves_adj = np.array(curves_adj)

    def equity_stats(r_arr, curve):
        wins = (r_arr > 0).sum()
        cum  = np.cumsum(r_arr)
        dd   = float(np.min(cum - np.maximum.accumulate(cum)))
        sh   = float(np.mean(r_arr)/np.std(r_arr)*np.sqrt(252)) if np.std(r_arr)>0 else 0
        # Max DD % from equity curve
        peak_c = np.maximum.accumulate(curve)
        dd_pct = float(np.max((peak_c - curve)/peak_c*100))
        return {
            'n':       len(r_arr),
            'wr':      round(wins/len(r_arr)*100, 1),
            'ev':      round(float(np.mean(r_arr)), 3),
            'total_r': round(float(np.sum(r_arr)), 2),
            'max_dd_r':round(dd, 2),
            'max_dd_pct': round(dd_pct, 1),
            'sharpe':  round(sh, 2),
            'final_bal': round(curve[-1], 2),
        }

    before = equity_stats(raw_r, curves_raw)
    after  = equity_stats(adj_r[adj_r != 0], curves_adj)  # non-zero for stats

    # ── PRINT RESULTS ─────────────────────────────────────────

    print(f"\n{'─'*72}")
    print(f"  BEFORE vs AFTER REGIME + DRAWDOWN FILTERING")
    print(f"{'─'*72}")
    print(f"  {'Metric':<28} {'Before':>14} {'After':>14} {'Delta':>12}")
    print(f"  {'─'*70}")
    metrics = [
        ('Total trades',        before['n'],          after['n']+n_blocked,  ''),
        ('Trades executed',     before['n'],          len(adj_r[adj_r!=0]),  ''),
        ('Trades blocked/0×',   0,                    n_blocked,             ''),
        ('Trades reduced',      0,                    n_regime_reduced+n_dd_reduced-n_blocked, ''),
        ('Win rate',            f"{before['wr']}%",   f"{after['wr']}%",     ''),
        ('Expectancy (R)',      before['ev'],         after['ev'],           ''),
        ('Total R',             before['total_r'],    after['total_r'],      ''),
        ('Max DD (R)',          before['max_dd_r'],   after['max_dd_r'],     ''),
        ('Max DD (%)',          f"{before['max_dd_pct']}%", f"{after['max_dd_pct']}%", ''),
        ('Sharpe',              before['sharpe'],     after['sharpe'],       ''),
        ('Final balance',       f"${before['final_bal']:,.0f}", f"${after['final_bal']:,.0f}", ''),
    ]
    for label, b, a, _ in metrics:
        if isinstance(b, (int, float)) and isinstance(a, (int, float)):
            delta = a - b
            delta_s = f"{delta:+.2f}" if isinstance(delta, float) else f"{delta:+d}"
        else:
            delta_s = ''
        print(f"  {label:<28} {str(b):>14} {str(a):>14} {delta_s:>12}")

    print(f"\n{'─'*72}")
    print(f"  REGIME BREAKDOWN")
    print(f"{'─'*72}")
    print(f"  {'Regime':<22} {'Trades':>7} {'WR':>6} {'Total R (raw)':>14} {'Total R (adj)':>14} {'Δ R':>8}")
    print(f"  {'─'*70}")
    for label, ls in sorted(label_stats.items(), key=lambda x: -x[1]['n']):
        wr = round(ls['wins']/ls['n']*100,1) if ls['n']>0 else 0
        delta = ls['total_adj'] - ls['total_raw']
        mult_avg = REGIME_MULT.get(label, REGIME_MULT['NORMAL'])['any']
        print(f"  {label:<22} {ls['n']:>7} {wr:>5.1f}%  {ls['total_raw']:>+13.2f}R  {ls['total_adj']:>+13.2f}R  {delta:>+7.2f}R")

    print(f"\n{'─'*72}")
    print(f"  DRAWDOWN REDUCTION IMPACT")
    print(f"{'─'*72}")
    n_full = sum(1 for m in mult_info if m == 1.0)
    n_075  = sum(1 for m in mult_info if 0.74 < m <= 0.76)
    n_050  = sum(1 for m in mult_info if 0.49 < m <= 0.51)
    n_zero = sum(1 for m in mult_info if m == 0)
    n_mixed= len(mult_info) - n_full - n_075 - n_050 - n_zero
    print(f"  Full size (1.0×):       {n_full:>5} trades")
    print(f"  Reduced 75% (0.75×):    {n_075:>5} trades")
    print(f"  Halved (0.5×):          {n_050:>5} trades")
    print(f"  Mixed (regime×dd):      {n_mixed:>5} trades")
    print(f"  Blocked (0×):           {n_zero:>5} trades")
    print(f"\n  Trades reduced by regime: {n_regime_reduced}")
    print(f"  Trades reduced by DD:     {n_dd_reduced}")

    # ── Current regime snapshot ───────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  CURRENT REGIME SNAPSHOT")
    print(f"{'─'*72}")
    rd2 = RegimeDetector()
    current = rd2.get_regime()
    print(f"  {current}")
    for d in ('long', 'short'):
        mult = rd2.get_regime_multiplier(d, current)
        print(f"  Risk multiplier ({d:>5}): {mult:.2f}×")

    return {
        'before': before, 'after': after,
        'n_blocked': n_blocked,
        'n_regime_reduced': n_regime_reduced,
        'n_dd_reduced': n_dd_reduced,
        'regime_breakdown': {k: {'n':v['n'],'wr':round(v['wins']/v['n']*100,1) if v['n']>0 else 0,
                                 'total_r_raw':round(v['total_raw'],2),
                                 'total_r_adj':round(v['total_adj'],2)}
                             for k,v in label_stats.items()},
        'current_regime': current.label,
    }


# ═════════════════════════════════════════════════════════════
#  CLI — run directly for live status + backtest impact
# ═════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import os; os.chdir(os.path.dirname(os.path.abspath(__file__)) or '.')

    print(f"\n{'='*72}")
    print(f"  APEX Risk Manager — Live Status")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"{'='*72}")

    # Live components
    rd   = RegimeDetector()
    dd   = DrawdownTracker()
    drm  = DailyRiskMonitor()
    gate = RiskGate()

    print(f"\n── Regime ──────────────────────────────────────────────────────────")
    regime = rd.get_regime()
    print(f"  {regime}")

    print(f"\n── Daily Risk ──────────────────────────────────────────────────────")
    ds = drm.get_status()
    print(f"  Daily P&L: {ds['daily_pnl_r']:+.2f}R  |  "
          f"Limit hit: {ds['limit_hit']}  |  Win protection: {ds['win_protection']}")

    print(f"\n── Drawdown ────────────────────────────────────────────────────────")
    dds = dd.get_status()
    print(f"  Balance: ${dds['current_balance']:,.2f}  HWM: ${dds['high_watermark']:,.2f}  "
          f"DD: {dds['drawdown_pct']:.1f}%  Status: {dds['status']}  "
          f"Multiplier: {dds['risk_multiplier']:.2f}×")

    print(f"\n── Sample Trade Evaluation (NQ LONG, $50k account, 20pt stop) ──────")
    decision = gate.evaluate('NQ', 'long', 50_000, 20.0, open_trades=[])
    print(f"  {decision}")
    if decision.notes:
        for n in decision.notes: print(f"  ⚠ {n}")

    # Backtest impact
    impact = run_backtest_impact()

    # Save to JSON
    with open('risk_manager_backtest.json','w') as f:
        json.dump({
            'meta': {'generated': datetime.now(timezone.utc).isoformat(),
                     'current_regime': impact['current_regime']},
            'backtest_impact': impact
        }, f, indent=2, default=str)
    print(f"\n✓ Backtest results saved to risk_manager_backtest.json")
    print()
