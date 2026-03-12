"""
APEX Strategy Configuration — strategy_config.py
==================================================
Locked-in optimised settings from v2 backtest with liquidity sweep upgrade.

Last optimised: 2026-03-12
Data: 10,000 bars (5min) per instrument, walk-forward backtest

NQ SCALP:  Sharpe 15.39 | Return 173%   | WR 53.4% | DD 14.5% | Expectancy 1.31R
NQ SWING:  Sharpe 30.28 | Return 191%   | WR 50.0% | DD 30.7% | Expectancy 2.52R
ES SWING:  Sharpe  6.18 | Return 1241%  | WR 33.5% | DD 38.3% | Expectancy 2.93R
GC SCALP:  Sharpe  9.68 | Return 101%   | WR 52.8% | DD 13.2% | Expectancy 0.64R
GC SWING:  Sharpe  9.03 | Return 104%   | WR 37.3% | DD 27.6% | Expectancy 0.69R

Instrument routing:
  NQ → Scalp (NY Open, Tue-Thu) + Swing (London, Tue-Thu)
  ES → Swing only (London, all week) — scalp DD 66% excluded
  GC → Scalp (NY Open, all week) + Swing (NY Open, all week)
"""

# =============================================================
#  GLOBAL SETTINGS
# =============================================================

STRATEGY = {
    'min_score':        70,
    'rr_ratio':         3.0,
    'risk_pct':         1.5,
    'allowed_days':     [1, 2, 3],
    'blocked_days':     [0, 4],
    'tradeable_sessions': [
        {'name': 'London Prime', 'start': (5, 0),  'end': (8, 0),  'quality': 90},
        {'name': 'NY Open',      'start': (9, 30), 'end': (10, 30),'quality': 95},
    ],
    'max_vix':          25.0,   # temporarily raised from 20, revert when markets calm
    'max_total_risk':   5.0,
    'max_trades_day':   3,
    'max_trades_session': 1,
    'allow_intraday':   True,
    'allow_overnight':  True,
    'partial_exit_r':   2.0,
    'breakeven_after':  1.5,
    'auto_paper_trade': True,
    'paper_balance':    10000,
    'send_alerts':      True,
    'alert_min_score':  70,
}

# =============================================================
#  NQ — NASDAQ 100 FUTURES
#  Sharpe 15.39 scalp | Sharpe 30.28 swing
# =============================================================

NQ_SCALP_CONFIG = {
    'min_score':     70,
    'rr_ratio':      2.5,
    'risk_pct':      1.5,
    'session':       'ny_open',
    'vix_max':       20,
    'dow':           [1, 2, 3],   # Tue/Wed/Thu only
    'stop_atr':      0.8,
    'max_hold_bars': 30,
}

NQ_SWING_CONFIG = {
    'min_score':     70,
    'rr_ratio':      4.0,
    'risk_pct':      2.0,
    'session':       'london',
    'vix_max':       20,
    'dow':           [1, 2, 3],   # Tue/Wed/Thu only
    'stop_atr':      1.5,
    'htf_strict':    False,
    'max_hold_bars': 100,
}

# =============================================================
#  ES — S&P 500 E-MINI FUTURES
#  Swing only — scalp DD 66% excluded
#  Sharpe 6.18 swing
# =============================================================

ES_SWING_CONFIG = {
    'min_score':     70,
    'rr_ratio':      4.0,
    'risk_pct':      2.0,
    'session':       'london',
    'vix_max':       20,
    'dow':           [0, 1, 2, 3, 4],  # all week
    'stop_atr':      2.0,
    'htf_strict':    True,
    'max_hold_bars': 100,
}

# =============================================================
#  GC — GOLD FUTURES
#  Both modes viable — trades all week, NY Open session
#  Sharpe 9.68 scalp | Sharpe 9.03 swing
# =============================================================

GC_SCALP_CONFIG = {
    'min_score':     70,
    'rr_ratio':      2.0,
    'risk_pct':      1.5,
    'session':       'ny_open',
    'vix_max':       20,
    'dow':           [0, 1, 2, 3, 4],  # all week
    'stop_atr':      1.5,
    'max_hold_bars': 30,
}

GC_SWING_CONFIG = {
    'min_score':     65,
    'rr_ratio':      4.0,
    'risk_pct':      2.0,
    'session':       'ny_open',
    'vix_max':       20,
    'dow':           [0, 1, 2, 3, 4],  # all week
    'stop_atr':      2.0,
    'htf_strict':    True,
    'max_hold_bars': 100,
}

# =============================================================
#  LEGACY ALIASES (used by strategy_router internals)
# =============================================================

SCALP_CONFIG   = NQ_SCALP_CONFIG
SWING_CONFIG   = NQ_SWING_CONFIG

MEANREV_CONFIG = {
    'min_score':       75,
    'rr_ratio':        2.0,
    'risk_pct':        0.5,
    'vix_max':         18,
    'dow':             [1, 2, 3],
    'vwap_dev_thresh': 2.5,
    'market_cond':     'ranging_only',
    'stop_atr':        0.6,
    'enabled':         False,
}

# =============================================================
#  INSTRUMENT ROUTING
# =============================================================

INSTRUMENT_MODES = {
    'NQ': ['scalp', 'swing'],
    'ES': ['swing'],          # scalp excluded — DD 66%
    'GC': ['scalp', 'swing'],
}

INSTRUMENT_CONFIGS = {
    'NQ': {'scalp': NQ_SCALP_CONFIG, 'swing': NQ_SWING_CONFIG},
    'ES': {'swing': ES_SWING_CONFIG},
    'GC': {'scalp': GC_SCALP_CONFIG, 'swing': GC_SWING_CONFIG},
}

# =============================================================
#  HELPER FUNCTIONS
# =============================================================

def is_tradeable_session(dt_ny=None):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    if dt_ny is None:
        from datetime import timezone
        dt_ny = datetime.now(timezone.utc).astimezone(ZoneInfo('America/New_York'))
    h   = dt_ny.hour
    m   = dt_ny.minute
    dow = dt_ny.weekday()
    if dow >= 5:
        return False, 'Weekend', 0
    if dow in STRATEGY['blocked_days']:
        day_names = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday'}
        return False, day_names[dow], 0
    for sess in STRATEGY['tradeable_sessions']:
        sh, sm = sess['start']
        eh, em = sess['end']
        if (sh*60+sm) <= (h*60+m) < (eh*60+em):
            return True, sess['name'], sess['quality']
    return False, 'Off Hours', 0


def get_mode_config(mode: str, symbol: str = 'NQ') -> dict:
    """Get instrument-specific config for a given mode."""
    inst_configs = INSTRUMENT_CONFIGS.get(symbol, {})
    if mode in inst_configs:
        return inst_configs[mode]
    return {'scalp': SCALP_CONFIG, 'swing': SWING_CONFIG, 'meanrev': MEANREV_CONFIG}.get(mode, STRATEGY)


def get_instrument_modes(symbol: str) -> list:
    """Get allowed trading modes for an instrument."""
    return INSTRUMENT_MODES.get(symbol, ['swing'])


def is_tradeable_day(symbol: str, dt_ny=None) -> bool:
    """Check if today is a valid trading day for this instrument."""
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    if dt_ny is None:
        dt_ny = datetime.now(timezone.utc).astimezone(ZoneInfo('America/New_York'))
    dow = dt_ny.weekday()
    if dow >= 5:
        return False
    configs = INSTRUMENT_CONFIGS.get(symbol, {})
    # Check any config's dow setting
    for cfg in configs.values():
        allowed = cfg.get('dow', [0,1,2,3,4])
        if dow in allowed:
            return True
    return False


def check_vix(vix_value):
    if vix_value is None:
        return True
    return float(vix_value) <= STRATEGY['max_vix']


def get_trade_risk(mode: str, score: int, session_quality: int, symbol: str = 'NQ') -> float:
    cfg  = get_mode_config(mode, symbol)
    base = cfg.get('risk_pct', STRATEGY['risk_pct'])
    if score >= 85:   return base
    elif score >= 75: return base * 0.85
    elif score >= 70: return base * 0.70
    return 0


def get_strategy_summary():
    return {
        'min_score':      f"{STRATEGY['min_score']}/100",
        'rr_ratio':       'Scalp 2.5:1 | Swing 4.0:1',
        'risk_per_trade': f"Scalp {NQ_SCALP_CONFIG['risk_pct']}% | Swing {NQ_SWING_CONFIG['risk_pct']}%",
        'max_total_risk': f"{STRATEGY['max_total_risk']}%",
        'trading_days':   'NQ/ES: Tue-Thu | GC: All week',
        'sessions':       'London 10am-1pm UK (Swing) | NY Open 1:30-2:30pm UK (Scalp)',
        'vix_filter':     f"Max VIX {STRATEGY['max_vix']}",
        'instruments':    'NQ (Scalp+Swing) | ES (Swing) | GC (Scalp+Swing)',
        # NQ
        'nq_scalp_sharpe':  '15.39', 'nq_scalp_return': '+173%',  'nq_scalp_wr': '53.4%', 'nq_scalp_exp': '1.31R',
        'nq_swing_sharpe':  '30.28', 'nq_swing_return': '+191%',  'nq_swing_wr': '50.0%', 'nq_swing_exp': '2.52R',
        # ES
        'es_swing_sharpe':   '6.18', 'es_swing_return': '+1241%', 'es_swing_wr': '33.5%', 'es_swing_exp': '2.93R',
        # GC
        'gc_scalp_sharpe':   '9.68', 'gc_scalp_return': '+101%',  'gc_scalp_wr': '52.8%', 'gc_scalp_exp': '0.64R',
        'gc_swing_sharpe':   '9.03', 'gc_swing_return': '+104%',  'gc_swing_wr': '37.3%', 'gc_swing_exp': '0.69R',
        # Legacy keys
        'scalp_sharpe': '15.39', 'scalp_return': '+173%', 'scalp_wr': '53.4%', 'scalp_exp': '1.31R',
        'swing_sharpe': '30.28', 'swing_return': '+191%', 'swing_wr': '50.0%', 'swing_exp': '2.52R',
    }
