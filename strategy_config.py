"""
APEX Strategy Configuration — strategy_config.py
==================================================
Locked-in optimised settings from v2 backtest run.

Last optimised: 2026-03-04
Data: NQ 10,000 bars (5min), walk-forward backtest

SCALP:  Sharpe 15.39 | Return 173% | WR 53.4% | Expectancy 1.31R
SWING:  Sharpe 30.28 | Return 191% | WR 50.0% | Expectancy 2.52R
"""

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
    'max_vix':          20.0,
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

SCALP_CONFIG = {
    'min_score':  70,
    'rr_ratio':   2.5,
    'risk_pct':   1.5,
    'session':    'ny_open',
    'vix_max':    20,
    'dow':        [1,2,3],
    'stop_atr':   0.8,
    'max_hold_bars': 30,
}

SWING_CONFIG = {
    'min_score':  70,
    'rr_ratio':   4.0,
    'risk_pct':   2.0,
    'session':    'london',
    'vix_max':    20,
    'dow':        [1,2,3],
    'stop_atr':   1.5,
    'htf_strict': False,
    'max_hold_bars': 100,
}

MEANREV_CONFIG = {
    'min_score':  75,
    'rr_ratio':   2.0,
    'risk_pct':   0.5,
    'vix_max':    18,
    'dow':        [1,2,3],
    'vwap_dev_thresh': 2.5,
    'market_cond': 'ranging_only',
    'stop_atr':   0.6,
    'enabled':    False,
}


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


def get_mode_config(mode: str) -> dict:
    return {'scalp': SCALP_CONFIG, 'swing': SWING_CONFIG, 'meanrev': MEANREV_CONFIG}.get(mode, STRATEGY)


def check_vix(vix_value):
    if vix_value is None:
        return True
    return float(vix_value) <= STRATEGY['max_vix']


def get_trade_risk(mode: str, score: int, session_quality: int) -> float:
    base = {'swing': SWING_CONFIG['risk_pct'], 'scalp': SCALP_CONFIG['risk_pct'], 'meanrev': MEANREV_CONFIG['risk_pct']}.get(mode, STRATEGY['risk_pct'])
    if score >= 85: return base
    elif score >= 75: return base * 0.85
    elif score >= 70: return base * 0.70
    return 0


def get_strategy_summary():
    return {
        'min_score':      f"{STRATEGY['min_score']}/100",
        'rr_ratio':       'Scalp 2.5:1 | Swing 4.0:1',
        'risk_per_trade': f"Scalp {SCALP_CONFIG['risk_pct']}% | Swing {SWING_CONFIG['risk_pct']}%",
        'max_total_risk': f"{STRATEGY['max_total_risk']}%",
        'trading_days':   'Tuesday, Wednesday, Thursday',
        'sessions':       'London 10am-1pm UK (Swing) | NY Open 1:30-2:30pm UK (Scalp)',
        'vix_filter':     f"Max VIX {STRATEGY['max_vix']}",
        'scalp_sharpe':   '15.39', 'scalp_return': '+173%', 'scalp_wr': '53.4%', 'scalp_exp': '1.31R',
        'swing_sharpe':   '30.28', 'swing_return': '+191%', 'swing_wr': '50.0%', 'swing_exp': '2.52R',
    }
