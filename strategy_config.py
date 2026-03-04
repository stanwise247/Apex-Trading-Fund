"""
APEX Strategy Configuration — strategy_config.py
==================================================
Locked-in optimised settings from parameter optimisation run.
Edit this file to update strategy parameters.

Last optimised: 2026-03-03
Data: NQ 14 months, 73 trades, Sharpe 2.02
"""

# =============================================================
#  OPTIMISED PARAMETERS (from optimiser.py results)
# =============================================================

STRATEGY = {

    # --- Scoring ---
    'min_score':        55,     # Minimum total score to fire alert (0-100)
    'rr_ratio':         3.0,    # Risk:Reward ratio for targets
    'risk_pct':         2.0,    # % of account to risk per trade

    # --- Timeframes ---
    'entry_timeframes': ['15min', '5min'],   # Timeframes to scan for entries

    # --- Days of week (0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri) ---
    'allowed_days':     [1, 2, 3],           # Tue, Wed, Thu only
    'blocked_days':     [0, 4],              # Block Monday and Friday

    # --- Sessions (all times Eastern/NY) ---
    # Based on optimiser: London 5-8am and NY Open 9:30-9:45 are the only
    # positive expectancy windows
    'tradeable_sessions': [
        {'name': 'London Prime',  'start': (5, 0),  'end': (8, 0),  'quality': 85},
        {'name': 'NY Open',       'start': (9, 30), 'end': (9, 45), 'quality': 95},
        {'name': 'NY Morning',    'start': (9, 45), 'end': (11, 30),'quality': 70},
    ],

    # --- VIX filter ---
    'max_vix':          25.0,   # Block trades when VIX above this level

    # --- Risk management ---
    'max_total_risk':   5.0,    # Max % total account at risk across all open trades
    'max_trades_day':   3,      # Max trades per day total
    'max_trades_session': 1,    # Max trades per session window

    # --- Trade types ---
    'allow_intraday':   True,   # Close by end of session
    'allow_overnight':  True,   # Hold open positions overnight
    'partial_exit_r':   1.5,    # Take 50% off at this R multiple
    'breakeven_after':  1.5,    # Move stop to BE after this R

    # --- Paper trading ---
    'auto_paper_trade': True,   # Auto open paper trades on signals
    'paper_balance':    10000,  # Starting paper balance

    # --- Telegram ---
    'send_alerts':      True,   # Send Telegram alerts
    'alert_min_score':  55,     # Minimum score to send alert
}

# =============================================================
#  SESSION HELPER — returns True if current time is tradeable
# =============================================================

def is_tradeable_session(dt_ny=None):
    """
    Returns (bool, session_name, quality)
    dt_ny: datetime in NY timezone. If None, uses current time.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    if dt_ny is None:
        from datetime import timezone
        dt_ny = datetime.now(timezone.utc).astimezone(ZoneInfo('America/New_York'))

    h = dt_ny.hour
    m = dt_ny.minute
    dow = dt_ny.weekday()

    # Block weekends
    if dow >= 5:
        return False, 'Weekend', 0

    # Block bad days
    if dow in STRATEGY['blocked_days']:
        day_names = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday'}
        return False, day_names[dow], 0

    # Check each tradeable session
    for sess in STRATEGY['tradeable_sessions']:
        sh, sm = sess['start']
        eh, em = sess['end']
        start_mins = sh * 60 + sm
        end_mins   = eh * 60 + em
        now_mins   = h  * 60 + m
        if start_mins <= now_mins < end_mins:
            return True, sess['name'], sess['quality']

    return False, 'Off Hours', 0


def check_vix(vix_value):
    """Returns True if VIX is within acceptable range"""
    if vix_value is None:
        return True
    return float(vix_value) <= STRATEGY['max_vix']


def get_trade_risk(score_pct, session_quality):
    """
    Returns suggested risk % based on setup quality.
    Higher quality setups get full risk, lower quality get reduced risk.
    """
    base_risk = STRATEGY['risk_pct']
    if score_pct >= 75 and session_quality >= 85:
        return base_risk          # Full risk on A+ setups
    elif score_pct >= 65:
        return base_risk * 0.75   # 75% risk on A setups
    elif score_pct >= 55:
        return base_risk * 0.5    # 50% risk on B setups
    return 0


def get_strategy_summary():
    """Returns human-readable summary of current strategy settings"""
    return {
        'min_score':    f"{STRATEGY['min_score']}/100",
        'rr_ratio':     f"{STRATEGY['rr_ratio']}:1",
        'risk_per_trade': f"{STRATEGY['risk_pct']}%",
        'max_total_risk': f"{STRATEGY['max_total_risk']}%",
        'entry_timeframes': ', '.join(STRATEGY['entry_timeframes']),
        'trading_days': 'Tuesday, Wednesday, Thursday',
        'sessions':     'London Prime (10am-1pm UK), NY Open (2:30-2:45pm UK)',
        'vix_filter':   f"Max VIX {STRATEGY['max_vix']}",
        'max_trades':   f"{STRATEGY['max_trades_day']} per day",
        'overnight':    'Yes — holds open positions overnight',
        'auto_paper':   'Yes — auto enters paper trades',
        'sharpe':       '2.02 (backtested)',
        'win_rate':     '28.8% (backtested)',
        'annual_return':'38.2% (backtested on $10k)',
        'max_drawdown': '23.6% (backtested)',
    }
