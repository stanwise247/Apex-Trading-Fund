"""
APEX Session Filter — session_filter.py
=========================================
Filters trades based on:
  - Time of day (session quality windows)
  - Day of week effects
  - VIX volatility regime
  - High-impact news event blackout periods
  - Pre/post market behaviour
"""

from datetime import datetime, time, timezone, timedelta
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo('America/New_York')

# =============================================================
#  SESSION WINDOWS (all times Eastern)
# =============================================================

SESSION_WINDOWS = {
    'pre_market': {
        'start': time(4, 0), 'end': time(9, 29),
        'quality': 20, 'label': 'Pre-Market',
        'note': 'Low liquidity — avoid unless strong overnight catalyst',
        'trade': False,
    },
    'ny_open_first15': {
        'start': time(9, 30), 'end': time(9, 45),
        'quality': 60, 'label': 'NY Open First 15min',
        'note': 'High volatility — wait for direction confirmation, not first candle',
        'trade': True,
    },
    'ny_open_prime': {
        'start': time(9, 45), 'end': time(11, 30),
        'quality': 95, 'label': 'NY Open Prime',
        'note': 'Best session. Highest volume, cleanest setups, most reliable signals',
        'trade': True,
    },
    'midday_chop': {
        'start': time(11, 30), 'end': time(13, 30),
        'quality': 25, 'label': 'Midday Chop',
        'note': 'Institutional lunch break. Low volume, choppy, stop-hunting zone. Avoid.',
        'trade': False,
    },
    'afternoon_reopen': {
        'start': time(13, 30), 'end': time(15, 0),
        'quality': 65, 'label': 'Afternoon Reopen',
        'note': 'Institutions return. Good for continuation of morning trend.',
        'trade': True,
    },
    'power_hour': {
        'start': time(15, 0), 'end': time(16, 0),
        'quality': 80, 'label': 'Power Hour',
        'note': 'Strong directional moves. End-of-day positioning and squaring.',
        'trade': True,
    },
    'post_market': {
        'start': time(16, 0), 'end': time(20, 0),
        'quality': 15, 'label': 'Post-Market',
        'note': 'Very low liquidity. Avoid.',
        'trade': False,
    },
    'overnight': {
        'start': time(20, 0), 'end': time(3, 59),
        'quality': 10, 'label': 'Overnight',
        'note': 'NQ futures trade overnight but liquidity is poor. Avoid.',
        'trade': False,
    },
    'london_open': {
        'start': time(3, 0), 'end': time(4, 30),
        'quality': 45, 'label': 'London Open',
        'note': 'Can create momentum that carries into NY. Watch for direction clue.',
        'trade': False,
    },
}

# Day of week score adjustments
DOW_SCORES = {
    0: {'label': 'Monday',    'adjustment': -5,
        'note': 'Often reverses Friday trend in first hour. Wait for confirmation.'},
    1: {'label': 'Tuesday',   'adjustment': +10,
        'note': 'Strong trending day historically. Best day for momentum trades.'},
    2: {'label': 'Wednesday', 'adjustment': +5,
        'note': 'Good trending day. FOMC days are Wednesdays — check calendar.'},
    3: {'label': 'Thursday',  'adjustment': +8,
        'note': 'Strong day. Often continuation of Wednesday moves.'},
    4: {'label': 'Friday',    'adjustment': -10,
        'note': 'Position squaring into weekend. Avoid holding through close.'},
}

# Known recurring high-impact events (approximations — real calendar via API)
HIGH_IMPACT_KEYWORDS = [
    'FOMC', 'Federal Reserve', 'Fed decision', 'rate decision',
    'CPI', 'inflation', 'PPI', 'PCE',
    'NFP', 'non-farm payroll', 'jobs report', 'unemployment',
    'GDP', 'retail sales', 'ISM',
    'earnings', 'NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META',
]


def get_session_quality(dt=None):
    """
    Given a datetime (or now), return the current session quality score
    and whether trading is recommended.
    """
    if dt is None:
        dt = datetime.now(NY_TZ)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=NY_TZ)
    else:
        dt = dt.astimezone(NY_TZ)

    t = dt.time()
    dow = dt.weekday()

    # Find matching session
    current_session = None
    for name, sess in SESSION_WINDOWS.items():
        start = sess['start']
        end   = sess['end']
        # Handle overnight crossing midnight
        if start > end:
            if t >= start or t < end:
                current_session = (name, sess)
                break
        else:
            if start <= t < end:
                current_session = (name, sess)
                break

    if current_session is None:
        current_session = ('overnight', SESSION_WINDOWS['overnight'])

    name, sess = current_session
    dow_info    = DOW_SCORES.get(dow, {'label': 'Unknown', 'adjustment': 0, 'note': ''})

    base_quality    = sess['quality']
    adjusted_quality= max(0, min(100, base_quality + dow_info['adjustment']))

    # Friday afternoon reduction
    if dow == 4 and t >= time(14, 0):
        adjusted_quality = max(0, adjusted_quality - 15)

    # Monday morning reduction
    if dow == 0 and t < time(10, 30):
        adjusted_quality = max(0, adjusted_quality - 10)

    return {
        'session':          name,
        'label':            sess['label'],
        'quality':          adjusted_quality,
        'trade_recommended':sess['trade'] and adjusted_quality >= 50,
        'note':             sess['note'],
        'day':              dow_info['label'],
        'day_note':         dow_info['note'],
        'ny_time':          dt.strftime('%H:%M ET'),
        'dow_adjustment':   dow_info['adjustment'],
    }


def get_session_score_for_bar(timestamp_utc):
    """Get session quality for a specific bar timestamp"""
    try:
        if isinstance(timestamp_utc, (int, float)):
            dt = datetime.fromtimestamp(timestamp_utc, tz=timezone.utc)
        elif isinstance(timestamp_utc, str):
            dt = datetime.fromisoformat(timestamp_utc)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = timestamp_utc
        return get_session_quality(dt)
    except Exception:
        return {'quality': 50, 'trade_recommended': True, 'session': 'unknown'}


# =============================================================
#  VIX REGIME FILTER
# =============================================================

def get_vix_regime(vix_value):
    """
    Classify VIX into trading regime.
    Each regime has different strategy implications.
    """
    if vix_value is None:
        return {'regime': 'unknown', 'score_mult': 1.0, 'note': 'VIX data unavailable'}

    v = float(vix_value)

    if v < 13:
        return {
            'regime':     'ultra_low',
            'vix':        v,
            'score_mult': 0.8,
            'note':       'Extreme complacency. Low volatility = small moves. Reduce position size.',
            'action':     'reduce_size',
        }
    elif v < 17:
        return {
            'regime':     'low',
            'vix':        v,
            'score_mult': 1.0,
            'note':       'Normal low-vol environment. Standard approach.',
            'action':     'normal',
        }
    elif v < 22:
        return {
            'regime':     'normal',
            'vix':        v,
            'score_mult': 1.1,
            'note':       'Healthy volatility. Good trending conditions.',
            'action':     'normal',
        }
    elif v < 28:
        return {
            'regime':     'elevated',
            'vix':        v,
            'score_mult': 1.15,
            'note':       'Elevated vol. Wider stops needed. Better moves when they come.',
            'action':     'widen_stops',
        }
    elif v < 35:
        return {
            'regime':     'high',
            'vix':        v,
            'score_mult': 0.9,
            'note':       'High vol. Choppy, whipsaws common. Only highest-quality setups.',
            'action':     'high_quality_only',
        }
    else:
        return {
            'regime':     'extreme',
            'vix':        v,
            'score_mult': 0.6,
            'note':       'Extreme fear. Gaps, halts, circuit breakers possible. Reduce size or stand aside.',
            'action':     'stand_aside',
        }


# =============================================================
#  NEWS BLACKOUT DETECTOR
# =============================================================

# Standard recurring blackout windows (Eastern Time)
RECURRING_BLACKOUTS = [
    # FOMC — 8 times per year, Wednesday 2pm ET announcement
    {'name': 'FOMC Decision', 'type': 'recurring',
     'blackout_before_min': 60, 'blackout_after_min': 30,
     'impact': 'extreme'},
    # NFP — first Friday of month, 8:30am ET
    {'name': 'Non-Farm Payrolls', 'type': 'monthly',
     'blackout_before_min': 30, 'blackout_after_min': 30,
     'impact': 'high'},
    # CPI — monthly, 8:30am ET
    {'name': 'CPI', 'type': 'monthly',
     'blackout_before_min': 30, 'blackout_after_min': 20,
     'impact': 'high'},
]

def check_news_blackout(news_headlines=None, current_time=None):
    """
    Check if we're in a news blackout period.
    Uses news headlines from APEX news feed to detect upcoming events.
    """
    if news_headlines is None:
        news_headlines = []

    blackout   = False
    reason     = None
    impact     = 'none'

    for headline in news_headlines[:20]:
        h_lower = headline.lower() if isinstance(headline, str) else ''
        for keyword in HIGH_IMPACT_KEYWORDS:
            if keyword.lower() in h_lower:
                blackout = True
                reason   = f'High-impact news detected: {keyword}'
                impact   = 'high'
                break
        if blackout:
            break

    return {
        'blackout':  blackout,
        'reason':    reason,
        'impact':    impact,
        'trade_ok':  not blackout,
    }


# =============================================================
#  COMBINED SESSION + REGIME SCORE
# =============================================================

def get_full_context_score(vix_value=None, news_headlines=None, dt=None):
    """
    Combined context score for the current market environment.
    Returns a score from 0-100 and a full breakdown.
    """
    session   = get_session_quality(dt)
    vix_reg   = get_vix_regime(vix_value)
    news_bl   = check_news_blackout(news_headlines)

    # Start with session quality
    score = session['quality']

    # VIX adjustment
    score = score * vix_reg.get('score_mult', 1.0)

    # News blackout kills the score
    if news_bl['blackout']:
        score = min(score, 20)

    score = max(0, min(100, score))

    trade_ok = (
        session['trade_recommended'] and
        not news_bl['blackout'] and
        vix_reg.get('action') not in ('stand_aside',) and
        score >= 40
    )

    return {
        'score':       round(score, 1),
        'trade_ok':    trade_ok,
        'session':     session,
        'vix_regime':  vix_reg,
        'news_status': news_bl,
        'summary':     (
            f"{session['label']} ({session['quality']}/100) | "
            f"VIX {vix_value or '?'} [{vix_reg['regime']}] | "
            f"{'⚠ NEWS BLACKOUT' if news_bl['blackout'] else 'News OK'}"
        ),
    }
