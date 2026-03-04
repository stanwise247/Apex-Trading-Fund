"""
APEX News Calendar Engine — news_calendar.py
=============================================
Protects against high-impact news events that cause
random, unpredictable price moves.

Features:
  - Economic calendar integration (via free API)
  - Auto-blackout 30min before / 15min after high impact news
  - Recurring event schedule (FOMC, CPI, NFP, etc.)
  - News scoring for trade context
  - Fed speak detection

High Impact Events (always blackout):
  - FOMC Rate Decision + Press Conference
  - CPI / Core CPI
  - NFP (Non-Farm Payrolls)
  - GDP (advance estimate)
  - PCE (Fed's preferred inflation measure)
  - ISM Manufacturing / Services
  - PPI / Retail Sales

Medium Impact (flag only):
  - ADP Employment, Jobless Claims
  - Consumer Confidence, JOLTS, Housing data
"""

import json
import time
import logging
import requests
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger('APEX.NewsCalendar')
NY_TZ  = ZoneInfo('America/New_York')

# =============================================================
#  FOMC DATES 2026
# =============================================================

FOMC_DATES_2026 = [
    '2026-01-29', '2026-03-18', '2026-04-29', '2026-06-10',
    '2026-07-29', '2026-09-16', '2026-10-28', '2026-12-09',
]

HIGH_IMPACT_KEYWORDS = [
    'fomc', 'federal reserve', 'fed decision', 'rate decision', 'interest rate',
    'cpi', 'consumer price', 'core cpi',
    'nfp', 'non-farm', 'payroll', 'jobs report', 'unemployment rate',
    'gdp', 'gross domestic',
    'pce', 'personal consumption',
    'ism manufacturing', 'ism services',
    'ppi', 'producer price',
    'retail sales',
    'powell', 'fed chair',
]

MEDIUM_IMPACT_KEYWORDS = [
    'adp employment', 'jobless claims', 'initial claims',
    'consumer confidence', 'consumer sentiment',
    'jolts', 'job openings',
    'housing starts', 'building permits',
    'durable goods', 'trade balance',
]

# Cache
_calendar_cache = {}
_cache_expiry   = {}
CACHE_TTL       = 3600


# =============================================================
#  CALENDAR FETCH
# =============================================================

def fetch_economic_calendar(days_ahead: int = 7) -> List[Dict]:
    cache_key = f'calendar_{days_ahead}'
    now       = time.time()
    if cache_key in _calendar_cache and now < _cache_expiry.get(cache_key, 0):
        return _calendar_cache[cache_key]

    events = []
    try:
        events = _fetch_from_api(days_ahead)
    except Exception:
        pass

    if not events:
        events = _fetch_from_hardcoded(days_ahead)

    _calendar_cache[cache_key] = events
    _cache_expiry[cache_key]   = now + CACHE_TTL
    return events


def _fetch_from_api(days_ahead: int) -> List[Dict]:
    now_utc  = datetime.now(timezone.utc)
    end_date = now_utc + timedelta(days=days_ahead)
    url      = 'https://economic-calendar.tradingview.com/events'
    params   = {
        'from':      now_utc.strftime('%Y-%m-%dT%H:%M:%S'),
        'to':        end_date.strftime('%Y-%m-%dT%H:%M:%S'),
        'countries': 'US',
    }
    resp = requests.get(url, params=params, timeout=5)
    if resp.status_code != 200:
        return []
    data   = resp.json()
    events = []
    for ev in data.get('result', []):
        impact = ev.get('importance', 0)
        if impact >= 1:
            events.append({
                'title':    ev.get('title', ''),
                'datetime': ev.get('date', ''),
                'impact':   'high' if impact >= 2 else 'medium',
                'country':  ev.get('country', 'US'),
                'source':   'tradingview',
            })
    return events


def _fetch_from_hardcoded(days_ahead: int) -> List[Dict]:
    now_ny = datetime.now(timezone.utc).astimezone(NY_TZ)
    today  = now_ny.date()
    events = []
    for date_str in FOMC_DATES_2026:
        try:
            ev_date   = datetime.strptime(date_str, '%Y-%m-%d').date()
            days_away = (ev_date - today).days
            if 0 <= days_away <= days_ahead:
                events.append({
                    'title':    'FOMC Rate Decision',
                    'datetime': f'{date_str}T14:00:00',
                    'impact':   'high',
                    'country':  'US',
                    'source':   'hardcoded',
                })
                events.append({
                    'title':    'Fed Press Conference',
                    'datetime': f'{date_str}T14:30:00',
                    'impact':   'high',
                    'country':  'US',
                    'source':   'hardcoded',
                })
        except Exception:
            pass
    return events


# =============================================================
#  BLACKOUT DETECTION
# =============================================================

def is_news_blackout(dt_ny: Optional[datetime] = None) -> Tuple[bool, str, int]:
    """
    Check if current time is in a news blackout window.
    Returns (is_blackout, reason, minutes_to_clear)

    High impact:   30min before -> 15min after
    Medium impact: 15min before -> 10min after
    """
    if dt_ny is None:
        dt_ny = datetime.now(timezone.utc).astimezone(NY_TZ)

    events = fetch_economic_calendar(days_ahead=2)

    for event in events:
        try:
            ev_dt_str = event.get('datetime', '')
            if not ev_dt_str:
                continue
            ev_dt = datetime.fromisoformat(ev_dt_str.replace('Z', '+00:00'))
            if ev_dt.tzinfo is None:
                ev_dt = ev_dt.replace(tzinfo=NY_TZ)
            ev_dt_ny = ev_dt.astimezone(NY_TZ)

            impact = event.get('impact', 'medium')
            title  = event.get('title', 'Unknown Event')

            if impact == 'high':
                window_before = timedelta(minutes=30)
                window_after  = timedelta(minutes=15)
            else:
                window_before = timedelta(minutes=15)
                window_after  = timedelta(minutes=10)

            blackout_start = ev_dt_ny - window_before
            blackout_end   = ev_dt_ny + window_after

            if blackout_start <= dt_ny <= blackout_end:
                if dt_ny < ev_dt_ny:
                    mins = int((ev_dt_ny - dt_ny).total_seconds() / 60) + int(window_after.total_seconds() / 60)
                else:
                    mins = int((blackout_end - dt_ny).total_seconds() / 60)
                return True, f'{title} ({impact.upper()})', max(1, mins)

        except Exception as e:
            logger.debug(f'Event parse error: {e}')

    # FOMC day extended blackout (1:30pm - 4pm ET)
    today_str = dt_ny.strftime('%Y-%m-%d')
    if today_str in FOMC_DATES_2026:
        start = dt_ny.replace(hour=13, minute=30, second=0, microsecond=0)
        end   = dt_ny.replace(hour=16, minute=0,  second=0, microsecond=0)
        if start <= dt_ny <= end:
            mins = int((end - dt_ny).total_seconds() / 60)
            return True, 'FOMC Day — extended blackout', max(1, mins)

    return False, '', 0


def check_keyword_blackout(text: str) -> Tuple[bool, str]:
    """Check if a news headline warrants a blackout"""
    if not text:
        return False, ''
    text_lower = text.lower()
    for kw in HIGH_IMPACT_KEYWORDS:
        if kw in text_lower:
            return True, f'High impact: {kw}'
    return False, ''


# =============================================================
#  NEWS SENTIMENT SCORING
# =============================================================

def score_news_context(headlines: List[str], direction: str) -> Tuple[int, Dict]:
    """
    Score news sentiment alignment with trade direction.
    Returns (score -10 to +10, details)
    """
    if not headlines:
        return 0, {}

    is_long = direction in ('long', 'bullish')

    bull_kws = [
        'beat', 'beats', 'surpass', 'record', 'strong', 'growth',
        'rise', 'rises', 'gain', 'rally', 'surge',
        'better than expected', 'above forecast',
        'upgrade', 'buy', 'bullish',
        'rate cut', 'dovish', 'stimulus',
    ]
    bear_kws = [
        'miss', 'misses', 'disappoint', 'weak', 'decline', 'fall',
        'drop', 'loss', 'recession', 'slowdown',
        'worse than expected', 'below forecast',
        'downgrade', 'sell', 'bearish',
        'rate hike', 'hawkish', 'inflation fears',
    ]

    bull_count = 0
    bear_count = 0
    flagged    = []

    for headline in headlines[:10]:
        hl = headline.lower()
        for kw in bull_kws:
            if kw in hl:
                bull_count += 1
                flagged.append(f'+{kw}')
                break
        for kw in bear_kws:
            if kw in hl:
                bear_count += 1
                flagged.append(f'-{kw}')
                break

    net   = bull_count - bear_count
    score = min(10, max(-10, net * 2)) if is_long else min(10, max(-10, -net * 2))

    return score, {
        'bull_signals':  bull_count,
        'bear_signals':  bear_count,
        'net_sentiment': net,
        'keywords':      flagged[:5],
        'aligned':       (is_long and net > 0) or (not is_long and net < 0),
    }


# =============================================================
#  UPCOMING EVENTS
# =============================================================

def get_upcoming_events(hours_ahead: int = 24) -> List[Dict]:
    """Get upcoming high-impact events in next N hours"""
    now_ny   = datetime.now(timezone.utc).astimezone(NY_TZ)
    events   = fetch_economic_calendar(days_ahead=3)
    upcoming = []

    for event in events:
        try:
            ev_dt_str = event.get('datetime', '')
            if not ev_dt_str:
                continue
            ev_dt    = datetime.fromisoformat(ev_dt_str.replace('Z', '+00:00'))
            if ev_dt.tzinfo is None:
                ev_dt = ev_dt.replace(tzinfo=NY_TZ)
            ev_dt_ny = ev_dt.astimezone(NY_TZ)
            hrs_away = (ev_dt_ny - now_ny).total_seconds() / 3600
            if 0 <= hrs_away <= hours_ahead:
                upcoming.append({
                    **event,
                    'hours_away': round(hrs_away, 1),
                    'ny_time':    ev_dt_ny.strftime('%a %H:%M ET'),
                    'uk_time':    ev_dt_ny.astimezone(
                                      ZoneInfo('Europe/London')
                                  ).strftime('%H:%M UK'),
                })
        except Exception:
            continue

    # Add FOMC from hardcoded
    today = now_ny.date()
    for date_str in FOMC_DATES_2026:
        try:
            ev_date   = datetime.strptime(date_str, '%Y-%m-%d').date()
            days_away = (ev_date - today).days
            hrs_away  = days_away * 24.0
            if 0 <= hrs_away <= hours_ahead:
                upcoming.append({
                    'title':      'FOMC Rate Decision',
                    'impact':     'high',
                    'ny_time':    f'{date_str} 14:00 ET',
                    'uk_time':    f'{date_str} 19:00 UK',
                    'hours_away': hrs_away,
                    'source':     'hardcoded',
                })
        except Exception:
            pass

    return sorted(upcoming, key=lambda x: x.get('hours_away', 999))


def get_news_summary_line() -> str:
    """One-line news summary for Telegram alerts"""
    blackout, reason, mins = is_news_blackout()
    if blackout:
        return f'⚠️ NEWS BLACKOUT: {reason} ({mins}min remaining)'
    upcoming = get_upcoming_events(hours_ahead=4)
    high     = [e for e in upcoming if e.get('impact') == 'high']
    if high:
        ev = high[0]
        return f'⚠️ {ev["title"]} in {ev["hours_away"]:.1f}h ({ev.get("uk_time", "")})'
    return '📰 No high-impact news in next 4h ✅'


# =============================================================
#  MASTER TRADE GATE
# =============================================================

def should_trade(dt_ny: Optional[datetime] = None) -> Tuple[bool, str]:
    """
    Master check: is it safe to trade from a news perspective?
    Returns (ok_to_trade, reason)
    """
    blackout, reason, mins = is_news_blackout(dt_ny)
    if blackout:
        return False, f'News blackout: {reason} ({mins}min to clear)'
    return True, 'Clear'


if __name__ == '__main__':
    # Quick test
    print('News blackout check:')
    ok, reason = should_trade()
    print(f'  OK to trade: {ok}  Reason: {reason}')
    print()
    print('Upcoming events (24h):')
    for ev in get_upcoming_events(24):
        print(f'  {ev.get("uk_time","?")} — {ev["title"]} [{ev["impact"]}]')
    print()
    print(get_news_summary_line())
