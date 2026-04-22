"""
APEX Economic Calendar Filter — calendar_filter.py
====================================================
Blocks all signals in a ±30/60 min window around HIGH-impact US economic events.
Completely standalone — zero dependencies on other setup files.

Block window:  30 min BEFORE  to  60 min AFTER  each event (90 min total).
Data source:   financialmodelingprep.com /economic_calendar (free tier).
Fallback:      hardcoded events for the next 60 days if API fails.
Cache:         6-hour TTL; refreshed on startup and every 6 hours.
"""

import os
import time
import logging
import requests
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger('APEX.Calendar')

# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────

FMP_API_KEY   = os.environ.get('FMP_API_KEY', '') or os.environ.get('FINANCIALMODELINGPREP_API_KEY', '')
FMP_BASE_URL  = 'https://financialmodelingprep.com/api/v3/economic_calendar'
CACHE_TTL_S   = 6 * 3600   # 6 hours

PRE_BLOCK_MIN  = 30   # minutes before event to start blocking
POST_BLOCK_MIN = 60   # minutes after event to stop blocking

# Keywords to match HIGH-impact events (case-insensitive substring match)
HIGH_IMPACT_KEYWORDS = [
    'fomc', 'federal open market', 'fed chair', 'fed press', 'powell',
    'non-farm payroll', 'nonfarm payroll', 'non farm payroll', 'nfp',
    'consumer price index', ' cpi ',
    'producer price index', ' ppi ',
    'gross domestic product', ' gdp ',
    'retail sales',
    'ism manufacturing', 'ism services', 'ism non-manufacturing',
    'jobless claims', 'initial claims', 'unemployment claims',
    'pce price index', 'personal consumption expenditure',
]

# Hardcoded fallback events — updated for next 60 days from 2026-04-22
# Format: (ISO UTC datetime string, event name)
FALLBACK_EVENTS = [
    ('2026-05-01T12:30:00Z', 'Non-Farm Payrolls'),
    ('2026-05-07T18:00:00Z', 'FOMC Rate Decision'),
    ('2026-05-07T18:30:00Z', 'FOMC Press Conference'),
    ('2026-05-13T12:30:00Z', 'CPI'),
    ('2026-05-15T12:30:00Z', 'PPI'),
    ('2026-05-28T12:30:00Z', 'GDP Advance'),
    ('2026-06-04T12:30:00Z', 'Non-Farm Payrolls'),
    ('2026-06-11T18:00:00Z', 'FOMC Rate Decision'),
    ('2026-06-11T18:30:00Z', 'FOMC Press Conference'),
    ('2026-06-12T12:30:00Z', 'CPI'),
]


# ─────────────────────────────────────────────────────────────
#  CALENDAR FILTER
# ─────────────────────────────────────────────────────────────

class CalendarFilter:
    """
    Fetch and cache HIGH-impact US economic events.
    Call is_blocked(symbol, utc_now) before any signal fires.
    """

    def __init__(self):
        self._events: list = []          # list of {'name': str, 'utc_dt': datetime}
        self._last_refresh: float = 0.0
        self._using_fallback: bool = False
        self._warn_sent: set = set()     # event keys for which 15-min warning was sent
        self._blackout_lifted_sent: set = set()

    # ── Refresh ───────────────────────────────────────────────

    def refresh_calendar(self) -> None:
        """Fetch next 7 days of HIGH-impact events from FMP API. Fallback on failure."""
        now_utc = datetime.now(timezone.utc)
        from_dt = now_utc.strftime('%Y-%m-%d')
        to_dt   = (now_utc + timedelta(days=7)).strftime('%Y-%m-%d')

        fetched = []
        if FMP_API_KEY:
            try:
                resp = requests.get(
                    FMP_BASE_URL,
                    params={'from': from_dt, 'to': to_dt, 'apikey': FMP_API_KEY},
                    timeout=10,
                )
                resp.raise_for_status()
                raw = resp.json()
                if isinstance(raw, list):
                    for ev in raw:
                        if ev.get('country', '').upper() != 'US':
                            continue
                        if ev.get('impact', '').lower() != 'high':
                            continue
                        name = ev.get('event', '')
                        if not self._is_high_impact(name):
                            continue
                        dt_str = ev.get('date', '')
                        try:
                            dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=timezone.utc)
                            fetched.append({'name': name, 'utc_dt': dt})
                        except Exception:
                            continue
                    if fetched:
                        self._events = sorted(fetched, key=lambda x: x['utc_dt'])
                        self._using_fallback = False
                        self._last_refresh = time.time()
                        logger.info(f'CalendarFilter: refreshed {len(self._events)} events from FMP API')
                        return
            except Exception as e:
                logger.warning(f'CalendarFilter: FMP API fetch failed ({e}), using fallback')

        # API unavailable — use hardcoded fallback
        self._load_fallback()

    def _load_fallback(self) -> None:
        """Load hardcoded fallback events, filtering to future only."""
        now = datetime.now(timezone.utc)
        events = []
        for dt_str, name in FALLBACK_EVENTS:
            try:
                dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                if dt > now - timedelta(hours=2):  # include events up to 2h in the past
                    events.append({'name': name, 'utc_dt': dt})
            except Exception:
                continue
        self._events = sorted(events, key=lambda x: x['utc_dt'])
        self._using_fallback = True
        self._last_refresh = time.time()
        logger.info(f'CalendarFilter: loaded {len(self._events)} hardcoded fallback events')

    def _is_high_impact(self, name: str) -> bool:
        name_lower = ' ' + name.lower() + ' '
        return any(kw in name_lower for kw in HIGH_IMPACT_KEYWORDS)

    def _ensure_fresh(self) -> None:
        if time.time() - self._last_refresh > CACHE_TTL_S or not self._events:
            self.refresh_calendar()

    # ── Core gate ─────────────────────────────────────────────

    def is_blocked(self, symbol: str, utc_now: datetime = None) -> tuple:
        """
        Return (True, reason_str) if trading is blocked due to an upcoming/recent event.
        Return (False, '') if clear to trade.
        """
        self._ensure_fresh()
        if utc_now is None:
            utc_now = datetime.now(timezone.utc)
        if utc_now.tzinfo is None:
            utc_now = utc_now.replace(tzinfo=timezone.utc)

        for ev in self._events:
            dt = ev['utc_dt']
            diff_min = (utc_now - dt).total_seconds() / 60
            # Blocked if within pre-window (before) or post-window (after)
            if -PRE_BLOCK_MIN <= diff_min <= POST_BLOCK_MIN:
                if diff_min < 0:
                    eta = int(-diff_min)
                    reason = f'{ev["name"]} in {eta} min ({dt.strftime("%H:%M UTC")})'
                else:
                    elapsed = int(diff_min)
                    reason = f'{ev["name"]} +{elapsed} min ago ({dt.strftime("%H:%M UTC")})'
                return True, reason

        return False, ''

    # ── Upcoming events ───────────────────────────────────────

    def get_upcoming_events(self, hours: int = 24) -> list:
        """
        Return list of events in the next N hours.
        Each entry: {'name', 'utc_dt', 'minutes_until', 'blocked_window_start', 'blocked_window_end'}
        """
        self._ensure_fresh()
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours)
        result = []
        for ev in self._events:
            dt = ev['utc_dt']
            if now - timedelta(hours=2) <= dt <= cutoff:
                mins_until = int((dt - now).total_seconds() / 60)
                result.append({
                    'name':                ev['name'],
                    'utc_time':            dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'utc_display':         dt.strftime('%Y-%m-%d %H:%M UTC'),
                    'minutes_until':       mins_until,
                    'blocked_window_start': (dt - timedelta(minutes=PRE_BLOCK_MIN)).strftime('%H:%M UTC'),
                    'blocked_window_end':   (dt + timedelta(minutes=POST_BLOCK_MIN)).strftime('%H:%M UTC'),
                })
        return result

    def get_next_event(self) -> dict | None:
        """Return the next upcoming event, or None."""
        upcoming = self.get_upcoming_events(hours=168)  # next 7 days
        return upcoming[0] if upcoming else None

    def get_current_status(self) -> dict:
        """Return dict for /api/apex/calendar endpoint."""
        self._ensure_fresh()
        now = datetime.now(timezone.utc)
        blocked, reason = self.is_blocked('ANY', now)
        next_ev = self.get_next_event()
        upcoming = self.get_upcoming_events(hours=24)
        return {
            'status':       'BLACKOUT' if blocked else 'CLEAR',
            'reason':       reason,
            'next_event':   next_ev,
            'upcoming':     upcoming[:5],
            'using_fallback': self._using_fallback,
            'last_refresh': datetime.fromtimestamp(self._last_refresh, tz=timezone.utc).strftime('%H:%M UTC')
                            if self._last_refresh else None,
        }

    # ── 15-min warning helper ─────────────────────────────────

    def check_warnings(self) -> list:
        """
        Check if any event is 12-17 minutes away (warning window).
        Returns list of event names that need a Telegram warning sent.
        De-duplicated so each event only warns once.
        """
        self._ensure_fresh()
        now = datetime.now(timezone.utc)
        to_warn = []
        for ev in self._events:
            dt = ev['utc_dt']
            mins_until = (dt - now).total_seconds() / 60
            key = f'warn_{ev["name"]}_{dt.strftime("%Y%m%d%H%M")}'
            if 12 <= mins_until <= 17 and key not in self._warn_sent:
                self._warn_sent.add(key)
                to_warn.append({
                    'name': ev['name'],
                    'utc_time': dt.strftime('%H:%M UTC'),
                    'block_end': (dt + timedelta(minutes=POST_BLOCK_MIN)).strftime('%H:%M UTC'),
                })
        return to_warn

    def check_blackout_lifted(self) -> list:
        """
        Check if blackout just ended (event + POST_BLOCK_MIN passed in last 5 min).
        Returns list of event names for which to send 'blackout lifted' alert.
        """
        self._ensure_fresh()
        now = datetime.now(timezone.utc)
        lifted = []
        for ev in self._events:
            dt     = ev['utc_dt']
            end_dt = dt + timedelta(minutes=POST_BLOCK_MIN)
            key    = f'lifted_{ev["name"]}_{dt.strftime("%Y%m%d%H%M")}'
            mins_since_end = (now - end_dt).total_seconds() / 60
            if 0 <= mins_since_end <= 5 and key not in self._blackout_lifted_sent:
                self._blackout_lifted_sent.add(key)
                lifted.append(ev['name'])
        return lifted


# ─────────────────────────────────────────────────────────────
#  SINGLETON
# ─────────────────────────────────────────────────────────────

_filter_instance: CalendarFilter | None = None


def get_filter() -> CalendarFilter:
    """Return the module-level singleton CalendarFilter (lazy-init)."""
    global _filter_instance
    if _filter_instance is None:
        _filter_instance = CalendarFilter()
        _filter_instance.refresh_calendar()
    return _filter_instance


if __name__ == '__main__':
    import logging as _l
    _l.basicConfig(level=_l.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    cf = CalendarFilter()
    cf.refresh_calendar()
    print('\nStatus:', cf.get_current_status())
    print('\nUpcoming 24h:', cf.get_upcoming_events(24))
    print('\nBlocked now?', cf.is_blocked('MNQ'))
