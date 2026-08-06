"""
ict_engine.py — ICT (Inner Circle Trader) Pattern Detection Engine
====================================================================
Pure computation only. Every function takes plain OHLCV bar lists (or the
output of another function in this module) and returns structured data.
No database access, no network calls, no side effects — this module never
imports db.py.

Bar contract: every function expects `bars` as a chronologically ascending
(oldest -> newest) list of dicts with keys: ts (int, unix seconds), open,
high, low, close, volume (floats). Callers are responsible for excluding
any still-forming/live candle — every bar passed in is assumed CLOSED.
This is a deliberate purity boundary: "is the last bar closed yet" is a
wall-clock question the scheduler answers, not this module.

Where the brief left a rule ambiguous, the choice made and the reasoning
is documented inline at the point of the decision — always the more
conservative (fewer false positives) reading.
"""

from __future__ import annotations
import math
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────
#  1.1 — SWING STRUCTURE
# ─────────────────────────────────────────────────────────────────────────

SWING_N = {'1min': 2, '5min': 2, '15min': 3, '1hour': 3, '4hour': 5}


def _swing_n_for(timeframe: str) -> int:
    return SWING_N.get(timeframe, 3)


def _find_swings(bars: list, n: int) -> list:
    """Core swing scan at a given lookback N. Only indices with N full bars
    on both sides are examined, so every swing returned is confirmed by
    construction — there is no unconfirmed case to filter out afterward."""
    swings = []
    total = len(bars)
    for i in range(n, total - n):
        c = bars[i]
        left = bars[i - n:i]
        right = bars[i + 1:i + n + 1]
        if c['high'] > max(b['high'] for b in left) and c['high'] > max(b['high'] for b in right):
            swings.append({'type': 'high', 'price': c['high'], 'timestamp': c['ts'], 'confirmed': True})
        if c['low'] < min(b['low'] for b in left) and c['low'] < min(b['low'] for b in right):
            swings.append({'type': 'low', 'price': c['low'], 'timestamp': c['ts'], 'confirmed': True})
    return swings


def detect_swings(bars: list, timeframe: str) -> list:
    """Swing high: candle's high > the N candles before AND after it.
    Swing low: candle's low < the N candles before AND after it.
    N = 2 for 1M/5M, 3 for 15M/1H, 5 for 4H. Strictly-greater/strictly-less
    (equal highs do not both qualify as swings — see detect_equal_highs_lows
    for that case, which is a deliberately separate concept)."""
    n = _swing_n_for(timeframe)
    swings = _find_swings(bars, n)
    for s in swings:
        s['timeframe'] = timeframe
    swings.sort(key=lambda s: s['timestamp'])
    return swings


# ─────────────────────────────────────────────────────────────────────────
#  1.2 — MARKET STRUCTURE
# ─────────────────────────────────────────────────────────────────────────

def classify_market_structure(swings: list) -> dict:
    """Looks at the last two swing highs and last two swing lows
    independently: last_hh/last_lh are mutually exclusive (a swing high is
    either higher or lower than its predecessor), same for last_hl/last_ll.
    structure = BULLISH only if the most recent high is a HH *and* the most
    recent low is a HL; BEARISH only if LH *and* LL. Any mixed or
    insufficient-data case is RANGING — the conservative default when the
    two swing types disagree about direction."""
    highs = [s for s in swings if s['type'] == 'high']
    lows = [s for s in swings if s['type'] == 'low']
    last_hh = last_lh = last_hl = last_ll = None

    if len(highs) >= 2:
        if highs[-1]['price'] > highs[-2]['price']:
            last_hh = highs[-1]['price']
        elif highs[-1]['price'] < highs[-2]['price']:
            last_lh = highs[-1]['price']
    if len(lows) >= 2:
        if lows[-1]['price'] > lows[-2]['price']:
            last_hl = lows[-1]['price']
        elif lows[-1]['price'] < lows[-2]['price']:
            last_ll = lows[-1]['price']

    if last_hh is not None and last_hl is not None:
        structure = 'BULLISH'
    elif last_lh is not None and last_ll is not None:
        structure = 'BEARISH'
    else:
        structure = 'RANGING'

    return {
        'structure': structure,
        'last_hh': last_hh, 'last_hl': last_hl,
        'last_lh': last_lh, 'last_ll': last_ll,
    }


def detect_bos(swings: list, bars: list) -> list:
    """Bullish BOS: a bar CLOSES above a prior swing high (wick-through does
    not count). Bearish BOS: a bar closes below a prior swing low. Each
    swing level fires at most once, on the first bar whose close breaks it
    (chronologically) — subsequent closes above/below an already-broken
    level are not additional BOS events."""
    events = []
    timeframe = swings[0]['timeframe'] if swings else None
    highs = [s for s in swings if s['type'] == 'high']
    lows = [s for s in swings if s['type'] == 'low']

    for sw in highs:
        for b in bars:
            if b['ts'] <= sw['timestamp']:
                continue
            if b['close'] > sw['price']:
                events.append({'type': 'bullish', 'broken_level': sw['price'],
                                'break_candle_timestamp': b['ts'], 'timeframe': timeframe,
                                'significance': 'bos'})
                break
    for sw in lows:
        for b in bars:
            if b['ts'] <= sw['timestamp']:
                continue
            if b['close'] < sw['price']:
                events.append({'type': 'bearish', 'broken_level': sw['price'],
                                'break_candle_timestamp': b['ts'], 'timeframe': timeframe,
                                'significance': 'bos'})
                break

    events.sort(key=lambda e: e['break_candle_timestamp'])
    return events


def detect_choch(swings: list, bars: list) -> list:
    """CHoCH is evaluated against the *current* established structure only
    (the most recent HL in a bullish structure, or most recent LH in a
    bearish one) — not every historical structure flip in the window. This
    matches the brief's framing of CHoCH as "the first sign of structural
    shift" from where things stand right now, and returns at most one
    event (0 or 1): the current pending/confirmed CHoCH condition, if any.
    RANGING structure has no "character" to change, so it never fires."""
    timeframe = swings[0]['timeframe'] if swings else None
    ms = classify_market_structure(swings)
    lows = [s for s in swings if s['type'] == 'low']
    highs = [s for s in swings if s['type'] == 'high']

    if ms['structure'] == 'BULLISH' and ms['last_hl'] is not None:
        ref_low = next(s for s in reversed(lows) if s['price'] == ms['last_hl'])
        for b in bars:
            if b['ts'] <= ref_low['timestamp']:
                continue
            if b['close'] < ref_low['price']:
                return [{'type': 'bearish', 'broken_level': ref_low['price'],
                          'break_candle_timestamp': b['ts'], 'timeframe': timeframe,
                          'significance': 'choch'}]
    elif ms['structure'] == 'BEARISH' and ms['last_lh'] is not None:
        ref_high = next(s for s in reversed(highs) if s['price'] == ms['last_lh'])
        for b in bars:
            if b['ts'] <= ref_high['timestamp']:
                continue
            if b['close'] > ref_high['price']:
                return [{'type': 'bullish', 'broken_level': ref_high['price'],
                          'break_candle_timestamp': b['ts'], 'timeframe': timeframe,
                          'significance': 'choch'}]
    return []


# ─────────────────────────────────────────────────────────────────────────
#  1.3 — FAIR VALUE GAPS
# ─────────────────────────────────────────────────────────────────────────

def detect_fvgs(bars: list, min_gap_pct: float = 0.05) -> list:
    """Three-candle imbalance. formed_at is timestamped at the THIRD
    candle's close — the point at which the gap becomes a confirmed
    pattern, not the first candle (which alone proves nothing). Status is
    evaluated only from bars strictly after the 3rd candle (the formation
    candle's own wick sits exactly on the zone boundary by construction and
    never counts as entering it).
      FRESH:     no bar since formation has traded into the zone at all
      MITIGATED: a bar has traded 50%+ into the zone but never closed
                 fully through it
      VOID:      a bar closed fully through the zone (gap fully filled)
    """
    fvgs = []
    n = len(bars)
    for i in range(n - 2):
        c1, c3 = bars[i], bars[i + 2]
        if c3['low'] > c1['high']:
            zone_low, zone_high = c1['high'], c3['low']
            kind = 'bullish'
        elif c3['high'] < c1['low']:
            zone_low, zone_high = c3['high'], c1['low']
            kind = 'bearish'
        else:
            continue
        if zone_high <= zone_low:
            continue
        mid = (zone_high + zone_low) / 2
        gap_pct = (zone_high - zone_low) / mid * 100 if mid else 0
        if gap_pct < min_gap_pct:
            continue

        status = 'FRESH'
        mitigation_pct = 0.0
        for b in bars[i + 3:]:
            if kind == 'bullish':
                if b['close'] < zone_low:
                    status, mitigation_pct = 'VOID', 100.0
                    break
                if b['low'] <= zone_high:
                    pen = (zone_high - max(b['low'], zone_low)) / (zone_high - zone_low) * 100
                    mitigation_pct = max(mitigation_pct, pen)
                    status = 'MITIGATED' if mitigation_pct >= 50 else status
            else:
                if b['close'] > zone_high:
                    status, mitigation_pct = 'VOID', 100.0
                    break
                if b['high'] >= zone_low:
                    pen = (min(b['high'], zone_high) - zone_low) / (zone_high - zone_low) * 100
                    mitigation_pct = max(mitigation_pct, pen)
                    status = 'MITIGATED' if mitigation_pct >= 50 else status

        fvgs.append({
            'type': kind, 'zone_high': zone_high, 'zone_low': zone_low,
            'formed_at': c3['ts'], 'status': status,
            'mitigation_pct': round(mitigation_pct, 1),
        })
    return fvgs


# ─────────────────────────────────────────────────────────────────────────
#  1.4 — ORDER BLOCKS
# ─────────────────────────────────────────────────────────────────────────

def _last_opposite_index(bars: list, before_idx: int, want: str) -> Optional[int]:
    """Nearest candle strictly before before_idx matching want ('red'/'green')."""
    for k in range(before_idx - 1, -1, -1):
        c = bars[k]
        if want == 'red' and c['close'] < c['open']:
            return k
        if want == 'green' and c['close'] > c['open']:
            return k
    return None


def detect_order_blocks(bars: list, swings: list) -> list:
    """OB candle = the last opposing-colour candle immediately before the
    impulse leg that produces a confirmed BOS. "Creates a BOS or takes out
    a swing high/low" is conservatively read as requiring an actual
    close-confirmed BOS (the stricter of the two, per instructions to
    prefer fewer false positives when a rule is ambiguous) — a wick-only
    high/low takeout without a closing BOS does not qualify an OB here.

    Impulse size = the maximum favourable excursion from the OB candle's
    opposing extreme (low for bullish, high for bearish), measured through
    the break candle. Must be >= 1.5x the OB candle's own high-low range.

    Status is evaluated only from bars AFTER the break candle, not from the
    OB candle itself — "price hasn't returned to the zone" means returned
    after the impulse leg is already confirmed (by its own BOS); a wick on
    the very next candle, still inside the same expansion move, is not a
    "return" and would otherwise mark almost every OB TESTED immediately:
      ACTIVE:   price has never returned to the zone since the break
      TESTED:   price has traded into zone_low..zone_high since the break
      BREACHED: a close has gone fully through the zone since the break
    """
    obs = []
    bos_events = detect_bos(swings, bars)
    ts_index = {b['ts']: i for i, b in enumerate(bars)}

    for ev in bos_events:
        break_idx = ts_index.get(ev['break_candle_timestamp'])
        if break_idx is None:
            continue
        if ev['type'] == 'bullish':
            k = _last_opposite_index(bars, break_idx, 'red')
            if k is None:
                continue
            ob_candle = bars[k]
            ob_range = ob_candle['high'] - ob_candle['low']
            if ob_range <= 0:
                continue
            max_high = max(b['high'] for b in bars[k + 1:break_idx + 1])
            impulse_size = max_high - ob_candle['low']
            if impulse_size < 1.5 * ob_range:
                continue
            zone_high, zone_low = ob_candle['high'], ob_candle['low']
            status = 'ACTIVE'
            for b in bars[break_idx + 1:]:
                if b['close'] < zone_low:
                    status = 'BREACHED'
                    break
                if b['low'] <= zone_high:
                    status = 'TESTED'
            obs.append({'type': 'bullish', 'zone_high': zone_high, 'zone_low': zone_low,
                        'formed_at': ob_candle['ts'], 'status': status,
                        'impulse_size': round(impulse_size, 4)})
        else:
            k = _last_opposite_index(bars, break_idx, 'green')
            if k is None:
                continue
            ob_candle = bars[k]
            ob_range = ob_candle['high'] - ob_candle['low']
            if ob_range <= 0:
                continue
            min_low = min(b['low'] for b in bars[k + 1:break_idx + 1])
            impulse_size = ob_candle['high'] - min_low
            if impulse_size < 1.5 * ob_range:
                continue
            zone_high, zone_low = ob_candle['high'], ob_candle['low']
            status = 'ACTIVE'
            for b in bars[break_idx + 1:]:
                if b['close'] > zone_high:
                    status = 'BREACHED'
                    break
                if b['high'] >= zone_low:
                    status = 'TESTED'
            obs.append({'type': 'bearish', 'zone_high': zone_high, 'zone_low': zone_low,
                        'formed_at': ob_candle['ts'], 'status': status,
                        'impulse_size': round(impulse_size, 4)})

    # De-duplicate: the same OB candle can be re-derived from multiple BOS
    # events further along the same leg — keep one row per (formed_at, type).
    seen = set()
    deduped = []
    for ob in obs:
        key = (ob['formed_at'], ob['type'])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ob)
    deduped.sort(key=lambda o: o['formed_at'])
    return deduped


# ─────────────────────────────────────────────────────────────────────────
#  1.5 — LIQUIDITY
# ─────────────────────────────────────────────────────────────────────────

def detect_equal_highs_lows(swings: list, tolerance_pct: float = 0.15) -> list:
    """Greedy price-clustering (not time-adjacency) of same-type swings:
    sort by price, start a new cluster whenever the gap to the running
    cluster average exceeds tolerance_pct. Needs >= 2 members to count.
    status is always INTACT here — sweep/reclaim status requires bars,
    which this function deliberately does not take (see
    detect_liquidity_sweep, a separate pass over the same clusters)."""
    results = []
    for kind, type_label in (('high', 'equal_high'), ('low', 'equal_low')):
        pts = sorted([s for s in swings if s['type'] == kind], key=lambda s: s['price'])
        cluster = []
        for s in pts:
            if not cluster:
                cluster = [s]
                continue
            avg = sum(m['price'] for m in cluster) / len(cluster)
            if avg and abs(s['price'] - avg) / avg * 100 <= tolerance_pct:
                cluster.append(s)
            else:
                if len(cluster) >= 2:
                    results.append(_cluster_to_row(cluster, type_label))
                cluster = [s]
        if len(cluster) >= 2:
            results.append(_cluster_to_row(cluster, type_label))
    results.sort(key=lambda r: r['formed_at'])
    return results


def _cluster_to_row(cluster: list, type_label: str) -> dict:
    prices = [m['price'] for m in cluster]
    return {
        'type': type_label,
        'price_avg': sum(prices) / len(prices),
        'price_range': max(prices) - min(prices),
        'swing_count': len(cluster),
        'formed_at': max(m['timestamp'] for m in cluster),
        'status': 'INTACT',
    }


def detect_liquidity_sweep(bars: list, eq_highs_lows: list, lookback: int = 5) -> list:
    """A sweep = price trades through the cluster's average level, then a
    CLOSE reclaims the other side, within `lookback` bars of the breach bar
    (inclusive of the breach bar itself). Only the first breach+reclaim per
    cluster is reported — this is a single real-time detection pass, not a
    full history of every touch."""
    sweeps = []
    for lvl in eq_highs_lows:
        level = lvl['price_avg']
        if lvl['type'] == 'equal_high':
            breach_idx = None
            for i, b in enumerate(bars):
                if b['ts'] <= lvl['formed_at']:
                    continue
                if b['high'] > level:
                    breach_idx = i
                    break
            if breach_idx is None:
                continue
            window = bars[breach_idx:breach_idx + lookback]
            sweep_high = max(b['high'] for b in window)
            for b in window:
                if b['close'] < level:
                    sweeps.append({'type': 'buy_side_swept', 'level': level,
                                    'sweep_high': sweep_high, 'sweep_low': None,
                                    'close_back_inside_bar': b['ts'],
                                    'timestamp': bars[breach_idx]['ts']})
                    break
        else:
            breach_idx = None
            for i, b in enumerate(bars):
                if b['ts'] <= lvl['formed_at']:
                    continue
                if b['low'] < level:
                    breach_idx = i
                    break
            if breach_idx is None:
                continue
            window = bars[breach_idx:breach_idx + lookback]
            sweep_low = min(b['low'] for b in window)
            for b in window:
                if b['close'] > level:
                    sweeps.append({'type': 'sell_side_swept', 'level': level,
                                    'sweep_high': None, 'sweep_low': sweep_low,
                                    'close_back_inside_bar': b['ts'],
                                    'timestamp': bars[breach_idx]['ts']})
                    break
    sweeps.sort(key=lambda s: s['timestamp'])
    return sweeps


def resolve_eql_status(lvl: dict, bars: list, lookback: int = 5) -> str:
    """Combines detect_liquidity_sweep's breach+reclaim signal with a raw
    breach check to produce the 3-state status the brief's equal-highs/lows
    spec calls for: INTACT (never breached), PARTIAL (currently breached,
    reclaim not yet confirmed / lookback window still open), SWEPT
    (breach + reclaim confirmed). Not part of any single detector function
    in the brief — this is the documented reconciliation step the DB-write
    layer runs once per refresh."""
    sweeps = detect_liquidity_sweep(bars, [lvl], lookback=lookback)
    if sweeps:
        return 'SWEPT'
    level = lvl['price_avg']
    for b in bars:
        if b['ts'] <= lvl['formed_at']:
            continue
        if lvl['type'] == 'equal_high' and b['high'] > level:
            return 'PARTIAL'
        if lvl['type'] == 'equal_low' and b['low'] < level:
            return 'PARTIAL'
    return 'INTACT'


def detect_inducement(bars: list, swings: list) -> list:
    """"Major" swings are re-derived using double the timeframe's normal N
    (a stricter, less sensitive lookback that only finds the larger
    structural points) — the brief does not define "major" numerically, so
    this is the documented conservative choice: inducement is only the
    minor swing that sits strictly between two *consecutive* major swings
    of opposite type, i.e. the false extreme the minor swings list found
    that the major pass filtered out entirely.
    """
    if not swings:
        return []
    timeframe = swings[0]['timeframe']
    major_n = _swing_n_for(timeframe) * 2
    major_swings = _find_swings(bars, major_n)
    major_swings.sort(key=lambda s: s['timestamp'])
    if len(major_swings) < 2:
        return []

    results = []
    for a, b in zip(major_swings, major_swings[1:]):
        if a['type'] == b['type']:
            continue
        t0, t1 = a['timestamp'], b['timestamp']
        if a['type'] == 'low' and b['type'] == 'high':
            # bullish leg — inducement is the highest minor swing HIGH between them
            minors = [s for s in swings if s['type'] == 'high' and t0 < s['timestamp'] < t1]
            if minors:
                m = max(minors, key=lambda s: s['price'])
                results.append({'type': 'inducement_high', 'price': m['price'],
                                'timestamp': m['timestamp'], 'major_target': b['price'],
                                'timeframe': timeframe})
        else:
            # bearish leg — inducement is the lowest minor swing LOW between them
            minors = [s for s in swings if s['type'] == 'low' and t0 < s['timestamp'] < t1]
            if minors:
                m = min(minors, key=lambda s: s['price'])
                results.append({'type': 'inducement_low', 'price': m['price'],
                                'timestamp': m['timestamp'], 'major_target': b['price'],
                                'timeframe': timeframe})
    results.sort(key=lambda r: r['timestamp'])
    return results


# ─────────────────────────────────────────────────────────────────────────
#  1.6 — OPTIMAL TRADE ENTRY (OTE)
# ─────────────────────────────────────────────────────────────────────────

def detect_ote_zones(swings: list, fvgs: list, bars: list) -> list:
    """Uses the single most recent structural break (BOS or CHoCH, CHoCH
    taking priority if both reference the same or a later point since it is
    the more significant signal) to define the active impulse leg. The leg
    extends from the swing that preceded the break through to the best
    price reached since (max high for a bullish leg, min low for a
    bearish one) — so the retracement zone stays current as the leg keeps
    extending, rather than freezing at the break candle.

    confluence_level, given this function's inputs are only (swings, fvgs,
    bars) — no order blocks — is necessarily FVG-only:
      'high'   — a FRESH FVG of matching direction overlaps the OTE zone
      'medium' — the triggering break was a CHoCH (the more significant
                 break type) but no FVG overlap
      'low'    — the triggering break was a plain BOS with no FVG overlap
    OB confluence is added separately at the score_ict_setup layer, which
    does receive OB data — see Part 1.8's own "+3 active OB" line.
    """
    if not swings or not bars:
        return []
    timeframe = swings[0]['timeframe']
    bos_events = detect_bos(swings, bars)
    choch_events = detect_choch(swings, bars)

    candidates = bos_events + choch_events
    if not candidates:
        return []
    # Most recent break; CHoCH wins ties since it is the more significant signal.
    candidates.sort(key=lambda e: (e['break_candle_timestamp'], e['significance'] == 'choch'))
    latest = candidates[-1]
    direction = 'bullish' if latest['type'] == 'bullish' else 'bearish'
    break_ts = latest['break_candle_timestamp']

    lows = [s for s in swings if s['type'] == 'low' and s['timestamp'] < break_ts]
    highs = [s for s in swings if s['type'] == 'high' and s['timestamp'] < break_ts]
    since_break = [b for b in bars if b['ts'] <= break_ts]
    if not since_break:
        return []

    if direction == 'bullish':
        if not lows:
            return []
        leg_low_swing = max(lows, key=lambda s: s['timestamp'])
        leg_low = leg_low_swing['price']
        leg_high = max(b['high'] for b in bars if b['ts'] >= leg_low_swing['timestamp'])
        if leg_high <= leg_low:
            return []
        fib_618 = leg_high - 0.618 * (leg_high - leg_low)
        fib_786 = leg_high - 0.786 * (leg_high - leg_low)
        ote_high, ote_low = fib_618, fib_786
    else:
        if not highs:
            return []
        leg_high_swing = max(highs, key=lambda s: s['timestamp'])
        leg_high = leg_high_swing['price']
        leg_low = min(b['low'] for b in bars if b['ts'] >= leg_high_swing['timestamp'])
        if leg_high <= leg_low:
            return []
        fib_618 = leg_low + 0.618 * (leg_high - leg_low)
        fib_786 = leg_low + 0.786 * (leg_high - leg_low)
        ote_low, ote_high = fib_618, fib_786

    last_bar = bars[-1]
    active = last_bar['low'] <= ote_high and last_bar['high'] >= ote_low

    confluence = 'low'
    for f in fvgs:
        if f['type'] != direction or f['status'] != 'FRESH':
            continue
        if f['zone_low'] <= ote_high and f['zone_high'] >= ote_low:
            confluence = 'high'
            break
    if confluence != 'high' and latest['significance'] == 'choch':
        confluence = 'medium'

    return [{
        'direction': direction, 'fib_618': fib_618, 'fib_786': fib_786,
        'ote_zone_high': ote_high, 'ote_zone_low': ote_low,
        'confluence_level': confluence, 'active': active, 'timeframe': timeframe,
    }]


# ─────────────────────────────────────────────────────────────────────────
#  1.7 — SESSION STRUCTURE
# ─────────────────────────────────────────────────────────────────────────

_KILLZONES = [
    ('Asian',    1200, 1440),   # 20:00-24:00 UTC
    ('Asian',       0,  240),   # 00:00-04:00 UTC (continuation, wraps midnight)
    ('London Open', 420,  600), # 07:00-10:00 UTC
    ('NY Open',     810,  960), # 13:30-16:00 UTC
    ('NY Close',   1140, 1200), # 19:00-20:00 UTC
]


def classify_killzone(utc_hour: int, utc_minute: int) -> dict:
    now_min = utc_hour * 60 + utc_minute
    for name, start, end in _KILLZONES:
        if start <= now_min < end:
            return {'name': name, 'active': True, 'minutes_remaining': end - now_min}
    return {'name': 'Between killzones', 'active': False, 'minutes_remaining': 0}


def detect_power_of_three(bars: list, session_start_bar: int) -> dict:
    """AMD phase classifier. Splits the session-so-far into an early third
    (used to define the initial range) and everything since. A sweep
    beyond the early range followed by a close back on the other side of
    that range = manipulation-then-reversal (distribution once the reversal
    is decisive; manipulation while still unfolding). A sweep with no
    reversal is read as a genuine breakout (distribution in the sweep's own
    direction, lower confidence since it may just be a trend day rather
    than a true AMD reversal). Sweeps in both directions, or too little
    data, resolve to 'unclear' — the conservative default."""
    session_bars = bars[session_start_bar:]
    if len(session_bars) < 6:
        return {'phase': 'unclear', 'direction_bias': 'neutral', 'confidence': 0.3}

    third = max(1, len(session_bars) // 3)
    early = session_bars[:third]
    rest = session_bars[third:]
    early_high = max(b['high'] for b in early)
    early_low = min(b['low'] for b in early)

    sweep_up = any(b['high'] > early_high for b in rest)
    sweep_down = any(b['low'] < early_low for b in rest)
    current_price = session_bars[-1]['close']

    if sweep_up and sweep_down:
        return {'phase': 'unclear', 'direction_bias': 'neutral', 'confidence': 0.3}

    if not sweep_up and not sweep_down:
        return {'phase': 'accumulation', 'direction_bias': 'neutral', 'confidence': 0.6}

    if sweep_up:
        if current_price < early_low:
            return {'phase': 'distribution', 'direction_bias': 'bearish', 'confidence': 0.8}
        if current_price < early_high:
            return {'phase': 'manipulation', 'direction_bias': 'bearish', 'confidence': 0.5}
        return {'phase': 'distribution', 'direction_bias': 'bullish', 'confidence': 0.5}

    # sweep_down
    if current_price > early_high:
        return {'phase': 'distribution', 'direction_bias': 'bullish', 'confidence': 0.8}
    if current_price > early_low:
        return {'phase': 'manipulation', 'direction_bias': 'bullish', 'confidence': 0.5}
    return {'phase': 'distribution', 'direction_bias': 'bearish', 'confidence': 0.5}


def detect_opening_range(bars: list, session_open_bar: int, minutes: int = 15) -> dict:
    """OR window = [session open ts, session open ts + minutes*60). Break
    direction requires the last 3 closes since the OR window ended to all
    sit on the same side of the range — a single wick or single close
    beyond ORH/ORL is a touch, not a confirmed "break and hold"."""
    if session_open_bar >= len(bars):
        return {'orh': None, 'orl': None, 'current_position': 'inside', 'break_direction': None}
    open_ts = bars[session_open_bar]['ts']
    window = [b for b in bars if open_ts <= b['ts'] < open_ts + minutes * 60]
    if not window:
        return {'orh': None, 'orl': None, 'current_position': 'inside', 'break_direction': None}
    orh = max(b['high'] for b in window)
    orl = min(b['low'] for b in window)

    last = bars[-1]
    if last['close'] > orh:
        position = 'above'
    elif last['close'] < orl:
        position = 'below'
    else:
        position = 'inside'

    after_window = [b for b in bars if b['ts'] >= open_ts + minutes * 60]
    break_direction = None
    if len(after_window) >= 3:
        last3 = after_window[-3:]
        if all(b['close'] > orh for b in last3):
            break_direction = 'bullish'
        elif all(b['close'] < orl for b in last3):
            break_direction = 'bearish'

    return {'orh': orh, 'orl': orl, 'current_position': position, 'break_direction': break_direction}


# ─────────────────────────────────────────────────────────────────────────
#  1.8 — MULTI-TIMEFRAME CONFLUENCE SCORING
# ─────────────────────────────────────────────────────────────────────────

# NOTE ON THE +3-EACH READING: summing every bullet at face value ("+3
# each" = +3 for 4H alignment AND +3 for 1H alignment, independently) gives
# a 30-point ceiling, not the 27 the brief states as the total range. The
# only reading that reconciles to exactly 27 is a single +3 bonus awarded
# once when 4H and 1H structure agree with each other (not two independent
# +3s) — "each" describing that *both* timeframes must individually confirm
# the same direction for the single bonus to apply. Implemented that way;
# every other line below sums to 27 unmodified.
MAX_SCORE = 27
HIGH_THRESHOLD = 15
VERY_HIGH_THRESHOLD = 21


def score_ict_setup(structure_1m: dict, structure_5m: dict, structure_15m: dict,
                     structure_1h: dict, structure_4h: dict, active_setups: dict) -> dict:
    """active_setups contract (all optional, missing keys treated as falsy):
      ote_zones:  list of detect_ote_zones() rows
      fvgs:       list of detect_fvgs() rows
      obs:        list of detect_order_blocks() rows
      sweeps:     list of detect_liquidity_sweep() rows already filtered by
                  the caller to "within the last 3 bars"
      choch_5m:   detect_choch() row for the 5M timeframe, or None
      choch_15m:  detect_choch() row for the 15M timeframe, or None
      power_of_three: detect_power_of_three() row
      killzone:   classify_killzone() row
      inducement_swept: bool
      htf_bias:   'bullish' | 'bearish' | None — the direction being traded,
                  used to check CHoCH alignment
    """
    score = 0
    components = {}

    s4h = (structure_4h or {}).get('structure')
    s1h = (structure_1h or {}).get('structure')
    htf_agree = s4h is not None and s4h == s1h and s4h != 'RANGING'
    if htf_agree:
        score += 3
    components['htf_alignment'] = 3 if htf_agree else 0

    ote_zones = active_setups.get('ote_zones') or []
    ote_reached = any(z.get('active') for z in ote_zones)
    if ote_reached:
        score += 4
    components['ote_reached'] = 4 if ote_reached else 0

    ote_dir = None
    for z in ote_zones:
        if z.get('active'):
            ote_dir = z.get('direction')
            break
    if ote_dir is None and ote_zones:
        ote_dir = ote_zones[0].get('direction')

    fvgs = active_setups.get('fvgs') or []
    fresh_fvg_in_dir = any(f.get('status') == 'FRESH' and f.get('type') == ote_dir for f in fvgs) if ote_dir else False
    if fresh_fvg_in_dir:
        score += 3
    components['fresh_fvg_in_ote_dir'] = 3 if fresh_fvg_in_dir else 0

    obs = active_setups.get('obs') or []
    ob_tested_in_dir = any(o.get('status') in ('ACTIVE', 'TESTED') and o.get('type') == ote_dir for o in obs) if ote_dir else False
    if ob_tested_in_dir:
        score += 3
    components['ob_tested_in_ote_dir'] = 3 if ob_tested_in_dir else 0

    sweeps = active_setups.get('sweeps') or []
    if sweeps:
        score += 4
    components['recent_sweep'] = 4 if sweeps else 0

    htf_bias = active_setups.get('htf_bias')
    choch_5m = active_setups.get('choch_5m')
    choch_15m = active_setups.get('choch_15m')

    def _choch_matches(c):
        return bool(c) and htf_bias is not None and c.get('type') == htf_bias

    choch_aligned = _choch_matches(choch_5m) or _choch_matches(choch_15m)
    if choch_aligned:
        score += 3
    components['choch_aligned'] = 3 if choch_aligned else 0

    po3 = active_setups.get('power_of_three') or {}
    po3_manip = po3.get('phase') == 'manipulation'
    if po3_manip:
        score += 2
    components['power_of_three_manipulation'] = 2 if po3_manip else 0

    kz = active_setups.get('killzone') or {}
    ny_active = kz.get('active') and kz.get('name') in ('NY Open', 'NY Close')
    london_active = kz.get('active') and kz.get('name') == 'London Open'
    if ny_active:
        score += 2
    components['ny_killzone'] = 2 if ny_active else 0
    if london_active:
        score += 1
    components['london_killzone'] = 1 if london_active else 0

    inducement_swept = bool(active_setups.get('inducement_swept'))
    if inducement_swept:
        score += 2
    components['inducement_swept'] = 2 if inducement_swept else 0

    if score >= VERY_HIGH_THRESHOLD:
        tier = 'VERY HIGH'
    elif score >= HIGH_THRESHOLD:
        tier = 'HIGH'
    else:
        tier = None

    return {'score': score, 'max_score': MAX_SCORE, 'tier': tier, 'components': components}
