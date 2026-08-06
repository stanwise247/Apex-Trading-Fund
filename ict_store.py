"""
ict_store.py — ICT engine database layer + live orchestration
=================================================================
Owns every ict_* table (creation, upsert, read) and the scheduler-facing
orchestration that runs the pure detectors in ict_engine.py against real
OHLCV data and writes the results. ict_engine.py itself never touches the
database — this module is the only place that does.

Table convention follows the codebase's existing pattern for newer feature
tables (daily_levels, week_ahead_reports): hardcoded PostgreSQL DDL
(SERIAL PRIMARY KEY, TIMESTAMPTZ), ensured lazily on first use, verified
against production only — matches this module's own operational reality
(MNQ/ES history and regime data only exist in Railway Postgres).
"""

import os
import time
import logging
from datetime import datetime, timezone, timedelta

import db as _db
import ict_engine as ict
import meridian_mri as _mri  # reuse call_anthropic / ANTHROPIC_MODEL — no circular import

logger = logging.getLogger('APEX.ICT')

SYMBOLS = ('ES', 'MNQ')
TIMEFRAMES = ('1min', '5min', '15min', '1hour', '4hour')
TF_SECONDS = {'1min': 60, '5min': 300, '15min': 900, '1hour': 3600, '4hour': 14400}
LOOKBACK = {'1min': 200, '5min': 200, '15min': 100, '1hour': 50, '4hour': 30}
INTRADAY_TFS = ('1min', '5min', '15min')

RETENTION_DAYS_INTRADAY = 5
RETENTION_DAYS_HTF = 30
ALERT_COOLDOWN_SECONDS = 1800   # 30 min, per symbol
SESSION_START_HOUR = 13         # NY session open, UTC — shared with meridian_mri's convention

_tables_ensured = False
_STATE = {'narrative': {'text': None, 'updated_at': 0}}


# ─────────────────────────────────────────────────────────────────────────
#  TABLES
# ─────────────────────────────────────────────────────────────────────────

def _ensure_ict_tables():
    global _tables_ensured
    if _tables_ensured:
        return
    conn = _db.connect()
    try:
        conn.execute('''CREATE TABLE IF NOT EXISTS ict_swings (
            id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
            swing_type TEXT NOT NULL, price REAL, timestamp BIGINT NOT NULL,
            confirmed BOOLEAN DEFAULT TRUE, created_at TIMESTAMPTZ NOT NULL,
            UNIQUE(symbol, timeframe, swing_type, timestamp)
        )''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_ict_swings_sym_tf ON ict_swings (symbol, timeframe)')

        conn.execute('''CREATE TABLE IF NOT EXISTS ict_market_structure (
            id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
            structure TEXT, last_hh REAL, last_hl REAL, last_lh REAL, last_ll REAL,
            updated_at TIMESTAMPTZ NOT NULL,
            UNIQUE(symbol, timeframe)
        )''')

        conn.execute('''CREATE TABLE IF NOT EXISTS ict_fvgs (
            id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
            type TEXT NOT NULL, zone_high REAL, zone_low REAL, formed_at BIGINT NOT NULL,
            status TEXT, mitigation_pct REAL, updated_at TIMESTAMPTZ NOT NULL,
            UNIQUE(symbol, timeframe, type, formed_at)
        )''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_ict_fvgs_sym_tf ON ict_fvgs (symbol, timeframe)')

        conn.execute('''CREATE TABLE IF NOT EXISTS ict_order_blocks (
            id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
            type TEXT NOT NULL, zone_high REAL, zone_low REAL, formed_at BIGINT NOT NULL,
            status TEXT, impulse_size REAL, updated_at TIMESTAMPTZ NOT NULL,
            UNIQUE(symbol, timeframe, type, formed_at)
        )''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_ict_obs_sym_tf ON ict_order_blocks (symbol, timeframe)')

        conn.execute('''CREATE TABLE IF NOT EXISTS ict_equal_levels (
            id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
            type TEXT NOT NULL, price_avg REAL, price_range REAL, swing_count INTEGER,
            formed_at BIGINT NOT NULL, status TEXT, updated_at TIMESTAMPTZ NOT NULL,
            UNIQUE(symbol, timeframe, type, formed_at)
        )''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_ict_eql_sym_tf ON ict_equal_levels (symbol, timeframe)')

        conn.execute('''CREATE TABLE IF NOT EXISTS ict_sweeps (
            id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
            type TEXT NOT NULL, level REAL, timestamp BIGINT NOT NULL, close_back_bar BIGINT,
            created_at TIMESTAMPTZ NOT NULL,
            UNIQUE(symbol, timeframe, type, timestamp)
        )''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_ict_sweeps_sym_tf ON ict_sweeps (symbol, timeframe)')

        conn.execute('''CREATE TABLE IF NOT EXISTS ict_ote_zones (
            id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
            direction TEXT, fib_618 REAL, fib_786 REAL, zone_high REAL, zone_low REAL,
            confluence_level TEXT, active BOOLEAN, formed_at BIGINT, updated_at TIMESTAMPTZ NOT NULL,
            UNIQUE(symbol, timeframe)
        )''')

        # price_at_alert / outcome_valid / outcome_checked_at extend the brief's
        # own literal column list — needed for Part 5's "valid setup in
        # hindsight, auto-updated after 4H" alert history requirement.
        conn.execute('''CREATE TABLE IF NOT EXISTS ict_alerts (
            id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, timeframe TEXT,
            alert_type TEXT, description TEXT, score INTEGER,
            timestamp TIMESTAMPTZ NOT NULL, sent_telegram BOOLEAN DEFAULT FALSE,
            price_at_alert REAL, outcome_valid BOOLEAN, outcome_checked_at TIMESTAMPTZ
        )''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_ict_alerts_sym ON ict_alerts (symbol, timestamp)')

        conn.commit()
        _tables_ensured = True
    except Exception as e:
        logger.debug(f'ICT tables ensure failed (may already exist): {e}')
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────
#  BAR FETCH
# ─────────────────────────────────────────────────────────────────────────

def fetch_ict_bars(symbol: str, timeframe: str, limit: int) -> list:
    """Ascending list-of-dicts, matching ict_engine.py's bar contract. The
    most recent row is dropped whenever its own timeframe period hasn't
    fully elapsed yet — the live feed keeps rewriting that row's OHLC as
    new trades arrive, so it is never a closed candle."""
    conn = _db.connect()
    try:
        cur = conn.execute(
            'SELECT ts, open, high, low, close, volume FROM ohlcv '
            'WHERE symbol=? AND timeframe=? ORDER BY ts DESC LIMIT ?',
            (symbol, timeframe, limit)
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    rows = list(reversed(rows))
    bars = [{'ts': int(r[0]), 'open': float(r[1]), 'high': float(r[2]),
              'low': float(r[3]), 'close': float(r[4]), 'volume': float(r[5] or 0)} for r in rows]
    if bars:
        tf_sec = TF_SECONDS.get(timeframe, 60)
        if bars[-1]['ts'] + tf_sec > int(time.time()):
            bars = bars[:-1]
    return bars


def _session_start_bar_index(bars: list) -> int:
    """Index of the first bar at/after today's 13:00 UTC session open."""
    if not bars:
        return 0
    now = datetime.now(timezone.utc)
    session_ts = int(datetime(now.year, now.month, now.day, SESSION_START_HOUR, tzinfo=timezone.utc).timestamp())
    for i, b in enumerate(bars):
        if b['ts'] >= session_ts:
            return i
    return max(0, len(bars) - 1)


# ─────────────────────────────────────────────────────────────────────────
#  WRITE-BACK (upsert, not insert — never duplicate)
# ─────────────────────────────────────────────────────────────────────────

def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _write_swings(conn, symbol, timeframe, swings):
    for s in swings:
        conn.execute(
            'INSERT INTO ict_swings (symbol,timeframe,swing_type,price,timestamp,confirmed,created_at) '
            'VALUES (?,?,?,?,?,?,?) '
            'ON CONFLICT (symbol,timeframe,swing_type,timestamp) DO NOTHING',
            (symbol, timeframe, s['type'], s['price'], s['timestamp'], s['confirmed'], _now_iso())
        )


def _write_market_structure(conn, symbol, timeframe, ms):
    conn.execute(
        'INSERT INTO ict_market_structure (symbol,timeframe,structure,last_hh,last_hl,last_lh,last_ll,updated_at) '
        'VALUES (?,?,?,?,?,?,?,?) '
        'ON CONFLICT (symbol,timeframe) DO UPDATE SET '
        'structure=EXCLUDED.structure, last_hh=EXCLUDED.last_hh, last_hl=EXCLUDED.last_hl, '
        'last_lh=EXCLUDED.last_lh, last_ll=EXCLUDED.last_ll, updated_at=EXCLUDED.updated_at',
        (symbol, timeframe, ms['structure'], ms['last_hh'], ms['last_hl'], ms['last_lh'], ms['last_ll'], _now_iso())
    )


def _write_fvgs(conn, symbol, timeframe, fvgs):
    for f in fvgs:
        conn.execute(
            'INSERT INTO ict_fvgs (symbol,timeframe,type,zone_high,zone_low,formed_at,status,mitigation_pct,updated_at) '
            'VALUES (?,?,?,?,?,?,?,?,?) '
            'ON CONFLICT (symbol,timeframe,type,formed_at) DO UPDATE SET '
            'status=EXCLUDED.status, mitigation_pct=EXCLUDED.mitigation_pct, updated_at=EXCLUDED.updated_at',
            (symbol, timeframe, f['type'], f['zone_high'], f['zone_low'], f['formed_at'],
             f['status'], f['mitigation_pct'], _now_iso())
        )


def _write_obs(conn, symbol, timeframe, obs):
    for o in obs:
        conn.execute(
            'INSERT INTO ict_order_blocks (symbol,timeframe,type,zone_high,zone_low,formed_at,status,impulse_size,updated_at) '
            'VALUES (?,?,?,?,?,?,?,?,?) '
            'ON CONFLICT (symbol,timeframe,type,formed_at) DO UPDATE SET '
            'status=EXCLUDED.status, updated_at=EXCLUDED.updated_at',
            (symbol, timeframe, o['type'], o['zone_high'], o['zone_low'], o['formed_at'],
             o['status'], o['impulse_size'], _now_iso())
        )


def _write_eqls(conn, symbol, timeframe, eqls):
    for e in eqls:
        conn.execute(
            'INSERT INTO ict_equal_levels (symbol,timeframe,type,price_avg,price_range,swing_count,formed_at,status,updated_at) '
            'VALUES (?,?,?,?,?,?,?,?,?) '
            'ON CONFLICT (symbol,timeframe,type,formed_at) DO UPDATE SET '
            'status=EXCLUDED.status, price_avg=EXCLUDED.price_avg, updated_at=EXCLUDED.updated_at',
            (symbol, timeframe, e['type'], e['price_avg'], e['price_range'], e['swing_count'],
             e['formed_at'], e['status'], _now_iso())
        )


def _write_sweeps(conn, symbol, timeframe, sweeps):
    for s in sweeps:
        conn.execute(
            'INSERT INTO ict_sweeps (symbol,timeframe,type,level,timestamp,close_back_bar,created_at) '
            'VALUES (?,?,?,?,?,?,?) '
            'ON CONFLICT (symbol,timeframe,type,timestamp) DO NOTHING',
            (symbol, timeframe, s['type'], s['level'], s['timestamp'], s['close_back_inside_bar'], _now_iso())
        )


def _write_ote(conn, symbol, timeframe, ote_row):
    conn.execute(
        'INSERT INTO ict_ote_zones (symbol,timeframe,direction,fib_618,fib_786,zone_high,zone_low,confluence_level,active,formed_at,updated_at) '
        'VALUES (?,?,?,?,?,?,?,?,?,?,?) '
        'ON CONFLICT (symbol,timeframe) DO UPDATE SET '
        'direction=EXCLUDED.direction, fib_618=EXCLUDED.fib_618, fib_786=EXCLUDED.fib_786, '
        'zone_high=EXCLUDED.zone_high, zone_low=EXCLUDED.zone_low, confluence_level=EXCLUDED.confluence_level, '
        'active=EXCLUDED.active, formed_at=EXCLUDED.formed_at, updated_at=EXCLUDED.updated_at',
        (symbol, timeframe, ote_row['direction'], ote_row['fib_618'], ote_row['fib_786'],
         ote_row['ote_zone_high'], ote_row['ote_zone_low'], ote_row['confluence_level'],
         ote_row['active'], ote_row.get('formed_at'), _now_iso())
    )


def _insert_alert(conn, symbol, timeframe, alert_type, description, score, price_at_alert):
    conn.execute(
        'INSERT INTO ict_alerts (symbol,timeframe,alert_type,description,score,timestamp,sent_telegram,price_at_alert) '
        'VALUES (?,?,?,?,?,?,?,?)',
        (symbol, timeframe, alert_type, description, score, _now_iso(), True, price_at_alert)
    )


# ─────────────────────────────────────────────────────────────────────────
#  PER-SYMBOL / PER-TIMEFRAME ANALYSIS PASS
# ─────────────────────────────────────────────────────────────────────────

def analyse_symbol_timeframe(symbol: str, timeframe: str) -> dict:
    """Runs every detector for one (symbol, timeframe) and returns the raw
    results — does not write to the database. Used both by the scheduler
    (which writes the results) and directly by the API/tests (which don't
    need a DB round-trip to see current state)."""
    bars = fetch_ict_bars(symbol, timeframe, LOOKBACK.get(timeframe, 100))
    if len(bars) < 10:
        return {'symbol': symbol, 'timeframe': timeframe, 'bars': bars, 'insufficient_data': True}

    swings = ict.detect_swings(bars, timeframe)
    ms = ict.classify_market_structure(swings)
    bos = ict.detect_bos(swings, bars)
    choch = ict.detect_choch(swings, bars)
    fvgs = ict.detect_fvgs(bars)
    obs = ict.detect_order_blocks(bars, swings)
    eqls_raw = ict.detect_equal_highs_lows(swings)
    eqls = []
    for lvl in eqls_raw:
        lvl = dict(lvl)
        lvl['status'] = ict.resolve_eql_status(lvl, bars)
        eqls.append(lvl)
    sweeps = ict.detect_liquidity_sweep(bars, eqls_raw)
    inducement = ict.detect_inducement(bars, swings)
    ote = ict.detect_ote_zones(swings, fvgs, bars)

    session_idx = _session_start_bar_index(bars)
    po3 = ict.detect_power_of_three(bars, session_idx) if timeframe in ('1min', '5min') else None
    orr = ict.detect_opening_range(bars, session_idx) if timeframe in ('1min', '5min') else None

    return {
        'symbol': symbol, 'timeframe': timeframe, 'bars': bars,
        'swings': swings, 'structure': ms, 'bos': bos, 'choch': choch,
        'fvgs': fvgs, 'obs': obs, 'equal_levels': eqls, 'sweeps': sweeps,
        'inducement': inducement, 'ote_zones': ote,
        'power_of_three': po3, 'opening_range': orr,
        'insufficient_data': False,
    }


def _write_analysis(conn, result: dict):
    if result.get('insufficient_data'):
        return
    symbol, timeframe = result['symbol'], result['timeframe']
    _write_swings(conn, symbol, timeframe, result['swings'])
    _write_market_structure(conn, symbol, timeframe, result['structure'])
    _write_fvgs(conn, symbol, timeframe, result['fvgs'])
    _write_obs(conn, symbol, timeframe, result['obs'])
    _write_eqls(conn, symbol, timeframe, result['equal_levels'])
    _write_sweeps(conn, symbol, timeframe, result['sweeps'])
    if result['ote_zones']:
        row = dict(result['ote_zones'][0])
        _write_ote(conn, symbol, timeframe, row)


# ─────────────────────────────────────────────────────────────────────────
#  KILLZONE / SCORING GLUE
# ─────────────────────────────────────────────────────────────────────────

def current_killzone() -> dict:
    now = datetime.now(timezone.utc)
    return ict.classify_killzone(now.hour, now.minute)


def _htf_bias_for(symbol: str) -> str:
    """'bullish'/'bearish'/None — reuses Meridian's own live HTF-bias
    fetcher rather than re-deriving direction from scratch, keeping ICT's
    "direction of HTF bias" consistent with what the rest of the dashboard
    already shows for this symbol."""
    try:
        b = _mri.fetch_htf_bias(symbol)
        return {'BULLISH': 'bullish', 'BEARISH': 'bearish'}.get(b)
    except Exception:
        return None


def score_symbol(symbol: str, per_tf: dict) -> dict:
    """per_tf: {timeframe: analyse_symbol_timeframe() result}"""
    def struct(tf):
        r = per_tf.get(tf)
        return r['structure'] if r and not r.get('insufficient_data') else {'structure': None}

    r5m = per_tf.get('5min') or {}
    r15m = per_tf.get('15min') or {}

    all_fvgs, all_obs, all_ote, recent_sweeps = [], [], [], []
    for tf, r in per_tf.items():
        if r.get('insufficient_data'):
            continue
        all_fvgs.extend(r.get('fvgs', []))
        all_obs.extend(r.get('obs', []))
        all_ote.extend(r.get('ote_zones', []))
        bars = r.get('bars') or []
        cutoff_ts = bars[-3]['ts'] if len(bars) >= 3 else 0
        recent_sweeps.extend(s for s in r.get('sweeps', []) if s['timestamp'] >= cutoff_ts)

    # Inducement swept = a recent sweep's level lines up with (within 0.05%)
    # a detected inducement point on the same timeframe — i.e. the minor
    # false-extreme that ICT says gets taken out before the real move was
    # actually the level that just got swept, not just "a sweep happened."
    inducement_swept = False
    for tf, r in per_tf.items():
        if r.get('insufficient_data'):
            continue
        for ind in r.get('inducement', []):
            for sw in r.get('sweeps', []):
                if ind['price'] and abs(sw['level'] - ind['price']) / ind['price'] * 100 <= 0.05:
                    inducement_swept = True
                    break
            if inducement_swept:
                break
        if inducement_swept:
            break

    active_setups = {
        'ote_zones': all_ote, 'fvgs': all_fvgs, 'obs': all_obs, 'sweeps': recent_sweeps,
        'choch_5m': (r5m.get('choch') or [None])[0], 'choch_15m': (r15m.get('choch') or [None])[0],
        'power_of_three': r5m.get('power_of_three') or {},
        'killzone': current_killzone(),
        'inducement_swept': inducement_swept,
        'htf_bias': _htf_bias_for(symbol),
    }
    result = ict.score_ict_setup(
        struct('1min'), struct('5min'), struct('15min'), struct('1hour'), struct('4hour'),
        active_setups
    )
    result['active_setups'] = active_setups
    return result


# ─────────────────────────────────────────────────────────────────────────
#  ALERTS
# ─────────────────────────────────────────────────────────────────────────

def _describe_setup(score_result: dict) -> str:
    c = score_result['components']
    parts = []
    if c.get('recent_sweep'):
        parts.append('Liquidity Sweep')
    if c.get('ote_reached'):
        parts.append('OTE')
    if c.get('choch_aligned'):
        parts.append('CHoCH')
    if not parts:
        parts.append('HTF Confluence')
    ote = score_result['active_setups'].get('ote_zones') or []
    direction = ote[0]['direction'] if ote else score_result['active_setups'].get('htf_bias')
    side = 'Long' if direction == 'bullish' else ('Short' if direction == 'bearish' else '')
    return ' + '.join(parts) + (f' {side}' if side else '')


def _maybe_alert(conn, symbol: str, score_result: dict, per_tf: dict):
    score = score_result['score']
    if score < ict.HIGH_THRESHOLD:
        return
    last = getattr(_maybe_alert, '_last_alert_ts', {})
    now = time.time()
    if symbol in last and now - last[symbol] < ALERT_COOLDOWN_SECONDS:
        return

    setup_desc = _describe_setup(score_result)
    r5m = per_tf.get('5min') or {}
    price = None
    if r5m.get('bars'):
        price = r5m['bars'][-1]['close']

    _insert_alert(conn, symbol, '5min', setup_desc, setup_desc, score, price)

    try:
        from live_scanner import send_telegram
        tier = score_result['tier'] or 'HIGH'
        s4h = (per_tf.get('4hour') or {}).get('structure', {}).get('structure', '—')
        s1h = (per_tf.get('1hour') or {}).get('structure', {}).get('structure', '—')
        s5m = (per_tf.get('5min') or {}).get('structure', {}).get('structure', '—')
        kz = score_result['active_setups']['killzone']
        kz_txt = kz['name'] if kz['active'] else 'Between killzones'
        htf_bias = score_result['active_setups'].get('htf_bias') or 'unknown'

        fvgs = score_result['active_setups'].get('fvgs') or []
        fresh_fvgs = [f for f in fvgs if f['status'] == 'FRESH']
        fvg_txt = f"{fresh_fvgs[0]['type']} {fresh_fvgs[0]['status']} [{fresh_fvgs[0]['zone_low']:.2f}-{fresh_fvgs[0]['zone_high']:.2f}]" if fresh_fvgs else 'none nearby'

        obs = score_result['active_setups'].get('obs') or []
        active_obs = [o for o in obs if o['status'] in ('ACTIVE', 'TESTED')]
        ob_txt = f"{active_obs[0]['type']} {active_obs[0]['status']} [{active_obs[0]['zone_low']:.2f}-{active_obs[0]['zone_high']:.2f}]" if active_obs else 'none nearby'

        eql = (r5m.get('equal_levels') or [])
        eql_txt = f"{eql[0]['type']} @ {eql[0]['price_avg']:.2f} ({eql[0]['status']})" if eql else 'none nearby'

        key_level = ''
        ote = score_result['active_setups'].get('ote_zones') or []
        if ote:
            key_level = f"OTE {ote[0]['ote_zone_low']:.2f}-{ote[0]['ote_zone_high']:.2f}"

        msg = (
            f'🎯 <b>ICT SETUP ALERT — {symbol}</b>\n'
            f'━━━━━━━━━━━━━━━━━━\n'
            f'Score: {score}/27 ({tier})\n'
            f'Structure: {s4h} → {s1h} → {s5m}\n'
            f'Setup: {setup_desc}\n'
            f'Key Level: {key_level or "—"}\n'
            f'Killzone: {kz_txt}\n'
            f'━━━━━━━━━━━━━━━━━━\n'
            f'HTF Bias: {htf_bias}\n'
            f'FVG: {fvg_txt}\n'
            f'OB: {ob_txt}\n'
            f'EQL: {eql_txt}'
        )
        send_telegram(msg, message_type='ict_alert')
    except Exception as e:
        logger.warning(f'ICT alert Telegram send failed: {e}')

    last[symbol] = now
    _maybe_alert._last_alert_ts = last


# ─────────────────────────────────────────────────────────────────────────
#  RETENTION
# ─────────────────────────────────────────────────────────────────────────

def prune_ict_tables():
    _ensure_ict_tables()
    now = int(time.time())
    intraday_cutoff = now - RETENTION_DAYS_INTRADAY * 86400
    htf_cutoff = now - RETENTION_DAYS_HTF * 86400
    conn = _db.connect()
    try:
        for table, ts_col in (('ict_swings', 'timestamp'), ('ict_fvgs', 'formed_at'),
                               ('ict_order_blocks', 'formed_at'), ('ict_equal_levels', 'formed_at'),
                               ('ict_sweeps', 'timestamp')):
            conn.execute(f"DELETE FROM {table} WHERE timeframe IN ('1min','5min','15min') AND {ts_col} < ?", (intraday_cutoff,))
            conn.execute(f"DELETE FROM {table} WHERE timeframe IN ('1hour','4hour') AND {ts_col} < ?", (htf_cutoff,))
        conn.commit()
    except Exception as e:
        logger.warning(f'ICT prune error: {e}')
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────
#  MAIN ORCHESTRATOR — called by background_scheduler every 60s
# ─────────────────────────────────────────────────────────────────────────

def run_ict_analysis() -> dict:
    _ensure_ict_tables()
    report = {}
    conn = _db.connect()
    try:
        for symbol in SYMBOLS:
            per_tf = {}
            for tf in TIMEFRAMES:
                try:
                    result = analyse_symbol_timeframe(symbol, tf)
                    per_tf[tf] = result
                    _write_analysis(conn, result)
                except Exception as e:
                    logger.warning(f'ICT analysis {symbol}/{tf} error: {e}')
                    per_tf[tf] = {'insufficient_data': True}
            conn.commit()

            try:
                score_result = score_symbol(symbol, per_tf)
                report[symbol] = {'score': score_result['score'], 'tier': score_result['tier']}
                _maybe_alert(conn, symbol, score_result, per_tf)
                conn.commit()
            except Exception as e:
                logger.warning(f'ICT scoring {symbol} error: {e}')
    finally:
        conn.close()

    if not hasattr(run_ict_analysis, '_last_prune') or time.time() - run_ict_analysis._last_prune >= 3600:
        run_ict_analysis._last_prune = time.time()
        try:
            prune_ict_tables()
        except Exception as e:
            logger.warning(f'ICT prune error: {e}')

    return report


def resolve_ict_alerts():
    """Hindsight validation: 4h after an alert, check whether price moved
    in the alert's implied direction. Mirrors meridian_direction.py's
    resolve_predictions() pattern already established in this codebase."""
    _ensure_ict_tables()
    conn = _db.connect()
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat()
        rows = conn.execute(
            'SELECT id, symbol, description, price_at_alert, timestamp FROM ict_alerts '
            'WHERE outcome_checked_at IS NULL AND timestamp <= ? AND price_at_alert IS NOT NULL',
            (cutoff,)
        ).fetchall()
        for row in rows:
            alert_id, symbol, description, price_at_alert, ts = row
            bars = fetch_ict_bars(symbol, '5min', 5)
            if not bars:
                continue
            current_price = bars[-1]['close']
            direction = 'bullish' if 'Long' in (description or '') else ('bearish' if 'Short' in (description or '') else None)
            if direction == 'bullish':
                valid = current_price > price_at_alert
            elif direction == 'bearish':
                valid = current_price < price_at_alert
            else:
                valid = None
            conn.execute(
                'UPDATE ict_alerts SET outcome_valid=?, outcome_checked_at=? WHERE id=?',
                (valid, _now_iso(), alert_id)
            )
        conn.commit()
    except Exception as e:
        logger.debug(f'resolve_ict_alerts error: {e}')
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────
#  READ HELPERS FOR API ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────

def get_ict_alerts(symbol: str, limit: int = 20) -> list:
    _ensure_ict_tables()
    conn = _db.connect()
    try:
        rows = conn.execute(
            'SELECT symbol, timeframe, alert_type, description, score, timestamp, sent_telegram, outcome_valid '
            'FROM ict_alerts WHERE symbol=? ORDER BY timestamp DESC LIMIT ?',
            (symbol, limit)
        ).fetchall()
    finally:
        conn.close()
    return [{'symbol': r[0], 'timeframe': r[1], 'alert_type': r[2], 'description': r[3],
             'score': r[4], 'timestamp': str(r[5]), 'sent_telegram': bool(r[6]), 'outcome_valid': r[7]}
            for r in rows]


def get_ict_levels(symbol: str, timeframe: str) -> dict:
    result = analyse_symbol_timeframe(symbol, timeframe)
    if result.get('insufficient_data'):
        return {'ok': False, 'error': 'insufficient_data', 'symbol': symbol, 'timeframe': timeframe}
    return {
        'ok': True, 'symbol': symbol, 'timeframe': timeframe,
        'swings': result['swings'], 'structure': result['structure'],
        'bos': result['bos'], 'choch': result['choch'],
        'fvgs': result['fvgs'], 'obs': result['obs'],
        'equal_levels': result['equal_levels'], 'sweeps': result['sweeps'],
        'inducement': result['inducement'], 'ote_zones': result['ote_zones'],
    }


def _generate_narrative(symbol_data: dict) -> str:
    now = time.time()
    if _STATE['narrative']['text'] and now - _STATE['narrative']['updated_at'] < 300:
        return _STATE['narrative']['text']
    try:
        prompt = (
            'You are a professional ICT (Inner Circle Trader) market analyst talking through '
            'the chart for a trader who is actively watching and ready to execute. Given this '
            'structured ICT data for ES and MNQ, write a 4-6 sentence analysis covering: the '
            'current structure bias, the highest-probability setup forming and what is driving '
            'its score, where the draw on liquidity is, and what price needs to do to confirm an '
            f'entry. Structured but conversational.\n\nDATA:\n{symbol_data}'
        )
        text = _mri.call_anthropic(prompt, max_tokens=700)
        _STATE['narrative'] = {'text': text, 'updated_at': now}
        return text
    except Exception as e:
        logger.warning(f'ICT narrative generation error: {e}')
        fallback = _STATE['narrative']['text'] or 'ICT narrative unavailable — analysis data is still populating.'
        return fallback


_TF_LABEL = {'1min': '1m', '5min': '5m', '15min': '15m', '1hour': '1h', '4hour': '4h'}


def get_ict_analysis(symbol: str) -> dict:
    per_tf = {tf: analyse_symbol_timeframe(symbol, tf) for tf in TIMEFRAMES}
    score_result = score_symbol(symbol, per_tf)

    structure, swings_by_tf, bos_by_tf, choch_by_tf, ote_by_tf = {}, {}, {}, {}, {}
    all_fvgs, all_obs, all_eql, all_sweeps = [], [], [], []

    for tf, r in per_tf.items():
        label = _TF_LABEL[tf]
        if r.get('insufficient_data'):
            structure[label] = None
            swings_by_tf[label] = []
            bos_by_tf[label] = []
            choch_by_tf[label] = []
            ote_by_tf[label] = []
            continue
        structure[label] = r['structure']['structure']
        swings_by_tf[label] = r['swings']
        bos_by_tf[label] = r['bos']
        choch_by_tf[label] = r['choch']
        ote_by_tf[label] = r['ote_zones']
        all_fvgs.extend(dict(f, timeframe=label) for f in r['fvgs'] if f['status'] != 'VOID')
        all_obs.extend(dict(o, timeframe=label) for o in r['obs'] if o['status'] != 'BREACHED')
        all_eql.extend(dict(e, timeframe=label) for e in r['equal_levels'])
        all_sweeps.extend(dict(s, timeframe=label) for s in r['sweeps'])

    all_sweeps.sort(key=lambda s: s['timestamp'], reverse=True)
    all_sweeps = all_sweeps[:10]

    r5m = per_tf.get('5min') or {}
    narrative_input = {
        'symbol': symbol, 'structure': structure, 'score': score_result['score'],
        'tier': score_result['tier'], 'components': score_result['components'],
        'fvgs': all_fvgs[:5], 'obs': all_obs[:5], 'equal_levels': all_eql[:5],
        'killzone': score_result['active_setups']['killzone'],
        'power_of_three': r5m.get('power_of_three'),
        'ote_zones': score_result['active_setups']['ote_zones'],
    }

    return {
        'symbol': symbol, 'timestamp': _now_iso(),
        'structure': structure,
        'active_fvgs': all_fvgs, 'active_obs': all_obs, 'equal_levels': all_eql,
        'recent_sweeps': all_sweeps, 'ote_zones': score_result['active_setups']['ote_zones'],
        'killzone': score_result['active_setups']['killzone'],
        'power_of_three': r5m.get('power_of_three'),
        'opening_range': r5m.get('opening_range'),
        'setup_score': score_result['score'], 'setup_tier': score_result['tier'],
        'narrative': _generate_narrative(narrative_input),
        # Extend beyond the brief's literal example shape (which omits swing/
        # break data entirely) — Part 5's chart explicitly needs swing
        # triangles and BOS/CHoCH markers, keyed per-timeframe so the
        # frontend's TF switcher doesn't need a fresh fetch on every click.
        'swings_by_tf': swings_by_tf, 'bos_by_tf': bos_by_tf, 'choch_by_tf': choch_by_tf,
        'ote_by_tf': ote_by_tf,
    }
