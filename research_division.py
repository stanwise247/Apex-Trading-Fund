"""
Wise Meridian Capital — Research Division
==========================================
Completely isolated from live trading.
Read-only access to apex_trades.
Writes only to: strategy_health_log, shadow_lab, research_decisions.
"""

import json
import logging
import math
from datetime import datetime, date, timedelta, timezone

import db as _db

logger = logging.getLogger('APEX.Research')

BENCHMARKS = {
    'D': {'sharpe': 10.60, 'wr': 0.61},
    'B': {'sharpe':  7.50, 'wr': 0.52},
    'E': {'sharpe':  5.80, 'wr': 0.49},
    'H': {'sharpe':  6.00, 'wr': 0.53},
    'I': {'sharpe':  8.40, 'wr': 0.67},
    'C': {'sharpe':  9.42, 'wr': 0.55},
    'A': {'sharpe': 12.81, 'wr': 0.60},
}

SETUP_NAMES = {
    'A': 'Sweep + OB',
    'B': 'ChoCh + OB',
    'C': 'BOS + OB',
    'D': 'FVG Fill',
    'E': 'EMA50 Pullback',
    'H': 'VWAP Reversion',
    'I': 'Mathematical Alpha',
}


def _sharpe(pnl_r_list):
    n = len(pnl_r_list)
    if n < 5:
        return None
    mean = sum(pnl_r_list) / n
    var  = sum((v - mean) ** 2 for v in pnl_r_list) / (n - 1) if n > 1 else 0
    std  = math.sqrt(var) if var > 0 else 0
    if std == 0:
        return None
    return (mean / std) * math.sqrt(252)


def _weekly_metrics(setup_id, conn, n_weeks=4):
    """Per-week trade metrics for the last n_weeks ISO weeks, oldest first."""
    today       = date.today()
    monday_this = today - timedelta(days=today.weekday())
    weeks = []
    for i in range(n_weeks - 1, -1, -1):
        w_start = monday_this - timedelta(weeks=i)
        w_end   = w_start + timedelta(days=7)
        rows = conn.execute(
            "SELECT pnl_r FROM apex_trades "
            "WHERE UPPER(SUBSTR(setup, 1, 1)) = ? "
            "  AND status = 'closed' AND pnl_r IS NOT NULL "
            "  AND exit_time >= ? AND exit_time < ?",
            (setup_id, w_start.isoformat(), w_end.isoformat())
        ).fetchall()
        pnl = [r[0] for r in rows]
        wr  = (sum(1 for v in pnl if v > 0) / len(pnl)) if pnl else None
        exp = (sum(pnl) / len(pnl)) if pnl else None
        weeks.append({
            'week_start': w_start,
            'count':      len(pnl),
            'wr':         wr,
            'expectancy': exp,
        })
    return weeks


def _consecutive(bool_list, n=2):
    """True if there are n consecutive True values in bool_list."""
    count = 0
    for v in bool_list:
        if v:
            count += 1
            if count >= n:
                return True
        else:
            count = 0
    return False


def calculate_health_score(setup_id, conn):
    """
    Calculate health metrics for a given setup.
    Returns a dict with all metrics and a health_score (0-100).
    Returns health_score=None and alert_level='INSUFFICIENT_DATA' for < 5 trades.
    """
    sid   = setup_id.upper()
    bench = BENCHMARKS.get(sid)
    if not bench:
        return {'setup_id': sid, 'health_score': None, 'alert_level': 'INSUFFICIENT_DATA'}

    now_utc = datetime.now(timezone.utc)
    cut_30d = (now_utc - timedelta(days=30)).isoformat()
    cut_28d = (now_utc - timedelta(days=28)).isoformat()

    # Last 20 closed trades (no time limit) — for WR and expectancy
    rows_20 = conn.execute(
        "SELECT pnl_r FROM apex_trades "
        "WHERE UPPER(SUBSTR(setup, 1, 1)) = ? "
        "  AND status = 'closed' AND pnl_r IS NOT NULL "
        "ORDER BY exit_time DESC LIMIT 20",
        (sid,)
    ).fetchall()

    if len(rows_20) < 5:
        return {
            'setup_id':          sid,
            'health_score':      None,
            'alert_level':       'INSUFFICIENT_DATA',
            'sharpe_30d':        None,
            'sharpe_benchmark':  bench['sharpe'],
            'win_rate':          None,
            'win_rate_benchmark':bench['wr'],
            'signal_count_week': None,
            'expectancy':        None,
        }

    pnl_20 = [r[0] for r in rows_20]

    # Sharpe on last 30 days of closed trades
    rows_30d = conn.execute(
        "SELECT pnl_r FROM apex_trades "
        "WHERE UPPER(SUBSTR(setup, 1, 1)) = ? "
        "  AND status = 'closed' AND pnl_r IS NOT NULL "
        "  AND exit_time >= ? "
        "ORDER BY exit_time ASC",
        (sid, cut_30d)
    ).fetchall()
    pnl_30d    = [r[0] for r in rows_30d]
    sharpe_30d = _sharpe(pnl_30d)

    # Win rate and expectancy (last 20)
    wins       = sum(1 for v in pnl_20 if v > 0)
    win_rate   = wins / len(pnl_20)
    expectancy = sum(pnl_20) / len(pnl_20)

    # Signal frequency (last 4 weeks)
    count_28d = conn.execute(
        "SELECT COUNT(*) FROM apex_trades "
        "WHERE UPPER(SUBSTR(setup, 1, 1)) = ? "
        "  AND status = 'closed' AND exit_time >= ?",
        (sid, cut_28d)
    ).fetchone()[0]
    signal_count_week = count_28d / 4.0

    # Per-week metrics for consecutive checks
    week_m = _weekly_metrics(sid, conn, n_weeks=4)

    # ── Health score ──────────────────────────────────────────
    score = 100

    # Sharpe vs benchmark
    if sharpe_30d is not None and bench['sharpe'] > 0:
        ratio = sharpe_30d / bench['sharpe']
        if ratio < 0.5:
            score -= 20
        elif ratio > 1.1:
            score += 5

    # WR > 5pp below benchmark for 2+ consecutive weeks
    wr_below = [w['wr'] is not None and w['wr'] < (bench['wr'] - 0.05) for w in week_m]
    if _consecutive(wr_below, n=2):
        score -= 15

    # Signal frequency declining > 30% vs 4-week average
    if len(week_m) >= 4:
        recent = (week_m[-1]['count'] + week_m[-2]['count']) / 2
        older  = (week_m[0]['count']  + week_m[1]['count'])  / 2
        if older > 0 and recent < older * 0.7:
            score -= 10

    # Expectancy negative for 2+ consecutive weeks
    exp_neg = [w['expectancy'] is not None and w['expectancy'] < 0 for w in week_m]
    if _consecutive(exp_neg, n=2):
        score -= 10

    score = max(0, min(100, score))

    alert_level = ('HEALTHY' if score > 75 else 'WATCH' if score >= 50 else 'ALERT')

    return {
        'setup_id':          sid,
        'sharpe_30d':        round(sharpe_30d, 3) if sharpe_30d is not None else None,
        'sharpe_benchmark':  bench['sharpe'],
        'win_rate':          round(win_rate, 4),
        'win_rate_benchmark':bench['wr'],
        'signal_count_week': round(signal_count_week, 2),
        'expectancy':        round(expectancy, 4),
        'health_score':      score,
        'alert_level':       alert_level,
    }


def run_weekly_health_check(conn):
    """Run health check for all live setups and write results to strategy_health_log."""
    setups  = ['A', 'B', 'C', 'D', 'E', 'H', 'I']
    today   = date.today()
    monday  = today - timedelta(days=today.weekday())
    results = {}

    for sid in setups:
        try:
            metrics = calculate_health_score(sid, conn)

            # Detect 3+ consecutive declining health scores in the log
            prev = conn.execute(
                "SELECT health_score FROM strategy_health_log "
                "WHERE setup_id = ? AND health_score IS NOT NULL "
                "ORDER BY week_start DESC LIMIT 3",
                (sid,)
            ).fetchall()
            notes = None
            if len(prev) >= 2:
                scores = [r[0] for r in prev]
                # scores[0] is most recent — declining means each is less than the next (older)
                if all(scores[i] > scores[i + 1] for i in range(len(scores) - 1)):
                    notes = f'Declining {len(scores) + 1} consecutive weeks'

            conn.execute(
                "INSERT INTO strategy_health_log "
                "(setup_id, week_start, sharpe_30d, sharpe_benchmark, win_rate, "
                " win_rate_benchmark, signal_count_week, expectancy, health_score, "
                " alert_level, notes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    sid,
                    monday.isoformat(),
                    metrics.get('sharpe_30d'),
                    metrics.get('sharpe_benchmark'),
                    metrics.get('win_rate'),
                    metrics.get('win_rate_benchmark'),
                    metrics.get('signal_count_week'),
                    metrics.get('expectancy'),
                    metrics.get('health_score'),
                    metrics.get('alert_level'),
                    notes,
                )
            )
            conn.commit()
            results[sid] = metrics
            logger.info(
                f'Research Health {sid}: score={metrics.get("health_score")} '
                f'alert={metrics.get("alert_level")}'
            )
        except Exception as e:
            logger.error(f'Health check failed {sid}: {e}')
            results[sid] = {'error': str(e)}

    return results


def seed_shadow_lab_candidates(conn):
    """Seed initial shadow lab candidates if the table is empty."""
    count = conn.execute("SELECT COUNT(*) FROM shadow_lab").fetchone()[0]
    if count > 0:
        logger.info(f'Shadow Lab: already seeded ({count} candidates)')
        return

    today = date.today()
    candidates = [
        {
            'name': 'Post-Low-Vol Expansion',
            'desc': (
                'Trades expansion moves following compressed volatility periods. '
                'Enters when ATR contracts below 20-day average then expands.'
            ),
            'sharpe': 29.91,
            'wr':      0.68,
        },
        {
            'name': 'Monday NY Open Long',
            'desc': (
                'Exploits Monday NY session upside bias on MNQ. Long only, '
                'first 30 minutes of NY open, requires bullish HTF bias.'
            ),
            'sharpe': 11.43,
            'wr':      0.61,
        },
        {
            'name': 'Value Area Continuation',
            'desc': (
                'Trades continuation moves from previous session value area high/low. '
                'Enters on retest of value area boundary with momentum confirmation.'
            ),
            'sharpe': 9.99,
            'wr':      0.58,
        },
    ]

    for c in candidates:
        promo = (today + timedelta(weeks=8)).isoformat()
        conn.execute(
            "INSERT INTO shadow_lab "
            "(strategy_name, description, entered_date, week_number, total_weeks, "
            " backtest_sharpe, backtest_win_rate, status, promotion_eligible_date) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                c['name'], c['desc'],
                today.isoformat(), 0, 8,
                c['sharpe'], c['wr'],
                'ACTIVE',
                promo,
            )
        )
    conn.commit()
    logger.info(f'Shadow Lab: seeded {len(candidates)} candidates')


def score_shadow_lab(conn):
    """Advance week numbers for active shadow lab candidates.
    Creates a PENDING research_decision when a candidate completes its total_weeks.
    """
    rows = conn.execute(
        "SELECT id, strategy_name, week_number, total_weeks, "
        "       paper_sharpe, backtest_sharpe "
        "FROM shadow_lab WHERE status = 'ACTIVE'"
    ).fetchall()

    updated = []
    for row in rows:
        sid, name, week_num, total_weeks, paper_sharpe, bt_sharpe = row
        new_week = (week_num or 0) + 1

        conn.execute(
            "UPDATE shadow_lab SET week_number = ? WHERE id = ?",
            (new_week, sid)
        )

        if new_week >= total_weeks:
            conn.execute(
                "INSERT INTO research_decisions "
                "(decision_type, subject, recommendation, supporting_data, status) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    'PROMOTION_REVIEW',
                    name,
                    (
                        f'{name} has completed {total_weeks} weeks in Shadow Lab. '
                        f'Review backtest vs paper performance for promotion decision.'
                    ),
                    json.dumps({
                        'week_number':     new_week,
                        'backtest_sharpe': bt_sharpe,
                        'paper_sharpe':    paper_sharpe,
                    }),
                    'PENDING',
                )
            )
            logger.info(f'Shadow Lab: {name} — promotion review created')

        conn.commit()
        updated.append({'id': sid, 'name': name, 'week_number': new_week})

    return updated


def generate_weekly_telegram_report(conn):
    """Compose and send the weekly Research Division Telegram report."""
    try:
        from telegram_alerts import load_telegram_config, send_telegram
        token, chat_id = load_telegram_config()
        if not token or not chat_id:
            logger.warning('Research report: Telegram not configured')
            return False
    except Exception as e:
        logger.error(f'Research report: Telegram import failed: {e}')
        return False

    today    = date.today()
    week_str = today.strftime('%d %b %Y')
    sep      = '━' * 21  # ━━━━━━━━━━━━━━━━━━━━━

    # Latest health per setup
    health_rows = conn.execute(
        "SELECT s.setup_id, s.health_score, s.alert_level, "
        "       s.sharpe_30d, s.sharpe_benchmark, s.win_rate "
        "FROM strategy_health_log s "
        "INNER JOIN ("
        "  SELECT setup_id, MAX(week_start) AS latest "
        "  FROM strategy_health_log GROUP BY setup_id"
        ") m ON s.setup_id = m.setup_id AND s.week_start = m.latest "
        "ORDER BY s.setup_id"
    ).fetchall()

    emoji_map = {
        'HEALTHY': '✅',
        'WATCH':   '⚠️',
        'ALERT':   '\U0001f534',
    }
    health_lines = []
    for row in health_rows:
        sid, score, alert, sharpe, sharpe_bench, wr = row
        name  = SETUP_NAMES.get(sid, sid)
        emoji = emoji_map.get(alert, '—')
        if score is None:
            health_lines.append(f'—  {sid} {name}: insufficient data')
        else:
            s_vs = ''
            if sharpe is not None and sharpe_bench:
                pct  = (sharpe / sharpe_bench - 1) * 100
                s_vs = f' ({pct:+.0f}%)'
            wr_s = f'{wr * 100:.1f}%' if wr is not None else '—'
            health_lines.append(
                f'{emoji}  {sid} {name}: {score}/100  '
                f'Sharpe {sharpe:.2f}{s_vs}  WR {wr_s}'
            )
    if not health_lines:
        health_lines = ['No health data yet — check runs Monday 06:00 UTC']

    # Active shadow lab candidates
    shadow_rows = conn.execute(
        "SELECT strategy_name, week_number, total_weeks, backtest_sharpe "
        "FROM shadow_lab WHERE status = 'ACTIVE' ORDER BY id"
    ).fetchall()

    shadow_lines = []
    for row in shadow_rows:
        name, week, total, bt_s = row
        shadow_lines.append(f'  {name}')
        shadow_lines.append(f'  Week {week or 0}/{total}  BT Sharpe: {bt_s:.2f}')
    if not shadow_lines:
        shadow_lines = ['  No active candidates']

    # Pending decisions
    dec_rows = conn.execute(
        "SELECT subject, recommendation FROM research_decisions "
        "WHERE status = 'PENDING' ORDER BY created_at DESC LIMIT 5"
    ).fetchall()

    dec_lines = []
    for row in dec_rows:
        subject, rec = row
        snippet = (rec or '')[:80]
        dec_lines.append(f'  ▸ {subject}: {snippet}')
    if not dec_lines:
        dec_lines = ['  None this week']

    msg = (
        '<b>WISE MERIDIAN CAPITAL</b>\n'
        f'Research Division — Week of {week_str}\n'
        f'{sep}\n\n'
        '<b>STRATEGY HEALTH</b>\n' + '\n'.join(health_lines) +
        '\n\n<b>SHADOW LAB</b>\n'  + '\n'.join(shadow_lines) +
        '\n\n<b>DECISIONS PENDING</b>\n' + '\n'.join(dec_lines) +
        f'\n\n{sep}\n'
        '<i>Wise Meridian Capital · Research Division</i>'
    )

    try:
        result = send_telegram(msg, token, chat_id, parse_mode='HTML')
        if result:
            logger.info('Research: weekly report sent via Telegram')
        return bool(result)
    except Exception as e:
        logger.error(f'Research: Telegram send failed: {e}')
        return False
