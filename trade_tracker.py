"""
APEX Trade Tracker — trade_tracker.py
=======================================
Logs signals as open trades, monitors exits, fires Telegram alerts.

Entry: called from live_scanner.py when signal fires
Monitor: called every 5 minutes from server.py scheduler
Exit alerts: stop hit | target hit | session end | max bars
"""

import sqlite3
import logging
import json
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

logger  = logging.getLogger('APEX.TradeTracker')
DB_PATH = 'apex_market.db'
NY_TZ   = ZoneInfo('America/New_York')
UTC     = ZoneInfo('UTC')

SESSION_END_UTC = {
    'NQ': 19, 'ES': 19, 'GC': 17
}
MAX_BARS = {
    'swing': 100, 'scalp': 30
}


# ─────────────────────────────────────────────────────────────
#  DATABASE SETUP
# ─────────────────────────────────────────────────────────────

def init_trades_table():
    """Create trades table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS apex_trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol      TEXT NOT NULL,
            direction   TEXT NOT NULL,
            setup       TEXT NOT NULL,
            mode        TEXT DEFAULT 'swing',
            entry_price REAL,
            stop        REAL,
            target      REAL,
            rr_planned  REAL,
            session     TEXT,
            quality     TEXT,
            entry_time  TEXT,
            exit_price  REAL,
            exit_time   TEXT,
            exit_reason TEXT,
            pnl_r       REAL,
            status      TEXT DEFAULT 'open',
            bars_held   INTEGER DEFAULT 0,
            notes       TEXT
        )
    ''')
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────
#  LOG TRADE (called when signal fires)
# ─────────────────────────────────────────────────────────────

def log_trade(signal: dict) -> int:
    """
    Log a new trade when a signal fires.
    Returns the trade ID.
    signal dict keys: symbol, direction, setup, mode, entry,
                      stop, target, rr, session, quality
    """
    init_trades_table()
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    now  = datetime.now(timezone.utc).isoformat()

    c.execute('''
        INSERT INTO apex_trades
        (symbol, direction, setup, mode, entry_price, stop, target,
         rr_planned, session, quality, entry_time, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
    ''', (
        signal.get('symbol'),
        signal.get('direction'),
        signal.get('setup'),
        signal.get('mode', 'swing'),
        signal.get('entry'),
        signal.get('stop'),
        signal.get('target'),
        signal.get('rr'),
        signal.get('session', ''),
        signal.get('quality', ''),
        now,
    ))
    trade_id = c.lastrowid
    conn.commit()
    conn.close()
    logger.info(f'Trade logged: #{trade_id} {signal.get("symbol")} {signal.get("direction").upper()}')
    return trade_id


# ─────────────────────────────────────────────────────────────
#  GET CURRENT PRICE
# ─────────────────────────────────────────────────────────────

def get_current_price(symbol: str, timeframe: str = '5min') -> float:
    """Get the most recent close price from the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df_row = conn.execute(
            'SELECT close FROM ohlcv WHERE symbol=? AND timeframe=? '
            'ORDER BY ts DESC LIMIT 1',
            (symbol, timeframe)
        ).fetchone()
        conn.close()
        return float(df_row[0]) if df_row else None
    except Exception as e:
        logger.error(f'get_current_price error {symbol}: {e}')
        return None


# ─────────────────────────────────────────────────────────────
#  CLOSE TRADE
# ─────────────────────────────────────────────────────────────

def close_trade(trade_id: int, exit_price: float, reason: str):
    """Close a trade and calculate P&L."""
    conn   = sqlite3.connect(DB_PATH)
    trade  = conn.execute(
        'SELECT * FROM apex_trades WHERE id=?', (trade_id,)
    ).fetchone()

    if not trade:
        conn.close()
        return None

    cols = ['id','symbol','direction','setup','mode','entry_price','stop',
            'target','rr_planned','session','quality','entry_time','exit_price',
            'exit_time','exit_reason','pnl_r','status','bars_held','notes']
    t = dict(zip(cols, trade))

    entry  = float(t['entry_price'])
    stop   = float(t['stop'])
    risk   = abs(entry - stop)

    if risk > 0:
        if t['direction'] == 'long':
            pnl_r = round((exit_price - entry) / risk, 3)
        else:
            pnl_r = round((entry - exit_price) / risk, 3)
    else:
        pnl_r = 0.0

    now = datetime.now(timezone.utc).isoformat()
    conn.execute('''
        UPDATE apex_trades
        SET exit_price=?, exit_time=?, exit_reason=?, pnl_r=?, status='closed'
        WHERE id=?
    ''', (exit_price, now, reason, pnl_r, trade_id))
    conn.commit()
    conn.close()

    t['exit_price']  = exit_price
    t['exit_time']   = now
    t['exit_reason'] = reason
    t['pnl_r']       = pnl_r
    return t


# ─────────────────────────────────────────────────────────────
#  TELEGRAM
# ─────────────────────────────────────────────────────────────

def send_exit_alert(trade: dict):
    """Send Telegram exit alert."""
    try:
        from live_scanner import send_telegram
    except Exception as e:
        logger.error(f'send_exit_alert: cannot import send_telegram: {e}')
        return

    pnl    = trade.get('pnl_r', 0)
    reason = trade.get('exit_reason', '')
    sym    = trade.get('symbol')
    dir_   = trade.get('direction', '').upper()
    setup  = trade.get('setup', '')
    entry  = trade.get('entry_price')
    exit_  = trade.get('exit_price')
    now_ny = datetime.now(timezone.utc).astimezone(NY_TZ).strftime('%Y-%m-%d %H:%M')

    if reason == 'target':
        emoji = '✅'
        title = 'TARGET HIT'
    elif reason == 'stop':
        emoji = '❌'
        title = 'STOP HIT'
    elif reason == 'session_end':
        emoji = '⏰'
        title = 'SESSION END — CLOSE NOW'
    elif reason == 'max_bars':
        emoji = '⏱'
        title = 'MAX HOLD — CLOSE NOW'
    else:
        emoji = '📋'
        title = 'EXIT'

    pnl_str = f'+{pnl:.2f}R' if pnl >= 0 else f'{pnl:.2f}R'

    msg = (
        f'{emoji} <b>APEX EXIT — {sym}</b>\n'
        f'━━━━━━━━━━━━━━━━━━━━\n'
        f'<b>{title}</b>\n'
        f'<b>Direction:</b> {dir_}\n'
        f'<b>Setup:</b>     {setup}\n'
        f'━━━━━━━━━━━━━━━━━━━━\n'
        f'<b>Entry:</b>     {entry:.2f}\n'
        f'<b>Exit:</b>      {exit_:.2f}\n'
        f'<b>P&L:</b>       {pnl_str}\n'
        f'━━━━━━━━━━━━━━━━━━━━\n'
        f'<i>{now_ny} ET</i>'
    )
    send_telegram(msg)
    logger.info(f'Exit alert sent: {sym} {dir_} {pnl_str} [{reason}]')


# ─────────────────────────────────────────────────────────────
#  MONITOR OPEN TRADES
# ─────────────────────────────────────────────────────────────

def monitor_trades():
    """
    Check all open trades against current price.
    Called every 5 minutes from server.py scheduler.
    """
    init_trades_table()
    conn   = sqlite3.connect(DB_PATH)
    trades = conn.execute(
        'SELECT * FROM apex_trades WHERE status=?', ('open',)
    ).fetchall()
    conn.close()

    if not trades:
        return

    cols = ['id','symbol','direction','setup','mode','entry_price','stop',
            'target','rr_planned','session','quality','entry_time','exit_price',
            'exit_time','exit_reason','pnl_r','status','bars_held','notes']

    now = datetime.now(timezone.utc)

    for row in trades:
        t = dict(zip(cols, row))
        sym       = t['symbol']
        direction = t['direction']
        entry     = float(t['entry_price'])
        stop      = float(t['stop'])
        target    = float(t['target'])
        mode      = t.get('mode', 'swing')

        is_fvg = t.get('setup', '').startswith('FVG')
        price = get_current_price(sym, timeframe='1min' if is_fvg else '5min')
        if price is None:
            continue

        exit_reason = None
        exit_price  = price

        # Check stop
        if direction == 'long'  and price <= stop:
            exit_reason = 'stop'
            exit_price  = stop
        elif direction == 'short' and price >= stop:
            exit_reason = 'stop'
            exit_price  = stop

        # Check target
        if exit_reason is None:
            if direction == 'long'  and price >= target:
                exit_reason = 'target'
                exit_price  = target
            elif direction == 'short' and price <= target:
                exit_reason = 'target'
                exit_price  = target

        # Check session end
        if exit_reason is None:
            sess_end = SESSION_END_UTC.get(sym, 19)
            if now.hour >= sess_end:
                exit_reason = 'session_end'
                exit_price  = price

        # Check max bars
        if exit_reason is None:
            entry_time = datetime.fromisoformat(t['entry_time'])
            if entry_time.tzinfo is None:
                entry_time = entry_time.replace(tzinfo=timezone.utc)
            bar_seconds = 60 if is_fvg else 300
            bars_held = int((now - entry_time).total_seconds() / bar_seconds)
            max_b = MAX_BARS.get(mode, 100)
            if bars_held >= max_b:
                exit_reason = 'max_bars'
                exit_price  = price

        if exit_reason:
            closed = close_trade(t['id'], exit_price, exit_reason)
            if closed:
                send_exit_alert(closed)
                logger.info(
                    f'Trade #{t["id"]} {sym} {direction} closed: '
                    f'{exit_reason} @ {exit_price:.2f} | {closed["pnl_r"]:+.2f}R'
                )


# ─────────────────────────────────────────────────────────────
#  GET STATS
# ─────────────────────────────────────────────────────────────

def get_stats() -> dict:
    """Return summary stats for dashboard."""
    init_trades_table()
    conn   = sqlite3.connect(DB_PATH)
    closed = conn.execute(
        'SELECT pnl_r FROM apex_trades WHERE status=?', ('closed',)
    ).fetchall()
    open_t = conn.execute(
        'SELECT COUNT(*) FROM apex_trades WHERE status=?', ('open',)
    ).fetchone()[0]

    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    today_trades = conn.execute(
        'SELECT pnl_r FROM apex_trades WHERE status=? AND entry_time LIKE ?',
        ('closed', f'{today}%')
    ).fetchall()
    conn.close()

    all_r    = [r[0] for r in closed if r[0] is not None]
    today_r  = [r[0] for r in today_trades if r[0] is not None]
    winners  = [r for r in all_r if r > 0]

    return {
        'total_trades':   len(all_r),
        'open_trades':    open_t,
        'win_rate':       round(len(winners) / len(all_r) * 100, 1) if all_r else 0,
        'total_r':        round(sum(all_r), 2),
        'today_r':        round(sum(today_r), 2),
        'expectancy':     round(sum(all_r) / len(all_r), 3) if all_r else 0,
    }


def get_open_trades() -> list:
    """Return all open trades with current P&L."""
    init_trades_table()
    conn   = sqlite3.connect(DB_PATH)
    trades = conn.execute(
        'SELECT * FROM apex_trades WHERE status=? ORDER BY entry_time DESC',
        ('open',)
    ).fetchall()
    conn.close()

    cols = ['id','symbol','direction','setup','mode','entry_price','stop',
            'target','rr_planned','session','quality','entry_time','exit_price',
            'exit_time','exit_reason','pnl_r','status','bars_held','notes']

    result = []
    for row in trades:
        t     = dict(zip(cols, row))
        price = get_current_price(t['symbol'])
        if price:
            entry = float(t['entry_price'])
            stop  = float(t['stop'])
            risk  = abs(entry - stop)
            if risk > 0:
                if t['direction'] == 'long':
                    t['current_pnl_r'] = round((price - entry) / risk, 3)
                else:
                    t['current_pnl_r'] = round((entry - price) / risk, 3)
            else:
                t['current_pnl_r'] = 0.0
            t['current_price'] = price
        result.append(t)
    return result
