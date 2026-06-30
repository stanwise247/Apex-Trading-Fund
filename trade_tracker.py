"""
APEX Trade Tracker — trade_tracker.py
=======================================
Logs signals as open trades, monitors exits, fires Telegram alerts.

Entry: called from live_scanner.py when signal fires
Monitor: called every 5 minutes from server.py scheduler
Exit alerts: stop hit | target hit | session end | max bars
"""

import logging
import json
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import db as _db

logger = logging.getLogger('APEX.TradeTracker')
NY_TZ   = ZoneInfo('America/New_York')
UTC     = ZoneInfo('UTC')

SESSION_END_UTC = {
    'MNQ': 19, 'NQ': 19, 'ES': 19, 'MES': 19, 'GC': 17, 'MGC': 17
}
MAX_BARS = {
    'swing':    100,   # ~8h on 5min bars
    'intraday':  48,   # 4h on 5min bars — Setup I/D/E/H
    'scalp':     30,   # 2.5h on 5min bars
}


# ─────────────────────────────────────────────────────────────
#  DATABASE SETUP
# ─────────────────────────────────────────────────────────────

def init_trades_table():
    """Create trades table if it doesn't exist."""
    conn = _db.connect()
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
            bars_held        INTEGER DEFAULT 0,
            notes            TEXT,
            broker_order_id  TEXT,
            contracts        INTEGER DEFAULT 1
        )
    ''')
    # Add columns if missing (migrations for existing SQLite DBs)
    if not _db.IS_POSTGRES:
        for _col, _ddl in [
            ('broker_order_id', 'ALTER TABLE apex_trades ADD COLUMN broker_order_id TEXT'),
            ('contracts',       'ALTER TABLE apex_trades ADD COLUMN contracts INTEGER DEFAULT 1'),
        ]:
            try:
                conn.execute(_ddl)
                conn.commit()
            except Exception:
                pass  # Column already exists
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────
#  LOG TRADE (called when signal fires)
# ─────────────────────────────────────────────────────────────

def log_trade(signal: dict) -> int:
    """
    Log a new trade when a signal fires.
    Returns the trade ID (always an int — raises RuntimeError if INSERT fails).
    signal dict keys: symbol, direction, setup, mode, entry,
                      stop, target, rr, session, quality
    """
    sym = signal.get('symbol')
    dirn = signal.get('direction')
    setup = signal.get('setup')
    entry = signal.get('entry')
    logger.info(f'log_trade: {sym} {dirn} {setup} entry={entry}')

    init_trades_table()
    conn = _db.connect()
    now  = datetime.now(timezone.utc).isoformat()

    try:
        trade_id = _db.insert_returning_id(conn, '''
            INSERT INTO apex_trades
            (symbol, direction, setup, mode, entry_price, stop, target,
             rr_planned, session, quality, entry_time, status, broker_order_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)
        ''', (
            sym,
            dirn,
            setup,
            signal.get('mode', 'swing'),
            entry,
            signal.get('stop'),
            signal.get('target'),
            signal.get('rr'),
            signal.get('session', ''),
            signal.get('quality', ''),
            now,
            signal.get('broker_order_id'),
        ))
        if not trade_id:
            raise RuntimeError(
                f'INSERT returned no id for {sym} {dirn} {setup} entry={entry} — row may not have committed'
            )
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        conn.close()
        raise
    conn.close()
    logger.info(f'Trade logged: #{trade_id} {sym} {dirn.upper() if dirn else "?"}')
    return trade_id


# ─────────────────────────────────────────────────────────────
#  GET CURRENT PRICE
# ─────────────────────────────────────────────────────────────

def get_current_price(symbol: str, timeframe: str = '5min') -> float:
    """Get the most recent close price from the database."""
    try:
        conn = _db.connect()
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
    conn   = _db.connect()
    trade  = conn.execute(
        'SELECT * FROM apex_trades WHERE id=?', (trade_id,)
    ).fetchone()

    if not trade:
        conn.close()
        return None

    cols = ['id','symbol','direction','setup','mode','entry_price','stop',
            'target','rr_planned','session','quality','entry_time','exit_price',
            'exit_time','exit_reason','pnl_r','status','bars_held','notes','broker_order_id','contracts']
    t = dict(zip(cols, trade))

    entry  = float(t['entry_price'])
    stop   = float(t['stop'])
    risk   = abs(entry - stop)

    if risk >= 1.0:
        if t['direction'] == 'long':
            pnl_r = round((exit_price - entry) / risk, 3)
        else:
            pnl_r = round((entry - exit_price) / risk, 3)
        # Sanity cap — anything beyond ±10R is a data/calculation error
        if abs(pnl_r) > 10.0:
            logger.warning(
                f'close_trade #{trade_id}: pnl_r={pnl_r:.2f}R exceeds ±10R cap '
                f'(entry={entry}, exit={exit_price}, risk={risk:.2f}pt) — '
                f'storing None to prevent bad data'
            )
            pnl_r = None
    else:
        # risk < 1pt — dividing by near-zero stop distance produces nonsense R
        logger.warning(
            f'close_trade #{trade_id}: stop_distance={risk:.3f}pt < 1pt — '
            f'pnl_r calculation skipped (calculation_error)'
        )
        pnl_r = None

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

    pnl    = trade.get('pnl_r')
    if pnl is None:
        pnl = 0.0  # capped by ±10R sanity check or sub-1pt stop
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

    if trade.get('quality') == 'shadow_lab':
        msg = (
            f'[SHADOW LAB] {setup}\n'
            f'{sym} {dir_} — {title}\n'
            f'Entry: {entry:.2f} | Exit: {exit_:.2f} | P&L: {pnl_str}\n'
            f'Paper trade — no real money moved\n'
            f'<i>{now_ny} ET</i>'
        )
    else:
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
    conn   = _db.connect()
    trades = conn.execute(
        'SELECT * FROM apex_trades WHERE status=?', ('open',)
    ).fetchall()
    conn.close()

    logger.info(f'monitor_trades: checking {len(trades)} open trades')

    if not trades:
        return

    cols = ['id','symbol','direction','setup','mode','entry_price','stop',
            'target','rr_planned','session','quality','entry_time','exit_price',
            'exit_time','exit_reason','pnl_r','status','bars_held','notes','broker_order_id','contracts']

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
        _tf = '1min' if is_fvg else '5min'
        SYMBOL_ALIASES = {'NQ': 'MNQ'}
        price = get_current_price(sym, timeframe=_tf)
        if price is None and sym in SYMBOL_ALIASES:
            price = get_current_price(SYMBOL_ALIASES[sym], timeframe=_tf)
            if price is not None:
                logger.info(f'monitor_trades: {sym} resolved via alias → {SYMBOL_ALIASES[sym]}')
        if price is None:
            logger.warning(f'monitor_trades: {sym} — could not get current price, skipping')
            continue

        risk = abs(entry - stop)
        if risk > 0:
            unreal_r = round((price - entry) / risk, 3) if direction == 'long' else round((entry - price) / risk, 3)
        else:
            unreal_r = 0.0
        logger.info(
            f'monitor_trades: {sym} {direction} {t["setup"]} | '
            f'entry={entry:.2f} current={price:.2f} unrealised={unreal_r:+.3f}R | '
            f'stop={stop:.2f} target={target:.2f}'
        )

        exit_reason = None
        exit_price  = price

        # Check stop — close must fully breach (strict), never bar high/low
        if direction == 'long'  and price < stop:
            exit_reason = 'stop'
            exit_price  = stop
        elif direction == 'short' and price > stop:
            exit_reason = 'stop'
            exit_price  = stop

        # Check target — close must fully breach (strict), never bar high/low
        if exit_reason is None:
            if direction == 'long'  and price > target:
                exit_reason = 'target'
                exit_price  = target
            elif direction == 'short' and price < target:
                exit_reason = 'target'
                exit_price  = target

        # Compute bars_held (needed for both session_end guard and max_bars check)
        entry_time = datetime.fromisoformat(t['entry_time'])
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        bar_seconds = 60 if is_fvg else 300
        bars_held = int((now - entry_time).total_seconds() / bar_seconds)

        # Check session end — require at least 5 bars open to prevent
        # crash-restart from force-closing fresh trades
        if exit_reason is None:
            sess_end = SESSION_END_UTC.get(sym, 19)
            if now.hour >= sess_end and bars_held >= 5:
                exit_reason = 'session_end'
                exit_price  = price

        # Check max bars
        if exit_reason is None:
            max_b = MAX_BARS.get(mode, 100)
            if bars_held >= max_b:
                exit_reason = 'max_bars'
                exit_price  = price

        if exit_reason:
            closed = close_trade(t['id'], exit_price, exit_reason)
            if closed:
                send_exit_alert(closed)
                _mt_pnl = closed.get('pnl_r')
                _mt_pnl_str = f'{_mt_pnl:+.2f}R' if _mt_pnl is not None else 'None'
                logger.info(
                    f'monitor_trades: EXIT #{t["id"]} {sym} {direction} [{t["setup"]}] '
                    f'reason={exit_reason} entry={entry:.2f} exit={exit_price:.2f} '
                    f'pnl={_mt_pnl_str}'
                )
                # Close Tradovate position — only if this trade was sent to the broker
                if t.get('broker_order_id'):
                    try:
                        from tradovate import place_market_close, TRADOVATE_ENABLED
                        if TRADOVATE_ENABLED:
                            num_contracts = int(t.get('contracts') or 1)
                            tv_result = place_market_close(sym, direction, contracts=num_contracts)
                            if tv_result.get('ok'):
                                logger.info(
                                    f'monitor_trades: Tradovate close sent — '
                                    f'#{t["id"]} {sym} {direction} {num_contracts}ct [{exit_reason}]'
                                )
                            else:
                                logger.warning(
                                    f'monitor_trades: Tradovate close FAILED — '
                                    f'#{t["id"]} {sym}: {tv_result.get("error")}'
                                )
                    except Exception as _tv_e:
                        logger.warning(
                            f'monitor_trades: Tradovate close exception — '
                            f'#{t["id"]} {sym}: {_tv_e}'
                        )


# ─────────────────────────────────────────────────────────────
#  GET STATS
# ─────────────────────────────────────────────────────────────

def get_stats() -> dict:
    """Return summary stats for dashboard."""
    init_trades_table()
    conn   = _db.connect()
    closed = conn.execute(
        'SELECT pnl_r FROM apex_trades WHERE status=?', ('closed',)
    ).fetchall()
    open_t = conn.execute(
        'SELECT COUNT(*) FROM apex_trades WHERE status=?', ('open',)
    ).fetchone()[0]

    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    today_trades = conn.execute(
        "SELECT pnl_r FROM apex_trades "
        "WHERE status='closed' AND entry_time LIKE ? "
        "AND quality NOT IN ('test') "
        "AND exit_reason NOT IN ('test_endpoint', 'test_cleanup', 'test_simulation') "
        "AND pnl_r IS NOT NULL AND ABS(pnl_r) <= 10",
        (f'{today}%',)
    ).fetchall()
    conn.close()

    all_r   = [r[0] for r in closed if r[0] is not None and abs(r[0]) <= 10]
    today_r = [r[0] for r in today_trades]
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
    conn   = _db.connect()
    trades = conn.execute(
        'SELECT * FROM apex_trades WHERE status=? ORDER BY entry_time DESC',
        ('open',)
    ).fetchall()
    conn.close()

    cols = ['id','symbol','direction','setup','mode','entry_price','stop',
            'target','rr_planned','session','quality','entry_time','exit_price',
            'exit_time','exit_reason','pnl_r','status','bars_held','notes','broker_order_id','contracts']

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
