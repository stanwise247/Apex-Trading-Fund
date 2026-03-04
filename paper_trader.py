"""
APEX Paper Trading Account — paper_trader.py
==============================================
Simulates a real trading account with:
  - Position management (entry, partial exits, full exit)
  - Breakeven stop management
  - Trailing stops
  - Real-time P&L tracking
  - Full trade journal
  - Performance statistics
  - Persisted to SQLite so survives server restarts
"""

import sqlite3
import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger('APEX.paper')

DB_PATH     = 'apex_market.db'
POINT_VALUE = 20.0   # $ per NQ point per contract
COMMISSION  = 5.0    # $ per round trip

# =============================================================
#  DATABASE SETUP
# =============================================================

def init_paper_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS paper_account (
        id INTEGER PRIMARY KEY,
        key TEXT UNIQUE,
        value TEXT
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS paper_positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        direction TEXT,
        entry_price REAL,
        current_price REAL,
        stop REAL,
        target1 REAL,
        target2 REAL,
        contracts INTEGER DEFAULT 1,
        contracts_remaining INTEGER DEFAULT 1,
        status TEXT DEFAULT 'open',
        setup_name TEXT,
        setup_score INTEGER,
        entry_ts INTEGER,
        exit_ts INTEGER,
        exit_price REAL,
        pnl_pts REAL DEFAULT 0,
        pnl_usd REAL DEFAULT 0,
        be_stop_moved INTEGER DEFAULT 0,
        partial_taken INTEGER DEFAULT 0,
        notes TEXT,
        alert_id TEXT
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS paper_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        position_id INTEGER,
        symbol TEXT,
        direction TEXT,
        entry_price REAL,
        exit_price REAL,
        contracts INTEGER,
        pnl_pts REAL,
        pnl_usd REAL,
        outcome TEXT,
        setup_name TEXT,
        setup_score INTEGER,
        entry_ts INTEGER,
        exit_ts INTEGER,
        hold_minutes INTEGER,
        partial INTEGER DEFAULT 0,
        notes TEXT
    )''')

    # Initialise account balance if not exists
    c.execute("INSERT OR IGNORE INTO paper_account (key, value) VALUES ('balance', '10000')")
    c.execute("INSERT OR IGNORE INTO paper_account (key, value) VALUES ('starting_balance', '10000')")
    c.execute("INSERT OR IGNORE INTO paper_account (key, value) VALUES ('peak_balance', '10000')")
    c.execute("INSERT OR IGNORE INTO paper_account (key, value) VALUES ('risk_pct', '2.0')")
    c.execute("INSERT OR IGNORE INTO paper_account (key, value) VALUES ('max_positions', '3')")
    c.execute("INSERT OR IGNORE INTO paper_account (key, value) VALUES ('total_trades', '0')")

    conn.commit()
    conn.close()
    logger.info('Paper trading account initialised')


def get_account_value(key):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT value FROM paper_account WHERE key=?', (key,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


def set_account_value(key, value):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('INSERT OR REPLACE INTO paper_account (key, value) VALUES (?,?)',
                 (key, str(value)))
    conn.commit()
    conn.close()


# =============================================================
#  ACCOUNT STATE
# =============================================================

def get_account_state():
    balance  = float(get_account_value('balance') or 10000)
    starting = float(get_account_value('starting_balance') or 10000)
    peak     = float(get_account_value('peak_balance') or balance)
    risk_pct = float(get_account_value('risk_pct') or 2.0)
    max_pos  = int(get_account_value('max_positions') or 3)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Open positions
    c.execute('SELECT * FROM paper_positions WHERE status="open" ORDER BY entry_ts DESC')
    cols = [d[0] for d in c.description]
    open_pos = [dict(zip(cols, r)) for r in c.fetchall()]

    # Recent closed trades
    c.execute('SELECT * FROM paper_trades ORDER BY exit_ts DESC LIMIT 50')
    cols = [d[0] for d in c.description]
    recent_trades = [dict(zip(cols, r)) for r in c.fetchall()]

    # Stats
    c.execute('SELECT COUNT(*), SUM(pnl_usd), SUM(CASE WHEN outcome="win" THEN 1 ELSE 0 END) FROM paper_trades WHERE partial=0')
    row = c.fetchone()
    total_trades = row[0] or 0
    total_pnl    = row[1] or 0
    total_wins   = row[2] or 0

    conn.close()

    win_rate = round(total_wins / total_trades * 100, 1) if total_trades > 0 else 0
    drawdown = round((peak - balance) / peak * 100, 2) if peak > 0 else 0
    ret      = round((balance - starting) / starting * 100, 2)

    # Unrealised P&L from open positions
    unrealised = sum(p.get('pnl_usd', 0) or 0 for p in open_pos)

    return {
        'balance':         round(balance, 2),
        'starting_balance':round(starting, 2),
        'peak_balance':    round(peak, 2),
        'total_return_pct':ret,
        'total_pnl_usd':   round(total_pnl, 2),
        'current_drawdown':drawdown,
        'risk_pct':        risk_pct,
        'max_positions':   max_pos,
        'open_positions':  open_pos,
        'open_count':      len(open_pos),
        'recent_trades':   recent_trades,
        'total_trades':    total_trades,
        'win_rate':        win_rate,
        'unrealised_pnl':  round(unrealised, 2),
        'equity':          round(balance + unrealised, 2),
    }


# =============================================================
#  POSITION SIZING
# =============================================================

def calculate_position_size(balance, risk_pct, entry, stop, point_value=POINT_VALUE):
    """
    Calculate how many contracts to trade.
    Risk = balance * risk_pct/100
    Stop distance in points * point_value * contracts = risk amount
    """
    risk_amount  = balance * (risk_pct / 100)
    stop_dist    = abs(entry - stop)
    if stop_dist <= 0:
        return 1, risk_amount
    contracts = max(1, int(risk_amount / (stop_dist * point_value)))
    contracts = min(contracts, 10)  # safety cap
    actual_risk = stop_dist * point_value * contracts
    return contracts, round(actual_risk, 2)


# =============================================================
#  OPEN POSITION
# =============================================================

def open_position(symbol, direction, entry_price, stop, target1, target2,
                  setup_name='', setup_score=0, alert_id='', notes=''):
    """Open a new paper trade position"""
    init_paper_db()

    state = get_account_state()
    balance   = state['balance']
    risk_pct  = state['risk_pct']
    max_pos   = state['max_positions']
    open_count= state['open_count']

    # Checks
    if open_count >= max_pos:
        return {'ok': False, 'error': f'Max positions ({max_pos}) already open'}

    # Check if already in this symbol/direction
    for p in state['open_positions']:
        if p['symbol'] == symbol and p['direction'] == direction:
            return {'ok': False, 'error': f'Already have {direction} {symbol} open'}

    contracts, risk_usd = calculate_position_size(balance, risk_pct, entry_price, stop)
    ts = int(datetime.now(timezone.utc).timestamp())

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO paper_positions
        (symbol, direction, entry_price, current_price, stop, target1, target2,
         contracts, contracts_remaining, status, setup_name, setup_score,
         entry_ts, pnl_pts, pnl_usd, notes, alert_id)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
        (symbol, direction, entry_price, entry_price, stop, target1, target2,
         contracts, contracts, 'open', setup_name, setup_score,
         ts, 0, 0, notes, alert_id)
    )
    pos_id = c.lastrowid
    conn.commit()
    conn.close()

    logger.info(f'📈 Paper trade opened: {direction.upper()} {symbol} @ {entry_price} | '
                f'Stop: {stop} | T1: {target1} | T2: {target2} | '
                f'{contracts} contract(s) | Risk: ${risk_usd:.2f}')

    return {
        'ok':         True,
        'position_id':pos_id,
        'symbol':     symbol,
        'direction':  direction,
        'entry':      entry_price,
        'stop':       stop,
        'target1':    target1,
        'target2':    target2,
        'contracts':  contracts,
        'risk_usd':   risk_usd,
        'setup':      setup_name,
        'score':      setup_score,
    }


# =============================================================
#  UPDATE POSITION (price update)
# =============================================================

def update_position_price(position_id, current_price):
    """Update unrealised P&L for an open position"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM paper_positions WHERE id=? AND status="open"', (position_id,))
    cols = [d[0] for d in c.description]
    row  = c.fetchone()
    if not row:
        conn.close()
        return None

    pos = dict(zip(cols, row))
    direction = pos['direction']
    entry     = float(pos['entry_price'])
    contracts = int(pos['contracts_remaining'])

    if direction in ('long', 'bullish'):
        pnl_pts = current_price - entry
    else:
        pnl_pts = entry - current_price

    pnl_usd = pnl_pts * POINT_VALUE * contracts - COMMISSION

    conn.execute('UPDATE paper_positions SET current_price=?, pnl_pts=?, pnl_usd=? WHERE id=?',
                 (current_price, round(pnl_pts, 2), round(pnl_usd, 2), position_id))
    conn.commit()

    # Check for stop or target hit
    result = check_exit_triggers(pos, current_price, conn)
    conn.close()
    return result


def check_exit_triggers(pos, current_price, conn=None):
    """Check if stop or target has been hit"""
    close_conn = conn is None
    if close_conn:
        conn = sqlite3.connect(DB_PATH)

    direction = pos['direction']
    stop  = float(pos['stop'])
    tgt1  = float(pos['target1'])
    tgt2  = float(pos['target2'])
    be    = bool(pos['be_stop_moved'])
    part  = bool(pos['partial_taken'])

    triggered = None

    if direction in ('long', 'bullish'):
        if current_price <= stop:
            triggered = 'stop'
        elif current_price >= tgt1 and not part:
            triggered = 'partial'
        elif current_price >= tgt2:
            triggered = 'target2'
    else:
        if current_price >= stop:
            triggered = 'stop'
        elif current_price <= tgt1 and not part:
            triggered = 'partial'
        elif current_price <= tgt2:
            triggered = 'target2'

    if triggered == 'partial':
        take_partial(pos['id'], current_price, conn)
    elif triggered in ('stop', 'target2'):
        close_position_db(pos['id'], current_price, triggered, conn)

    if close_conn:
        conn.close()

    return triggered


# =============================================================
#  PARTIAL EXIT
# =============================================================

def take_partial(position_id, exit_price, conn=None):
    """
    Take partial profit — exit half position at T1, move stop to breakeven.
    """
    close_conn = conn is None
    if close_conn:
        conn = sqlite3.connect(DB_PATH)

    c = conn.cursor()
    c.execute('SELECT * FROM paper_positions WHERE id=?', (position_id,))
    cols = [d[0] for d in c.description]
    row  = c.fetchone()
    if not row:
        if close_conn: conn.close()
        return

    pos        = dict(zip(cols, row))
    direction  = pos['direction']
    entry      = float(pos['entry_price'])
    contracts  = int(pos['contracts'])
    partial_ct = max(1, contracts // 2)

    if direction in ('long', 'bullish'):
        pnl_pts = exit_price - entry
    else:
        pnl_pts = entry - exit_price

    pnl_usd = pnl_pts * POINT_VALUE * partial_ct - COMMISSION

    # Log the partial trade
    ts = int(datetime.now(timezone.utc).timestamp())
    entry_ts = pos['entry_ts']
    hold_min = (ts - entry_ts) // 60

    conn.execute('''INSERT INTO paper_trades
        (position_id, symbol, direction, entry_price, exit_price, contracts,
         pnl_pts, pnl_usd, outcome, setup_name, setup_score, entry_ts, exit_ts,
         hold_minutes, partial)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)''',
        (position_id, pos['symbol'], direction, entry, exit_price, partial_ct,
         round(pnl_pts, 2), round(pnl_usd, 2), 'win',
         pos['setup_name'], pos['setup_score'], entry_ts, ts, hold_min)
    )

    # Move stop to breakeven and reduce contract count
    remaining = contracts - partial_ct
    conn.execute('''UPDATE paper_positions SET
        stop=?, contracts_remaining=?, partial_taken=1, be_stop_moved=1,
        pnl_usd=pnl_usd+?
        WHERE id=?''',
        (entry, remaining, round(pnl_usd, 2), position_id)
    )

    # Update account balance
    balance = float(get_account_value('balance') or 10000)
    new_balance = balance + pnl_usd
    set_account_value('balance', round(new_balance, 2))
    peak = float(get_account_value('peak_balance') or new_balance)
    if new_balance > peak:
        set_account_value('peak_balance', round(new_balance, 2))

    conn.commit()
    if close_conn:
        conn.close()

    logger.info(f'💰 Partial exit: {pos["symbol"]} @ {exit_price} | '
                f'+{pnl_pts:.1f}pts | ${pnl_usd:.2f} | Stop moved to BE @ {entry}')

    return {'ok': True, 'pnl_pts': pnl_pts, 'pnl_usd': pnl_usd}


# =============================================================
#  CLOSE POSITION
# =============================================================

def close_position(position_id, exit_price, reason='manual'):
    """Close an open position"""
    conn = sqlite3.connect(DB_PATH)
    close_position_db(position_id, exit_price, reason, conn)
    conn.close()


def close_position_db(position_id, exit_price, reason, conn):
    c = conn.cursor()
    c.execute('SELECT * FROM paper_positions WHERE id=? AND status="open"', (position_id,))
    cols = [d[0] for d in c.description]
    row  = c.fetchone()
    if not row:
        return

    pos       = dict(zip(cols, row))
    direction = pos['direction']
    entry     = float(pos['entry_price'])
    contracts = int(pos['contracts_remaining'])
    entry_ts  = pos['entry_ts']
    ts        = int(datetime.now(timezone.utc).timestamp())
    hold_min  = (ts - entry_ts) // 60

    if direction in ('long', 'bullish'):
        pnl_pts = exit_price - entry
    else:
        pnl_pts = entry - exit_price

    pnl_usd = pnl_pts * POINT_VALUE * contracts - COMMISSION
    outcome = 'win' if pnl_pts > 0 else 'loss' if pnl_pts < 0 else 'breakeven'

    # Log trade
    conn.execute('''INSERT INTO paper_trades
        (position_id, symbol, direction, entry_price, exit_price, contracts,
         pnl_pts, pnl_usd, outcome, setup_name, setup_score, entry_ts, exit_ts,
         hold_minutes, partial, notes)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,?)''',
        (position_id, pos['symbol'], direction, entry, exit_price, contracts,
         round(pnl_pts, 2), round(pnl_usd, 2), outcome,
         pos['setup_name'], pos['setup_score'],
         entry_ts, ts, hold_min, reason)
    )

    # Close position
    conn.execute('''UPDATE paper_positions SET
        status="closed", exit_ts=?, exit_price=?, pnl_pts=?, pnl_usd=?
        WHERE id=?''',
        (ts, exit_price, round(pnl_pts + float(pos.get('pnl_pts') or 0), 2),
         round(pnl_usd + float(pos.get('pnl_usd') or 0), 2), position_id)
    )

    # Update balance
    balance = float(get_account_value('balance') or 10000)
    new_balance = balance + pnl_usd
    set_account_value('balance', round(new_balance, 2))
    peak = float(get_account_value('peak_balance') or new_balance)
    if new_balance > peak:
        set_account_value('peak_balance', round(new_balance, 2))
    total = int(get_account_value('total_trades') or 0) + 1
    set_account_value('total_trades', total)

    conn.commit()

    emoji = '✅' if outcome == 'win' else '❌' if outcome == 'loss' else '➖'
    logger.info(f'{emoji} Trade closed: {pos["symbol"]} {direction.upper()} | '
                f'Entry: {entry} Exit: {exit_price} | '
                f'{pnl_pts:+.1f}pts | ${pnl_usd:+.2f} | {reason} | {hold_min}min')


# =============================================================
#  MANUAL CLOSE ALL
# =============================================================

def close_all_positions(current_prices: dict):
    """Emergency close all open positions"""
    state = get_account_state()
    for pos in state['open_positions']:
        symbol = pos['symbol']
        price  = current_prices.get(symbol, pos['current_price'])
        close_position(pos['id'], price, 'manual_close_all')
    logger.info('All positions closed')


# =============================================================
#  PERFORMANCE REPORT
# =============================================================

def get_performance_report():
    state = get_account_state()
    trades = state['recent_trades']

    if not trades:
        return {**state, 'performance': 'No trades yet'}

    wins   = [t for t in trades if t['outcome'] == 'win' and not t['partial']]
    losses = [t for t in trades if t['outcome'] == 'loss']
    total  = [t for t in trades if not t['partial']]

    avg_win  = sum(t['pnl_usd'] for t in wins)   / len(wins)   if wins   else 0
    avg_loss = sum(t['pnl_usd'] for t in losses) / len(losses) if losses else 0
    pf = abs(avg_win * len(wins)) / abs(avg_loss * len(losses)) if losses and avg_loss != 0 else 999

    return {
        **state,
        'performance': {
            'total_closed_trades': len(total),
            'wins':                len(wins),
            'losses':              len(losses),
            'win_rate_pct':        round(len(wins)/len(total)*100, 1) if total else 0,
            'avg_win_usd':         round(avg_win, 2),
            'avg_loss_usd':        round(avg_loss, 2),
            'profit_factor':       round(pf, 2),
            'best_trade_usd':      max((t['pnl_usd'] for t in total), default=0),
            'worst_trade_usd':     min((t['pnl_usd'] for t in total), default=0),
        }
    }
