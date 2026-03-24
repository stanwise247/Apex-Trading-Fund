"""
APEX Live Scanner — live_scanner.py
=====================================
Session 6: Live signal detection, Telegram alerts.

Runs every minute, executes full gate check on 5min bar closes.
Alerts via Telegram when Setup B forms on NQ, ES, or GC.

Run:
  python3 live_scanner.py
"""

import sqlite3
import logging
import time
import json
import requests
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from setup_engine import check_setup
from setup_e import check_setup_e, format_alert as format_alert_e

logging.basicConfig(
    level  = logging.INFO,
    format = '%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger('APEX.Scanner')

NY_TZ = ZoneInfo('America/New_York')
UTC   = ZoneInfo('UTC')

DB_PATH    = 'apex_market.db'
INSTRUMENTS = ['NQ', 'ES', 'GC']
MODES       = ['swing']
SCAN_EVERY  = 60
BAR_MINUTES = 5

DAY_FILTERS = {
    'NQ': [0, 2, 3, 4],
    'ES': [0, 1, 3],
    'GC': [0, 1, 2, 3],
}


def load_telegram_config():
    import os
    token   = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID', '')
    try:
        with open('config.json') as f:
            cfg = json.load(f)
        token   = token   or cfg.get('TELEGRAM_BOT_TOKEN', cfg.get('telegram_token', ''))
        chat_id = chat_id or cfg.get('TELEGRAM_CHAT_ID', cfg.get('telegram_chat_id', ''))
    except Exception:
        pass
    return token, chat_id


def send_telegram(message: str):
    token, chat_id = load_telegram_config()
    if not token or not chat_id:
        logger.warning('Telegram not configured — alert not sent')
        logger.info(f'Alert: {message}')
        return False
    try:
        url  = f'https://api.telegram.org/bot{token}/sendMessage'
        resp = requests.post(url, json={
            'chat_id':    chat_id,
            'text':       message,
            'parse_mode': 'HTML',
        }, timeout=10)
        if resp.status_code == 200:
            logger.info('Telegram alert sent')
            return True
        else:
            logger.warning(f'Telegram error: {resp.status_code} {resp.text}')
            return False
    except Exception as e:
        logger.error(f'Telegram failed: {e}')
        return False


def format_alert(result) -> str:
    now_ny = datetime.now(timezone.utc).astimezone(NY_TZ)
    direction_emoji = '🟢' if result.direction == 'long' else '🔴'
    quality_tag     = '⭐ PRIMARY' if result.quality == 'primary' else 'SECONDARY'
    setup_names = {
        'A_sweep_ob':       'Setup A — Sweep + OB',
        'B_choch_breaker':  'Setup B — CHoCH + Breaker',
        'C_bos_ob':         'Setup C — BOS + OB',
        'E_ema50_pullback': 'Setup E — EMA50 Pullback',
    }
    setup_name = setup_names.get(result.setup, result.setup)
    msg = (
        f"{direction_emoji} <b>APEX SIGNAL — {result.symbol}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Direction:</b> {result.direction.upper()}\n"
        f"<b>Setup:</b>     {setup_name}\n"
        f"<b>Session:</b>   {quality_tag}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Entry:</b>     {result.entry:.2f}\n"
        f"<b>Stop:</b>      {result.stop:.2f} ({abs(round(result.entry - result.stop, 2))} pts)\n"
        f"<b>Target:</b>    {result.target:.2f} ({abs(round(result.entry - result.target, 2))} pts)\n"
        f"<b>R:R:</b>       {result.rr:.1f}x\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>{now_ny.strftime('%Y-%m-%d %H:%M')} ET</i>\n"
        f"<i>Gates: all 6 passed ✅</i>"
    )
    return msg


def init_swing_dedup_table():
    """Create swing_alerted_signals table for DB-backed dedup (matches fvg_alerted_zones pattern)."""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS swing_alerted_signals (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol    TEXT NOT NULL,
                direction TEXT NOT NULL,
                setup     TEXT NOT NULL,
                date      TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_swing_alert_key ON swing_alerted_signals (symbol, direction, setup, date)')
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f'swing dedup table init failed: {e}')


def is_swing_signal_new(symbol: str, direction: str, setup: str, dt: datetime = None) -> bool:
    """Return True if this signal has NOT been alerted today (DB-backed check)."""
    date_str = (dt or datetime.now(timezone.utc)).strftime('%Y-%m-%d')
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        row = conn.execute(
            'SELECT id FROM swing_alerted_signals WHERE symbol=? AND direction=? AND setup=? AND date=?',
            (symbol, direction, setup, date_str)
        ).fetchone()
        conn.close()
        return row is None
    except Exception as e:
        logger.warning(f'swing dedup check failed: {e}')
        return True  # fail open — allow alert


def record_swing_signal(symbol: str, direction: str, setup: str, dt: datetime = None):
    """Insert a row into swing_alerted_signals to mark this signal as alerted."""
    date_str = (dt or datetime.now(timezone.utc)).strftime('%Y-%m-%d')
    now_str  = datetime.now(timezone.utc).isoformat()
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.execute(
            'INSERT INTO swing_alerted_signals (symbol, direction, setup, date, created_at) VALUES (?,?,?,?,?)',
            (symbol, direction, setup, date_str, now_str)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f'swing dedup insert failed: {e}')


class SignalTracker:
    def __init__(self):
        self.sent = {}
        self.ttl  = 4 * 3600
        init_swing_dedup_table()

    def is_new(self, result) -> bool:
        key = (result.symbol, result.direction, result.setup,
               round(result.entry, 0) if result.entry else 0)
        now = time.time()
        if key in self.sent and now - self.sent[key] < self.ttl:
            return False
        # Also check DB dedup
        if not is_swing_signal_new(result.symbol, result.direction, result.setup):
            return False
        self.sent[key] = now
        return True

    def mark_sent(self, result):
        """Record in DB after successfully sending alert."""
        record_swing_signal(result.symbol, result.direction, result.setup)

    def cleanup(self):
        now = time.time()
        self.sent = {k: v for k, v in self.sent.items() if now - v < self.ttl}


def run_scan(dt: datetime = None) -> list:
    if dt is None:
        dt = datetime.now(timezone.utc)
    dow     = dt.weekday()
    results = []
    for symbol in INSTRUMENTS:
        allowed_days = DAY_FILTERS.get(symbol, [0,1,2,3,4])
        if dow not in allowed_days:
            logger.debug(f'{symbol}: skipped — day filter')
            continue
        for mode in MODES:
            for direction in ('long', 'short'):
                try:
                    result = check_setup(symbol, direction, mode, dt)
                    if result.valid:
                        results.append(result)
                        logger.info(
                            f'SIGNAL: {symbol} {direction.upper()} '
                            f'{result.setup} entry={result.entry} '
                            f'stop={result.stop} target={result.target}'
                        )
                except Exception as e:
                    logger.error(f'Gate check failed {symbol} {direction}: {e}')

    # Setup E — EMA50 Pullback, NQ only, session-gated internally
    if dow < 5:
        for direction in ('long', 'short'):
            try:
                result = check_setup_e('NQ', direction, dt)
                if result.valid:
                    results.append(result)
                    logger.info(
                        f'SIGNAL: NQ {direction.upper()} E_ema50_pullback '
                        f'entry={result.entry} stop={result.stop} target={result.target}'
                    )
            except Exception as e:
                logger.error(f'Setup E gate check failed NQ {direction}: {e}')

    return results


def scan_now() -> list:
    """Single scan pass — callable from server.py scheduler."""
    results = run_scan()
    return [
        {
            'symbol':    r.symbol,
            'direction': r.direction,
            'setup':     r.setup,
            'entry':     r.entry,
            'stop':      r.stop,
            'target':    r.target,
            'rr':        r.rr,
            'quality':   r.quality,
            'timestamp': str(r.timestamp),
        }
        for r in results
    ]


def main():
    logger.info('='*50)
    logger.info('  APEX Live Scanner starting')
    logger.info(f'  Instruments: {INSTRUMENTS}')
    logger.info(f'  Modes:       {MODES}')
    logger.info(f'  Scan every:  {SCAN_EVERY}s on 5min bar closes')
    logger.info('='*50)

    tracker         = SignalTracker()
    last_bar_minute = -1

    send_telegram(
        '🚀 <b>APEX Scanner Online</b>\n'
        f'Monitoring: {", ".join(INSTRUMENTS)}\n'
        f'Setups: A/B/C (swing) + E (EMA50 Pullback, NQ)\n'
        f'<i>{datetime.now(timezone.utc).astimezone(NY_TZ).strftime("%Y-%m-%d %H:%M")} ET</i>'
    )

    while True:
        try:
            now        = datetime.now(timezone.utc)
            minute     = now.minute
            bar_minute = (minute // BAR_MINUTES) * BAR_MINUTES

            if bar_minute == last_bar_minute:
                time.sleep(10)
                continue

            last_bar_minute = bar_minute
            logger.info(f'Bar close {now.strftime("%H:%M")} UTC — scanning...')

            signals = run_scan(now)

            for result in signals:
                if tracker.is_new(result):
                    msg = format_alert_e(result) if getattr(result, 'setup', '') == 'E_ema50_pullback' else format_alert(result)
                    send_telegram(msg)
                    tracker.mark_sent(result)
                    logger.info(f'Alert sent: {result.symbol} {result.direction}')
                    # Log trade to tracker
                    try:
                        from trade_tracker import log_trade
                        log_trade({
                            'symbol':    result.symbol,
                            'direction': result.direction,
                            'setup':     result.setup,
                            'mode':      'swing',
                            'entry':     result.entry,
                            'stop':      result.stop,
                            'target':    result.target,
                            'rr':        result.rr,
                            'session':   result.session,
                            'quality':   result.quality,
                        })
                    except Exception as e:
                        logger.error(f'Trade log error: {e}')
                else:
                    logger.debug(f'Duplicate suppressed: {result.symbol} {result.direction}')

            tracker.cleanup()

            if not signals:
                logger.info('No signals this bar')

            time.sleep(SCAN_EVERY)

        except KeyboardInterrupt:
            logger.info('Scanner stopped by user')
            send_telegram('⏹ <b>APEX Scanner stopped</b>')
            break
        except Exception as e:
            logger.error(f'Scanner error: {e}')
            time.sleep(30)


if __name__ == '__main__':
    main()
