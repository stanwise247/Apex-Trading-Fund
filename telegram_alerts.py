"""
APEX Telegram Alert System — telegram_alerts.py
=================================================
Sends rich, formatted alerts to your Telegram when:
  - A high-conviction setup is detected
  - A position hits partial profit (T1)
  - A stop loss is hit
  - A target is hit
  - Daily summary at market close

Setup:
  1. Message @BotFather on Telegram
  2. Create new bot: /newbot
  3. Copy the token it gives you
  4. Message your bot once (so it can find your chat ID)
  5. Visit: https://api.telegram.org/bot<TOKEN>/getUpdates
  6. Find your chat_id in the response
  7. Add both to config.json as telegram_token and telegram_chat_id
"""

import json
import logging
import sqlite3
import time
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timezone

logger = logging.getLogger('APEX.telegram')

DB_PATH = 'apex_market.db'


# =============================================================
#  TELEGRAM API
# =============================================================

def send_telegram(message, token, chat_id, parse_mode='HTML', silent=False):
    """Send a message via Telegram Bot API"""
    if not token or not chat_id:
        logger.warning('Telegram not configured — skipping alert')
        return False

    url  = f'https://api.telegram.org/bot{token}/sendMessage'
    data = {
        'chat_id':              str(chat_id),
        'text':                 message,
        'parse_mode':           parse_mode,
        'disable_notification': silent,
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode())
            if result.get('ok'):
                logger.info(f'Telegram sent: {message[:60]}...')
                return True
            else:
                logger.error(f'Telegram error: {result}')
                return False
    except Exception as e:
        logger.error(f'Telegram send failed: {e}')
        return False


def load_telegram_config():
    """Load Telegram credentials from config.json"""
    try:
        with open('config.json') as f:
            cfg = json.load(f)
        token   = cfg.get('telegram_token', '')
        chat_id = cfg.get('telegram_chat_id', '')
        return token, chat_id
    except Exception:
        return '', ''


def test_telegram():
    """Test Telegram connection"""
    token, chat_id = load_telegram_config()
    if not token or not chat_id:
        return False, 'Telegram not configured in config.json'
    result = send_telegram(
        '🤖 <b>APEX Connected</b>\n\nTelegram alerts are working. You will receive trade signals here.',
        token, chat_id
    )
    return result, 'OK' if result else 'Failed to send'


# =============================================================
#  ALERT FORMATTERS
# =============================================================

def format_setup_alert(setup):
    """
    Format a trade setup alert with full context.
    Includes entry, stop, targets, reasoning, and paper trade instructions.
    """
    direction = setup.get('direction', 'long')
    symbol    = setup.get('symbol', 'NQ')
    score     = setup.get('score', 0)
    entry     = setup.get('entry', 0)
    stop      = setup.get('stop', 0)
    t1        = setup.get('target1', 0)
    t2        = setup.get('target2', 0)
    rr        = setup.get('rr', 0)
    signal    = setup.get('signal', '')
    tf        = setup.get('timeframe', '')
    session   = setup.get('session_label', 'Unknown Session')
    vix_reg   = setup.get('vix_regime', 'normal')
    conf      = setup.get('confluence', 0)
    reasons   = setup.get('reasons', [])
    ob_conf   = setup.get('ob_confluence', False)
    bos       = setup.get('bos_confirmed', False)
    fvg       = setup.get('fvg_inside', False)
    sweep     = setup.get('liquidity_sweep', False)
    pd_zone   = setup.get('pd_zone', '')

    # Direction emoji
    dir_emoji = '🟢' if direction in ('long','bullish') else '🔴'
    dir_label = 'LONG' if direction in ('long','bullish') else 'SHORT'

    # Score badge
    if score >= 85:   badge = '⭐⭐⭐ A+'
    elif score >= 75: badge = '⭐⭐ A'
    elif score >= 65: badge = '⭐ B'
    else:             badge = 'C'

    # Risk/reward
    risk_pts   = abs(entry - stop)
    reward_pts = abs(t1 - entry)

    # Build confluence summary
    confluence_lines = []
    if ob_conf:  confluence_lines.append('✅ Order Block + FVG confluence')
    elif fvg:    confluence_lines.append('✅ Fair Value Gap at entry')
    if bos:      confluence_lines.append('✅ Break of Structure confirmed')
    if sweep:    confluence_lines.append('✅ Liquidity sweep reversal')
    if pd_zone:  confluence_lines.append(f'✅ {pd_zone.replace("_"," ").title()} zone')
    for r in reasons[:3]:
        if r and 'MISSING' not in r:
            confluence_lines.append(f'✅ {r}')

    confluence_text = '\n'.join(confluence_lines) if confluence_lines else '— Standard setup'

    msg = f"""{dir_emoji} <b>APEX ALERT — {symbol} {dir_label}</b>
<b>Score: {score}/100</b> {badge}

📊 <b>Trade Levels</b>
Entry:    <b>{entry:,.1f}</b>
Stop:     {stop:,.1f}  ({risk_pts:.0f} pts risk)
Target 1: <b>{t1:,.1f}</b>  ({reward_pts:.0f} pts — partial exit)
Target 2: <b>{t2:,.1f}</b>  ({abs(t2-entry):.0f} pts — full target)
R:R Ratio: {rr:.1f}:1

📋 <b>Setup Details</b>
Signal:    {signal.replace('_',' ')}
Timeframe: {tf}
Session:   {session}
MTF Conf:  {conf}%

🔍 <b>Why This Setup</b>
{confluence_text}

💼 <b>Paper Trade Action</b>
→ Enter {dir_label} @ market ({entry:,.1f})
→ Set stop at {stop:,.1f}
→ Take HALF off at {t1:,.1f} — move stop to BE
→ Let remainder run to {t2:,.1f}

⚠️ Paper trade only — not financial advice"""

    return msg


def format_partial_exit_alert(position, exit_price, pnl_pts, pnl_usd):
    """Alert when T1 is hit and partial profit taken"""
    direction = position.get('direction','long')
    symbol    = position.get('symbol','NQ')
    entry     = position.get('entry_price', 0)
    t2        = position.get('target2', 0)

    msg = f"""💰 <b>PARTIAL PROFIT — {symbol}</b>

T1 Hit! Half position closed.

Entry:   {entry:,.1f}
Exit T1: <b>{exit_price:,.1f}</b>
Points:  +{pnl_pts:.1f} pts
P&L:     <b>+${pnl_usd:.2f}</b>

🔄 <b>Remaining Position</b>
Stop moved to breakeven ({entry:,.1f})
Next target: {t2:,.1f}
→ Free trade now — can only win or breakeven"""

    return msg


def format_close_alert(position, exit_price, pnl_pts, pnl_usd, reason):
    """Alert when position is fully closed"""
    direction = position.get('direction','long')
    symbol    = position.get('symbol','NQ')
    entry     = position.get('entry_price', 0)
    setup     = position.get('setup_name','')
    hold_min  = position.get('hold_minutes', 0)

    if pnl_usd > 0:
        emoji = '✅'
        result = 'WIN'
    elif pnl_usd < 0:
        emoji = '❌'
        result = 'LOSS'
    else:
        emoji = '➖'
        result = 'BREAKEVEN'

    hours = hold_min // 60
    mins  = hold_min % 60
    hold_str = f'{hours}h {mins}m' if hours > 0 else f'{mins}m'

    msg = f"""{emoji} <b>TRADE CLOSED — {symbol} {result}</b>

Entry:  {entry:,.1f}
Exit:   <b>{exit_price:,.1f}</b>
Reason: {reason.replace('_',' ').title()}
Points: {pnl_pts:+.1f} pts
P&L:    <b>${pnl_usd:+.2f}</b>
Held:   {hold_str}

Setup: {setup.replace('_',' ')}"""

    return msg


def format_daily_summary(account_state):
    """Daily P&L summary at market close"""
    bal    = account_state.get('balance', 10000)
    start  = account_state.get('starting_balance', 10000)
    ret    = account_state.get('total_return_pct', 0)
    trades = account_state.get('total_trades', 0)
    wr     = account_state.get('win_rate', 0)
    dd     = account_state.get('current_drawdown', 0)
    perf   = account_state.get('performance', {})

    emoji = '📈' if ret >= 0 else '📉'
    today = datetime.now().strftime('%A %d %B')

    msg = f"""{emoji} <b>APEX Daily Summary — {today}</b>

💰 <b>Account</b>
Balance:    ${bal:,.2f}
Return:     {ret:+.1f}%
Drawdown:   {dd:.1f}%

📊 <b>Trading Stats</b>
Total Trades: {trades}
Win Rate:     {wr:.1f}%
Profit Factor:{perf.get('profit_factor', 0):.2f}

Today's trades: checking..."""

    return msg


# =============================================================
#  ALERT QUEUE (rate limiting + deduplication)
# =============================================================

_sent_alerts = {}  # alert_key -> timestamp

def send_setup_alert(setup, force=False):
    """
    Send a setup alert with deduplication.
    Won't re-send the same setup within 30 minutes.
    """
    token, chat_id = load_telegram_config()
    if not token or not chat_id:
        return False

    # Deduplication key
    alert_key = f"{setup.get('symbol')}_{setup.get('signal')}_{setup.get('direction')}_{setup.get('entry', 0):.0f}"
    now = time.time()

    if not force and alert_key in _sent_alerts:
        if now - _sent_alerts[alert_key] < 1800:  # 30 min cooldown
            return False

    msg = format_setup_alert(setup)
    result = send_telegram(msg, token, chat_id)

    if result:
        _sent_alerts[alert_key] = now
        # Log to database
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.execute('''INSERT OR IGNORE INTO scan_log
                (ts, symbol, pattern_name, score, direction, entry, stop, target1, target2, regime, notes)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
                (int(now), setup.get('symbol'), setup.get('signal'), setup.get('score'),
                 setup.get('direction'), setup.get('entry'), setup.get('stop'),
                 setup.get('target1'), setup.get('target2'),
                 setup.get('best_regime',''), 'telegram_sent')
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    return result


def send_trade_update(update_type, data):
    """Send position update alerts (partial, close, stop)"""
    token, chat_id = load_telegram_config()
    if not token or not chat_id:
        return False

    if update_type == 'partial':
        msg = format_partial_exit_alert(
            data.get('position', {}),
            data.get('exit_price', 0),
            data.get('pnl_pts', 0),
            data.get('pnl_usd', 0)
        )
    elif update_type == 'close':
        msg = format_close_alert(
            data.get('position', {}),
            data.get('exit_price', 0),
            data.get('pnl_pts', 0),
            data.get('pnl_usd', 0),
            data.get('reason', 'manual')
        )
    else:
        return False

    return send_telegram(msg, token, chat_id)


def send_daily_summary():
    """Send end of day summary"""
    token, chat_id = load_telegram_config()
    if not token or not chat_id:
        return False
    try:
        from paper_trader import get_performance_report
        state = get_performance_report()
        msg   = format_daily_summary(state)
        return send_telegram(msg, token, chat_id)
    except Exception as e:
        logger.error(f'Daily summary error: {e}')
        return False
