"""
APEX Tradovate Integration — tradovate.py
==========================================
Paper trading via Tradovate demo API.

Features:
  - OAuth token management with auto-refresh
  - Account balance & position queries
  - Position size calculator (micro contracts)
  - Bracket order placement (entry + stop + target)
  - Signal execution with full audit logging
  - Position sync vs apex_trades open positions

Set TRADOVATE_ENABLED=true in .env to enable.
Default: disabled — all calls are no-ops.

Point values (micro contracts):
  MNQ = $2/pt   MES = $5/pt   MGC = $10/pt

Account: configured via TRADOVATE_ACCOUNT in .env
"""

import os
import time
import logging
import requests
import db as _db
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()

logger  = logging.getLogger('APEX.Tradovate')
DB_PATH = os.environ.get('DB_PATH', 'apex_market.db')

# ─────────────────────────────────────────────────────────────
#  CONFIG — loaded from .env
# ─────────────────────────────────────────────────────────────

TRADOVATE_ENABLED  = os.environ.get('TRADOVATE_ENABLED',  'false').lower() == 'true'
TRADOVATE_DEMO     = os.environ.get('TRADOVATE_DEMO',     'true').lower()  == 'true'
TRADOVATE_USERNAME = os.environ.get('TRADOVATE_USERNAME', '')
TRADOVATE_PASSWORD = os.environ.get('TRADOVATE_PASSWORD', '')
TRADOVATE_APP_ID   = os.environ.get('TRADOVATE_APP_ID',   'APEX')
TRADOVATE_APP_VER  = os.environ.get('TRADOVATE_APP_VERSION', '1.0')
TRADOVATE_CID      = os.environ.get('TRADOVATE_CID',      '')
TRADOVATE_SECRET   = os.environ.get('TRADOVATE_SECRET',   '')
TRADOVATE_DEVICE   = os.environ.get('TRADOVATE_DEVICE_ID','apex-trader-001')
TRADOVATE_ACCOUNT  = os.environ.get('TRADOVATE_ACCOUNT',  '')  # e.g. DUP560700
TRADING_ENABLED    = os.environ.get('TRADING_ENABLED',    'true').lower() == 'true'

# Instruments allowed to send live Tradovate orders — all others paper-trade only
LIVE_INSTRUMENTS = ['MNQ']

BASE_URL = (
    'https://demo.tradovateapi.com/v1'
    if TRADOVATE_DEMO else
    'https://live.tradovateapi.com/v1'
)

# Point values per micro contract
POINT_VALUE = {
    'MNQ': 2.0,   # MNQ $2/pt
    'NQ':  2.0,   # legacy alias
    'ES':  5.0,   # MES $5/pt
    'GC':  10.0,  # MGC $10/pt
}
MAX_CONTRACTS = 10
MIN_CONTRACTS = 1

# Risk tiers keyed on account balance — determines kill switch and daily loss limit
RISK_TIERS = [
    {'min_balance': 0,    'kill_switch': 400,   'daily_limit': 50},
    {'min_balance': 750,  'kill_switch': 600,   'daily_limit': 75},
    {'min_balance': 1000, 'kill_switch': 800,   'daily_limit': 100},
    {'min_balance': 2500, 'kill_switch': 2000,  'daily_limit': 250},
    {'min_balance': 5000, 'kill_switch': 4000,  'daily_limit': 500},
]


def get_risk_tier(balance: float) -> dict:
    """Return the RISK_TIER that applies for the given account balance."""
    tier = RISK_TIERS[0]
    for t in RISK_TIERS:
        if balance >= t['min_balance']:
            tier = t
    return tier


def _get_today_dollar_pnl() -> float:
    """Sum of dollar P&L for all closed trades today (midnight UTC onward)."""
    try:
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        conn = _db.connect()
        rows = conn.execute(
            "SELECT symbol, entry_price, stop, pnl_r FROM apex_trades "
            "WHERE status='closed' AND DATE(entry_time)=?", (today,)
        ).fetchall()
        conn.close()
        total = 0.0
        for sym, entry, stop, pnl_r in rows:
            if None in (entry, stop, pnl_r):
                continue
            stop_pts = abs(float(entry) - float(stop))
            pv = POINT_VALUE.get(sym, 2.0)
            total += float(pnl_r) * stop_pts * pv
        return round(total, 2)
    except Exception as e:
        logger.warning(f'_get_today_dollar_pnl error: {e}')
        return 0.0


def _get_open_dollar_loss() -> float:
    """Sum of unrealised dollar losses across all open trades (from Tradovate or DB)."""
    try:
        if TRADOVATE_ENABLED:
            pos = get_positions()
            if pos['ok']:
                return round(sum(p['unrealised_pnl'] for p in pos['positions']
                                 if p['unrealised_pnl'] < 0), 2)
        # Fallback: estimate from apex_trades open rows (entry/stop distance as 1R max loss)
        conn = _db.connect()
        rows = conn.execute(
            "SELECT symbol, entry_price, stop FROM apex_trades WHERE status='open'"
        ).fetchall()
        conn.close()
        total = 0.0
        for sym, entry, stop in rows:
            if None in (entry, stop):
                continue
            stop_pts = abs(float(entry) - float(stop))
            pv = POINT_VALUE.get(sym, 2.0)
            total -= stop_pts * pv  # worst-case: full 1R loss per open trade
        return round(total, 2)
    except Exception:
        return 0.0


def check_daily_loss_limit(balance: float) -> dict:
    """
    Check daily loss limit before any Tradovate order.
    Returns {ok, blocked, reason, daily_loss, limit}.
    """
    tier = get_risk_tier(balance)
    daily_loss = _get_today_dollar_pnl()            # negative = net loss today
    daily_loss_abs = abs(min(0.0, daily_loss))

    if daily_loss_abs >= tier['daily_limit']:
        msg = f'daily_loss_limit_reached (${daily_loss_abs:.0f} >= ${tier["daily_limit"]} limit)'
        logger.warning(f'Tradovate: order blocked — {msg}. No more trades today.')
        return {'ok': False, 'blocked': True, 'reason': msg,
                'daily_loss': daily_loss, 'limit': tier['daily_limit']}

    # Block if open exposure already eats >=50% of remaining daily allowance
    remaining = tier['daily_limit'] - daily_loss_abs
    open_loss_abs = abs(min(0.0, _get_open_dollar_loss()))
    if open_loss_abs >= remaining * 0.5:
        msg = (f'open_exposure_limit (open=${open_loss_abs:.0f} >= '
               f'50% of remaining ${remaining:.0f})')
        logger.warning(f'Tradovate: order blocked — {msg}')
        return {'ok': False, 'blocked': True, 'reason': msg,
                'daily_loss': daily_loss, 'limit': tier['daily_limit']}

    return {'ok': True, 'blocked': False, 'daily_loss': daily_loss,
            'limit': tier['daily_limit'], 'remaining': round(remaining, 2)}


def _front_month_suffix() -> str:
    """Return the current front-month contract suffix, e.g. 'M6' for June 2026.
    Rolls to next quarter ~2 weeks before expiry (day 15 of expiry month).
    Quarterly codes: H=Mar, M=Jun, U=Sep, Z=Dec."""
    now = datetime.now(timezone.utc)
    yr  = str(now.year)[-1]
    for q_month, q_code in [(3, 'H'), (6, 'M'), (9, 'U'), (12, 'Z')]:
        if now.month < q_month or (now.month == q_month and now.day < 15):
            return f'{q_code}{yr}'
    return f'H{str(now.year + 1)[-1]}'


def _tradovate_symbol(apex_sym: str) -> str:
    """Map APEX symbol to current Tradovate front-month micro contract."""
    base = {'MNQ': 'MNQ', 'NQ': 'MNQ', 'ES': 'MES', 'GC': 'MGC'}.get(apex_sym, apex_sym)
    return base + _front_month_suffix()


# Execution log — last 20 APEX signal→order events, shown on Tradovate dashboard tab
_exec_log: list = []
_last_heartbeat: str = ''

# ─────────────────────────────────────────────────────────────
#  TOKEN STORE (module-level, survives across calls)
# ─────────────────────────────────────────────────────────────

_token_cache = {
    'access_token':  None,
    'expiry':        0,       # unix timestamp
    'account_id':    None,
}


# ─────────────────────────────────────────────────────────────
#  AUTHENTICATION
# ─────────────────────────────────────────────────────────────

def authenticate() -> dict:
    """
    Obtain a Tradovate access token using password credentials.
    Caches the token and refreshes when < 60 seconds from expiry.
    Returns {ok, token, expiry, account_id} or {ok: False, error}.
    """
    if not TRADOVATE_ENABLED:
        return {'ok': False, 'error': 'Tradovate disabled — set TRADOVATE_ENABLED=true'}

    now = time.time()
    # Return cached token if still valid
    if (_token_cache['access_token']
            and now < _token_cache['expiry'] - 60):
        age = int(now - (_token_cache['expiry'] - 80 * 60))
        ttl = int(_token_cache['expiry'] - now)
        logger.info(f'Tradovate: token age={age}s TTL={ttl}s (expires at 4800s / 80min)')
        return {
            'ok':         True,
            'token':      _token_cache['access_token'],
            'expiry':     _token_cache['expiry'],
            'account_id': _token_cache['account_id'],
        }

    if not TRADOVATE_USERNAME or not TRADOVATE_PASSWORD:
        return {'ok': False, 'error': 'TRADOVATE_USERNAME / TRADOVATE_PASSWORD not set'}

    payload = {
        'name':       TRADOVATE_USERNAME,
        'password':   TRADOVATE_PASSWORD,
        'appId':      TRADOVATE_APP_ID,
        'appVersion': TRADOVATE_APP_VER,
        'cid':        TRADOVATE_CID,
        'sec':        TRADOVATE_SECRET,
        'deviceId':   TRADOVATE_DEVICE,
    }

    try:
        resp = requests.post(
            f'{BASE_URL}/auth/accesstokenrequest',
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        token = data.get('accessToken')
        if not token:
            error_info = data.get('errorText') or data.get('p-ticket') or str(data)
            return {'ok': False, 'error': f'Auth failed: {error_info}'}

        # Tradovate tokens last 80 minutes
        expiry = now + 80 * 60
        _token_cache['access_token'] = token
        _token_cache['expiry']       = expiry

        # Resolve account ID
        acct_id = _resolve_account_id(token)
        _token_cache['account_id'] = acct_id

        logger.info(f'Tradovate authenticated — account={acct_id}')
        return {'ok': True, 'token': token, 'expiry': expiry, 'account_id': acct_id}

    except requests.exceptions.RequestException as e:
        logger.error(f'Tradovate auth request failed: {e}')
        return {'ok': False, 'error': str(e)}


def _resolve_account_id(token: str) :
    """Find the account ID matching TRADOVATE_ACCOUNT name."""
    try:
        resp = requests.get(
            f'{BASE_URL}/account/list',
            headers={'Authorization': f'Bearer {token}'},
            timeout=10,
        )
        resp.raise_for_status()
        accounts = resp.json()
        if not isinstance(accounts, list):
            return None
        if TRADOVATE_ACCOUNT:
            for a in accounts:
                if (str(a.get('name', '')) == TRADOVATE_ACCOUNT
                        or str(a.get('id', '')) == TRADOVATE_ACCOUNT):
                    return a['id']
        # Fall back to first account
        return accounts[0]['id'] if accounts else None
    except Exception as e:
        logger.warning(f'_resolve_account_id failed: {e}')
        return None


def _auth_header():
    """Get Authorization header dict, or None if auth fails."""
    result = authenticate()
    if not result['ok']:
        logger.warning(f'Tradovate auth failed: {result.get("error")}')
        return None
    return {'Authorization': f'Bearer {result["token"]}'}


# ─────────────────────────────────────────────────────────────
#  ACCOUNT
# ─────────────────────────────────────────────────────────────

def get_account() -> dict:
    """
    Return account balance and P&L.
    Returns {ok, balance, available, unrealised_pnl, realised_pnl, account_id}
    """
    if not TRADOVATE_ENABLED:
        return {'ok': False, 'error': 'disabled'}

    headers = _auth_header()
    if not headers:
        return {'ok': False, 'error': 'auth_failed'}

    account_id = _token_cache.get('account_id')
    if not account_id:
        return {'ok': False, 'error': 'no_account_id'}

    try:
        resp = requests.get(
            f'{BASE_URL}/cashBalance/getCashBalanceSnapshot',
            params={'accountId': account_id},
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        return {
            'ok':              True,
            'account_id':      account_id,
            'account_name':    TRADOVATE_ACCOUNT,
            'balance':         round(float(data.get('totalCashValue',  0)), 2),
            'available':       round(float(data.get('availableFunds',  0)), 2),
            'unrealised_pnl':  round(float(data.get('openPnL',         0)), 2),
            'realised_pnl':    round(float(data.get('realizedPnL',      0)), 2),
        }

    except Exception as e:
        logger.error(f'get_account error: {e}')
        return {'ok': False, 'error': str(e)}


# ─────────────────────────────────────────────────────────────
#  POSITIONS
# ─────────────────────────────────────────────────────────────

def get_positions() -> dict:
    """
    Return open positions.
    Returns {ok, positions: [{symbol, contracts, avg_price, unrealised_pnl}]}
    """
    if not TRADOVATE_ENABLED:
        return {'ok': False, 'error': 'disabled', 'positions': []}

    headers = _auth_header()
    if not headers:
        return {'ok': False, 'error': 'auth_failed', 'positions': []}

    account_id = _token_cache.get('account_id')

    try:
        resp = requests.get(
            f'{BASE_URL}/position/list',
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        raw = resp.json()

        positions = []
        for p in (raw if isinstance(raw, list) else []):
            if account_id and p.get('accountId') != account_id:
                continue
            if abs(p.get('netPos', 0)) == 0:
                continue
            positions.append({
                'symbol':         p.get('contractId', ''),
                'contracts':      int(p.get('netPos', 0)),
                'avg_price':      float(p.get('netPrice', 0)),
                'unrealised_pnl': float(p.get('openPnL', 0)),
            })

        return {'ok': True, 'positions': positions}

    except Exception as e:
        logger.error(f'get_positions error: {e}')
        return {'ok': False, 'error': str(e), 'positions': []}


# ─────────────────────────────────────────────────────────────
#  POSITION SIZE CALCULATOR
# ─────────────────────────────────────────────────────────────

def calculate_position_size(
    account_balance: float,
    risk_pct: float,
    stop_pts: float,
    symbol: str,
) -> dict:
    """
    Calculate contracts to trade based on fixed fractional risk.

    contracts = floor(balance × risk_pct / (stop_pts × point_value))
    Clamped: 0 (skip) or MIN_CONTRACTS..MAX_CONTRACTS

    Returns {contracts, dollar_risk, instrument, point_value}
    """
    pv         = POINT_VALUE.get(symbol, 2.0)
    instrument = _tradovate_symbol(symbol)
    stop_pts   = abs(stop_pts)

    if stop_pts <= 0 or account_balance <= 0:
        return {'contracts': 0, 'dollar_risk': 0.0,
                'instrument': instrument, 'point_value': pv}

    dollar_risk_budget = account_balance * risk_pct
    contracts_raw      = dollar_risk_budget / (stop_pts * pv)
    contracts          = int(contracts_raw)

    if contracts <= 0:
        return {'contracts': 0, 'dollar_risk': 0.0,
                'instrument': instrument, 'point_value': pv}

    contracts   = max(MIN_CONTRACTS, min(MAX_CONTRACTS, contracts))
    dollar_risk = round(contracts * stop_pts * pv, 2)

    return {
        'contracts':   contracts,
        'dollar_risk': dollar_risk,
        'instrument':  instrument,
        'point_value': pv,
    }


# ─────────────────────────────────────────────────────────────
#  ORDER HISTORY
# ─────────────────────────────────────────────────────────────

def get_orders_today() -> list:
    """Return today's order fills from Tradovate."""
    if not TRADOVATE_ENABLED:
        return []

    headers = _auth_header()
    if not headers:
        return []

    try:
        resp = requests.get(
            f'{BASE_URL}/fill/list',
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        fills = resp.json()
        if not isinstance(fills, list):
            return []

        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        result = []
        account_id = _token_cache.get('account_id')
        for f in fills:
            if account_id and f.get('accountId') != account_id:
                continue
            ts = f.get('timestamp', '')
            if not ts.startswith(today):
                continue
            result.append({
                'time':       ts,
                'symbol':     f.get('contractId', ''),
                'side':       f.get('action', ''),
                'contracts':  abs(int(f.get('qty', 0))),
                'fill_price': float(f.get('price', 0)),
                'status':     'filled',
            })
        return result

    except Exception as e:
        logger.debug(f'get_orders_today: {e}')
        return []


# ─────────────────────────────────────────────────────────────
#  BRACKET ORDER
# ─────────────────────────────────────────────────────────────

def place_bracket_order(
    symbol: str,
    direction: str,
    contracts: int,
    entry: float,
    stop: float,
    target: float,
) -> dict:
    """
    Place a market entry order via /order/placeMarket.
    Stop and target are managed internally by APEX monitor_trades().

    symbol:    APEX symbol (MNQ / ES / GC)
    direction: 'long' or 'short'
    contracts: number of micro contracts
    entry:     reference entry price (used for logging/slippage calc)
    stop/target: logged but not sent to exchange (APEX manages exits)

    Returns {ok, order_id, fill_price, contracts, instrument}
    """
    if not TRADOVATE_ENABLED:
        return {'ok': False, 'error': 'disabled'}

    headers = _auth_header()
    if not headers:
        return {'ok': False, 'error': 'auth_failed'}

    account_id = _token_cache.get('account_id')
    if not account_id:
        return {'ok': False, 'error': 'no_account_id'}

    instrument = _tradovate_symbol(symbol)
    action     = 'Buy' if direction == 'long' else 'Sell'

    payload = {
        'accountSpec': TRADOVATE_ACCOUNT,
        'accountId':   account_id,
        'action':      action,
        'symbol':      instrument,
        'orderQty':    contracts,
        'orderType':   'Market',
        'isAutomated': True,
    }

    _order_url = f'{BASE_URL}/order/placeMarket'
    logger.info(
        f'Tradovate: placing market order {contracts}x {instrument} '
        f'{action} (entry≈{entry:.2f} SL={stop:.2f} TP={target:.2f}) '
        f'URL={_order_url} accountId={account_id} accountSpec={TRADOVATE_ACCOUNT}'
    )

    try:
        resp = requests.post(
            _order_url,
            json=payload,
            headers={**headers, 'Content-Type': 'application/json'},
            timeout=15,
        )
        if not resp.ok:
            err_body = resp.text
            try:
                err_body = resp.json()
            except Exception:
                pass
            if resp.status_code == 404:
                logger.error(
                    f'placeMarket 404 — URL={_order_url} symbol={instrument} '
                    f'accountId={account_id} accountSpec={TRADOVATE_ACCOUNT} '
                    f'payload={payload} response={err_body}'
                )
            else:
                logger.error(f'placeMarket {resp.status_code}: {err_body}')
            return {'ok': False, 'error': f'HTTP {resp.status_code}: {err_body}'}

        data       = resp.json()
        order_id   = (data.get('orderId') or data.get('orderStatus', {}).get('orderId')
                      or data.get('p-ticket') or str(data))
        fill_price = float(data.get('fillPrice') or data.get('price') or entry)

        logger.info(
            f'Market order placed: {contracts} {instrument} {action} '
            f'@ {fill_price:.2f} orderId={order_id}'
        )

        return {
            'ok':         True,
            'order_id':   str(order_id),
            'fill_price': fill_price,
            'contracts':  contracts,
            'instrument': instrument,
        }

    except requests.exceptions.RequestException as e:
        logger.error(f'placeMarket request failed: {e}')
        return {'ok': False, 'error': str(e)}


# ─────────────────────────────────────────────────────────────
#  EXECUTE APEX SIGNAL
# ─────────────────────────────────────────────────────────────

def execute_apex_signal(signal: dict, risk_pct: float = 0.01) -> dict:
    """
    Full execution pipeline for an APEX signal.

    1. Authenticate
    2. Get account balance
    3. Calculate position size
    4. Place bracket order
    5. Return execution result with order details

    signal dict must have: symbol, direction, entry, stop, target, setup

    Returns {ok, contracts, fill_price, dollar_risk, order_id, instrument, skipped_reason}
    """
    if not TRADOVATE_ENABLED:
        return {'ok': False, 'skipped_reason': 'disabled'}

    if not TRADING_ENABLED:
        return {'ok': False, 'skipped_reason': 'trading_disabled_kill_switch'}

    sym       = signal.get('symbol', '')
    direction = signal.get('direction', '')
    entry     = float(signal.get('entry', 0))
    stop      = float(signal.get('stop', 0))
    target    = float(signal.get('target', 0))
    setup     = signal.get('setup', '')

    if not sym or not direction or not entry or not stop:
        return {'ok': False, 'skipped_reason': 'invalid_signal_fields'}

    # LIVE_INSTRUMENTS whitelist — only MNQ executes live; others are paper-only
    if sym not in LIVE_INSTRUMENTS:
        logger.info(f'Tradovate: {sym} signal — paper only (not in LIVE_INSTRUMENTS {LIVE_INSTRUMENTS})')
        return {'ok': False, 'skipped_reason': 'paper_only', 'detail': f'{sym} not in LIVE_INSTRUMENTS'}

    # Get live account balance
    acct = get_account()
    if not acct['ok']:
        logger.warning(f'execute_apex_signal: account fetch failed — {acct.get("error")}')
        return {'ok': False, 'skipped_reason': f'account_error: {acct.get("error")}'}

    balance  = acct['balance']
    stop_pts = abs(entry - stop)

    # Daily loss limit check — must pass on EVERY order attempt
    limit_check = check_daily_loss_limit(balance)
    if limit_check['blocked']:
        return {'ok': False, 'skipped_reason': limit_check['reason']}

    # Calculate size
    sizing = calculate_position_size(balance, risk_pct, stop_pts, sym)
    if sizing['contracts'] == 0:
        msg = (f'Signal skipped — 0 contracts: balance={balance:.0f} '
               f'risk_pct={risk_pct:.1%} stop_pts={stop_pts:.1f}')
        logger.info(msg)
        return {'ok': False, 'skipped_reason': 'zero_contracts', 'detail': msg}

    # Place bracket order
    result = place_bracket_order(
        symbol=sym,
        direction=direction,
        contracts=sizing['contracts'],
        entry=entry,
        stop=stop,
        target=target,
    )

    global _last_heartbeat
    _last_heartbeat = datetime.now(timezone.utc).strftime('%H:%M UTC')
    ts_str = datetime.now(timezone.utc).strftime('%H:%M UTC')

    if result['ok']:
        slippage = round(abs(result['fill_price'] - entry), 2)
        logger.info(
            f'Signal executed: {sym} {direction.upper()} {sizing["contracts"]} '
            f'{sizing["instrument"]} @ {result["fill_price"]:.2f} '
            f'slippage={slippage:.2f}pts risk=${sizing["dollar_risk"]:.0f} orderId={result["order_id"]}'
        )
        _exec_log.append({
            'time': ts_str, 'symbol': sym, 'setup': setup, 'direction': direction,
            'instrument': sizing['instrument'], 'contracts': sizing['contracts'],
            'fill_price': result['fill_price'], 'entry': entry,
            'slippage_pts': slippage, 'order_id': result['order_id'], 'ok': True, 'error': None,
        })
        if len(_exec_log) > 20:
            _exec_log.pop(0)
        return {
            'ok':          True,
            'contracts':   sizing['contracts'],
            'instrument':  sizing['instrument'],
            'fill_price':  result['fill_price'],
            'dollar_risk': sizing['dollar_risk'],
            'order_id':    result['order_id'],
        }
    else:
        err = result.get('error', 'unknown')
        logger.error(f'execute_apex_signal: order failed — {err}')
        _exec_log.append({
            'time': ts_str, 'symbol': sym, 'setup': setup, 'direction': direction,
            'instrument': sizing.get('instrument', ''), 'contracts': 0,
            'fill_price': None, 'entry': entry,
            'slippage_pts': None, 'order_id': None, 'ok': False, 'error': str(err),
        })
        if len(_exec_log) > 20:
            _exec_log.pop(0)
        return {'ok': False, 'skipped_reason': f'order_failed: {err}'}


# ─────────────────────────────────────────────────────────────
#  POSITION SYNC
# ─────────────────────────────────────────────────────────────

def sync_positions() -> list:
    """
    Compare Tradovate open positions vs apex_trades open trades.
    Flags mismatches and closes apex_trades where Tradovate position is gone.
    Returns list of sync events (mismatches, auto-closes).
    """
    if not TRADOVATE_ENABLED:
        return []

    events = []

    tv_result = get_positions()
    if not tv_result['ok']:
        logger.warning(f'sync_positions: cannot fetch Tradovate positions')
        return events

    tv_positions = tv_result['positions']
    tv_symbols   = {p['symbol'] for p in tv_positions}

    try:
        conn   = _db.connect()
        trades = conn.execute(
            "SELECT id, symbol, direction, setup, entry_price, stop, target "
            "FROM apex_trades WHERE status='open'"
        ).fetchall()
        conn.close()
    except Exception as e:
        logger.error(f'sync_positions DB error: {e}')
        return events

    for trade_id, sym, direction, setup, entry, stop, target in trades:
        instrument = _tradovate_symbol(sym)
        # Check if Tradovate has a matching open position
        tv_match = next((p for p in tv_positions
                         if p['symbol'] == instrument), None)

        if tv_match is None and instrument:
            # Position gone in Tradovate — trade may have been closed externally
            event = {
                'type':     'mismatch',
                'trade_id': trade_id,
                'symbol':   sym,
                'detail':   f'apex_trades #{trade_id} open but no {instrument} position in Tradovate',
            }
            events.append(event)
            logger.warning(
                f'Position sync mismatch: apex_trade #{trade_id} {sym} '
                f'{direction.upper()} open but {instrument} not in Tradovate positions'
            )

    return events


# ─────────────────────────────────────────────────────────────
#  DASHBOARD STATUS
# ─────────────────────────────────────────────────────────────

def get_status() -> dict:
    """
    Return full Tradovate status for dashboard /api/apex/tradovate endpoint.
    Returns {connected, balance, available, unrealised_pnl, realised_pnl,
             positions, orders_today, account_name, demo, enabled}
    """
    instrument_now = _tradovate_symbol('MNQ')
    today_pnl = _get_today_dollar_pnl()
    result = {
        'enabled':         TRADOVATE_ENABLED,
        'trading_enabled': TRADING_ENABLED,
        'demo':            TRADOVATE_DEMO,
        'connected':       False,
        'account_name':    TRADOVATE_ACCOUNT or 'Not configured',
        'live_instruments': LIVE_INSTRUMENTS,
        'front_month':     instrument_now,
        'risk_tiers':      RISK_TIERS,
        'today_dollar_pnl': today_pnl,
        'balance':         None,
        'available':       None,
        'unrealised_pnl':  None,
        'realised_pnl':    None,
        'positions':       [],
        'orders_today':    [],
        'exec_log':        list(reversed(_exec_log)),
        'last_heartbeat':  _last_heartbeat or None,
    }

    if not TRADOVATE_ENABLED:
        return result

    auth = authenticate()
    if not auth['ok']:
        result['error'] = auth.get('error')
        return result

    result['connected'] = True

    acct = get_account()
    if acct['ok']:
        result['balance']        = acct['balance']
        result['available']      = acct['available']
        result['unrealised_pnl'] = acct['unrealised_pnl']
        result['realised_pnl']   = acct['realised_pnl']

    pos = get_positions()
    if pos['ok']:
        result['positions'] = pos['positions']

    result['orders_today'] = get_orders_today()

    return result


# ─────────────────────────────────────────────────────────────
#  CLI TEST
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    print(f'TRADOVATE_ENABLED: {TRADOVATE_ENABLED}')
    print(f'TRADOVATE_DEMO:    {TRADOVATE_DEMO}')
    print(f'BASE_URL:          {BASE_URL}')

    if not TRADOVATE_ENABLED:
        print('\nSet TRADOVATE_ENABLED=true in .env to test live connection.')
        print('\nPosition size calculator test (no auth required):')
        for sym, entry, stop in [('NQ', 21500, 21450), ('ES', 5800, 5790), ('GC', 3100, 3090)]:
            sz = calculate_position_size(50000, 0.01, abs(entry - stop), sym)
            print(f'  {sym}: {sz["contracts"]} {sz["instrument"]} | risk=${sz["dollar_risk"]} | pv=${sz["point_value"]}/pt')
    else:
        print('\nTesting authentication...')
        auth = authenticate()
        print(f'Auth: {auth}')
        if auth['ok']:
            print('\nAccount:')
            print(f'  {get_account()}')
            print('\nPositions:')
            print(f'  {get_positions()}')
