"""
APEX Strategy Router — strategy_router.py
==========================================
The brain of APEX v2.

Detects current market conditions and routes to the
appropriate strategy mode:

  Mode 1 — Scalp:       Fast moves, 5-30min holds
  Mode 2 — Swing:       Structure trades, 1-4hr holds
  Mode 3 — Mean Rev:    Fade extremes, 15-90min holds
  Mode 4 — Daily:       Overnight/multi-day holds (future)

Routing logic:
  - Check session window (London/NY only)
  - Check news blackout
  - Check VIX regime
  - Detect market condition (trending/ranging/volatile)
  - Run appropriate strategy scanner(s)
  - Return best setup across all modes

Multiple setups can fire simultaneously — router ranks
them by score and returns the best one(s) up to max
concurrent trades.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Optional, Dict, List, Any
import db as _db

logger = logging.getLogger('APEX.Router')
NY_TZ  = ZoneInfo('America/New_York')


# =============================================================
#  DATA LOADING
# =============================================================

def load_dataframes(symbol: str, limits: Dict[str, int] = None) -> Dict[str, Optional[pd.DataFrame]]:
    """Load all timeframes for a symbol from the database"""
    if limits is None:
        limits = {
            '1min':  500,
            '5min':  1000,
            '15min': 800,
            '1hour': 500,
            '4hour': 300,
            '1day':  500,
            '1week': 200,
        }

    dfs = {}
    try:
        conn = _db.connect()
        for tf, limit in limits.items():
            try:
                df = _db.read_sql(
                    'SELECT ts,open,high,low,close,volume FROM ohlcv '
                    'WHERE symbol=? AND timeframe=? ORDER BY ts DESC LIMIT ?',
                    conn, params=(symbol, tf, limit)
                )
                if len(df) >= 10:
                    df = df.iloc[::-1].reset_index(drop=True)
                    df['dt'] = pd.to_datetime(df['ts'], unit='s', utc=True)
                    df.set_index('dt', inplace=True)
                    for col in ['open','high','low','close','volume']:
                        df[col] = df[col].astype(float)
                    dfs[tf] = df
                else:
                    dfs[tf] = None
            except Exception:
                dfs[tf] = None
        conn.close()
    except Exception as e:
        logger.error(f'DB load error: {e}')

    return dfs


def get_latest_vix() -> Optional[float]:
    """Get latest VIX close from database"""
    try:
        conn = _db.connect()
        row  = conn.execute(
            "SELECT close FROM ohlcv WHERE symbol='VIX' AND timeframe='1day' "
            "ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        conn.close()
        return float(row[0]) if row else None
    except Exception:
        return None


# =============================================================
#  MAIN ROUTER
# =============================================================

def run_strategy_router(
    symbol:             str = 'NQ',
    min_score:          int = 55,
    alert:              bool = True,
    paper_trade:        bool = True,
    allowed_modes:      list = None,
    skip_session_check: bool = False,
) -> Optional[Dict]:
    """
    Main entry point. Runs all strategy modes and returns the best setup.

    Returns dict with setup details or None if no qualifying setup.
    skip_session_check: set True when caller already validated session.
    """
    try:
        from strategy_config import is_tradeable_session, STRATEGY
        from news_calendar import should_trade
        from market_structure import get_market_condition

        dt_ny = datetime.now(timezone.utc).astimezone(NY_TZ)

        # Gate 1: Session check (skip if caller already validated)
        in_session, sess_name, sess_quality = is_tradeable_session(dt_ny)
        if not skip_session_check and not in_session:
            return None
        if skip_session_check:
            sess_name, sess_quality = "Active", 90

        # Gate 2: News blackout
        news_ok, news_reason = should_trade(dt_ny)
        if not news_ok:
            logger.info(f'[{symbol}] Blocked — {news_reason}')
            return None

        # Gate 3: VIX check
        vix = get_latest_vix()
        if vix and vix > STRATEGY['max_vix']:
            logger.info(f'[{symbol}] VIX {vix:.1f} > {STRATEGY["max_vix"]} — skipping')
            return None

        # Gate 4: Max concurrent trades
        if paper_trade:
            try:
                from paper_trader import get_account_state
                state     = get_account_state()
                open_pos  = state.get('open_positions', [])
                total_risk = sum(p.get('risk_pct', 2) for p in open_pos)
                if total_risk >= STRATEGY['max_total_risk']:
                    logger.info(f'[{symbol}] Max risk reached ({total_risk:.1f}%)')
                    return None
            except Exception:
                pass

        # Load all data
        dfs = load_dataframes(symbol)
        if not dfs.get('15min') and not dfs.get('5min'):
            return None

        # Detect market condition
        mc = get_market_condition(dfs.get('15min') or dfs.get('5min'))

        # Route to strategy modes based on condition and session
        setups = []

        # Filter modes per instrument (e.g. ES swing only)
        if allowed_modes is None:
            from strategy_config import get_instrument_modes
            allowed_modes = get_instrument_modes(symbol)

        # Always try swing in prime sessions
        if sess_quality >= 70 and 'swing' in allowed_modes:
            try:
                from strategy_swing import scan_swing
                swing = scan_swing(
                    df_5min  = dfs.get('5min'),
                    df_15min = dfs.get('15min'),
                    df_1hour = dfs.get('1hour'),
                    df_4hour = dfs.get('4hour'),
                    df_1day  = dfs.get('1day'),
                    df_1week = dfs.get('1week'),
                    symbol   = symbol,
                    min_score= min_score,
                )
                if swing:
                    setups.append(('swing', swing))
            except Exception as e:
                logger.debug(f'Swing scan error: {e}')

        # Scalp in NY open or London prime
        if ('NY Open' in sess_name or 'London' in sess_name) and 'scalp' in allowed_modes:
            try:
                from strategy_scalp import scan_scalp
                scalp = scan_scalp(
                    df_1min  = dfs.get('1min'),
                    df_5min  = dfs.get('5min'),
                    df_15min = dfs.get('15min'),
                    df_1hour = dfs.get('1hour'),
                    df_4hour = dfs.get('4hour'),
                    df_1day  = dfs.get('1day'),
                    symbol   = symbol,
                    min_score= min_score + 5,  # Higher bar for scalps
                )
                if scalp:
                    setups.append(('scalp', scalp))
            except Exception as e:
                logger.debug(f'Scalp scan error: {e}')

        # Mean reversion in ranging markets
        if (mc.condition == 'ranging' or mc.condition == 'transitioning') and 'meanrev' in allowed_modes:
            try:
                from strategy_meanrev import scan_meanrev
                mr = scan_meanrev(
                    df_5min  = dfs.get('5min'),
                    df_15min = dfs.get('15min'),
                    df_1hour = dfs.get('1hour'),
                    symbol   = symbol,
                    min_score= min_score + 10,  # Higher bar for mean rev
                )
                if mr:
                    setups.append(('meanrev', mr))
            except Exception as e:
                logger.debug(f'MeanRev scan error: {e}')

        if not setups:
            logger.info(f'[{symbol}] No qualifying setup')
            return None

        # Pick best setup by score
        best_mode, best_setup = max(setups, key=lambda x: x[1].score)

        # Build result dict
        result = _build_result(symbol, best_mode, best_setup, sess_name, vix, mc)

        logger.info(
            f'🎯 [{symbol}] {best_mode.upper()} {result["direction"].upper()} '
            f'score={result["score"]}% entry={result["entry"]} '
            f'stop={result["stop"]} t1={result["target1"]}'
        )

        # Send Telegram alert
        if alert:
            try:
                _send_alert(result)
            except Exception as e:
                logger.debug(f'Alert error: {e}')

        # Open paper trade
        if paper_trade:
            try:
                _open_paper_trade(result)
            except Exception as e:
                logger.debug(f'Paper trade error: {e}')

        return result

    except Exception as e:
        logger.error(f'Router error: {e}')
        return None


# =============================================================
#  RESULT BUILDER
# =============================================================

def _build_result(symbol, mode, setup, sess_name, vix, mc) -> Dict:
    """Convert setup dataclass to standard result dict"""
    result = {
        'symbol':     symbol,
        'mode':       mode,
        'direction':  setup.direction,
        'score':      setup.score,
        'score_pct':  setup.score,
        'entry':      setup.entry,
        'stop':       setup.stop,
        'target1':    setup.target1,
        'target2':    setup.target2,
        'risk_pct':   setup.risk_pct,
        'risk_pts':   setup.risk_pts,
        'rr_ratio':   setup.rr_ratio,
        'timeframe':  setup.timeframe,
        'reason':     setup.reason,
        'confluences':setup.confluences,
        'warnings':   getattr(setup, 'warnings', []),
        'session':    sess_name,
        'vix':        vix,
        'market_condition': mc.condition,
        'timestamp':  datetime.now(timezone.utc).isoformat(),
    }

    # Add target3 if swing setup
    if hasattr(setup, 'target3'):
        result['target3'] = setup.target3
    if hasattr(setup, 'hold_overnight'):
        result['hold_overnight'] = setup.hold_overnight

    return result


# =============================================================
#  TELEGRAM ALERT
# =============================================================

def _send_alert(result: Dict):
    """Send formatted Telegram alert for v2 setup"""
    try:
        from telegram_alerts import send_telegram
        from news_calendar import get_news_summary_line

        mode_emoji = {
            'scalp':   '⚡',
            'swing':   '🎯',
            'meanrev': '↩️',
            'daily':   '📈',
        }.get(result['mode'], '🎯')

        dir_emoji = '🟢 LONG' if result['direction'] in ('long','bullish') else '🔴 SHORT'

        confluences_str = '\n'.join(f'  • {c}' for c in result.get('confluences', [])[:5])
        warnings_str    = ''
        if result.get('warnings'):
            warnings_str = '\n⚠️ ' + ' | '.join(result['warnings'][:2])

        risk_pts  = result.get('risk_pts', 0)
        t1_pts    = abs(result['target1'] - result['entry'])
        rr_actual = round(t1_pts / risk_pts, 1) if risk_pts > 0 else 0

        msg = (
            f"{mode_emoji} APEX {result['mode'].upper()} SETUP\n"
            f"{'='*32}\n"
            f"{dir_emoji} {result['symbol']}\n"
            f"Score: {result['score']}/100\n"
            f"Session: {result['session']}\n"
            f"\n📊 LEVELS\n"
            f"Entry:    {result['entry']:,.0f}\n"
            f"Stop:     {result['stop']:,.0f} ({risk_pts:.0f}pts)\n"
            f"Target 1: {result['target1']:,.0f} ({t1_pts:.0f}pts | {rr_actual}R)\n"
            f"Target 2: {result.get('target2', 0):,.0f}\n"
        )

        if result.get('target3'):
            msg += f"Target 3: {result['target3']:,.0f}\n"

        msg += (
            f"\n💡 CONFLUENCES\n{confluences_str}\n"
            f"\n⚙️ Risk: {result['risk_pct']}% | "
            f"Market: {result.get('market_condition','?')}\n"
        )

        if warnings_str:
            msg += warnings_str + '\n'

        msg += f"\n{get_news_summary_line()}"

        if result.get('hold_overnight'):
            msg += '\n🌙 Hold overnight — HTF target'

        send_telegram(msg)

    except Exception as e:
        logger.debug(f'Alert send error: {e}')


# =============================================================
#  PAPER TRADE
# =============================================================

def _open_paper_trade(result: Dict):
    """Open paper trade for v2 setup"""
    try:
        from paper_trader import open_position
        open_position({
            'symbol':    result['symbol'],
            'direction': result['direction'],
            'entry':     result['entry'],
            'stop':      result['stop'],
            'target1':   result['target1'],
            'target2':   result.get('target2', result['target1']),
            'risk_pct':  result['risk_pct'],
            'mode':      result['mode'],
            'score':     result['score'],
            'reason':    result['reason'],
        })
    except Exception as e:
        logger.debug(f'Paper trade error: {e}')


# =============================================================
#  DAILY TRADE LIMIT
# =============================================================

_daily_trade_counts: Dict[str, int] = {}
_last_trade_date: str = ''

def check_daily_limit(symbol: str) -> bool:
    """Returns True if we can still trade today"""
    global _daily_trade_counts, _last_trade_date

    try:
        from strategy_config import STRATEGY
        today = datetime.now(timezone.utc).astimezone(NY_TZ).strftime('%Y-%m-%d')

        # Reset counts on new day
        if today != _last_trade_date:
            _daily_trade_counts = {}
            _last_trade_date    = today

        count = _daily_trade_counts.get(symbol, 0)
        return count < STRATEGY['max_trades_day']
    except Exception:
        return True


def record_trade(symbol: str):
    """Record that a trade was taken today"""
    global _daily_trade_counts
    _daily_trade_counts[symbol] = _daily_trade_counts.get(symbol, 0) + 1
