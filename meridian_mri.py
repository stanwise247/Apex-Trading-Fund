"""
Meridian MRI — standalone market-intelligence scoring engine for ES/MNQ.

Synthesises macro conditions, ICT price structure, multi-timeframe trend, and
live news into a composite -10..+10 "MRI" score. Never imports server.py
(server.py imports this module — the reverse would be circular). All fetch
functions degrade gracefully (never raise) so a missing model/feed/data point
renders as "unavailable" rather than a fabricated number.

See docs/meridian_mri.md for scoring methodology and how to adjust weights,
and docs/meridian_news.md for the news pipeline.
"""

import os
import re
import json
import time
import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests
import numpy as np
import pandas as pd

from market_structure import load_bars
from liquidity_engine import analyse_liquidity, calc_atr
from liquidity_sweep import find_equal_levels
from setup_j_value_area import get_setup_j_state
from setup_h_vwap import get_h_state
import meridian_l3
import regime_engine

logger = logging.getLogger('MERIDIAN')


# =============================================================
#  CONFIG / CONSTANTS
# =============================================================

def _load_config():
    try:
        with open('config.json') as f:
            return json.load(f)
    except Exception:
        return {}

_cfg = _load_config()
ANTHROPIC_KEY   = os.environ.get('ANTHROPIC_KEY', _cfg.get('anthropic_key', ''))
# claude-sonnet-4-20250514 (originally chosen to match server.py's existing
# convention) was confirmed 404/not_found_error on Railway's key as of
# 2026-07-23 — deprecated/retired. claude-sonnet-5 confirmed working live.
ANTHROPIC_MODEL = 'claude-sonnet-5'

SYMBOLS = ('ES', 'MNQ')

# (label, db_timeframe, resample_rule_or_None, bars_to_load) — mirrors
# server.py's /api/apex/mtf _TFS list exactly (kept as an independent copy
# here to avoid importing server.py, which would be circular).
MTF_TIMEFRAMES = [
    ('1m',  '1min',  None,    120),
    ('5m',  '5min',  None,    100),
    ('15m', '15min', None,     80),
    ('30m', '5min',  '30min', 400),
    ('1h',  '1hour', None,     60),
    ('4h',  '4hour', None,     50),
]
MTF_TABLE_ROWS   = ['Monthly', 'Weekly', 'Daily', '4H', '1H', '15M', '5M', '1M']
MTF_TABLE_TF_MAP = {'4H': '4hour', '1H': '1hour', '15M': '15min', '5M': '5min', '1M': '1min'}
WEEKLY_STALE_DAYS = 14

WEIGHTS_SHORT  = {'macro': 0.15, 'regime': 0.30, 'ict': 0.25, 'mtf': 0.20, 'news': 0.10}
WEIGHTS_MEDIUM = {'macro': 0.30, 'regime': 0.20, 'ict': 0.20, 'mtf': 0.25, 'news': 0.05}

NEWS_WINDOW_SECONDS = 3 * 3600
EQUAL_LEVELS_LOOKBACK = 400   # ~5 sessions of 5min bars

# Bloomberg discontinued free public RSS years ago (no auth-free feed exists
# today) — substituted with equivalent, already-proven-working free feeds.
# See docs/meridian_news.md for the substitution rationale.
NEWS_RSS_FEEDS = [
    ('Reuters Business',        'macro',       'https://feeds.reuters.com/reuters/businessNews'),
    ('Reuters World',           'macro',       'https://feeds.reuters.com/reuters/worldNews'),
    ('CNBC Markets',            'markets',     'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069'),
    ('CNBC Markets (Top News)', 'markets',     'https://www.cnbc.com/id/20910258/device/rss/rss.html'),
    ('Yahoo Finance',           'markets',     'https://finance.yahoo.com/news/rssindex'),
    ('Investing.com Commodities', 'commodities', 'https://www.investing.com/rss/commodities.rss'),
    ('MarketWatch',              'markets',     'https://feeds.marketwatch.com/marketwatch/topstories/'),
    ('WSJ Economy',              'macro',       'https://feeds.a.dj.com/rss/RSSEconomics.xml'),
]


def _ema20(series: pd.Series) -> pd.Series:
    return series.ewm(span=20, adjust=False).mean()


# =============================================================
#  A. PURE SCORING FUNCTIONS — no IO, unit-testable with literal inputs
# =============================================================

def vix_subscore(vix_level):
    if vix_level is None:
        return 0
    if vix_level < 15:
        return 2
    if vix_level < 20:
        return 0
    if vix_level < 25:
        return -1
    return -2


def dxy_subscore(trend):
    return {'rising': -1, 'falling': 1}.get(trend, 0)


def yield_subscore(direction):
    return {'rising_sharply': -1, 'stable_or_falling': 1}.get(direction, 0)


def oil_subscore(chg_pct_5d):
    if chg_pct_5d is None:
        return 0
    if chg_pct_5d > 1.5:
        return -1
    if chg_pct_5d < -1.5:
        return 1
    return 0


def macro_layer_score(subscores: dict) -> float:
    """subscores: {'vix':-2..2, 'dxy':-1..1, 'yield':-1..1, 'oil':-1..1} — any may be None."""
    vals = [v for v in subscores.values() if v is not None]
    if not vals:
        return 0.0
    return round(float(np.clip(np.mean(vals) * 5, -10, 10)), 2)


def regime_strength_subscore(regime, confidence):
    if regime == 'TRENDING':
        return 3 if (confidence or 0) >= 0.50 else 1
    if regime == 'CHOPPY':
        return 0
    if regime == 'MEAN_REVERTING':
        return -1
    return 0


def l3_normalize(prob):
    if prob is None:
        return 0.0
    return float(np.clip(2 * prob - 1, -1, 1))


def regime_momentum_score(regime, confidence, l3_prob, htf_bias):
    """
    Regime/L3 measure trend *strength*, not direction — there's no schema field
    for "regime is bullish." Direction is derived from htf_bias (documented
    assumption, see docs/meridian_mri.md). L3 probability modulates magnitude.
    """
    base = regime_strength_subscore(regime, confidence)
    l3n  = l3_normalize(l3_prob)
    direction = {'BULLISH': 1, 'BEARISH': -1}.get(htf_bias, 0)
    if direction == 0:
        return 0.0
    magnitude = base * (1 + 0.7 * l3n)
    return float(np.clip(magnitude * direction, -10, 10))


def ict_structure_score(price, vah, val, active_fvgs, equal_levels):
    if price is None:
        return 0.0
    score = 0.0
    if vah is not None and val is not None:
        if price > vah:
            score += 2
        elif price < val:
            score -= 2

    bull_above, bear_below = 0, 0
    for f in (active_fvgs or []):
        mid = (f.get('high', 0) + f.get('low', 0)) / 2
        if f.get('kind') == 'bull' and price > mid:
            bull_above += 1
        elif f.get('kind') == 'bear' and price < mid:
            bear_below += 1
    score += min(bull_above, 3)
    score -= min(bear_below, 3)

    for lvl in (equal_levels or []):
        lp = lvl.get('price')
        if lp is None or not price:
            continue
        dist_pct = abs(lp - price) / price * 100
        if dist_pct <= 0.3:
            if lvl.get('type') == 'high':
                score += 1
            elif lvl.get('type') == 'low':
                score -= 1

    return float(np.clip(score, -10, 10))


def mtf_trend_score(biases: dict) -> float:
    total, counted = 0, 0
    for b in (biases or {}).values():
        if b == 'BULLISH':
            total += 1; counted += 1
        elif b == 'BEARISH':
            total -= 1; counted += 1
        elif b == 'NEUTRAL':
            counted += 1
        # UNKNOWN excluded from denominator entirely
    if counted == 0:
        return 0.0
    return float(np.clip(total / counted * 10, -10, 10))


def pct_bullish(biases: dict) -> float:
    vals = list((biases or {}).values())
    if not vals:
        return 0.0
    bull = sum(1 for b in vals if b == 'BULLISH')
    return round(bull / len(vals) * 100, 1)


def pct_alignment(biases: dict) -> float:
    """% of timeframes agreeing on direction — excludes insufficient-history/UNKNOWN rows."""
    vals = [b for b in (biases or {}).values() if b in ('BULLISH', 'BEARISH')]
    if not vals:
        return 0.0
    bull = sum(1 for b in vals if b == 'BULLISH')
    bear = len(vals) - bull
    return round(max(bull, bear) / len(vals) * 100, 1)


def news_layer_score(items, now_ts=None) -> float:
    if now_ts is None:
        now_ts = time.time()
    score = 0
    for it in (items or []):
        if it.get('relevance') != 'High':
            continue
        ts = it.get('timestamp')
        if ts is None or now_ts - ts > NEWS_WINDOW_SECONDS:
            continue
        d = it.get('direction')
        if d == 'Bullish':
            score += 2
        elif d == 'Bearish':
            score -= 2
    return float(np.clip(score, -6, 6))


def composite_score(layers: dict, weights: dict) -> float:
    total = sum(layers.get(k, 0) * w for k, w in weights.items())
    return round(float(np.clip(total, -10, 10)), 1)


def mri_label(composite) -> str:
    if composite >= 5:
        return 'BULLISH'
    if composite >= 2:
        return 'CAUTIOUSLY BULLISH'
    if composite > -2:
        return 'NEUTRAL'
    if composite > -5:
        return 'CAUTIOUSLY BEARISH'
    return 'BEARISH'


# =============================================================
#  B. IO / FETCH FUNCTIONS — always return a safe fallback, never raise
# =============================================================

def fetch_regime(symbol: str) -> dict:
    try:
        return regime_engine.get_current_regime(symbol)
    except Exception as e:
        return {'symbol': symbol, 'regime': None, 'error': str(e)}


def fetch_l3(symbol: str) -> float:
    try:
        return meridian_l3.predict(symbol)
    except Exception:
        return 0.5


def fetch_l3_multiplier(symbol: str, hurst=0.5, conf=0.5):
    try:
        return meridian_l3.get_position_multiplier(symbol, hurst or 0.5, conf or 0.5)
    except Exception:
        return (1.0, 0.5)


def fetch_htf_bias(symbol: str) -> str:
    """get_h_state returns lowercase 'bullish'/'bearish'/'neutral' — normalise to upper."""
    try:
        h = get_h_state(symbol)
        return (h.get('htf_bias') or 'NEUTRAL').upper()
    except Exception:
        return 'NEUTRAL'


def fetch_value_area(symbol: str) -> dict:
    try:
        j = get_setup_j_state(symbol)
        return {'vah': j.get('vah'), 'val': j.get('val'), 'in_session': j.get('in_session')}
    except Exception:
        return {'vah': None, 'val': None, 'in_session': False}


def _fvg_to_dict(f):
    return {'kind': f.kind, 'high': round(f.high, 2), 'low': round(f.low, 2),
            'size': round(f.size, 2), 'filled': f.filled}


def _ob_to_dict(o):
    return {'kind': o.kind, 'high': round(o.high, 2), 'low': round(o.low, 2), 'broken': o.broken}


def fetch_liquidity(symbol: str, timeframe: str = '1hour') -> dict:
    try:
        data = analyse_liquidity(symbol, timeframe)
        return {
            'active_fvgs': [_fvg_to_dict(f) for f in data['active_fvgs']],
            'active_obs':  [_ob_to_dict(o) for o in data['active_obs']],
        }
    except Exception as e:
        logger.debug(f'fetch_liquidity {symbol} error: {e}')
        return {'active_fvgs': [], 'active_obs': []}


def fetch_equal_levels(symbol: str, timeframe: str = '5min', lookback: int = EQUAL_LEVELS_LOOKBACK) -> list:
    try:
        df = load_bars(symbol, timeframe, limit=lookback)
        levels = find_equal_levels(df, lookback=lookback, tolerance_pct=0.1)
        # find_equal_levels returns numpy float64 prices — cast to native float for JSON safety
        for lv in levels:
            lv['price'] = float(lv['price'])
        return levels
    except Exception as e:
        logger.debug(f'fetch_equal_levels {symbol} error: {e}')
        return []


def fetch_mtf_biases(symbol: str) -> dict:
    biases = {}
    for label, db_tf, resample, n in MTF_TIMEFRAMES:
        try:
            df = load_bars(symbol, db_tf, limit=n)
            if resample:
                df = df[['open', 'high', 'low', 'close', 'volume']].resample(resample).agg(
                    {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
                ).dropna(subset=['close'])
            if len(df) < 5:
                biases[label] = 'UNKNOWN'
                continue
            closes = df['close'].astype(float)
            ema20  = float(_ema20(closes).iloc[-1])
            last   = float(closes.iloc[-1])
            biases[label] = 'BULLISH' if last > ema20 else 'BEARISH'
        except Exception:
            biases[label] = 'UNKNOWN'
    return biases


def fetch_price(symbol: str) -> dict:
    try:
        df1   = load_bars(symbol, '1min', limit=5)
        price = float(df1['close'].iloc[-1])
    except Exception:
        return {'price': None, 'change_pts': None, 'change_pct': None}
    try:
        dfd = load_bars(symbol, '1day', limit=3)
        prior_close = float(dfd['close'].iloc[-2]) if len(dfd) >= 2 else float(dfd['close'].iloc[-1])
        change_pts  = round(price - prior_close, 2)
        change_pct  = round(change_pts / prior_close * 100, 3) if prior_close else None
    except Exception:
        change_pts, change_pct = None, None
    return {'price': round(price, 2), 'change_pts': change_pts, 'change_pct': change_pct}


def fetch_atr_vs_avg(symbol: str, timeframe: str = '15min') -> dict:
    try:
        df = load_bars(symbol, timeframe, limit=200)
        atr_series = calc_atr(df, 14)
        atr14 = float(atr_series.iloc[-1])
        avg20 = float(atr_series.dropna().tail(20).mean())
        expanding = (atr14 > avg20) if (not np.isnan(atr14) and not np.isnan(avg20)) else None
        return {
            'atr14': round(atr14, 3) if not np.isnan(atr14) else None,
            'avg20': round(avg20, 3) if not np.isnan(avg20) else None,
            'expanding': expanding,
        }
    except Exception:
        return {'atr14': None, 'avg20': None, 'expanding': None}


def _pdh_pdl(symbol: str):
    try:
        df = load_bars(symbol, '1day', limit=3)
        row = df.iloc[-2] if len(df) >= 2 else df.iloc[-1]
        return round(float(row['high']), 2), round(float(row['low']), 2)
    except Exception:
        return None, None


# --- macro (yfinance) --------------------------------------------------

def _yf_quote(ticker):
    try:
        import yfinance as yf
        info  = yf.Ticker(ticker).fast_info
        price = info.last_price or info.previous_close
        return float(price) if price else None
    except Exception as e:
        logger.debug(f'yf quote failed {ticker}: {e}')
        return None


def _yf_pct_change(ticker, days=5):
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period=f'{days}d')
        if hist.empty or len(hist) < 2:
            return None
        first, last = float(hist['Close'].iloc[0]), float(hist['Close'].iloc[-1])
        return round((last - first) / first * 100, 3) if first else None
    except Exception as e:
        logger.debug(f'yf history failed {ticker}: {e}')
        return None


def _yf_abs_change(ticker, days=5):
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period=f'{days}d')
        if hist.empty or len(hist) < 2:
            return None
        return round(float(hist['Close'].iloc[-1] - hist['Close'].iloc[0]), 2)
    except Exception:
        return None


def _yf_trend(ticker, threshold_pct=0.15):
    pct = _yf_pct_change(ticker, days=5)
    if pct is None:
        return 'flat'
    if pct > threshold_pct:
        return 'rising'
    if pct < -threshold_pct:
        return 'falling'
    return 'flat'


def _vix_interpretation(vix):
    if vix is None:
        return 'Data unavailable.'
    if vix < 15:
        return f'VIX {vix:.1f} — Low. Calm conditions, typically supportive for trend-continuation setups.'
    if vix < 20:
        return f'VIX {vix:.1f} — Elevated. Typically a mild headwind for momentum setups.'
    if vix < 25:
        return f'VIX {vix:.1f} — High. Choppier price action likely; favor caution on breakout entries.'
    return f'VIX {vix:.1f} — Extreme. Elevated risk of violent moves in both directions.'


def _dxy_interpretation(dxy, trend):
    if dxy is None:
        return 'Data unavailable.'
    if trend == 'rising':
        return f'DXY {dxy:.2f} — rising. A stronger dollar is typically a headwind for ES/MNQ.'
    if trend == 'falling':
        return f'DXY {dxy:.2f} — falling. A weaker dollar is typically supportive for ES/MNQ.'
    return f'DXY {dxy:.2f} — flat. Limited directional pressure from the dollar.'


def _yield_interpretation(y, direction):
    if y is None:
        return 'Data unavailable.'
    if direction == 'rising_sharply':
        return f'10Y {y:.2f}% — rising sharply. Higher discount rates typically pressure valuations, especially growth/tech (MNQ).'
    if direction == 'stable_or_falling':
        return f'10Y {y:.2f}% — stable/falling. Supportive backdrop for equity valuations.'
    return f'10Y {y:.2f}% — mixed signal.'


def _oil_interpretation(oil, chg_pct):
    if oil is None:
        return 'Data unavailable.'
    chg_txt = f'{chg_pct:+.1f}% 5d' if chg_pct is not None else 'change n/a'
    return f'WTI ${oil:.2f} ({chg_txt}) — relevant given ongoing Middle East / Strait of Hormuz supply-risk context.'


def fetch_macro() -> dict:
    macro = {}

    vix     = _yf_quote('^VIX')
    vix_chg = _yf_abs_change('^VIX', days=5)
    vix_regime = None
    if vix is not None:
        vix_regime = 'Low' if vix < 15 else 'Elevated' if vix < 20 else 'High' if vix < 25 else 'Extreme'
    macro['vix'] = {
        'value': round(vix, 2) if vix is not None else None,
        'change_5d': vix_chg,
        'regime': vix_regime,
        'sub_score': vix_subscore(vix) if vix is not None else None,
        'available': vix is not None,
        'interpretation': _vix_interpretation(vix),
    }

    dxy_val   = _yf_quote('DX-Y.NYB')
    dxy_trend = _yf_trend('DX-Y.NYB') if dxy_val is not None else None
    macro['dxy'] = {
        'value': round(dxy_val, 2) if dxy_val is not None else None,
        'trend': dxy_trend,
        'sub_score': dxy_subscore(dxy_trend) if dxy_val is not None else None,
        'available': dxy_val is not None,
        'interpretation': _dxy_interpretation(dxy_val, dxy_trend),
    }

    yield_val  = _yf_quote('^TNX')
    yield_chg  = _yf_pct_change('^TNX', days=5) if yield_val is not None else None
    yield_dir  = None
    if yield_chg is not None:
        yield_dir = 'rising_sharply' if yield_chg > 3.0 else ('stable_or_falling' if yield_chg <= 0 else 'mixed')
    macro['yield_10y'] = {
        'value': round(yield_val, 2) if yield_val is not None else None,
        'change_pct_5d': round(yield_chg, 2) if yield_chg is not None else None,
        'direction': yield_dir,
        'sub_score': yield_subscore(yield_dir) if yield_dir else None,
        'available': yield_val is not None,
        'interpretation': _yield_interpretation(yield_val, yield_dir),
    }

    oil_val = _yf_quote('CL=F')
    oil_chg = _yf_pct_change('CL=F', days=5) if oil_val is not None else None
    macro['oil'] = {
        'value': round(oil_val, 2) if oil_val is not None else None,
        'change_pct_5d': round(oil_chg, 2) if oil_chg is not None else None,
        'sub_score': oil_subscore(oil_chg) if oil_chg is not None else None,
        'available': oil_val is not None,
        'interpretation': _oil_interpretation(oil_val, oil_chg),
    }

    # No reliable free, documented, no-auth API exists for Put/Call ratio or
    # Fear & Greed (CNN's is an undocumented internal endpoint) — shown
    # honestly as unavailable per the brief's own explicit fallback rule.
    macro['put_call'] = {
        'value': None,
        'available': False,
        'interpretation': 'Data unavailable — no free, reliable, no-auth API exists for Put/Call ratio or Fear & Greed.',
    }

    return macro


# --- news (RSS + Anthropic) --------------------------------------------

def _parse_pub_ts(pub_str):
    try:
        return parsedate_to_datetime(pub_str).timestamp()
    except Exception:
        try:
            return datetime.fromisoformat(pub_str.replace('Z', '+00:00')).timestamp()
        except Exception:
            return time.time()


def _fetch_rss_feed(name, category, url):
    try:
        r = requests.get(url, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/rss+xml, application/xml, text/xml',
        }, timeout=8)
        if r.status_code != 200:
            return []

        import xml.etree.ElementTree as ET
        xml_content = re.sub(r' xmlns[^=]*="[^"]*"', '', r.text)
        xml_content = re.sub(r'<[a-z]+:', '<', xml_content)
        xml_content = re.sub(r'</[a-z]+:', '</', xml_content)
        root  = ET.fromstring(xml_content)
        items = root.findall('.//item') or root.findall('.//entry')

        articles = []
        for item in items[:8]:
            title = item.findtext('title') or ''
            title = re.sub(r'<[^>]+>', '', title).strip()
            if not title or len(title) < 10:
                continue
            pub  = item.findtext('pubDate') or item.findtext('published') or item.findtext('updated') or ''
            desc = item.findtext('description') or item.findtext('summary') or ''
            desc = re.sub(r'<[^>]+>', '', desc).strip()[:200]
            articles.append({
                'headline': title, 'source': name, 'category': category,
                'published': pub, 'description': desc,
                'timestamp': _parse_pub_ts(pub) if pub else time.time(),
            })
        return articles
    except Exception as e:
        logger.debug(f'RSS failed {name}: {e}')
        return []


def fetch_rss_all() -> list:
    import concurrent.futures
    articles = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(_fetch_rss_feed, name, cat, url): name for name, cat, url in NEWS_RSS_FEEDS}
        for fut in concurrent.futures.as_completed(futures, timeout=15):
            try:
                articles.extend(fut.result())
            except Exception:
                pass
    return articles


def call_anthropic(prompt: str, max_tokens: int = 600) -> dict:
    if not ANTHROPIC_KEY:
        raise Exception('ANTHROPIC_KEY not configured')
    r = requests.post(
        'https://api.anthropic.com/v1/messages',
        headers={'Content-Type': 'application/json', 'x-api-key': ANTHROPIC_KEY,
                  'anthropic-version': '2023-06-01'},
        json={'model': ANTHROPIC_MODEL, 'max_tokens': max_tokens,
              'messages': [{'role': 'user', 'content': prompt}]},
        timeout=60,
    )
    data = r.json()
    if data.get('error'):
        raise Exception(data['error'].get('message', 'anthropic error'))
    text  = data['content'][0]['text'].strip()
    match = re.search(r'\{[\s\S]*\}', text)
    if not match:
        raise Exception('Could not parse JSON from AI response')
    return json.loads(match.group())


def classify_news_item(headline: str, description: str = '') -> dict:
    prompt = (
        'You are a futures markets analyst. Classify this news headline for its '
        'relevance to ES (S&P 500) and MNQ (Nasdaq 100) futures trading.\n\n'
        f'Headline: {headline}\n'
        f'Summary: {description}\n\n'
        'Respond with ONLY a JSON object: {"relevance": "High"|"Medium"|"Low"|"Irrelevant", '
        '"direction": "Bullish"|"Bearish"|"Neutral", "explanation": "one sentence on why this matters for ES/MNQ"}'
    )
    try:
        result      = call_anthropic(prompt, max_tokens=200)
        relevance   = result.get('relevance', 'Low')
        direction   = result.get('direction', 'Neutral')
        explanation = result.get('explanation', '')
        if relevance not in ('High', 'Medium', 'Low', 'Irrelevant'):
            relevance = 'Low'
        if direction not in ('Bullish', 'Bearish', 'Neutral'):
            direction = 'Neutral'
        return {'relevance': relevance, 'direction': direction, 'explanation': explanation, 'ok': True}
    except Exception as e:
        # Classification genuinely failed (API/network/parse error) — distinct
        # from a real Low/Irrelevant classification. refresh_news() still
        # shows these items, tagged 'Unclassified', rather than silently
        # dropping every headline whenever the AI call has a problem.
        logger.debug(f'classify_news_item error: {e}')
        return {'relevance': 'Unclassified', 'direction': 'Neutral',
                'explanation': 'Classification unavailable', 'ok': False}


# =============================================================
#  C. IN-PROCESS CACHE / STATE
#     Single Flask process; background_scheduler() thread is the only writer.
# =============================================================

_STATE = {
    'narrative':  {'text': None, 'updated_at': 0},
    'news_items': {'items': [], 'seen_keys': set(), 'updated_at': 0},
    'last_short': None,
}


def check_threshold_cross(new_short):
    """Edge-triggered ±5 crossing detector — fires once per crossing, not every tick."""
    old = _STATE.get('last_short')
    _STATE['last_short'] = new_short
    if old is None or new_short is None:
        return None
    if old <= 5 < new_short:
        return 'crossed_up_5'
    if old >= -5 > new_short:
        return 'crossed_down_5'
    return None


def refresh_news():
    """Fetch + classify new RSS headlines, dedupe, cap at 10. Returns newly-seen High items."""
    try:
        raw = fetch_rss_all()
    except Exception as e:
        logger.warning(f'refresh_news RSS error: {e}')
        raw = []

    seen = _STATE['news_items']['seen_keys']
    new_high = []
    fresh = []
    for art in raw:
        key = (art['headline'][:120], art['source'])
        if key in seen:
            continue
        seen.add(key)
        c = classify_news_item(art['headline'], art.get('description', ''))
        # Show Medium/High (genuinely relevant) and Unclassified (classification
        # failed — better to show with a badge than silently vanish). Genuine
        # Low/Irrelevant classifications are still filtered out as intended.
        if c['relevance'] not in ('Medium', 'High', 'Unclassified'):
            continue
        item = {
            'headline': art['headline'], 'source': art['source'],
            'published': art['published'], 'timestamp': art['timestamp'],
            'relevance': c['relevance'], 'direction': c['direction'],
            'explanation': c['explanation'],
        }
        fresh.append(item)
        if c['relevance'] == 'High':
            new_high.append(item)

    combined = fresh + list(_STATE['news_items']['items'])
    combined.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
    combined = combined[:10]

    if len(seen) > 500:
        _STATE['news_items']['seen_keys'] = set(list(seen)[-300:])

    _STATE['news_items']['items'] = combined
    _STATE['news_items']['updated_at'] = int(time.time())
    return new_high


def generate_narrative(scored: dict) -> str:
    layers = scored['layers']
    es = scored['per_symbol'].get('ES', {})
    mnq = scored['per_symbol'].get('MNQ', {})
    prompt = (
        'You are a professional macro/futures desk analyst. Given these market-intelligence '
        'layer scores (-10 to +10 scale) for ES and MNQ futures, write a 2-3 sentence plain-English '
        'narrative describing the current market picture. Be concise and professional, no bullet points.\n\n'
        f"Macro: {layers['macro']:+.1f}\n"
        f"Regime/Momentum: {layers['regime']:+.1f}\n"
        f"ICT Structure: {layers['ict']:+.1f}\n"
        f"MTF Trend Alignment: {layers['mtf']:+.1f}\n"
        f"News Sentiment: {layers['news']:+.1f}\n"
        f"Composite short-term (4-24h): {scored['short_term']:+.1f}\n"
        f"Composite medium-term (1-5d): {scored['medium_term']:+.1f}\n"
        f"Label: {scored['label']}\n"
        f"ES: price {es.get('price')}, regime {es.get('regime')}, HTF bias {es.get('htf_bias')}\n"
        f"MNQ: price {mnq.get('price')}, regime {mnq.get('regime')}, HTF bias {mnq.get('htf_bias')}\n\n"
        'Respond with ONLY a JSON object: {"narrative": "..."}'
    )
    try:
        result = call_anthropic(prompt, max_tokens=300)
        text = (result.get('narrative') or '').strip()
        if text:
            return text
    except Exception as e:
        logger.warning(f'generate_narrative Anthropic error: {e}')
    return _fallback_narrative(scored)


def _fallback_narrative(scored: dict) -> str:
    layers = scored['layers']
    return (
        f"MRI composite is {scored['label']} at {scored['short_term']:+.1f} (short-term) / "
        f"{scored['medium_term']:+.1f} (medium-term). Layers — Macro {layers['macro']:+.1f}, "
        f"Regime {layers['regime']:+.1f}, ICT {layers['ict']:+.1f}, MTF {layers['mtf']:+.1f}, "
        f"News {layers['news']:+.1f}."
    )


def refresh_narrative():
    try:
        scored = _score_all_layers()
        text = generate_narrative(scored)
        _STATE['narrative'] = {'text': text, 'updated_at': int(time.time())}
    except Exception as e:
        logger.warning(f'refresh_narrative error: {e}')


# =============================================================
#  LAYER EXPLANATIONS — deterministic, template-based from real score
#  inputs (never hardcoded text, never an extra Anthropic call).
# =============================================================

def explain_macro_layer(macro: dict) -> str:
    parts, negatives, positives = [], 0, 0

    vix = macro.get('vix', {})
    if vix.get('available'):
        parts.append(f"VIX {(vix.get('regime') or '?').lower()} ({vix.get('value')})")
        sub = vix.get('sub_score') or 0
        negatives += sub < 0; positives += sub > 0

    dxy = macro.get('dxy', {})
    if dxy.get('available'):
        parts.append(f"DXY {dxy.get('trend')}")
        sub = dxy.get('sub_score') or 0
        negatives += sub < 0; positives += sub > 0

    yld = macro.get('yield_10y', {})
    if yld.get('available'):
        dir_txt = {'rising_sharply': 'rising sharply', 'stable_or_falling': 'stable/falling'}.get(
            yld.get('direction'), 'mixed')
        parts.append(f"10Y yield {dir_txt}")
        sub = yld.get('sub_score') or 0
        negatives += sub < 0; positives += sub > 0

    oil = macro.get('oil', {})
    if oil.get('available'):
        sub = oil.get('sub_score') or 0
        if sub != 0:
            parts.append(f"Oil {'rising' if sub < 0 else 'falling'} ({oil.get('change_pct_5d')}% 5d)")
            negatives += sub < 0; positives += sub > 0

    if not parts:
        return 'Macro data unavailable.'

    lead  = ', '.join(parts) + '.'
    total = negatives + positives
    if total == 0:
        summary = 'Net neutral for ES/MNQ momentum.'
    elif negatives == total:
        summary = f"All {total} headwind{'s' if total != 1 else ''} for ES/MNQ momentum."
    elif positives == total:
        summary = f"All {total} tailwind{'s' if total != 1 else ''} for ES/MNQ momentum."
    else:
        summary = f"{positives} tailwind{'s' if positives != 1 else ''}, {negatives} headwind{'s' if negatives != 1 else ''} — mixed picture."
    return f'{lead} {summary}'


def _regime_symbol_phrase(sym: str, info: dict) -> str:
    regime = info.get('regime') or 'regime unavailable'
    l3     = info.get('l3_probability')
    if l3 is None:
        return f'{sym} {regime}, L3 unavailable'
    if abs(l3 - 0.5) < 0.10:
        return f'{sym} {regime} but L3 neutral'
    return f'{sym} in {regime} regime ({l3 * 100:.1f}% L3 probability)'


def explain_regime_layer(per_symbol: dict) -> str:
    phrases = [_regime_symbol_phrase(sym, per_symbol.get(sym, {})) for sym in SYMBOLS if sym in per_symbol]
    if not phrases:
        return 'Regime data unavailable.'
    return '. '.join(phrases) + '.'


def _ict_symbol_phrase(sym: str, info: dict) -> str:
    price, vah, val = info.get('price'), info.get('vah'), info.get('val')
    parts = []
    if price is not None and vah is not None and val is not None:
        if price > vah:
            parts.append('above VAH')
        elif price < val:
            parts.append('below VAL')
        else:
            parts.append('inside value area')
    bull_fvg = info.get('bull_fvg_below', 0)
    bear_fvg = info.get('bear_fvg_above', 0)
    if bull_fvg:
        parts.append(f"{bull_fvg} bullish FVG{'s' if bull_fvg != 1 else ''} below")
    if bear_fvg:
        parts.append(f"{bear_fvg} bearish FVG{'s' if bear_fvg != 1 else ''} above")
    if not parts:
        return f'{sym} no clear structural signal'
    return f"{sym} {', '.join(parts)}"


def explain_ict_layer(per_symbol: dict) -> str:
    phrases = [_ict_symbol_phrase(sym, per_symbol.get(sym, {})) for sym in SYMBOLS if sym in per_symbol]
    if not phrases:
        return 'ICT structure data unavailable.'
    return '. '.join(phrases) + '.'


def explain_mtf_layer(per_symbol: dict) -> str:
    phrases = []
    for sym in SYMBOLS:
        info = per_symbol.get(sym)
        if not info or not info.get('mtf_total'):
            continue
        phrases.append(f"{sym} {info['mtf_bull_count']}/{info['mtf_total']} bullish")
    if not phrases:
        return 'MTF data unavailable.'
    return ', '.join(phrases) + '.'


def explain_news_layer(items, now_ts=None) -> str:
    if now_ts is None:
        now_ts = time.time()
    bullish = bearish = 0
    for it in (items or []):
        if it.get('relevance') != 'High':
            continue
        ts = it.get('timestamp')
        if ts is None or now_ts - ts > NEWS_WINDOW_SECONDS:
            continue
        if it.get('direction') == 'Bullish':
            bullish += 1
        elif it.get('direction') == 'Bearish':
            bearish += 1
    if bullish == 0 and bearish == 0:
        return 'No high-relevance news in the last 3 hours.'
    return (f"{bullish} high-relevance bullish headline{'s' if bullish != 1 else ''}, "
            f"{bearish} bearish, in the last 3 hours.")


def layer_explanations(scored: dict) -> dict:
    """Compose all five layer explanations from the same real inputs _score_all_layers() already fetched."""
    return {
        'macro':  explain_macro_layer(scored.get('macro', {})),
        'regime': explain_regime_layer(scored.get('per_symbol', {})),
        'ict':    explain_ict_layer(scored.get('per_symbol', {})),
        'mtf':    explain_mtf_layer(scored.get('per_symbol', {})),
        'news':   explain_news_layer(_STATE['news_items']['items']),
    }


# =============================================================
#  ORCHESTRATION
# =============================================================

def _score_all_layers() -> dict:
    macro = fetch_macro()
    macro_layer = macro_layer_score({
        'vix':   macro['vix'].get('sub_score'),
        'dxy':   macro['dxy'].get('sub_score'),
        'yield': macro['yield_10y'].get('sub_score'),
        'oil':   macro['oil'].get('sub_score'),
    })

    per_symbol = {}
    regime_scores, ict_scores, mtf_scores = [], [], []

    for sym in SYMBOLS:
        regime   = fetch_regime(sym)
        l3_prob  = fetch_l3(sym)
        htf_bias = fetch_htf_bias(sym)
        rscore   = regime_momentum_score(regime.get('regime'), regime.get('confidence'), l3_prob, htf_bias)
        regime_scores.append(rscore)

        price_info = fetch_price(sym)
        price      = price_info.get('price')
        va         = fetch_value_area(sym)
        liq        = fetch_liquidity(sym, '1hour')
        eq_levels  = fetch_equal_levels(sym, '5min')
        iscore     = ict_structure_score(price, va.get('vah'), va.get('val'), liq.get('active_fvgs'), eq_levels)
        ict_scores.append(iscore)

        mtf_biases = fetch_mtf_biases(sym)
        mscore     = mtf_trend_score(mtf_biases)
        mtf_scores.append(mscore)

        # Extra detail captured purely so layer_explanations() can describe
        # *why* a score is what it is, from the same real inputs already
        # fetched above — not a second round of IO.
        bull_fvg_below, bear_fvg_above = 0, 0
        if price is not None:
            for f in (liq.get('active_fvgs') or []):
                mid = (f.get('high', 0) + f.get('low', 0)) / 2
                if f.get('kind') == 'bull' and price > mid:
                    bull_fvg_below += 1
                elif f.get('kind') == 'bear' and price < mid:
                    bear_fvg_above += 1
        mtf_bull_count = sum(1 for b in mtf_biases.values() if b == 'BULLISH')
        mtf_bear_count = sum(1 for b in mtf_biases.values() if b == 'BEARISH')

        per_symbol[sym] = {
            'price': price, 'regime': regime.get('regime'), 'htf_bias': htf_bias,
            'regime_score': round(rscore, 2), 'ict_score': round(iscore, 2), 'mtf_score': round(mscore, 2),
            'l3_probability': round(l3_prob, 3) if l3_prob is not None else None,
            'regime_confidence': regime.get('confidence'),
            'vah': va.get('vah'), 'val': va.get('val'),
            'bull_fvg_below': bull_fvg_below, 'bear_fvg_above': bear_fvg_above,
            'mtf_bull_count': mtf_bull_count, 'mtf_bear_count': mtf_bear_count, 'mtf_total': len(mtf_biases),
        }

    regime_layer = float(np.clip(np.mean(regime_scores), -10, 10)) if regime_scores else 0.0
    ict_layer    = float(np.clip(np.mean(ict_scores), -10, 10)) if ict_scores else 0.0
    mtf_layer    = float(np.clip(np.mean(mtf_scores), -10, 10)) if mtf_scores else 0.0
    news_layer   = news_layer_score(_STATE['news_items']['items'])

    layers = {'macro': macro_layer, 'regime': regime_layer, 'ict': ict_layer, 'mtf': mtf_layer, 'news': news_layer}
    short  = composite_score(layers, WEIGHTS_SHORT)
    medium = composite_score(layers, WEIGHTS_MEDIUM)

    return {
        'layers': {k: round(v, 2) for k, v in layers.items()},
        'short_term': short,
        'medium_term': medium,
        'label': mri_label(short),
        'per_symbol': per_symbol,
        'macro': macro,
    }


def compute_composite() -> dict:
    """Top-level orchestrator for GET /api/mri/composite. Reads (never writes) the narrative cache."""
    scored = _score_all_layers()
    narrative = _STATE['narrative']['text'] or _fallback_narrative(scored)
    return {
        'layers': scored['layers'],
        'short_term': scored['short_term'],
        'medium_term': scored['medium_term'],
        'label': scored['label'],
        'narrative': narrative,
        'narrative_updated_at': _STATE['narrative']['updated_at'],
        'per_symbol': scored['per_symbol'],
        'layer_explanations': layer_explanations(scored),
        'updated_at': int(time.time()),
    }


def build_price_ladder(symbol: str) -> dict:
    price_info = fetch_price(symbol)
    price      = price_info.get('price')
    liq        = fetch_liquidity(symbol, '1hour')
    va         = fetch_value_area(symbol)
    eq_levels  = fetch_equal_levels(symbol, '5min')
    pdh, pdl   = _pdh_pdl(symbol)

    above, below = [], []

    def add(level_type, lvl_price, color, zone_high=None, zone_low=None):
        if lvl_price is None or price is None:
            return
        dist_pts = round(lvl_price - price, 2)
        dist_pct = round(dist_pts / price * 100, 3) if price else None
        entry = {'type': level_type, 'price': round(lvl_price, 2),
                  'distance_pts': dist_pts, 'distance_pct': dist_pct, 'color_group': color,
                  # zone_high/zone_low: real bounds for FVGs/order blocks (None for
                  # single-price levels like VAH/PDH) — lets a chart draw the actual
                  # zone rather than approximating from the midpoint alone.
                  'zone_high': round(zone_high, 2) if zone_high is not None else None,
                  'zone_low': round(zone_low, 2) if zone_low is not None else None}
        (above if lvl_price > price else below).append(entry)

    if va.get('vah') is not None:
        add('VAH', va['vah'], 'green')
    if va.get('val') is not None:
        add('VAL', va['val'], 'green')
    if pdh is not None:
        add('PDH', pdh, 'neutral')
    if pdl is not None:
        add('PDL', pdl, 'neutral')
    for f in liq.get('active_fvgs', []):
        mid = (f['high'] + f['low']) / 2
        add('Bullish FVG' if f['kind'] == 'bull' else 'Bearish FVG', mid, 'blue',
            zone_high=f['high'], zone_low=f['low'])
    for o in liq.get('active_obs', []):
        mid = (o['high'] + o['low']) / 2
        add('Bullish OB' if o['kind'] == 'bull' else 'Bearish OB', mid, 'purple',
            zone_high=o['high'], zone_low=o['low'])
    for lvl in eq_levels:
        add('Equal High' if lvl['type'] == 'high' else 'Equal Low', lvl['price'], 'amber')

    above.sort(key=lambda x: x['price'])
    below.sort(key=lambda x: -x['price'])
    return {'symbol': symbol, 'price': price, 'above': above, 'below': below}


def instrument_snapshot(symbol: str) -> dict:
    price_info = fetch_price(symbol)
    price      = price_info.get('price')
    regime     = fetch_regime(symbol)
    l3_prob    = fetch_l3(symbol)
    mult, _    = fetch_l3_multiplier(symbol, regime.get('hurst'), regime.get('confidence'))
    htf_bias   = fetch_htf_bias(symbol)
    atr        = fetch_atr_vs_avg(symbol)
    mtf        = fetch_mtf_biases(symbol)
    liq        = fetch_liquidity(symbol, '1hour')
    va         = fetch_value_area(symbol)
    eq_levels  = fetch_equal_levels(symbol, '5min')

    nearest_fvg_item = None
    if liq['active_fvgs'] and price is not None:
        nearest_fvg_item = min(liq['active_fvgs'], key=lambda f: abs((f['high'] + f['low']) / 2 - price))
    nearest_eq = None
    if eq_levels and price is not None:
        nearest_eq = min(eq_levels, key=lambda lv: abs(lv['price'] - price))

    return {
        'symbol': symbol,
        'price': price,
        'change_pts': price_info.get('change_pts'),
        'change_pct': price_info.get('change_pct'),
        'htf_bias_4h': htf_bias,
        'regime': regime.get('regime'),
        'regime_confidence': regime.get('confidence'),
        'l3_probability': round(l3_prob, 3) if l3_prob is not None else None,
        'l3_multiplier': mult,
        'atr14': atr.get('atr14'),
        'atr_avg20': atr.get('avg20'),
        'atr_expanding': atr.get('expanding'),
        'mtf_biases': mtf,
        'mtf_bullish_count': sum(1 for b in mtf.values() if b == 'BULLISH'),
        'mtf_total': len(mtf),
        'mtf_pct_bullish': pct_bullish(mtf),
        'nearest_fvg': nearest_fvg_item,
        'vah': va.get('vah'),
        'val': va.get('val'),
        'nearest_equal_level': nearest_eq,
    }


def _tf_trend(symbol: str, timeframe: str, limit: int = 60) -> dict:
    try:
        df = load_bars(symbol, timeframe, limit=limit)
        if len(df) < 20:
            return {'trend': 'insufficient history'}
        closes = df['close'].astype(float)
        ema20  = float(_ema20(closes).iloc[-1])
        last   = float(closes.iloc[-1])
        return {'trend': 'BULLISH' if last > ema20 else 'BEARISH', 'price': round(last, 2), 'ema20': round(ema20, 2)}
    except Exception:
        return {'trend': 'insufficient history'}


def _weekly_trend(symbol: str) -> dict:
    try:
        df = load_bars(symbol, '1week', limit=25)
        if len(df) < 10:
            return {'trend': 'insufficient history'}
        age_days = (pd.Timestamp.now(tz='UTC') - df.index[-1]).days
        if age_days > WEEKLY_STALE_DAYS:
            return {'trend': 'insufficient history'}
        closes = df['close'].astype(float)
        ema20  = float(_ema20(closes).iloc[-1])
        last   = float(closes.iloc[-1])
        return {'trend': 'BULLISH' if last > ema20 else 'BEARISH', 'price': round(last, 2), 'ema20': round(ema20, 2)}
    except Exception:
        return {'trend': 'insufficient history'}


def build_mtf_table() -> dict:
    table = {row: {} for row in MTF_TABLE_ROWS}
    for sym in SYMBOLS:
        table['Monthly'][sym] = {'trend': 'insufficient history'}
        table['Weekly'][sym]  = _weekly_trend(sym)
        table['Daily'][sym]   = _tf_trend(sym, '1day', limit=60)
        for row, tf in MTF_TABLE_TF_MAP.items():
            table[row][sym] = _tf_trend(sym, tf, limit=60)

    alignment = {}
    for sym in SYMBOLS:
        biases = {row: table[row][sym].get('trend') for row in MTF_TABLE_ROWS}
        alignment[sym] = pct_alignment(biases)

    return {'table': table, 'rows': MTF_TABLE_ROWS, 'symbols': list(SYMBOLS), 'alignment': alignment}
