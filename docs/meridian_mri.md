# Meridian MRI — Market Resonance Index

## What it is

The MRI (Meridian Resonance Index) is a composite -10..+10 read on ES/MNQ synthesised from five
independent layers: Macro conditions, Regime/Momentum, ICT price structure, MTF trend alignment,
and News sentiment. Two composites are produced — a short-term read (4-24h horizon, weighted
toward Regime/Momentum) and a medium-term read (1-5 day horizon, weighted toward Macro and MTF).
A plain-English narrative is regenerated every 5 minutes via the Anthropic API from the current
layer scores.

**Interpretation:** `>=+5` BULLISH, `[+2,+5)` CAUTIOUSLY BULLISH, `(-2,+2)` NEUTRAL, `(-5,-2]`
CAUTIOUSLY BEARISH, `<=-5` BEARISH. This is a market-intelligence read, not a trade signal — it
describes conditions, it doesn't recommend entries.

---

## The five layers

### Macro (`meridian_mri.macro_layer_score`)

Simple mean of four sub-scores, scaled ×5 to reach the -10..+10 range, clipped:

| Metric | Rule | Sub-score |
|---|---|---|
| VIX | `<15` / `15-20` / `20-25` / `>25` | `+2` / `0` / `-1` / `-2` |
| DXY | rising / falling | `-1` / `+1` |
| 10Y Yield | rising sharply (>3% over 5 sessions) / stable-or-falling | `-1` / `+1` |
| Oil (WTI) | rising >1.5% (5d) / falling <-1.5% (5d) | `-1` / `+1` |

Put/Call ratio and Fear & Greed have no sub-score — see "Known assumptions" below.

### Regime / Momentum (`meridian_mri.regime_momentum_score`)

`regime_log` and the Meridian L3 model measure trend *strength*, not *direction* — see the
direction-sign assumption below for how this layer gets signed.

| Regime | Confidence | Strength |
|---|---|---|
| TRENDING | `>=0.50` | `+3` |
| TRENDING | `<0.50` | `+1` |
| CHOPPY | — | `0` |
| MEAN_REVERTING | — | `-1` |

Strength is then modulated by the Meridian L3 probability (`P(TRENDING next 30min)`, normalised to
-1..+1) and signed by HTF 4h bias direction.

### ICT Structure (`meridian_mri.ict_structure_score`)

| Condition | Points |
|---|---|
| Price above VAH | `+2` |
| Price below VAL | `-2` |
| Price inside value area | `0` |
| Above an unfilled bullish FVG (each, max 3) | `+1` |
| Below an unfilled bearish FVG (each, max 3) | `-1` |
| Within 0.3% of an equal high | `+1` |
| Within 0.3% of an equal low | `-1` |

### MTF Trend (`meridian_mri.mtf_trend_score`)

Each of the six timeframes (1m/5m/15m/30m/1h/4h) contributes `+1` (BULLISH), `-1` (BEARISH), or
`0` (NEUTRAL); UNKNOWN timeframes are excluded from the denominator entirely (never treated as
neutral). Sum is normalised to -10..+10.

### News (`meridian_mri.news_layer_score`)

Only High-relevance items from the last 3 hours count: `+2` bullish, `-2` bearish each, capped at
±6.

---

## Weighting — short-term vs medium-term

| Layer | Short-term (4-24h) | Medium-term (1-5d) |
|---|---|---|
| Macro | 15% | 30% |
| Regime/Momentum | 30% | 20% |
| ICT Structure | 25% | 20% |
| MTF Trend | 20% | 25% |
| News | 10% | 5% |

Regime/Momentum dominates the short-term read because trend strength and structure decay fastest;
Macro dominates the medium-term read because rate/dollar/vol regimes persist over days while
intraday structure resets each session. Both weight dicts must sum to 1.0 — `test_mri6` guards
this as a regression check.

---

## Known assumptions and how to adjust them

The brief's formulas assume data the schema doesn't actually carry. These defaults are isolated
behind named constants/functions so they're easy to change:

1. **Regime direction.** `regime_log`/L3 measure strength, not direction. Direction is taken from
   `setup_h_vwap.get_h_state(symbol)['htf_bias']` and applied as a sign multiplier in
   `regime_momentum_score`. To use a different direction source, edit that one function.
2. **MTF is not double-counted.** The brief's Regime/Momentum bullet also mentions "MTF alignment
   % mapped to ±5" — this is treated as describing the separate MTF layer only, not folded into
   Regime/Momentum a second time.
3. **Macro sub-weights.** VIX/DXY/Yield/Oil sub-scores are simple-averaged (equal weight) in
   `macro_layer_score`. To weight one metric more heavily, replace the `np.mean` call with a
   weighted average.
4. **Oil sub-score shape.** No formula was given for oil; it uses the same rising/falling shape as
   the yield sub-score, at ±1.5% over 5 sessions (`oil_subscore`).
5. **MRI label thresholds.** `mri_label`'s boundaries (±5 / ±2) aren't specified in the brief —
   they were set to match the brief's own ±5 alert-crossing threshold. Edit `mri_label` to change.
6. **Equal highs/lows symmetry.** The brief only specifies "+1 near equal highs"; equal lows are
   treated symmetrically (`-1`) in `ict_structure_score`.

---

## Narrative generation

`meridian_mri.generate_narrative(scored)` builds a prompt containing all five layer scores, both
composites, the label, and each symbol's price/regime/HTF bias, then calls the Anthropic API
(`meridian_mri.call_anthropic`, raw HTTP — matches the existing codebase convention of not using
the `anthropic` SDK) asking for a 2-3 sentence narrative as JSON. Model is `claude-sonnet-5`.

**History:** originally shipped as `claude-sonnet-4-20250514` (matching server.py's existing
`call_anthropic` at the time). That snapshot was confirmed `404 not_found_error` on 2026-07-23 —
deprecated/retired — which silently broke both the narrative (silently falling back to the
templated `_fallback_narrative` every cycle) and news classification (every headline falling back
to `Unclassified`, filtering the feed down to nothing). Switched to `claude-sonnet-5`, confirmed
working live. If narrative/news classification silently stop working again, check the model string
first — `call_anthropic` raising `not_found_error` is a very easy failure to miss since both
callers already degrade gracefully instead of crashing.

Refreshed every 5 minutes by a `background_scheduler()` job in server.py. If the API call fails
(missing key, network error, bad response), `generate_narrative` falls back to a templated
description built directly from the real layer scores (`_fallback_narrative`) — never a blank or
fake narrative.

---

## API

### `GET /api/mri/composite`

```json
{
  "ok": true,
  "layers": {"macro": -2.5, "regime": 3.89, "ict": 1.5, "mtf": 1.67, "news": 0.0},
  "short_term": 1.5,
  "medium_term": 0.7,
  "label": "NEUTRAL",
  "narrative": "The market picture is mixed...",
  "narrative_updated_at": 1784664057,
  "per_symbol": {
    "ES":  {"price": 7508.0, "regime": "TRENDING", "htf_bias": "BULLISH", "regime_score": 4.25, "ict_score": 0.0, "mtf_score": 0.0,
            "l3_probability": 0.796, "regime_confidence": 0.75, "vah": 7522.5, "val": 7510.25,
            "bull_fvg_below": 0, "bear_fvg_above": 0, "mtf_bull_count": 4, "mtf_bear_count": 2, "mtf_total": 6},
    "MNQ": {"price": 28910.0, "regime": "TRENDING", "htf_bias": "BULLISH", "regime_score": 3.54, "ict_score": 3.0, "mtf_score": 3.33}
  },
  "layer_explanations": {
    "macro": "VIX elevated (16.9), DXY rising, 10Y yield rising sharply. All 3 headwinds for ES/MNQ momentum.",
    "regime": "ES in TRENDING regime (79.6% L3 probability). MNQ in TRENDING regime (62.9% L3 probability).",
    "ict": "ES inside value area. MNQ above VAH.",
    "mtf": "ES 4/6 bullish, MNQ 4/6 bullish.",
    "news": "No high-relevance news in the last 3 hours."
  },
  "updated_at": 1784664057
}
```

| Field | Type | Notes |
|---|---|---|
| `layers` | object | Five layer scores, -10..+10 |
| `short_term` / `medium_term` | float | Weighted composites, -10..+10, 1 decimal |
| `label` | string | BULLISH / CAUTIOUSLY BULLISH / NEUTRAL / CAUTIOUSLY BEARISH / BEARISH |
| `narrative` | string | Cached, refreshed every 5 min |
| `per_symbol` | object | Per-symbol contribution detail for ES/MNQ, including the extra fields (`l3_probability`, `vah`/`val`, FVG counts, MTF bull/bear counts) that `layer_explanations` is built from |
| `layer_explanations` | object | One plain-English sentence per layer (`macro`/`regime`/`ict`/`mtf`/`news`), generated deterministically from the same real inputs above — see `meridian_mri.layer_explanations()`. Never hardcoded, never a second Anthropic call. |

### `GET /api/mri/mtf`

Returns the full 8-row MTF table (Monthly/Weekly/Daily/4H/1H/15M/5M/1M × ES/MNQ) plus per-symbol
alignment %, and bundles both instrument snapshots (Section 2 card data) in the same response.

### `GET /api/mri/levels`, `GET /api/mri/macro`, `GET /api/mri/news`

See the ICT price ladder, macro conditions, and news feed sections of `meridian_dashboard.html`
respectively; `docs/meridian_news.md` covers the news pipeline in detail.

---

## Debugging

- **A layer score always shows 0 for one symbol** — most commonly means `regime_log` has no rows
  for that symbol (check `SELECT * FROM regime_log WHERE symbol=? ORDER BY timestamp DESC LIMIT 1`)
  or `htf_bias` came back `NEUTRAL`/unavailable, which zeroes `regime_momentum_score` by design
  (no directional read available) rather than guessing a sign.
- **Narrative never changes** — check Railway logs for `MRI narrative refresh error`; the most
  common cause locally is `ANTHROPIC_KEY not configured` (only set in Railway's environment, not
  local `.env`/`config.json`).
- **Monthly/Weekly always "insufficient history"** — expected. No ingestion path writes
  `'1month'`/`'1week'` timeframes to `ohlcv` on a live basis; Weekly additionally requires the most
  recent bar to be less than `WEEKLY_STALE_DAYS` (14) old, so a stale historical backfill doesn't
  get presented as a live trend.
- **Composite differs between `/api/mri/composite` calls and the morning brief** — both call
  `meridian_mri.compute_composite()` fresh each time (not cached), so a few seconds of price/regime
  drift between calls is expected, not a bug.

---

## Tests

Eighteen tests cover this module (`test_apex_full.py`, `MRI1`-`MRI11` pure scoring tests + `MRIAPI1`-`MRIAPI7` endpoint tests):

| Test | What it verifies |
|---|---|
| `MRI1` | VIX sub-score thresholds |
| `MRI2` | Macro layer score matches hand-computed mean |
| `MRI3` | Regime/momentum score sign flips with HTF bias direction (not just magnitude) |
| `MRI4` | ICT score with `vah=val=None` returns `0.0`, never raises |
| `MRI5` | MTF trend score hits ±10.0 at the extremes |
| `MRI6` | Both weight dicts sum to 1.0 (regression guard) |
| `MRI7` | Composite score matches a hand-computed weighted sum |
| `MRI8` | MRI label boundaries |
| `MRI9` | News score's 3h window and ±6 cap |
| `MRI10` | `pct_bullish` and `pct_alignment` are genuinely different metrics |
| `MRI11` | Threshold-cross fires once per crossing, not every tick |
| `MRIAPI1`-`MRIAPI7` | Endpoint shape, graceful degradation, root route swap |
