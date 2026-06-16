# Multi-Timeframe Bias Panel

## What it shows

The MTF panel on the Market Intel tab shows EMA20 directional bias for MNQ and ES across six timeframes: 1min, 5min, 15min, 30min, 1hr, and 4hr. Each timeframe is rendered as a coloured pill — green for BULLISH, red for BEARISH, grey for UNKNOWN. The panel refreshes every 30 seconds alongside the rest of the Market Intel tab.

**Interpretation:** When a timeframe shows BULLISH, price closed above its 20-period EMA in the most recent completed bar for that timeframe. BEARISH means close is below EMA20. Alignment across multiple timeframes suggests a stronger directional bias.

---

## How EMA20 bias is calculated

1. Bars are fetched from the `ohlcv` table for the requested symbol and timeframe.
2. For native timeframes (1min, 5min, 15min, 1hr, 4hr), bars are pulled directly using the `timeframe` column.
3. For 30min, 5min bars are fetched (400 bars) and resampled: `open=first, high=max, low=min, close=last, volume=sum`.
4. EMA20 is computed on the `close` series using pandas exponential weighted mean (`span=20, adjust=False`).
5. `bias = BULLISH if last_close > ema20_val else BEARISH`.
6. If fewer than 5 bars are available, `bias = UNKNOWN`.

---

## API

### `GET /api/apex/mtf`

Returns EMA20 bias for MNQ and ES across all six timeframes.

**Response:**

```json
{
  "ok": true,
  "ts": 1749686400.0,
  "mtf": {
    "MNQ": {
      "1min":  {"bias": "BULLISH", "close": 21345.50, "ema20": 21340.12},
      "5min":  {"bias": "BEARISH", "close": 21340.00, "ema20": 21352.80},
      "15min": {"bias": "BULLISH", "close": 21345.50, "ema20": 21330.00},
      "30min": {"bias": "BULLISH", "close": 21345.50, "ema20": 21310.00},
      "1hr":   {"bias": "BEARISH", "close": 21340.00, "ema20": 21360.00},
      "4hr":   {"bias": "BULLISH", "close": 21345.50, "ema20": 21280.00}
    },
    "ES": {
      "1min":  {"bias": "UNKNOWN", "error": "insufficient bars"},
      ...
    }
  }
}
```

Fields per timeframe:

| Field | Type | Notes |
|-------|------|-------|
| `bias` | string | `BULLISH`, `BEARISH`, or `UNKNOWN` |
| `close` | float | Last close price |
| `ema20` | float | EMA20 value for the most recent bar |
| `error` | string | Only present when `bias` is `UNKNOWN` |

**Error response:**

```json
{"ok": false, "error": "...message..."}
```

---

## Debugging

**Panel shows "MTF data unavailable"**

The frontend `_renderMIMTF()` function received an `ok: false` response or a network error. Check:

1. Railway logs for the `/api/apex/mtf` route — look for Python tracebacks.
2. `ohlcv` table has data: `SELECT count(*), timeframe FROM ohlcv WHERE symbol='MNQ' GROUP BY timeframe;`
3. Bar counts — if all timeframes show UNKNOWN, the `ohlcv` feed may have stopped writing.

**All pills show UNKNOWN**

The `< 5 bars` guard fired for every timeframe. This means the OHLCV feed is down or the `ohlcv` table has no recent data. The panel correctly shows UNKNOWN rather than crashing or showing stale data.

**30min shows UNKNOWN but others are fine**

The 30min bar is resampled from 5min bars. If there are fewer than two complete 30min windows in the last 400 5min bars (unlikely but possible after a gap), resampling can produce fewer than 5 rows. Check 5min bar recency for the affected symbol.

**`AttributeError: 'ExponentialMovingWindow' object has no attribute 'iloc'`**

This was the original bug (fixed). The EMA must be computed as:

```python
ema_series = closes.ewm(span=20, adjust=False).mean()   # correct
ema20_val  = float(ema_series.iloc[-1])
```

Not:

```python
closes.ewm(span=20, adjust=False).iloc[-1]              # wrong — raises AttributeError
```

If you see this error in logs, the fix is already in `server.py`; verify the deployed version matches.

---

## Tests

Five unit/integration tests cover this panel (`test_apex_full.py`):

| Test | What it verifies |
|------|-----------------|
| `test_m1_mtf_endpoint_returns_ok` | Live Railway endpoint returns `ok=True` with `mtf` key |
| `test_m2_mtf_all_timeframes_present` | Both symbols have all 6 TFs with valid bias values |
| `test_m3_mtf_bias_correct_against_known_data` | Rising closes → BULLISH, falling → BEARISH; `.ewm().iloc[-1]` raises `AttributeError` (regression guard) |
| `test_m4_mtf_handles_insufficient_bars` | <5 bars → UNKNOWN; 5 bars → valid bias; all live values are BULLISH/BEARISH/UNKNOWN |
| `test_m5_mtf_resample_30min_from_5min` | 12×5min bars resample to correct 30min OHLCV aggregation |
