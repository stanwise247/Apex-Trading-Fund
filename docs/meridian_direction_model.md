# Meridian Direction Model

## What it is

A supervised machine-learning model that predicts whether price will be **higher or lower** in N minutes. Trained separately per symbol (MNQ, ES) and per horizon (30min, 1hr, 4hr) for six models total. Predictions are calibrated probabilities — a 60% prediction should resolve correctly roughly 60% of the time.

This is different from Meridian L3, which predicts regime transitions (TRENDING vs not). L3 answers "will conditions favour trend trading?". The Direction model answers "will price go up or down from here?".

---

## Features (31 total)

### Regime state (from regime_log)
| Feature | Description |
|---------|-------------|
| `regime_trending` | 1 if current regime is TRENDING |
| `regime_choppy` | 1 if current regime is CHOPPY |
| `regime_mr` | 1 if current regime is MEAN_REVERTING |
| `regime_conf` | Regime classification confidence (0–1) |
| `hurst` | Hurst exponent from R/S analysis (0.5 = random walk) |
| `autocorr` | Lag-1 autocorrelation of 5min returns |
| `vol_ratio` | Realised vol 20-bar vs 100-bar ratio |

### Regime dynamics
| Feature | Description |
|---------|-------------|
| `hurst_chg3` | Hurst change over last 3 regime observations (~15 min) |
| `autocorr_chg3` | Autocorrelation change over last 3 observations |
| `consec_choppy` | Count of CHOPPY classifications in last 6 regime ticks |

### MTF EMA20 biases (+1 bullish / –1 bearish)
| Feature | Timeframe |
|---------|-----------|
| `ema_1m` | 1 minute |
| `ema_5m` | 5 minute |
| `ema_15m` | 15 minute |
| `ema_30m` | 30 minute (resampled from 5min) |
| `ema_1h` | 1 hour |
| `ema_4h` | 4 hour |
| `mtf_bull_count` | Count of bullish timeframes (0–6) |

### Price position
| Feature | Description |
|---------|-------------|
| `vwap_dist_atr` | (close – session VWAP) / ATR14, clipped ±5 |
| `pct_range_pdhl` | (close – prior day low) / (prior day high – low), 0–1 |
| `above_pdh` | 1 if close > prior day high (breakout) |
| `below_pdl` | 1 if close < prior day low (breakdown) |

### Momentum
| Feature | Description |
|---------|-------------|
| `ret_5` | 5-bar cumulative log return |
| `ret_10` | 10-bar cumulative log return |
| `ret_20` | 20-bar cumulative log return |

### Volatility
| Feature | Description |
|---------|-------------|
| `atr_ratio` | ATR5 / ATR20 — volatility expansion signal |
| `vol_ratio_20` | Current volume / 20-bar volume MA |

### Time
| Feature | Description |
|---------|-------------|
| `hour_sin`, `hour_cos` | UTC hour encoded cyclically |
| `dow_sin`, `dow_cos` | Day of week encoded cyclically |
| `session_progress` | (hour – 13) / 7.0 during 13:00–20:00 UTC, else 0.5 |

---

## Target construction

For each 5min bar at index `i`:
- **30min target**: `close[i+6] > close[i]` → 1 else 0
- **1hr target**: `close[i+12] > close[i]` → 1 else 0
- **4hr target**: `close[i+48] > close[i]` → 1 else 0

The last N rows (6/12/48 bars depending on horizon) have NaN targets and are excluded from training. Training is restricted to NY primary session bars (13:00–20:00 UTC, weekdays only) since that is when APEX trades.

---

## Validation methodology

Walk-forward validation with `TimeSeriesSplit(n_splits=5)`. Each fold trains on earlier data and tests on later data — no future information leaks into training.

### Calibration

After walk-forward validation, a final model is trained on the first 80% of data and calibrated using `CalibratedClassifierCV(method='sigmoid', cv='prefit')` on the remaining 20%. This isotonic calibration corrects the GBT's probability estimates to match actual frequencies.

A calibration table is stored alongside each model showing mean predicted vs actual frequency per bin. A well-calibrated model has small gaps across all bins.

### Deployment gate

**AUC gate: 0.54** — horizons where OOS AUC < 0.54 across all folds are not deployed. The dashboard shows "Insufficient Edge" for those horizons rather than a fabricated number.

---

## Validation results

*These results are updated each Sunday after the weekly retrain. Query ml_models for current trained_at and oos_auc.*

**Note:** Future equity prices are close to efficient. An AUC of 0.54–0.60 in this domain is genuine signal, not a modelling error. Do not expect 0.70+ — that level would indicate lookahead bias.

To check current model status:
```sql
SELECT model_name, oos_auc, training_samples, trained_at
FROM ml_models
WHERE model_name LIKE 'meridian_dir_%'
ORDER BY model_name;
```

---

## API

### `GET /api/apex/forecast`

Returns directional forecasts for MNQ and ES across all three horizons, plus live accuracy stats.

**Response structure:**

```json
{
  "ok": true,
  "ts": 1749686400.0,
  "forecast": {
    "MNQ": {
      "30min": {
        "deployed": true,
        "direction": "UP",
        "probability": 0.583,
        "prob_pct": 58.3,
        "auc": 0.561,
        "trained_at": "2026-06-15T03:00:00+00:00",
        "breakdown": {
          "regime":        {"label": "TRENDING",   "leans": "UP",   "conf": 0.67},
          "htf_bias":      {"label": "1h=BULL 4h=BULL", "leans": "UP",   "score": 2.0},
          "mtf_alignment": {"label": "5/6 bullish", "leans": "UP",   "count": 5},
          "vwap_pos":      {"label": "ABOVE VWAP", "leans": "UP",   "dist": 0.8},
          "momentum":      {"label": "ret5=0.123%", "leans": "UP",   "value": 0.00123}
        }
      },
      "1hr": {
        "deployed": false,
        "reason": "insufficient_edge",
        "auc": 0.521
      },
      "4hr": {
        "deployed": true,
        ...
      }
    },
    "ES": { ... }
  },
  "accuracy": {
    "MNQ": {
      "30min": {"total": 48, "correct": 27, "accuracy": 0.5625, "pct": 56.2},
      "1hr": null,
      "4hr": {"total": 45, "correct": 25, "accuracy": 0.5556, "pct": 55.6}
    }
  }
}
```

**When a horizon is not deployed:**
```json
{"deployed": false, "reason": "insufficient_edge", "auc": 0.521}
```

Possible reasons: `insufficient_edge` (AUC < gate), `no_model` (not yet trained), `no_features` (DB unavailable), `predict_error`.

---

## Retraining

The model retrains automatically every **Sunday at 03:00 UTC** (one hour after Meridian L3 retrains at 02:00 UTC).

To retrain manually:
```bash
DATABASE_URL="postgresql://..." python3 meridian_direction.py MNQ ES
```

The CLI output includes per-horizon AUC, fold breakdowns, and calibration tables. After a successful retrain, models are written to `ml_models` and reloaded into server memory automatically.

---

## Live accuracy tracker

Every 30 minutes the server logs the current prediction for each deployed horizon. When the horizon window elapses (30min/1hr/4hr), the prediction is resolved against the actual close price and marked correct or incorrect.

The dashboard shows the last 50 resolved predictions per horizon per symbol. This is a live self-audit: if live accuracy drops significantly below AUC (say 5%+ for 20+ predictions), it warrants investigation.

The table `direction_forecast_log` stores all predictions:
```sql
SELECT symbol, horizon, direction, probability, price_at_log, predicted_at,
       resolved, correct
FROM direction_forecast_log
ORDER BY predicted_at DESC LIMIT 20;
```

---

## What this can and cannot tell you

### What it can tell you

- The *direction* the model currently favours across three time horizons
- The *relative conviction* behind that direction (probability away from 50%)
- *Which signals* are leaning which way (breakdown section)
- *Whether it has been right* over the recent past (live accuracy tracker)
- *How confident to be* in the number (AUC from walk-forward validation)

### What it cannot tell you

- The *magnitude* of the move (direction only, not size)
- Whether a specific setup signal should be taken (the direction model has no knowledge of entry/stop/target)
- How the model will perform in the next 30 minutes specifically (probabilities only hold in expectation over many predictions)
- Whether the model is still in-distribution (if regime has shifted dramatically from training data, the model's edge may have decayed — watch the live accuracy tracker)

### Realistic accuracy expectations

In highly efficient futures markets (MNQ, ES) even the best direction models built on public features achieve 53–60% accuracy over large samples. If the live accuracy tracker shows 60%+ over 50+ predictions, do not assume it will continue — this is well within the tail of random variance. Use the model as one input, not a certainty.

An "Insufficient Edge" horizon is not a failure — it is an honest answer. A 52% AUC model shown as 52% would mislead you into thinking you have edge you don't. The model that says "I don't know" is more useful than one that shows you a number with no validation behind it.

---

## Debugging

**All horizons show "Insufficient Edge" after first deployment:**
Normal. The first training run must complete before models appear. Check Railway logs for `meridian_direction: training complete`. If it hasn't run, check that `DATABASE_URL` is set on Railway and the startup thread started.

**Dashboard shows "Forecast unavailable":**
The `/api/apex/forecast` endpoint returned `ok: false`. Check Railway logs for the stack trace. The most common cause is a missing `direction_forecast_log` table — the migration runs at startup via `init_schema()`.

**Accuracy shows "no resolved predictions yet":**
Normal for the first 4 hours after deployment. Predictions are logged every 30min and resolved when the horizon window passes. The 4hr horizon takes 4+ hours to produce its first resolved row.

**AUC looks suspiciously high (>0.65):**
Check for lookahead bias in feature construction. The `build_targets()` function uses strictly future bars; the `build_features()` function uses only past bars + current bar. Confirm by looking at the calibration table — a biased model will show high AUC but poorly calibrated probabilities.
