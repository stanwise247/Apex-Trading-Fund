# Meridian News Feed

## What it shows

Meridian pulls headlines from free RSS feeds every 5 minutes, runs each new one through the
Anthropic API for relevance/direction classification, and shows only Medium and High relevance
items — newest first, capped at 10. High-relevance items also trigger an immediate Telegram alert.

**Interpretation:** relevance tells you whether a headline is worth reading at all for ES/MNQ;
direction tells you which way it leans. A High-relevance item with a colored left border is the
kind of headline that would make a macro desk look up from their screens.

---

## Sources

The original brief asked for "Reuters markets, Bloomberg markets RSS, Fed press releases,
economic calendar." **Bloomberg discontinued its free public RSS feeds years ago** — no free,
no-auth Bloomberg feed exists today. It's substituted with a subset of feeds already proven
working elsewhere in this codebase (server.py's `RSS_FEEDS`):

| Source | Category |
|---|---|
| Reuters Business | macro |
| Reuters World | macro |
| CNBC Markets | markets |
| MarketWatch | markets |
| WSJ Economy | macro |

Feeds are hand-parsed via `requests` + `xml.etree.ElementTree` (namespace-stripped with regex) —
same approach as server.py, deliberately not using the `feedparser` package (keeps the module
dependency-free, matching the rest of this codebase's convention).

---

## Relevance & direction classification

Each new headline (title + up to 200 chars of description) is sent to the Anthropic API
(`meridian_mri.classify_news_item`) with a prompt asking for:

- **Relevance**: `High` / `Medium` / `Low` / `Irrelevant`
- **Direction**: `Bullish` / `Bearish` / `Neutral` (for ES)
- **Explanation**: one sentence on why it matters for ES/MNQ

Only `Medium`/`High` items are kept and shown. If the Anthropic call fails for any reason (missing
key, network error, malformed response), classification fails toward `Low`/`Neutral` — the item is
silently dropped, never shown with a fabricated High-relevance tag.

---

## Refresh cadence and dedup

A `background_scheduler()` job in server.py calls `meridian_mri.refresh_news()` every 5 minutes.
Each fetched headline is deduped against an in-process `seen_keys` set (`(headline[:120], source)`
tuples) so re-fetched RSS items already classified aren't re-sent to Anthropic. The set is capped
at 500 entries to bound memory. Kept items are merged with the existing cached list, re-sorted
newest-first, and truncated to 10.

---

## Telegram alerting

Any item classified `High` relevance that wasn't already seen triggers an immediate Telegram alert
(headline + direction + one-line explanation), sent via `live_scanner.send_telegram(...,
message_type='mri_alert')`. Because dedup happens before classification, an item is never
re-alerted on a later refresh cycle.

---

## How to add a new source

Add a `(name, category, url)` tuple to `meridian_mri.NEWS_RSS_FEEDS`. No other code changes are
needed — the parser tolerates both RSS `<item>` and Atom `<entry>` formats. Verify a new feed
parses by running it directly:

```python
import meridian_mri as m
print(m._fetch_rss_feed('New Source', 'macro', 'https://example.com/rss'))
```

If it returns an empty list, check the feed's XML structure (some feeds wrap content in CDATA or
use non-standard namespaces the regex-strip step doesn't anticipate).

---

## API

### `GET /api/mri/news`

```json
{
  "ok": true,
  "news": [
    {
      "headline": "Fed holds rates steady, signals patience on cuts",
      "source": "Reuters Business",
      "published": "Mon, 21 Jul 2026 18:00:00 GMT",
      "timestamp": 1784663000.0,
      "relevance": "High",
      "direction": "Neutral",
      "explanation": "No surprise on rates keeps near-term volatility contained for ES/MNQ."
    }
  ],
  "updated_at": 1784663100
}
```

| Field | Type | Notes |
|---|---|---|
| `headline` | string | RSS title |
| `source` | string | Feed name |
| `timestamp` | float | Unix seconds, parsed from the feed's publish date |
| `relevance` | string | Always `Medium` or `High` — `Low`/`Irrelevant` are filtered out server-side |
| `direction` | string | `Bullish` / `Bearish` / `Neutral` |
| `explanation` | string | One-sentence AI-generated rationale |

This endpoint only reads the 5-minute cache — it never triggers a live RSS/Anthropic call, so it
stays fast regardless of feed latency.

---

## Debugging

- **News list stays empty** — check Railway logs for `MRI news refresh error`. Most common local
  cause: `ANTHROPIC_KEY not configured` (classification fails toward Low, so nothing clears the
  Medium/High filter). RSS fetch itself degrades silently per-feed (`_fetch_rss_feed` returns `[]`
  on any HTTP/XML error) — a single feed being down never blocks the others.
- **A High item never sent a Telegram alert** — check `live_scanner.py`'s `_ALLOWED_TYPES` still
  includes `'mri_alert'` (added specifically for Meridian; if this set gets edited in an unrelated
  change, alerts get silently dropped, not errored).
- **Same headline reappears after 5 minutes** — shouldn't happen; the dedup key is
  `(headline[:120], source)`, so a genuinely different source or a headline edited past 120 chars
  is treated as new. If this reoccurs for identical items, check whether `seen_keys` was reset by a
  server restart (it's in-process state, not persisted).
