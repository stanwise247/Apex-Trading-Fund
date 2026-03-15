# APEX Trading Engine — Phase 1
## Live Market Intelligence Dashboard

---

## What This Is

A fully local trading intelligence system with:
- **Live prices** from Polygon.io (real-time, no proxies, no CORS issues)
- **Live financial news** from NewsAPI fed directly into AI analysis
- **Real historical data** stored locally in SQLite — grows forever
- **Multi-timeframe technical analysis** computed server-side
- **Market regime classification** updated every 30 seconds
- **AI daily brief** powered by Claude — synthesises all live data
- **Historical database** — foundation for the edge engine (Phase 2)

---

## Setup — 5 Minutes

### Step 1: Get API Keys (all free tiers work)

**Polygon.io** (live prices + historical data)
1. Go to https://polygon.io
2. Sign up — free tier gives you delayed quotes + historical data
3. Go to Dashboard → API Keys → copy your key

**NewsAPI** (live financial headlines)
1. Go to https://newsapi.org
2. Sign up free — 100 requests/day on free tier
3. Copy your API key from the dashboard

**Anthropic** (AI analysis — you already have this)
- From https://console.anthropic.com

### Step 2: Install Python
- Windows: https://python.org → Download → check "Add to PATH"
- Mac: `brew install python3` or download from python.org
- Check it works: open Terminal/Command Prompt, type `python --version`

### Step 3: Start the Engine

**Windows:**
- Double-click `START_WINDOWS.bat`

**Mac/Linux:**
```bash
chmod +x START_MAC_LINUX.sh
./START_MAC_LINUX.sh
```

### Step 4: Open the Dashboard
- Open `apex_dashboard.html` in Chrome or Firefox
- Click ⚙ SETUP (top right)
- Enter your API keys
- Click SAVE CONFIGURATION

### Step 5: Backfill Historical Data
- Select an instrument (NQ, ES, etc.)
- Click "⬇ Backfill History" in the Intraday Plan column
- This downloads up to 5 years of OHLCV data in the background
- Run this for each instrument you trade

---

## How It Works

```
Polygon.io ──→ server.py ──→ apex_market.db (grows forever)
NewsAPI    ──→ server.py ──→ AI prompt
                    ↓
             apex_dashboard.html
                    ↓
             Anthropic API ──→ Daily Brief
```

The server runs locally on port 5000. The dashboard connects to it.
All data is stored on YOUR machine. Nothing is sent to the cloud
except the AI prompt to Anthropic.

---

## Phase 2 (Coming Next)

- `patterns.py` — mines historical database for statistical edges
- `backtest.py` — validates every pattern across all regimes
- `scanner.py` — real-time setup detector with edge scoring
- `regime.py` — enhanced regime classification
- `alerts.py` — notifications when high-conviction setups trigger

---

## Files

```
apex/
├── server.py           — Main backend server
├── apex_dashboard.html — Live dashboard
├── requirements.txt    — Python dependencies
├── config.json         — Your API keys (never share this)
├── apex_market.db      — Historical database (created on first run)
├── START_WINDOWS.bat   — Windows launcher
├── START_MAC_LINUX.sh  — Mac/Linux launcher
└── README.md           — This file
```

---

## ⚠ Disclaimer

For educational and informational purposes only.
Not financial advice. Futures trading involves substantial risk of loss.
