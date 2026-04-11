<div align="center">

# ✈ AeroSignal
### Geopolitical Flight Risk Intelligence

*Most flight tools tell you prices.*  
*AeroSignal tells you why prices are about to change.*

[Live Demo →](PLACEHOLDER_URL) · [Report Bug](https://github.com/VaibhavLakshmiS/aerosignal/issues)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-Agent-green)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

</div>

---

## The problem

You want to fly Toronto → Dubai next month.
Gulf tensions are escalating. Oil is rising.
Airlines haven't repriced yet — but they will.

Google Flights shows you today's price.
Nobody tells you what it'll be next week, or why.

**AeroSignal does.**

---

## What makes this different

### 1. Full corridor risk scoring
Most tools score the destination.
AeroSignal scores the entire flight path.

A Toronto → Mumbai flight routed via the Gulf inherits Gulf airspace risk —
even though neither Toronto nor Mumbai is in a conflict zone.

We calculate this using great circle interpolation across 3,500+ airports from
the OurAirports database. Any route worldwide works automatically. Nothing is hardcoded.

### 2. Cascade effect detection
When Gulf airspace closes, passengers reroute through Istanbul and Frankfurt.
Those hubs see a demand spike. Fares rise — even on routes with nothing to do
with the conflict.

AeroSignal detects this second-order effect and flags which hub airports will
be affected and by how much.

No consumer flight tool currently does this.

### 3. 7-day risk and fare forecast
Projects how risk and estimated fares will change over the next 7 days using:
- Oil price momentum (daily rate of change)
- News cycle decay (events lose relevance ~15%/day)
- IATA fuel cost methodology for fare impact

Tells you the optimal day to book — and what it'll cost if you wait.

### 4. True AI agent — not a pipeline
AeroSignal is built as a LangChain agent with 5 tools. The LLM decides which
tools to call and in what order based on the query.

For a Gulf route it might call `search_news`, then `get_oil_trend`, then
`analyze_route_risk`. For a Russia route it skips oil and checks airspace
sanctions directly.

That adaptive reasoning is what makes it an agent — not a fixed pipeline.

---

## Architecture

```
User enters route (e.g. YYZ → DXB)
        │
        ▼
┌─────────────────────────────────────────────┐
│              Stage 1 — Fast (3-4s)          │
│                                             │
│  OurAirports   →  Great circle waypoints   │
│  GDELT API     →  News events per region   │
│  yfinance      →  WTI oil prices + trend   │
│  Serpapi       →  Live flight fares        │
│  IsolationForest → Oil anomaly detection   │
│                                             │
│  score_route()        → 0-100 risk score   │
│  forecast_route()     → 7-day projection   │
│  detect_cascade_risk() → hub impacts       │
│  build_map_html()     → 3D globe           │
└─────────────────────────────────────────────┘
        │
        ▼  st.rerun() — globe shows immediately
        │
┌─────────────────────────────────────────────┐
│              Stage 2 — AI (15-30s)          │
│                                             │
│  LangChain Agent (Groq Llama 3.1 70B)      │
│    ├── search_news (ChromaDB RAG)           │
│    ├── get_oil_trend                        │
│    ├── analyze_route_risk                   │
│    ├── get_forecast_summary                 │
│    └── detect_cascade                       │
│                                             │
│  → AI Analysis tab populates               │
└─────────────────────────────────────────────┘
```

**Two-stage loading** means the globe, risk score, and forecast appear in
3-4 seconds. The AI agent reasoning appears in the Analysis tab when ready.
No blank screen. No waiting 30 seconds before seeing anything.

---

## Tech stack

| Component | Technology | Why |
|-----------|-----------|-----|
| AI Agent | LangChain + Groq Llama 3.1 70B | Free, fast, strong tool calling |
| Vector store | ChromaDB | Semantic news search |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | Local — no API cost |
| News data | GDELT Project API | Free, global, real-time |
| Oil prices | yfinance WTI CL=F | Free, reliable |
| Flight prices | Serpapi Google Flights | Real pricing data |
| Airport database | OurAirports CSV | 3,500+ airports, free |
| Database | SQLite + SQLAlchemy | Caching, deduplication |
| Anomaly detection | scikit-learn IsolationForest | Unsupervised, no labels needed |
| UI | Streamlit + Plotly | Fast to build, interactive globe |

---

## Key engineering decisions

**Great circle routing — not hardcoded**  
Route waypoints are calculated geometrically using spherical linear
interpolation. No hardcoded route lists — any route worldwide works
automatically using 3,500+ airports from OurAirports.

**Data-driven oil relevance**  
Oil scores only amplify regions that already have news activity.
Zero news = zero oil amplification. Prevents false positives from oil
spikes in unaffected regions.

**Destination-proximity weighting**  
Regions close to the destination get a 1.8× score multiplier; regions
far away get 0.6×. This prevents high-media events (Ukraine war news
volume) from outranking the actual destination region (Gulf for YYZ→DXB).

**Cascade detection via bounding boxes**  
Hub airports are excluded from cascade risk if their coordinates fall
inside the trigger region's geographic bounding box. No hardcoded airport
lists — pure geometry.

**OR IGNORE deduplication**  
Every API response is cached in SQLite with OR IGNORE on all inserts.
The same article never gets stored twice. Routes analyzed within 24 hours
return cached results.

**ECB live FX rates**  
Flight prices are converted to USD using live European Central Bank rates.
Falls back to fixed rates if ECB is unreachable.

---

## Model validation

Backtested against 3 historical events using only data available **before**
each event peaked:

| Event | Date | Pre-event score | Oil Δ (14 days after) |
|-------|------|----------------|----------------------|
| Ukraine invasion | Feb 2022 | 27.5/100 | +17.7% |
| Iran attacks Israel | Oct 2024 | 15.0/100 | +8.3% |
| Gaza conflict start | Oct 2023 | 35.0/100 | +11.2% |

**Directional accuracy: 1/3 events correctly flagged**

Honest note: Ukraine and Iran attacks show low pre-event scores because oil
signals were not yet elevated 30 days before the events peaked. The model
detects gradual risk escalation — not sudden shocks. This is an expected and
documented limitation.

See the **Backtest tab** in the live demo.

---

## Fare projection methodology

Based on IATA published fuel cost research:

> Fuel represents ~25% of airline operating costs. When oil rises, airlines
> pass costs through via fuel surcharges.

| Route type | 10% oil rise → fare impact |
|------------|---------------------------|
| Long haul (>8 hrs) | ~3.5% |
| Medium haul (3-8 hrs) | ~2.5% |
| Short haul (<3 hrs) | ~1.5% |

Route type is calculated automatically from haversine distance — not hardcoded.

*Projections are directional estimates. Not precise forecasts.*

---

## Limitations

**Being transparent about what this doesn't do:**

- **No NOTAM data** — real airlines use Notice to Airmen data for routing.
  This model approximates from news and geography.
- **GDELT rate limits** — free tier has limits. SQLite cache used as fallback.
- **News volume bias** — high-media events (Ukraine) score higher regardless
  of actual aviation impact. Proximity weighting partially corrects this.
- **Backtest is thin** — 3 events is not statistically significant.
  20-30 events needed for proper validation.
- **Not financial or travel advice** — always check official airline and
  government advisories before booking.

---

## Setup

```bash
# Clone
git clone https://github.com/VaibhavLakshmiS/aerosignal.git
cd aerosignal

# Virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env and add your keys

# Run
streamlit run app.py
```

### API keys required

| API | Cost | Where to get |
|-----|------|------|
| Groq | Free | [console.groq.com](https://console.groq.com) |
| Serpapi | 100 req/month free | [serpapi.com](https://serpapi.com) |
| GDELT | Free, no key | automatic |
| yfinance | Free, no key | automatic |
| OurAirports | Free, no key | automatic |
| ECB FX rates | Free, no key | automatic |

### `.env` file

```
GROQ_API_KEY=your_groq_key_here
SERPAPI_KEY=your_serpapi_key_here
```

---

## Project structure

```
aerosignal/
├── app.py                  # Streamlit entry point, two-stage loading, rate limiting
├── data/
│   ├── airports.py         # OurAirports CSV loader (3,500+ airports)
│   ├── fetch_events.py     # GDELT news fetcher with SQLite cache + backoff
│   ├── fetch_flights.py    # Serpapi Google Flights + ECB live FX rates
│   └── fetch_prices.py     # yfinance WTI oil prices + trend calculation
├── rag/
│   ├── chain.py            # LangChain agent, Groq LLM, 5 tool definitions
│   ├── embed.py            # ChromaDB embeddings (HuggingFace local)
│   └── signals.py          # Risk scoring, proximity weighting, forecast,
│                           # cascade detection, backtest validation
├── viz/
│   ├── charts.py           # Plotly forecast, oil, cascade bar charts
│   └── map.py              # Plotly 3D orthographic globe with route animation
├── db/
│   └── database.py         # SQLAlchemy models, OR IGNORE deduplication
└── .streamlit/
    └── config.toml         # Dark theme configuration
```

---

## License

MIT — see [LICENSE](LICENSE)

---

<div align="center">
Built by <a href="https://github.com/VaibhavLakshmiS">Vaibhav Lakshmi</a>
</div>
