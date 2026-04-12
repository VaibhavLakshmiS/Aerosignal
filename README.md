# AeroSignal

A Python tool that connects geopolitical news, oil prices, and flight data to estimate how world events might affect airfare on a given route.

Built as a learning project to explore data pipelines, API integration, and LLM-based agents.

[Report Bug](https://github.com/VaibhavLakshmiS/aerosignal/issues)

---

## What it does

You give it a route (e.g. Toronto → Dubai). It:

1. Pulls recent geopolitical news from GDELT
2. Checks current oil prices and trends via yfinance
3. Looks up live flight prices via Serpapi
4. Scores the route's risk based on news activity near the flight path
5. Projects how fares might change over the next 7 days
6. Flags hub airports that could see demand spikes (cascade effects)

A LangChain agent ties it together — the LLM decides which data sources to query based on the route.

---

## How it works

```
User enters route (e.g. YYZ → DXB)
        │
        ▼
┌──────────────────────────────────────┐
│         Data Collection (3-4s)       │
│                                      │
│  GDELT       → news events          │
│  yfinance    → WTI oil prices       │
│  Serpapi     → flight prices         │
│  OurAirports → airport coordinates   │
│                                      │
│  Risk score, forecast, cascade       │
│  detection all run here              │
└──────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│         AI Analysis (15-30s)         │
│                                      │
│  LangChain agent (Groq Llama 3.1)   │
│  picks which tools to call based     │
│  on the route and available data     │
│                                      │
│  Tools: search_news, get_oil_trend,  │
│  analyze_route_risk, get_forecast,   │
│  detect_cascade                      │
└──────────────────────────────────────┘
        │
        ▼
   Streamlit UI with Plotly charts
```

---

## Tech stack

| Component | Technology |
|-----------|-----------|
| Agent | LangChain + Groq Llama 3.1 70B |
| Vector store | ChromaDB + HuggingFace embeddings (local) |
| News | GDELT Project API |
| Oil prices | yfinance (WTI CL=F) |
| Flight prices | Serpapi Google Flights |
| Airports | OurAirports CSV (~3,500 airports) |
| Database | SQLite + SQLAlchemy |
| Anomaly detection | scikit-learn IsolationForest |
| UI | Streamlit + Plotly |

---

## How routing works

Routes are scored by checking what's happening along the actual flight path, not just at the origin and destination.

Waypoints are calculated using great circle interpolation between the two airports. News events near any waypoint contribute to the route's risk score. This means a Toronto → Mumbai flight picks up Gulf region risk if the great circle path passes through that airspace.

No routes are hardcoded. Any origin/destination pair works as long as both airports exist in the OurAirports dataset.

---

## How fare projection works

Based on the general relationship between oil prices and airline operating costs:

- Fuel is roughly 25% of airline costs (per IATA research)
- When oil rises, airlines pass some of that through as fuel surcharges
- Longer routes are more fuel-sensitive than shorter ones

The 7-day projection uses oil price momentum and news cycle decay (~15%/day) to estimate directional fare movement. These are rough estimates, not precise forecasts.

---

## Limitations

- **Not deployed** — runs locally only. Streamlit Cloud deployment had dependency issues I haven't resolved yet.
- **Backtest is thin** — only validated against 3 historical events. Not statistically meaningful.
- **No NOTAM data** — real airlines use official airspace notices. This approximates from news.
- **GDELT rate limits** — free tier has request limits. SQLite caching helps but doesn't eliminate the issue.
- **News volume bias** — heavily covered events (e.g. Ukraine) can inflate scores for routes that aren't actually affected. Proximity weighting partially addresses this but doesn't fully solve it.
- **Vibecoded in parts** — this was a learning project. Some modules were built with heavy AI assistance. I'm actively refactoring to understand and own every line.
- **Not financial or travel advice.**

---

## Project structure

```
aerosignal/
├── app.py                  # Streamlit entry point
├── data/
│   ├── airports.py         # OurAirports CSV loader
│   ├── fetch_events.py     # GDELT news fetcher with caching
│   ├── fetch_flights.py    # Serpapi flight prices + FX conversion
│   └── fetch_prices.py     # yfinance oil prices
├── rag/
│   ├── chain.py            # LangChain agent + tool definitions
│   ├── embed.py            # ChromaDB embeddings
│   └── signals.py          # Risk scoring, forecast, cascade detection
├── viz/
│   ├── charts.py           # Plotly charts
│   └── map.py              # 3D globe visualization
├── db/
│   └── database.py         # SQLAlchemy models, deduplication
└── .streamlit/
    └── config.toml         # Theme config
```

---

## Setup

```bash
git clone https://github.com/VaibhavLakshmiS/aerosignal.git
cd aerosignal

python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

pip install -r requirements.txt

cp .env.example .env
# Add your API keys to .env
```

### API keys needed

| API | Cost |
|-----|------|
| Groq | Free — [console.groq.com](https://console.groq.com) |
| Serpapi | 100 req/month free — [serpapi.com](https://serpapi.com) |
| GDELT | Free, no key needed |
| yfinance | Free, no key needed |

### `.env` file

```
GROQ_API_KEY=your_key
SERPAPI_KEY=your_key
```

### Run

```bash
streamlit run app.py
```

---

## What I learned building this

- Designing ETL pipelines that pull from multiple APIs and join data meaningfully
- Working with LangChain's agent/tool-calling pattern and understanding when an agent adds value vs. when a simple pipeline would suffice
- Vector embeddings and RAG for contextual news retrieval
- The difference between building something that works locally and deploying it reliably
- How to cache API responses properly to avoid rate limits and redundant calls

---

## License

MIT

---

Built by [Vaibhav](https://github.com/VaibhavLakshmiS)
