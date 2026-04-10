# AeroSignal — Claude Code context

## What this project is
A geopolitical flight price intelligence AI agent. Users enter a flight 
route and the agent analyzes geopolitical news, oil prices, and historical 
patterns to output a risk score and AI summary.

## Stack
- Python, LangChain 1.x agent with tool-calling (create_agent, LangGraph-based)
- ChromaDB for vector storage (embedded news)
- SQLite via SQLAlchemy for events/prices/risk history
- Groq Llama 3.3 70b Versatile as the LLM (llama-3.3-70b-versatile via langchain-groq)
- Gemini embedding-001 for ChromaDB embeddings (GEMINI_API_KEY)
- Serpapi for flight prices
- GDELT for geopolitical news (free, no key)
- yfinance for oil prices
- Streamlit + Folium + Plotly for the UI

## Folder structure
- app.py — Streamlit entry point
- data/ — fetch_events.py, fetch_prices.py, fetch_flights.py
- rag/ — embed.py, chain.py, signals.py
- viz/ — map.py, charts.py
- db/ — database.py

## Rules when writing code
- Always use the venv Python at venv/Scripts/python.exe
- Never hardcode API keys — always load from .env using python-dotenv
- Keep each file focused — one responsibility per file
- Add a short docstring at the top of every file explaining what it does
- Use type hints on all functions

## Key design decisions (do not change)
- Great circle routing via spherical interpolation
- OurAirports CSV for airport coordinates (3500+)
- Data-driven oil relevance (not hardcoded regions)
- Cascade detection via geographic bounding boxes
- OR IGNORE deduplication on all SQLite inserts
- Two-stage loading in app.py (fast data first, agent second)
- Groq Llama 3.1 70B as LLM
- HuggingFace all-MiniLM-L6-v2 for embeddings (local)
- IATA fuel cost methodology for fare projections
- ECB live FX rates with fixed rate fallback

## Current status
Project is feature complete and ready to deploy.

## Completed files
- db/database.py ✓
- data/airports.py ✓
- data/fetch_events.py ✓
- data/fetch_prices.py ✓
- data/fetch_flights.py ✓
- rag/embed.py ✓
- rag/signals.py ✓ (includes forecast, cascade, backtest)
- rag/chain.py ✓
- viz/map.py ✓ (Plotly 3D globe)
- viz/charts.py ✓
- app.py ✓ (full dashboard)
- .streamlit/config.toml ✓

## Remaining tasks
- README.md (not written yet)
- Deploy to Streamlit Cloud
- Add live URL to README
- Update resume

## Known issues to fix post-deploy
- GDELT rate limiting (caching helps but not perfect)
- Eastern Europe occasionally scores higher than Gulf due to Ukraine war news volume

## APIs in use
- Groq API (LLM) — GROQ_API_KEY
- Serpapi (flights) — SERPAPI_KEY
- GDELT (news) — no key needed
- yfinance (oil) — no key needed
- OurAirports CSV — no key needed
- ECB FX rates — no key needed
