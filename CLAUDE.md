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

## Key design decisions

### Route risk scoring uses full flight path, not just destination
A route like Dallas → Mumbai routed via Gulf inherits Gulf airspace
risk even though neither endpoint is in the Gulf.

In signals.py, implement a ROUTE_WAYPOINTS dictionary mapping
(origin_iata, destination_iata) tuples to a list of airspace
regions the flight passes through. Example:

ROUTE_WAYPOINTS = {
    ("YYZ", "DXB"): ["Gulf", "Eastern Europe", "Central Asia"],
    ("YYZ", "BOM"): ["Gulf", "Pakistan", "India"],
    ("LHR", "TLV"): ["Eastern Mediterranean"],
    ("JFK", "BOM"): ["Gulf", "Pakistan", "India"],
}

Score every region in the waypoints list and return the highest
score as the overall route risk score. This is a key differentiator
vs tools like Hopper that only score destination.

## Current phase
Phase 3 complete — full agent pipeline confirmed working end-to-end.
- rag/chain.py complete and tested: LangChain 1.x create_agent (LangGraph-based)
- Groq Llama 3.3 70b as LLM — tool-calling and multi-step reasoning confirmed
- ChromaDB embedding working with gemini-embedding-001
- GDELT news fetching working (429s handled gracefully as empty results)
- Real oil prices from yfinance working
- Smart great-circle waypoint routing working (slerp interpolation)
- Data-driven oil relevance scoring working (oil/anomaly only amplify regions with news activity)
- get_oil_trend tool takes no parameters (Groq schema strictness — int default causes string mismatch)
- Agent output confirmed: risk score, riskiest region, headlines with dates, oil trend, recommendation

Next: viz/map.py — Folium route map with risk region overlays
