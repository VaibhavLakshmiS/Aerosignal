# AeroSignal — Claude Code context

## What this project is
A geopolitical flight price intelligence AI agent. Users enter a flight 
route and the agent analyzes geopolitical news, oil prices, and historical 
patterns to output a risk score and AI summary.

## Stack
- Python, LangChain agent with tool-calling
- ChromaDB for vector storage (embedded news)
- SQLite via SQLAlchemy for events/prices/risk history
- Gemini Flash (free tier) as the LLM
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
Phase 2 in progress — database.py complete.
Next: fetch_events.py
