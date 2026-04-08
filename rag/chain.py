"""
rag/chain.py — LangChain tool-calling agent for AeroSignal.

Connects the data layer (GDELT news, oil prices, flight prices) and RAG layer
(ChromaDB embeddings, route risk scoring) into a single Gemini Flash agent.
Exposes run_agent() as the main entry point for the Streamlit UI.

Uses the LangChain 1.x create_agent API (LangGraph-based), which replaces
the legacy AgentExecutor + create_tool_calling_agent pattern.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import os
from datetime import datetime

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from sqlalchemy import select
from sqlalchemy.orm import Session

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def search_news(region: str) -> str:
    """
    Fetch and semantically search recent geopolitical news for a given airspace region.

    Calls GDELT for fresh articles, embeds them into ChromaDB, then runs a
    semantic search for conflict/tension signals. Use this after analyze_route_risk
    identifies a high-risk region.

    Args:
        region: Airspace region name, e.g. "Gulf", "Eastern Europe", "Pakistan".
    """
    from data.fetch_events import fetch_events
    from rag.embed import embed_events, search_events

    events = fetch_events(region)
    if events:
        embed_events(events)

    semantic_results = search_events(
        f"{region} conflict tension military geopolitical", n_results=5
    )

    if not events and not semantic_results:
        return f"No recent news found for region: {region}"

    seen_titles: set[str] = set()
    all_items: list[dict] = []

    for e in events[:5]:
        t = e.get("title", "")
        if t and t not in seen_titles:
            seen_titles.add(t)
            all_items.append(e)

    for e in semantic_results:
        t = e.get("title", "")
        if t and t not in seen_titles:
            seen_titles.add(t)
            all_items.append(e)

    lines = [f"News for {region} ({len(events)} articles fetched from GDELT):"]
    for e in all_items[:8]:
        date = e.get("date", "unknown date")
        title = e.get("title", "")
        lines.append(f"  - [{date}] {title}")

    return "\n".join(lines)


@tool
def get_oil_trend() -> str:
    """
    Fetch the Brent crude oil price trend over the last 30 days via Yahoo Finance.

    Oil price spikes correlate with geopolitical instability in oil-producing regions
    and directly affect airline fuel surcharges.
    """
    from data.fetch_prices import fetch_oil_prices
    from data.fetch_prices import get_oil_trend as _get_oil_trend

    prices = fetch_oil_prices(days=30)
    if not prices:
        return "Could not fetch oil price data."

    trend = _get_oil_trend(prices)
    direction = "RISING" if trend["is_rising"] else "FALLING"
    sign = "+" if trend["price_change_pct"] >= 0 else ""

    return (
        f"Brent crude oil (30-day trend):\n"
        f"  Current price: ${trend['current_price']:.2f}/bbl\n"
        f"  30-day change: {sign}{trend['price_change_pct']:.2f}%\n"
        f"  Trend: {direction}"
    )


@tool
def check_flight_prices(origin: str, destination: str) -> str:
    """
    Fetch current one-way flight prices for a route from Google Flights via SerpApi.

    Returns the top 3 cheapest options. Requires SERPAPI_KEY in .env.
    Large price spikes vs. historical norms can themselves be a risk signal.

    Args:
        origin:      Departure airport IATA code, e.g. "YYZ".
        destination: Arrival airport IATA code, e.g. "DXB".
    """
    from data.fetch_flights import fetch_flights

    flights = fetch_flights(origin, destination)
    if not flights:
        return (
            f"No flight price data available for {origin}→{destination}. "
            "SERPAPI_KEY may not be set or no results returned."
        )

    cheapest = sorted(flights, key=lambda f: f["price"])[:3]
    lines = [f"Top 3 cheapest flights for {origin}→{destination}:"]
    for i, f in enumerate(cheapest, 1):
        lines.append(
            f"  {i}. {f['airline']}: ${f['price']:.0f} USD "
            f"(departing {f['departure_date']})"
        )
    return "\n".join(lines)


@tool
def analyze_route_risk(origin: str, destination: str) -> str:
    """
    Compute the geopolitical risk score for a flight route. CALL THIS FIRST.

    Traces the great-circle path, identifies every airspace region crossed,
    scores each region using news activity + oil trend + anomaly detection,
    and returns the highest single-region score as the overall route risk.
    Saves the result to the SQLite database for historical tracking.

    Args:
        origin:      Departure airport IATA code, e.g. "YYZ".
        destination: Arrival airport IATA code, e.g. "DXB".
    """
    from data.fetch_events import fetch_events
    from data.fetch_prices import fetch_oil_prices
    from data.fetch_prices import get_oil_trend as _get_oil_trend
    from db.database import engine, init_db, insert_risk_score
    from rag.signals import get_waypoints, score_route

    # Fetch news only for regions this route actually crosses
    waypoints = get_waypoints(origin, destination)
    all_events: list[dict] = []
    for region in waypoints:
        all_events.extend(fetch_events(region))

    prices = fetch_oil_prices()
    oil_trend = _get_oil_trend(prices)

    result = score_route(origin, destination, all_events, oil_trend, prices)

    # Persist so query_historical can track trends
    init_db()
    with Session(engine) as session:
        insert_risk_score(session, {
            "route": result["route"],
            "origin": origin,
            "destination": destination,
            "score": result["score"],
            "summary": (
                f"{result['label']} risk via "
                f"{', '.join(result['waypoints']) or 'no risk regions'}"
            ),
            "date": datetime.utcnow().date(),
        })

    breakdown_str = "\n".join(
        f"    {region}: {score}/100"
        for region, score in sorted(
            result["breakdown"].items(), key=lambda x: -x[1]
        )
    ) or "    (no risk regions on this path)"

    waypoints_str = (
        " → ".join(result["waypoints"]) if result["waypoints"]
        else "No risk regions crossed"
    )

    return (
        f"Route risk analysis: {origin} → {destination}\n"
        f"  Overall score:    {result['score']}/100 ({result['label']})\n"
        f"  Riskiest region:  {result['riskiest_region']}\n"
        f"  Airspace crossed: {waypoints_str}\n"
        f"  Regional breakdown:\n{breakdown_str}"
    )


@tool
def query_historical(route: str) -> str:
    """
    Return the last 5 historical risk scores for a route from the SQLite database.

    Use this to determine whether geopolitical risk on this route is trending
    up, down, or stable over recent assessments.

    Args:
        route: Route string in "ORIGIN-DESTINATION" format, e.g. "YYZ-DXB".
    """
    from db.database import RiskScore, engine, init_db

    init_db()
    with Session(engine) as session:
        stmt = (
            select(RiskScore)
            .where(RiskScore.route == route)
            .order_by(RiskScore.created_at.desc())
            .limit(5)
        )
        rows = list(session.scalars(stmt))

    if not rows:
        return f"No historical risk data found for route {route}."

    lines = [f"Historical risk scores for {route} (most recent first):"]
    for row in rows:
        lines.append(f"  [{row.date}] Score: {row.score}/100 — {row.summary}")

    scores = [row.score for row in rows]
    if len(scores) >= 2:
        if scores[0] > scores[-1]:
            trend = f"INCREASING ({scores[-1]} → {scores[0]})"
        elif scores[0] < scores[-1]:
            trend = f"DECREASING ({scores[-1]} → {scores[0]})"
        else:
            trend = "STABLE"
        lines.append(f"  Trend: {trend}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent  (LangChain 1.x create_agent — LangGraph-based)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are AeroSignal, an expert geopolitical flight intelligence agent.

Your mission: analyze flight routes for geopolitical risk and give travelers
and airlines a clear, data-driven assessment of whether a route is safe to fly.

When asked about a route, follow this sequence:
1. ALWAYS call analyze_route_risk first — this gives you the score and airspace breakdown.
2. Call search_news for the riskiest region(s) to get specific headlines and dates.
3. Call get_oil_trend to check whether rising oil prices are amplifying instability.
4. Call query_historical to determine if risk is improving or deteriorating over time.
5. Call check_flight_prices only if the user asks about cost or pricing anomalies.

In your final response:
- Lead with the risk score and label, e.g. "Risk score: 42/100 — Moderate".
- Name every airspace region crossed and which is the riskiest.
- Cite specific news headlines with dates — never give vague warnings.
- Note the oil price trend and whether it amplifies regional risk.
- Reference historical trend if data is available.
- End with a clear, actionable recommendation: fly / fly with awareness / avoid.

Be direct, specific, and data-driven.\
"""

_tools = [analyze_route_risk, search_news, get_oil_trend, query_historical, check_flight_prices]

agent_executor = create_agent(
    model=llm,
    tools=_tools,
    system_prompt=_SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_agent(query: str) -> str:
    """
    Run the AeroSignal agent on a natural language flight risk query.

    Args:
        query: Natural language question, e.g. "Is YYZ to DXB safe right now?"

    Returns:
        The agent's final response string.
    """
    result = agent_executor.invoke({"messages": [HumanMessage(content=query)]})
    messages = result.get("messages", [])
    if messages:
        return messages[-1].content
    return "No response generated."


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    response = run_agent("Analyze the risk for flying YYZ to DXB")
    print(response)
