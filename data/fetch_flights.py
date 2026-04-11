"""
data/fetch_flights.py — Fetches live flight prices from Google Flights via SerpApi.

Uses the serpapi GoogleFlightsSearch to retrieve one-way fares for a given
origin/destination IATA pair and departure date. Persists results to the
AeroSignal SQLite database via helpers in db/database.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

from dotenv import load_dotenv
from serpapi import Client as SerpapiClient
from sqlalchemy.orm import Session

from db.database import engine, init_db, insert_data_source, insert_flight

load_dotenv()
logger = logging.getLogger(__name__)

AIRPORT_FALLBACKS: dict[str, str] = {
    "MAA": "BOM",  # Chennai → Mumbai
    "CCU": "DEL",  # Kolkata → Delhi
    "HYD": "BOM",  # Hyderabad → Mumbai
    "COK": "BOM",  # Kochi → Mumbai
}


def get_live_fx_rates() -> dict:
    """Fetch live FX rates from ECB — free, no key needed."""
    try:
        import urllib.request
        url = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml"
        with urllib.request.urlopen(url, timeout=5) as r:
            tree = ET.parse(r)
        root = tree.getroot()
        ns = "{http://www.ecb.int/vocabulary/2002-08-01/eurofxref}"
        rates = {"EUR": 1.0}
        for cube in root.iter(f"{ns}Cube"):
            if "currency" in cube.attrib:
                rates[cube.attrib["currency"]] = float(cube.attrib["rate"])
        # Convert to USD base
        usd_rate = rates.get("USD", 1.0)
        return {k: usd_rate / v for k, v in rates.items()}
    except Exception:
        # Fallback to fixed rates if ECB is unreachable
        return {"USD": 1.0, "CAD": 0.73, "GBP": 1.27, "EUR": 1.08}


def convert_to_usd(price: float, currency: str, rates: dict = None) -> float:
    """Convert price to USD using live ECB rates.
    Falls back to fixed rates if ECB unreachable.

    Pass pre-fetched rates to avoid a redundant HTTP call per flight.
    """
    if rates is None:
        rates = get_live_fx_rates()
    return round(price * rates.get(currency, 1.0), 2)


def fetch_flights(
    origin: str, destination: str, date: str = None
) -> list[dict]:
    """
    Fetch one-way flight prices from Google Flights via SerpApi.

    Args:
        origin:      Departure airport IATA code, e.g. "YYZ".
        destination: Arrival airport IATA code, e.g. "DXB".
        date:        Departure date as "YYYY-MM-DD". Defaults to 2 weeks from today.

    Returns:
        List of dicts with keys: route, origin, destination, price, currency,
        departure_date, airline. Returns an empty list if the API key is
        missing or the request fails.
    """
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        logger.warning(
            "SERPAPI_KEY not set — skipping flight fetch. "
            "Add it to your .env file to enable live prices."
        )
        return []

    if date is None:
        date = (datetime.utcnow() + timedelta(weeks=2)).strftime("%Y-%m-%d")

    route = f"{origin}-{destination}"

    try:
        client = SerpapiClient(api_key=api_key)
        results = client.search({
            "engine": "google_flights",
            "departure_id": origin,
            "arrival_id": destination,
            "outbound_date": date,
            "currency": "USD",
            "type": "2",          # 2 = one-way
        })
    except Exception as e:
        logger.error("SerpApi request failed: %s", e)
        return []

    currency = results.get("search_parameters", {}).get("currency", "USD")

    best_flights = results.get("best_flights") or []
    other_flights = results.get("other_flights") or []
    all_flights = best_flights + other_flights

    if not all_flights:
        logger.info("SerpApi returned 0 flights for %s on %s", route, date)
        return []

    try:
        departure_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        departure_date = datetime.utcnow().date()

    fx_rates = get_live_fx_rates()
    flights: list[dict] = []
    for result in all_flights:
        legs = result.get("flights") or []
        airline = legs[0].get("airline", "Unknown") if legs else "Unknown"
        price = result.get("price")
        if price is None:
            continue
        flights.append({
            "route": route,
            "origin": origin,
            "destination": destination,
            "price_local": float(price),
            "price_usd": convert_to_usd(float(price), currency, fx_rates),
            "currency": currency,
            "departure_date": departure_date,
            "airline": airline,
        })

    return flights


def save_flights(flights: list[dict], session: Session) -> int:
    """
    Persist a list of flight dicts to the database.

    Inserts each row via insert_flight (OR IGNORE on duplicate
    route/departure_date/airline), then logs the fetch to the
    data_sources audit table.

    Args:
        flights: List of flight dicts as returned by fetch_flights().
        session: Active SQLAlchemy session.

    Returns:
        Number of rows passed in (duplicates silently skipped by DB).
    """
    success = True
    error_message = None
    count = 0

    try:
        for flight in flights:
            insert_flight(session, flight)
        count = len(flights)
    except Exception as e:
        success = False
        error_message = str(e)
        logger.error("Failed to save flights: %s", e)

    insert_data_source(session, {
        "source_name": "serpapi_flights",
        "fetched_at": datetime.utcnow(),
        "record_count": count,
        "success": success,
        "error_message": error_message,
    })

    return count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
    flights = fetch_flights("YYZ", "DXB")
    print(f"Fetched {len(flights)} flights")
    for f in flights[:3]:
        print(f"{f['airline']}: {f['currency']} {f['price_local']} (${f['price_usd']} USD)")
