"""
data/fetch_events.py — Fetches geopolitical news events from the GDELT Project API.

Queries GDELT's free v2 doc API for articles related to a given region,
parses the response into a standard dict format, and persists results to
the AeroSignal SQLite database via helpers in db/database.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
from datetime import datetime
from typing import Any

import requests
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from db.database import Event, insert_data_source, insert_event

logger = logging.getLogger(__name__)

GDELT_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_TIMEOUT = 15  # seconds


def fetch_events(region: str, days: int = 14) -> list[dict]:
    """
    Fetch geopolitical news articles for a region from GDELT.

    Args:
        region: Airspace region name, e.g. "Gulf", "Eastern Europe".
        days:   Unused — window is controlled by timespan=2w in the URL.
                Kept for API compatibility.

    Returns:
        List of dicts with keys: title, url, region, country,
        date, relevance_score, raw_json.
        Returns an empty list if GDELT is unreachable or returns no articles.
    """
    url = (
        f"{GDELT_BASE_URL}?query={region}+conflict"
        f"&mode=artlist&maxrecords=50&format=json&timespan=2w"
    )

    try:
        response = requests.get(url, timeout=_TIMEOUT)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
    except requests.RequestException as e:
        logger.error("GDELT request failed: %s", e)
        return []
    except ValueError as e:
        logger.error("GDELT response not valid JSON: %s", e)
        return []

    articles = data.get("articles") or []
    articles = [a for a in articles if a.get("language") == "English"]
    if not articles:
        logger.info("GDELT returned 0 English articles for region '%s'", region)
        return []

    region_lower = region.lower()
    events: list[dict] = []
    for article in articles:
        date_str: str = article.get("seendate", "")
        try:
            # GDELT seendate format: "20260323T064500Z"
            date = datetime.strptime(date_str, "%Y%m%dT%H%M%SZ").date()
        except (ValueError, TypeError):
            date = datetime.utcnow().date()

        title: str = article.get("title", "")
        relevance_score = max(1, title.lower().count(region_lower))

        events.append({
            "title": title,
            "url": article.get("url", ""),
            "region": region,
            "country": article.get("sourcecountry", ""),
            "date": date,
            "relevance_score": float(relevance_score),
            "raw_json": json.dumps(article),
        })

    return events


def save_events(events: list[dict], session: Session) -> int:
    """
    Persist a list of event dicts to the database.

    Inserts each event via insert_event (OR IGNORE on duplicate url),
    then logs the fetch to the data_sources audit table with the exact
    count of newly inserted rows.

    Args:
        events:  List of event dicts as returned by fetch_events().
        session: Active SQLAlchemy session.

    Returns:
        Number of new rows actually inserted (duplicates excluded).
    """
    success = True
    error_message = None
    new_count = 0

    try:
        urls = [e["url"] for e in events if e.get("url")]
        existing_count: int = session.execute(
            select(func.count()).where(Event.url.in_(urls))
        ).scalar() or 0

        for event in events:
            insert_event(session, event)

        new_count = len(events) - existing_count
    except Exception as e:
        success = False
        error_message = str(e)
        logger.error("Failed to save events: %s", e)

    insert_data_source(session, {
        "source_name": "gdelt",
        "fetched_at": datetime.utcnow(),
        "record_count": new_count,
        "success": success,
        "error_message": error_message,
    })

    return new_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    events = fetch_events("Gulf")
    print(f"Fetched {len(events)} events")
