"""
data/fetch_events.py — Fetches geopolitical news events from the GDELT Project API.

Queries GDELT's free v2 doc API for articles related to a given region,
parses the response into a standard dict format, and persists results to
the AeroSignal SQLite database via helpers in db/database.py.

Includes SQLite caching (1-day fresh / 30-day stale fallback) and
exponential backoff retry to handle GDELT rate limiting gracefully.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
import time
from datetime import datetime
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from urllib3.util.retry import Retry

from db.database import (
    Event, engine, get_recent_events, init_db,
    insert_data_source, insert_event,
)

logger = logging.getLogger(__name__)

GDELT_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_TIMEOUT = 15  # seconds


def create_session() -> requests.Session:
    """
    Create a requests Session with exponential backoff retry.

    Retries up to 3 times on 429/5xx responses with 2s, 4s, 8s delays.
    """
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


def _events_from_cache(cached: list[Event]) -> list[dict]:
    """Convert ORM Event rows to the standard fetch_events dict format."""
    return [
        {
            "title": e.title,
            "url": e.url,
            "region": e.region,
            "country": e.country,
            "date": str(e.date),
            "relevance_score": e.relevance_score,
            "raw_json": e.raw_json,
        }
        for e in cached
    ]


def fetch_events(region: str, days: int = 14) -> list[dict]:
    """
    Fetch geopolitical news articles for a region from GDELT.

    Checks SQLite cache first (1-day window). Only hits GDELT if the cache
    has fewer than 5 articles. On GDELT failure, falls back to a 30-day
    stale cache rather than returning empty.

    Args:
        region: Airspace region name, e.g. "Gulf", "Eastern Europe".
        days:   Unused — window is controlled by timespan=2w in the URL.
                Kept for API compatibility.

    Returns:
        List of dicts with keys: title, url, region, country,
        date, relevance_score, raw_json.
        Returns an empty list only if GDELT fails and no cache exists.
    """
    init_db()

    # ── Cache check ───────────────────────────────────────────────────────────
    with Session(engine) as session:
        cached = get_recent_events(session, region, days=1)
        if len(cached) >= 5:
            logger.info("Cache hit: %d articles for '%s'", len(cached), region)
            return _events_from_cache(cached)

    # ── GDELT fetch ───────────────────────────────────────────────────────────
    time.sleep(3)  # rate limit protection between calls

    url = (
        f"{GDELT_BASE_URL}?query={region}+conflict"
        f"&mode=artlist&maxrecords=50&format=json&timespan=2w"
    )
    http = create_session()

    data: dict[str, Any] = {}
    for attempt in range(3):
        try:
            response = http.get(url, timeout=_TIMEOUT)
            if response.status_code == 429:
                wait = 10 * (attempt + 1)
                logger.warning(
                    "GDELT 429 — waiting %ds (attempt %d/3)", wait, attempt + 1
                )
                time.sleep(wait)
                continue
            response.raise_for_status()
            data = response.json()
            break
        except (requests.RequestException, ValueError) as e:
            logger.error("GDELT attempt %d failed: %s", attempt + 1, e)
            time.sleep(5)
            continue
    else:
        # All 3 attempts failed — stale cache fallback
        with Session(engine) as session:
            cached = get_recent_events(session, region, days=30)
            if cached:
                logger.warning(
                    "GDELT unavailable — using stale cache (%d articles) for '%s'",
                    len(cached), region,
                )
                return _events_from_cache(cached)
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

    # Persist so the next call within 24 hours hits cache
    with Session(engine) as session:
        save_events(events, session)

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
