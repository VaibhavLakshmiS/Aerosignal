"""
rag/signals.py — Route risk scoring engine for AeroSignal.

Computes a geopolitical risk score for a flight route by interpolating
points along the great circle path, checking which risk region bounding
boxes each point falls inside, then combining news event counts, oil price
trends, and price anomaly detection into a 0-100 score per region.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

import numpy as np
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Region bounding boxes  (lat_min, lat_max), (lng_min, lng_max)
# ---------------------------------------------------------------------------

REGION_BOXES: dict[str, dict] = {
    "Gulf":                   {"lat": (22, 30),  "lng": (48, 60)},
    "Eastern Europe":         {"lat": (44, 55),  "lng": (22, 40)},
    "Eastern Mediterranean":  {"lat": (30, 42),  "lng": (25, 37)},
    "Russia":                 {"lat": (50, 75),  "lng": (30, 180)},
    "Pakistan":               {"lat": (23, 37),  "lng": (60, 77)},
    "India":                  {"lat": (8, 37),   "lng": (68, 97)},
    "Southeast Asia":         {"lat": (1, 20),   "lng": (100, 120)},
    "East Asia":              {"lat": (20, 45),  "lng": (110, 145)},
    "North Africa":           {"lat": (15, 35),  "lng": (-5, 35)},
    "Central Asia":           {"lat": (35, 55),  "lng": (55, 80)},
}

# ---------------------------------------------------------------------------
# Airport coordinates  (lat, lng)
# ---------------------------------------------------------------------------

AIRPORT_COORDS: dict[str, tuple[float, float]] = {
    "YYZ": (43.68, -79.63),  "JFK": (40.64, -73.78),
    "LHR": (51.48,  -0.46),  "CDG": (49.01,   2.55),
    "DXB": (25.25,  55.36),  "BOM": (19.09,  72.87),
    "DEL": (28.56,  77.10),  "SIN": ( 1.36, 103.99),
    "NRT": (35.77, 140.39),  "SVO": (55.97,  37.41),
    "TLV": (32.01,  34.89),  "IST": (41.27,  28.74),
    "DOH": (25.27,  51.61),  "LAX": (33.94, -118.41),
    "ORD": (41.98, -87.91),  "YVR": (49.19, -123.18),
    "YUL": (45.47, -73.74),  "SYD": (-33.95, 151.18),
    "MEL": (-37.67, 144.84), "GRU": (-23.43, -46.47),
    "EZE": (-34.82, -58.54),
}

_N_WAYPOINTS = 20


def get_waypoints(origin: str, destination: str) -> list[str]:
    """
    Return the list of risk regions crossed on the great circle path.

    Interpolates _N_WAYPOINTS points between origin and destination using
    spherical linear interpolation, then checks each point against
    REGION_BOXES. Duplicates are removed while preserving order.

    Args:
        origin:      IATA code of the departure airport.
        destination: IATA code of the arrival airport.

    Returns:
        Ordered, deduplicated list of region names the route passes through.
        Returns ["Unknown"] if either airport is not in AIRPORT_COORDS.
    """
    if origin not in AIRPORT_COORDS or destination not in AIRPORT_COORDS:
        logger.warning("Unknown airport(s): %s, %s", origin, destination)
        return ["Unknown"]

    lat1, lng1 = AIRPORT_COORDS[origin]
    lat2, lng2 = AIRPORT_COORDS[destination]

    # Convert to radians for slerp
    lat1r, lng1r = np.radians(lat1), np.radians(lng1)
    lat2r, lng2r = np.radians(lat2), np.radians(lng2)

    # Cartesian unit vectors on the sphere
    def to_xyz(latr: float, lngr: float) -> np.ndarray:
        return np.array([
            np.cos(latr) * np.cos(lngr),
            np.cos(latr) * np.sin(lngr),
            np.sin(latr),
        ])

    p1 = to_xyz(lat1r, lng1r)
    p2 = to_xyz(lat2r, lng2r)

    dot = float(np.clip(np.dot(p1, p2), -1.0, 1.0))
    omega = np.arccos(dot)

    regions_seen: list[str] = []
    seen_set: set[str] = set()

    for t in np.linspace(0, 1, _N_WAYPOINTS):
        if abs(omega) < 1e-10:
            pt = p1
        else:
            pt = (np.sin((1 - t) * omega) * p1 + np.sin(t * omega) * p2) / np.sin(omega)

        lat_deg = float(np.degrees(np.arcsin(np.clip(pt[2], -1.0, 1.0))))
        lng_deg = float(np.degrees(np.arctan2(pt[1], pt[0])))

        for region, box in REGION_BOXES.items():
            if (box["lat"][0] <= lat_deg <= box["lat"][1] and
                    box["lng"][0] <= lng_deg <= box["lng"][1]):
                if region not in seen_set:
                    regions_seen.append(region)
                    seen_set.add(region)

    return regions_seen if regions_seen else []


def detect_anomaly(prices: list[dict]) -> bool:
    """
    Detect whether the most recent oil price is anomalous using IsolationForest.

    Args:
        prices: List of price dicts with a "close" key (from fetch_oil_prices).

    Returns:
        True if the latest price is flagged as an outlier, False otherwise.
        Always returns False if fewer than 10 data points are available.
    """
    if len(prices) < 10:
        return False

    closes = np.array([p["close"] for p in prices]).reshape(-1, 1)
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(closes)
    prediction = model.predict(closes[-1].reshape(1, -1))
    return bool(prediction[0] == -1)


def score_region(
    region: str,
    events: list[dict],
    oil_trend: dict,
    anomaly: bool,
) -> float:
    """
    Compute a 0-100 risk score for a single airspace region.

    Scoring breakdown:
      - news_score:    5 pts per relevant event, capped at 50
      - oil_score:     2 × |price_change_pct|, capped at 30 — only applied
                       when oil is rising AND the region has news activity,
                       making oil relevance data-driven rather than hardcoded
      - anomaly_score: 20 — only applied when the region has news activity,
                       so anomalies only amplify already-active regions

    Args:
        region:    Region name (used to filter events by region field).
        events:    List of event dicts from fetch_events / search_events.
        oil_trend: Dict with keys price_change_pct (float) and is_rising (bool).
        anomaly:   Whether the latest oil price is anomalous.

    Returns:
        Total risk score rounded to 1 decimal place.
    """
    region_events = [e for e in events if e.get("region", "") == region]
    news_score = min(len(region_events) * 5, 50)

    if oil_trend.get("is_rising") and len(region_events) > 0:
        oil_score = min(abs(oil_trend.get("price_change_pct", 0)) * 2, 30)
    else:
        oil_score = 0.0

    anomaly_score = 20.0 if (anomaly and len(region_events) > 0) else 0.0

    return round(news_score + oil_score + anomaly_score, 1)


def label_from_score(score: float) -> str:
    """
    Convert a numeric risk score to a human-readable label.

    Args:
        score: Risk score in range 0-100.

    Returns:
        One of: "Low", "Moderate", "High", "Very High".
    """
    if score <= 25:
        return "Low"
    elif score <= 50:
        return "Moderate"
    elif score <= 75:
        return "High"
    else:
        return "Very High"


def score_route(
    origin: str,
    destination: str,
    events: list[dict],
    oil_trend: dict,
    prices: list[dict],
) -> dict:
    """
    Score a full flight route by checking every airspace region it crosses.

    Interpolates the great circle path, scores each region the path passes
    through, and returns the highest single-region score as the overall
    route risk.

    Args:
        origin:      IATA departure code, e.g. "YYZ".
        destination: IATA arrival code, e.g. "DXB".
        events:      List of geopolitical event dicts from fetch_events.
        oil_trend:   Dict from get_oil_trend with price_change_pct, is_rising.
        prices:      Raw price list from fetch_oil_prices (for anomaly detection).

    Returns:
        Dict with keys:
          route (str), score (float), riskiest_region (str),
          waypoints (list[str]), breakdown (dict[str, float]), label (str).
    """
    waypoints = get_waypoints(origin, destination)
    anomaly = detect_anomaly(prices)

    breakdown: dict[str, float] = {}
    for region in waypoints:
        breakdown[region] = score_region(region, events, oil_trend, anomaly)

    if breakdown:
        riskiest_region = max(breakdown, key=lambda r: breakdown[r])
        top_score = breakdown[riskiest_region]
    else:
        riskiest_region = "Unknown"
        top_score = 0.0

    return {
        "route": f"{origin}-{destination}",
        "score": top_score,
        "riskiest_region": riskiest_region,
        "waypoints": waypoints,
        "breakdown": breakdown,
        "label": label_from_score(top_score),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from data.fetch_prices import fetch_oil_prices, get_oil_trend
    prices = fetch_oil_prices()
    trend = get_oil_trend(prices)

    sample_events = [
        {"title": "Gulf tensions escalate", "region": "Gulf", "url": "http://test.com/1"},
        {"title": "Oil tanker attacked in Strait of Hormuz", "region": "Gulf", "url": "http://test.com/2"},
        {"title": "Iran threatens shipping lanes", "region": "Gulf", "url": "http://test.com/3"},
    ]

    result = score_route("YYZ", "DXB", sample_events, trend, prices)
    print(f"Route: {result['route']}")
    print(f"Score: {result['score']} — {result['label']}")
    print(f"Waypoints: {result['waypoints']}")
    print(f"Riskiest: {result['riskiest_region']}")
    print(f"Breakdown: {result['breakdown']}")
