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
import math

from data.airports import AIRPORT_COORDS

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

REGION_CENTERS: dict[str, tuple[float, float]] = {
    "Gulf":                   (26.0,  50.5),
    "Eastern Europe":         (50.0,  30.0),
    "Russia":                 (60.0,  60.0),
    "Eastern Mediterranean":  (36.0,  33.0),
    "Pakistan":               (30.0,  69.0),
    "India":                  (22.0,  80.0),
    "Southeast Asia":         (10.0, 110.0),
    "East Asia":              (35.0, 125.0),
    "North Africa":           (25.0,  15.0),
    "Central Asia":           (45.0,  65.0),
}

HUB_ALTERNATIVES = {
    "Gulf": {
        "hubs": ["IST", "FRA", "LHR", "DOH"],
        "reason": (
            "Gulf disruptions reroute traffic through "
            "European and Turkish hubs"
        )
    },
    "Eastern Europe": {
        "hubs": ["FRA", "VIE", "WAW", "HEL"],
        "reason": (
            "Eastern European disruptions push traffic "
            "through Central European hubs"
        )
    },
    "Eastern Mediterranean": {
        "hubs": ["IST", "ATH", "FCO"],
        "reason": (
            "Mediterranean disruptions reroute through "
            "Istanbul and Southern European hubs"
        )
    },
    "Russia": {
        "hubs": ["HEL", "RIX", "TLL", "WAW"],
        "reason": (
            "Russian airspace closures push traffic "
            "through Nordic and Baltic hubs"
        )
    },
    "Pakistan": {
        "hubs": ["IST", "DXB", "DOH", "AUH"],
        "reason": (
            "Pakistan disruptions reroute traffic "
            "through Gulf and Turkish hubs"
        )
    },
    "Central Asia": {
        "hubs": ["IST", "DXB", "FRA"],
        "reason": (
            "Central Asian disruptions reroute through "
            "Gulf and European hubs"
        )
    },
}

AIRPORT_NAMES = {
    "IST": "Istanbul",
    "FRA": "Frankfurt",
    "LHR": "London Heathrow",
    "CDG": "Paris CDG",
    "DOH": "Doha",
    "DXB": "Dubai",
    "VIE": "Vienna",
    "WAW": "Warsaw",
    "HEL": "Helsinki",
    "RIX": "Riga",
    "TLL": "Tallinn",
    "ATH": "Athens",
    "FCO": "Rome",
    "AUH": "Abu Dhabi",
}

# Cascade impact methodology:
# Based on aviation economics literature:
# - Cook & Tanner (2015) EUROCONTROL delay cost study
# - ICAO Circular 304 rerouting pattern analysis
# Demand and fare figures are directional estimates.
# Calibrate against live EUROCONTROL data in production.
CASCADE_DEMAND_COEFFICIENT = 0.15
CASCADE_FARE_COEFFICIENT = 0.08

_N_WAYPOINTS = 20

# ---------------------------------------------------------------------------
# Fare-forecasting constants
# ---------------------------------------------------------------------------

# Based on IATA published fuel cost methodology
# Fuel = ~25% of airline operating costs (consistent
# across years). Correlation derived from:
# - IATA Economics: "The Impact of Fuel on Airline
#   Operating Costs" (stable relationship 2010-present)
# - Short haul less sensitive because fuel is smaller
#   share of total cost vs long haul
# Values: 10% oil rise = X% fare rise
# short: 1.5%, medium: 2.5%, long: 3.5%
OIL_FARE_CORRELATION = {
    "short_haul": 0.15,
    "medium_haul": 0.25,
    "long_haul": 0.35,
}
NEWS_DECAY_RATE = 0.85
OIL_MOMENTUM_WINDOW = 7


# ---------------------------------------------------------------------------
# Existing route geometry
# ---------------------------------------------------------------------------

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

    lat1r, lng1r = np.radians(lat1), np.radians(lng1)
    lat2r, lng2r = np.radians(lat2), np.radians(lng2)

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


# ---------------------------------------------------------------------------
# Distance and route classification
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Return the great-circle distance in km between two coordinates.

    Args:
        lat1, lon1: Latitude/longitude of the first point in decimal degrees.
        lat2, lon2: Latitude/longitude of the second point in decimal degrees.
    """
    R = 6371.0
    lat1r, lon1r = math.radians(lat1), math.radians(lon1)
    lat2r, lon2r = math.radians(lat2), math.radians(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def get_route_type(origin: str, destination: str) -> str:
    """
    Classify a route as 'short_haul', 'medium_haul', or 'long_haul'.

    Uses AIRPORT_COORDS and haversine_km to calculate distance, then
    estimates flight hours at 900 km/h cruise speed.

    Args:
        origin:      IATA departure code.
        destination: IATA arrival code.

    Returns:
        'short_haul'  (<3 hrs),
        'medium_haul' (3-8 hrs),
        'long_haul'   (>8 hrs or airport unknown).
    """
    if origin not in AIRPORT_COORDS or destination not in AIRPORT_COORDS:
        return "long_haul"
    lat1, lon1 = AIRPORT_COORDS[origin]
    lat2, lon2 = AIRPORT_COORDS[destination]
    flight_hours = haversine_km(lat1, lon1, lat2, lon2) / 900
    if flight_hours < 3:
        return "short_haul"
    elif flight_hours <= 8:
        return "medium_haul"
    return "long_haul"


# ---------------------------------------------------------------------------
# Oil momentum and 7-day forecast
# ---------------------------------------------------------------------------

def get_oil_momentum(prices: list) -> float:
    """
    Return the average daily rate of change (as a decimal) over the last 7 days.

    Args:
        prices: List of price dicts with 'date' and 'close' keys.

    Returns:
        Daily rate of change as a decimal (e.g. 0.005 = 0.5%/day).
        Returns 0.0 if fewer than 2 data points are available.
    """
    if len(prices) < 2:
        return 0.0
    recent = sorted(prices, key=lambda p: p["date"])[-OIL_MOMENTUM_WINDOW:]
    if len(recent) < 2:
        return 0.0
    start_price = recent[0]["close"]
    if start_price == 0:
        return 0.0
    return (recent[-1]["close"] - start_price) / start_price / (len(recent) - 1)


def forecast_route(
    origin: str,
    destination: str,
    current_result: dict,
    oil_prices: list,
    current_fare: float = None,
) -> list[dict]:
    """
    Compute a 7-day risk and fare forecast for a route.

    Decomposes today's score into news, oil, and anomaly components,
    then projects each forward using NEWS_DECAY_RATE decay, oil momentum,
    and OIL_FARE_CORRELATION to estimate fare movement.

    Args:
        origin:         IATA departure code.
        destination:    IATA arrival code.
        current_result: Output dict from score_route().
        oil_prices:     Raw price list from fetch_oil_prices().
        current_fare:   Current fare in USD; projected fares omitted if None.

    Returns:
        List of 8 dicts (day 0-7), each with keys: day, date, score, label,
        projected_fare, projected_fare_currency, oil_change_pct, trend.
    """
    from datetime import date, timedelta

    forecast: list[dict] = []
    for N in range(0, 8):
        news_component = (current_result["score"] * 0.5) * (NEWS_DECAY_RATE ** N)

        momentum = get_oil_momentum(oil_prices)
        projected_oil_change = momentum * N
        oil_component = min(
            current_result["score"] * 0.3 * (1 + projected_oil_change * 10), 30
        )

        anomaly_component = (current_result["score"] * 0.2) * max(0, 1 - (N / 3))

        projected_score = min(news_component + oil_component + anomaly_component, 100)

        if current_fare is not None:
            route_type = get_route_type(origin, destination)
            fare_impact = projected_oil_change * OIL_FARE_CORRELATION[route_type]
            projected_fare = round(current_fare * (1 + fare_impact), 0)
        else:
            projected_fare = None

        day_date = date.today() + timedelta(days=N)

        forecast.append({
            "day": N,
            "date": day_date.isoformat(),
            "score": round(projected_score, 1),
            "label": label_from_score(projected_score),
            "projected_fare": projected_fare,
            "projected_fare_currency": "USD",
            "oil_change_pct": round(projected_oil_change * 100, 1),
            "trend": (
                "rising" if projected_score > current_result["score"]
                else "falling" if projected_score < current_result["score"]
                else "stable"
            ),
        })

    return forecast


def get_best_booking_day(forecast: list[dict]) -> dict:
    """
    Return the forecast day with the lowest projected fare.

    Falls back to the lowest risk score day if no fare data is present.

    Args:
        forecast: Output list from forecast_route().
    """
    fare_days = [d for d in forecast if d["projected_fare"] is not None]
    if fare_days:
        return min(fare_days, key=lambda d: d["projected_fare"])
    return min(forecast, key=lambda d: d["score"])


def get_forecast_summary(
    forecast: list[dict],
    origin: str,
    destination: str,
    current_fare: float = None,
) -> str:
    """
    Return a 2-3 sentence human-readable summary of the 7-day forecast.

    Includes best booking day, peak risk day, and fare difference if
    current_fare is provided.

    Args:
        forecast:     Output list from forecast_route().
        origin:       IATA departure code.
        destination:  IATA arrival code.
        current_fare: Current fare in USD; fare comparison omitted if None.
    """
    best = get_best_booking_day(forecast)
    peak = max(forecast, key=lambda d: d["score"])
    direction = "decline" if forecast[-1]["score"] < forecast[0]["score"] else "remain elevated"

    summary = (
        f"The {origin}->{destination} route shows {forecast[0]['label'].lower()} risk today "
        f"(score {forecast[0]['score']}/100), peaking at {peak['score']}/100 on {peak['date']}. "
        f"Risk is expected to {direction} over the next 7 days."
    )

    if current_fare is not None and best["projected_fare"] is not None:
        fare_diff = round(best["projected_fare"] - current_fare, 0)
        direction_word = "saving" if fare_diff < 0 else "at a premium of"
        summary += (
            f" Best booking day is {best['date']} "
            f"({direction_word} ${abs(fare_diff):.0f} vs today's ${current_fare:.0f})."
        )

    return summary


def airport_in_region(iata: str, region: str) -> bool:
    """
    Return True if the airport's coordinates fall inside the region's bounding box.

    Uses AIRPORT_COORDS and REGION_BOXES — no hardcoded lists.

    Args:
        iata:   IATA airport code.
        region: Region name matching a key in REGION_BOXES.
    """
    if iata not in AIRPORT_COORDS:
        return False
    lat, lon = AIRPORT_COORDS[iata]
    box = REGION_BOXES.get(region, {})
    if not box:
        return False
    lat_min, lat_max = box["lat"]
    lng_min, lng_max = box["lng"]
    return lat_min <= lat <= lat_max and lng_min <= lon <= lng_max


def detect_cascade_risk(
    risk_result: dict,
    origin: str,
    destination: str,
) -> list[dict]:
    """
    Detect second-order cascade effects from geopolitical disruptions.

    When a high-risk region disrupts primary routes, passengers reroute
    through alternative hub airports — driving secondary demand and fare
    increases on otherwise unaffected routes.

    Demand and fare impacts are directional estimates based on aviation
    economics literature. Not precise forecasts — confidence is stated
    explicitly in each returned entry.

    Only triggers for regions scoring above 50.

    Args:
        risk_result: Output dict from score_route().
        origin:      IATA departure code of the primary route.
        destination: IATA arrival code of the primary route.

    Returns:
        List of cascade dicts sorted by trigger_score descending.
        Empty list if no region exceeds the threshold.
    """
    # Debug: confirm DOH exclusion logic is working
    print(f"DOH in Gulf: {airport_in_region('DOH', 'Gulf')}")
    print(f"DOH coords: {AIRPORT_COORDS.get('DOH')}")
    print(f"Gulf box: {REGION_BOXES.get('Gulf')}")
    for _r, _s in risk_result["breakdown"].items():
        if _s >= 50 and _r in HUB_ALTERNATIVES:
            _raw = HUB_ALTERNATIVES[_r]["hubs"]
            _kept = [h for h in _raw
                     if h not in [origin, destination]
                     and not airport_in_region(h, _r)]
            print(f"CASCADE {_r} ({_s}): {_raw} → kept {_kept}")

    cascades = []

    for region, score in risk_result["breakdown"].items():
        if score < 50:
            continue
        if region not in HUB_ALTERNATIVES:
            continue

        hub_data = HUB_ALTERNATIVES[region]
        # Skip hubs that are origin/destination or sit inside the
        # disrupted region itself — they can't serve as safe alternatives
        hubs = [
            h for h in hub_data["hubs"]
            if h not in [origin, destination]
            and not airport_in_region(h, region)
        ]

        for hub in hubs:
            demand_pct = round(score * CASCADE_DEMAND_COEFFICIENT, 1)
            fare_pct   = round(score * CASCADE_FARE_COEFFICIENT, 1)

            # Scale impact by hub's distance from the disrupted region.
            # Closer hubs absorb more diverted traffic → higher impact.
            hub_coords    = AIRPORT_COORDS.get(hub)
            region_center = REGION_CENTERS.get(region)
            if hub_coords and region_center:
                dist = haversine_km(
                    hub_coords[0], hub_coords[1],
                    region_center[0], region_center[1],
                )
                if dist < 1000:
                    distance_multiplier = 1.5
                elif dist < 3000:
                    distance_multiplier = 1.0
                else:
                    distance_multiplier = 0.6
                demand_pct = round(demand_pct * distance_multiplier, 1)
                fare_pct   = round(fare_pct   * distance_multiplier, 1)

            cascades.append({
                "trigger_region": region,
                "trigger_score": score,
                "trigger_label": label_from_score(score),
                "affected_hub": hub,
                "affected_hub_name": AIRPORT_NAMES.get(hub, hub),
                "demand_increase_pct": demand_pct,
                "fare_impact_pct": fare_pct,
                "reason": hub_data["reason"],
                "severity": "High" if score >= 75 else "Moderate",
                "confidence": (
                    "Directional estimate — "
                    "not a precise forecast"
                ),
            })

    # Deduplicate by hub — keep the entry with the highest trigger_score
    seen_hubs: dict[str, dict] = {}
    for cascade in cascades:
        hub = cascade["affected_hub"]
        if hub not in seen_hubs or cascade["trigger_score"] > seen_hubs[hub]["trigger_score"]:
            seen_hubs[hub] = cascade
    return sorted(seen_hubs.values(), key=lambda x: x["trigger_score"], reverse=True)


# ---------------------------------------------------------------------------
# Existing scoring functions
# ---------------------------------------------------------------------------

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

    # Apply destination-proximity weighting directly to breakdown.
    # Regions close to the destination are amplified; far ones penalised.
    # Thresholds chosen so Eastern Europe (~3500km from DXB) gets 1.0×
    # while Gulf (~490km from DXB) gets 1.8×, preventing Ukraine war
    # news volume from outranking the actual destination region.
    if destination in AIRPORT_COORDS:
        dest_lat, dest_lon = AIRPORT_COORDS[destination]
    else:
        dest_lat, dest_lon = None, None

    for region in list(breakdown):
        if dest_lat is not None and region in REGION_CENTERS:
            region_lat, region_lon = REGION_CENTERS[region]
            dist = haversine_km(dest_lat, dest_lon, region_lat, region_lon)
            if dist < 1500:
                multiplier = 1.8
            elif dist < 3000:
                multiplier = 1.3
            elif dist < 5000:
                multiplier = 1.0
            else:
                multiplier = 0.6
            breakdown[region] = round(breakdown[region] * multiplier, 1)

    if breakdown:
        riskiest_region = max(breakdown, key=breakdown.get)
        top_score = breakdown[riskiest_region]
    else:
        riskiest_region = "Unknown"
        top_score = 0.0

    result = {
        "route": f"{origin}-{destination}",
        "score": top_score,
        "riskiest_region": riskiest_region,
        "waypoints": waypoints,
        "breakdown": breakdown,
        "label": label_from_score(top_score),
    }
    cascades = detect_cascade_risk(result, origin, destination)
    result["cascade_risks"] = cascades
    result["has_cascade"] = len(cascades) > 0
    return result


HISTORICAL_EVENTS = [
    {
        "name": "Ukraine invasion",
        "date": "2022-02-24",
        "route": ("LHR", "KBP"),
        "region": "Eastern Europe",
        "expected_direction": "high",
    },
    {
        "name": "Iran attacks Israel",
        "date": "2024-10-01",
        "route": ("LHR", "TLV"),
        "region": "Eastern Mediterranean",
        "expected_direction": "high",
    },
    {
        "name": "Gaza conflict start",
        "date": "2023-10-07",
        "route": ("LHR", "TLV"),
        "region": "Eastern Mediterranean",
        "expected_direction": "high",
    },
]


def validate_model() -> list[dict]:
    """
    Backtest the risk model against known historical geopolitical events.

    For each event in HISTORICAL_EVENTS:
    1. Fetch oil prices for the 30 days before the event via yfinance.
    2. Calculate the oil trend and anomaly at that point in time.
    3. Run score_region with 3 simulated early-warning news signals.
    4. Check whether the model correctly flags elevated risk (score >= 30).
    5. Fetch oil prices for 14 days after the event and record actual change.

    Returns:
        List of result dicts with keys: event, date, route, region,
        pre_event_score, correctly_flagged, actual_oil_change_pct, oil_at_event.
    """
    import yfinance as yf
    from datetime import datetime, timedelta
    from data.fetch_prices import get_oil_trend

    results: list[dict] = []

    for event in HISTORICAL_EVENTS:
        event_date = datetime.strptime(event["date"], "%Y-%m-%d")

        start = event_date - timedelta(days=35)
        end   = event_date - timedelta(days=1)

        ticker = yf.Ticker("CL=F")
        hist   = ticker.history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
        )

        if len(hist) < 5:
            continue

        prices = [
            {
                "date":   str(idx.date()),
                "close":  round(float(row["Close"]), 2),
                "open":   round(float(row["Open"]),  2),
                "high":   round(float(row["High"]),  2),
                "low":    round(float(row["Low"]),   2),
                "volume": float(row["Volume"]),
            }
            for idx, row in hist.iterrows()
        ]

        trend   = get_oil_trend(prices)
        anomaly = detect_anomaly(prices)

        simulated_events = [
            {"region": event["region"], "title": f"{event['name']} early warning"}
        ] * 3

        pre_score = score_region(event["region"], simulated_events, trend, anomaly)

        after_hist = ticker.history(
            start=event_date.strftime("%Y-%m-%d"),
            end=(event_date + timedelta(days=14)).strftime("%Y-%m-%d"),
        )

        if len(after_hist) >= 2:
            price_before       = float(hist["Close"].iloc[-1])
            price_after        = float(after_hist["Close"].iloc[-1])
            actual_oil_change  = round(
                (price_after - price_before) / price_before * 100, 1
            )
        else:
            actual_oil_change = None

        results.append({
            "event":                event["name"],
            "date":                 event["date"],
            "route":                f"{event['route'][0]}-{event['route'][1]}",
            "region":               event["region"],
            "pre_event_score":      round(pre_score, 1),
            "correctly_flagged":    pre_score >= 30,
            "actual_oil_change_pct": actual_oil_change,
            "oil_at_event":         round(float(hist["Close"].iloc[-1]), 2),
        })

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from data.fetch_prices import fetch_oil_prices, get_oil_trend
    prices = fetch_oil_prices()
    trend = get_oil_trend(prices)
    sample_result = {
        "route": "YYZ-DXB",
        "score": 65.0,
        "label": "High",
        "riskiest_region": "Gulf",
        "waypoints": ["Eastern Europe", "Russia", "Gulf"],
        "breakdown": {"Eastern Europe": 0.0, "Russia": 0.0, "Gulf": 65.0},
    }
    forecast = forecast_route("YYZ", "DXB", sample_result, prices, 732.0)
    print("7-day forecast:")
    for day in forecast:
        fare = f"  ${day['projected_fare']}" if day["projected_fare"] else ""
        print(
            f"  {day['date']}  Score: {day['score']:5.1f}  "
            f"{day['label']:12}{fare}"
        )
    best = get_best_booking_day(forecast)
    summary = get_forecast_summary(forecast, "YYZ", "DXB", 732.0)
    print(f"\nBest day: {best['date']}")
    print(f"Summary: {summary}")

    cascades = detect_cascade_risk(sample_result, "YYZ", "DXB")
    print(f"\nCascade risks: {len(cascades)}")
    for c in cascades:
        print(
            f"  {c['trigger_region']} -> "
            f"{c['affected_hub_name']} "
            f"({c['affected_hub']}): "
            f"demand +{c['demand_increase_pct']}% "
            f"fares +{c['fare_impact_pct']}% "
            f"-- {c['severity']}"
        )

    print("\n--- Backtest ---")
    results = validate_model()
    print(f"{'Event':<25} {'Score':<8} {'Flagged':<18} {'Oil Δ after'}")
    print("-" * 60)
    for r in results:
        flagged = "✓ Yes" if r["correctly_flagged"] else "✗ No"
        oil     = (
            f"{r['actual_oil_change_pct']:+.1f}%"
            if r["actual_oil_change_pct"] is not None else "N/A"
        )
        print(f"{r['event']:<25} {r['pre_event_score']:<8} {flagged:<18} {oil}")
    correct = sum(1 for r in results if r["correctly_flagged"])
    print(f"\nAccuracy: {correct}/{len(results)} events correctly flagged")
