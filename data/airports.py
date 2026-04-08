"""
data/airports.py — Loads airport coordinates from the OurAirports free dataset.

Downloads airports.csv on first run and caches it locally. Exposes
AIRPORT_COORDS as a module-level dict mapping IATA codes to (lat, lon) tuples,
covering all large and medium airports worldwide.
"""

import csv
import requests
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

AIRPORTS_CSV_URL = (
    "https://ourairports.com/data/airports.csv"
)
AIRPORTS_CACHE = (
    Path(__file__).parent / "airports_cache.csv"
)


def download_airports_csv() -> None:
    """Download OurAirports dataset if not cached."""
    if AIRPORTS_CACHE.exists():
        return
    response = requests.get(
        AIRPORTS_CSV_URL, timeout=30
    )
    response.raise_for_status()
    AIRPORTS_CACHE.write_text(
        response.text, encoding="utf-8"
    )
    print(f"Downloaded airports database — "
          f"{len(response.text.splitlines())} airports")


def load_airport_coords() -> dict[str, tuple]:
    """
    Load IATA code -> (lat, lon) for all airports
    with valid IATA codes from OurAirports dataset.
    Only includes large and medium airports.
    Returns dict like {"YYZ": (43.68, -79.63)}
    """
    download_airports_csv()
    coords = {}
    with open(
        AIRPORTS_CACHE, encoding="utf-8"
    ) as f:
        reader = csv.DictReader(f)
        for row in reader:
            iata = row.get("iata_code", "").strip()
            airport_type = row.get("type", "")
            if not iata:
                continue
            if airport_type not in [
                "large_airport", "medium_airport"
            ]:
                continue
            try:
                lat = float(row["latitude_deg"])
                lon = float(row["longitude_deg"])
                coords[iata] = (lat, lon)
            except (ValueError, KeyError):
                continue
    print(f"Loaded {len(coords)} airports "
          f"with IATA codes")
    return coords


# Module-level load — happens once on import
AIRPORT_COORDS = load_airport_coords()


if __name__ == "__main__":
    print(f"\nSample lookups:")
    for code in ["YYZ", "DXB", "LHR", "JFK", "BOM"]:
        coord = AIRPORT_COORDS.get(code)
        print(f"  {code}: {coord}")
