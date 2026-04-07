"""
data/fetch_prices.py — Fetches crude oil price history from Yahoo Finance.

Downloads Brent crude (BZ=F) OHLCV data for the past 30 days via yfinance,
persists results to the AeroSignal SQLite database, and exposes a trend
summary dict used by the risk scoring pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from datetime import datetime, timedelta

import yfinance as yf
from sqlalchemy.orm import Session

from db.database import engine, init_db, insert_data_source, insert_oil_price

logger = logging.getLogger(__name__)

OIL_TICKER = "BZ=F"  # Brent crude futures


def fetch_oil_prices(days: int = 30) -> list[dict]:
    """
    Download the last `days` days of Brent crude OHLCV data from Yahoo Finance.

    Returns:
        List of dicts with keys: date, open, high, low, close, volume.
        Returns an empty list if the download fails.
    """
    end = datetime.utcnow()
    start = end - timedelta(days=days)

    try:
        ticker = yf.Ticker(OIL_TICKER)
        df = ticker.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
    except Exception as e:
        logger.error("yfinance download failed: %s", e)
        return []

    if df.empty:
        logger.warning("yfinance returned empty DataFrame for %s", OIL_TICKER)
        return []

    prices: list[dict] = []
    for ts, row in df.iterrows():
        prices.append({
            "date": ts.date(),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row.get("Volume", 0) or 0),
        })

    return prices


def save_oil_prices(prices: list[dict], session: Session) -> int:
    """
    Persist a list of oil price dicts to the database.

    Inserts each row via insert_oil_price (OR IGNORE on duplicate date),
    then logs the fetch to the data_sources audit table.

    Args:
        prices:  List of price dicts as returned by fetch_oil_prices().
        session: Active SQLAlchemy session.

    Returns:
        Number of rows passed in (duplicates silently skipped by DB).
    """
    success = True
    error_message = None
    count = 0

    try:
        for price in prices:
            insert_oil_price(session, price)
        count = len(prices)
    except Exception as e:
        success = False
        error_message = str(e)
        logger.error("Failed to save oil prices: %s", e)

    insert_data_source(session, {
        "source_name": "yfinance",
        "fetched_at": datetime.utcnow(),
        "record_count": count,
        "success": success,
        "error_message": error_message,
    })

    return count


def get_oil_trend(prices: list = None) -> dict:
    """
    Summarise the 30-day oil price trend.

    Args:
        prices: Optional list of price dicts (from fetch_oil_prices).
                If None, fetches fresh data automatically.

    Returns:
        Dict with keys: current_price (float), price_change_pct (float),
        is_rising (bool). Returns zeros/False if prices list is empty.
    """
    if prices is None:
        prices = fetch_oil_prices()

    if not prices:
        return {"current_price": 0.0, "price_change_pct": 0.0, "is_rising": False}

    sorted_prices = sorted(prices, key=lambda p: p["date"])
    current_price = sorted_prices[-1]["close"]
    oldest_price = sorted_prices[0]["close"]

    if oldest_price == 0:
        price_change_pct = 0.0
    else:
        price_change_pct = round((current_price - oldest_price) / oldest_price * 100, 2)

    return {
        "current_price": round(current_price, 2),
        "price_change_pct": price_change_pct,
        "is_rising": price_change_pct > 0,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
    prices = fetch_oil_prices()
    trend = get_oil_trend(prices)
    print(f"Current oil price: {trend['current_price']}")
    print(f"30 day change: {trend['price_change_pct']}%")
    print(f"Rising: {trend['is_rising']}")
