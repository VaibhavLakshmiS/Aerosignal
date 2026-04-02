"""
db/database.py — SQLAlchemy setup for AeroSignal.

Creates and manages a local SQLite database (aerosignal.db inside db/) with
five tables: events (geopolitical news), oil_prices (OHLCV history),
risk_scores (per-route AI assessments), flights (scraped price snapshots),
and data_sources (fetch audit log). Provides insert and query helpers for
the rest of the data layer.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    Boolean, Column, Date, DateTime, Float, Integer, String, Text,
    UniqueConstraint, create_engine, select,
)
from sqlalchemy.orm import DeclarativeBase, Session


# aerosignal.db lives inside the db/ folder alongside this file
_DB_PATH = Path(__file__).parent / "aerosignal.db"
DATABASE_URL = f"sqlite:///{_DB_PATH}"
engine = create_engine(DATABASE_URL, echo=False)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Table definitions
# ---------------------------------------------------------------------------

class Event(Base):
    __tablename__ = "events"
    __table_args__ = (
        UniqueConstraint("url", name="uq_events_url"),
    )

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    title: str = Column(String, nullable=False)
    url: str = Column(String)
    region: str = Column(String)
    country: str = Column(String)
    date: datetime = Column(Date, default=lambda: datetime.utcnow().date())
    relevance_score: float = Column(Float)
    raw_json: str = Column(Text)
    created_at: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)


class OilPrice(Base):
    __tablename__ = "oil_prices"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    date: datetime = Column(Date, unique=True, nullable=False)
    open: float = Column(Float)
    high: float = Column(Float)
    low: float = Column(Float)
    close: float = Column(Float)
    volume: float = Column(Float)
    created_at: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)


class RiskScore(Base):
    __tablename__ = "risk_scores"
    __table_args__ = (
        UniqueConstraint("route", "date", name="uq_risk_route_date"),
    )

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    route: str = Column(String, nullable=False)
    origin: str = Column(String)
    destination: str = Column(String)
    score: float = Column(Float)
    summary: str = Column(Text)
    date: datetime = Column(Date, default=lambda: datetime.utcnow().date())
    created_at: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)


class Flight(Base):
    __tablename__ = "flights"
    __table_args__ = (
        UniqueConstraint("route", "departure_date", "airline", name="uq_flights_route_date_airline"),
    )

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    route: str = Column(String, nullable=False)
    origin: str = Column(String)
    destination: str = Column(String)
    price: float = Column(Float)
    currency: str = Column(String)
    departure_date: datetime = Column(Date, default=lambda: datetime.utcnow().date())
    airline: str = Column(String)
    created_at: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)


class DataSource(Base):
    __tablename__ = "data_sources"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    source_name: str = Column(String, nullable=False)
    fetched_at: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)
    record_count: int = Column(Integer)
    success: bool = Column(Boolean, nullable=False)
    error_message: str = Column(Text)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(engine)


# ---------------------------------------------------------------------------
# Insert helpers  (OR IGNORE via SQLite dialect for duplicate safety)
# ---------------------------------------------------------------------------

def insert_event(session: Session, event_dict: dict) -> None:
    """Insert a geopolitical event, silently ignoring duplicates (by url)."""
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert
    stmt = sqlite_insert(Event).values(**event_dict).prefix_with("OR IGNORE")
    session.execute(stmt)
    session.commit()


def insert_oil_price(session: Session, price_dict: dict) -> None:
    """Insert an oil price OHLCV record, silently ignoring duplicate dates."""
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert
    stmt = sqlite_insert(OilPrice).values(**price_dict).prefix_with("OR IGNORE")
    session.execute(stmt)
    session.commit()


def insert_risk_score(session: Session, score_dict: dict) -> None:
    """Insert a risk score record, silently ignoring duplicate (route, date)."""
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert
    stmt = sqlite_insert(RiskScore).values(**score_dict).prefix_with("OR IGNORE")
    session.execute(stmt)
    session.commit()


def insert_flight(session: Session, flight_dict: dict) -> None:
    """Insert a flight price snapshot, silently ignoring duplicate (route, departure_date, airline)."""
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert
    stmt = sqlite_insert(Flight).values(**flight_dict).prefix_with("OR IGNORE")
    session.execute(stmt)
    session.commit()


def insert_data_source(session: Session, source_dict: dict) -> None:
    """Log a fetch attempt to the data_sources audit table."""
    record = DataSource(**source_dict)
    session.add(record)
    session.commit()


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_recent_events(
    session: Session, region: str, days: int = 14
) -> list[Event]:
    """Return events for a given region within the last `days` days."""
    cutoff = datetime.utcnow().date() - timedelta(days=days)
    stmt = (
        select(Event)
        .where(Event.region == region)
        .where(Event.date >= cutoff)
        .order_by(Event.date.desc())
    )
    return list(session.scalars(stmt))


def get_oil_history(session: Session, days: int = 30) -> list[OilPrice]:
    """Return oil price rows for the last `days` days, newest first."""
    cutoff = datetime.utcnow().date() - timedelta(days=days)
    stmt = (
        select(OilPrice)
        .where(OilPrice.date >= cutoff)
        .order_by(OilPrice.date.desc())
    )
    return list(session.scalars(stmt))


def get_risk_score(
    session: Session, route: str
) -> Optional[RiskScore]:
    """Return cached risk score for a route if one exists from the last 24 hours, else None."""
    cutoff = datetime.utcnow() - timedelta(hours=24)
    stmt = (
        select(RiskScore)
        .where(RiskScore.route == route)
        .where(RiskScore.created_at >= cutoff)
        .order_by(RiskScore.created_at.desc())
        .limit(1)
    )
    return session.scalars(stmt).first()


def get_recent_flights(
    session: Session, route: str, days: int = 1
) -> list[Flight]:
    """Return flight snapshots for a route fetched within the last `days` days."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    stmt = (
        select(Flight)
        .where(Flight.route == route)
        .where(Flight.created_at >= cutoff)
        .order_by(Flight.created_at.desc())
    )
    return list(session.scalars(stmt))
