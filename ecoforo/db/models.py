"""SQLAlchemy ORM models for ecoforo."""

import datetime
from sqlalchemy import (
    Column, Integer, String, Text, Float, SMALLINT, DateTime,
    ForeignKey, UniqueConstraint, Index, JSON, Enum as SAEnum,
)
from sqlalchemy.orm import DeclarativeBase, relationship
import enum


class Base(DeclarativeBase):
    pass


class ImpactLevel(str, enum.Enum):
    low = "low"
    medium = "medium"
    high = "high"


class SourceType(str, enum.Enum):
    calendar = "calendar"
    indicator = "indicator"
    news = "news"


class Frequency(str, enum.Enum):
    daily = "daily"
    weekly = "weekly"
    monthly = "monthly"
    quarterly = "quarterly"
    yearly = "yearly"


class EventSource(Base):
    __tablename__ = "event_sources"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False)
    source_type = Column(SAEnum(SourceType), nullable=False)
    base_url = Column(String(500))
    config = Column(JSON)
    is_active = Column(SMALLINT, default=1)
    last_fetch_at = Column(DateTime(timezone=True))
    last_backfill_at = Column(DateTime(timezone=True))
    fetch_interval_seconds = Column(Integer, default=86400)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc))

    events = relationship("EconomicEvent", back_populates="source")
    indicators = relationship("EconomicIndicator", back_populates="source")


class EventCategory(Base):
    __tablename__ = "event_categories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False)
    parent_id = Column(Integer, ForeignKey("event_categories.id"))
    keywords = Column(JSON)  # list[str] for auto-classification
    importance_default = Column(SMALLINT, default=3)

    parent = relationship("EventCategory", remote_side=[id])
    events = relationship("EconomicEvent", back_populates="category")


class EconomicEvent(Base):
    __tablename__ = "economic_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(Integer, ForeignKey("event_sources.id"), nullable=False)
    category_id = Column(Integer, ForeignKey("event_categories.id"))
    title = Column(String(500), nullable=False)
    description = Column(Text)
    event_date = Column(DateTime(timezone=True), nullable=False, index=True)
    actual_value = Column(Float)
    forecast_value = Column(Float)
    previous_value = Column(Float)
    impact = Column(SAEnum(ImpactLevel))
    country = Column(String(2))  # ISO 3166-1 alpha-2
    importance = Column(SMALLINT, default=3)
    sentiment = Column(Float)  # -1 to 1, populated in Phase 2
    url = Column(String(2000))
    raw_data = Column(JSON)
    dedup_key = Column(String(64), unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.datetime.now(datetime.timezone.utc))

    source = relationship("EventSource", back_populates="events")
    category = relationship("EventCategory", back_populates="events")
    indicator_links = relationship("EventIndicatorLink", back_populates="event")

    __table_args__ = (
        Index("idx_events_date_country_impact", "event_date", "country", "impact"),
        Index("idx_events_dedup", "dedup_key"),
    )


class EconomicIndicator(Base):
    __tablename__ = "economic_indicators"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(50), unique=True, nullable=False)
    name = Column(String(300), nullable=False)
    unit = Column(String(50))
    frequency = Column(SAEnum(Frequency))
    source_id = Column(Integer, ForeignKey("event_sources.id"))
    country = Column(String(2))
    category = Column(String(100))
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc))

    source = relationship("EventSource", back_populates="indicators")
    values = relationship("IndicatorValue", back_populates="indicator")
    event_links = relationship("EventIndicatorLink", back_populates="indicator")


class IndicatorValue(Base):
    __tablename__ = "indicator_values"

    id = Column(Integer, primary_key=True, autoincrement=True)
    time = Column(DateTime(timezone=True), nullable=False, index=True)
    indicator_id = Column(Integer, ForeignKey("economic_indicators.id"), nullable=False)
    value = Column(Float, nullable=False)
    metadata_ = Column("metadata", JSON)

    indicator = relationship("EconomicIndicator", back_populates="values")

    __table_args__ = (
        UniqueConstraint("indicator_id", "time", name="uq_indicator_time"),
    )


class EventIndicatorLink(Base):
    __tablename__ = "event_indicator_links"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, ForeignKey("economic_events.id"), nullable=False)
    indicator_id = Column(Integer, ForeignKey("economic_indicators.id"), nullable=False)
    relationship_type = Column(String(50))
    correlation_score = Column(Float)

    event = relationship("EconomicEvent", back_populates="indicator_links")
    indicator = relationship("EconomicIndicator", back_populates="event_links")

    __table_args__ = (
        UniqueConstraint("event_id", "indicator_id", name="uq_event_indicator"),
    )


class DeadLetter(Base):
    """Records that failed validation or processing."""

    __tablename__ = "dead_letters"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_name = Column(String(100), nullable=False)
    raw_data = Column(JSON)
    error_message = Column(Text)
    traceback = Column(Text)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc))
