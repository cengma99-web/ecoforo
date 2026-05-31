"""Unit tests for BaseFetcher ABC."""

import pytest
from datetime import date, datetime, timezone

from ecoforo.fetchers.base import BaseFetcher, FetcherError


# ---------------------------------------------------------------------------
# Minimal concrete fetcher for testing
# ---------------------------------------------------------------------------

class DummyFetcher(BaseFetcher):
    source_name = "dummy"
    source_type = "indicator"

    def fetch(self, start, end):
        return [
            {"title": "Test Event", "event_date": "2026-01-15", "country": "US"},
        ]

    def normalize(self, raw):
        return {
            "source_name": self.source_name,
            "title": raw["title"],
            "event_date": raw["event_date"],
            "country": raw["country"],
            "description": raw.get("description"),
            "actual_value": raw.get("actual_value"),
            "forecast_value": raw.get("forecast_value"),
            "previous_value": raw.get("previous_value"),
            "impact": raw.get("impact"),
            "importance": raw.get("importance", 3),
            "url": raw.get("url"),
            "raw_data": raw,
        }


# ---------------------------------------------------------------------------
# Tests: normalize
# ---------------------------------------------------------------------------

def test_fetcher_normalize_produces_dict():
    """normalize() must return a dict with expected keys."""
    f = DummyFetcher()
    result = f.normalize({"title": "X", "event_date": "2026-01-01", "country": "CN"})
    assert isinstance(result, dict)
    assert result["title"] == "X"


# ---------------------------------------------------------------------------
# Tests: _make_dedup_key
# ---------------------------------------------------------------------------

def test_fetcher_dedup_key_deterministic():
    """Same inputs must produce the same 64-char hex key."""
    f = DummyFetcher()
    record = {
        "title": "GDP Release",
        "source_name": "dummy",
        "event_date": "2026-01-01",
        "country": "US",
    }
    k1 = f._make_dedup_key(record)
    k2 = f._make_dedup_key(record)
    assert k1 == k2
    assert len(k1) == 64


def test_fetcher_dedup_key_different_for_different_inputs():
    """Different titles must yield different dedup keys."""
    f = DummyFetcher()
    a = {
        "title": "A",
        "source_name": "dummy",
        "event_date": "2026-01-01",
        "country": "US",
    }
    b = {
        "title": "B",
        "source_name": "dummy",
        "event_date": "2026-01-01",
        "country": "US",
    }
    assert f._make_dedup_key(a) != f._make_dedup_key(b)


# ---------------------------------------------------------------------------
# Tests: validate
# ---------------------------------------------------------------------------

def test_fetcher_validate_field_checks():
    """Default validate() checks title, event_date, country."""
    f = DummyFetcher()

    # Valid record
    assert f.validate({"title": "OK", "event_date": "2026-01-01", "country": "US"})

    # Empty title
    assert not f.validate({"title": "", "event_date": "2026-01-01", "country": "US"})

    # Invalid event_date
    assert not f.validate({"title": "OK", "event_date": "invalid-date", "country": "US"})


# ---------------------------------------------------------------------------
# Additional tests for thorough coverage
# ---------------------------------------------------------------------------

def test_fetcher_validate_missing_country():
    f = DummyFetcher()
    assert not f.validate({"title": "OK", "event_date": "2026-01-01", "country": ""})


def test_fetcher_validate_missing_event_date():
    f = DummyFetcher()
    assert not f.validate({"title": "OK", "event_date": "", "country": "US"})


def test_fetcher_validate_accepts_datetime_object():
    """validate() should accept a datetime object for event_date."""
    f = DummyFetcher()
    assert f.validate({
        "title": "OK",
        "event_date": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "country": "US",
    })


def test_fetcher_validate_accepts_date_object():
    """validate() should accept a date object for event_date."""
    f = DummyFetcher()
    assert f.validate({
        "title": "OK",
        "event_date": date(2026, 1, 1),
        "country": "US",
    })


def test_fetcher_error_is_exception():
    """FetcherError must be a subclass of Exception."""
    assert issubclass(FetcherError, Exception)


def test_fetcher_error_can_be_raised():
    """FetcherError can be raised and caught."""
    with pytest.raises(FetcherError):
        raise FetcherError("test error")


def test_base_fetcher_cannot_be_instantiated():
    """BaseFetcher is abstract and cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseFetcher()  # type: ignore[abstract]


def test_dummy_fetcher_can_be_instantiated():
    """A concrete subclass with fetch+normalize can be instantiated."""
    f = DummyFetcher()
    assert f.source_name == "dummy"
    assert f.source_type == "indicator"


def test_dedup_key_includes_country():
    """Different countries should produce different dedup keys."""
    f = DummyFetcher()
    base = {"title": "GDP", "source_name": "dummy", "event_date": "2026-01-01"}
    a = {**base, "country": "US"}
    b = {**base, "country": "CN"}
    assert f._make_dedup_key(a) != f._make_dedup_key(b)


def test_dedup_key_includes_event_date():
    """Different dates should produce different dedup keys."""
    f = DummyFetcher()
    base = {"title": "GDP", "source_name": "dummy", "country": "US"}
    a = {**base, "event_date": "2026-01-01"}
    b = {**base, "event_date": "2026-01-02"}
    assert f._make_dedup_key(a) != f._make_dedup_key(b)


def test_dedup_key_is_hex():
    """Dedup key must be all lowercase hex characters."""
    f = DummyFetcher()
    record = {"title": "X", "source_name": "dummy", "event_date": "2026-01-01", "country": "US"}
    key = f._make_dedup_key(record)
    assert all(c in "0123456789abcdef" for c in key)


def test_parse_event_date_iso_string():
    """_parse_event_date handles ISO-8601 strings."""
    result = BaseFetcher._parse_event_date("2026-06-15T12:30:00+00:00")
    assert result == datetime(2026, 6, 15, 12, 30, tzinfo=timezone.utc)


def test_parse_event_date_naive_string():
    """_parse_event_date assumes naive strings are UTC."""
    result = BaseFetcher._parse_event_date("2026-06-15")
    assert result == datetime(2026, 6, 15, tzinfo=timezone.utc)


def test_parse_event_date_datetime_object():
    """_parse_event_date passes through aware datetimes unchanged."""
    dt = datetime(2026, 6, 15, tzinfo=timezone.utc)
    result = BaseFetcher._parse_event_date(dt)
    assert result == dt
    assert result.tzinfo is not None


def test_default_retry_config():
    """Default max_retries and retry_delays are set."""
    f = DummyFetcher()
    assert f.max_retries == 3
    assert f.retry_delays == (1, 5, 25)
