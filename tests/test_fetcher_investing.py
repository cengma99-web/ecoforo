import pytest
from datetime import date
from ecoforo.fetchers.investing import InvestingFetcher


def test_investing_fetcher_exists():
    f = InvestingFetcher()
    assert f.source_name == "investing"
    assert f.source_type == "calendar"


def test_investing_normalize():
    f = InvestingFetcher()
    raw = {
        "title": "Fed Interest Rate Decision",
        "date": "2026-06-15",
        "time": "14:00",
        "country": "US",
        "actual": "5.25%",
        "forecast": "5.25%",
        "previous": "5.25%",
        "importance": 5,
        "currency": "USD",
    }
    result = f.normalize(raw)
    assert result["title"] == "Fed Interest Rate Decision"
    assert result["country"] == "US"
    assert result["importance"] == 5
    assert result["impact"] == "high"


def test_investing_importance_to_impact():
    f = InvestingFetcher()
    assert f._importance_to_impact(5) == "high"
    assert f._importance_to_impact(3) == "medium"
    assert f._importance_to_impact(1) == "low"


def test_parse_value_percent():
    f = InvestingFetcher()
    assert f._parse_numeric("5.25%") == 5.25
    assert f._parse_numeric("3.2M") == 3_200_000
    assert f._parse_numeric("-2.1B") == -2_100_000_000
    assert f._parse_numeric("N/A") is None
    assert f._parse_numeric("") is None


def test_parse_country():
    f = InvestingFetcher()
    assert f._parse_country("United States") == "US"
    assert f._parse_country("China") == "CN"
    assert f._parse_country("Eurozone") == "EU"
    assert f._parse_country("UnknownLand") == "UN"  # fallback: first 2 chars uppercase
