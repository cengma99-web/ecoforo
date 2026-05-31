import pytest
from datetime import date
from ecoforo.fetchers.fred import FREDFetcher


def test_fred_fetcher_exists():
    f = FREDFetcher()
    assert f.source_name == "fred"
    assert f.source_type == "indicator"


def test_fred_normalize_indicator():
    f = FREDFetcher()
    raw = {
        "code": "CPIAUCSL",
        "name": "Consumer Price Index for All Urban Consumers: All Items",
        "date": "2026-01-01",
        "value": 315.2,
    }
    result = f.normalize(raw)
    assert result["title"] == "Consumer Price Index for All Urban Consumers: All Items"
    assert result["event_date"] == "2026-01-01"
    assert result["country"] == "US"
    assert result["actual_value"] == 315.2
    assert result["raw_data"]["indicator_code"] == "CPIAUCSL"


def test_fred_series_list_is_nonempty():
    f = FREDFetcher()
    assert len(f.SERIES) > 0
    assert all("code" in s and "name" in s for s in f.SERIES)


def test_fred_skips_without_api_key():
    f = FREDFetcher(api_key="")
    result = f.fetch(date(2026, 1, 1), date(2026, 1, 31))
    assert result == []
