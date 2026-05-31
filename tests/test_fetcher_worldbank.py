import pytest
from datetime import date
from ecoforo.fetchers.worldbank import WBFetcher


def test_wb_fetcher_exists():
    f = WBFetcher()
    assert f.source_name == "worldbank"
    assert f.source_type == "indicator"


def test_wb_normalize_indicator():
    f = WBFetcher()
    raw = {
        "code": "NY.GDP.MKTP.CD",
        "name": "GDP (current US$)",
        "date": "2025-01-01",
        "value": 2.5e13,
        "country": "CN",
    }
    result = f.normalize(raw)
    assert result["title"] == "GDP (current US$) — CN"
    assert result["actual_value"] == 2.5e13
    assert result["country"] == "CN"


def test_wb_indicators_list():
    f = WBFetcher()
    codes = [ind["code"] for ind in f.INDICATORS]
    assert "NY.GDP.MKTP.CD" in codes
    assert "FP.CPI.TOTL.ZG" in codes
