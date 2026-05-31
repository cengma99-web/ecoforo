import pytest
from datetime import date
from ecoforo.fetchers.metals import MetalsFetcher


def test_metals_fetcher_exists():
    f = MetalsFetcher()
    assert f.source_name == "metals"
    assert f.source_type == "indicator"


def test_metals_normalize():
    f = MetalsFetcher()
    raw = {
        "ticker": "HG=F",
        "name": "Copper Futures",
        "date": "2026-05-15",
        "close": 4.85,
        "open": 4.80,
        "high": 4.88,
        "low": 4.78,
        "volume": 125000,
        "exchange": "COMEX",
        "unit": "USD/lb",
    }
    result = f.normalize(raw)
    assert result["title"] == "Copper Futures — Close"
    assert result["actual_value"] == 4.85
    assert result["country"] == "GLOBAL"
    assert result["raw_data"]["ticker"] == "HG=F"


def test_metals_tickers_defined():
    f = MetalsFetcher()
    tickers = [t["ticker"] for t in f.METALS]
    assert "HG=F" in tickers
    assert "ALI=F" in tickers
