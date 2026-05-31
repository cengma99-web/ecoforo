"""FRED (Federal Reserve Economic Data) fetcher."""

import logging
from datetime import date
from typing import Optional

from ecoforo.fetchers.base import BaseFetcher

logger = logging.getLogger(__name__)

# Key indicators to track
SERIES = [
    # Inflation
    {"code": "CPIAUCSL", "name": "US CPI (All Urban Consumers, All Items)", "unit": "index", "frequency": "monthly"},
    {"code": "CPILFESL", "name": "US Core CPI (All Items Less Food & Energy)", "unit": "index", "frequency": "monthly"},
    {"code": "PCEPI", "name": "US PCE Price Index", "unit": "index", "frequency": "monthly"},
    # Employment
    {"code": "UNRATE", "name": "US Unemployment Rate", "unit": "percent", "frequency": "monthly"},
    {"code": "PAYEMS", "name": "US Total Nonfarm Payrolls", "unit": "thousands", "frequency": "monthly"},
    # GDP & Growth
    {"code": "GDP", "name": "US Gross Domestic Product", "unit": "billions_usd", "frequency": "quarterly"},
    {"code": "GDPC1", "name": "US Real GDP", "unit": "billions_chained_2017", "frequency": "quarterly"},
    # Interest Rates
    {"code": "DFF", "name": "US Federal Funds Effective Rate", "unit": "percent", "frequency": "daily"},
    {"code": "DGS10", "name": "US 10-Year Treasury Yield", "unit": "percent", "frequency": "daily"},
    {"code": "T10Y2Y", "name": "US 10Y-2Y Treasury Spread", "unit": "percent", "frequency": "daily"},
    # Housing
    {"code": "HOUST", "name": "US Housing Starts", "unit": "thousands", "frequency": "monthly"},
    {"code": "MSPUS", "name": "US Median Home Sales Price", "unit": "usd", "frequency": "quarterly"},
    # Industrial
    {"code": "INDPRO", "name": "US Industrial Production Index", "unit": "index_2017", "frequency": "monthly"},
    {"code": "TCU", "name": "US Capacity Utilization", "unit": "percent", "frequency": "monthly"},
    # Trade
    {"code": "BOPGSTB", "name": "US Trade Balance (Goods & Services)", "unit": "millions_usd", "frequency": "monthly"},
]


class FREDFetcher(BaseFetcher):
    source_name = "fred"
    source_type = "indicator"
    SERIES = SERIES

    def __init__(self, api_key: Optional[str] = None):
        from ecoforo.config import config
        self.api_key = api_key or config.FRED_API_KEY
        self._client = None

    def _get_client(self):
        if self._client is None:
            from fredapi import Fred
            kwargs = {"api_key": self.api_key}
            self._client = Fred(**kwargs)
        return self._client

    def fetch(self, start: date, end: date) -> list[dict]:
        records = []
        if not self.api_key:
            logger.warning("FRED_API_KEY not set — skipping FRED fetch")
            return records

        fred = self._get_client()
        for series in self.SERIES:
            try:
                s = fred.get_series(series["code"], observation_start=start, observation_end=end)
                for dt, val in s.dropna().items():
                    records.append({
                        "code": series["code"],
                        "name": series["name"],
                        "unit": series["unit"],
                        "frequency": series["frequency"],
                        "date": dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)[:10],
                        "value": float(val),
                    })
            except Exception as e:
                logger.warning(f"FRED series {series['code']} fetch failed: {e}")
                continue
        return records

    def normalize(self, raw: dict) -> dict:
        return {
            "source_name": self.source_name,
            "title": raw["name"],
            "description": f"FRED series {raw['code']}: {raw['name']} ({raw.get('frequency', '')})",
            "event_date": raw["date"],
            "actual_value": raw["value"],
            "forecast_value": None,
            "previous_value": None,
            "country": "US",
            "impact": "medium",
            "importance": 4,
            "url": f"https://fred.stlouisfed.org/series/{raw['code']}",
            "raw_data": {
                "indicator_code": raw["code"],
                "unit": raw.get("unit"),
                "frequency": raw.get("frequency"),
            },
        }
