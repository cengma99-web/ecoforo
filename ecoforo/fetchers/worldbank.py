"""World Bank global development indicators fetcher."""

import logging
from datetime import date

from ecoforo.fetchers.base import BaseFetcher

logger = logging.getLogger(__name__)

INDICATORS = [
    {"code": "NY.GDP.MKTP.CD", "name": "GDP (current US$)", "unit": "usd", "frequency": "yearly"},
    {"code": "NY.GDP.MKTP.KD.ZG", "name": "GDP Growth (annual %)", "unit": "percent", "frequency": "yearly"},
    {"code": "FP.CPI.TOTL.ZG", "name": "Inflation, Consumer Prices (annual %)", "unit": "percent", "frequency": "yearly"},
    {"code": "SL.UEM.TOTL.ZS", "name": "Unemployment Rate (% of total labor force)", "unit": "percent", "frequency": "yearly"},
    {"code": "NE.EXP.GNFS.ZS", "name": "Exports (% of GDP)", "unit": "percent", "frequency": "yearly"},
    {"code": "BX.KLT.DINV.WD.GD.ZS", "name": "FDI Net Inflow (% of GDP)", "unit": "percent", "frequency": "yearly"},
    {"code": "GC.DOD.TOTL.GD.ZS", "name": "Government Debt (% of GDP)", "unit": "percent", "frequency": "yearly"},
    {"code": "SP.POP.TOTL", "name": "Population, Total", "unit": "count", "frequency": "yearly"},
    {"code": "AG.LND.TOTL.K2", "name": "Land Area (sq km)", "unit": "sq_km", "frequency": "yearly"},
]

COUNTRIES = ["CN", "US", "JP", "DE", "GB", "IN", "KR", "RU", "BR", "ZA", "1W"]


class WBFetcher(BaseFetcher):
    source_name = "worldbank"
    source_type = "indicator"
    INDICATORS = INDICATORS
    COUNTRIES = COUNTRIES

    def _get_session(self):
        import requests
        s = requests.Session()
        from ecoforo.config import config
        if config.PROXY_URL:
            s.proxies = {"http": config.PROXY_URL, "https": config.PROXY_URL}
        return s

    def fetch(self, start: date, end: date) -> list[dict]:
        records = []
        session = self._get_session()
        api_base = "https://api.worldbank.org/v2"

        for indicator in self.INDICATORS:
            code = indicator["code"]
            for country in self.COUNTRIES:
                url = f"{api_base}/country/{country}/indicator/{code}"
                params = {
                    "format": "json",
                    "date": f"{start.year}:{end.year}",
                    "per_page": 2000,
                }
                try:
                    resp = session.get(url, params=params, timeout=60)
                    if resp.status_code != 200:
                        continue
                    data = resp.json()
                    if not data or len(data) < 2 or data[1] is None:
                        continue
                    for entry in data[1]:
                        val = entry.get("value")
                        if val is None:
                            continue
                        records.append({
                            "code": code,
                            "name": indicator["name"],
                            "unit": indicator["unit"],
                            "frequency": indicator["frequency"],
                            "date": entry.get("date", ""),
                            "value": float(val),
                            "country": country,
                        })
                except Exception as e:
                    logger.debug(f"WB {code}/{country}: {e}")
                    continue

        return records

    def normalize(self, raw: dict) -> dict:
        # World Bank API returns bare years like "2024" — convert to full date
        date_str = raw.get("date", "")
        if len(date_str) == 4:  # "2024"
            date_str = f"{date_str}-01-01"
        return {
            "source_name": self.source_name,
            "title": f"{raw['name']} — {raw.get('country', '')}",
            "description": f"World Bank indicator {raw['code']}: {raw['name']} ({raw.get('frequency', '')})",
            "event_date": date_str,
            "actual_value": raw["value"],
            "forecast_value": None,
            "previous_value": None,
            "country": raw.get("country", ""),
            "impact": "medium",
            "importance": 3,
            "url": f"https://data.worldbank.org/indicator/{raw['code']}",
            "raw_data": {
                "indicator_code": raw["code"],
                "unit": raw.get("unit"),
                "frequency": raw.get("frequency"),
                "country": raw.get("country"),
            },
        }
