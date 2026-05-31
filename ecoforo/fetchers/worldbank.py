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

    def fetch(self, start: date, end: date) -> list[dict]:
        records = []
        try:
            import wbdata
        except ImportError:
            logger.warning("wbdata not installed — skipping World Bank fetch")
            return records

        codes = [ind["code"] for ind in self.INDICATORS]
        try:
            df = wbdata.get_dataframe(
                {"indicator": codes, "country": self.COUNTRIES},
                date=(str(start.year), str(end.year)),
            )
            for (country_code, indicator_code), row in df.iterrows():
                val = row.get("value")
                if val is None:
                    continue
                date_str = str(row.get("date", ""))[:10] or f"{start.year}-01-01"
                indicator_info = next((i for i in self.INDICATORS if i["code"] == indicator_code), None)
                records.append({
                    "code": indicator_code,
                    "name": indicator_info["name"] if indicator_info else indicator_code,
                    "unit": indicator_info["unit"] if indicator_info else "",
                    "frequency": indicator_info["frequency"] if indicator_info else "yearly",
                    "date": date_str,
                    "value": float(val),
                    "country": str(country_code),
                })
        except Exception as e:
            logger.error(f"World Bank fetch failed: {e}")
        return records

    def normalize(self, raw: dict) -> dict:
        return {
            "source_name": self.source_name,
            "title": f"{raw['name']} — {raw.get('country', '')}",
            "description": f"World Bank indicator {raw['code']}: {raw['name']} ({raw.get('frequency', '')})",
            "event_date": raw["date"],
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
