"""Metal commodities price fetcher via yfinance (LME/COMEX futures)."""

import logging
from datetime import date, timedelta

from ecoforo.fetchers.base import BaseFetcher

logger = logging.getLogger(__name__)

METALS = [
    {"ticker": "HG=F", "name": "Copper Futures", "unit": "USD/lb", "exchange": "COMEX"},
    {"ticker": "ALI=F", "name": "Aluminum Futures", "unit": "USD/ton", "exchange": "LME"},
    {"ticker": "NI=F", "name": "Nickel Futures", "unit": "USD/ton", "exchange": "LME"},
    {"ticker": "ZNC=F", "name": "Zinc Futures", "unit": "USD/ton", "exchange": "LME"},
    {"ticker": "LEAD=F", "name": "Lead Futures", "unit": "USD/ton", "exchange": "LME"},
    {"ticker": "TIN=F", "name": "Tin Futures", "unit": "USD/ton", "exchange": "LME"},
    {"ticker": "SI=F", "name": "Silver Futures", "unit": "USD/oz", "exchange": "COMEX"},
    {"ticker": "GC=F", "name": "Gold Futures", "unit": "USD/oz", "exchange": "COMEX"},
    {"ticker": "PA=F", "name": "Palladium Futures", "unit": "USD/oz", "exchange": "NYMEX"},
    {"ticker": "PL=F", "name": "Platinum Futures", "unit": "USD/oz", "exchange": "NYMEX"},
    {"ticker": "CL=F", "name": "Crude Oil WTI Futures", "unit": "USD/bbl", "exchange": "NYMEX"},
    {"ticker": "RB=F", "name": "RBOB Gasoline Futures", "unit": "USD/gal", "exchange": "NYMEX"},
    {"ticker": "NG=F", "name": "Natural Gas Futures", "unit": "USD/MMBtu", "exchange": "NYMEX"},
    {"ticker": "ZC=F", "name": "Corn Futures", "unit": "USC/bu", "exchange": "CBOT"},
    {"ticker": "BZ=F", "name": "Brent Crude Futures", "unit": "USD/bbl", "exchange": "ICE"},
]


class MetalsFetcher(BaseFetcher):
    source_name = "metals"
    source_type = "indicator"
    METALS = METALS

    def __init__(self):
        self._ticker = None

    def _get_ticker(self):
        if self._ticker is None:
            import yfinance as yf
            self._ticker = yf
        return self._ticker

    def fetch(self, start: date, end: date) -> list[dict]:
        records = []
        yf = self._get_ticker()
        tickers = [m["ticker"] for m in self.METALS]

        try:
            data = yf.download(
                tickers=" ".join(tickers),
                start=str(start),
                end=str(end + timedelta(days=1)),
                group_by="ticker",
                auto_adjust=False,
                progress=False,
            )
        except Exception as e:
            logger.error(f"yfinance download failed: {e}")
            return records

        for metal in self.METALS:
            t = metal["ticker"]
            if t not in data or data[t].empty:
                continue
            df = data[t]
            for idx, row in df.iterrows():
                if row.isna().all():
                    continue
                records.append({
                    "ticker": t,
                    "name": metal["name"],
                    "unit": metal["unit"],
                    "exchange": metal["exchange"],
                    "date": idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10],
                    "close": float(row["Close"]) if "Close" in row and not row.isna()["Close"] else None,
                    "open": float(row["Open"]) if "Open" in row and not row.isna()["Open"] else None,
                    "high": float(row["High"]) if "High" in row and not row.isna()["High"] else None,
                    "low": float(row["Low"]) if "Low" in row and not row.isna()["Low"] else None,
                    "volume": int(row["Volume"]) if "Volume" in row and not row.isna()["Volume"] else None,
                })
        return records

    def normalize(self, raw: dict) -> dict:
        return {
            "source_name": self.source_name,
            "title": f"{raw['name']} — Close",
            "description": f"{raw['name']} ({raw['exchange']}) close price: {raw['close']} {raw['unit']}",
            "event_date": raw["date"],
            "actual_value": raw["close"],
            "forecast_value": None,
            "previous_value": None,
            "country": "GLOBAL",
            "impact": "medium",
            "importance": 4,
            "url": f"https://finance.yahoo.com/quote/{raw['ticker']}",
            "raw_data": {
                "ticker": raw["ticker"],
                "open": raw.get("open"),
                "high": raw.get("high"),
                "low": raw.get("low"),
                "volume": raw.get("volume"),
                "exchange": raw.get("exchange"),
                "unit": raw.get("unit"),
            },
        }
