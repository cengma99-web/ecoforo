"""Investing.com economic calendar scraper."""

import logging
import re
from datetime import date, datetime, timedelta
from typing import Optional

import requests
from bs4 import BeautifulSoup

from ecoforo.fetchers.base import BaseFetcher

logger = logging.getLogger(__name__)


class InvestingFetcher(BaseFetcher):
    source_name = "investing"
    source_type = "calendar"
    BASE_URL = "https://www.investing.com/economic-calendar/"

    def _get_session(self):
        s = requests.Session()
        s.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml",
        })
        from ecoforo.config import config
        if config.PROXY_URL:
            s.proxies = {"http": config.PROXY_URL, "https": config.PROXY_URL}
        return s

    @staticmethod
    def _importance_to_impact(stars: int) -> str:
        if stars >= 4:
            return "high"
        elif stars >= 2:
            return "medium"
        return "low"

    @staticmethod
    def _parse_numeric(val: str) -> Optional[float]:
        """Parse values like '5.25%', '3.2M', '-2.1B', 'N/A'."""
        if not val or val.strip() in ("", "N/A", "-", "—"):
            return None
        val = val.strip().replace(",", "")
        multiplier = 1
        if val.endswith("%"):
            return float(val[:-1])
        for suffix, mul in [("K", 1e3), ("M", 1e6), ("B", 1e9), ("T", 1e12)]:
            if val.upper().endswith(suffix):
                val = val[:-1]
                multiplier = mul
                break
        try:
            return float(val) * multiplier
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_country(raw_country: str) -> str:
        """Map Investing.com country names to ISO codes."""
        MAP = {
            "United States": "US", "China": "CN", "Japan": "JP", "Germany": "DE",
            "United Kingdom": "GB", "France": "FR", "Italy": "IT", "Canada": "CA",
            "Australia": "AU", "South Korea": "KR", "India": "IN", "Brazil": "BR",
            "Russia": "RU", "Switzerland": "CH", "Eurozone": "EU", "Euro Zone": "EU",
            "European Union": "EU", "Spain": "ES", "Netherlands": "NL", "Mexico": "MX",
            "Indonesia": "ID", "Turkey": "TR", "Saudi Arabia": "SA", "South Africa": "ZA",
        }
        return MAP.get(raw_country, raw_country[:2].upper())

    def fetch(self, start: date, end: date) -> list[dict]:
        """Scrape Investing.com economic calendar for a date range."""
        records = []
        session = self._get_session()
        current = start

        while current <= end:
            url = f"{self.BASE_URL}?date={current.strftime('%Y-%m-%d')}"
            try:
                resp = session.get(url, timeout=30)
                if resp.status_code != 200:
                    logger.warning(f"Investing.com returned {resp.status_code} for {current}")
                    current += timedelta(days=7)
                    continue

                soup = BeautifulSoup(resp.text, "lxml")
                rows = soup.select("tr.js-event-item")
                if not rows:
                    rows = soup.select("table.genTbl tbody tr")

                for row in rows:
                    try:
                        record = self._parse_row(row)
                        if record and start <= record["date_obj"] <= end:
                            records.append(record)
                    except Exception as e:
                        logger.debug(f"Failed to parse Investing.com row: {e}")
                        continue

                logger.debug(f"Investing.com: fetched {len(records)} events so far (week of {current})")
            except Exception as e:
                logger.warning(f"Investing.com fetch error for {current}: {e}")

            current += timedelta(days=7)

        return records

    def _parse_row(self, row) -> Optional[dict]:
        """Parse a single table row from Investing.com economic calendar."""
        cells = row.find_all("td")
        if len(cells) < 5:
            return None

        time_str = cells[0].get_text(strip=True)
        if not time_str or time_str == "No events found":
            return None

        # Currency / country column
        currency_cell = cells[1]
        country_flag = currency_cell.find("span", class_="ceFlags")
        country_name = country_flag.get("title", "") if country_flag else ""
        currency = currency_cell.get_text(strip=True)

        # Importance (star count)
        importance = 1
        impact_cell = cells[2] if len(cells) > 2 else None
        if impact_cell:
            stars = impact_cell.find_all("i", class_=lambda c: c and "grayFull" in c)
            importance = len(stars) if stars else 1

        # Event title
        event_cell = cells[3] if len(cells) > 3 else None
        title = ""
        url = ""
        if event_cell:
            link = event_cell.find("a")
            if link:
                title = link.get_text(strip=True)
                url = link.get("href", "")
                if url and not url.startswith("http"):
                    url = "https://www.investing.com" + url
            if not title:
                title = event_cell.get_text(strip=True)

        # Actual / Forecast / Previous values
        actual_val = None
        forecast_val = None
        previous_val = None
        if len(cells) > 4:
            actual_val = self._parse_numeric(cells[4].get_text(strip=True))
        if len(cells) > 5:
            forecast_val = self._parse_numeric(cells[5].get_text(strip=True))
        if len(cells) > 6:
            previous_val = self._parse_numeric(cells[6].get_text(strip=True))

        # Parse date from time string
        date_obj = None
        try:
            date_obj = datetime.fromisoformat(time_str)
        except (ValueError, TypeError):
            date_obj = datetime.now().date()

        return {
            "title": title,
            "date": date_obj.strftime("%Y-%m-%d") if hasattr(date_obj, "strftime") else str(date_obj)[:10],
            "time": time_str,
            "date_obj": date_obj if hasattr(date_obj, "date") else datetime.strptime(str(date_obj)[:10], "%Y-%m-%d").date(),
            "country": country_name or currency[:2],
            "currency": currency,
            "actual": cells[4].get_text(strip=True) if len(cells) > 4 else "",
            "forecast": cells[5].get_text(strip=True) if len(cells) > 5 else "",
            "previous": cells[6].get_text(strip=True) if len(cells) > 6 else "",
            "importance": importance,
            "url": url,
        }

    def normalize(self, raw: dict) -> dict:
        country_iso = self._parse_country(raw.get("country", ""))
        importance = raw.get("importance", 3)
        return {
            "source_name": self.source_name,
            "title": raw["title"],
            "description": f"{raw['title']} — actual: {raw.get('actual', 'N/A')}, forecast: {raw.get('forecast', 'N/A')}, previous: {raw.get('previous', 'N/A')}",
            "event_date": raw["date"],
            "actual_value": self._parse_numeric(raw.get("actual", "")),
            "forecast_value": self._parse_numeric(raw.get("forecast", "")),
            "previous_value": self._parse_numeric(raw.get("previous", "")),
            "country": country_iso,
            "impact": self._importance_to_impact(importance),
            "importance": importance,
            "url": raw.get("url", ""),
            "raw_data": {
                "time": raw.get("time"),
                "currency": raw.get("currency"),
                "raw_actual": raw.get("actual"),
                "raw_forecast": raw.get("forecast"),
                "raw_previous": raw.get("previous"),
            },
        }
