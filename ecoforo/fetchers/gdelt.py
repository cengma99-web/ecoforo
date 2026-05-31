"""GDELT global news event fetcher — economic event tracking."""

import json
import logging
import subprocess
import time
from datetime import date, datetime, timedelta
from typing import Optional

from ecoforo.fetchers.base import BaseFetcher

logger = logging.getLogger(__name__)

# Broad economic keywords for GDELT article queries (kept minimal for rate limits)
ECONOMIC_QUERIES = [
    "inflation central bank",
    "GDP economic growth",
    "trade tariff commodity",
]

# Country name to ISO code mapping (common in GDELT)
COUNTRY_MAP = {
    "united states": "US", "china": "CN", "japan": "JP", "germany": "DE",
    "united kingdom": "GB", "france": "FR", "italy": "IT", "canada": "CA",
    "australia": "AU", "south korea": "KR", "india": "IN", "brazil": "BR",
    "russia": "RU", "european union": "EU", "eurozone": "EU",
    "spain": "ES", "netherlands": "NL", "mexico": "MX", "indonesia": "ID",
    "turkey": "TR", "saudi arabia": "SA", "south africa": "ZA",
    "switzerland": "CH", "singapore": "SG", "hong kong": "HK",
    "taiwan": "TW", "vietnam": "VN", "malaysia": "MY", "thailand": "TH",
}


class GDELTFetcher(BaseFetcher):
    source_name = "gdelt"
    source_type = "news"
    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def _http_get(self, url: str, params: dict) -> Optional[dict]:
        """Fetch JSON via curl subprocess (bypasses Python TLS/DPI issues)."""
        import time
        from urllib.parse import urlencode
        from ecoforo.config import config

        full_url = f"{url}?{urlencode(params)}"
        for attempt in range(3):
            cmd = ["curl", "-x", config.PROXY_URL, "-s", "--max-time", "30",
                   "-H", "Accept: application/json",
                   "-H", "User-Agent: ecoforo/0.1",
                   full_url]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=35)
                if result.returncode != 0 or not result.stdout.strip():
                    time.sleep(2)
                    continue
                data = json.loads(result.stdout)
                if "error" not in data:
                    return data
                time.sleep(2)
            except Exception as e:
                logger.debug(f"GDELT curl attempt {attempt+1} failed: {e}")
                time.sleep(2)
        return None

    @staticmethod
    def _guess_country(title: str, url: str = "") -> Optional[str]:
        """Try to guess country from article title/url by keyword matching."""
        text = (title + " " + url).lower()
        for name, code in COUNTRY_MAP.items():
            if name in text:
                return code
        return None

    def validate(self, record: dict) -> bool:
        """Override: allow empty country for news articles."""
        if not record.get("title") or not str(record.get("title")).strip():
            return False
        if not record.get("event_date"):
            return False
        return True

    @staticmethod
    def _guess_impact(title: str) -> str:
        """Heuristic impact level from keywords."""
        high_keywords = ["rate decision", "rate cut", "rate hike", "recession",
                         "crash", "crisis", "default", "sanction", "war"]
        medium_keywords = ["inflation", "gdp", "tariff", "trade war", "bailout",
                           "stimulus", "policy change"]
        title_lower = title.lower()
        if any(k in title_lower for k in high_keywords):
            return "high"
        if any(k in title_lower for k in medium_keywords):
            return "medium"
        return "low"

    def fetch(self, start: date, end: date) -> list[dict]:
        """Query GDELT for economic news articles across date range.

        GDELT timespan is relative: '15min', '1h', '1d', '5d', '30d', etc.
        We batch by 5-day windows to handle large ranges.
        """
        records = []
        window_size = timedelta(days=30)
        current = start

        # GDELT free API only accepts specific timespan values
        VALID_TIMESPANS = ["1d", "3d", "7d", "30d", "3m", "6m"]

        while current <= end:
            window_end = min(current + window_size, end)
            days_in_window = (window_end - current).days + 1
            # Map to closest valid GDELT timespan
            if days_in_window <= 1:
                timespan = "1d"
            elif days_in_window <= 3:
                timespan = "3d"
            elif days_in_window <= 7:
                timespan = "7d"
            elif days_in_window <= 30:
                timespan = "30d"
            elif days_in_window <= 90:
                timespan = "3m"
            else:
                timespan = "6m"

            for query in ECONOMIC_QUERIES:
                params = {
                    "query": query,
                    "mode": "artlist",
                    "timespan": timespan,
                    "maxrecords": 250,
                    "format": "json",
                }
                try:
                    data = self._http_get(self.BASE_URL, params)
                    if not data:
                        continue
                    articles = data.get("articles", [])
                    for art in articles:
                        seen_date = art.get("seendate", "")
                        if not seen_date:
                            continue
                        try:
                            art_date = datetime.strptime(seen_date[:8], "%Y%m%d").date()
                        except (ValueError, TypeError):
                            continue
                        if start <= art_date <= end:
                            records.append({
                                "title": art.get("title", ""),
                                "date": art_date.strftime("%Y-%m-%d"),
                                "url": art.get("url", ""),
                                "source_country": art.get("sourcecountry", ""),
                                "language": art.get("language", ""),
                                "tone": float(art.get("tone", 0)),
                                "domain": art.get("domain", ""),
                            })
                except Exception as e:
                    logger.debug(f"GDELT query failed: {query} @ {current}: {e}")
                    continue
                time.sleep(6)  # GDELT free API: 1 request per 5 seconds

            current = window_end + timedelta(days=1)

        # Deduplicate by title
        seen = set()
        unique = []
        for r in records:
            key = r["title"][:100]
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique

    def normalize(self, raw: dict) -> dict:
        title = raw["title"]
        country = self._guess_country(title, raw.get("url", ""))
        impact = self._guess_impact(title)
        tone = raw.get("tone", 0)

        return {
            "source_name": self.source_name,
            "title": title,
            "description": f"GDELT news: {title} (tone: {tone:.1f}, source: {raw.get('source_country', 'N/A')})",
            "event_date": raw["date"],
            "actual_value": None,
            "forecast_value": None,
            "previous_value": None,
            "country": country or "",
            "impact": impact,
            "importance": 2 if impact == "low" else (3 if impact == "medium" else 4),
            "url": raw.get("url", ""),
            "raw_data": {
                "tone": tone,
                "source_country": raw.get("source_country"),
                "domain": raw.get("domain"),
                "language": raw.get("language"),
            },
        }
