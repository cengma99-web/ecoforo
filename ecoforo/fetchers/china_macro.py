"""China macroeconomic data fetcher via AkShare."""

import logging
from datetime import date, datetime, timedelta
from typing import Optional

from ecoforo.fetchers.base import BaseFetcher

logger = logging.getLogger(__name__)


class ChinaMacroFetcher(BaseFetcher):
    """Fetches Chinese macroeconomic indicators via AkShare.

    Sources: NBS (National Bureau of Statistics), PBOC, Customs, Sina.
    """

    source_name = "china_macro"
    source_type = "indicator"

    @staticmethod
    def _parse_date(val) -> Optional[str]:
        """Parse various date formats to YYYY-MM-DD string."""
        if isinstance(val, (datetime, date)):
            return val.strftime("%Y-%m-%d") if hasattr(val, "strftime") else str(val)[:10]
        s = str(val).strip()
        # "2025年07月份" → "2025-07-01"
        if "年" in s and "月" in s:
            import re
            m = re.match(r"(\d{4})年(\d{1,2})月", s)
            if m:
                return f"{m.group(1)}-{int(m.group(2)):02d}-01"
        # "2025-07-09"
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            pass
        return None

    @staticmethod
    def _parse_chinese_number(val) -> Optional[float]:
        """Parse number, handling Chinese formatting and NaN."""
        import math
        try:
            f = float(val)
            return None if math.isnan(f) else f
        except (ValueError, TypeError):
            return None

    # ── fetch ────────────────────────────────────────────────

    def fetch(self, start: date, end: date) -> list[dict]:
        records = []

        # 1) CPI (yearly)
        self._fetch_cpi_yearly(records, start, end)
        # 2) PMI
        self._fetch_pmi(records, start, end)
        # 3) PPI
        self._fetch_ppi(records, start, end)
        # 4) Real estate
        self._fetch_real_estate(records, start, end)
        # 5) M2 money supply
        self._fetch_m2(records, start, end)
        # 6) Trade balance
        self._fetch_trade(records, start, end)
        # 7) GDP
        self._fetch_gdp(records, start, end)

        # Filter by date range (parse date string to date for comparison)
        filtered = []
        for r in records:
            d_str = r.get("date", "")
            try:
                rd = date.fromisoformat(d_str[:10])
                if start <= rd <= end:
                    filtered.append(r)
            except (ValueError, TypeError):
                continue
        return filtered

    # ── individual fetchers ──────────────────────────────────

    def _fetch_cpi_yearly(self, records, start, end):
        try:
            import akshare as ak
            df = ak.macro_china_cpi_yearly()
            for _, row in df.iterrows():
                d = self._parse_date(row["日期"])
                val = self._parse_chinese_number(row.get("今值"))
                if d:
                    records.append({
                        "title": "中国CPI年率",
                        "date": d,
                        "value": val,
                        "forecast": self._parse_chinese_number(row.get("预测值")),
                        "previous": self._parse_chinese_number(row.get("前值")),
                        "country": "CN",
                        "unit": "percent",
                        "importance": 5,
                    })
        except Exception as e:
            logger.warning(f"China CPI fetch failed: {e}")

    def _fetch_pmi(self, records, start, end):
        try:
            import akshare as ak
            df = ak.macro_china_pmi()
            for _, row in df.iterrows():
                d = self._parse_date(row["月份"])
                if d:
                    mfg = self._parse_chinese_number(row.get("制造业-指数"))
                    non_mfg = self._parse_chinese_number(row.get("非制造业-指数"))
                    if mfg is not None:
                        records.append({
                            "title": "中国制造业PMI", "date": d, "value": mfg,
                            "forecast": None, "previous": None,
                            "country": "CN", "unit": "index", "importance": 5,
                        })
                    if non_mfg is not None:
                        records.append({
                            "title": "中国非制造业PMI", "date": d, "value": non_mfg,
                            "forecast": None, "previous": None,
                            "country": "CN", "unit": "index", "importance": 4,
                        })
        except Exception as e:
            logger.warning(f"China PMI fetch failed: {e}")

    def _fetch_ppi(self, records, start, end):
        try:
            import akshare as ak
            df = ak.macro_china_ppi_yearly()
            for _, row in df.iterrows():
                d = self._parse_date(row["日期"])
                val = self._parse_chinese_number(row.get("今值"))
                if d and val is not None:
                    records.append({
                        "title": "中国PPI年率", "date": d, "value": val,
                        "forecast": None, "previous": self._parse_chinese_number(row.get("前值")),
                        "country": "CN", "unit": "percent", "importance": 4,
                    })
        except Exception as e:
            logger.warning(f"China PPI fetch failed: {e}")

    def _fetch_real_estate(self, records, start, end):
        try:
            import akshare as ak
            df = ak.macro_china_real_estate()
            for _, row in df.iterrows():
                d = self._parse_date(row["日期"])
                val = self._parse_chinese_number(row.get("最新值"))
                if d and val is not None:
                    records.append({
                        "title": "中国国房景气指数", "date": d, "value": val,
                        "forecast": None, "previous": None,
                        "country": "CN", "unit": "index", "importance": 3,
                    })
        except Exception as e:
            logger.warning(f"China real estate fetch failed: {e}")

    def _fetch_m2(self, records, start, end):
        try:
            import akshare as ak
            df = ak.macro_china_money_supply()
            for _, row in df.iterrows():
                d = self._parse_date(row["月份"])
                m2_val = self._parse_chinese_number(row.get("货币和准货币(M2)-数量(亿元)"))
                m2_yoy = self._parse_chinese_number(row.get("货币和准货币(M2)-同比增长"))
                if d:
                    if m2_val is not None:
                        records.append({
                            "title": "中国M2货币供应量", "date": d, "value": m2_val,
                            "forecast": None, "previous": None,
                            "country": "CN", "unit": "亿人民币", "importance": 4,
                        })
                    if m2_yoy is not None:
                        records.append({
                            "title": "中国M2同比增速", "date": d, "value": m2_yoy,
                            "forecast": None, "previous": None,
                            "country": "CN", "unit": "percent", "importance": 4,
                        })
        except Exception as e:
            logger.warning(f"China M2 fetch failed: {e}")

    def _fetch_trade(self, records, start, end):
        try:
            import akshare as ak
            df = ak.macro_china_trade_balance()
            for _, row in df.iterrows():
                d = self._parse_date(row["日期"])
                val = self._parse_chinese_number(row.get("今值"))
                if d and val is not None:
                    records.append({
                        "title": "中国贸易差额（美元）", "date": d, "value": val,
                        "forecast": self._parse_chinese_number(row.get("预测值")),
                        "previous": self._parse_chinese_number(row.get("前值")),
                        "country": "CN", "unit": "亿美元", "importance": 3,
                    })
        except Exception as e:
            logger.warning(f"China trade balance fetch failed: {e}")

    def _fetch_gdp(self, records, start, end):
        try:
            import akshare as ak
            df = ak.macro_china_gdp_yearly()
            for _, row in df.iterrows():
                d = self._parse_date(row["日期"])
                val = self._parse_chinese_number(row.get("今值"))
                if d and val is not None:
                    records.append({
                        "title": "中国GDP年率", "date": d, "value": val,
                        "forecast": self._parse_chinese_number(row.get("预测值")),
                        "previous": self._parse_chinese_number(row.get("前值")),
                        "country": "CN", "unit": "percent", "importance": 5,
                    })
        except Exception as e:
            logger.warning(f"China GDP fetch failed: {e}")

    # ── normalize ────────────────────────────────────────────

    def normalize(self, raw: dict) -> dict:
        return {
            "source_name": self.source_name,
            "title": raw["title"],
            "description": f"China macro: {raw['title']}",
            "event_date": raw["date"],
            "actual_value": raw.get("value"),
            "forecast_value": raw.get("forecast"),
            "previous_value": raw.get("previous"),
            "country": raw.get("country", "CN"),
            "impact": "high" if raw.get("importance", 3) >= 4 else "medium",
            "importance": raw.get("importance", 3),
            "url": "",
            "raw_data": {
                "unit": raw.get("unit"),
            },
        }
