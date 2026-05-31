"""Economic event classification and sentiment analysis.

Keyword-based classification + sentiment scoring.
No LLM dependency — fast, offline, and sufficient for feature engineering.
"""

import logging
import re
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# ── Event Categories ─────────────────────────────────────
CATEGORIES = {
    "rate_decision": {
        "keywords": [
            "rate hike", "rate cut", "rate decision", "interest rate",
            "central bank", "fed", "ECB", "PBOC", "BOK", "BOJ", "BOE",
            "monetary policy", "tighten", "ease", "加息", "降息", "利率",
            "央行", "federal reserve", "basis point", "bps", "hawkish", "dovish",
        ],
        "importance": 5,
    },
    "inflation": {
        "keywords": [
            "inflation", "CPI", "PPI", "consumer price", "deflation",
            "price index", "HICP", "通胀", "物价", "purchasing power",
        ],
        "importance": 4,
    },
    "trade_policy": {
        "keywords": [
            "tariff", "trade war", "trade deal", "trade deficit",
            "export ban", "import ban", "sanction", "protectionism",
            "WTO", "trade agreement", "关税", "贸易战", "进出口",
        ],
        "importance": 4,
    },
    "recession": {
        "keywords": [
            "recession", "economic contraction", "downturn", "slowdown",
            "negative growth", "depression", "crisis", "meltdown",
            "衰退", "经济危机", "负增长",
        ],
        "importance": 5,
    },
    "commodity": {
        "keywords": [
            "commodity", "copper", "aluminum", "oil", "crude", "gold",
            "metal", "mineral", "LME", "OPEC", "大宗商品", "铜", "原油",
            "supply chain", "shortage", "inventory",
        ],
        "importance": 3,
    },
    "employment": {
        "keywords": [
            "unemployment", "jobless", "payroll", "labor market",
            "employment", "wage", "失业", "就业",
        ],
        "importance": 3,
    },
    "geopolitical": {
        "keywords": [
            "war", "conflict", "invasion", "sanction", "embargo",
            "geopolitical", "military", "missile", "coup", "regime",
            "战争", "冲突", "制裁",
        ],
        "importance": 5,
    },
    "market_turmoil": {
        "keywords": [
            "crash", "selloff", "plunge", "surge", "volatile",
            "rally", "panic", "bear market", "bull market",
            "暴跌", "暴涨", "股灾",
        ],
        "importance": 4,
    },
    "fiscal_policy": {
        "keywords": [
            "stimulus", "bailout", "subsidy", "tax cut", "tax hike",
            "government spending", "deficit", "debt ceiling",
            "财政", "刺激", "补贴",
        ],
        "importance": 3,
    },
    "housing": {
        "keywords": [
            "housing", "real estate", "property", "mortgage",
            "home price", "construction", "房地产", "房价",
        ],
        "importance": 3,
    },
}

# ── Sentiment Lexicon ────────────────────────────────────
POSITIVE_WORDS = {
    "growth", "recovery", "surge", "rally", "boom", "expansion",
    "improve", "strong", "bullish", "optimism", "upbeat",
    "增长", "回升", "改善", "强劲", "复苏", "利好",
    "cut", "ease", "stimulus", "support",
}

NEGATIVE_WORDS = {
    "recession", "crisis", "crash", "plunge", "collapse", "depression",
    "downturn", "slowdown", "weak", "bearish", "pessimism", "fear",
    "hawkish", "tighten", "sanction", "war", "conflict", "default",
    "衰退", "危机", "暴跌", "下降", "疲软", "利空",
    "hike", "tariff", "inflation", "debt",
}

NEGATION_WORDS = {"no", "not", "never", "avoid", "prevent", "halt"}


def classify_event(title: str) -> Tuple[str, int]:
    """Classify an event title into a category. Returns (category, importance)."""
    title_lower = title.lower()
    best_score = 0
    best_category = "uncategorized"
    best_importance = 2

    for cat, info in CATEGORIES.items():
        score = sum(1 for kw in info["keywords"] if kw.lower() in title_lower)
        if score > best_score:
            best_score = score
            best_category = cat
            best_importance = info["importance"]

    return best_category, best_importance


def analyze_sentiment(title: str) -> float:
    """Simple keyword-based sentiment score. Range: -1 (very negative) to +1 (very positive)."""
    words = set(re.findall(r'\w+', title.lower()))
    if not words:
        return 0.0

    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    negations = sum(1 for w in words if w in NEGATION_WORDS)

    # Negations flip nearby sentiment
    if negations > 0:
        pos, neg = neg, pos

    total = pos + neg
    if total == 0:
        return 0.0

    return (pos - neg) / total


def classify_and_score(title: str) -> dict:
    """Full classification + sentiment for one event title."""
    category, importance = classify_event(title)
    sentiment = analyze_sentiment(title)
    return {
        "category": category,
        "importance_override": importance,
        "sentiment": round(sentiment, 2),
    }


def classify_events_db(session=None):
    """Run classification on all uncategorized GDELT events in DB."""
    from sqlalchemy.orm.attributes import flag_modified
    from ecoforo.db.session import SessionLocal
    from ecoforo.db.models import EconomicEvent, EventSource

    db = session or SessionLocal()
    try:
        gdelt = db.query(EventSource).filter(EventSource.name == 'gdelt').first()
        if not gdelt:
            return {"classified": 0}

        events = (
            db.query(EconomicEvent)
            .filter(
                EconomicEvent.source_id == gdelt.id,
                EconomicEvent.raw_data.isnot(None),
            )
            .all()
        )

        classified = 0
        for e in events:
            result = classify_and_score(e.title)
            if result["category"] != "uncategorized" or result["sentiment"] != 0:
                if e.raw_data is None:
                    e.raw_data = {}
                e.raw_data["category"] = result["category"]
                e.raw_data["importance_classified"] = result["importance_override"]
                e.raw_data["sentiment"] = result["sentiment"]

                # Update importance if classified higher
                if result["importance_override"] > (e.importance or 0):
                    e.importance = result["importance_override"]
                if result["sentiment"] != 0:
                    e.sentiment = result["sentiment"]

                flag_modified(e, "raw_data")
                classified += 1

        db.commit()
        logger.info(f"Classified {classified}/{len(events)} GDELT events")
        return {"classified": classified, "total": len(events)}
    finally:
        if session is None:
            db.close()
