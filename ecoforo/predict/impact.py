"""Event impact analysis — quantify how economic events affect copper prices."""

import logging
import warnings
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from ecoforo.db.session import SessionLocal
from ecoforo.db.models import EconomicEvent, EventSource

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def _get_copper_returns() -> pd.Series:
    """Get daily copper returns."""
    db = SessionLocal()
    try:
        rows = (
            db.query(EconomicEvent)
            .filter(
                EconomicEvent.title == "Copper Futures — Close",
                EconomicEvent.actual_value.isnot(None),
            )
            .order_by(EconomicEvent.event_date.asc())
            .all()
        )
        df = pd.DataFrame([{
            'date': pd.Timestamp(e.event_date.date()),
            'price': e.actual_value,
        } for e in rows])
        if df.empty:
            return pd.Series(dtype=float)
        df = df.dropna().drop_duplicates('date', keep='last').set_index('date')
        return df['price'].pct_change()
    finally:
        db.close()


def _get_classified_events() -> pd.DataFrame:
    """Get GDELT events with categories."""
    db = SessionLocal()
    try:
        gdelt = db.query(EventSource).filter(EventSource.name == 'gdelt').first()
        if not gdelt:
            return pd.DataFrame()
        rows = (
            db.query(EconomicEvent)
            .filter(EconomicEvent.source_id == gdelt.id)
            .order_by(EconomicEvent.event_date.asc())
            .all()
        )
        return pd.DataFrame([{
            'date': pd.Timestamp(e.event_date.date()),
            'title': e.title,
            'category': (e.raw_data or {}).get('category', 'uncategorized'),
            'sentiment': (e.raw_data or {}).get('sentiment', 0),
            'impact': e.impact or 'low',
        } for e in rows])
    finally:
        db.close()


def analyze_event_impact(horizon_days: int = 30) -> str:
    """Measure copper price changes following different event types."""
    copper_returns = _get_copper_returns()
    events = _get_classified_events()

    if copper_returns.empty or events.empty:
        return "No data available for impact analysis."

    lines = []
    lines.append("═" * 65)
    lines.append(f"📊 事件冲击分析 — {horizon_days}日铜价反应")
    lines.append("═" * 65)

    categories = events['category'].value_counts()
    lines.append(f"事件分类分布: {dict(categories)}")
    lines.append("")

    # For each category, measure forward returns after events
    lines.append(f"{'事件类型':<18s} {'事件数':>6s} {'7日':>8s} {'30日':>8s} {'冲击等级':>8s}")
    lines.append("-" * 55)

    category_impacts = {}
    for cat in categories.index:
        if cat == 'uncategorized':
            continue
        cat_events = events[events['category'] == cat]
        returns_7d = []
        returns_30d = []

        for _, evt in cat_events.iterrows():
            evt_date = evt['date']
            # Forward returns
            fwd_7 = copper_returns.loc[evt_date:evt_date + timedelta(days=7)]
            fwd_30 = copper_returns.loc[evt_date:evt_date + timedelta(days=horizon_days)]
            if len(fwd_7) > 0:
                returns_7d.append((1 + fwd_7).prod() - 1)
            if len(fwd_30) > 0:
                returns_30d.append((1 + fwd_30).prod() - 1)

        if returns_7d:
            avg_7 = np.mean(returns_7d) * 100
            avg_30 = np.mean(returns_30d) * 100 if returns_30d else 0

            direction = "📈利好" if avg_30 > 1 else ("📉利空" if avg_30 < -1 else "➡️中性")
            severity = "🔴高" if abs(avg_30) > 3 else ("🟡中" if abs(avg_30) > 1 else "🟢低")

            category_impacts[cat] = {
                "count": len(returns_7d),
                "avg_7d": avg_7,
                "avg_30d": avg_30,
                "direction": direction,
            }
            lines.append(
                f"{cat:<18s} {len(returns_7d):>6d} {avg_7:>+7.1f}% {avg_30:>+7.1f}% {direction} {severity}"
            )

    # Sentiment impact
    lines.append("")
    lines.append("═" * 65)
    lines.append("💬 新闻情感对铜价的影响")
    lines.append("═" * 65)

    pos_events = events[events['sentiment'] > 0]
    neg_events = events[events['sentiment'] < 0]

    for label, evt_df in [("正面新闻", pos_events), ("负面新闻", neg_events)]:
        if evt_df.empty:
            continue
        returns_30d = []
        for _, evt in evt_df.iterrows():
            fwd = copper_returns.loc[evt['date']:evt['date'] + timedelta(days=horizon_days)]
            if len(fwd) > 0:
                returns_30d.append((1 + fwd).prod() - 1)
        if returns_30d:
            avg_ret = np.mean(returns_30d) * 100
            win_rate = sum(1 for r in returns_30d if r > 0) / len(returns_30d) * 100
            lines.append(f"  {label} ({len(returns_30d)}条): 均值 {avg_ret:+.1f}%, 胜率 {win_rate:.0f}%")

    return "\n".join(lines)
