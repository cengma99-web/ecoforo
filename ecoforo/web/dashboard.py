"""ecoforo Streamlit dashboard — economic events & indicators browser."""

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
from sqlalchemy import func, desc

from ecoforo.db.session import SessionLocal
from ecoforo.db.models import EconomicEvent, EventSource, DeadLetter

st.set_page_config(page_title="ecoforo — Economic Dashboard", page_icon="📊", layout="wide")

st.title("📊 ecoforo — Economic Data Pipeline")
st.caption("Real-time economic events & indicators monitor")


@st.cache_data(ttl=300)
def load_summary():
    db = SessionLocal()
    try:
        total_events = db.query(func.count(EconomicEvent.id)).scalar()
        total_sources = db.query(func.count(EventSource.id)).scalar()
        total_dead = db.query(func.count(DeadLetter.id)).scalar()

        by_country = (
            db.query(EconomicEvent.country, func.count(EconomicEvent.id).label("count"))
            .filter(EconomicEvent.country.isnot(None))
            .group_by(EconomicEvent.country)
            .order_by(desc("count"))
            .limit(10)
            .all()
        )

        by_source = (
            db.query(EventSource.name, func.count(EconomicEvent.id).label("count"))
            .join(EconomicEvent)
            .group_by(EventSource.name)
            .all()
        )

        recent = (
            db.query(EconomicEvent)
            .order_by(EconomicEvent.event_date.desc())
            .limit(100)
            .all()
        )
        return {
            "total_events": total_events,
            "total_sources": total_sources,
            "total_dead": total_dead,
            "by_country": by_country,
            "by_source": by_source,
            "recent": recent,
        }
    finally:
        db.close()


@st.cache_data(ttl=300)
def load_events_for_date_range(start, end, countries=None, importance_min=0):
    db = SessionLocal()
    try:
        q = db.query(EconomicEvent).filter(
            EconomicEvent.event_date >= start,
            EconomicEvent.event_date <= end,
        )
        if countries:
            q = q.filter(EconomicEvent.country.in_(countries))
        if importance_min > 0:
            q = q.filter(EconomicEvent.importance >= importance_min)
        return q.order_by(EconomicEvent.event_date.desc()).all()
    finally:
        db.close()


# Sidebar
st.sidebar.header("Filters")
date_range = st.sidebar.date_input(
    "Date Range",
    value=(date.today() - timedelta(days=7), date.today()),
)
imp_min = st.sidebar.slider("Min Importance", 1, 5, 1)
countries_filter = st.sidebar.multiselect(
    "Countries", ["US", "CN", "JP", "DE", "GB", "EU", "KR", "IN"],
    default=["US", "CN"],
)

# Main
summary = load_summary()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Events", f"{summary['total_events']:,}")
col2.metric("Active Sources", summary["total_sources"])
col3.metric("Dead Letters", summary["total_dead"])
latest = max((e.event_date for e in summary["recent"]), default=None)
col4.metric("Latest Event", str(latest)[:10] if latest else "N/A")

tab1, tab2, tab3 = st.tabs(["📅 Event Timeline", "📈 By Country", "📋 Raw Data"])

with tab1:
    st.subheader("Recent Economic Events")
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_dt = datetime.combine(date_range[0], datetime.min.time()).replace(tzinfo=timezone.utc)
        end_dt = datetime.combine(date_range[1], datetime.max.time()).replace(tzinfo=timezone.utc)
        events = load_events_for_date_range(start_dt, end_dt, countries_filter, imp_min)

        if events:
            df = pd.DataFrame([{
                "Date": str(e.event_date)[:10],
                "Country": e.country or "—",
                "Importance": "★" * (e.importance or 0),
                "Title": e.title,
                "Actual": e.actual_value if e.actual_value is not None else "—",
                "Forecast": e.forecast_value if e.forecast_value is not None else "—",
                "Previous": e.previous_value if e.previous_value is not None else "—",
            } for e in events])
            st.dataframe(df, use_container_width=True, height=500, hide_index=True)
        else:
            st.info("No events match the current filters.")

with tab2:
    st.subheader("Events by Country")
    if summary["by_country"]:
        df_country = pd.DataFrame(summary["by_country"], columns=["Country", "Count"])
        st.bar_chart(df_country.set_index("Country"), use_container_width=True)

    st.subheader("Events by Source")
    if summary["by_source"]:
        df_source = pd.DataFrame(summary["by_source"], columns=["Source", "Count"])
        st.bar_chart(df_source.set_index("Source"), use_container_width=True)

with tab3:
    st.subheader("Data Pipeline Status")
    db = SessionLocal()
    try:
        sources = db.query(EventSource).all()
        if sources:
            df_src = pd.DataFrame([{
                "Name": s.name,
                "Type": str(s.source_type),
                "Active": bool(s.is_active),
                "Last Fetch": str(s.last_fetch_at)[:19] if s.last_fetch_at else "never",
                "Last Backfill": str(s.last_backfill_at)[:19] if s.last_backfill_at else "never",
            } for s in sources])
            st.dataframe(df_src, use_container_width=True, hide_index=True)
        else:
            st.info("No sources registered yet. Run 'ecoforo collect --all' first.")
    finally:
        db.close()

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
