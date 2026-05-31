"""ecoforo Streamlit dashboard — events, analysis & prediction."""

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import func, desc

from ecoforo.db.session import SessionLocal
from ecoforo.db.models import EconomicEvent, EventSource, DeadLetter

st.set_page_config(page_title="ecoforo — Economic Dashboard", page_icon="📊", layout="wide")

st.title("📊 ecoforo — Economic Intelligence")
st.caption("Data pipeline · Market analysis · Price prediction")

# ═══════════════ Data Loaders ═══════════════

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
            .group_by(EconomicEvent.country).order_by(desc("count")).limit(10).all()
        )
        by_source = (
            db.query(EventSource.name, func.count(EconomicEvent.id).label("count"))
            .join(EconomicEvent).group_by(EventSource.name).all()
        )
        recent = (
            db.query(EconomicEvent).order_by(EconomicEvent.event_date.desc()).limit(100).all()
        )
        return {
            "total_events": total_events, "total_sources": total_sources,
            "total_dead": total_dead, "by_country": by_country,
            "by_source": by_source, "recent": recent,
        }
    finally:
        db.close()


@st.cache_data(ttl=300)
def load_events_for_date_range(start, end, countries=None, importance_min=0):
    db = SessionLocal()
    try:
        q = db.query(EconomicEvent).filter(
            EconomicEvent.event_date >= start, EconomicEvent.event_date <= end)
        if countries:
            q = q.filter(EconomicEvent.country.in_(countries))
        if importance_min > 0:
            q = q.filter(EconomicEvent.importance >= importance_min)
        return q.order_by(EconomicEvent.event_date.desc()).all()
    finally:
        db.close()


@st.cache_data(ttl=600)
def load_copper_data(months=24):
    db = SessionLocal()
    try:
        rows = (
            db.query(EconomicEvent)
            .filter(EconomicEvent.title == "Copper Futures — Close",
                    EconomicEvent.actual_value.isnot(None),
                    EconomicEvent.event_date >= datetime.now(timezone.utc) - timedelta(days=months*31))
            .order_by(EconomicEvent.event_date.asc()).all()
        )
        df = pd.DataFrame([{
            'date': e.event_date, 'price': e.actual_value
        } for e in rows])
        if not df.empty:
            df = df.dropna().drop_duplicates('date', keep='last').set_index('date')
        return df
    finally:
        db.close()


@st.cache_data(ttl=600)
def load_macro_snapshot():
    """Load latest China + US macro values."""
    db = SessionLocal()
    try:
        indicators = [
            ("中国CPI年率", "China CPI"),
            ("中国制造业PMI", "China PMI"),
            ("中国M2同比增速", "China M2"),
            ("中国GDP年率", "China GDP"),
            ("US Federal Funds Effective Rate", "US Rate"),
            ("US 10-Year Treasury Yield", "US 10Y"),
            ("US CPI (All Urban Consumers, All Items)", "US CPI"),
        ]
        result = {}
        for name, key in indicators:
            e = (
                db.query(EconomicEvent)
                .filter(EconomicEvent.title.ilike(f"%{name}%"),
                        EconomicEvent.actual_value.isnot(None))
                .order_by(EconomicEvent.event_date.desc()).first()
            )
            if e:
                result[key] = {"value": e.actual_value, "date": str(e.event_date)[:10]}
        return result
    finally:
        db.close()


@st.cache_data(ttl=600)
def load_correlation_data():
    """Compute commodity correlations with copper."""
    db = SessionLocal()
    try:
        commodities = [
            "Aluminum Futures — Close", "Crude Oil WTI Futures — Close",
            "Gold Futures — Close", "Silver Futures — Close",
            "Zinc Futures — Close", "Natural Gas Futures — Close",
        ]
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        corr_results = {}

        # Copper monthly returns
        cu_rows = (db.query(EconomicEvent).filter(
            EconomicEvent.title == "Copper Futures — Close",
            EconomicEvent.actual_value.isnot(None),
            EconomicEvent.event_date >= start).order_by(EconomicEvent.event_date.asc()).all())
        cu_df = pd.DataFrame([{'date': e.event_date, 'val': e.actual_value} for e in cu_rows])
        if cu_df.empty:
            return {}
        cu_df = cu_df.dropna().drop_duplicates('date', keep='last').set_index('date')
        cu_m = cu_df['val'].resample('ME').last().pct_change().dropna()

        for cm in commodities:
            rows = (db.query(EconomicEvent).filter(
                EconomicEvent.title == cm, EconomicEvent.actual_value.isnot(None),
                EconomicEvent.event_date >= start).order_by(EconomicEvent.event_date.asc()).all())
            df = pd.DataFrame([{'date': e.event_date, 'val': e.actual_value} for e in rows])
            if df.empty:
                continue
            df = df.dropna().drop_duplicates('date', keep='last').set_index('date')
            s_m = df['val'].resample('ME').last().pct_change().dropna()
            common = cu_m.index.intersection(s_m.index)
            if len(common) < 3:
                continue
            corr = cu_m.loc[common].corr(s_m.loc[common])
            if not np.isnan(corr):
                label = cm.split(" — ")[0].replace(" Futures", "")
                corr_results[label] = corr
        return corr_results
    finally:
        db.close()


@st.cache_data(ttl=600)
def load_prediction():
    """Load latest model prediction."""
    try:
        from ecoforo.predict.predict import predict_latest
        return predict_latest()
    except Exception:
        return None


# ═══════════════ Sidebar ═══════════════

st.sidebar.header("📊 Dashboard")

summary = load_summary()
st.sidebar.metric("Total Events", f"{summary['total_events']:,}")
st.sidebar.metric("Sources", summary['total_sources'])

st.sidebar.divider()
st.sidebar.header("Filters")

date_range = st.sidebar.date_input(
    "Event Date Range",
    value=(date.today() - timedelta(days=7), date.today()),
)
imp_min = st.sidebar.slider("Min Importance", 1, 5, 1)
countries_filter = st.sidebar.multiselect(
    "Countries", ["US", "CN", "JP", "DE", "GB", "EU", "KR", "IN"],
    default=["US", "CN"],
)

# ═══════════════ KPI Row ═══════════════

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Events", f"{summary['total_events']:,}")
col2.metric("Active Sources", summary['total_sources'])
col3.metric("Dead Letters", summary['total_dead'])
latest = max((e.event_date for e in summary['recent']), default=None)
col4.metric("Latest Event", str(latest)[:10] if latest else "N/A")

# ═══════════════ Tabs ═══════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📅 Events", "📈 Charts", "🔬 Analysis", "🔮 Prediction", "📋 Status"
])

# ── Tab 1: Event Timeline ───────────────────────────────

with tab1:
    st.subheader("Economic Events")
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_dt = datetime.combine(date_range[0], datetime.min.time()).replace(tzinfo=timezone.utc)
        end_dt = datetime.combine(date_range[1], datetime.max.time()).replace(tzinfo=timezone.utc)
        events = load_events_for_date_range(start_dt, end_dt, countries_filter, imp_min)

        if events:
            df = pd.DataFrame([{
                "Date": str(e.event_date)[:10],
                "Country": e.country or "—",
                "★": "★" * (e.importance or 0),
                "Title": e.title,
                "Actual": e.actual_value if e.actual_value is not None else "—",
                "Forecast": e.forecast_value if e.forecast_value is not None else "—",
            } for e in events])
            st.dataframe(df, use_container_width=True, height=500, hide_index=True)
        else:
            st.info("No events match filters.")

# ── Tab 2: Charts ───────────────────────────────────────

with tab2:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Events by Country")
        if summary["by_country"]:
            df_country = pd.DataFrame(summary["by_country"], columns=["Country", "Count"])
            st.bar_chart(df_country.set_index("Country"), use_container_width=True)

    with col_right:
        st.subheader("Events by Source")
        if summary["by_source"]:
            df_source = pd.DataFrame(summary["by_source"], columns=["Source", "Count"])
            st.bar_chart(df_source.set_index("Source"), use_container_width=True)

# ── Tab 3: Analysis ─────────────────────────────────────

with tab3:
    st.subheader("🔧 Copper Price Trend")

    df_cu = load_copper_data()
    if not df_cu.empty and 'price' in df_cu.columns:
        df_cu['MA20'] = df_cu['price'].rolling(20).mean()
        df_cu['MA60'] = df_cu['price'].rolling(60).mean()

        # Price chart with MAs
        chart_data = df_cu[['price', 'MA20', 'MA60']].rename(
            columns={'price': 'Price', 'MA20': 'MA20', 'MA60': 'MA60'})
        st.line_chart(chart_data, use_container_width=True, height=350)

        # Quick stats
        c1, c2, c3, c4 = st.columns(4)
        latest_price = df_cu['price'].iloc[-1]
        c1.metric("Latest", f"${latest_price:.2f}")
        if len(df_cu) >= 20:
            ma20 = df_cu['MA20'].iloc[-1]
            c2.metric("MA20", f"${ma20:.2f}", f"{latest_price - ma20:+.2f}")
        if len(df_cu) >= 60:
            ma60 = df_cu['MA60'].iloc[-1]
            c3.metric("MA60", f"${ma60:.2f}", f"{latest_price - ma60:+.2f}")
        c4.metric("30d Vol", f"{df_cu['price'].pct_change().tail(30).std()*100:.1f}%")
    else:
        st.warning("No copper price data available.")

    # Macro snapshot
    st.divider()
    st.subheader("🌐 Macro Snapshot")
    macro = load_macro_snapshot()
    if macro:
        cols = st.columns(len(macro))
        for i, (key, val) in enumerate(macro.items()):
            cols[i].metric(key, f"{val['value']:.2f}", f"as of {val['date']}")

    # Correlations
    st.divider()
    st.subheader("🔗 Commodity Correlations with Copper")

    corr_data = load_correlation_data()
    if corr_data:
        corr_df = pd.DataFrame({
            'Commodity': list(corr_data.keys()),
            'Correlation': list(corr_data.values()),
        }).sort_values('Correlation')
        corr_df['Color'] = corr_df['Correlation'].apply(
            lambda x: '#22c55e' if x > 0 else '#ef4444')
        st.bar_chart(corr_df.set_index('Commodity')['Correlation'], use_container_width=True)

# ── Tab 4: Prediction ───────────────────────────────────

with tab4:
    st.subheader("🔮 Copper 30-Day Prediction")

    pred = load_prediction()
    if pred:
        # Signal card
        signal_color = {
            "🟢 买入": "#22c55e", "🔴 卖出": "#ef4444", "🟡 等待": "#eab308"
        }
        bg = signal_color.get(pred.get('signal', ''), '#6b7280')

        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.markdown(f"""
            <div style="background:{bg};padding:24px;border-radius:12px;text-align:center;color:white">
                <h1 style="margin:0;font-size:48px">{pred.get('signal', '')}</h1>
                <p style="font-size:20px;margin:8px 0">{pred.get('direction_label', '')}</p>
                <p style="opacity:0.9">Confidence: {pred.get('probabilities', {}).get('up', 0)*100 if pred.get('probabilities', {}).get('up', 0) > pred.get('probabilities', {}).get('down', 0) else pred.get('probabilities', {}).get('down', 0)*100:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)

        with col_right:
            # Probability bars
            probs = pred.get('probabilities', {})
            st.markdown("**Direction Probabilities**")
            if probs:
                st.progress(probs.get('up', 0), text=f"📈 Up: {probs.get('up', 0):.0%}")
                if 'flat' in probs:
                    st.progress(probs.get('flat', 0), text=f"➡️ Flat: {probs.get('flat', 0):.0%}")
                st.progress(probs.get('down', 0), text=f"📉 Down: {probs.get('down', 0):.0%}")

        # Prediction details
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        current = pred.get('current_price', 0)
        target = pred.get('predicted_price', 0)
        ret = pred.get('predicted_return_pct', 0)
        c1.metric("Current Price", f"${current:.2f}/lb" if current else "N/A")
        c2.metric("Predicted (30d)", f"${target:.2f}/lb" if target else "N/A",
                  f"{ret:+.1f}%" if ret else None)
        c3.metric("Date", pred.get('date', 'N/A'))
        metrics = pred.get('model_metrics', {}).get('classifier', {})
        c4.metric("CV Accuracy", f"{metrics.get('cv_accuracy', 0):.1%}")

        # Feature importance
        features = pred.get('model_metrics', {}).get('features', [])
        if features:
            st.divider()
            st.subheader("📊 Top Predictive Features")
            top = features[:8]
            feat_df = pd.DataFrame({
                'Feature': [f[0] for f in top],
                'Importance': [f[1] for f in top],
            }).sort_values('Importance')
            st.bar_chart(feat_df.set_index('Feature'), use_container_width=True)
    else:
        st.warning("No prediction available. Run 'ecoforo predict copper --train' first.")

# ── Tab 5: Pipeline Status ──────────────────────────────

with tab5:
    st.subheader("Data Pipeline")
    db = SessionLocal()
    try:
        sources = db.query(EventSource).all()
        if sources:
            df_src = pd.DataFrame([{
                "Source": s.name,
                "Type": str(s.source_type.value) if hasattr(s.source_type, 'value') else str(s.source_type),
                "Active": "✅" if s.is_active else "❌",
                "Events": db.query(func.count(EconomicEvent.id)).filter(
                    EconomicEvent.source_id == s.id).scalar(),
                "Last Fetch": str(s.last_fetch_at)[:19] if s.last_fetch_at else "never",
                "Last Backfill": str(s.last_backfill_at)[:19] if s.last_backfill_at else "never",
            } for s in sources])
            st.dataframe(df_src, use_container_width=True, hide_index=True)

            # Event count over time
            st.divider()
            st.subheader("Event Collection Timeline")
            event_dates = (
                db.query(
                    func.date(EconomicEvent.event_date).label('d'),
                    func.count(EconomicEvent.id).label('c'))
                .filter(EconomicEvent.event_date >= datetime.now(timezone.utc) - timedelta(days=90))
                .group_by('d').order_by('d').all()
            )
            if event_dates:
                df_timeline = pd.DataFrame(event_dates, columns=['Date', 'Count'])
                df_timeline = df_timeline.set_index('Date')
                st.area_chart(df_timeline, use_container_width=True)
        else:
            st.info("No sources registered. Run 'ecoforo collect --all' first.")
    finally:
        db.close()

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
