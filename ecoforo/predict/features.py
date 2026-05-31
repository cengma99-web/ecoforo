"""Feature engineering for copper price prediction."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import func

from ecoforo.db.session import SessionLocal
from ecoforo.db.models import EconomicEvent, EventSource

logger = logging.getLogger(__name__)

# Feature definitions
PRICE_FEATURES = [
    "lag_1", "lag_7", "lag_30",          # Price lags
    "ma_20", "ma_60",                     # Moving averages
    "volatility_30", "rsi_14",            # Technical indicators
]

MACRO_FEATURES = [
    "cn_pmi", "cn_m2_yoy", "cn_cpi", "cn_ppi",    # China macro
    "us_rate", "us_10y",                            # US macro
]

COMMODITY_FEATURES = [
    "aluminum_30d_pct", "silver_30d_pct", "oil_30d_pct",
]

NEWS_FEATURES = [
    "news_high_30d", "news_rate_30d",
]

ALL_FEATURES = PRICE_FEATURES + MACRO_FEATURES + COMMODITY_FEATURES + NEWS_FEATURES
TARGET_CLASS = "direction_30d"       # 1=up, 0=flat, -1=down
TARGET_REGRESSION = "return_30d"     # 30-day forward return


def _fetch_series(title: str, start_date=None) -> pd.Series:
    """Fetch a single time series from the database."""
    db = SessionLocal()
    try:
        q = db.query(EconomicEvent).filter(
            EconomicEvent.title == title,
            EconomicEvent.actual_value.isnot(None),
        )
        if start_date:
            q = q.filter(EconomicEvent.event_date >= start_date)
        rows = q.order_by(EconomicEvent.event_date.asc()).all()
        df = pd.DataFrame([{
            'date': pd.Timestamp(e.event_date.date()),
            'value': e.actual_value,
        } for e in rows])
        if df.empty:
            return pd.Series(dtype=float)
        df = df.dropna().drop_duplicates('date', keep='last').set_index('date')
        return df['value']
    finally:
        db.close()


def _fetch_indicator(name_match: str, start_date=None) -> pd.Series:
    """Fetch indicator by partial name match, returning latest per month."""
    db = SessionLocal()
    try:
        q = db.query(EconomicEvent).filter(
            EconomicEvent.title.ilike(f"%{name_match}%"),
            EconomicEvent.actual_value.isnot(None),
        )
        if start_date:
            q = q.filter(EconomicEvent.event_date >= start_date)
        rows = q.order_by(EconomicEvent.event_date.asc()).all()
        df = pd.DataFrame([{
            'date': pd.Timestamp(e.event_date.date()),
            'value': e.actual_value,
        } for e in rows])
        if df.empty:
            return pd.Series(dtype=float)
        df = df.dropna().drop_duplicates('date', keep='last').set_index('date')
        return df['value'].resample('ME').last()
    finally:
        db.close()


def _fetch_news_count(days: int = 30, keyword: Optional[str] = None) -> pd.Series:
    """Count GDELT news articles per day, optionally filtered by keyword."""
    db = SessionLocal()
    try:
        gdelt = db.query(EventSource).filter(EventSource.name == 'gdelt').first()
        if not gdelt:
            return pd.Series(dtype=float)
        q = db.query(EconomicEvent).filter(
            EconomicEvent.source_id == gdelt.id,
        )
        if keyword:
            q = q.filter(EconomicEvent.title.ilike(f"%{keyword}%"))
        rows = q.order_by(EconomicEvent.event_date.asc()).all()
        df = pd.DataFrame([{
            'date': pd.Timestamp(e.event_date.date()),
            'count': 1,
        } for e in rows])
        if df.empty:
            return pd.Series(dtype=float)
        daily = df.groupby('date').sum()
        # Rolling sum over N days
        return daily['count'].rolling(days).sum()
    finally:
        db.close()


def build_features(start_date: str = "2018-01-01") -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build feature matrix and target variables for copper prediction.

    Returns:
        X: DataFrame with features indexed by date
        y_cls: Series with direction classification (1, 0, -1)
        y_reg: Series with 30-day forward return
    """
    start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)

    # ── Core: Copper prices ──────────────────────────────────
    copper = _fetch_series("Copper Futures — Close", start_dt)
    if len(copper) < 60:
        raise ValueError(f"Insufficient copper data: {len(copper)} rows")

    df = pd.DataFrame(index=copper.index)
    df['copper_price'] = copper

    # Lag features
    df['lag_1'] = copper.shift(1)
    df['lag_7'] = copper.shift(7)
    df['lag_30'] = copper.shift(30)

    # Moving averages
    df['ma_20'] = copper.rolling(20).mean()
    df['ma_60'] = copper.rolling(60).mean()

    # Volatility
    df['volatility_30'] = copper.pct_change().rolling(30).std()

    # RSI(14)
    delta = copper.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # ── Target variables ─────────────────────────────────────
    forward = copper.shift(-30)
    df['return_30d'] = (forward - copper) / copper
    df['direction_30d'] = np.select(
        [df['return_30d'] > 0.02, df['return_30d'] < -0.02],
        [2, 0],   # 2=up, 0=down, 1=flat (XGBoost requires 0-indexed)
        default=1,
    )
    # Class mapping: 0=down, 1=flat, 2=up

    # ── China macro ──────────────────────────────────────────
    cn_pmi = _fetch_indicator("制造业PMI", start_dt)
    cn_m2 = _fetch_indicator("M2同比", start_dt)
    cn_cpi = _fetch_indicator("CPI年率", start_dt)
    cn_ppi = _fetch_indicator("PPI年率", start_dt)

    # Forward-fill (then back-fill) monthly data to daily
    for name, series in [("cn_pmi", cn_pmi), ("cn_m2_yoy", cn_m2),
                          ("cn_cpi", cn_cpi), ("cn_ppi", cn_ppi)]:
        if not series.empty:
            combined = series.reindex(df.index).ffill().bfill()
            df[name] = combined

    # ── US macro ─────────────────────────────────────────────
    us_rate = _fetch_series("US Federal Funds Effective Rate", start_dt)
    us_10y = _fetch_series("US 10-Year Treasury Yield", start_dt)
    for name, series in [("us_rate", us_rate), ("us_10y", us_10y)]:
        if not series.empty:
            combined = series.reindex(df.index).ffill().bfill()
            df[name] = combined

    # ── Commodity linkages ───────────────────────────────────
    for title, col in [
        ("Aluminum Futures — Close", "aluminum_30d_pct"),
        ("Silver Futures — Close", "silver_30d_pct"),
        ("Crude Oil WTI Futures — Close", "oil_30d_pct"),
    ]:
        s = _fetch_series(title, start_dt)
        if not s.empty:
            pct = s.pct_change(30).reindex(df.index).ffill().bfill()
            df[col] = pct

    # ── News features ────────────────────────────────────────
    news_total = _fetch_news_count(30)
    news_rate = _fetch_news_count(30, "rate")

    if not news_total.empty:
        df['news_high_30d'] = news_total.reindex(df.index).ffill().bfill().fillna(0)
    else:
        df['news_high_30d'] = 0

    if not news_rate.empty:
        df['news_rate_30d'] = news_rate.reindex(df.index).ffill().bfill().fillna(0)
    else:
        df['news_rate_30d'] = 0

    # ── Cleanup ──────────────────────────────────────────────
    # Drop rows with NaN in any feature or target
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    df_clean = df.dropna(subset=feature_cols + ['direction_30d', 'return_30d'])

    X = df_clean[feature_cols].copy()
    y_cls = df_clean['direction_30d'].copy()
    y_reg = df_clean['return_30d'].copy()

    logger.info(f"Features built: {len(X)} rows, {len(feature_cols)} features")
    return X, y_cls, y_reg
