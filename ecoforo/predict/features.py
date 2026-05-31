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
    "lag_1", "lag_7", "lag_30", "lag_60", "lag_90",  # Price lags
    "ma_20", "ma_60", "ma_120",                         # Moving averages
    "ma_ratio_20_60", "ma_ratio_20_120",               # MA cross signals
    "price_vs_ma20", "price_vs_ma60",                   # Price position vs MA
    "volatility_30", "volatility_90",                   # Volatility
    "rsi_14", "rsi_30",                                 # RSI
    "macd", "macd_signal", "macd_hist",                 # MACD
    "bb_position", "bb_width",                          # Bollinger Bands
    "momentum_7", "momentum_30",                        # Momentum
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
TARGET_CLASS = "direction_30d"       # Binary: 0=down, 1=up
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

    # ── Lag features ────────────────────────────────────────
    df['lag_1'] = copper.shift(1)
    df['lag_7'] = copper.shift(7)
    df['lag_30'] = copper.shift(30)
    df['lag_60'] = copper.shift(60)
    df['lag_90'] = copper.shift(90)

    # ── Moving averages ─────────────────────────────────────
    df['ma_20'] = copper.rolling(20).mean()
    df['ma_60'] = copper.rolling(60).mean()
    df['ma_120'] = copper.rolling(120).mean()

    # MA cross signals
    df['ma_ratio_20_60'] = df['ma_20'] / df['ma_60'] - 1
    df['ma_ratio_20_120'] = df['ma_20'] / df['ma_120'] - 1

    # Price position vs MA
    df['price_vs_ma20'] = copper / df['ma_20'] - 1
    df['price_vs_ma60'] = copper / df['ma_60'] - 1

    # ── Volatility ──────────────────────────────────────────
    returns = copper.pct_change()
    df['volatility_30'] = returns.rolling(30).std()
    df['volatility_90'] = returns.rolling(90).std()

    # ── RSI ─────────────────────────────────────────────────
    delta = copper.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))

    avg_gain_14 = gain.rolling(14).mean()
    avg_loss_14 = loss.rolling(14).mean()
    rs_14 = avg_gain_14 / avg_loss_14.replace(0, np.nan)
    df['rsi_14'] = 100 - (100 / (1 + rs_14))

    avg_gain_30 = gain.rolling(30).mean()
    avg_loss_30 = loss.rolling(30).mean()
    rs_30 = avg_gain_30 / avg_loss_30.replace(0, np.nan)
    df['rsi_30'] = 100 - (100 / (1 + rs_30))

    # ── MACD ────────────────────────────────────────────────
    ema_12 = copper.ewm(span=12, adjust=False).mean()
    ema_26 = copper.ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # ── Bollinger Bands ─────────────────────────────────────
    bb_mid = copper.rolling(20).mean()
    bb_std = copper.rolling(20).std()
    df['bb_position'] = (copper - bb_mid) / (2 * bb_std + 1e-9)  # position within bands
    df['bb_width'] = (2 * bb_std) / bb_mid  # band width as % of price

    # ── Momentum ────────────────────────────────────────────
    df['momentum_7'] = copper / copper.shift(7) - 1
    df['momentum_30'] = copper / copper.shift(30) - 1

    # ── Target variables ─────────────────────────────────────
    forward = copper.shift(-30)
    df['return_30d'] = (forward - copper) / copper
    df['direction_30d'] = (df['return_30d'] > 0).astype(int)  # 0=down, 1=up

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
