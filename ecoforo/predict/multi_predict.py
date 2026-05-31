"""Multi-commodity prediction — extend copper model to other metals and energy."""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

COMMODITY_CONFIGS = {
    "copper": {
        "title": "Copper Futures — Close",
        "name": "铜",
        "unit": "USD/lb",
        "emoji": "🔧",
    },
    "aluminum": {
        "title": "Aluminum Futures — Close",
        "name": "铝",
        "unit": "USD/ton",
        "emoji": "🪶",
    },
    "oil": {
        "title": "Crude Oil WTI Futures — Close",
        "name": "原油",
        "unit": "USD/bbl",
        "emoji": "🛢️",
    },
    "gold": {
        "title": "Gold Futures — Close",
        "name": "黄金",
        "unit": "USD/oz",
        "emoji": "🥇",
    },
    "silver": {
        "title": "Silver Futures — Close",
        "name": "白银",
        "unit": "USD/oz",
        "emoji": "🥈",
    },
    "natural_gas": {
        "title": "Natural Gas Futures — Close",
        "name": "天然气",
        "unit": "USD/MMBtu",
        "emoji": "🔥",
    },
}

DIRECTION_LABELS = {1: "📈 上涨 (UP)", 0: "📉 下跌 (DOWN)"}
SIGNAL_LABELS = {1: "🟢 买入", 0: "🔴 卖出"}


def build_features_for_commodity(commodity_key: str, start_date: str = "2018-01-01"):
    """Build features using the target commodity's price instead of copper."""
    from datetime import datetime, timezone

    config = COMMODITY_CONFIGS.get(commodity_key)
    if not config:
        raise ValueError(f"Unknown commodity: {commodity_key}")

    from ecoforo.db.session import SessionLocal
    from ecoforo.db.models import EconomicEvent
    from ecoforo.predict import features as ft

    start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)

    # Fetch target commodity prices
    db = SessionLocal()
    try:
        rows = (
            db.query(EconomicEvent)
            .filter(
                EconomicEvent.title == config["title"],
                EconomicEvent.actual_value.isnot(None),
            )
            .order_by(EconomicEvent.event_date.asc())
            .all()
        )
        price_df = pd.DataFrame([{
            'date': pd.Timestamp(e.event_date.date()),
            'value': e.actual_value,
        } for e in rows])
    finally:
        db.close()

    if price_df.empty:
        raise ValueError(f"No price data for {commodity_key}")

    price_df = price_df.dropna().drop_duplicates('date', keep='last').set_index('date')
    prices = price_df['value']

    if len(prices) < 120:
        raise ValueError(f"Insufficient data for {commodity_key}: {len(prices)} rows")

    # Build price-derived features (same as copper)
    df = pd.DataFrame(index=prices.index)
    df['lag_1'] = prices.shift(1)
    df['lag_7'] = prices.shift(7)
    df['lag_30'] = prices.shift(30)
    df['lag_60'] = prices.shift(60)
    df['lag_90'] = prices.shift(90)
    df['ma_20'] = prices.rolling(20).mean()
    df['ma_60'] = prices.rolling(60).mean()
    df['ma_120'] = prices.rolling(120).mean()
    df['ma_ratio_20_60'] = df['ma_20'] / df['ma_60'] - 1
    df['ma_ratio_20_120'] = df['ma_20'] / df['ma_120'] - 1
    df['price_vs_ma20'] = prices / df['ma_20'] - 1
    df['price_vs_ma60'] = prices / df['ma_60'] - 1

    returns = prices.pct_change()
    df['volatility_30'] = returns.rolling(30).std()
    df['volatility_90'] = returns.rolling(90).std()

    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    rs_14 = gain.rolling(14).mean() / loss.rolling(14).mean().replace(0, np.nan)
    df['rsi_14'] = 100 - (100 / (1 + rs_14))
    rs_30 = gain.rolling(30).mean() / loss.rolling(30).mean().replace(0, np.nan)
    df['rsi_30'] = 100 - (100 / (1 + rs_30))

    ema12 = prices.ewm(span=12).mean()
    ema26 = prices.ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    bb_mid = prices.rolling(20).mean()
    bb_std = prices.rolling(20).std()
    df['bb_position'] = (prices - bb_mid) / (2 * bb_std + 1e-9)
    df['bb_width'] = (2 * bb_std) / bb_mid
    df['momentum_7'] = prices / prices.shift(7) - 1
    df['momentum_30'] = prices / prices.shift(30) - 1

    # Forward returns for target
    forward = prices.shift(-30)
    df['return_30d'] = (forward - prices) / prices
    df['direction_30d'] = (df['return_30d'] > 0).astype(int)

    # Macro features (same as copper model)
    cn_pmi = ft._fetch_indicator("制造业PMI", start_dt)
    cn_m2 = ft._fetch_indicator("M2同比", start_dt)
    cn_cpi = ft._fetch_indicator("CPI年率", start_dt)
    cn_ppi = ft._fetch_indicator("PPI年率", start_dt)

    for name, series in [("cn_pmi", cn_pmi), ("cn_m2_yoy", cn_m2),
                          ("cn_cpi", cn_cpi), ("cn_ppi", cn_ppi)]:
        if not series.empty:
            df[name] = series.reindex(df.index).ffill().bfill()

    us_rate = ft._fetch_series("US Federal Funds Effective Rate", start_dt)
    us_10y = ft._fetch_series("US 10-Year Treasury Yield", start_dt)
    for name, series in [("us_rate", us_rate), ("us_10y", us_10y)]:
        if not series.empty:
            df[name] = series.reindex(df.index).ffill().bfill()

    # News features
    news_total = ft._fetch_news_count(30)
    news_rate = ft._fetch_news_count(30, "rate")
    df['news_high_30d'] = news_total.reindex(df.index).ffill().bfill().fillna(0) if not news_total.empty else 0
    df['news_rate_30d'] = news_rate.reindex(df.index).ffill().bfill().fillna(0) if not news_rate.empty else 0

    # ── Commodity cross-features ─────────────────────────────
    for cm_title, cm_col in [
        ("Aluminum Futures — Close", "aluminum_30d_pct"),
        ("Silver Futures — Close", "silver_30d_pct"),
        ("Crude Oil WTI Futures — Close", "oil_30d_pct"),
    ]:
        s = ft._fetch_series(cm_title, start_dt)
        if not s.empty:
            df[cm_col] = s.pct_change(30).reindex(df.index).ffill().bfill()
        else:
            df[cm_col] = 0.0

    # Clean
    feature_cols = [c for c in ft.ALL_FEATURES if c in df.columns]
    df_clean = df.dropna(subset=feature_cols + ['direction_30d', 'return_30d'])

    X = df_clean[feature_cols].copy()
    y_cls = df_clean['direction_30d'].copy()
    y_reg = df_clean['return_30d'].copy()

    return X, y_cls, y_reg


def train_commodity(commodity_key: str) -> dict:
    """Train a model for a specific commodity."""
    from ecoforo.predict.train import train_models, save_models, MODEL_DIR

    logger.info(f"Training model for {commodity_key}...")
    X, y_cls, y_reg = build_features_for_commodity(commodity_key)
    clf, reg, metrics = train_models(X, y_cls, y_reg)

    model_path = MODEL_DIR / f"{commodity_key}_model.pkl"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump({'classifier': clf, 'regressor': reg, 'metrics': metrics}, f)

    config = COMMODITY_CONFIGS[commodity_key]
    return {
        "commodity": commodity_key,
        "name": config["name"],
        "cv_accuracy": metrics['classifier']['cv_accuracy'],
        "baseline": metrics['classifier']['baseline_accuracy'],
        "improvement": metrics['classifier']['improvement'],
        "n_samples": metrics['n_samples'],
        "model_path": str(model_path),
    }


def predict_commodity(commodity_key: str) -> dict:
    """Predict for a specific commodity."""
    config = COMMODITY_CONFIGS[commodity_key]
    model_path = Path(__file__).resolve().parent.parent.parent / "data" / f"{commodity_key}_model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"No model for {commodity_key}. Train first.")

    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    X, y_cls, y_reg = build_features_for_commodity(commodity_key)
    feature_cols = [c for c in X.columns if c in X.columns]
    latest = X[feature_cols].iloc[-1:]

    clf = data['classifier']
    reg = data['regressor']
    metrics = data['metrics']

    dir_pred = int(clf.predict(latest)[0])
    dir_proba = clf.predict_proba(latest)[0]
    ret_pred = float(reg.predict(latest)[0])
    current_price = float(X['lag_1'].iloc[-1]) if 'lag_1' in X.columns else 0.0

    return {
        "commodity": commodity_key,
        "name": config["name"],
        "emoji": config["emoji"],
        "date": str(X.index[-1])[:10],
        "current_price": current_price,
        "direction": dir_pred,
        "direction_label": DIRECTION_LABELS.get(dir_pred, ""),
        "signal": SIGNAL_LABELS.get(dir_pred, ""),
        "probabilities": {
            "down": float(dir_proba[0]),
            "up": float(dir_proba[1]),
        },
        "predicted_return_pct": round(ret_pred * 100, 2),
        "predicted_price": round(current_price * (1 + ret_pred), 2),
        "unit": config["unit"],
        "cv_accuracy": metrics['classifier']['cv_accuracy'],
    }
