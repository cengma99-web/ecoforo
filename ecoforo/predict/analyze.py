"""Economic data analysis — copper trends, correlations, macro overview."""

import logging
import warnings
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import func

from ecoforo.db.session import SessionLocal
from ecoforo.db.models import EconomicEvent, EventSource

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def _fetch_df(title: str, start_date=None):
    db = SessionLocal()
    try:
        q = db.query(EconomicEvent).filter(
            EconomicEvent.title == title,
            EconomicEvent.actual_value.isnot(None),
        )
        if start_date:
            q = q.filter(EconomicEvent.event_date >= start_date)
        rows = q.order_by(EconomicEvent.event_date.asc()).all()
        return pd.DataFrame([{
            'date': pd.Timestamp(e.event_date.date()),
            'value': e.actual_value,
        } for e in rows])
    finally:
        db.close()


def _fetch_latest(name_like: str, n=4):
    db = SessionLocal()
    try:
        rows = (
            db.query(EconomicEvent)
            .filter(EconomicEvent.title.ilike(f"%{name_like}%"),
                    EconomicEvent.actual_value.isnot(None))
            .order_by(EconomicEvent.event_date.desc())
            .limit(n).all()
        )
        return rows
    finally:
        db.close()


def analyze_copper(months: int = 12) -> list[str]:
    """Copper price technical analysis."""
    out = []
    out.append("═" * 60)
    out.append("🔧 铜价技术分析 (COMEX)")
    out.append("═" * 60)

    start = datetime.now(timezone.utc) - timedelta(days=months * 31)
    df = _fetch_df("Copper Futures — Close", start)
    if df.empty:
        out.append("  无铜价数据")
        return out

    df = df.dropna().drop_duplicates('date', keep='last').set_index('date')
    series = df['value']

    latest = series.iloc[-1]
    ma20 = series.rolling(20).mean().iloc[-1]
    ma60 = series.rolling(60).mean().iloc[-1] if len(series) >= 60 else series.mean()
    ma120 = series.rolling(120).mean().iloc[-1] if len(series) >= 120 else series.mean()

    out.append(f"  最新:  ${latest:.2f}/lb")
    out.append(f"  MA20:  ${ma20:.2f}/lb  {'▲ 站上均线' if latest > ma20 else '▼ 跌破'}")
    out.append(f"  MA60:  ${ma60:.2f}/lb  {'▲ 站上均线' if latest > ma60 else '▼ 跌破'}")
    out.append(f"  MA120: ${ma120:.2f}/lb  {'▲ 站上均线' if latest > ma120 else '▼ 跌破'}")

    recent = series[-60:]
    out.append(f"  60日高: ${recent.max():.2f}  60日低: ${recent.min():.2f}")
    out.append(f"  30日波动率: {recent[-30:].pct_change().std()*100:.1f}%")

    # RSI
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_now = rsi.iloc[-1]
    if not np.isnan(rsi_now):
        status = "⚠️ 超买" if rsi_now > 70 else ("💡 超卖" if rsi_now < 30 else "✓ 中性")
        out.append(f"  RSI(14): {rsi_now:.0f} [{status}]")

    # MACD
    ema12 = series.ewm(span=12).mean()
    ema26 = series.ewm(span=26).mean()
    macd = (ema12 - ema26).iloc[-1]
    macd_signal = (ema12 - ema26).ewm(span=9).mean().iloc[-1]
    macd_state = "▲ 金叉" if macd > macd_signal else "▼ 死叉"
    out.append(f"  MACD: {macd:.3f} vs Signal {macd_signal:.3f} [{macd_state}]")

    # Monthly averages
    out.append("")
    out.append("  月度均价:")
    df['month'] = df.index.to_period('M')
    monthly = df.groupby('month')['value'].agg(['mean', 'min', 'max']).round(2)
    max_val = monthly['mean'].max()
    min_val = monthly['mean'].min()
    for m, row in monthly.tail(6).iterrows():
        n = max(1, int((row['mean'] - min_val) / (max_val - min_val) * 20))
        bar = "█" * n
        out.append(f"    {str(m):<8s} ${row['mean']:<6.2f} {bar}")

    return out


def analyze_correlations() -> list[str]:
    """Commodity correlations with copper."""
    out = []
    out.append("")
    out.append("═" * 60)
    out.append("🔗 大宗商品与铜价相关性（2024-2026 月度）")
    out.append("═" * 60)

    commodities = {
        "Aluminum Futures — Close": "铝",
        "Crude Oil WTI Futures — Close": "原油",
        "Gold Futures — Close": "黄金",
        "Silver Futures — Close": "白银",
        "Zinc Futures — Close": "锌",
        "Natural Gas Futures — Close": "天然气",
    }

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    cu_df = _fetch_df("Copper Futures — Close", start)
    if cu_df.empty:
        return out
    cu_df = cu_df.dropna().drop_duplicates('date', keep='last').set_index('date')
    cu = cu_df['value'].resample('ME').last().dropna()
    cu_ret = cu.pct_change().dropna()

    out.append(f"{'商品':<6s} {'相关':>8s}  走势")
    out.append("-" * 45)

    for title, label in commodities.items():
        s_df = _fetch_df(title, start)
        if s_df.empty:
            continue
        s_df = s_df.dropna().drop_duplicates('date', keep='last').set_index('date')
        s = s_df['value'].resample('ME').last().dropna()
        s_ret = s.pct_change().dropna()
        common = cu_ret.index.intersection(s_ret.index)
        if len(common) < 3:
            continue
        corr = cu_ret.loc[common].corr(s_ret.loc[common])
        if np.isnan(corr):
            continue
        direction = "同向" if corr > 0 else "反向"
        n = max(1, int(abs(corr) * 25))
        bar = "█" * n + "░" * (25 - n)
        out.append(f"  {label:<6s} {corr:>+7.3f}  {bar} {direction}")

    return out


def analyze_china_macro() -> list[str]:
    """China macro overview."""
    out = []
    out.append("")
    out.append("═" * 60)
    out.append("🇨🇳 中国经济核心指标")
    out.append("═" * 60)

    indicators = [
        ("中国CPI年率", "percent", "CPI"),
        ("中国PPI年率", "percent", "PPI"),
        ("中国制造业PMI", "index", "PMI"),
        ("中国M2同比增速", "percent", "M2"),
        ("中国GDP年率", "percent", "GDP"),
        ("中国贸易差额（美元）", "亿美元", "贸易差额"),
        ("中国国房景气指数", "index", "房地产"),
    ]

    for name, unit, label in indicators:
        rows = _fetch_latest(name, n=4)
        if not rows:
            continue
        vals = [e.actual_value for e in rows if e.actual_value is not None]
        if not vals:
            continue
        trend = ""
        if len(vals) >= 2:
            d = vals[0] - vals[1]
            trend = f" {'↑' if d>0 else '↓' if d<0 else '→'}"
        out.append(f"  {label:<8s}: {vals[0]:>8.1f}{trend}  ({str(rows[0].event_date)[:10]})")

    return out


def analyze_us_macro() -> list[str]:
    """US macro overview."""
    out = []
    out.append("")
    out.append("═" * 60)
    out.append("🇺🇸 美国经济核心指标")
    out.append("═" * 60)

    indicators = [
        ("US Federal Funds Effective Rate", "联邦利率", "percent"),
        ("US 10-Year Treasury Yield", "10Y国债", "percent"),
        ("US CPI (All Urban Consumers, All Items)", "CPI指数", "index"),
        ("US Unemployment Rate", "失业率", "percent"),
    ]

    for name, label, unit in indicators:
        rows = _fetch_latest(name, n=2)
        if not rows:
            continue
        val = rows[0].actual_value
        prev = rows[1].actual_value if len(rows) > 1 else None
        trend = ""
        if prev and val:
            d = val - prev
            trend = f" {'↑' if d>0 else '↓' if d<0 else '→'}"
        out.append(f"  {label:<8s}: {val:>8.2f}{trend}  ({str(rows[0].event_date)[:10]})")

    return out


def analyze_news(days: int = 30) -> list[str]:
    """Recent high-impact news from GDELT."""
    out = []
    out.append("")
    out.append("═" * 60)
    out.append(f"📰 近{days}日高影响经济新闻 (GDELT)")
    out.append("═" * 60)

    db = SessionLocal()
    try:
        gdelt = db.query(EventSource).filter(EventSource.name == 'gdelt').first()
        if not gdelt:
            return out

        news = (
            db.query(EconomicEvent)
            .filter(
                EconomicEvent.source_id == gdelt.id,
                EconomicEvent.event_date >= datetime.now(timezone.utc) - timedelta(days=days),
                EconomicEvent.impact == 'high',
            )
            .order_by(EconomicEvent.event_date.desc())
            .limit(10)
            .all()
        )

        if not news:
            out.append("  无高影响新闻")
            return out

        for e in news:
            out.append(f"  {str(e.event_date)[:10]} | {e.title[:65]}")
    finally:
        db.close()

    return out


def analyze_market_regime() -> list[str]:
    """Determine current market regime based on multiple indicators."""
    out = []
    out.append("")
    out.append("═" * 60)
    out.append("🎯 市场状态评估")
    out.append("═" * 60)

    db = SessionLocal()
    try:
        # Copper trend
        cu = (
            db.query(EconomicEvent)
            .filter(EconomicEvent.title == "Copper Futures — Close",
                    EconomicEvent.actual_value.isnot(None))
            .order_by(EconomicEvent.event_date.desc()).limit(30).all()
        )
        if cu:
            prices = [e.actual_value for e in cu]
            trend_30 = (prices[0] - prices[-1]) / prices[-1] if prices[-1] else 0
            out.append(f"  铜价 30 日趋势: {'📈' if trend_30 > 0 else '📉'} {trend_30*100:+.1f}%")

        # China PMI trend
        pmi = _fetch_latest("制造业PMI", n=3)
        if pmi and len(pmi) >= 2:
            pmi_now = pmi[0].actual_value
            pmi_prev = pmi[1].actual_value
            if pmi_now and pmi_prev:
                direction = "扩张↑" if pmi_now > pmi_prev else ("收缩↓" if pmi_now < pmi_prev else "持平→")
                zone = "荣枯线上" if pmi_now >= 50 else "荣枯线下⚠️"
                out.append(f"  中国 PMI: {pmi_now:.1f} ({zone}, {direction})")

        # US rate direction
        rate = _fetch_latest("Federal Funds Effective Rate", n=2)
        if rate and len(rate) >= 2:
            r_now, r_prev = rate[0].actual_value, rate[1].actual_value
            if r_now and r_prev:
                rate_state = "加息周期" if r_now > r_prev else ("降息周期" if r_now < r_prev else "暂停")
                out.append(f"  美国利率: {r_now:.2f}% ({rate_state})")

    finally:
        db.close()

    return out


def run_full_analysis() -> str:
    """Run full economic analysis and return formatted text."""
    sections = []
    sections.extend(analyze_copper())
    sections.extend(analyze_correlations())
    sections.extend(analyze_china_macro())
    sections.extend(analyze_us_macro())
    sections.extend(analyze_market_regime())
    sections.extend(analyze_news())
    return "\n".join(sections)
