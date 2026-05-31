"""Daily economic brief — automated Markdown report."""

import logging
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

REPORT_DIR = Path(__file__).resolve().parent.parent / "data" / "reports"


def _fetch_one(title_like: str, n: int = 1):
    from ecoforo.db.session import SessionLocal
    from ecoforo.db.models import EconomicEvent
    db = SessionLocal()
    try:
        return (
            db.query(EconomicEvent)
            .filter(EconomicEvent.title.ilike(f"%{title_like}%"),
                    EconomicEvent.actual_value.isnot(None))
            .order_by(EconomicEvent.event_date.desc()).limit(n).all()
        )
    finally:
        db.close()


def _copper_section() -> str:
    """Copper price and technical analysis."""
    from ecoforo.db.session import SessionLocal
    from ecoforo.db.models import EconomicEvent

    db = SessionLocal()
    try:
        rows = (
            db.query(EconomicEvent)
            .filter(EconomicEvent.title == "Copper Futures — Close",
                    EconomicEvent.actual_value.isnot(None))
            .order_by(EconomicEvent.event_date.desc()).limit(120).all()
        )
        if not rows:
            return "> No copper data available.\n"

        prices = pd.Series([e.actual_value for e in reversed(rows)])
        latest = prices.iloc[-1]
        ma20 = prices.rolling(20).mean().iloc[-1]
        ma60 = prices.rolling(60).mean().iloc[-1] if len(prices) >= 60 else prices.mean()

        # RSI
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = (100 - (100 / (1 + rs))).iloc[-1]

        # MACD
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = (ema12 - ema26).iloc[-1]
        macd_sig = (ema12 - ema26).ewm(span=9).mean().iloc[-1]

        # 7-day change
        change_7d = (latest - prices.iloc[-8]) / prices.iloc[-8] * 100 if len(prices) >= 8 else 0

        trend = "▲ 多头" if latest > ma20 else "▼ 空头"
        rsi_status = "超买 ⚠️" if rsi > 70 else ("超卖 💡" if rsi < 30 else "中性")
        macd_status = "金叉 ▲" if macd > macd_sig else "死叉 ▼"

        return f"""## 🔧 铜价

| 指标 | 值 | 信号 |
|------|-----|------|
| COMEX 铜 | **${latest:.2f}/lb** | {trend} |
| 7日涨跌 | {change_7d:+.1f}% | |
| MA20 | ${ma20:.2f} | {'站上' if latest > ma20 else '跌破'}均线 |
| MA60 | ${ma60:.2f} | {'站上' if latest > ma60 else '跌破'}均线 |
| RSI(14) | {rsi:.0f} | {rsi_status} |
| MACD | {macd:.3f} | {macd_status} |

"""
    finally:
        db.close()


def _prediction_section() -> str:
    """Model prediction signal."""
    try:
        from ecoforo.predict.multi_predict import predict_commodity
        pred = predict_commodity("copper")
        return f"""## 🔮 30日预测

| 信号 | 方向 | 预测涨幅 | 目标价 | 概率 |
|------|------|----------|--------|------|
| {pred['signal']} | {pred['direction_label']} | {pred['predicted_return_pct']:+.1f}% | ${pred['predicted_price']:.2f}/lb | 涨 {pred['probabilities']['up']:.0%} / 跌 {pred['probabilities']['down']:.0%} |

> 模型 CV 准确率: {pred['cv_accuracy']:.0%}，基于 34 特征 XGB+LGB 集成

"""
    except Exception as e:
        return f"> 预测模型暂不可用: {e}\n\n"


def _events_section() -> str:
    """Recent high-impact events."""
    from ecoforo.db.session import SessionLocal
    from ecoforo.db.models import EconomicEvent, EventSource

    db = SessionLocal()
    try:
        # High-impact GDELT news (last 24h)
        gdelt = db.query(EventSource).filter(EventSource.name == 'gdelt').first()
        lines = ["## 📰 24小时要闻\n\n"]
        if gdelt:
            news = (
                db.query(EconomicEvent)
                .filter(
                    EconomicEvent.source_id == gdelt.id,
                    EconomicEvent.event_date >= datetime.now(timezone.utc) - timedelta(hours=24),
                )
                .order_by(EconomicEvent.importance.desc(), EconomicEvent.event_date.desc())
                .limit(8).all()
            )
            if news:
                for e in news:
                    cat = (e.raw_data or {}).get('category', '')
                    sentiment = (e.raw_data or {}).get('sentiment', 0)
                    sent_emoji = "🟢" if sentiment > 0 else ("🔴" if sentiment < 0 else "⚪")
                    cat_tag = f"`{cat}`" if cat else ""
                    lines.append(f"- {sent_emoji} {cat_tag} {e.title[:80]}\n")
            else:
                lines.append("> 近 24 小时无高影响事件\n")
        else:
            lines.append("> GDELT 数据源未注册\n")
        return "".join(lines) + "\n"
    finally:
        db.close()


def _macro_section() -> str:
    """China + US macro snapshot."""
    indicators_cn = [
        ("中国制造业PMI", "制造业 PMI", "index", 50),
        ("中国CPI年率", "CPI 年率", "%", 2),
        ("中国M2同比增速", "M2 增速", "%", 8),
        ("中国GDP年率", "GDP 年率", "%", 5),
    ]
    indicators_us = [
        ("US Federal Funds Effective Rate", "联邦利率", "%", 3),
        ("US 10-Year Treasury Yield", "10Y 国债", "%", 4),
        ("US Unemployment Rate", "失业率", "%", 4),
    ]

    out = "## 🌐 宏观速览\n\n"
    out += "### 🇨🇳 中国\n\n| 指标 | 最新值 | 日期 |\n|------|--------|------|\n"
    for name, label, unit, _ in indicators_cn:
        rows = _fetch_one(name, 1)
        if rows:
            out += f"| {label} | {rows[0].actual_value:.1f}{unit} | {str(rows[0].event_date)[:10]} |\n"

    out += "\n### 🇺🇸 美国\n\n| 指标 | 最新值 | 日期 |\n|------|--------|------|\n"
    for name, label, unit, _ in indicators_us:
        rows = _fetch_one(name, 1)
        if rows:
            out += f"| {label} | {rows[0].actual_value:.2f}{unit} | {str(rows[0].event_date)[:10]} |\n"

    return out + "\n"


def _alerts_section() -> str:
    """Risk alerts."""
    alerts = []
    from ecoforo.db.session import SessionLocal
    from ecoforo.db.models import EconomicEvent, EventSource

    db = SessionLocal()
    try:
        # Check copper RSI extreme
        cu = _fetch_one("Copper Futures — Close", 120)
        if cu and len(cu) >= 14:
            prices = pd.Series([e.actual_value for e in reversed(cu)])
            delta = prices.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = (100 - (100 / (1 + rs))).iloc[-1]
            if rsi > 70:
                alerts.append(f"⚠️ 铜 RSI={rsi:.0f}，超买区域，警惕回调")
            elif rsi < 30:
                alerts.append(f"💡 铜 RSI={rsi:.0f}，超卖区域，反弹可能")

        # High-impact events in last 24h
        gdelt = db.query(EventSource).filter(EventSource.name == 'gdelt').first()
        if gdelt:
            high = (
                db.query(EconomicEvent)
                .filter(
                    EconomicEvent.source_id == gdelt.id,
                    EconomicEvent.event_date >= datetime.now(timezone.utc) - timedelta(hours=24),
                    EconomicEvent.impact == 'high',
                ).count()
            )
            if high >= 5:
                alerts.append(f"🔴 近 24h 有 {high} 条高影响事件，市场波动风险上升")

        # China PMI warning
        pmi = _fetch_one("制造业PMI", 1)
        if pmi and pmi[0].actual_value and pmi[0].actual_value < 50:
            alerts.append(f"⚠️ 中国制造业 PMI {pmi[0].actual_value:.1f}，低于荣枯线")

    finally:
        db.close()

    if not alerts:
        alerts.append("✅ 无特别预警")

    return "## 🚨 风险预警\n\n" + "\n".join(f"- {a}" for a in alerts) + "\n"


def generate_daily_report() -> str:
    """Generate a complete daily economic brief in Markdown."""
    today = datetime.now().strftime("%Y-%m-%d")
    sections = [
        f"# 📊 ecoforo 经济早报 — {today}\n",
        "> 自动生成 · 数据管道 5 源 · XGB+LGB 预测\n",
        "---\n",
        _copper_section(),
        _prediction_section(),
        _events_section(),
        _macro_section(),
        _alerts_section(),
        "---\n",
        f"*报告生成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · [GitHub](https://github.com/cengma99-web/ecoforo)*\n",
    ]
    return "\n".join(sections)


def save_report(content: str) -> Path:
    """Save report to disk and return path."""
    today = datetime.now().strftime("%Y-%m-%d")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{today}-report.md"
    path.write_text(content, encoding="utf-8")
    logger.info(f"Report saved to {path}")
    return path
