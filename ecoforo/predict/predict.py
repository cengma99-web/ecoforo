"""Prediction and backtesting for copper price."""

import logging
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Class labels: 0=down, 1=flat, 2=up
DIRECTION_MAP = {2: "📈 上涨 (UP)", 1: "➡️ 持平 (FLAT)", 0: "📉 下跌 (DOWN)"}
SIGNAL_MAP = {2: "🟢 买入", 1: "🟡 等待", 0: "🔴 卖出"}


def predict_latest() -> dict:
    """Predict copper direction and price for the next 30 days."""
    from ecoforo.predict.features import build_features, ALL_FEATURES
    from ecoforo.predict.train import load_models

    X, y_cls, y_reg = build_features()

    clf, reg, metrics = load_models()

    # Get latest row
    feature_cols = [c for c in ALL_FEATURES if c in X.columns]
    latest = X[feature_cols].iloc[-1:]
    latest_date = X.index[-1]

    # Predict (classes: 0=down, 1=flat, 2=up)
    dir_pred = int(clf.predict(latest)[0])
    dir_proba = clf.predict_proba(latest)[0]
    ret_pred = float(reg.predict(latest)[0])

    # Current price
    copper_price = float(X['lag_1'].iloc[-1]) if 'lag_1' in X.columns else None

    # Predicted price
    predicted_price = copper_price * (1 + ret_pred) if copper_price else None

    return {
        "date": str(latest_date)[:10],
        "current_price": copper_price,
        "direction": dir_pred,
        "direction_label": DIRECTION_MAP.get(dir_pred, "UNKNOWN"),
        "signal": SIGNAL_MAP.get(dir_pred, "❓"),
        "probabilities": {
            "down": float(dir_proba[0]),
            "flat": float(dir_proba[1]),
            "up": float(dir_proba[2]),
        },
        "predicted_return_pct": round(ret_pred * 100, 2),
        "predicted_price": round(predicted_price, 2) if predicted_price else None,
        "model_metrics": metrics,
    }


def run_backtest() -> dict:
    """Run full backtest and return detailed metrics."""
    from ecoforo.predict.features import build_features, ALL_FEATURES
    from ecoforo.predict.train import load_models

    X, y_cls, y_reg = build_features()

    clf, reg, metrics = load_models()

    feature_cols = [c for c in ALL_FEATURES if c in X.columns]

    # Predict on all data
    dir_preds = clf.predict(X[feature_cols])
    ret_preds = reg.predict(X[feature_cols])

    results = pd.DataFrame({
        'date': X.index,
        'actual_direction': y_cls.values,
        'predicted_direction': dir_preds,
        'actual_return': y_reg.values,
        'predicted_return': ret_preds,
    })

    # Direction accuracy
    correct = (results['actual_direction'] == results['predicted_direction']).sum()
    total = len(results)
    accuracy = correct / total if total > 0 else 0

    # Per-class accuracy (0=down, 1=flat, 2=up)
    per_class = {}
    for label, name in [(0, 'down'), (1, 'flat'), (2, 'up')]:
        mask = results['actual_direction'] == label
        if mask.sum() > 0:
            per_class[name] = float((results.loc[mask, 'actual_direction'] == results.loc[mask, 'predicted_direction']).mean())

    # Regression MAE
    mae = float(np.mean(np.abs(results['actual_return'] - results['predicted_return'])))

    # Recent performance (last 90 days)
    recent = results.tail(90)
    recent_acc = float((recent['actual_direction'] == recent['predicted_direction']).mean()) if len(recent) > 0 else 0

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_cls, dir_preds, labels=[0, 1, 2])

    return {
        "total_samples": total,
        "direction_accuracy": round(accuracy, 3),
        "recent_90d_accuracy": round(recent_acc, 3),
        "per_class_accuracy": per_class,
        "regression_mae_pct": round(mae * 100, 2),
        "confusion_matrix": cm.tolist(),
        "cv_metrics": metrics,
    }


def format_prediction(result: dict) -> str:
    """Format prediction result for CLI output."""
    lines = []
    lines.append("═" * 55)
    lines.append("🔮 铜价 30 日预测")
    lines.append("═" * 55)
    lines.append(f"日期: {result['date']}")
    lines.append(f"当前价格: ${result['current_price']:.2f}/lb" if result['current_price'] else "")
    lines.append(f"")
    lines.append(f"信号: {result['signal']} {result['direction_label']}")
    lines.append(f"预测 30 日涨跌幅: {result['predicted_return_pct']:+.1f}%")
    if result['predicted_price']:
        lines.append(f"预测价格: ${result['predicted_price']:.2f}/lb")
    lines.append(f"")
    probs = result.get('probabilities', {})
    lines.append(f"概率分布: 涨 {probs.get('up', 0):.0%}  |  平 {probs.get('flat', 0):.0%}  |  跌 {probs.get('down', 0):.0%}")
    lines.append("")
    m = result.get('model_metrics', {}).get('classifier', {})
    lines.append(f"模型 CV 准确率: {m.get('cv_accuracy', 0):.1%}  (基线: {m.get('baseline_accuracy', 0):.1%})")
    lines.append("═" * 55)
    return "\n".join(lines)


def format_backtest(result: dict) -> str:
    """Format backtest results for CLI output."""
    lines = []
    lines.append("═" * 55)
    lines.append("📊 回测报告")
    lines.append("═" * 55)
    lines.append(f"样本: {result['total_samples']}")
    lines.append(f"方向准确率: {result['direction_accuracy']:.1%}")
    lines.append(f"近 90 日准确率: {result['recent_90d_accuracy']:.1%}")
    lines.append(f"回归 MAE: {result['regression_mae_pct']:.2f}%")
    lines.append("")
    per_class = result.get('per_class_accuracy', {})
    lines.append(f"分类准确率: 涨 {per_class.get('up', 0):.1%}  |  平 {per_class.get('flat', 0):.1%}  |  跌 {per_class.get('down', 0):.1%}")
    lines.append("")
    cm = result.get('confusion_matrix', [])
    if cm:
        lines.append("混淆矩阵 (行=实际, 列=预测):")
        lines.append(f"        跌   平   涨")
        labels = ['跌', '平', '涨']
        for i, label in enumerate(labels):
            if i < len(cm):
                lines.append(f"  {label}  {cm[i][0]:>4d} {cm[i][1]:>4d} {cm[i][2]:>4d}")
    lines.append("")
    cv = result.get('cv_metrics', {}).get('classifier', {})
    lines.append(f"CV 准确率: {cv.get('cv_accuracy', 0):.1%} ± {cv.get('cv_accuracy_std', 0):.1%}")
    lines.append(f"顶级特征: {', '.join(f[0] for f in result.get('cv_metrics', {}).get('features', [])[:5])}")
    lines.append("═" * 55)
    return "\n".join(lines)
