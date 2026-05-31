"""Model training and persistence for copper price prediction."""

import logging
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data"
MODEL_FILE = MODEL_DIR / "copper_model.pkl"


def train_models(X: pd.DataFrame, y_cls: pd.Series, y_reg: pd.Series
                 ) -> Tuple[object, object, dict]:
    """Train XGBoost classifier and regressor with walk-forward validation.

    Returns:
        clf: Trained XGBoost classifier (up/flat/down)
        reg: Trained XGBoost regressor (return %)
        metrics: dict with backtest results
    """
    from xgboost import XGBClassifier, XGBRegressor

    logger.info(f"Training on {len(X)} samples, {X.shape[1]} features")

    # Time-series split: 5 folds, each training on past, testing on future
    tscv = TimeSeriesSplit(n_splits=5)

    # ── Classifier ───────────────────────────────────────────
    clf = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        eval_metric='mlogloss',
    )

    clf_accuracies = []
    clf_predictions = []
    clf_actuals = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_cls.iloc[train_idx], y_cls.iloc[test_idx]

        if len(X_test) < 5:
            continue

        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        clf_accuracies.append(acc)
        clf_predictions.extend(preds)
        clf_actuals.extend(y_test)

    # Final fit on all data
    clf.fit(X, y_cls)

    # ── Regressor ────────────────────────────────────────────
    reg = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )

    reg_maes = []
    reg_predictions = []
    reg_actuals = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]

        if len(X_test) < 5:
            continue

        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        reg_maes.append(mae)
        reg_predictions.extend(preds)
        reg_actuals.extend(y_test)

    reg.fit(X, y_reg)

    # ── Feature importance ───────────────────────────────────
    importance = sorted(
        zip(X.columns, clf.feature_importances_),
        key=lambda x: -x[1],
    )

    # ── Baseline comparison ──────────────────────────────────
    # Baseline: always predict "up" (most common class)
    baseline_acc = max(
        (y_cls == 1).mean(),
        (y_cls == 0).mean(),
        (y_cls == -1).mean(),
    )

    metrics = {
        "classifier": {
            "cv_accuracy": float(np.mean(clf_accuracies)) if clf_accuracies else 0,
            "cv_accuracy_std": float(np.std(clf_accuracies)) if clf_accuracies else 0,
            "baseline_accuracy": float(baseline_acc),
            "improvement": float(np.mean(clf_accuracies) - baseline_acc) if clf_accuracies else 0,
        },
        "regressor": {
            "cv_mae_pct": float(np.mean(reg_maes)) if reg_maes else 0,  # MAE in return %
            "cv_mae_std": float(np.std(reg_maes)) if reg_maes else 0,
        },
        "features": importance[:10],
        "n_samples": len(X),
        "n_features": X.shape[1],
    }

    # Cross-validation fold details
    metrics["classifier"]["fold_accuracies"] = [float(a) for a in clf_accuracies]

    logger.info(f"Classifier CV accuracy: {metrics['classifier']['cv_accuracy']:.3f} "
                f"(baseline: {baseline_acc:.3f}, +{metrics['classifier']['improvement']:.3f})")
    logger.info(f"Regressor CV MAE: {metrics['regressor']['cv_mae_pct']:.4f} return units")

    return clf, reg, metrics


def save_models(clf, reg, metrics: dict) -> Path:
    """Save models and metadata to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({
            'classifier': clf,
            'regressor': reg,
            'metrics': metrics,
            'features': ALL_FEATURES if hasattr(__import__('ecoforo.predict.features', fromlist=['ALL_FEATURES']), 'ALL_FEATURES') else [],
        }, f)
    logger.info(f"Models saved to {MODEL_FILE}")
    return MODEL_FILE


def load_models() -> Tuple[object, object, dict]:
    """Load trained models from disk."""
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"No model found at {MODEL_FILE}. Train first with: ecoforo predict copper --train")
    with open(MODEL_FILE, 'rb') as f:
        data = pickle.load(f)
    return data['classifier'], data['regressor'], data['metrics']


# import for save_models
from ecoforo.predict.features import ALL_FEATURES
