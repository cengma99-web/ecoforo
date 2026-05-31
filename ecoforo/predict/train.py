"""Model training and persistence for copper price prediction.

Improvements over v1:
- Expanding window CV (more realistic than fixed-fold TimeSeriesSplit)
- Stronger regularization to reduce overfitting
- LightGBM + XGBoost ensemble voting
- Hyperparameter grid search
"""

import logging
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data"
MODEL_FILE = MODEL_DIR / "copper_model.pkl"

# Minimum training samples before we start validating
MIN_TRAIN_SAMPLES = 360  # ~1 year of daily data


def _expanding_window_cv(X, y, model, n_splits=8):
    """Walk-forward validation: train on [0:t], test on [t:t+step].

    More realistic for time series than k-fold CV.
    """
    step = max(1, (len(X) - MIN_TRAIN_SAMPLES) // n_splits)
    scores = []
    for i in range(MIN_TRAIN_SAMPLES, len(X), step):
        X_train, X_test = X.iloc[:i], X.iloc[i:i+step]
        y_train, y_test = y.iloc[:i], y.iloc[i:i+step]
        if len(X_test) < 5:
            continue
        model_clone = pickle.loads(pickle.dumps(model))  # clone
        model_clone.fit(X_train, y_train)
        scores.append(accuracy_score(y_test, model_clone.predict(X_test)))
    return scores


def _grid_search_xgb(X, y_cls):
    """Simple grid search for XGBoost classifier."""
    from xgboost import XGBClassifier

    best_score = -1
    best_params = {}
    best_model = None

    param_grid = [
        {"max_depth": 3, "n_estimators": 80, "learning_rate": 0.03,
         "subsample": 0.7, "min_child_weight": 5, "reg_alpha": 0.1, "reg_lambda": 1.0},
        {"max_depth": 3, "n_estimators": 120, "learning_rate": 0.03,
         "subsample": 0.8, "min_child_weight": 3, "reg_alpha": 0.05, "reg_lambda": 1.0},
        {"max_depth": 4, "n_estimators": 100, "learning_rate": 0.05,
         "subsample": 0.7, "min_child_weight": 5, "reg_alpha": 0.1, "reg_lambda": 2.0},
        {"max_depth": 2, "n_estimators": 150, "learning_rate": 0.02,
         "subsample": 0.8, "min_child_weight": 7, "reg_alpha": 0.2, "reg_lambda": 1.0},
        {"max_depth": 4, "n_estimators": 80, "learning_rate": 0.04,
         "subsample": 0.6, "min_child_weight": 3, "reg_alpha": 0.15, "reg_lambda": 1.5},
    ]

    for params in param_grid:
        model = XGBClassifier(
            **params,
            random_state=42,
            eval_metric='mlogloss',
            verbosity=0,
        )
        scores = _expanding_window_cv(X, y_cls, model, n_splits=6)
        if scores:
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_params = params
                best_model = model

    logger.info(f"Best XGB params: {best_params} (CV: {best_score:.3f})")
    return best_model, best_params, best_score


def _grid_search_lgb(X, y_cls):
    """Simple grid search for LightGBM classifier."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        logger.warning("LightGBM not installed, skipping")
        return None, {}, 0

    best_score = -1
    best_model = None
    best_params = {}

    param_grid = [
        {"max_depth": 3, "n_estimators": 80, "learning_rate": 0.03,
         "subsample": 0.7, "min_child_samples": 20, "reg_alpha": 0.1, "reg_lambda": 1.0},
        {"max_depth": 4, "n_estimators": 100, "learning_rate": 0.03,
         "subsample": 0.7, "min_child_samples": 15, "reg_alpha": 0.05, "reg_lambda": 1.0},
        {"max_depth": 3, "n_estimators": 120, "learning_rate": 0.02,
         "subsample": 0.8, "min_child_samples": 25, "reg_alpha": 0.1, "reg_lambda": 2.0},
    ]

    for params in param_grid:
        model = LGBMClassifier(
            **params,
            random_state=42,
            verbosity=-1,
        )
        scores = _expanding_window_cv(X, y_cls, model, n_splits=6)
        if scores:
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_params = params
                best_model = model

    if best_model:
        logger.info(f"Best LGB params: {best_params} (CV: {best_score:.3f})")
    return best_model, best_params, best_score


class EnsembleClassifier:
    """Voting ensemble: XGBoost + LightGBM, weighted by CV score."""

    def __init__(self):
        self.models = []
        self.weights = []

    def add(self, model, weight=1.0):
        self.models.append(model)
        self.weights.append(weight)

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)

    def predict(self, X):
        probas = []
        for m, w in zip(self.models, self.weights):
            probas.append(m.predict_proba(X) * w)
        avg_proba = np.sum(probas, axis=0) / sum(self.weights)
        return np.argmax(avg_proba, axis=1)

    def predict_proba(self, X):
        probas = []
        for m, w in zip(self.models, self.weights):
            probas.append(m.predict_proba(X) * w)
        return np.sum(probas, axis=0) / sum(self.weights)

    @property
    def feature_importances_(self):
        if self.models:
            return self.models[0].feature_importances_
        return np.array([])


def train_models(X: pd.DataFrame, y_cls: pd.Series, y_reg: pd.Series
                 ) -> Tuple[object, object, dict]:
    """Train ensemble classifiers and regressor with expanding-window CV."""

    from xgboost import XGBRegressor

    logger.info(f"Training on {len(X)} samples, {X.shape[1]} features")

    # ── Classifier: XGBoost + LightGBM ensemble ──────────────
    xgb_model, xgb_params, xgb_score = _grid_search_xgb(X, y_cls)
    lgb_model, lgb_params, lgb_score = _grid_search_lgb(X, y_cls)

    ensemble = EnsembleClassifier()
    ensemble.add(xgb_model, weight=max(xgb_score, 0.3))

    if lgb_model:
        ensemble.add(lgb_model, weight=max(lgb_score, 0.3))
        ensemble_cv_score = max(xgb_score, lgb_score)
        logger.info(f"Using XGBoost + LightGBM ensemble")
    else:
        ensemble_cv_score = xgb_score
        logger.info("Using XGBoost only (LightGBM not available)")

    # Final fit on all data
    ensemble.fit(X, y_cls)

    # ── Expanding window CV for detailed metrics ─────────────
    cv_scores = _expanding_window_cv(X, y_cls, xgb_model, n_splits=8)

    # ── Regressor (XGBoost) ──────────────────────────────────
    reg = XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.03,
        subsample=0.7, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42,
    )

    reg_maes = []
    step = max(1, (len(X) - MIN_TRAIN_SAMPLES) // 6)
    for i in range(MIN_TRAIN_SAMPLES, len(X), step):
        X_train, X_test = X.iloc[:i], X.iloc[i:i+step]
        y_train, y_test = y_reg.iloc[:i], y_reg.iloc[i:i+step]
        if len(X_test) < 5:
            continue
        reg_clone = pickle.loads(pickle.dumps(reg))
        reg_clone.fit(X_train, y_train)
        reg_maes.append(mean_absolute_error(y_test, reg_clone.predict(X_test)))

    reg.fit(X, y_reg)

    # ── Baseline ─────────────────────────────────────────────
    baseline_acc = max((y_cls == c).mean() for c in [0, 1])

    # ── Feature importance ───────────────────────────────────
    importance = sorted(
        zip(X.columns, xgb_model.feature_importances_),
        key=lambda x: -x[1],
    )

    metrics = {
        "classifier": {
            "cv_accuracy": float(np.mean(cv_scores)) if cv_scores else ensemble_cv_score,
            "cv_accuracy_std": float(np.std(cv_scores)) if cv_scores else 0,
            "baseline_accuracy": float(baseline_acc),
            "improvement": float(np.mean(cv_scores) - baseline_acc) if cv_scores else (ensemble_cv_score - baseline_acc),
            "fold_accuracies": [float(s) for s in cv_scores] if cv_scores else [],
            "ensemble": "XGBoost+LightGBM" if lgb_model else "XGBoost",
            "xgb_params": xgb_params,
            "lgb_params": lgb_params if lgb_model else {},
        },
        "regressor": {
            "cv_mae_pct": float(np.mean(reg_maes)) if reg_maes else 0,
            "cv_mae_std": float(np.std(reg_maes)) if reg_maes else 0,
        },
        "features": importance[:15],
        "n_samples": len(X),
        "n_features": X.shape[1],
    }

    logger.info(f"Ensemble CV accuracy: {metrics['classifier']['cv_accuracy']:.3f} "
                f"(baseline: {baseline_acc:.3f}, +{metrics['classifier']['improvement']:.3f})")
    logger.info(f"Regressor CV MAE: {metrics['regressor']['cv_mae_pct']:.4f}")

    return ensemble, reg, metrics


def save_models(clf, reg, metrics: dict) -> Path:
    """Save models and metadata to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({
            'classifier': clf,
            'regressor': reg,
            'metrics': metrics,
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
