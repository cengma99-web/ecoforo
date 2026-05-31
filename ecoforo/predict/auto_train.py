"""Automatic model retraining with champion/challenger comparison."""

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data"
REGISTRY_FILE = MODEL_DIR / "model_registry.json"
CHANGELOG_FILE = MODEL_DIR / "model_changelog.md"


def _load_registry() -> dict:
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text())
    return {}


def _save_registry(registry: dict):
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(registry, indent=2, default=str))


def _append_changelog(commodity: str, action: str, details: dict):
    """Append a line to the model changelog."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    CHANGELOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    cv = details.get('cv_accuracy', 0)
    baseline = details.get('baseline', 0)
    imp = details.get('improvement', 0)
    samples = details.get('n_samples', 0)
    line = (
        f"| {now} | {commodity} | {action} | "
        f"CV={cv:.1%} | baseline={baseline:.1%} | Δ={imp:+.1%} | "
        f"n={samples} |\n"
    )
    if not CHANGELOG_FILE.exists():
        CHANGELOG_FILE.write_text(
            "| Time | Commodity | Action | Metrics |\n"
            "|------|-----------|--------|----------|\n"
        )
    with open(CHANGELOG_FILE, 'a') as f:
        f.write(line)


def retrain_and_compare(commodity_key: str = "copper") -> dict:
    """Retrain model for a commodity. Replace if better than current.

    Returns dict with comparison results.
    """
    from ecoforo.predict.multi_predict import (
        build_features_for_commodity, train_commodity, COMMODITY_CONFIGS,
    )
    from ecoforo.predict.train import load_models

    config = COMMODITY_CONFIGS.get(commodity_key)
    if not config:
        return {"error": f"Unknown commodity: {commodity_key}"}

    model_path = MODEL_DIR / f"{commodity_key}_model.pkl"
    name = config['name']

    # Load current champion
    current_metrics = None
    current_cv = 0
    if model_path.exists():
        try:
            _, _, current_metrics = load_models()
            current_cv = current_metrics.get('classifier', {}).get('cv_accuracy', 0)
        except Exception as e:
            logger.warning(f"Failed to load current model: {e}")

    # Train challenger
    logger.info(f"Training challenger model for {name}...")
    try:
        X, y_cls, y_reg = build_features_for_commodity(commodity_key)
        from ecoforo.predict.train import train_models
        clf, reg, challenger_metrics = train_models(X, y_cls, y_reg)
    except Exception as e:
        logger.error(f"Challenger training failed: {e}")
        return {"error": str(e), "commodity": commodity_key}

    challenger_cv = challenger_metrics['classifier']['cv_accuracy']

    # Compare
    registry = _load_registry()
    current_entry = registry.get(commodity_key, {})

    result = {
        "commodity": commodity_key,
        "name": name,
        "current_cv": round(current_cv, 4),
        "challenger_cv": round(challenger_cv, 4),
        "current_samples": current_entry.get('n_samples', 0),
        "challenger_samples": challenger_metrics['n_samples'],
        "challenger_baseline": round(challenger_metrics['classifier']['baseline_accuracy'], 4),
        "challenger_improvement": round(challenger_metrics['classifier']['improvement'], 4),
    }

    MIN_IMPROVEMENT = 0.005  # 0.5% minimum gain to replace

    if current_cv == 0 or challenger_cv >= current_cv - MIN_IMPROVEMENT:
        # Replace: challenger is better or within epsilon
        action = "REPLACED" if current_cv > 0 else "INITIAL"

        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump({'classifier': clf, 'regressor': reg, 'metrics': challenger_metrics}, f)

        # Update registry
        registry[commodity_key] = {
            "last_trained": datetime.now(timezone.utc).isoformat(),
            "cv_accuracy": round(challenger_cv, 4),
            "baseline": round(challenger_metrics['classifier']['baseline_accuracy'], 4),
            "improvement": round(challenger_metrics['classifier']['improvement'], 4),
            "n_samples": challenger_metrics['n_samples'],
            "n_features": challenger_metrics['n_features'],
            "version": current_entry.get('version', 0) + 1,
            "previous_cv": round(current_cv, 4),
        }
        _save_registry(registry)

        result["action"] = action
        result["version"] = registry[commodity_key]["version"]
        logger.info(f"[{name}] {action}: CV {current_cv:.3f} → {challenger_cv:.3f}")
    else:
        # Keep current: challenger is worse
        action = "KEPT"
        result["action"] = action
        result["version"] = current_entry.get('version', 0)
        logger.info(f"[{name}] KEPT current: CV {current_cv:.3f} > {challenger_cv:.3f}")

    _append_changelog(commodity_key, action, result)
    return result


def retrain_all() -> list[dict]:
    """Retrain all commodity models."""
    from ecoforo.predict.multi_predict import COMMODITY_CONFIGS

    # Only retrain commodities that have data
    priority = ["copper", "aluminum", "oil", "gold", "silver"]
    results = []
    for key in priority:
        if key in COMMODITY_CONFIGS:
            try:
                r = retrain_and_compare(key)
                results.append(r)
            except Exception as e:
                logger.error(f"Retrain failed for {key}: {e}")
                results.append({"commodity": key, "error": str(e)})
    return results


def get_model_status() -> list[dict]:
    """Get status of all models."""
    registry = _load_registry()
    status = []
    for commodity, info in registry.items():
        status.append({
            "commodity": commodity,
            "version": info.get("version", 0),
            "cv_accuracy": info.get("cv_accuracy", 0),
            "last_trained": info.get("last_trained", "never"),
            "n_samples": info.get("n_samples", 0),
        })
    return sorted(status, key=lambda x: x["commodity"])
