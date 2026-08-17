"""Model artifact readiness and metadata access."""

import json
from pathlib import Path
from typing import Any

from src.services.exceptions import ModelNotReadyError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REQUIRED_ARTIFACTS = (
    ARTIFACTS_DIR / "schema.json",
    ARTIFACTS_DIR / "preprocessor.pkl",
    ARTIFACTS_DIR / "encoder.pkl",
    ARTIFACTS_DIR / "model.pkl",
)


def load_metadata() -> dict[str, Any]:
    """Load raw training metadata without making health checks depend on it."""
    defaults = {"training_date": "unknown", "model_name": "churn_predictor"}
    try:
        with (ARTIFACTS_DIR / "metadata.json").open(encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return defaults


def prediction_metadata() -> dict[str, str]:
    metadata = load_metadata()
    return {
        "model_name": metadata.get("model_name", "churn_predictor"),
        "model_version": metadata.get("version", "1.0.0"),
    }


def artifacts_ready() -> bool:
    return all(path.exists() for path in REQUIRED_ARTIFACTS)


def ensure_artifacts_ready() -> None:
    if not artifacts_ready():
        raise ModelNotReadyError()
