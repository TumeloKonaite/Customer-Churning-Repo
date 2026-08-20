"""Model artifact readiness and metadata access."""

import json
import os
from pathlib import Path
from typing import Any

from src.services.exceptions import ModelNotReadyError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REQUIRED_ARTIFACTS = (
    ARTIFACTS_DIR / "schema.json",
    ARTIFACTS_DIR / "model.pkl",
)


def deployment_package_dir() -> Path:
    return Path(os.getenv("DEPLOYMENT_PACKAGE_DIR", PROJECT_ROOT / "build" / "model"))


def deployment_artifacts_ready() -> bool:
    package = deployment_package_dir()
    return all(
        path.exists()
        for path in (
            package / "deployment_metadata.json",
            package / "feature_schema.json",
            package / "model" / "MLmodel",
        )
    )


def load_metadata() -> dict[str, Any]:
    """Load raw training metadata without making health checks depend on it."""
    if deployment_artifacts_ready():
        try:
            with (deployment_package_dir() / "deployment_metadata.json").open(
                encoding="utf-8"
            ) as file:
                return json.load(file)
        except (json.JSONDecodeError, OSError):
            return {}
    defaults = {"model_name": "churn_predictor"}
    try:
        with (ARTIFACTS_DIR / "metadata.json").open(encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return defaults


def prediction_metadata() -> dict[str, str]:
    metadata = load_metadata()
    result = {"model_name": metadata.get("model_name", "churn_predictor")}
    version = metadata.get("model_version", metadata.get("version"))
    if version is not None:
        result["model_version"] = str(version)
    for field in ("deployment_id", "model_version_id", "mlflow_run_id"):
        if metadata.get(field):
            result[field] = str(metadata[field])
    return result


def artifacts_ready() -> bool:
    if deployment_artifacts_ready():
        return True
    if os.getenv("APP_ENV", "development") == "production":
        return False
    return all(path.exists() for path in REQUIRED_ARTIFACTS)


def ensure_artifacts_ready() -> None:
    if not artifacts_ready():
        raise ModelNotReadyError()
