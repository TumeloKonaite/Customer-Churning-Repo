"""Optional MLflow tracking for completed local training runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
import math
import os
import platform
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from src.components.model_trainer import ModelTrainingResult
from src.config import DagsHubSettings, safe_error_message
from src.logger import logging
from src.model_schema import CANONICAL_FEATURE_ORDER
from src.schemas.prediction import SINGLE_PREDICTION_EXAMPLE


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TrackingSetupError(RuntimeError):
    """Raised when an explicitly requested tracking backend cannot be configured."""


@dataclass(frozen=True, slots=True)
class TrackingResult:
    status: str
    artifact_dir: str
    selected_model: str | None = None
    candidate_scores: dict[str, float] | None = None
    run_id: str | None = None
    registered_model_name: str | None = None
    registered_model_version: str | None = None
    model_version_id: str | None = None
    pipeline_sha256: str | None = None
    warning: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def configure_tracking_backend(settings: DagsHubSettings) -> None:
    """Configure DagsHub as the only supported remote MLflow backend."""
    if not settings.enabled:
        raise TrackingSetupError("DagsHub tracking is disabled")
    if not settings.dagshub_repo_owner or not settings.dagshub_repo_name:
        raise TrackingSetupError(
            "DAGSHUB_REPO_OWNER and DAGSHUB_REPO_NAME are required"
        )
    if settings.dagshub_token and "DAGSHUB_USER_TOKEN" not in os.environ:
        os.environ["DAGSHUB_USER_TOKEN"] = settings.dagshub_token.get_secret_value()
    try:
        dagshub = importlib.import_module("dagshub")
        dagshub.init(
            repo_owner=settings.dagshub_repo_owner,
            repo_name=settings.dagshub_repo_name,
            mlflow=True,
        )
    except Exception as exc:
        raise TrackingSetupError("DagsHub tracking initialization failed") from exc


class ExperimentTracker:
    """Log one completed experiment; disabled mode is a safe no-op."""

    def __init__(self, settings: DagsHubSettings | None = None):
        self.settings = settings or DagsHubSettings()

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    def track(self, training: ModelTrainingResult, config: dict) -> TrackingResult:
        selection_metric = config["model"].get("selection_metric", "roc_auc")
        candidate_scores = {
            name: float(metrics[selection_metric])
            for name, metrics in training.candidate_metrics.items()
        }
        if not self.enabled:
            return TrackingResult(
                status="disabled",
                artifact_dir=str(training.artifact_dir),
                selected_model=training.model_name,
                candidate_scores=candidate_scores,
            )

        try:
            mlflow = importlib.import_module("mlflow")
            configure_tracking_backend(self.settings)
            mlflow.set_experiment(self.settings.experiment_name)
            example = pd.DataFrame(
                [SINGLE_PREDICTION_EXAMPLE], columns=CANONICAL_FEATURE_ORDER
            )
            lineage = self._source_lineage()
            with mlflow.start_run(run_name=training.model_name) as run:
                run_id = run.info.run_id
                mlflow.log_params(self._parameters(training, config, lineage))
                mlflow.log_metrics(
                    self._metrics("validation", training.validation_metrics)
                )
                mlflow.log_metrics(self._metrics("test", training.test_metrics))
                for candidate, metrics in training.candidate_metrics.items():
                    mlflow.log_metrics(self._metrics(f"selection/{candidate}", metrics))
                mlflow.set_tags(
                    {
                        **lineage,
                        "positive_class": "1",
                        "validation_status": "eligible",
                    }
                )
                mlflow.log_artifacts(str(training.artifact_dir))
                signature = mlflow.models.infer_signature(
                    example, training.pipeline.predict(example)
                )
                logged_model = mlflow.sklearn.log_model(
                    sk_model=training.pipeline,
                    name="model",
                    serialization_format=(
                        mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
                    ),
                    signature=signature,
                    input_example=example,
                    pip_requirements=[
                        f"scikit-learn=={sklearn.__version__}",
                        f"pandas=={pd.__version__}",
                        f"numpy=={np.__version__}",
                    ],
                    metadata={
                        "classification_threshold": training.threshold,
                        "positive_class": 1,
                    },
                )

            checksum = hashlib.sha256(
                (training.artifact_dir / "model.pkl").read_bytes()
            ).hexdigest()
            if not self.settings.register_model:
                return TrackingResult(
                    status="tracked",
                    artifact_dir=str(training.artifact_dir),
                    selected_model=training.model_name,
                    candidate_scores=candidate_scores,
                    run_id=run_id,
                    pipeline_sha256=checksum,
                )
            model_uri = getattr(logged_model, "model_uri", None)
            if not model_uri:
                raise RuntimeError("MLflow did not return the logged model URI")
            return self._register(
                mlflow,
                model_uri,
                run_id,
                training,
                config,
                lineage,
            )
        except Exception as exc:
            message = safe_error_message(exc)
            if self.settings.register_model:
                raise RuntimeError(f"Required model publication failed: {message}") from exc
            logging.warning(
                "MLflow tracking failed; local artifacts remain available: %s",
                message,
            )
            return TrackingResult(
                status="failed",
                artifact_dir=str(training.artifact_dir),
                selected_model=training.model_name,
                candidate_scores=candidate_scores,
                warning=message,
            )

    def _register(
        self,
        mlflow,
        model_uri: str,
        run_id: str,
        training: ModelTrainingResult,
        config: dict,
        lineage: dict[str, str],
    ) -> TrackingResult:
        from src.mlops.registry import model_version_id, pipeline_checksum

        model_dir = Path(mlflow.artifacts.download_artifacts(artifact_uri=model_uri))
        checksum = pipeline_checksum(model_dir)
        client = mlflow.tracking.MlflowClient()
        client.set_tag(run_id, "pipeline_sha256", checksum)
        registration = mlflow.register_model(
            model_uri=model_uri, name=self.settings.registered_model_name
        )
        version = str(registration.version)
        contracts = config["contracts"]
        tags = {
            "training_run_id": run_id,
            "source_commit_sha": lineage["source_commit_sha"],
            "feature_schema_version": contracts["feature_schema_version"],
            "prediction_contract_version": contracts["prediction_contract_version"],
            "positive_class": "1",
            "classification_threshold": str(training.threshold),
            "validation_status": "validated",
            "pipeline_sha256": checksum,
        }
        for key, value in tags.items():
            client.set_model_version_tag(
                self.settings.registered_model_name, version, key, value
            )
        example = pd.DataFrame(
            [SINGLE_PREDICTION_EXAMPLE], columns=CANONICAL_FEATURE_ORDER
        )
        mlflow.sklearn.load_model(
            f"models:/{self.settings.registered_model_name}/{version}"
        ).predict(example)
        return TrackingResult(
            status="registered",
            artifact_dir=str(training.artifact_dir),
            selected_model=training.model_name,
            candidate_scores={
                name: float(metrics[config["model"].get("selection_metric", "roc_auc")])
                for name, metrics in training.candidate_metrics.items()
            },
            run_id=run_id,
            registered_model_name=self.settings.registered_model_name,
            registered_model_version=version,
            model_version_id=model_version_id(
                self.settings.dagshub_repo_owner or "",
                self.settings.dagshub_repo_name or "",
                self.settings.registered_model_name,
                version,
            ),
            pipeline_sha256=checksum,
        )

    @staticmethod
    def _parameters(
        training: ModelTrainingResult, config: dict, lineage: dict[str, str]
    ) -> dict[str, str | int | float | bool]:
        split = config["split"]
        dataset = config["dataset"]
        values: dict[str, str | int | float | bool] = {
            "workflow": "customer_churn_training",
            "model_type": training.model_name,
            "candidate_models": ",".join(sorted(config["model"]["candidates"])),
            "selection_metric": config["model"].get("selection_metric", "roc_auc"),
            "random_seed": int(split["random_seed"]),
            "classification_threshold": training.threshold,
            "dataset_name": dataset["name"],
            "dataset_source": dataset["source_identity"],
            "test_size": float(split["test_size"]),
            "validation_size": float(split["validation_size"]),
            "feature_count": len(CANONICAL_FEATURE_ORDER),
            "training_configuration_version": config["version"],
            **lineage,
        }
        values.update(
            {
                f"model.{key}": str(value)
                for key, value in training.model_parameters.items()
            }
        )
        for candidate, candidate_config in config["model"]["candidates"].items():
            values.update(
                {
                    f"candidate.{candidate}.{key}": str(value)
                    for key, value in candidate_config.get("parameters", {}).items()
                }
            )
        return values

    @staticmethod
    def _metrics(prefix: str, values: dict) -> dict[str, float]:
        return {
            f"{prefix}/{key}": float(value)
            for key, value in values.items()
            if isinstance(value, (int, float, np.integer, np.floating))
            and math.isfinite(float(value))
        }

    @staticmethod
    def _source_lineage() -> dict[str, str]:
        def git(*arguments: str) -> str:
            result = subprocess.run(
                ["git", *arguments],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"

        return {
            "source_commit_sha": git("rev-parse", "HEAD"),
            "source_branch": git("branch", "--show-current") or "detached",
            "source_worktree_dirty": str(bool(git("status", "--porcelain"))).lower(),
            "python_version": platform.python_version(),
        }


def create_experiment_tracker(
    settings: DagsHubSettings | None = None,
) -> ExperimentTracker:
    return ExperimentTracker(settings)
