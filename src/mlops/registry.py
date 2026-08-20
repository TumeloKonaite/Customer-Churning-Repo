"""Independent validation of exact DagsHub MLflow registered model versions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.models import Model
import mlflow.sklearn
import pandas as pd

from src.config import DagsHubSettings
from src.mlops.tracking import configure_tracking_backend
from src.model_schema import (
    CANONICAL_FEATURE_ORDER,
    TARGET_COLUMN,
    reject_prohibited_columns,
)


REQUIRED_ARTIFACTS = (
    "contracts/feature_schema.json",
    "contracts/prediction_contract.json",
    "contracts/monitoring_contract.json",
    "contracts/privacy_contract.json",
    "evaluation/metrics.json",
    "evaluation/confusion_matrix.json",
    "evaluation/classification_report.json",
    "lineage/dataset_identities.json",
    "references/drift_reference.parquet",
    "references/drift_reference_metadata.json",
    "references/evaluation_reference.parquet",
    "references/evaluation_reference_metadata.json",
)
REQUIRED_VERSION_TAGS = (
    "source_commit_sha",
    "training_run_id",
    "feature_schema_version",
    "prediction_contract_version",
    "positive_class",
    "classification_threshold",
    "validation_status",
    "pipeline_sha256",
)
REQUIRED_PARAMETER_KEYS = (
    "model_type",
    "dataset_name",
    "dataset_source",
    "test_size",
    "validation_size",
    "classification_threshold",
    "random_seed",
    "training_configuration_version",
    "source_commit_sha",
)


_EXACT_URI = re.compile(r"^models:/([^/@]+)/(\d+)$")


@dataclass(frozen=True, slots=True)
class ExactModelReference:
    name: str
    version: str


def parse_exact_model_uri(uri: str) -> ExactModelReference:
    match = _EXACT_URI.fullmatch(uri.strip())
    if not match or int(match.group(2)) < 1:
        raise ValueError(
            "Model URI must use an exact positive numeric version, for example "
            "models:/churn_predictor/7; latest, aliases, and stages are not allowed"
        )
    return ExactModelReference(name=match.group(1), version=match.group(2))


def exact_model_uri(model_name: str, version: str | int) -> str:
    reference = parse_exact_model_uri(f"models:/{model_name}/{version}")
    return f"models:/{reference.name}/{reference.version}"


def model_version_id(
    owner: str, repository: str, model_name: str, version: str | int
) -> str:
    reference = parse_exact_model_uri(f"models:/{model_name}/{version}")
    if not owner or not repository:
        raise ValueError("DagsHub owner and repository are required")
    return f"dagshub:{owner}/{repository}:{reference.name}:{reference.version}"


@dataclass(frozen=True, slots=True)
class ValidationResult:
    valid: bool
    model_name: str
    model_version: str
    model_version_id: str
    mlflow_run_id: str
    pipeline_sha256: str
    feature_schema_version: str
    source_commit_sha: str
    duration_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def configure_mlflow(settings: DagsHubSettings) -> MlflowClient:
    configure_tracking_backend(settings)
    return MlflowClient()


def pipeline_checksum(model_dir: Path) -> str:
    candidates = sorted(model_dir.rglob("model.pkl"))
    if len(candidates) != 1:
        raise ValueError("MLflow model must contain exactly one serialized sklearn pipeline")
    return hashlib.sha256(candidates[0].read_bytes()).hexdigest()


def _require_mapping_keys(mapping: dict, required: tuple[str, ...], label: str) -> None:
    missing = sorted(key for key in required if key not in mapping or mapping[key] is None)
    if missing:
        raise ValueError(f"Registered model is missing required {label}: {missing}")


def _validate_artifacts(client: MlflowClient, run_id: str, destination: Path) -> dict:
    downloaded: dict[str, Path] = {}
    for artifact in REQUIRED_ARTIFACTS:
        try:
            downloaded[artifact] = Path(client.download_artifacts(run_id, artifact, str(destination)))
        except Exception as exc:
            raise ValueError(f"Required MLflow artifact is unavailable: {artifact}") from exc
    drift_metadata = json.loads(downloaded["references/drift_reference_metadata.json"].read_text())
    evaluation_metadata = json.loads(
        downloaded["references/evaluation_reference_metadata.json"].read_text()
    )
    dataset_identities = json.loads(
        downloaded["lineage/dataset_identities.json"].read_text()
    )
    if set(dataset_identities) != {"training", "validation", "evaluation"}:
        raise ValueError("Training dataset identities are incomplete")
    for cohort, identity in dataset_identities.items():
        if not identity.get("dataset_digest") or not identity.get("row_count"):
            raise ValueError(f"{cohort} dataset identity is incomplete")
        reject_prohibited_columns(identity.get("feature_list", []))
    if drift_metadata.get("dataset_purpose") != "drift_reference":
        raise ValueError("Drift reference metadata has the wrong purpose")
    if evaluation_metadata.get("dataset_purpose") != "evaluation_reference":
        raise ValueError("Evaluation reference metadata has the wrong purpose")
    drift = pd.read_parquet(downloaded["references/drift_reference.parquet"])
    evaluation = pd.read_parquet(downloaded["references/evaluation_reference.parquet"])
    if list(drift.columns) != CANONICAL_FEATURE_ORDER:
        raise ValueError("Drift reference has incompatible columns")
    if list(evaluation.columns) != CANONICAL_FEATURE_ORDER + [TARGET_COLUMN]:
        raise ValueError("Evaluation reference has incompatible columns")
    if drift.empty or evaluation.empty:
        raise ValueError("Reference datasets must not be empty")
    reject_prohibited_columns(drift.columns)
    reject_prohibited_columns(evaluation.columns)
    return {
        "feature_schema": json.loads(downloaded["contracts/feature_schema.json"].read_text()),
        "prediction_contract": json.loads(
            downloaded["contracts/prediction_contract.json"].read_text()
        ),
    }


def validate_registered_model(
    model_uri: str,
    *,
    settings: DagsHubSettings | None = None,
    expected_run_id: str | None = None,
    expected_pipeline_sha256: str | None = None,
) -> ValidationResult:
    started = time.perf_counter()
    reference = parse_exact_model_uri(model_uri)
    settings = settings or DagsHubSettings()
    if reference.name != settings.registered_model_name:
        raise ValueError(
            f"Expected registered model {settings.registered_model_name!r}, got {reference.name!r}"
        )
    client = configure_mlflow(settings)
    version = client.get_model_version(reference.name, reference.version)
    if str(version.version) != reference.version or version.name != reference.name:
        raise ValueError("DagsHub returned a different registered model identity")
    run_id = version.run_id
    if not run_id:
        raise ValueError("Registered model version has no source MLflow run")
    if expected_run_id and run_id != expected_run_id:
        raise ValueError("Registered model run ID does not match EXPECTED_MLFLOW_RUN_ID")
    run = client.get_run(run_id)
    _require_mapping_keys(run.data.params, REQUIRED_PARAMETER_KEYS, "run parameters")
    required_metrics = tuple(
        f"{cohort}/{metric}"
        for cohort in ("validation", "test")
        for metric in ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "log_loss", "row_count")
    )
    _require_mapping_keys(run.data.metrics, required_metrics, "run metrics")
    _require_mapping_keys(version.tags, REQUIRED_VERSION_TAGS, "model-version tags")
    if version.tags["training_run_id"] != run_id:
        raise ValueError("Model-version training_run_id tag does not match its source run")

    with tempfile.TemporaryDirectory(prefix="validate-churn-model-") as temporary:
        temp = Path(temporary)
        artifacts = _validate_artifacts(client, run_id, temp / "artifacts")
        model_dir = Path(
            mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=str(temp / "model"))
        )
        mlflow_model = Model.load(str(model_dir / "MLmodel"))
        if mlflow_model.signature is None:
            raise ValueError("MLflow model signature is missing")
        if mlflow_model.signature.outputs.input_names() != [
            "predicted_class",
            "churn_probability",
        ]:
            raise ValueError("MLflow model output signature is incompatible")
        input_example = mlflow_model.load_input_example(str(model_dir))
        if input_example is None:
            raise ValueError("MLflow input example is missing")
        reject_prohibited_columns(input_example.columns)
        fitted_pipeline = mlflow.sklearn.load_model(str(model_dir))
        fitted_pipeline.predict(input_example)
        actual_checksum = pipeline_checksum(model_dir)
        tagged_checksum = version.tags["pipeline_sha256"].lower()
        if actual_checksum != tagged_checksum:
            raise ValueError("Downloaded pipeline checksum does not match model-version metadata")
        if expected_pipeline_sha256 and actual_checksum != expected_pipeline_sha256.lower():
            raise ValueError("Downloaded pipeline checksum does not match expected checksum")

    schema_version = artifacts["feature_schema"].get("schema_version")
    if schema_version != version.tags["feature_schema_version"]:
        raise ValueError("Feature-schema artifact and model-version tag disagree")
    if str(artifacts["prediction_contract"].get("positive_class")) != version.tags["positive_class"]:
        raise ValueError("Positive-class contract and model-version tag disagree")
    if artifacts["feature_schema"].get("canonical_feature_order") != CANONICAL_FEATURE_ORDER:
        raise ValueError("Feature-schema artifact is incompatible with this application")
    return ValidationResult(
        valid=True,
        model_name=reference.name,
        model_version=reference.version,
        model_version_id=model_version_id(
            settings.dagshub_repo_owner,
            settings.dagshub_repo_name,
            reference.name,
            reference.version,
        ),
        mlflow_run_id=run_id,
        pipeline_sha256=actual_checksum,
        feature_schema_version=schema_version,
        source_commit_sha=version.tags["source_commit_sha"],
        duration_seconds=time.perf_counter() - started,
    )
