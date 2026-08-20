"""Create and validate inference-only packages for Modal."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import uuid
from typing import Any

import mlflow
from mlflow.models import Model
import mlflow.sklearn

from src.config import DagsHubSettings, DeploymentSettings
from src.mlops.registry import (
    ValidationResult,
    configure_mlflow,
    pipeline_checksum,
    validate_registered_model,
)
from src.model_schema import CANONICAL_FEATURE_ORDER, reject_prohibited_columns


DEPLOYMENT_METADATA_FILENAME = "deployment_metadata.json"
MODEL_DIRECTORY_NAME = "model"


def _write_package(
    destination: Path,
    *,
    validation: ValidationResult,
    settings: DagsHubSettings,
    environment: str,
    application_version: str,
    modal_application: str,
) -> dict[str, Any]:
    client = configure_mlflow(settings)
    model_uri = f"models:/{validation.model_name}/{validation.model_version}"
    model_download = Path(
        mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri, dst_path=str(destination / "download")
        )
    )
    shutil.copytree(model_download, destination / MODEL_DIRECTORY_NAME)
    shutil.rmtree(destination / "download")
    feature_contract = Path(
        client.download_artifacts(
            validation.mlflow_run_id,
            "contracts/feature_schema.json",
            str(destination),
        )
    )
    contract_destination = destination / "feature_schema.json"
    if feature_contract != contract_destination:
        shutil.move(str(feature_contract), contract_destination)
        contracts_dir = destination / "contracts"
        if contracts_dir.exists() and not any(contracts_dir.iterdir()):
            contracts_dir.rmdir()
    metadata = {
        "deployment_id": str(uuid.uuid4()),
        "environment": environment,
        "deployment_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "modal_application": modal_application,
        "model_name": validation.model_name,
        "model_version": validation.model_version,
        "model_version_id": validation.model_version_id,
        "mlflow_run_id": validation.mlflow_run_id,
        "source_commit_sha": validation.source_commit_sha,
        "pipeline_sha256": validation.pipeline_sha256,
        "application_version": application_version,
        "feature_schema_version": validation.feature_schema_version,
        "validation_status": "validated",
    }
    (destination / DEPLOYMENT_METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metadata


def prepare_deployment(
    model_uri: str,
    output_dir: str | Path,
    *,
    mlflow_settings: DagsHubSettings | None = None,
    expected_run_id: str | None = None,
    expected_pipeline_sha256: str | None = None,
    expected_model_version_id: str | None = None,
    environment: str = "production",
    application_version: str = "0.1.0",
    modal_application: str = "customer-churn-backend",
) -> dict[str, Any]:
    """Validate a numeric model version, then emit only inference-required files."""
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Deployment output directory must be absent or empty: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    settings = mlflow_settings or DagsHubSettings()
    validation = validate_registered_model(
        model_uri,
        settings=settings,
        expected_run_id=expected_run_id,
        expected_pipeline_sha256=expected_pipeline_sha256,
    )
    if (
        expected_model_version_id is not None
        and validation.model_version_id != expected_model_version_id
    ):
        raise ValueError("Registered model identity does not match EXPECTED_MODEL_VERSION_ID")
    with tempfile.TemporaryDirectory(prefix="prepare-churn-deployment-", dir=output.parent) as temp:
        staged = Path(temp) / "package"
        staged.mkdir()
        metadata = _write_package(
            staged,
            validation=validation,
            settings=settings,
            environment=environment,
            application_version=application_version,
            modal_application=modal_application,
        )
        if output.exists():
            output.rmdir()
        os.replace(staged, output)
    validate_packaged_model(output, expected=metadata)
    return metadata


def load_deployment_metadata(package_dir: str | Path) -> dict[str, Any]:
    path = Path(package_dir) / DEPLOYMENT_METADATA_FILENAME
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Deployment metadata is missing or invalid") from exc
    required = {
        "deployment_id", "environment", "deployment_timestamp_utc", "modal_application",
        "model_name", "model_version", "model_version_id", "mlflow_run_id",
        "source_commit_sha", "pipeline_sha256", "application_version",
        "feature_schema_version", "validation_status",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"Deployment metadata is incomplete: {missing}")
    return metadata


def validate_packaged_model(
    package_dir: str | Path,
    *,
    expected: dict[str, Any] | DeploymentSettings | None = None,
) -> dict[str, Any]:
    """Validate identity, checksum, schema, deserialization, and a smoke prediction."""
    started = time.perf_counter()
    package = Path(package_dir)
    metadata = load_deployment_metadata(package)
    model_dir = package / MODEL_DIRECTORY_NAME
    contract_path = package / "feature_schema.json"
    if not model_dir.is_dir() or not contract_path.is_file():
        raise ValueError("Deployment package is missing the model or feature contract")
    if not str(metadata["model_version"]).isdigit():
        raise ValueError("Packaged model version is not an exact numeric version")
    if isinstance(expected, DeploymentSettings):
        assertions = {
            "model_name": expected.model_name,
            "model_version": expected.model_version,
            "mlflow_run_id": expected.expected_run_id,
            "model_version_id": expected.expected_model_version_id,
            "pipeline_sha256": expected.expected_pipeline_sha256,
            "environment": expected.environment.value,
        }
    else:
        assertions = expected or {}
    for key, expected_value in assertions.items():
        if expected_value is not None and str(metadata.get(key)) != str(expected_value):
            raise ValueError(f"Packaged deployment identity mismatch for {key}")
    actual_checksum = pipeline_checksum(model_dir)
    if actual_checksum != metadata["pipeline_sha256"]:
        raise ValueError("Packaged pipeline checksum mismatch")
    schema = json.loads(contract_path.read_text(encoding="utf-8"))
    if schema.get("schema_version") != metadata["feature_schema_version"]:
        raise ValueError("Packaged feature contract version mismatch")
    if schema.get("canonical_feature_order") != CANONICAL_FEATURE_ORDER:
        raise ValueError("Packaged feature contract is incompatible with this application")
    reject_prohibited_columns(schema.get("canonical_feature_order", []))
    mlflow_model = Model.load(str(model_dir / "MLmodel"))
    input_example = mlflow_model.load_input_example(str(model_dir))
    if input_example is None:
        raise ValueError("Packaged MLflow model has no input example")
    fitted_pipeline = mlflow.sklearn.load_model(str(model_dir))
    fitted_pipeline.predict(input_example)
    metadata["startup_validation_duration_seconds"] = time.perf_counter() - started
    return metadata


def validate_production_startup(package_dir: str | Path | None = None) -> dict[str, Any]:
    settings = DeploymentSettings()
    package = package_dir or settings.deployment_package_dir
    return validate_packaged_model(package, expected=settings)
