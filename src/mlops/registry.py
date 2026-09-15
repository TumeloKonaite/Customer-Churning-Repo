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
from src.mlops.integrity import (
    COMPLETION_MARKER_ARTIFACT_PATH,
    MANIFEST_ARTIFACT_PATH,
    artifact_manifest_checksum,
    normalize_artifact_path,
    validate_artifact_manifest,
    verify_completion_marker,
    verify_protected_artifacts,
)
from src.mlops.tracking import configure_tracking_backend
from src.model_schema import (
    CANONICAL_FEATURE_ORDER,
    TARGET_COLUMN,
    reject_prohibited_columns,
)


LEGACY_REQUIRED_ARTIFACTS = (
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
INTEGRITY_VERSION_TAGS = (
    "integrity_status",
    "artifact_manifest_schema_version",
    "artifact_manifest_sha256",
)
REQUIRED_PARAMETER_KEYS = (
    "model_type",
    "selection_metric",
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
    integrity_status: str
    model_name: str
    model_version: str
    model_version_id: str
    mlflow_run_id: str
    pipeline_sha256: str
    artifact_manifest_sha256: str | None
    manifest_schema_version: str | None
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


def _download_run_artifact(
    client: MlflowClient, run_id: str, artifact: str, destination: Path
) -> Path:
    try:
        return Path(client.download_artifacts(run_id, artifact, str(destination)))
    except Exception as exc:
        raise ValueError(f"Required MLflow artifact is unavailable: {artifact}") from exc


def _try_download_run_artifact(
    client: MlflowClient, run_id: str, artifact: str, destination: Path
) -> Path | None:
    try:
        return Path(client.download_artifacts(run_id, artifact, str(destination)))
    except Exception:
        return None


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _list_run_artifact_files(client: MlflowClient, run_id: str) -> set[str]:
    files: set[str] = set()
    pending: list[str | None] = [None]
    while pending:
        prefix = pending.pop()
        try:
            artifacts = client.list_artifacts(run_id, prefix)
        except Exception as exc:
            raise ValueError("Unable to enumerate the source MLflow run artifacts") from exc
        for artifact in artifacts:
            path = normalize_artifact_path(artifact.path)
            if artifact.is_dir:
                pending.append(path)
            else:
                files.add(path)
    return files


def _validate_artifact_contents(downloaded: dict[str, Path]) -> dict[str, Any]:
    drift_metadata = _read_json(
        downloaded["references/drift_reference_metadata.json"],
        "Drift reference metadata",
    )
    evaluation_metadata = _read_json(
        downloaded["references/evaluation_reference_metadata.json"],
        "Evaluation reference metadata",
    )
    dataset_identities = _read_json(
        downloaded["lineage/dataset_identities.json"], "Dataset identities"
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
        "feature_schema": _read_json(
            downloaded["contracts/feature_schema.json"], "Feature schema"
        ),
        "prediction_contract": _read_json(
            downloaded["contracts/prediction_contract.json"], "Prediction contract"
        ),
        "privacy_contract": _read_json(
            downloaded["contracts/privacy_contract.json"], "Privacy contract"
        ),
        "monitoring_contract": _read_json(
            downloaded["contracts/monitoring_contract.json"], "Monitoring contract"
        ),
        "dataset_identities": dataset_identities,
    }


def _validate_contract_consistency(
    artifacts: dict[str, Any], version_tags: dict[str, str], manifest: dict[str, Any] | None
) -> str:
    feature_schema = artifacts["feature_schema"]
    prediction_contract = artifacts["prediction_contract"]
    privacy_contract = artifacts["privacy_contract"]
    monitoring_contract = artifacts["monitoring_contract"]
    schema_version = feature_schema.get("schema_version")
    if schema_version != version_tags["feature_schema_version"]:
        raise ValueError("Feature-schema artifact and model-version tag disagree")
    if str(prediction_contract.get("positive_class")) != version_tags["positive_class"]:
        raise ValueError("Positive-class contract and model-version tag disagree")
    if feature_schema.get("canonical_feature_order") != CANONICAL_FEATURE_ORDER:
        raise ValueError("Feature-schema artifact is incompatible with this application")
    if not privacy_contract.get("prohibited_columns"):
        raise ValueError("Privacy contract has no prohibited-column policy")
    if manifest is not None:
        comparisons = {
            "feature_schema_version": schema_version,
            "prediction_contract_version": prediction_contract.get("version"),
            "prediction_horizon_contract_version": prediction_contract.get(
                "prediction_horizon_version"
            ),
            "identity_privacy_contract_version": privacy_contract.get("version"),
            "monitoring_contract_version": monitoring_contract.get("version"),
        }
        for key, actual in comparisons.items():
            if str(manifest[key]) != str(actual):
                raise ValueError(f"Manifest {key} does not match its contract artifact")
        if manifest["training_dataset_identity"] != artifacts["dataset_identities"]["training"]:
            raise ValueError("Manifest training dataset identity does not match its artifact")
        if float(manifest["classification_threshold"]) != float(
            prediction_contract["classification_threshold"]
        ):
            raise ValueError("Manifest classification threshold does not match its contract")
    return str(schema_version)


def _validate_mlflow_model(model_dir: Path) -> None:
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


def _validate_legacy_version(
    *,
    client: MlflowClient,
    run_id: str,
    model_uri: str,
    version: Any,
    expected_pipeline_sha256: str | None,
    temporary: Path,
) -> tuple[dict[str, Any], str]:
    downloaded = {
        artifact: _download_run_artifact(client, run_id, artifact, temporary / "artifacts")
        for artifact in LEGACY_REQUIRED_ARTIFACTS
    }
    artifacts = _validate_artifact_contents(downloaded)
    model_dir = Path(
        mlflow.artifacts.download_artifacts(
            artifact_uri=getattr(version, "source", None) or model_uri,
            dst_path=str(temporary / "model"),
        )
    )
    actual_checksum = pipeline_checksum(model_dir)
    tagged_checksum = version.tags["pipeline_sha256"].lower()
    if actual_checksum != tagged_checksum:
        raise ValueError("Downloaded pipeline checksum does not match model-version metadata")
    if expected_pipeline_sha256 and actual_checksum != expected_pipeline_sha256.lower():
        raise ValueError("Downloaded pipeline checksum does not match expected checksum")
    _validate_mlflow_model(model_dir)
    return artifacts, actual_checksum


def _validate_integrity_version(
    *,
    client: MlflowClient,
    run: Any,
    run_id: str,
    model_uri: str,
    version: Any,
    reference: ExactModelReference,
    settings: DagsHubSettings,
    expected_pipeline_sha256: str | None,
    expected_artifact_manifest_sha256: str | None,
    manifest_path: Path,
    temporary: Path,
) -> tuple[dict[str, Any], str, str, str, dict[str, Any]]:
    _require_mapping_keys(version.tags, INTEGRITY_VERSION_TAGS, "integrity tags")
    if version.tags["integrity_status"] != "complete":
        raise ValueError("Registered model publication is not integrity-complete")
    if run.data.tags.get("integrity_status") != "complete":
        raise ValueError("Source MLflow run is not integrity-complete")
    manifest = _read_json(manifest_path, "Artifact integrity manifest")
    validate_artifact_manifest(manifest)
    manifest_checksum = artifact_manifest_checksum(manifest)
    tagged_manifest_checksum = version.tags["artifact_manifest_sha256"].lower()
    if manifest_checksum != tagged_manifest_checksum:
        raise ValueError("Artifact manifest checksum does not match model-version metadata")
    run_manifest_checksum = run.data.tags.get("artifact_manifest_sha256")
    if not run_manifest_checksum or manifest_checksum != run_manifest_checksum.lower():
        raise ValueError("Artifact manifest checksum does not match MLflow run metadata")
    if expected_artifact_manifest_sha256 and (
        manifest_checksum != expected_artifact_manifest_sha256.lower()
    ):
        raise ValueError("Artifact manifest checksum does not match expected checksum")
    if manifest["manifest_schema_version"] != version.tags[
        "artifact_manifest_schema_version"
    ]:
        raise ValueError("Artifact manifest schema version disagrees with its tag")

    marker_path = _download_run_artifact(
        client, run_id, COMPLETION_MARKER_ARTIFACT_PATH, temporary / "integrity"
    )
    marker = _read_json(marker_path, "Publication completion marker")
    verify_completion_marker(marker, manifest, manifest_checksum=manifest_checksum)

    expected_identity = model_version_id(
        settings.dagshub_repo_owner or "",
        settings.dagshub_repo_name or "",
        reference.name,
        reference.version,
    )
    identity_values = {
        "dagshub_repository_owner": settings.dagshub_repo_owner,
        "dagshub_repository_name": settings.dagshub_repo_name,
        "mlflow_experiment_name": settings.experiment_name,
        "mlflow_run_id": run_id,
        "registered_model_name": reference.name,
        "registered_model_version": reference.version,
        "model_version_id": expected_identity,
        "source_commit_sha": version.tags["source_commit_sha"],
        "source_branch": run.data.tags.get("source_branch"),
        "source_worktree_dirty": (
            str(run.data.tags.get("source_worktree_dirty", "")).lower() == "true"
        ),
        "selected_classifier_name": run.data.params["model_type"],
        "selection_metric": run.data.params["selection_metric"],
        "feature_schema_version": version.tags["feature_schema_version"],
        "prediction_contract_version": version.tags["prediction_contract_version"],
        "prediction_horizon_contract_version": version.tags.get(
            "prediction_horizon_contract_version"
        ),
        "identity_privacy_contract_version": version.tags.get(
            "identity_privacy_contract_version"
        ),
        "monitoring_contract_version": version.tags.get(
            "monitoring_contract_version"
        ),
        "positive_class": version.tags["positive_class"],
    }
    for key, expected in identity_values.items():
        if str(manifest[key]) != str(expected):
            raise ValueError(f"Artifact manifest identity mismatch for {key}")
    if run.data.params["source_commit_sha"] != version.tags["source_commit_sha"]:
        raise ValueError("Source commit differs between the run and model version")
    experiment_id = getattr(getattr(run, "info", None), "experiment_id", None)
    if not experiment_id:
        raise ValueError("Source MLflow run has no experiment identity")
    try:
        experiment = client.get_experiment(experiment_id)
    except Exception as exc:
        raise ValueError("Unable to resolve the source MLflow experiment") from exc
    if experiment.name != manifest["mlflow_experiment_name"]:
        raise ValueError("Artifact manifest experiment name disagrees with the source run")
    if float(manifest["classification_threshold"]) != float(
        run.data.params["classification_threshold"]
    ):
        raise ValueError("Artifact manifest classification threshold disagrees with the run")

    run_files = _list_run_artifact_files(client, run_id)
    declared = {entry["path"] for entry in manifest["protected_artifacts"]}
    declared_run_files = {path for path in declared if not path.startswith("model/")}
    actual_run_files = {
        path
        for path in run_files
        if not path.startswith("integrity/") and not path.startswith("model/")
    }
    if declared_run_files != actual_run_files:
        missing = sorted(declared_run_files - actual_run_files)
        unprotected = sorted(actual_run_files - declared_run_files)
        raise ValueError(
            f"MLflow run artifact inventory differs from the manifest; "
            f"missing={missing}, unprotected={unprotected}"
        )

    model_dir = Path(
        mlflow.artifacts.download_artifacts(
            artifact_uri=getattr(version, "source", None) or model_uri,
            dst_path=str(temporary / "model"),
        )
    )
    actual_model_files = {
        f"model/{normalize_artifact_path(path.relative_to(model_dir))}"
        for path in model_dir.rglob("*")
        if path.is_file()
    }
    declared_model_files = {path for path in declared if path.startswith("model/")}
    if actual_model_files != declared_model_files:
        missing = sorted(declared_model_files - actual_model_files)
        unprotected = sorted(actual_model_files - declared_model_files)
        raise ValueError(
            f"MLflow model artifact inventory differs from the manifest; "
            f"missing={missing}, unprotected={unprotected}"
        )

    downloaded: dict[str, Path] = {}
    for path in sorted(declared_run_files):
        downloaded[path] = _download_run_artifact(
            client, run_id, path, temporary / "protected"
        )

    def resolve_path(path: str) -> Path:
        if path.startswith("model/"):
            return model_dir.joinpath(*_posix_parts(path)[1:])
        return downloaded[path]

    verify_protected_artifacts(manifest, resolve_path)
    actual_checksum = pipeline_checksum(model_dir)
    checksum_sources = {
        "manifest": manifest["pipeline_sha256"],
        "model-version tag": version.tags["pipeline_sha256"],
        "MLflow run tag or parameter": run.data.tags.get("pipeline_sha256")
        or run.data.params.get("pipeline_sha256"),
    }
    for label, checksum in checksum_sources.items():
        if not checksum or actual_checksum != str(checksum).lower():
            raise ValueError(f"Pipeline checksum does not match {label}")
    if expected_pipeline_sha256 and actual_checksum != expected_pipeline_sha256.lower():
        raise ValueError("Downloaded pipeline checksum does not match expected checksum")

    artifacts = _validate_artifact_contents(downloaded)
    _validate_mlflow_model(model_dir)
    return (
        artifacts,
        actual_checksum,
        manifest_checksum,
        str(manifest["manifest_schema_version"]),
        manifest,
    )


def _posix_parts(path: str) -> tuple[str, ...]:
    """Return already-validated POSIX parts without host path semantics."""
    return tuple(normalize_artifact_path(path).split("/"))


def validate_registered_model(
    model_uri: str,
    *,
    settings: DagsHubSettings | None = None,
    expected_run_id: str | None = None,
    expected_pipeline_sha256: str | None = None,
    expected_artifact_manifest_sha256: str | None = None,
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
        for metric in (
            "roc_auc",
            "pr_auc",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "log_loss",
            "row_count",
        )
    )
    _require_mapping_keys(run.data.metrics, required_metrics, "run metrics")
    _require_mapping_keys(version.tags, REQUIRED_VERSION_TAGS, "model-version tags")
    if version.tags["training_run_id"] != run_id:
        raise ValueError("Model-version training_run_id tag does not match its source run")

    with tempfile.TemporaryDirectory(prefix="validate-churn-model-") as temporary:
        temp = Path(temporary)
        manifest_path = _try_download_run_artifact(
            client, run_id, MANIFEST_ARTIFACT_PATH, temp / "manifest"
        )
        has_integrity_tags = any(key in version.tags for key in INTEGRITY_VERSION_TAGS)
        if manifest_path is None and not has_integrity_tags:
            if expected_artifact_manifest_sha256:
                raise ValueError("Legacy model version has no artifact manifest checksum")
            artifacts, actual_checksum = _validate_legacy_version(
                client=client,
                run_id=run_id,
                model_uri=model_uri,
                version=version,
                expected_pipeline_sha256=expected_pipeline_sha256,
                temporary=temp,
            )
            manifest_checksum = None
            manifest_schema_version = None
            integrity_status = "legacy"
            manifest = None
        else:
            if manifest_path is None:
                raise ValueError("Integrity-enabled model version is missing its manifest")
            (
                artifacts,
                actual_checksum,
                manifest_checksum,
                manifest_schema_version,
                manifest,
            ) = _validate_integrity_version(
                client=client,
                run=run,
                run_id=run_id,
                model_uri=model_uri,
                version=version,
                reference=reference,
                settings=settings,
                expected_pipeline_sha256=expected_pipeline_sha256,
                expected_artifact_manifest_sha256=expected_artifact_manifest_sha256,
                manifest_path=manifest_path,
                temporary=temp,
            )
            integrity_status = "complete"

    schema_version = _validate_contract_consistency(artifacts, version.tags, manifest)
    return ValidationResult(
        valid=True,
        integrity_status=integrity_status,
        model_name=reference.name,
        model_version=reference.version,
        model_version_id=model_version_id(
            settings.dagshub_repo_owner or "",
            settings.dagshub_repo_name or "",
            reference.name,
            reference.version,
        ),
        mlflow_run_id=run_id,
        pipeline_sha256=actual_checksum,
        artifact_manifest_sha256=manifest_checksum,
        manifest_schema_version=manifest_schema_version,
        feature_schema_version=schema_version,
        source_commit_sha=version.tags["source_commit_sha"],
        duration_seconds=time.perf_counter() - started,
    )
