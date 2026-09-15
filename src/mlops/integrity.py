"""Canonical integrity manifests for DagsHub MLflow model publications."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_SCHEMA_PATH = (
    PROJECT_ROOT / "configs" / "contracts" / "artifact_manifest.schema.json"
)
MANIFEST_SCHEMA_VERSION = "1.0.0"
CANONICALIZATION_VERSION = "sorted-compact-json-v1"
COMPLETION_MARKER_SCHEMA_VERSION = "1.0.0"
PUBLISHER_VERSION = "0.1.0"
MANIFEST_ARTIFACT_PATH = "integrity/artifact_manifest.json"
COMPLETION_MARKER_ARTIFACT_PATH = "integrity/publication_complete.json"
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")

REQUIRED_RUN_ARTIFACTS = frozenset(
    {
        "contracts/feature_schema.json",
        "contracts/prediction_contract.json",
        "contracts/privacy_contract.json",
        "contracts/monitoring_contract.json",
        "lineage/training_config.json",
        "lineage/dataset_identities.json",
        "evaluation/metrics.json",
        "evaluation/model_comparison.json",
        "references/drift_reference_metadata.json",
        "references/drift_reference.parquet",
        "references/evaluation_reference_metadata.json",
        "references/evaluation_reference.parquet",
    }
)
REQUIRED_MODEL_FILES = frozenset(
    {
        "model/MLmodel",
        "model/model.pkl",
        "model/conda.yaml",
        "model/python_env.yaml",
        "model/requirements.txt",
        "model/input_example.json",
    }
)
APPROVED_TOP_LEVEL_PATHS = frozenset(
    {"contracts", "evaluation", "lineage", "references", "model", "model.pkl"}
)
_PROHIBITED_METADATA_KEYS = frozenset(
    {
        "authorization",
        "authenticationheader",
        "authenticationheaders",
        "credential",
        "credentials",
        "dagshubtoken",
        "databaseurl",
        "connectionstring",
        "password",
        "secret",
        "token",
        "customerid",
        "rownumber",
        "surname",
        "email",
        "phone",
        "telephone",
        "address",
        "rawfeaturerows",
        "rawcustomeridentifiers",
    }
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]+"),
    re.compile(r"(?i)\b(?:password|token|secret)\s*[=:]\s*\S+"),
    re.compile(r"^[a-z][a-z0-9+.-]*://[^/\s]*@", re.IGNORECASE),
)
_EXCLUDED_ARTIFACT_PARTS = frozenset(
    {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git"}
)
_EXCLUDED_ARTIFACT_NAMES = frozenset(
    {".env", "credentials", "credentials.json", "secrets.json"}
)


def utc_timestamp(value: datetime | None = None) -> str:
    timestamp = value or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("Integrity timestamps must be timezone-aware")
    return timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically for publication and verification."""
    _reject_non_finite_numbers(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_json(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def artifact_manifest_checksum(manifest: Mapping[str, Any]) -> str:
    if "artifact_manifest_sha256" in manifest:
        raise ValueError("The artifact manifest must not contain its own checksum")
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_artifact_path(path: str | Path | PurePosixPath) -> str:
    raw = str(path).replace("\\", "/")
    if not raw or raw.startswith("/") or re.match(r"^[A-Za-z]:", raw):
        raise ValueError("Artifact path must be repository-relative")
    normalized = PurePosixPath(raw)
    if any(part in {"", ".", ".."} for part in normalized.parts):
        raise ValueError("Artifact path is not normalized")
    value = normalized.as_posix()
    if value != raw:
        raise ValueError("Artifact path is not normalized")
    return value


def artifact_role(path: str) -> str:
    _reject_excluded_artifact_path(path)
    roles = {
        "model/MLmodel": "mlflow_model_metadata",
        "model/model.pkl": "fitted_pipeline",
        "model/input_example.json": "model_input_example",
        "model/serving_input_example.json": "serving_input_example",
        "contracts/feature_schema.json": "feature_contract",
        "contracts/prediction_contract.json": "prediction_contract",
        "contracts/privacy_contract.json": "privacy_contract",
        "contracts/monitoring_contract.json": "monitoring_contract",
        "lineage/training_config.json": "training_configuration",
        "lineage/dataset_identities.json": "dataset_identity",
        "evaluation/metrics.json": "evaluation_metrics",
        "evaluation/model_comparison.json": "model_comparison",
        "references/drift_reference_metadata.json": "drift_reference_metadata",
        "references/drift_reference.parquet": "drift_reference_dataset",
        "references/evaluation_reference_metadata.json": "evaluation_reference_metadata",
        "references/evaluation_reference.parquet": "evaluation_reference_dataset",
    }
    if path in roles:
        return roles[path]
    if path == "model.pkl":
        return "local_fitted_pipeline"
    if path.startswith("model/"):
        if PurePosixPath(path).name in {"conda.yaml", "python_env.yaml", "requirements.txt"}:
            return "model_environment"
        return "mlflow_model_artifact"
    if path.startswith("contracts/"):
        return "versioned_contract"
    if path.startswith("evaluation/"):
        return "evaluation_artifact"
    if path.startswith("lineage/"):
        return "lineage_artifact"
    if path.startswith("references/"):
        return "approved_reference_artifact"
    raise ValueError(f"Artifact is outside the approved production set: {path}")


def build_protected_artifact_entries(
    root: str | Path,
    *,
    paths: Iterable[str] | None = None,
    require_complete_set: bool = True,
) -> list[dict[str, Any]]:
    """Hash the approved artifact inventory rooted at ``root``."""
    base = Path(root)
    candidates = (
        [normalize_artifact_path(path) for path in paths]
        if paths is not None
        else [
            normalize_artifact_path(path.relative_to(base))
            for path in base.rglob("*")
            if path.is_file() and path.relative_to(base).parts[0] != "integrity"
        ]
    )
    if len(candidates) != len(set(candidates)):
        raise ValueError("Protected artifact paths must be unique")
    entries: list[dict[str, Any]] = []
    for relative in sorted(candidates):
        _reject_excluded_artifact_path(relative)
        if PurePosixPath(relative).parts[0] not in APPROVED_TOP_LEVEL_PATHS:
            raise ValueError(f"Artifact is outside the approved production set: {relative}")
        artifact = base.joinpath(*PurePosixPath(relative).parts)
        if artifact.is_symlink():
            raise ValueError(f"Protected artifact must not be a symbolic link: {relative}")
        if not artifact.is_file():
            raise ValueError(f"Protected artifact is unavailable: {relative}")
        entries.append(
            {
                "path": relative,
                "role": artifact_role(relative),
                "sha256": file_sha256(artifact),
                "size_bytes": artifact.stat().st_size,
            }
        )
    validate_protected_artifact_entries(
        entries, require_complete_set=require_complete_set
    )
    return entries


def validate_publication_source_artifacts(root: str | Path) -> None:
    """Reject unsafe local files and metadata before anything is sent remotely."""
    base = Path(root)
    integrity_dir = base / "integrity"
    if integrity_dir.exists() and any(integrity_dir.rglob("*")):
        raise ValueError(
            "Local training artifacts must not contain integrity publication files"
        )
    entries = build_protected_artifact_entries(
        base, require_complete_set=False
    )
    for entry in entries:
        if not entry["path"].endswith(".json"):
            continue
        path = base.joinpath(*entry["path"].split("/"))
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Approved JSON artifact is invalid: {entry['path']}"
            ) from exc
        _reject_unsafe_metadata(value)


def validate_protected_artifact_entries(
    entries: Iterable[Mapping[str, Any]],
    *,
    require_complete_set: bool = True,
) -> None:
    materialized = list(entries)
    paths = [normalize_artifact_path(str(entry.get("path", ""))) for entry in materialized]
    if paths != sorted(paths):
        raise ValueError("Protected artifacts must use stable path ordering")
    if len(paths) != len(set(paths)):
        raise ValueError("Protected artifact paths must be unique")
    for entry, path in zip(materialized, paths):
        if set(entry) != {"path", "role", "sha256", "size_bytes"}:
            raise ValueError(f"Protected artifact entry has unexpected fields: {path}")
        if entry["role"] != artifact_role(path):
            raise ValueError(f"Protected artifact role is invalid: {path}")
        if not SHA256_PATTERN.fullmatch(str(entry["sha256"])):
            raise ValueError(f"Protected artifact checksum is invalid: {path}")
        size = entry["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"Protected artifact size is invalid: {path}")
    if require_complete_set:
        missing = sorted((REQUIRED_RUN_ARTIFACTS | REQUIRED_MODEL_FILES) - set(paths))
        if missing:
            raise ValueError(f"Integrity manifest is missing required artifacts: {missing}")


def build_artifact_manifest(
    *,
    repository_owner: str,
    repository_name: str,
    experiment_name: str,
    run_id: str,
    model_name: str,
    model_version: str | int,
    model_version_id: str,
    source_commit_sha: str,
    source_branch: str,
    source_worktree_dirty: bool,
    selected_classifier_name: str,
    selection_metric: str,
    feature_schema_version: str,
    prediction_contract_version: str,
    prediction_horizon_contract_version: str,
    identity_privacy_contract_version: str,
    monitoring_contract_version: str,
    training_dataset_identity: Mapping[str, Any],
    classification_threshold: float,
    positive_class: str | int,
    python_version: str,
    scikit_learn_version: str,
    mlflow_version: str,
    pipeline_sha256: str,
    protected_artifacts: Iterable[Mapping[str, Any]],
    publication_timestamp_utc: str | None = None,
) -> dict[str, Any]:
    version = str(model_version)
    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "dagshub_repository_owner": repository_owner,
        "dagshub_repository_name": repository_name,
        "mlflow_experiment_name": experiment_name,
        "mlflow_run_id": run_id,
        "registered_model_name": model_name,
        "registered_model_version": version,
        "model_version_id": model_version_id,
        "source_commit_sha": source_commit_sha,
        "source_branch": source_branch,
        "source_worktree_dirty": bool(source_worktree_dirty),
        "selected_classifier_name": selected_classifier_name,
        "selection_metric": selection_metric,
        "feature_schema_version": feature_schema_version,
        "prediction_contract_version": prediction_contract_version,
        "prediction_horizon_contract_version": prediction_horizon_contract_version,
        "identity_privacy_contract_version": identity_privacy_contract_version,
        "monitoring_contract_version": monitoring_contract_version,
        "training_dataset_identity": dict(training_dataset_identity),
        "classification_threshold": float(classification_threshold),
        "positive_class": positive_class,
        "python_version": python_version,
        "scikit_learn_version": scikit_learn_version,
        "mlflow_version": mlflow_version,
        "pipeline_sha256": str(pipeline_sha256).lower(),
        "protected_artifacts": [dict(entry) for entry in protected_artifacts],
        "publication_timestamp_utc": publication_timestamp_utc or utc_timestamp(),
    }
    validate_artifact_manifest(manifest)
    return manifest


def validate_artifact_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the repository schema plus canonical and security invariants."""
    if "artifact_manifest_sha256" in manifest:
        raise ValueError("The artifact manifest must not contain its own checksum")
    schema = json.loads(MANIFEST_SCHEMA_PATH.read_text(encoding="utf-8"))
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError:
        _fallback_schema_validation(manifest, schema)
    else:
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        errors = sorted(validator.iter_errors(manifest), key=lambda error: list(error.path))
        if errors:
            location = ".".join(str(item) for item in errors[0].path) or "manifest"
            raise ValueError(
                "Artifact manifest JSON Schema violation at "
                f"{location} ({errors[0].validator})"
            )
    if manifest.get("canonicalization_version") != CANONICALIZATION_VERSION:
        raise ValueError("Unsupported manifest canonicalization version")
    entries = manifest.get("protected_artifacts")
    if not isinstance(entries, list):
        raise ValueError("Artifact manifest protected_artifacts must be a list")
    validate_protected_artifact_entries(entries)
    _reject_unsafe_metadata(manifest)
    canonical_json_bytes(manifest)


def verify_protected_artifacts(
    manifest: Mapping[str, Any],
    resolve_path: Callable[[str], Path],
) -> None:
    """Verify every declared file before any serialized model is loaded."""
    validate_artifact_manifest(manifest)
    for entry in manifest["protected_artifacts"]:
        path = entry["path"]
        artifact = resolve_path(path)
        if not artifact.is_file() or artifact.is_symlink():
            raise ValueError(f"Protected artifact is unavailable: {path}")
        if artifact.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"Protected artifact size mismatch: {path}")
        if file_sha256(artifact) != entry["sha256"]:
            raise ValueError(f"Protected artifact checksum mismatch: {path}")


def build_completion_marker(
    *,
    model_name: str,
    model_version: str | int,
    model_version_id: str,
    run_id: str,
    pipeline_sha256: str,
    artifact_manifest_sha256: str,
    completion_timestamp_utc: str | None = None,
    publisher_version: str = PUBLISHER_VERSION,
) -> dict[str, Any]:
    marker = {
        "completion_marker_schema_version": COMPLETION_MARKER_SCHEMA_VERSION,
        "model_name": model_name,
        "model_version": str(model_version),
        "model_version_id": model_version_id,
        "mlflow_run_id": run_id,
        "pipeline_sha256": pipeline_sha256.lower(),
        "artifact_manifest_sha256": artifact_manifest_sha256.lower(),
        "completion_timestamp_utc": completion_timestamp_utc or utc_timestamp(),
        "publisher_version": publisher_version,
    }
    validate_completion_marker(marker)
    return marker


def validate_completion_marker(marker: Mapping[str, Any]) -> None:
    required = {
        "completion_marker_schema_version",
        "model_name",
        "model_version",
        "model_version_id",
        "mlflow_run_id",
        "pipeline_sha256",
        "artifact_manifest_sha256",
        "completion_timestamp_utc",
        "publisher_version",
    }
    if set(marker) != required:
        raise ValueError("Publication completion marker has missing or unexpected fields")
    if marker["completion_marker_schema_version"] != COMPLETION_MARKER_SCHEMA_VERSION:
        raise ValueError("Unsupported completion-marker schema version")
    if marker["model_name"] != "churn_predictor":
        raise ValueError("Completion marker model name is invalid")
    if not str(marker["model_version"]).isdigit() or int(marker["model_version"]) < 1:
        raise ValueError("Completion marker model version must be numeric")
    if not str(marker["model_version_id"]).endswith(
        f":{marker['model_name']}:{marker['model_version']}"
    ):
        raise ValueError("Completion marker model_version_id is inconsistent")
    for field in ("mlflow_run_id", "publisher_version"):
        if not isinstance(marker[field], str) or not marker[field].strip():
            raise ValueError(f"Completion marker {field} is invalid")
    for field in ("pipeline_sha256", "artifact_manifest_sha256"):
        if not SHA256_PATTERN.fullmatch(str(marker[field])):
            raise ValueError(f"Completion marker {field} is invalid")
    _reject_unsafe_metadata(marker)
    _validate_utc_timestamp(
        str(marker["completion_timestamp_utc"]), "Completion marker timestamp"
    )
    canonical_json_bytes(marker)


def verify_completion_marker(
    marker: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    manifest_checksum: str,
) -> None:
    validate_completion_marker(marker)
    comparisons = {
        "model_name": "registered_model_name",
        "model_version": "registered_model_version",
        "model_version_id": "model_version_id",
        "mlflow_run_id": "mlflow_run_id",
        "pipeline_sha256": "pipeline_sha256",
    }
    for marker_key, manifest_key in comparisons.items():
        if str(marker[marker_key]) != str(manifest[manifest_key]):
            raise ValueError(f"Completion marker {marker_key} does not match the manifest")
    if marker["artifact_manifest_sha256"] != manifest_checksum:
        raise ValueError("Completion marker artifact manifest checksum mismatch")


def _reject_non_finite_numbers(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Canonical JSON does not allow NaN or infinity")
    if isinstance(value, Mapping):
        for nested in value.values():
            _reject_non_finite_numbers(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_non_finite_numbers(nested)


def _normalized_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).casefold())


def _reject_excluded_artifact_path(path: str) -> None:
    parts = PurePosixPath(path).parts
    name = parts[-1].casefold()
    if (
        any(part.casefold() in _EXCLUDED_ARTIFACT_PARTS for part in parts)
        or name in _EXCLUDED_ARTIFACT_NAMES
        or name.startswith(".env.")
        or name.endswith((".log", ".pyc", ".pyo"))
    ):
        raise ValueError(f"Transient or sensitive artifact is prohibited: {path}")


def _validate_utc_timestamp(value: str, label: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{label} must be UTC")


def _reject_unsafe_metadata(value: Any, *, parent_key: str = "") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = _normalized_key(key)
            if normalized in _PROHIBITED_METADATA_KEYS:
                raise ValueError(f"Prohibited field in integrity metadata: {key}")
            _reject_unsafe_metadata(nested, parent_key=str(key))
    elif isinstance(value, list):
        for nested in value:
            _reject_unsafe_metadata(nested, parent_key=parent_key)
    elif isinstance(value, str):
        if parent_key == "path":
            return
        if value.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", value):
            raise ValueError("Integrity metadata must not contain absolute local paths")
        if any(pattern.search(value) for pattern in _SECRET_VALUE_PATTERNS):
            raise ValueError("Integrity metadata contains a credential-like value")


def _fallback_schema_validation(manifest: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    """Keep validation fail-closed in minimal environments before dependencies install."""
    if not isinstance(manifest, Mapping):
        raise ValueError("Artifact manifest must be a JSON object")
    required = set(schema["required"])
    missing = sorted(required - set(manifest))
    extra = sorted(set(manifest) - set(schema["properties"]))
    if missing or extra:
        raise ValueError(
            f"Artifact manifest JSON Schema fields are invalid; missing={missing}, extra={extra}"
        )
    if manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported artifact manifest schema version")
    if not SHA256_PATTERN.fullmatch(str(manifest.get("pipeline_sha256", ""))):
        raise ValueError("Artifact manifest pipeline checksum is invalid")
    if not isinstance(manifest.get("source_worktree_dirty"), bool):
        raise ValueError("Artifact manifest dirty-worktree status must be boolean")
