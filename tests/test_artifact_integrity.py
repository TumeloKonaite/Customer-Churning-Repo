from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.mlops.integrity import (
    REQUIRED_MODEL_FILES,
    REQUIRED_RUN_ARTIFACTS,
    artifact_manifest_checksum,
    build_artifact_manifest,
    build_completion_marker,
    build_protected_artifact_entries,
    canonical_json_bytes,
    normalize_artifact_path,
    validate_artifact_manifest,
    validate_publication_source_artifacts,
    validate_protected_artifact_entries,
    verify_completion_marker,
    verify_protected_artifacts,
)
from src.mlops.registry import ValidationResult


def _bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    for relative in sorted(REQUIRED_RUN_ARTIFACTS | REQUIRED_MODEL_FILES):
        path = root.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"protected:{relative}".encode())
    (root / "model" / "serving_input_example.json").write_text(
        "{}", encoding="utf-8"
    )
    return root


def _manifest(tmp_path: Path) -> tuple[Path, dict]:
    root = _bundle(tmp_path)
    entries = build_protected_artifact_entries(root)
    manifest = build_artifact_manifest(
        repository_owner="owner",
        repository_name="repository",
        experiment_name="customer-churn-production",
        run_id="run-1",
        model_name="churn_predictor",
        model_version="7",
        model_version_id="dagshub:owner/repository:churn_predictor:7",
        source_commit_sha="abc123",
        source_branch="release-7",
        source_worktree_dirty=False,
        selected_classifier_name="logistic_regression",
        selection_metric="roc_auc",
        feature_schema_version="1.0.0",
        prediction_contract_version="1.0.0",
        prediction_horizon_contract_version="90-days-v1",
        identity_privacy_contract_version="1.0.0",
        monitoring_contract_version="1.0.0",
        training_dataset_identity={
            "dataset_name": "bank_customer_churn",
            "source_identity": "repository:dataset/Churn_Modelling.csv",
            "dataset_digest": "d" * 64,
            "row_count": 100,
            "feature_list": ["CreditScore", "Age"],
            "target_column": "Exited",
        },
        classification_threshold=0.5,
        positive_class=1,
        python_version="3.12.3",
        scikit_learn_version="1.7.0",
        mlflow_version="3.1.0",
        pipeline_sha256=next(
            entry["sha256"] for entry in entries if entry["path"] == "model/model.pkl"
        ),
        protected_artifacts=entries,
        publication_timestamp_utc="2026-08-20T12:00:00Z",
    )
    return root, manifest


def test_canonical_json_is_deterministic_compact_utf8_and_rejects_non_finite():
    first = {"z": "München", "a": {"y": 2, "x": 1}}
    second = {"a": {"x": 1, "y": 2}, "z": "München"}

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert canonical_json_bytes(first) == (
        '{"a":{"x":1,"y":2},"z":"München"}'.encode("utf-8")
    )
    with pytest.raises(ValueError, match="NaN or infinity"):
        canonical_json_bytes({"unsafe": float("nan")})


def test_artifact_entries_are_sorted_and_record_size_and_sha256(tmp_path):
    root = _bundle(tmp_path)
    entries = build_protected_artifact_entries(root)

    assert [entry["path"] for entry in entries] == sorted(
        entry["path"] for entry in entries
    )
    pipeline = next(entry for entry in entries if entry["path"] == "model/model.pkl")
    assert pipeline["size_bytes"] == (root / "model" / "model.pkl").stat().st_size
    assert len(pipeline["sha256"]) == 64
    assert pipeline["role"] == "fitted_pipeline"


def test_path_normalization_is_repository_independent_and_rejects_unsafe_paths():
    assert normalize_artifact_path(r"contracts\feature_schema.json") == (
        "contracts/feature_schema.json"
    )
    for path in ("/tmp/model.pkl", r"C:\model.pkl", "model/../secret", "model//file"):
        with pytest.raises(ValueError, match="relative|normalized"):
            normalize_artifact_path(path)


def test_manifest_schema_rejects_duplicates_secrets_and_self_checksum(tmp_path):
    _, manifest = _manifest(tmp_path)
    validate_artifact_manifest(manifest)

    duplicate = deepcopy(manifest)
    duplicate["protected_artifacts"].append(
        deepcopy(duplicate["protected_artifacts"][0])
    )
    with pytest.raises(ValueError, match="stable path ordering|unique"):
        validate_artifact_manifest(duplicate)

    secret = deepcopy(manifest)
    secret["dagshub_token"] = "very-secret"
    with pytest.raises(ValueError, match="Schema|fields|Prohibited"):
        validate_artifact_manifest(secret)

    self_referential = deepcopy(manifest)
    self_referential["artifact_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="own checksum"):
        artifact_manifest_checksum(self_referential)


def test_publication_source_rejects_transient_files_and_secret_metadata(tmp_path):
    root = tmp_path / "source"
    (root / "lineage").mkdir(parents=True)
    (root / "model.pkl").write_bytes(b"model")
    (root / "lineage" / "training_config.json").write_text(
        json.dumps({"dataset": {"path": "dataset/train.csv"}}), encoding="utf-8"
    )
    validate_publication_source_artifacts(root)

    (root / ".env").write_text("DAGSHUB_TOKEN=secret", encoding="utf-8")
    with pytest.raises(ValueError, match="Transient or sensitive"):
        validate_publication_source_artifacts(root)
    (root / ".env").unlink()

    (root / "lineage" / "training_config.json").write_text(
        json.dumps({"dagshub_token": "secret"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Prohibited field"):
        validate_publication_source_artifacts(root)

    (root / "lineage" / "training_config.json").write_text("{}", encoding="utf-8")
    (root / "integrity").mkdir()
    (root / "integrity" / "publication_complete.json").write_text(
        "{}", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="must not contain integrity"):
        validate_publication_source_artifacts(root)


def test_protected_artifact_verification_detects_size_and_checksum_changes(tmp_path):
    root, manifest = _manifest(tmp_path)
    resolve = lambda path: root.joinpath(*path.split("/"))
    verify_protected_artifacts(manifest, resolve)

    target = root / "contracts" / "feature_schema.json"
    original = target.read_bytes()
    target.write_bytes(original + b"changed")
    with pytest.raises(ValueError, match="size mismatch"):
        verify_protected_artifacts(manifest, resolve)

    target.write_bytes(b"x" * len(original))
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_protected_artifacts(manifest, resolve)


def test_required_artifact_and_completion_identity_fail_closed(tmp_path):
    root, manifest = _manifest(tmp_path)
    entries = deepcopy(manifest["protected_artifacts"])
    entries = [entry for entry in entries if entry["path"] != "evaluation/metrics.json"]
    with pytest.raises(ValueError, match="missing required artifacts"):
        validate_protected_artifact_entries(entries)

    checksum = artifact_manifest_checksum(manifest)
    marker = build_completion_marker(
        model_name="churn_predictor",
        model_version="7",
        model_version_id="dagshub:owner/repository:churn_predictor:7",
        run_id="run-1",
        pipeline_sha256=manifest["pipeline_sha256"],
        artifact_manifest_sha256=checksum,
        completion_timestamp_utc="2026-08-20T12:01:00Z",
    )
    verify_completion_marker(marker, manifest, manifest_checksum=checksum)

    wrong = {**marker, "mlflow_run_id": "run-2"}
    with pytest.raises(ValueError, match="mlflow_run_id"):
        verify_completion_marker(wrong, manifest, manifest_checksum=checksum)
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_completion_marker(marker, manifest, manifest_checksum="0" * 64)


def test_prepare_deployment_requires_and_pins_manifest_checksum(tmp_path, monkeypatch):
    import src.mlops.deployment as deployment

    checksum = "b" * 64
    validation = ValidationResult(
        valid=True,
        integrity_status="complete",
        model_name="churn_predictor",
        model_version="7",
        model_version_id="dagshub:owner/repository:churn_predictor:7",
        mlflow_run_id="run-1",
        pipeline_sha256="a" * 64,
        artifact_manifest_sha256=checksum,
        manifest_schema_version="1.0.0",
        feature_schema_version="1.0.0",
        source_commit_sha="abc123",
        duration_seconds=0.1,
    )
    calls = []

    def validate(*args, **kwargs):
        calls.append(kwargs)
        return validation

    def write(destination, **kwargs):
        return {
            "artifact_manifest_sha256": kwargs[
                "validation"
            ].artifact_manifest_sha256,
            "integrity_status": "complete",
        }

    monkeypatch.setattr(deployment, "validate_registered_model", validate)
    monkeypatch.setattr(deployment, "_write_package", write)
    monkeypatch.setattr(
        deployment, "validate_packaged_model", lambda package, expected=None: expected
    )

    output = tmp_path / "package"
    metadata = deployment.prepare_deployment(
        "models:/churn_predictor/7",
        output,
        expected_artifact_manifest_sha256=checksum,
    )
    assert metadata["artifact_manifest_sha256"] == checksum
    assert calls[0]["expected_artifact_manifest_sha256"] == checksum

    with pytest.raises(ValueError, match="EXPECTED_ARTIFACT_MANIFEST_SHA256"):
        deployment.prepare_deployment(
            "models:/churn_predictor/7", tmp_path / "missing-pin"
        )

    monkeypatch.setattr(
        deployment,
        "validate_registered_model",
        lambda *args, **kwargs: ValidationResult(
            **{
                **validation.as_dict(),
                "artifact_manifest_sha256": "c" * 64,
            }
        ),
    )
    with pytest.raises(ValueError, match="does not match"):
        deployment.prepare_deployment(
            "models:/churn_predictor/7",
            tmp_path / "wrong-pin",
            expected_artifact_manifest_sha256=checksum,
        )
