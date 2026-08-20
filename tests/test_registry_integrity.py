from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pandas as pd
import pytest

from src.config import DagsHubSettings
from src.mlops.integrity import (
    REQUIRED_MODEL_FILES,
    REQUIRED_RUN_ARTIFACTS,
    artifact_manifest_checksum,
    build_artifact_manifest,
    build_completion_marker,
    build_protected_artifact_entries,
    canonical_json_bytes,
)
from src.mlops.registry import pipeline_checksum, validate_registered_model
from src.model_schema import CANONICAL_FEATURE_ORDER, TARGET_COLUMN, build_model_schema


def _publication(tmp_path: Path):
    run_root = tmp_path / "run"
    model_dir = tmp_path / "model"
    for relative in sorted(REQUIRED_RUN_ARTIFACTS):
        path = run_root.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    for relative in (
        "evaluation/confusion_matrix.json",
        "evaluation/classification_report.json",
    ):
        path = run_root.joinpath(*relative.split("/"))
        path.write_text("{}", encoding="utf-8")
    for relative in sorted(REQUIRED_MODEL_FILES):
        path = model_dir.joinpath(*relative.split("/")[1:])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"protected:{relative}".encode())

    feature_schema = build_model_schema()
    (run_root / "contracts" / "feature_schema.json").write_text(
        json.dumps(feature_schema), encoding="utf-8"
    )
    (run_root / "contracts" / "prediction_contract.json").write_text(
        json.dumps(
            {
                "version": "1.0.0",
                "positive_class": 1,
                "classification_threshold": 0.5,
                "prediction_horizon_version": "90-days-v1",
            }
        ),
        encoding="utf-8",
    )
    (run_root / "contracts" / "privacy_contract.json").write_text(
        json.dumps({"version": "1.0.0", "prohibited_columns": ["CustomerId"]}),
        encoding="utf-8",
    )
    (run_root / "contracts" / "monitoring_contract.json").write_text(
        json.dumps({"version": "1.0.0"}), encoding="utf-8"
    )
    identities = {
        cohort: {
            "dataset_name": "churn",
            "source_identity": "repository:data.csv",
            "dataset_digest": character * 64,
            "row_count": 2,
            "feature_list": CANONICAL_FEATURE_ORDER,
            "target_column": TARGET_COLUMN,
        }
        for cohort, character in (
            ("training", "a"),
            ("validation", "b"),
            ("evaluation", "c"),
        )
    }
    (run_root / "lineage" / "dataset_identities.json").write_text(
        json.dumps(identities), encoding="utf-8"
    )
    (run_root / "references" / "drift_reference_metadata.json").write_text(
        json.dumps({"dataset_purpose": "drift_reference"}), encoding="utf-8"
    )
    (run_root / "references" / "evaluation_reference_metadata.json").write_text(
        json.dumps({"dataset_purpose": "evaluation_reference"}), encoding="utf-8"
    )
    row = {name: 1 for name in CANONICAL_FEATURE_ORDER}
    pd.DataFrame([row, row], columns=CANONICAL_FEATURE_ORDER).to_parquet(
        run_root / "references" / "drift_reference.parquet", index=False
    )
    pd.DataFrame(
        [{**row, TARGET_COLUMN: 0}, {**row, TARGET_COLUMN: 1}],
        columns=CANONICAL_FEATURE_ORDER + [TARGET_COLUMN],
    ).to_parquet(run_root / "references" / "evaluation_reference.parquet", index=False)

    bundle = tmp_path / "bundle"
    shutil.copytree(run_root, bundle)
    shutil.copytree(model_dir, bundle / "model")
    pipeline_sha = pipeline_checksum(model_dir)
    manifest = build_artifact_manifest(
        repository_owner="owner",
        repository_name="repo",
        experiment_name="customer-churn-production",
        run_id="run-1",
        model_name="churn_predictor",
        model_version="7",
        model_version_id="dagshub:owner/repo:churn_predictor:7",
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
        training_dataset_identity=identities["training"],
        classification_threshold=0.5,
        positive_class=1,
        python_version="3.12.3",
        scikit_learn_version="1.7.0",
        mlflow_version="3.1.0",
        pipeline_sha256=pipeline_sha,
        protected_artifacts=build_protected_artifact_entries(bundle),
        publication_timestamp_utc="2026-08-20T12:00:00Z",
    )
    manifest_sha = artifact_manifest_checksum(manifest)
    marker = build_completion_marker(
        model_name="churn_predictor",
        model_version="7",
        model_version_id="dagshub:owner/repo:churn_predictor:7",
        run_id="run-1",
        pipeline_sha256=pipeline_sha,
        artifact_manifest_sha256=manifest_sha,
        completion_timestamp_utc="2026-08-20T12:01:00Z",
    )
    integrity = run_root / "integrity"
    integrity.mkdir()
    (integrity / "artifact_manifest.json").write_bytes(canonical_json_bytes(manifest))
    (integrity / "publication_complete.json").write_bytes(canonical_json_bytes(marker))
    return run_root, model_dir, manifest_sha, pipeline_sha


class FakeClient:
    def __init__(self, run_root, model_dir, manifest_sha, pipeline_sha):
        self.run_root = run_root
        self.model_dir = model_dir
        self.version = SimpleNamespace(
            name="churn_predictor",
            version="7",
            run_id="run-1",
            source="model-source",
            tags={
                "source_commit_sha": "abc123",
                "training_run_id": "run-1",
                "feature_schema_version": "1.0.0",
                "prediction_contract_version": "1.0.0",
                "prediction_horizon_contract_version": "90-days-v1",
                "identity_privacy_contract_version": "1.0.0",
                "monitoring_contract_version": "1.0.0",
                "positive_class": "1",
                "classification_threshold": "0.5",
                "validation_status": "validated",
                "pipeline_sha256": pipeline_sha,
                "integrity_status": "complete",
                "artifact_manifest_schema_version": "1.0.0",
                "artifact_manifest_sha256": manifest_sha,
            },
        )
        metrics = {
            f"{cohort}/{metric}": 1.0
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
        }
        self.run = SimpleNamespace(
            info=SimpleNamespace(experiment_id="experiment-1"),
            data=SimpleNamespace(
                params={
                    "model_type": "logistic_regression",
                    "selection_metric": "roc_auc",
                    "dataset_name": "churn",
                    "dataset_source": "repository:data.csv",
                    "test_size": "0.2",
                    "validation_size": "0.2",
                    "classification_threshold": "0.5",
                    "random_seed": "42",
                    "training_configuration_version": "1.0.0",
                    "source_commit_sha": "abc123",
                },
                metrics=metrics,
                tags={
                    "source_branch": "release-7",
                    "source_worktree_dirty": "false",
                    "pipeline_sha256": pipeline_sha,
                    "artifact_manifest_sha256": manifest_sha,
                    "integrity_status": "complete",
                },
            ),
        )

    def get_model_version(self, name, version):
        return self.version

    def get_run(self, run_id):
        return self.run

    def get_experiment(self, experiment_id):
        return SimpleNamespace(name="customer-churn-production")

    def download_artifacts(self, run_id, path, destination):
        artifact = self.run_root.joinpath(*path.split("/"))
        if not artifact.exists():
            raise FileNotFoundError(path)
        return str(artifact)

    def list_artifacts(self, run_id, prefix=None):
        base = self.run_root if prefix is None else self.run_root.joinpath(*prefix.split("/"))
        result = []
        for path in sorted(base.iterdir()):
            relative = path.relative_to(self.run_root).as_posix()
            result.append(SimpleNamespace(path=relative, is_dir=path.is_dir()))
        return result


def test_integrity_validator_checks_everything_before_loading(tmp_path, monkeypatch):
    import src.mlops.registry as registry

    run_root, model_dir, manifest_sha, pipeline_sha = _publication(tmp_path)
    client = FakeClient(run_root, model_dir, manifest_sha, pipeline_sha)
    load_calls = []
    monkeypatch.setattr(registry, "configure_mlflow", lambda settings: client)
    monkeypatch.setattr(
        registry.mlflow.artifacts,
        "download_artifacts",
        lambda **kwargs: str(model_dir),
    )
    monkeypatch.setattr(
        registry, "_validate_mlflow_model", lambda path: load_calls.append(path)
    )
    settings = DagsHubSettings(
        enabled=True,
        dagshub_repo_owner="owner",
        dagshub_repo_name="repo",
    )

    result = validate_registered_model(
        "models:/churn_predictor/7",
        settings=settings,
        expected_pipeline_sha256=pipeline_sha,
        expected_artifact_manifest_sha256=manifest_sha,
    )

    assert result.valid is True
    assert result.integrity_status == "complete"
    assert result.artifact_manifest_sha256 == manifest_sha
    assert load_calls == [model_dir]

    marker_path = run_root / "integrity" / "publication_complete.json"
    marker_bytes = marker_path.read_bytes()
    marker_path.unlink()
    load_calls.clear()
    with pytest.raises(ValueError, match="publication_complete.json"):
        validate_registered_model(
            "models:/churn_predictor/7", settings=settings
        )
    assert load_calls == []
    marker_path.write_bytes(marker_bytes)

    client.version.tags["integrity_status"] = "incomplete"
    with pytest.raises(ValueError, match="not integrity-complete"):
        validate_registered_model(
            "models:/churn_predictor/7", settings=settings
        )
    client.version.tags["integrity_status"] = "complete"

    (model_dir / "requirements.txt").write_bytes(b"same-size-corruption")
    load_calls.clear()
    with pytest.raises(ValueError, match="size mismatch|checksum mismatch"):
        validate_registered_model(
            "models:/churn_predictor/7", settings=settings
        )
    assert load_calls == []


def test_legacy_version_is_explicit_and_not_integrity_verified(tmp_path, monkeypatch):
    import src.mlops.registry as registry

    run_root, model_dir, manifest_sha, pipeline_sha = _publication(tmp_path)
    shutil.rmtree(run_root / "integrity")
    client = FakeClient(run_root, model_dir, manifest_sha, pipeline_sha)
    for key in (
        "integrity_status",
        "artifact_manifest_schema_version",
        "artifact_manifest_sha256",
    ):
        client.version.tags.pop(key)
    monkeypatch.setattr(registry, "configure_mlflow", lambda settings: client)
    monkeypatch.setattr(
        registry.mlflow.artifacts,
        "download_artifacts",
        lambda **kwargs: str(model_dir),
    )
    monkeypatch.setattr(registry, "_validate_mlflow_model", lambda path: None)
    settings = DagsHubSettings(
        enabled=True,
        dagshub_repo_owner="owner",
        dagshub_repo_name="repo",
    )

    result = validate_registered_model(
        "models:/churn_predictor/7", settings=settings
    )

    assert result.valid is True
    assert result.integrity_status == "legacy"
    assert result.artifact_manifest_sha256 is None
    assert result.manifest_schema_version is None
