from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import json

import pandas as pd

from src.config import DagsHubSettings
from src.mlops.tracking import ExperimentTracker, configure_tracking_backend
from src.mlops.integrity import REQUIRED_MODEL_FILES, REQUIRED_RUN_ARTIFACTS
from src.mlops.registry import pipeline_checksum


def _training(tmp_path: Path):
    artifact_dir = tmp_path / "training"
    artifact_dir.mkdir()
    (artifact_dir / "model.pkl").write_bytes(b"local-model")

    class Pipeline:
        def predict(self, frame):
            return pd.DataFrame(
                {"predicted_class": [0], "churn_probability": [0.2]}
            )

    return SimpleNamespace(
        artifact_dir=artifact_dir,
        model_name="logistic_regression",
        model_parameters={"C": 1.0},
        threshold=0.5,
        validation_metrics={"roc_auc": 0.8, "ignored": None},
        test_metrics={"roc_auc": 0.75},
        candidate_metrics={
            "logistic_regression": {"roc_auc": 0.8},
            "decision_tree": {"roc_auc": 0.7},
        },
        pipeline=Pipeline(),
    )


def _config():
    return {
        "version": "1.0.0",
        "dataset": {"name": "churn", "source_identity": "repository:data.csv"},
        "split": {"random_seed": 42, "test_size": 0.2, "validation_size": 0.2},
        "model": {
            "selection_metric": "roc_auc",
            "candidates": {
                "logistic_regression": {"parameters": {"C": 1.0}},
                "decision_tree": {"parameters": {"max_depth": 5}},
            },
        },
        "contracts": {
            "feature_schema_version": "1.0.0",
            "prediction_contract_version": "1.0.0",
            "prediction_horizon_version": "90-days-v1",
            "identity_privacy_contract_version": "1.0.0",
            "monitoring_contract_version": "1.0.0",
        },
    }


def test_disabled_tracking_does_not_import_mlflow(tmp_path, monkeypatch):
    def fail_import(name):
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr("src.mlops.tracking.importlib.import_module", fail_import)
    settings = DagsHubSettings(enabled=False, register_model=False)
    result = ExperimentTracker(settings).track(_training(tmp_path), _config())
    assert result.status == "disabled"
    assert result.selected_model == "logistic_regression"
    assert result.candidate_scores == {
        "logistic_regression": 0.8,
        "decision_tree": 0.7,
    }
    assert result.run_id is None


def test_dagshub_opens_one_run_and_logs_local_artifacts(tmp_path, monkeypatch):
    calls = []
    dagshub = SimpleNamespace(
        init=lambda **kwargs: calls.append(("dagshub.init", kwargs))
    )

    @contextmanager
    def start_run(**kwargs):
        calls.append(("start_run", kwargs))
        yield SimpleNamespace(info=SimpleNamespace(run_id="run-1"))
        calls.append(("end_run", {}))

    mlflow = SimpleNamespace(
        set_experiment=lambda name: calls.append(("set_experiment", name)),
        start_run=start_run,
        log_params=lambda values: calls.append(("params", values)),
        log_metrics=lambda values: calls.append(("metrics", values)),
        set_tags=lambda values: calls.append(("tags", values)),
        log_artifacts=lambda path: calls.append(("artifacts", path)),
        models=SimpleNamespace(infer_signature=lambda inputs, outputs: "signature"),
        sklearn=SimpleNamespace(
            SERIALIZATION_FORMAT_CLOUDPICKLE="cloudpickle",
            log_model=lambda **kwargs: (
                calls.append(("model", kwargs))
                or SimpleNamespace(model_uri="models:/m-123")
            ),
        ),
    )

    def import_module(name):
        return dagshub if name == "dagshub" else mlflow

    monkeypatch.setattr("src.mlops.tracking.importlib.import_module", import_module)
    monkeypatch.setenv("DAGSHUB_USER_TOKEN", "existing-token")
    settings = DagsHubSettings(
        enabled=True,
        register_model=False,
        dagshub_repo_owner="owner",
        dagshub_repo_name="repo",
        dagshub_token="secret-token",
    )
    result = ExperimentTracker(settings).track(_training(tmp_path), _config())

    assert result.status == "tracked"
    assert calls[0] == (
        "dagshub.init",
        {"repo_owner": "owner", "repo_name": "repo", "mlflow": True},
    )
    assert calls[1][0] == "set_experiment"
    assert __import__("os").environ["DAGSHUB_USER_TOKEN"] == "existing-token"
    assert [name for name, _ in calls].count("start_run") == 1
    assert [name for name, _ in calls].count("end_run") == 1
    assert any(name == "artifacts" for name, _ in calls)
    model_call = next(value for name, value in calls if name == "model")
    assert model_call["name"] == "model"
    assert "artifact_path" not in model_call


def test_tracking_failure_is_reported_without_exposing_secret(tmp_path, monkeypatch):
    def fail_import(name):
        raise RuntimeError("token=very-secret")

    monkeypatch.setattr("src.mlops.tracking.importlib.import_module", fail_import)
    settings = DagsHubSettings(
        enabled=True,
        register_model=False,
        dagshub_repo_owner="owner",
        dagshub_repo_name="repo",
    )
    result = ExperimentTracker(settings).track(_training(tmp_path), _config())
    assert result.status == "failed"
    assert result.run_id is None
    assert "very-secret" not in result.warning


def test_dagshub_token_is_copied_only_when_user_token_is_absent(monkeypatch):
    dagshub = SimpleNamespace(init=lambda **kwargs: None)
    monkeypatch.setattr(
        "src.mlops.tracking.importlib.import_module", lambda name: dagshub
    )
    monkeypatch.delenv("DAGSHUB_USER_TOKEN", raising=False)
    settings = DagsHubSettings(
        enabled=True,
        register_model=False,
        dagshub_repo_owner="owner",
        dagshub_repo_name="repo",
        dagshub_token="secret-token",
    )
    configure_tracking_backend(settings)
    assert __import__("os").environ["DAGSHUB_USER_TOKEN"] == "secret-token"


def test_explicit_production_mode_registers_an_exact_version(tmp_path, monkeypatch):
    model_dir = tmp_path / "mlflow-model"
    model_dir.mkdir()
    (model_dir / "model.pkl").write_bytes(b"registered-model")

    class Client:
        def set_tag(self, *args):
            pass

        def set_model_version_tag(self, *args):
            pass

    def download_artifacts(**kwargs):
        assert kwargs["artifact_uri"] == "models:/m-123"
        return str(model_dir)

    def register_model(**kwargs):
        assert kwargs["model_uri"] == "models:/m-123"
        return SimpleNamespace(version="7")

    mlflow = SimpleNamespace(
        artifacts=SimpleNamespace(download_artifacts=download_artifacts),
        tracking=SimpleNamespace(MlflowClient=Client),
        register_model=register_model,
        sklearn=SimpleNamespace(
            load_model=lambda uri: SimpleNamespace(predict=lambda frame: [0])
        ),
    )
    settings = DagsHubSettings(
        enabled=True,
        register_model=True,
        dagshub_repo_owner="owner",
        dagshub_repo_name="repo",
    )
    tracker = ExperimentTracker(settings)
    monkeypatch.setattr(tracker, "_publish_integrity", lambda **kwargs: "b" * 64)
    result = tracker._register(
        mlflow,
        "models:/m-123",
        "run-1",
        _training(tmp_path),
        _config(),
        {
            "source_commit_sha": "abc123",
            "source_branch": "release-7",
            "source_worktree_dirty": "false",
            "python_version": "3.12.0",
        },
    )
    assert result.status == "registered"
    assert result.registered_model_version == "7"
    assert result.model_version_id == "dagshub:owner/repo:churn_predictor:7"
    assert result.artifact_manifest_sha256 == "b" * 64
    assert result.integrity_status == "complete"


def test_integrity_publication_logs_completion_marker_last(tmp_path):
    run_root = tmp_path / "run"
    model_dir = tmp_path / "mlflow-model"
    for relative in sorted(REQUIRED_RUN_ARTIFACTS):
        path = run_root.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    identities = {
        cohort: {
            "dataset_name": "churn",
            "source_identity": "repository:data.csv",
            "dataset_digest": character * 64,
            "row_count": 10,
            "feature_list": ["CreditScore"],
            "target_column": "Exited",
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
    for relative in sorted(REQUIRED_MODEL_FILES):
        path = tmp_path.joinpath("mlflow-model", *relative.split("/")[1:])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"protected:{relative}".encode())

    events = []

    class Client:
        def __init__(self):
            self.artifacts = {}

        def log_artifact(self, run_id, path, artifact_path=None):
            events.append(("artifact", Path(path).name, artifact_path))
            self.artifacts[f"{artifact_path}/{Path(path).name}"] = Path(path)

        def download_artifacts(self, run_id, path, destination):
            return str(self.artifacts[path])

        def set_tag(self, run_id, key, value):
            events.append(("run_tag", key, value))

        def set_model_version_tag(self, name, version, key, value):
            events.append(("version_tag", key, value))

    def download_run_artifacts(**kwargs):
        assert kwargs["run_id"] == "run-1"
        assert kwargs["artifact_path"] == ""
        assert Path(kwargs["dst_path"]).name == "run"
        return str(run_root)

    mlflow = SimpleNamespace(
        __version__="3.1.0",
        artifacts=SimpleNamespace(download_artifacts=download_run_artifacts),
        sklearn=SimpleNamespace(
            load_model=lambda path: SimpleNamespace(predict=lambda frame: [0])
        ),
    )
    settings = DagsHubSettings(
        enabled=True,
        register_model=True,
        dagshub_repo_owner="owner",
        dagshub_repo_name="repo",
    )
    tracker = ExperimentTracker(settings)
    digest = tracker._publish_integrity(
        mlflow=mlflow,
        client=Client(),
        model_uri="models:/m-123",
        model_dir=model_dir,
        run_id="run-1",
        version="7",
        identity="dagshub:owner/repo:churn_predictor:7",
        checksum=pipeline_checksum(model_dir),
        training=_training(tmp_path),
        config=_config(),
        lineage={
            "source_commit_sha": "abc123",
            "source_branch": "release-7",
            "source_worktree_dirty": "false",
            "python_version": "3.12.0",
        },
    )

    artifact_events = [event for event in events if event[0] == "artifact"]
    assert [event[1] for event in artifact_events] == [
        "artifact_manifest.json",
        "publication_complete.json",
    ]
    assert events.index(artifact_events[-1]) < events.index(
        ("version_tag", "integrity_status", "complete")
    )
    assert len(digest) == 64
