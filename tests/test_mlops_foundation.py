import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from src.config import (
    DatabaseSettings,
    DeploymentSettings,
    Environment,
    DagsHubSettings,
    safe_error_message,
)
from src.database import connection
from src.mlops.registry import exact_model_uri, model_version_id, parse_exact_model_uri
from src.model_schema import (
    CANONICAL_FEATURE_ORDER,
    TARGET_COLUMN,
    build_model_schema,
    prohibited_columns,
    reject_prohibited_columns,
)
from src.schemas.prediction import SINGLE_PREDICTION_EXAMPLE
from src.services import model_service


def test_tracking_settings_are_optional_and_redact_secrets(monkeypatch):
    for name in (
        "ENABLE_DAGSHUB_TRACKING",
        "MLFLOW_EXPERIMENT_NAME",
        "DAGSHUB_REPO_OWNER",
        "DAGSHUB_REPO_NAME",
        "DAGSHUB_TOKEN",
        "ENABLE_MODEL_REGISTRATION",
        "MLFLOW_REGISTERED_MODEL_NAME",
    ):
        monkeypatch.delenv(name, raising=False)

    assert DagsHubSettings().enabled is False
    settings = DagsHubSettings(
        enabled=True,
        dagshub_repo_owner="owner",
        dagshub_repo_name="repository",
        dagshub_token="dagshub-token",
    )
    assert settings.registered_model_name == "churn_predictor"
    assert "dagshub-token" not in repr(settings)

    with pytest.raises(ValidationError, match="ENABLE_DAGSHUB_TRACKING"):
        DagsHubSettings(register_model=True)


def test_error_messages_redact_url_credentials_and_tokens():
    message = safe_error_message(
        ValueError(
            "failed postgresql+psycopg://app:database-secret@host/db "
            "Bearer abc.def and token=dagshub-secret"
        )
    )
    assert "database-secret" not in message
    assert "abc.def" not in message
    assert "dagshub-secret" not in message


def test_production_database_requires_remote_postgres_and_ssl():
    valid = DatabaseSettings(
        environment=Environment.PRODUCTION,
        database_url="postgresql+psycopg://app:secret@pool.neon.tech/churn_monitoring?sslmode=require",
    )
    assert valid.pool_size == 2
    assert "secret" not in repr(valid)

    for url in (
        None,
        "sqlite:///local.db",
        "postgresql+psycopg://app:secret@localhost/churn_monitoring?sslmode=require",
        "postgresql+psycopg://app:secret@pool.neon.tech/churn_monitoring",
    ):
        with pytest.raises(ValidationError):
            DatabaseSettings(environment=Environment.PRODUCTION, database_url=url)


def test_exact_model_identity_rejects_latest_aliases_and_stages():
    assert parse_exact_model_uri("models:/churn_predictor/7").version == "7"
    assert exact_model_uri("churn_predictor", 7) == "models:/churn_predictor/7"
    assert (
        model_version_id("owner", "repo", "churn_predictor", 7)
        == "dagshub:owner/repo:churn_predictor:7"
    )
    for uri in (
        "models:/churn_predictor/latest",
        "models:/churn_predictor@champion",
        "models:/churn_predictor/Production",
        "models:/churn_predictor/0",
    ):
        with pytest.raises(ValueError, match="exact positive numeric version"):
            parse_exact_model_uri(uri)


def test_deployment_settings_require_consistent_exact_identity():
    values = {
        "environment": "production",
        "model_name": "churn_predictor",
        "model_version": "7",
        "expected_run_id": "run-1",
        "expected_model_version_id": "dagshub:owner/repo:churn_predictor:7",
        "expected_pipeline_sha256": "a" * 64,
        "expected_artifact_manifest_sha256": "b" * 64,
    }
    assert DeploymentSettings(**values).model_version == "7"
    with pytest.raises(ValidationError):
        DeploymentSettings(**{**values, "model_version": "latest"})
    with pytest.raises(ValidationError):
        DeploymentSettings(
            **{**values, "expected_model_version_id": "dagshub:owner/repo:churn_predictor:8"}
        )


def test_prohibited_columns_are_case_insensitive_and_cover_aliases():
    assert prohibited_columns(["CUSTOMER_ID", "EmailAddress", "safe_feature"]) == [
        "CUSTOMER_ID",
        "EmailAddress",
    ]
    with pytest.raises(ValueError, match="Prohibited identifier"):
        reject_prohibited_columns(["teLEPhone"])
    assert list(SINGLE_PREDICTION_EXAMPLE) == CANONICAL_FEATURE_ORDER


def test_model_trainer_owns_fit_evaluation_and_local_artifacts(tmp_path):
    from mlflow.models import infer_signature

    from src.components.data_ingestion import DatasetCohorts
    from src.components.model_trainer import ModelTrainer

    first = dict(SINGLE_PREDICTION_EXAMPLE)
    second = {**first, "Geography": "Germany", "Gender": "Male", "Age": 55}
    features = pd.DataFrame([first, second, first, second])
    cohort = features.assign(Exited=[0, 1, 0, 1])
    result = ModelTrainer().train(
        DatasetCohorts(train=cohort, validation=cohort, test=cohort),
        {
            "selection_metric": "roc_auc",
            "candidates": {
                "logistic_regression": {"parameters": {"max_iter": 1000}},
                "decision_tree": {"parameters": {"max_depth": 2}},
                "random_forest": {"parameters": {"n_estimators": 10}},
                "gradient_boosting": {"parameters": {"n_estimators": 10}},
            },
            "classification_threshold": 0.5,
        },
        {"minimum_validation_roc_auc": 0.0, "minimum_test_roc_auc": 0.0},
        random_seed=42,
        output_dir=tmp_path / "training",
        training_config={"dataset": {"name": "test", "source_identity": "test:data"}},
    )
    example = pd.DataFrame(
        [SINGLE_PREDICTION_EXAMPLE], columns=CANONICAL_FEATURE_ORDER
    )
    signature = infer_signature(example, result.pipeline.predict(example))
    assert set(result.candidate_metrics) == set(ModelTrainer.CLASSIFIERS)
    assert result.model_name == max(
        result.candidate_metrics,
        key=lambda name: result.candidate_metrics[name]["roc_auc"],
    )
    assert "roc_auc" in result.validation_metrics
    assert "roc_auc" in result.test_metrics
    assert (result.artifact_dir / "model.pkl").is_file()
    assert (result.artifact_dir / "evaluation" / "metrics.json").is_file()
    assert (result.artifact_dir / "evaluation" / "model_comparison.json").is_file()
    assert (result.artifact_dir / "references" / "drift_reference.parquet").is_file()
    assert [column.name for column in signature.outputs.inputs] == [
        "predicted_class",
        "churn_probability",
    ]


def _build_package(tmp_path: Path) -> tuple[Path, dict]:
    from src.mlops.registry import pipeline_checksum

    package = tmp_path / "package"
    model_dir = package / "model"
    model_dir.parent.mkdir(parents=True)
    model_dir.mkdir()
    (model_dir / "MLmodel").write_text("flavors: {}\n")
    (model_dir / "model.pkl").write_bytes(b"trusted-serialized-pipeline")
    schema = build_model_schema()
    (package / "feature_schema.json").write_text(json.dumps(schema))
    checksum = pipeline_checksum(model_dir)
    metadata = {
        "deployment_id": "deployment-1",
        "environment": "production",
        "deployment_timestamp_utc": "2026-08-20T00:00:00+00:00",
        "modal_application": "customer-churn-backend",
        "model_name": "churn_predictor",
        "model_version": "7",
        "model_version_id": "dagshub:owner/repo:churn_predictor:7",
        "mlflow_run_id": "run-1",
        "source_commit_sha": "abc123",
        "pipeline_sha256": checksum,
        "artifact_manifest_sha256": "b" * 64,
        "integrity_status": "complete",
        "application_version": "0.1.0",
        "feature_schema_version": schema["schema_version"],
        "validation_status": "validated",
    }
    (package / "deployment_metadata.json").write_text(json.dumps(metadata))
    return package, metadata


def _patch_smoke_load(monkeypatch):
    import src.mlops.deployment as deployment

    example = pd.DataFrame([SINGLE_PREDICTION_EXAMPLE])

    class MLflowModel:
        def load_input_example(self, path):
            return example

    class ModelLoader:
        @staticmethod
        def load(path):
            return MLflowModel()

    class Pipeline:
        def predict(self, frame):
            assert list(frame.columns) == CANONICAL_FEATURE_ORDER
            return [0]

    monkeypatch.setattr(deployment, "Model", ModelLoader)
    monkeypatch.setattr(deployment.mlflow.sklearn, "load_model", lambda path: Pipeline())


def test_packaged_model_round_trip_identity_and_checksum(tmp_path, monkeypatch):
    from src.mlops.deployment import load_deployment_metadata, validate_packaged_model

    package, metadata = _build_package(tmp_path)
    _patch_smoke_load(monkeypatch)
    result = validate_packaged_model(package, expected=metadata)
    assert result["model_version_id"] == metadata["model_version_id"]
    assert result["startup_validation_duration_seconds"] >= 0
    assert load_deployment_metadata(package)["deployment_id"] == "deployment-1"
    assert result["artifact_manifest_sha256"] == "b" * 64

    metadata["pipeline_sha256"] = "0" * 64
    (package / "deployment_metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_packaged_model(package)


def test_deployment_package_metadata_is_used_without_fabricating_version(tmp_path, monkeypatch):
    package, metadata = _build_package(tmp_path)
    monkeypatch.setenv("DEPLOYMENT_PACKAGE_DIR", str(package))
    assert model_service.prediction_metadata() == {
        "model_name": "churn_predictor",
        "model_version": "7",
        "deployment_id": "deployment-1",
        "model_version_id": "dagshub:owner/repo:churn_predictor:7",
        "mlflow_run_id": "run-1",
    }
    monkeypatch.setattr(model_service, "deployment_artifacts_ready", lambda: False)
    monkeypatch.setattr(model_service, "load_metadata", lambda: {"model_name": "churn_predictor"})
    assert model_service.prediction_metadata() == {"model_name": "churn_predictor"}


def test_database_connectivity_check_does_not_return_credentials(monkeypatch):
    class Result:
        def scalar_one(self):
            return 1

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def execute(self, statement):
            assert str(statement) == "SELECT 1"
            return Result()

    class Engine:
        disposed = False

        def connect(self):
            return Connection()

        def dispose(self):
            self.disposed = True

    engine = Engine()
    monkeypatch.setattr(connection, "create_database_engine", lambda settings=None: engine)
    result = connection.check_connectivity()
    assert result["status"] == "ok"
    assert "url" not in result
    assert engine.disposed is True
