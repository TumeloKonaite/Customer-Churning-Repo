from __future__ import annotations

from pathlib import Path

import pytest

from src.config import MonitoringArtifactBackend, MonitoringSettings
from src.monitoring.shared.artifacts import MLflowArtifactStore


def test_mlflow_artifact_store_downloads_exact_run_artifact(monkeypatch):
    expected = b"approved-reference"

    def download_artifacts(*, artifact_uri, dst_path):
        assert artifact_uri == (
            "runs:/0123456789abcdef0123456789abcdef/references/base.parquet"
        )
        target = Path(dst_path) / "base.parquet"
        target.write_bytes(expected)
        return str(target)

    monkeypatch.setattr("mlflow.artifacts.download_artifacts", download_artifacts)

    body = MLflowArtifactStore().read_uri(
        "runs:/0123456789abcdef0123456789abcdef/references/base.parquet"
    )

    assert body == expected


@pytest.mark.parametrize(
    "uri",
    [
        "models:/churn_predictor/5",
        "runs:/short/references/base.parquet",
        "runs:/0123456789abcdef0123456789abcdef/../secret",
    ],
)
def test_mlflow_artifact_store_rejects_non_exact_run_uri(uri):
    with pytest.raises(ValueError, match="runs:/"):
        MLflowArtifactStore().read_uri(uri)


def test_production_monitoring_accepts_mlflow_backend():
    settings = MonitoringSettings(
        APP_ENV="production",
        EXPECTED_MODEL_VERSION_ID="dagshub:owner/repo:churn_predictor:5",
        MONITORING_ARTIFACT_BACKEND="mlflow",
    )

    assert settings.artifact_backend is MonitoringArtifactBackend.MLFLOW


def test_production_monitoring_rejects_local_backend(tmp_path):
    with pytest.raises(ValueError, match="cannot use local"):
        MonitoringSettings(
            APP_ENV="production",
            EXPECTED_MODEL_VERSION_ID="dagshub:owner/repo:churn_predictor:5",
            MONITORING_ARTIFACT_BACKEND="local",
            MONITORING_LOCAL_ARTIFACT_DIR=tmp_path,
        )


def test_legacy_s3_configuration_is_still_inferred():
    settings = MonitoringSettings(
        EXPECTED_MODEL_VERSION_ID="dagshub:owner/repo:churn_predictor:5",
        MONITORING_ARTIFACT_BUCKET="approved-baselines",
    )

    assert settings.artifact_backend is MonitoringArtifactBackend.S3
