"""Opt-in checks for explicitly configured non-production external resources."""

import os

import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_EXTERNAL_MLOPS_TESTS") != "1",
    reason="external DagsHub/Neon integration tests are opt-in",
)


def test_exact_registered_model_can_be_validated_and_packaged(tmp_path):
    from src.mlops.deployment import prepare_deployment

    uri = os.environ["TEST_MLFLOW_MODEL_URI"]
    metadata = prepare_deployment(uri, tmp_path / "model", environment="test")
    assert metadata["model_version"].isdigit()
    assert (tmp_path / "model" / "deployment_metadata.json").is_file()


def test_test_neon_connectivity():
    from src.config import DatabaseSettings
    from src.database import check_connectivity

    settings = DatabaseSettings(
        environment="test",
        database_url=os.environ["TEST_DATABASE_URL"],
    )
    assert check_connectivity(settings)["status"] == "ok"
