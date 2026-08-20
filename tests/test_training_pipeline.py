from types import SimpleNamespace

from src.mlops.tracking import TrackingResult
from src.pipeline.training_pipeline import TrainingPipeline


def test_training_pipeline_uses_existing_components_in_order():
    calls = []
    cohorts = object()
    fitted_training = SimpleNamespace(artifact_dir="artifacts/training")
    tracking = TrackingResult(
        status="registered",
        artifact_dir="artifacts/training",
        run_id="run-1",
        registered_model_name="churn_predictor",
        registered_model_version="7",
        model_version_id="dagshub:owner/repo:churn_predictor:7",
        pipeline_sha256="a" * 64,
    )

    class Ingestion:
        def load(self, dataset, split):
            calls.append("ingestion")
            assert dataset["name"] == "bank_customer_churn"
            return cohorts

    class Trainer:
        def train(
            self,
            received,
            model,
            eligibility,
            *,
            random_seed,
            output_dir,
            training_config,
        ):
            calls.append("trainer")
            assert received is cohorts
            assert set(model["candidates"]) == {
                "logistic_regression",
                "decision_tree",
                "random_forest",
                "gradient_boosting",
            }
            assert random_seed == 42
            assert output_dir == "artifacts/training"
            assert training_config["version"] == "1.0.0"
            return fitted_training

    class Tracker:
        def track(self, received_training, config):
            calls.append("tracker")
            assert received_training is fitted_training
            assert config["version"] == "1.0.0"
            return tracking

    result = TrainingPipeline(
        ingestion=Ingestion(), trainer=Trainer(), tracker=Tracker()
    ).run("configs/training.yaml")

    assert result is tracking
    assert calls == ["ingestion", "trainer", "tracker"]
