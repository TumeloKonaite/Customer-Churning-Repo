import json
from datetime import datetime

import pytest

from src.config import DatabaseSettings
from src.monitoring.prediction_repository import (
    PendingPredictionEvent,
    PredictionEventRepository,
)
from src.schemas.prediction import REQUIRED_FIELDS
from src.services import (
    batch_prediction_service,
    model_service,
    prediction_event_service,
    single_prediction_service,
)
from src.services.exceptions import PredictionExecutionError
from src.services.prediction_event_service import PredictionPersistenceError


def valid_record():
    return {
        "CreditScore": 619,
        "Geography": "France",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 2,
        "Balance": 0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 101348.88,
    }


def deployment_metadata():
    return {
        "environment": "production",
        "model_version_id": "dagshub:owner/repo:churn_predictor:5",
        "feature_schema_version": "1.0.0",
        "deployment_id": "deployment-1",
    }


def production_settings():
    return DatabaseSettings(
        environment="production",
        database_url=(
            "postgresql+psycopg://prediction_app:secret@pool.neon.tech/"
            "churn_monitoring?sslmode=require"
        ),
    )


class CapturingRepository:
    def __init__(self):
        self.events = ()

    def append(self, events):
        self.events = tuple(events)
        return len(self.events)


def test_persistence_builds_canonical_append_only_events(monkeypatch):
    repository = CapturingRepository()
    monkeypatch.setattr(
        prediction_event_service,
        "_runtime",
        lambda: (production_settings(), repository),
    )

    prediction_ids = prediction_event_service.persist_prediction_events(
        feature_rows=[valid_record(), {**valid_record(), "Age": 55}],
        labels=[1, 0],
        probabilities=[0.81, 0.12],
        prediction_timestamp=datetime.fromisoformat("2026-08-24T12:00:00+00:00"),
        metadata=deployment_metadata(),
    )

    assert len(prediction_ids) == 2
    assert len(set(prediction_ids)) == 2
    assert [event.prediction_id for event in repository.events] == list(prediction_ids)
    assert all(event.environment == "production" for event in repository.events)
    assert all(event.feature_schema_version == "1.0.0" for event in repository.events)
    assert list(repository.events[0].features) == REQUIRED_FIELDS


def test_persistence_requires_probability_when_database_is_configured(monkeypatch):
    repository = CapturingRepository()
    monkeypatch.setattr(
        prediction_event_service,
        "_runtime",
        lambda: (production_settings(), repository),
    )

    with pytest.raises(PredictionPersistenceError, match="probability is required"):
        prediction_event_service.persist_prediction_events(
            feature_rows=[valid_record()],
            labels=[1],
            probabilities=[None],
            prediction_timestamp=datetime.fromisoformat(
                "2026-08-24T12:00:00+00:00"
            ),
            metadata=deployment_metadata(),
        )

    assert repository.events == ()


def test_repository_uses_one_transaction_and_explicitly_marks_events_ineligible():
    executions = []

    class Connection:
        def execute(self, statement, parameters):
            executions.append((str(statement), parameters))

    class Transaction:
        def __enter__(self):
            return Connection()

        def __exit__(self, *args):
            return None

    class Engine:
        def begin(self):
            return Transaction()

    event = PendingPredictionEvent(
        prediction_id="prediction-1",
        environment="production",
        model_version_id="dagshub:owner/repo:churn_predictor:5",
        prediction_timestamp=datetime.fromisoformat("2026-08-24T12:00:00+00:00"),
        feature_schema_version="1.0.0",
        features=valid_record(),
        prediction_probability=0.81,
        predicted_class="1",
        deployment_id="deployment-1",
    )

    count = PredictionEventRepository(Engine()).append([event])

    assert count == 1
    statement, parameters = executions[0]
    assert "monitoring_eligible" in statement
    assert "FALSE" in statement
    assert len(parameters) == 1
    stored_features = json.loads(parameters[0]["features"])
    assert set(stored_features) == set(REQUIRED_FIELDS)
    assert not {"customer_id", "row_id", "id"} & set(stored_features)


def test_single_prediction_persists_before_returning_success(monkeypatch):
    captured = {}
    monkeypatch.setattr(model_service, "artifacts_ready", lambda: True)
    monkeypatch.setattr(
        single_prediction_service, "_predict_one", lambda payload: (1, 0.81)
    )
    monkeypatch.setattr(
        model_service, "prediction_metadata", lambda: {"model_name": "test"}
    )
    monkeypatch.setattr(model_service, "load_metadata", deployment_metadata)
    monkeypatch.setattr(model_service, "operational_metadata", lambda: {})
    monkeypatch.setattr(
        prediction_event_service,
        "persist_prediction_events",
        lambda **kwargs: captured.update(kwargs) or ("prediction-1",),
    )

    result = single_prediction_service.predict_single(valid_record())

    assert result["status"] == "success"
    assert captured["feature_rows"] == [valid_record()]
    assert captured["labels"] == [1]
    assert captured["probabilities"] == [0.81]
    assert result["timestamp"] == captured["prediction_timestamp"].isoformat()


def test_batch_persistence_strips_caller_identifiers(monkeypatch):
    captured = {}

    class Pipeline:
        def predict(self, frame):
            return [1], [0.73]

    monkeypatch.setattr(model_service, "artifacts_ready", lambda: True)
    monkeypatch.setattr(batch_prediction_service, "PredictPipeline", Pipeline)
    monkeypatch.setattr(model_service, "load_metadata", deployment_metadata)
    monkeypatch.setattr(model_service, "operational_metadata", lambda: {})
    monkeypatch.setattr(
        prediction_event_service,
        "persist_prediction_events",
        lambda **kwargs: captured.update(kwargs) or ("prediction-1",),
    )
    record = {**valid_record(), "customer_id": "must-not-be-stored"}

    result = batch_prediction_service.predict_batch([record], {"mode": "partial"})

    assert result["status"] == "success"
    assert captured["feature_rows"] == [valid_record()]
    assert captured["labels"] == [1]
    assert captured["probabilities"] == [0.73]


def test_persistence_failure_prevents_success_response(monkeypatch):
    monkeypatch.setattr(model_service, "artifacts_ready", lambda: True)
    monkeypatch.setattr(
        single_prediction_service, "_predict_one", lambda payload: (1, 0.81)
    )
    monkeypatch.setattr(
        model_service, "prediction_metadata", lambda: {"model_name": "test"}
    )
    monkeypatch.setattr(model_service, "load_metadata", deployment_metadata)
    monkeypatch.setattr(
        prediction_event_service,
        "persist_prediction_events",
        lambda **kwargs: (_ for _ in ()).throw(
            PredictionPersistenceError("database unavailable")
        ),
    )

    with pytest.raises(PredictionExecutionError, match="could not be persisted"):
        single_prediction_service.predict_single(valid_record())
