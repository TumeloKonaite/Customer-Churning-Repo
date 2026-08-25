import application
import src.services.batch_prediction_service as batch_prediction_service
from fastapi.testclient import TestClient
from src.schemas.batch_prediction import MAX_BATCH_SIZE
from src.schemas.prediction import REQUIRED_FIELDS
from src.services import model_service


client = TestClient(application.app)


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


class FakePredictPipeline:
    call_count = 0
    last_df = None
    labels = []
    probabilities = []

    def predict(self, frame):
        type(self).call_count += 1
        type(self).last_df = frame.copy()
        probabilities = None if type(self).probabilities is None else list(type(self).probabilities)
        return list(type(self).labels), probabilities


def patch_batch_execution(monkeypatch, *, labels, probabilities):
    FakePredictPipeline.call_count = 0
    FakePredictPipeline.last_df = None
    FakePredictPipeline.labels = list(labels)
    FakePredictPipeline.probabilities = None if probabilities is None else list(probabilities)
    monkeypatch.setattr(batch_prediction_service, "PredictPipeline", FakePredictPipeline)
    monkeypatch.setattr(model_service, "artifacts_ready", lambda: True)
    monkeypatch.setattr(
        batch_prediction_service,
        "_load_model_metadata",
        lambda: {"model_name": "test-model", "model_version": "test-version"},
    )


def assert_prediction_only_envelope(body):
    assert set(body) == {"status", "results", "errors", "summary", "metadata", "timestamp"}
    for result in body["results"]:
        assert set(result) == {"index", "id", "predicted_label", "p_churn"}


def test_batch_contract_requires_records():
    response = client.post("/api/predict/batch", json={})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == "records"


def test_batch_over_limit_returns_413():
    response = client.post(
        "/api/predict/batch", json={"records": [valid_record()] * (MAX_BATCH_SIZE + 1)}
    )
    assert response.status_code == 422


def test_batch_fail_fast_stops_before_model_call(monkeypatch):
    patch_batch_execution(monkeypatch, labels=[], probabilities=[])
    invalid = valid_record()
    invalid.pop("Age")
    response = client.post(
        "/api/predict/batch",
        json={"records": [valid_record(), invalid, valid_record()], "options": {"mode": "fail_fast"}},
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == "Age"
    assert FakePredictPipeline.call_count == 0


def test_json_batch_partial_still_rejects_invalid_records_before_prediction(monkeypatch):
    patch_batch_execution(monkeypatch, labels=[1, 0], probabilities=[0.91, 0.08])
    invalid = valid_record()
    invalid["Balance"] = "bad"
    final = valid_record()
    final["Age"] = 50
    response = client.post(
        "/api/predict/batch",
        json={"records": [valid_record(), invalid, final], "options": {"mode": "partial"}},
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == "Balance"
    assert FakePredictPipeline.call_count == 0


def test_batch_preserves_supported_ids_without_model_features(monkeypatch):
    patch_batch_execution(monkeypatch, labels=[1, 0, 1], probabilities=[0.8, 0.2, 0.7])
    records = [valid_record(), valid_record(), valid_record()]
    records[0]["customer_id"] = "customer-1"
    records[1]["row_id"] = 42
    records[2]["id"] = "generic-3"
    response = client.post("/api/predict/batch", json={"records": records})

    body = response.json()
    assert [item["id"] for item in body["results"]] == ["customer-1", 42, "generic-3"]
    assert list(FakePredictPipeline.last_df.columns) == REQUIRED_FIELDS


def test_batch_probability_unavailable_is_null(monkeypatch):
    patch_batch_execution(monkeypatch, labels=[1, 0], probabilities=None)
    response = client.post(
        "/api/predict/batch", json={"records": [valid_record(), valid_record()]}
    )

    body = response.json()
    assert_prediction_only_envelope(body)
    assert [item["p_churn"] for item in body["results"]] == [None, None]


def test_batch_partial_with_no_valid_rows_returns_errors(monkeypatch):
    patch_batch_execution(monkeypatch, labels=[], probabilities=[])
    missing = valid_record()
    missing.pop("Age")
    numeric = valid_record()
    numeric["Tenure"] = "bad"
    response = client.post(
        "/api/predict/batch",
        json={"records": [missing, numeric], "options": {"mode": "partial"}},
    )

    assert response.status_code == 422
    assert [error["loc"][-1] for error in response.json()["detail"]] == ["Age", "Tenure"]
    assert FakePredictPipeline.call_count == 0


def test_batch_returns_503_when_model_is_not_ready(monkeypatch):
    monkeypatch.setattr(model_service, "artifacts_ready", lambda: False)
    response = client.post(
        "/api/predict/batch", json={"records": [valid_record()]}
    )
    assert response.status_code == 503


def test_removed_prediction_aliases_are_not_routable():
    assert client.post("/api/batch_predict", json={}).status_code == 404
    assert client.post("/api/batch_predict_csv").status_code == 404
