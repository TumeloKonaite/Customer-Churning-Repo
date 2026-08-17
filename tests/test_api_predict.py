import numpy as np

import application


def valid_payload():
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


class FakeCustomData:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_data_as_data_frame(self):
        return object()


def patch_prediction(monkeypatch, *, label=1, probability=0.82):
    class FakePredictPipeline:
        def predict(self, features):
            probabilities = None if probability is None else np.array([probability])
            return np.array([label]), probabilities

    monkeypatch.setattr(application, "artifacts_ready", lambda: True)
    monkeypatch.setattr(application, "CustomData", FakeCustomData)
    monkeypatch.setattr(application, "PredictPipeline", FakePredictPipeline)
    monkeypatch.setattr(
        application,
        "load_metadata",
        lambda: {"model_name": "test_model", "version": "9.9.9"},
    )


def test_predict_returns_prediction_only_contract(monkeypatch):
    patch_prediction(monkeypatch, label=1, probability=0.82)
    response = application.app.test_client().post("/api/predict", json=valid_payload())

    assert response.status_code == 200
    body = response.get_json()
    assert set(body) == {
        "status", "predicted_label", "p_churn", "model_name", "model_version", "timestamp"
    }
    assert body["status"] == "success"
    assert body["predicted_label"] == 1
    assert body["p_churn"] == 0.82
    assert body["model_name"] == "test_model"
    assert body["model_version"] == "9.9.9"


def test_predict_probability_unavailable_is_null(monkeypatch):
    patch_prediction(monkeypatch, label=0, probability=None)
    response = application.app.test_client().post("/api/predict", json=valid_payload())

    assert response.status_code == 200
    body = response.get_json()
    assert body["predicted_label"] == 0
    assert body["p_churn"] is None


def test_predict_reports_all_missing_required_fields():
    payload = valid_payload()
    payload.pop("Age")
    payload.pop("Balance")
    response = application.app.test_client().post("/api/predict", json=payload)

    assert response.status_code == 400
    body = response.get_json()
    assert body["status"] == "error"
    assert body["errors"] == ["Missing required field: Age", "Missing required field: Balance"]


def test_predict_rejects_invalid_numeric_input():
    payload = valid_payload()
    payload["Age"] = "not-a-number"
    response = application.app.test_client().post("/api/predict", json=payload)

    assert response.status_code == 400
    assert "Field 'Age' must be a number" in response.get_json()["errors"][0]


def test_predict_returns_503_when_model_is_not_ready(monkeypatch):
    monkeypatch.setattr(application, "artifacts_ready", lambda: False)
    response = application.app.test_client().post("/api/predict", json=valid_payload())

    assert response.status_code == 503
    assert response.get_json()["status"] == "error"


def test_health_reports_artifact_readiness_and_metadata(monkeypatch):
    monkeypatch.setattr(application, "artifacts_ready", lambda: False)
    monkeypatch.setattr(application, "load_metadata", lambda: {"model_name": "test_model"})
    response = application.app.test_client().get("/health")

    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is False
    assert body["metadata"] == {"model_name": "test_model"}
    assert "timestamp" in body
