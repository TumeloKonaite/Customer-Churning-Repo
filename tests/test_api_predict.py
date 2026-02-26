import numpy as np

import application


def test_predict_proba_missing_returns_null_decisioning_fields(monkeypatch):
    class FakeCustomData:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def get_data_as_data_frame(self):
            return object()

    class FakePredictPipeline:
        def predict(self, features):
            return np.array([0]), None

    monkeypatch.setattr(application, "artifacts_ready", lambda: True)
    monkeypatch.setattr(application, "CustomData", FakeCustomData)
    monkeypatch.setattr(application, "PredictPipeline", FakePredictPipeline)
    monkeypatch.setattr(
        application,
        "load_metadata",
        lambda: {"model_name": "test_model", "version": "9.9.9"},
    )

    client = application.app.test_client()
    payload = {
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

    response = client.post("/api/predict", json=payload)

    assert response.status_code == 200
    body = response.get_json()

    assert body["status"] == "success"
    assert body["predicted_label"] == 0
    assert "clv" in body
    assert body["clv"] is not None

    assert "p_churn" in body
    assert body["p_churn"] is None
    assert "recommended_action" in body
    assert body["recommended_action"] is None
    assert "net_gain" in body
    assert body["net_gain"] is None

    assert body["model_name"] == "test_model"
    assert body["model_version"] == "9.9.9"
