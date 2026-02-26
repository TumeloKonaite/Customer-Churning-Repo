import application
import src.services.prediction_service as prediction_service
from src.services.prediction_service import MAX_BATCH_SIZE
from src.services.prediction_service import REQUIRED_FIELDS


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

    def predict(self, df):
        FakePredictPipeline.call_count += 1
        FakePredictPipeline.last_df = df.copy()
        return list(FakePredictPipeline.labels), list(FakePredictPipeline.probabilities)


def patch_batch_execution(monkeypatch, *, labels, probabilities):
    FakePredictPipeline.call_count = 0
    FakePredictPipeline.last_df = None
    FakePredictPipeline.labels = list(labels)
    FakePredictPipeline.probabilities = list(probabilities)
    monkeypatch.setattr(prediction_service, "PredictPipeline", FakePredictPipeline)
    monkeypatch.setattr(application, "artifacts_ready", lambda: True)
    monkeypatch.setattr(
        prediction_service,
        "_load_model_metadata",
        lambda: {"model_name": "test-model", "model_version": "test-version"},
    )


def test_batch_contract_requires_records():
    client = application.app.test_client()

    response = client.post("/api/predict/batch", json={})

    assert response.status_code == 400
    body = response.get_json()
    assert body["status"] == "error"
    assert "records" in body["message"]
    assert body["contract_version"] == "v1"


def test_batch_over_limit_returns_413():
    client = application.app.test_client()
    payload = {
        "records": [valid_record() for _ in range(MAX_BATCH_SIZE + 1)],
        "options": {"mode": "partial"},
    }

    response = client.post("/api/predict/batch", json=payload)

    assert response.status_code == 413
    body = response.get_json()
    assert body["status"] == "error"
    assert body["contract_version"] == "v1"


def test_batch_mode_fail_fast_rejects_invalid(monkeypatch):
    patch_batch_execution(monkeypatch, labels=[0], probabilities=[0.1])
    client = application.app.test_client()
    invalid = valid_record()
    invalid.pop("Age")
    payload = {
        "records": [valid_record(), invalid, valid_record()],
        "options": {"mode": "fail_fast"},
    }

    response = client.post("/api/predict/batch", json=payload)

    assert response.status_code == 400
    body = response.get_json()
    assert body["status"] == "error"
    assert body["results"] == []
    assert len(body["errors"]) == 1
    assert body["errors"][0]["row_index"] == 1
    assert body["errors"][0]["field"] == "Age"
    assert body["errors"][0]["message"] == "Missing required field: Age"
    assert set(body) == {"status", "results", "errors", "summary", "metadata", "timestamp"}
    assert FakePredictPipeline.call_count == 0
    assert body["summary"]["mode"] == "fail_fast"
    assert body["metadata"]["model_version"] == "test-version"


def test_batch_mode_partial_predicts_once_and_maps_indices(monkeypatch):
    patch_batch_execution(monkeypatch, labels=[1, 0], probabilities=[0.91, 0.08])
    client = application.app.test_client()
    missing_age = valid_record()
    missing_age.pop("Age")
    trailing_valid = valid_record()
    trailing_valid["Age"] = 50
    payload = {
        "records": [valid_record(), missing_age, trailing_valid],
        "options": {"mode": "partial"},
    }

    response = client.post("/api/predict/batch", json=payload)

    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "partial"
    assert set(body) == {"status", "results", "errors", "summary", "metadata", "timestamp"}
    assert FakePredictPipeline.call_count == 1
    assert list(FakePredictPipeline.last_df.index) == [0, 1]
    assert len(FakePredictPipeline.last_df) == 2
    assert len(body["results"]) == 2
    assert [item["index"] for item in body["results"]] == [0, 2]
    assert [item["predicted_label"] for item in body["results"]] == [1, 0]
    assert [item["p_churn"] for item in body["results"]] == [0.91, 0.08]
    assert len(body["errors"]) == 1
    assert [err["row_index"] for err in body["errors"]] == [1]
    assert body["errors"][0]["field"] == "Age"
    assert body["errors"][0]["message"] == "Missing required field: Age"
    assert body["summary"]["mode"] == "partial"
    assert body["summary"]["total_records"] == 3
    assert body["summary"]["valid_records"] == 2
    assert body["summary"]["invalid_records"] == 1
    assert body["metadata"]["model_name"] == "test-model"
    assert body["metadata"]["model_version"] == "test-version"


def test_batch_all_valid_default_mode_predicts_once(monkeypatch):
    patch_batch_execution(monkeypatch, labels=[0, 1], probabilities=[0.12, 0.87])
    client = application.app.test_client()
    payload = {
        "records": [valid_record(), valid_record()],
    }

    response = client.post("/api/predict/batch", json=payload)

    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "success"
    assert body["errors"] is None
    assert [item["index"] for item in body["results"]] == [0, 1]
    assert set(body) == {"status", "results", "errors", "summary", "metadata", "timestamp"}
    assert FakePredictPipeline.call_count == 1
    assert body["summary"]["mode"] == "fail_fast"
    assert body["summary"]["valid_records"] == 2
    assert body["summary"]["invalid_records"] == 0


def test_batch_partial_with_no_valid_rows_returns_failed_and_skips_model(monkeypatch):
    patch_batch_execution(monkeypatch, labels=[], probabilities=[])
    client = application.app.test_client()
    invalid_one = valid_record()
    invalid_one.pop("Age")
    invalid_two = valid_record()
    invalid_two["Balance"] = "bad"
    payload = {
        "records": [invalid_one, invalid_two],
        "options": {"mode": "partial"},
    }

    response = client.post("/api/predict/batch", json=payload)

    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "failed"
    assert body["results"] == []
    assert FakePredictPipeline.call_count == 0
    assert len(body["errors"]) == 2
    assert [err["row_index"] for err in body["errors"]] == [0, 1]
    assert body["summary"]["total_records"] == 2
    assert body["summary"]["valid_records"] == 0
    assert body["summary"]["invalid_records"] == 2


def test_batch_id_passthrough_results(monkeypatch):
    patch_batch_execution(monkeypatch, labels=[1, 0], probabilities=[0.91, 0.08])
    client = application.app.test_client()
    first = valid_record()
    first["customer_id"] = "cust-123"
    second = valid_record()
    second["row_id"] = 42
    second["Age"] = 50
    payload = {"records": [first, second], "options": {"mode": "partial"}}

    response = client.post("/api/predict/batch", json=payload)

    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "success"
    assert [item["id"] for item in body["results"]] == ["cust-123", 42]
    assert [item["index"] for item in body["results"]] == [0, 1]


def test_batch_id_passthrough_errors(monkeypatch):
    patch_batch_execution(monkeypatch, labels=[], probabilities=[])
    client = application.app.test_client()
    invalid = valid_record()
    invalid["customer_id"] = "cust-bad"
    invalid.pop("Age")
    payload = {"records": [invalid], "options": {"mode": "partial"}}

    response = client.post("/api/predict/batch", json=payload)

    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "failed"
    assert body["results"] == []
    assert len(body["errors"]) == 1
    assert body["errors"][0]["id"] == "cust-bad"
    assert body["errors"][0]["row_index"] == 0
    assert body["errors"][0]["field"] == "Age"


def test_batch_df_excludes_id_fields(monkeypatch):
    patch_batch_execution(monkeypatch, labels=[0, 1], probabilities=[0.12, 0.87])
    client = application.app.test_client()
    first = valid_record()
    first["customer_id"] = "cust-1"
    second = valid_record()
    second["row_id"] = "row-2"
    second["Age"] = 50
    payload = {"records": [first, second], "options": {"mode": "partial"}}

    response = client.post("/api/predict/batch", json=payload)

    assert response.status_code == 200
    assert FakePredictPipeline.call_count == 1
    assert FakePredictPipeline.last_df is not None
    assert "customer_id" not in FakePredictPipeline.last_df.columns
    assert "row_id" not in FakePredictPipeline.last_df.columns
    assert "id" not in FakePredictPipeline.last_df.columns
    assert set(FakePredictPipeline.last_df.columns) == set(REQUIRED_FIELDS)
