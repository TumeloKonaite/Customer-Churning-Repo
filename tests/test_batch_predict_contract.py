import io
import json

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


def csv_upload_from_records(records):
    extra_headers = [key for key in ("customer_id", "row_id", "id") if any(key in row for row in records)]
    headers = list(REQUIRED_FIELDS) + extra_headers
    lines = [",".join(headers)]
    for record in records:
        lines.append(",".join(str(record.get(header, "")) for header in headers))
    return {"file": ("batch.csv", io.BytesIO(("\n".join(lines) + "\n").encode()), "text/csv")}


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


def test_batch_csv_uses_same_prediction_contract(monkeypatch):
    patch_batch_execution(monkeypatch, labels=[1], probabilities=[0.73])
    record = valid_record()
    record["customer_id"] = "csv-1"
    response = client.post(
        "/api/batch_predict_csv",
        files=csv_upload_from_records([record]),
        data={"options": json.dumps({"mode": "partial"})},
    )

    assert response.status_code == 200
    body = response.json()
    assert_prediction_only_envelope(body)
    assert body["results"] == [
        {"index": 0, "id": "csv-1", "predicted_label": 1, "p_churn": 0.73}
    ]


def test_batch_csv_rejects_invalid_options_and_missing_columns():
    invalid_options = client.post(
        "/api/batch_predict_csv",
        files=csv_upload_from_records([valid_record()]),
        data={"options": "{bad"},
    )
    assert invalid_options.status_code == 400
    assert "Invalid options JSON" in invalid_options.json()["message"]

    csv_body = b"CreditScore,Geography\n619,France\n"
    missing_columns = client.post(
        "/api/batch_predict_csv",
        files={"file": ("batch.csv", io.BytesIO(csv_body), "text/csv")},
    )
    assert missing_columns.status_code == 400
    assert "missing required columns" in missing_columns.json()["message"]


def test_batch_csv_missing_file_preserves_contract_error():
    response = client.post("/api/batch_predict_csv", data={"options": '{"mode":"partial"}'})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == "file"
