import io
import json

import application
import src.services.prediction_service as prediction_service
from src.services.prediction_service import MAX_BATCH_SIZE, REQUIRED_FIELDS


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
    return {"file": (io.BytesIO(("\n".join(lines) + "\n").encode()), "batch.csv")}


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
    monkeypatch.setattr(prediction_service, "PredictPipeline", FakePredictPipeline)
    monkeypatch.setattr(application, "artifacts_ready", lambda: True)
    monkeypatch.setattr(
        prediction_service,
        "_load_model_metadata",
        lambda: {"model_name": "test-model", "model_version": "test-version"},
    )


def assert_prediction_only_envelope(body):
    assert set(body) == {"status", "results", "errors", "summary", "metadata", "timestamp"}
    for result in body["results"]:
        assert set(result) == {"index", "id", "predicted_label", "p_churn"}


def test_batch_contract_requires_records():
    response = application.app.test_client().post("/api/predict/batch", json={})
    assert response.status_code == 400
    assert response.get_json()["contract_version"] == "v1"


def test_batch_over_limit_returns_413():
    response = application.app.test_client().post(
        "/api/predict/batch", json={"records": [valid_record()] * (MAX_BATCH_SIZE + 1)}
    )
    assert response.status_code == 413


def test_batch_fail_fast_stops_before_model_call(monkeypatch):
    patch_batch_execution(monkeypatch, labels=[], probabilities=[])
    invalid = valid_record()
    invalid.pop("Age")
    response = application.app.test_client().post(
        "/api/predict/batch",
        json={"records": [valid_record(), invalid, valid_record()], "options": {"mode": "fail_fast"}},
    )

    assert response.status_code == 400
    body = response.get_json()
    assert_prediction_only_envelope(body)
    assert body["status"] == "error"
    assert body["results"] == []
    assert body["errors"][0]["row_index"] == 1
    assert body["errors"][0]["field"] == "Age"
    assert FakePredictPipeline.call_count == 0


def test_batch_partial_scores_valid_rows_and_preserves_indices(monkeypatch):
    patch_batch_execution(monkeypatch, labels=[1, 0], probabilities=[0.91, 0.08])
    invalid = valid_record()
    invalid["Balance"] = "bad"
    final = valid_record()
    final["Age"] = 50
    response = application.app.test_client().post(
        "/api/predict/batch",
        json={"records": [valid_record(), invalid, final], "options": {"mode": "partial"}},
    )

    assert response.status_code == 200
    body = response.get_json()
    assert_prediction_only_envelope(body)
    assert body["status"] == "partial"
    assert [item["index"] for item in body["results"]] == [0, 2]
    assert [item["predicted_label"] for item in body["results"]] == [1, 0]
    assert [item["p_churn"] for item in body["results"]] == [0.91, 0.08]
    assert body["errors"][0]["field"] == "Balance"
    assert body["summary"] == {
        "total_records": 3,
        "valid_records": 2,
        "invalid_records": 1,
        "error_count": 1,
        "mode": "partial",
    }
    assert FakePredictPipeline.call_count == 1


def test_batch_preserves_supported_ids_without_model_features(monkeypatch):
    patch_batch_execution(monkeypatch, labels=[1, 0, 1], probabilities=[0.8, 0.2, 0.7])
    records = [valid_record(), valid_record(), valid_record()]
    records[0]["customer_id"] = "customer-1"
    records[1]["row_id"] = 42
    records[2]["id"] = "generic-3"
    response = application.app.test_client().post("/api/predict/batch", json={"records": records})

    body = response.get_json()
    assert [item["id"] for item in body["results"]] == ["customer-1", 42, "generic-3"]
    assert list(FakePredictPipeline.last_df.columns) == REQUIRED_FIELDS


def test_batch_probability_unavailable_is_null(monkeypatch):
    patch_batch_execution(monkeypatch, labels=[1, 0], probabilities=None)
    response = application.app.test_client().post(
        "/api/predict/batch", json={"records": [valid_record(), valid_record()]}
    )

    body = response.get_json()
    assert_prediction_only_envelope(body)
    assert [item["p_churn"] for item in body["results"]] == [None, None]


def test_batch_partial_with_no_valid_rows_returns_errors(monkeypatch):
    patch_batch_execution(monkeypatch, labels=[], probabilities=[])
    missing = valid_record()
    missing.pop("Age")
    numeric = valid_record()
    numeric["Tenure"] = "bad"
    response = application.app.test_client().post(
        "/api/predict/batch",
        json={"records": [missing, numeric], "options": {"mode": "partial"}},
    )

    body = response.get_json()
    assert body["status"] == "failed"
    assert body["results"] == []
    assert [error["field"] for error in body["errors"]] == ["Age", "Tenure"]
    assert FakePredictPipeline.call_count == 0


def test_batch_returns_503_when_model_is_not_ready(monkeypatch):
    monkeypatch.setattr(application, "artifacts_ready", lambda: False)
    response = application.app.test_client().post(
        "/api/predict/batch", json={"records": [valid_record()]}
    )
    assert response.status_code == 503


def test_batch_csv_uses_same_prediction_contract(monkeypatch):
    patch_batch_execution(monkeypatch, labels=[1], probabilities=[0.73])
    record = valid_record()
    record["customer_id"] = "csv-1"
    response = application.app.test_client().post(
        "/api/batch_predict_csv",
        data={**csv_upload_from_records([record]), "options": json.dumps({"mode": "partial"})},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    body = response.get_json()
    assert_prediction_only_envelope(body)
    assert body["results"] == [
        {"index": 0, "id": "csv-1", "predicted_label": 1, "p_churn": 0.73}
    ]


def test_batch_csv_rejects_invalid_options_and_missing_columns():
    client = application.app.test_client()
    invalid_options = client.post(
        "/api/batch_predict_csv",
        data={**csv_upload_from_records([valid_record()]), "options": "{bad"},
        content_type="multipart/form-data",
    )
    assert invalid_options.status_code == 400
    assert "Invalid options JSON" in invalid_options.get_json()["message"]

    csv_body = b"CreditScore,Geography\n619,France\n"
    missing_columns = client.post(
        "/api/batch_predict_csv",
        data={"file": (io.BytesIO(csv_body), "batch.csv")},
        content_type="multipart/form-data",
    )
    assert missing_columns.status_code == 400
    assert "missing required columns" in missing_columns.get_json()["message"]
