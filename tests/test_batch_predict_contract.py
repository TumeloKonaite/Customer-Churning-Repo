import application
import src.services.prediction_service as prediction_service
from src.decisioning import ACTION_DISCOUNT_CALL
from src.services.prediction_service import MAX_BATCH_SIZE
from src.services.prediction_service import REQUIRED_FIELDS
import io
import json


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
    headers = list(REQUIRED_FIELDS)
    all_keys = set(headers)
    for record in records:
        if isinstance(record, dict):
            all_keys.update(record.keys())
    ordered_headers = headers + [key for key in ("customer_id", "row_id", "id") if key in all_keys]

    lines = [",".join(ordered_headers)]
    for record in records:
        values = []
        for header in ordered_headers:
            value = record.get(header, "")
            values.append("" if value is None else str(value))
        lines.append(",".join(values))

    csv_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    return {"file": (io.BytesIO(csv_bytes), "batch.csv")}


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
    assert body["email_candidates"] == []
    assert len(body["errors"]) == 1
    assert body["errors"][0]["row_index"] == 1
    assert body["errors"][0]["field"] == "Age"
    assert body["errors"][0]["message"] == "Missing required field: Age"
    assert set(body) == {"status", "results", "email_candidates", "errors", "summary", "metadata", "timestamp"}
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
    assert set(body) == {"status", "results", "email_candidates", "errors", "summary", "metadata", "timestamp"}
    assert FakePredictPipeline.call_count == 1
    assert list(FakePredictPipeline.last_df.index) == [0, 1]
    assert len(FakePredictPipeline.last_df) == 2
    assert len(body["results"]) == 2
    assert [item["index"] for item in body["results"]] == [0, 2]
    assert [item["predicted_label"] for item in body["results"]] == [1, 0]
    assert [item["p_churn"] for item in body["results"]] == [0.91, 0.08]
    assert all("clv" in item for item in body["results"])
    assert all("recommended_action" in item for item in body["results"])
    assert all("net_gain" in item for item in body["results"])
    assert len(body["email_candidates"]) == 1
    assert body["email_candidates"][0]["index"] == 0
    assert body["email_candidates"][0]["p_churn"] == 0.91
    assert body["email_candidates"][0]["recommended_action"] == ACTION_DISCOUNT_CALL
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
    assert len(body["email_candidates"]) == 1
    assert body["email_candidates"][0]["index"] == 1
    assert body["email_candidates"][0]["p_churn"] == 0.87
    assert [item["index"] for item in body["results"]] == [0, 1]
    assert set(body) == {"status", "results", "email_candidates", "errors", "summary", "metadata", "timestamp"}
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
    assert body["email_candidates"] == []
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
    assert body["email_candidates"][0]["id"] == "cust-123"


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
    assert body["email_candidates"] == []
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


def test_batch_post_processing_is_null_safe_and_excludes_none_probability_from_candidates(monkeypatch):
    patch_batch_execution(monkeypatch, labels=[1, 1], probabilities=[None, 0.75])
    client = application.app.test_client()
    first = valid_record()
    first["customer_id"] = "cust-none-proba"
    second = valid_record()
    second["customer_id"] = "cust-keep"
    second["Age"] = 55
    payload = {"records": [first, second], "options": {"mode": "partial"}}

    response = client.post("/api/predict/batch", json=payload)

    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "success"
    assert len(body["results"]) == 2
    assert body["results"][0]["p_churn"] is None
    assert body["results"][0]["recommended_action"] is None
    assert body["results"][0]["net_gain"] is None
    assert body["results"][0]["clv"] is not None
    assert [item["id"] for item in body["email_candidates"]] == ["cust-keep"]
    assert body["email_candidates"][0]["recommended_action"] == ACTION_DISCOUNT_CALL


def test_batch_csv_happy_path_returns_same_contract_with_email_candidates(monkeypatch):
    patch_batch_execution(monkeypatch, labels=[1, 0], probabilities=[0.91, 0.08])
    client = application.app.test_client()
    first = valid_record()
    first["customer_id"] = "cust-1"
    second = valid_record()
    second["customer_id"] = "cust-2"
    second["Age"] = 50

    response = client.post(
        "/api/batch_predict_csv",
        data={
            **csv_upload_from_records([first, second]),
            "options": json.dumps({"mode": "partial"}),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    body = response.get_json()
    assert set(body) == {"status", "results", "email_candidates", "errors", "summary", "metadata", "timestamp"}
    assert body["status"] == "success"
    assert isinstance(body["email_candidates"], list)
    assert [item["id"] for item in body["email_candidates"]] == ["cust-1"]
    assert [item["index"] for item in body["results"]] == [0, 1]


def test_batch_csv_missing_file_returns_400():
    client = application.app.test_client()

    response = client.post(
        "/api/batch_predict_csv",
        data={},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    body = response.get_json()
    assert body["status"] == "error"
    assert "file" in body["message"]
    assert body["contract_version"] == "v1"


def test_batch_csv_invalid_options_json_returns_400():
    client = application.app.test_client()

    response = client.post(
        "/api/batch_predict_csv",
        data={
            **csv_upload_from_records([valid_record()]),
            "options": "{bad-json",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    body = response.get_json()
    assert body["status"] == "error"
    assert "Invalid options JSON" in body["message"]
    assert body["contract_version"] == "v1"


def test_batch_csv_missing_required_column_returns_400():
    client = application.app.test_client()
    csv_body = (
        "CreditScore,Geography,Gender,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary\n"
        "619,France,Female,2,0,1,1,1,101348.88\n"
    ).encode("utf-8")

    response = client.post(
        "/api/batch_predict_csv",
        data={"file": (io.BytesIO(csv_body), "missing_age.csv")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    body = response.get_json()
    assert body["status"] == "error"
    assert "missing required columns" in body["message"]
    assert "Age" in body["message"]
    assert body["contract_version"] == "v1"


def test_batch_csv_over_limit_returns_413():
    client = application.app.test_client()
    records = [valid_record() for _ in range(MAX_BATCH_SIZE + 1)]

    response = client.post(
        "/api/batch_predict_csv",
        data=csv_upload_from_records(records),
        content_type="multipart/form-data",
    )

    assert response.status_code == 413
    body = response.get_json()
    assert body["status"] == "error"
    assert body["contract_version"] == "v1"
    assert "MAX_BATCH_SIZE" in body["message"]


def test_batch_csv_null_safe_candidates_exclude_none_probability(monkeypatch):
    patch_batch_execution(monkeypatch, labels=[1, 1], probabilities=[None, 0.75])
    client = application.app.test_client()
    first = valid_record()
    first["customer_id"] = "cust-none-proba"
    second = valid_record()
    second["customer_id"] = "cust-keep"
    second["Age"] = 55

    response = client.post(
        "/api/batch_predict_csv",
        data={
            **csv_upload_from_records([first, second]),
            "options": json.dumps({"mode": "partial"}),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "success"
    assert len(body["results"]) == 2
    assert body["results"][0]["p_churn"] is None
    assert isinstance(body["email_candidates"], list)
    assert [item["id"] for item in body["email_candidates"]] == ["cust-keep"]
