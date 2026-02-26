import application
from src.services.prediction_service import MAX_BATCH_SIZE


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


def test_batch_mode_fail_fast_rejects_invalid():
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
    assert body["mode"] == "fail_fast"
    assert len(body["errors"]) == 1
    assert body["errors"][0]["row_index"] == 1
    assert body["errors"][0]["field"] == "Age"
    assert body["errors"][0]["message"] == "Missing required field: Age"


def test_batch_mode_partial_collects_errors():
    client = application.app.test_client()
    missing_age = valid_record()
    missing_age.pop("Age")
    bad_balance = valid_record()
    bad_balance["Balance"] = "not-a-number"
    payload = {
        "records": [valid_record(), missing_age, bad_balance],
        "options": {"mode": "partial"},
    }

    response = client.post("/api/predict/batch", json=payload)

    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "partial"
    assert body["contract_version"] == "v1"
    assert body["mode"] == "partial"
    assert len(body["valid_rows"]) == 1
    assert len(body["errors"]) == 2
    assert [err["row_index"] for err in body["errors"]] == [1, 2]
    assert body["errors"][0]["field"] == "Age"
    assert body["errors"][0]["message"] == "Missing required field: Age"
    assert body["errors"][1]["field"] == "Balance"
    assert body["errors"][1]["message"].startswith("Field 'Balance' must be a number")
    assert body["row_map"]["0"] == 0
