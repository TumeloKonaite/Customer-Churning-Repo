import io

import pytest
from pydantic import ValidationError

from src.schemas.batch_prediction import MAX_BATCH_SIZE, BatchPredictionRequest
from src.schemas.prediction import REQUIRED_FIELDS
from src.services import batch_prediction_service, model_service, single_prediction_service
from src.services.exceptions import APIServiceError


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


def test_single_prediction_service_builds_the_public_contract(monkeypatch):
    monkeypatch.setattr(model_service, "artifacts_ready", lambda: True)
    monkeypatch.setattr(single_prediction_service, "_predict_one", lambda payload: (1, 0.81))
    monkeypatch.setattr(
        model_service,
        "prediction_metadata",
        lambda: {"model_name": "test-model", "model_version": "2.0.0"},
    )

    result = single_prediction_service.predict_single(valid_record())

    assert result["status"] == "success"
    assert result["predicted_label"] == 1
    assert result["p_churn"] == 0.81
    assert result["model_name"] == "test-model"
    assert result["model_version"] == "2.0.0"
    assert "timestamp" in result


def test_single_prediction_service_reports_all_missing_fields():
    record = valid_record()
    record.pop("Age")
    record.pop("Balance")

    with pytest.raises(APIServiceError) as caught:
        single_prediction_service.predict_single(record)

    assert caught.value.message == "Invalid input payload"
    assert caught.value.errors == [
        "Missing required field: Age",
        "Missing required field: Balance",
    ]


def test_batch_payload_validation_uses_the_pydantic_contract():
    with pytest.raises(ValidationError):
        BatchPredictionRequest.model_validate({})

    with pytest.raises(ValidationError):
        BatchPredictionRequest.model_validate(
            {"records": [valid_record()] * (MAX_BATCH_SIZE + 1)}
        )


def test_csv_parser_uses_the_canonical_prediction_fields():
    record = valid_record()
    csv = ",".join(REQUIRED_FIELDS) + "\n"
    csv += ",".join(str(record[field]) for field in REQUIRED_FIELDS) + "\n"

    records = batch_prediction_service.parse_csv_upload_records(
        "batch.csv", io.BytesIO(csv.encode())
    )

    assert len(records) == 1
    assert set(REQUIRED_FIELDS) <= set(records[0])
