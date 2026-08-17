"""Single-customer prediction orchestration."""

from datetime import datetime, timezone
import logging
from typing import Any

from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
from src.services import model_service
from src.services.exceptions import APIServiceError, PredictionExecutionError
from src.services.prediction_validation import validate_payload


logger = logging.getLogger(__name__)


def _predict_one(data: dict) -> tuple[int, float | None]:
    customer = CustomData(
        credit_score=float(data["CreditScore"]),
        geography=str(data["Geography"]),
        gender=str(data["Gender"]),
        age=float(data["Age"]),
        tenure=float(data["Tenure"]),
        balance=float(data["Balance"]),
        num_of_products=float(data["NumOfProducts"]),
        has_cr_card=float(data["HasCrCard"]),
        is_active_member=float(data["IsActiveMember"]),
        estimated_salary=float(data["EstimatedSalary"]),
    )
    labels, probabilities = PredictPipeline().predict(customer.get_data_as_data_frame())
    probability = float(probabilities[0]) if probabilities is not None else None
    return int(labels[0]), probability


def predict_single(payload: Any) -> dict[str, Any]:
    """Validate, score, and construct the single-prediction response."""
    if payload is None:
        raise APIServiceError("Invalid JSON body")
    ok, errors = validate_payload(payload)
    if not ok:
        raise APIServiceError("Invalid input payload", errors=errors)
    model_service.ensure_artifacts_ready()

    try:
        label, probability = _predict_one(payload)
        metadata = model_service.prediction_metadata()
    except Exception as exc:
        logger.exception("Single prediction failed")
        raise PredictionExecutionError(f"Internal server error: {exc}") from exc

    return {
        "status": "success",
        "predicted_label": label,
        "p_churn": probability,
        **metadata,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
