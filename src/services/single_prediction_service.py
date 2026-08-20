"""Single-customer prediction orchestration."""

from datetime import datetime, timezone
import logging
from typing import Any

import pandas as pd

from src.pipeline.prediction_pipeline import PredictPipeline
from src.schemas.prediction import SinglePredictionRequest
from src.services import model_service
from src.services.exceptions import APIServiceError, PredictionExecutionError
from src.services.prediction_validation import validate_record


logger = logging.getLogger(__name__)


def _predict_one(request: SinglePredictionRequest) -> tuple[int, float | None]:
    record = pd.DataFrame([request.model_dump()])
    labels, probabilities = PredictPipeline().predict(record)
    probability = float(probabilities[0]) if probabilities is not None else None
    return int(labels[0]), probability


def predict_single(payload: Any) -> dict[str, Any]:
    """Score a validated request and construct the public response."""
    if payload is None:
        raise APIServiceError("Invalid JSON body")
    if isinstance(payload, SinglePredictionRequest):
        request = payload
    else:
        ok, errors, canonical = validate_record(payload, allow_identifiers=False)
        if not ok:
            raise APIServiceError("Invalid input payload", errors=errors)
        request = SinglePredictionRequest.model_validate(canonical)

    model_service.ensure_artifacts_ready()
    try:
        label, probability = _predict_one(request)
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
