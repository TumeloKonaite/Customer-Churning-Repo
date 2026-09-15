"""HTTP adapters for the single and JSON batch prediction endpoints."""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from src.schemas.batch_prediction import (
    BATCH_PREDICTION_EXAMPLE,
    BatchPredictionRequest,
    BatchResponse,
)
from src.schemas.errors import StandardAPIError
from src.schemas.prediction import (
    SINGLE_PREDICTION_EXAMPLE,
    SinglePredictionRequest,
    SinglePredictionResponse,
)
from src.services import batch_prediction_service, single_prediction_service
from src.services.exceptions import APIServiceError


router = APIRouter()


def require_json_content_type(request: Request) -> None:
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type != "application/json" and not media_type.endswith("+json"):
        raise APIServiceError("Content-Type must be application/json", status_code=415)


@router.post(
    "/api/predict",
    response_model=SinglePredictionResponse,
    responses={
        415: {"model": StandardAPIError, "description": "Unsupported content type"},
        503: {"model": StandardAPIError, "description": "Model artifacts are unavailable"},
        500: {"model": StandardAPIError, "description": "Prediction failure"},
    },
    tags=["Predictions"],
    summary="Predict churn for one customer",
    response_model_exclude_unset=True,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/SinglePredictionRequest"},
                    "example": SINGLE_PREDICTION_EXAMPLE,
                }
            },
        }
    },
)
def predict_api(
    payload: SinglePredictionRequest,
    _: None = Depends(require_json_content_type),
):
    return single_prediction_service.predict_single(payload)


BATCH_RESPONSES = {
    415: {"model": StandardAPIError, "description": "Unsupported content type"},
    503: {"model": StandardAPIError, "description": "Model artifacts are unavailable"},
    500: {"model": StandardAPIError, "description": "Prediction failure"},
}
BATCH_OPENAPI_BODY = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {
                "schema": {"$ref": "#/components/schemas/BatchPredictionRequest"},
                "example": BATCH_PREDICTION_EXAMPLE,
            }
        },
    }
}


def _batch_response(payload: BatchPredictionRequest) -> JSONResponse:
    body = batch_prediction_service.predict_batch(payload.records, payload.options.model_dump())
    status_code = 400 if body.get("status") == "error" else 200
    return JSONResponse(body, status_code=status_code)


@router.post(
    "/api/predict/batch",
    response_model=BatchResponse,
    responses=BATCH_RESPONSES,
    tags=["Predictions"],
    summary="Predict churn for a JSON batch",
    response_model_exclude_unset=True,
    openapi_extra=BATCH_OPENAPI_BODY,
)
def predict_batch_api(
    payload: BatchPredictionRequest,
    _: None = Depends(require_json_content_type),
):
    return _batch_response(payload)
