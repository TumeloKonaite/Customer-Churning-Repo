"""Thin HTTP adapters for single, JSON batch, and CSV batch predictions."""

from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from src.schemas.batch_prediction import (
    BATCH_PREDICTION_EXAMPLE,
    MAX_BATCH_SIZE,
    BatchResponse,
)
from src.schemas.errors import BatchContractError, StandardAPIError
from src.schemas.prediction import SINGLE_PREDICTION_EXAMPLE, SinglePredictionResponse
from src.services import batch_prediction_service, single_prediction_service
from src.services.exceptions import APIServiceError


router = APIRouter()
JSON_REQUEST_BODY = Annotated[Any, Body()]


def require_json_content_type(request: Request) -> None:
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type != "application/json" and not media_type.endswith("+json"):
        raise APIServiceError("Content-Type must be application/json", status_code=415)


@router.post(
    "/api/predict",
    response_model=SinglePredictionResponse,
    responses={
        400: {"model": StandardAPIError, "description": "Invalid JSON or customer fields"},
        415: {"model": StandardAPIError, "description": "Unsupported content type"},
        503: {"model": StandardAPIError, "description": "Model artifacts are unavailable"},
        500: {"model": StandardAPIError, "description": "Prediction failure"},
    },
    tags=["Predictions"],
    summary="Predict churn for one customer",
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
    payload: JSON_REQUEST_BODY,
    _: None = Depends(require_json_content_type),
):
    return single_prediction_service.predict_single(payload)


BATCH_BAD_REQUEST_MODEL = BatchResponse | BatchContractError
BATCH_RESPONSES = {
    400: {
        "model": BATCH_BAD_REQUEST_MODEL,
        "description": "Batch contract or row validation error",
    },
    413: {"model": BatchContractError, "description": f"More than {MAX_BATCH_SIZE} records"},
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


def _batch_response(payload: Any) -> JSONResponse:
    body = batch_prediction_service.predict_batch_payload(payload)
    status_code = 400 if body.get("status") == "error" else 200
    return JSONResponse(body, status_code=status_code)


@router.post(
    "/api/predict/batch",
    response_model=BatchResponse,
    responses=BATCH_RESPONSES,
    tags=["Predictions"],
    summary="Predict churn for a JSON batch",
    openapi_extra=BATCH_OPENAPI_BODY,
)
def predict_batch_api(
    payload: JSON_REQUEST_BODY,
    _: None = Depends(require_json_content_type),
):
    return _batch_response(payload)


@router.post(
    "/api/batch_predict",
    response_model=BatchResponse,
    responses=BATCH_RESPONSES,
    tags=["Predictions"],
    summary="Predict churn for a JSON batch (compatibility alias)",
    openapi_extra=BATCH_OPENAPI_BODY,
)
def predict_batch_api_alias(
    payload: JSON_REQUEST_BODY,
    _: None = Depends(require_json_content_type),
):
    return _batch_response(payload)


@router.post(
    "/api/batch_predict_csv",
    response_model=BatchResponse,
    responses={
        400: {
            "model": BATCH_BAD_REQUEST_MODEL,
            "description": "CSV or batch contract validation error",
        },
        413: {"model": BatchContractError, "description": f"More than {MAX_BATCH_SIZE} records"},
        503: {"model": StandardAPIError, "description": "Model artifacts are unavailable"},
        500: {"model": StandardAPIError, "description": "Prediction failure"},
    },
    tags=["Predictions"],
    summary="Predict churn from a CSV file",
)
def predict_batch_csv_api(
    file: UploadFile = File(description="CSV containing the ten required model fields"),
    options: str | None = Form(
        default=None,
        description='Optional JSON object, for example {"mode":"partial"}',
        examples=['{"mode":"partial"}'],
    ),
):
    body = batch_prediction_service.predict_csv_batch(file.filename, file.file, options)
    status_code = 400 if body.get("status") == "error" else 200
    return JSONResponse(body, status_code=status_code)
