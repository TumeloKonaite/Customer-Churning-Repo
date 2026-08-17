"""FastAPI entrypoint for training-backed customer churn prediction."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from typing import Annotated, Any, Literal

import pandas as pd
from fastapi import Body, Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field

from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
from src.services.prediction_service import (
    MAX_BATCH_SIZE,
    REQUIRED_FIELDS,
    VALID_BATCH_MODES,
    predict_batch_records,
    validate_record,
)

logger = logging.getLogger(__name__)

BATCH_CONTRACT_VERSION = "v1"
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
REQUIRED_ARTIFACTS = [
    os.path.join(ARTIFACTS_DIR, "schema.json"),
    os.path.join(ARTIFACTS_DIR, "preprocessor.pkl"),
    os.path.join(ARTIFACTS_DIR, "encoder.pkl"),
    os.path.join(ARTIFACTS_DIR, "model.pkl"),
]
BATCH_UI_SAMPLE_OPTIONS = json.dumps({"mode": "partial"}, indent=2)

SINGLE_PREDICTION_EXAMPLE = {
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
BATCH_PREDICTION_EXAMPLE = {
    "records": [{"customer_id": "CUST_001", **SINGLE_PREDICTION_EXAMPLE}],
    "options": {"mode": "partial"},
}


class HealthResponse(BaseModel):
    """Service health and model-artifact readiness."""

    status: Literal["healthy"]
    timestamp: str
    model_loaded: bool
    metadata: dict[str, Any]


class SinglePredictionRequest(BaseModel):
    """The ten customer features accepted by the churn model."""

    model_config = ConfigDict(json_schema_extra={"examples": [SINGLE_PREDICTION_EXAMPLE]})

    CreditScore: float = Field(description="Customer credit score", examples=[619])
    Geography: str = Field(description="Customer country", examples=["France"])
    Gender: str = Field(description="Customer gender", examples=["Female"])
    Age: float = Field(description="Customer age in years", examples=[42])
    Tenure: float = Field(description="Years as a customer", examples=[2])
    Balance: float = Field(description="Account balance", examples=[0])
    NumOfProducts: float = Field(description="Number of bank products", examples=[1])
    HasCrCard: float = Field(description="Whether the customer has a credit card (0 or 1)", examples=[1])
    IsActiveMember: float = Field(description="Whether the customer is active (0 or 1)", examples=[1])
    EstimatedSalary: float = Field(description="Estimated annual salary", examples=[101348.88])


class SinglePredictionResponse(BaseModel):
    status: Literal["success"]
    predicted_label: int
    p_churn: float | None = Field(description="Churn probability, or null when unavailable")
    model_name: str
    model_version: str
    timestamp: str


class BatchPredictionRecord(BaseModel):
    """One batch record, with optional caller-provided identifier fields."""

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"examples": [BATCH_PREDICTION_EXAMPLE["records"][0]]},
    )

    CreditScore: float
    Geography: str
    Gender: str
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: float
    HasCrCard: float
    IsActiveMember: float
    EstimatedSalary: float
    customer_id: str | int | float | None = Field(default=None, description="Preferred optional record identifier")
    row_id: str | int | float | None = Field(default=None, description="Optional fallback record identifier")
    id: str | int | float | None = Field(default=None, description="Optional generic record identifier")


class BatchOptions(BaseModel):
    model_config = ConfigDict(extra="allow")

    mode: Literal["fail_fast", "partial"] = Field(
        default="fail_fast",
        description="fail_fast stops at the first invalid record; partial scores all valid records",
    )


class BatchPredictionRequest(BaseModel):
    """The JSON batch contract, containing at most 100 records."""

    model_config = ConfigDict(json_schema_extra={"examples": [BATCH_PREDICTION_EXAMPLE]})

    records: list[BatchPredictionRecord] = Field(max_length=MAX_BATCH_SIZE)
    options: BatchOptions = Field(default_factory=BatchOptions)


class BatchResultItem(BaseModel):
    index: int
    id: Any | None = Field(default=None, description="Passed through from customer_id, row_id, or id")
    predicted_label: int
    p_churn: float | None = Field(description="Churn probability, or null when unavailable")


class BatchValidationError(BaseModel):
    row_index: int
    id: Any | None = None
    message: str
    field: str | None = None


class BatchSummary(BaseModel):
    total_records: int
    valid_records: int
    invalid_records: int
    error_count: int
    mode: Literal["fail_fast", "partial"]


class BatchMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    model_name: str
    model_version: str


class BatchResponse(BaseModel):
    status: Literal["success", "partial", "failed", "error"]
    results: list[BatchResultItem]
    errors: list[BatchValidationError] | None
    summary: BatchSummary
    metadata: BatchMetadata
    timestamp: str


class StandardAPIError(BaseModel):
    status: Literal["error"]
    message: str
    errors: list[str] | None = None


class BatchContractError(BaseModel):
    status: Literal["error"]
    message: str
    contract_version: Literal["v1"]


app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Train-backed customer churn predictions for individual customers and JSON or CSV batches. "
        f"Batch requests support fail_fast and partial modes and at most {MAX_BATCH_SIZE} records."
    ),
    version="1.0.0",
    openapi_tags=[
        {"name": "Health", "description": "Service and model readiness."},
        {"name": "Predictions", "description": "Single and batch churn inference."},
    ],
)
# Keep the historical module-level name available to deployment/import callers.
application = app
templates = Jinja2Templates(directory=os.path.join(PROJECT_ROOT, "templates"))


def load_metadata():
    """Load model metadata without making health checks depend on it."""
    metadata_path = os.path.join(ARTIFACTS_DIR, "metadata.json")
    defaults = {"training_date": "unknown", "model_name": "churn_predictor"}
    try:
        with open(metadata_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return defaults


def json_error(message: str, status_code: int = 400, errors=None):
    payload = {"status": "error", "message": message}
    if errors:
        payload["errors"] = errors
    return JSONResponse(payload, status_code=status_code)


def batch_contract_error(message: str, status_code: int = 400):
    return JSONResponse(
        {
            "status": "error",
            "message": message,
            "contract_version": BATCH_CONTRACT_VERSION,
        },
        status_code=status_code,
    )


def validate_payload(data):
    ok, errors, _ = validate_record(data)
    return ok, (errors or None)


def artifacts_ready() -> bool:
    return all(os.path.exists(path) for path in REQUIRED_ARTIFACTS)


def parse_batch_options_json(options_raw: str | None):
    if options_raw is None or not str(options_raw).strip():
        return {}
    try:
        options = json.loads(options_raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid options JSON: {exc.msg}") from exc
    if not isinstance(options, dict):
        raise ValueError("Field 'options' must be an object")
    return options


def parse_csv_upload_records(uploaded_file: UploadFile | None):
    if uploaded_file is None:
        raise ValueError("Field 'file' is required")

    filename = (uploaded_file.filename or "").strip()
    if not filename:
        raise ValueError("Uploaded filename must not be empty")
    if not filename.lower().endswith(".csv"):
        raise ValueError("Uploaded file must be a .csv")

    try:
        frame = pd.read_csv(uploaded_file.file)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"CSV could not be parsed: {exc}") from exc

    frame = frame.dropna(how="all")
    if frame.empty:
        raise ValueError("CSV must contain at least one data row")
    missing_columns = [field for field in REQUIRED_FIELDS if field not in frame.columns]
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing_columns)}")

    records = frame.to_dict(orient="records")
    if len(records) > MAX_BATCH_SIZE:
        raise OverflowError(f"Batch size exceeds MAX_BATCH_SIZE ({MAX_BATCH_SIZE})")
    return records


def execute_batch_prediction(records, options):
    mode = options.get("mode", "fail_fast")
    if mode not in VALID_BATCH_MODES:
        return batch_contract_error("options.mode must be one of: fail_fast, partial")
    if not artifacts_ready():
        return json_error(
            "Model artifacts are not ready yet. Please wait for training to finish.",
            status_code=503,
        )
    try:
        response_body = predict_batch_records(records, options)
    except ValueError as exc:
        return batch_contract_error(str(exc))
    except Exception as exc:
        logger.exception("Batch prediction failed")
        return json_error(f"Internal server error: {exc}", status_code=500)
    status_code = 400 if response_body.get("status") == "error" else 200
    return JSONResponse(response_body, status_code=status_code)


def _predict_one(data):
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


def require_json_content_type(request: Request):
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type != "application/json" and not media_type.endswith("+json"):
        return json_error("Content-Type must be application/json", status_code=415)
    return None


@app.exception_handler(RequestValidationError)
def request_validation_error_handler(request: Request, exc: RequestValidationError):
    """Translate FastAPI's body parsing errors to the established 400 contracts."""
    if request.url.path == "/api/batch_predict_csv":
        return batch_contract_error("Field 'file' is required")
    return json_error("Invalid JSON body")


JSON_REQUEST_BODY = Annotated[Any, Body()]


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Check service and model readiness",
)
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_loaded": artifacts_ready(),
        "metadata": load_metadata(),
    }


@app.post(
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
    content_type_error: JSONResponse | None = Depends(require_json_content_type),
):
    if content_type_error is not None:
        return content_type_error
    if payload is None:
        return json_error("Invalid JSON body")

    ok, errors = validate_payload(payload)
    if not ok:
        return json_error("Invalid input payload", errors=errors)
    if not artifacts_ready():
        return json_error(
            "Model artifacts are not ready yet. Please wait for training to finish.",
            status_code=503,
        )

    try:
        label, probability = _predict_one(payload)
        metadata = load_metadata()
        return {
            "status": "success",
            "predicted_label": label,
            "p_churn": probability,
            "model_name": metadata.get("model_name", "churn_predictor"),
            "model_version": metadata.get("version", "1.0.0"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        logger.exception("Single prediction failed")
        return json_error(f"Internal server error: {exc}", status_code=500)


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


def _predict_batch_json(payload, content_type_error):
    if content_type_error is not None:
        return content_type_error
    if payload is None:
        return json_error("Invalid JSON body")
    if not isinstance(payload, dict):
        return json_error("JSON body must be an object")
    if "records" not in payload:
        return batch_contract_error("Field 'records' is required and must be a list")

    records = payload.get("records")
    if not isinstance(records, list):
        return batch_contract_error("Field 'records' must be a list")
    if len(records) > MAX_BATCH_SIZE:
        return batch_contract_error(
            f"Batch size exceeds MAX_BATCH_SIZE ({MAX_BATCH_SIZE})", status_code=413
        )
    options = payload.get("options", {})
    if not isinstance(options, dict):
        return batch_contract_error("Field 'options' must be an object")
    return execute_batch_prediction(records, options)


@app.post(
    "/api/predict/batch",
    response_model=BatchResponse,
    responses=BATCH_RESPONSES,
    tags=["Predictions"],
    summary="Predict churn for a JSON batch",
    openapi_extra=BATCH_OPENAPI_BODY,
)
def predict_batch_api(
    payload: JSON_REQUEST_BODY,
    content_type_error: JSONResponse | None = Depends(require_json_content_type),
):
    return _predict_batch_json(payload, content_type_error)


@app.post(
    "/api/batch_predict",
    response_model=BatchResponse,
    responses=BATCH_RESPONSES,
    tags=["Predictions"],
    summary="Predict churn for a JSON batch (compatibility alias)",
    openapi_extra=BATCH_OPENAPI_BODY,
)
def predict_batch_api_alias(
    payload: JSON_REQUEST_BODY,
    content_type_error: JSONResponse | None = Depends(require_json_content_type),
):
    return _predict_batch_json(payload, content_type_error)


@app.post(
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
    try:
        parsed_options = parse_batch_options_json(options)
        records = parse_csv_upload_records(file)
    except OverflowError as exc:
        return batch_contract_error(str(exc), status_code=413)
    except ValueError as exc:
        return batch_contract_error(str(exc))
    return execute_batch_prediction(records, parsed_options)


@app.api_route(
    "/predictbatch",
    methods=["GET", "POST"],
    response_class=HTMLResponse,
    include_in_schema=False,
)
def predict_batch_form(
    request: Request,
    csv_options_json: str = Form(default=BATCH_UI_SAMPLE_OPTIONS),
    csv_file: UploadFile | None = File(default=None),
):
    options_json = csv_options_json.strip()
    response_body = None
    response_status_code = None
    error = None
    uploaded_filename = None

    if request.method == "POST":
        uploaded_filename = (csv_file.filename or "").strip() if csv_file else None
        if not artifacts_ready():
            error = "Model artifacts are not ready yet. Please wait for training to finish."
            response_status_code = 503
        else:
            try:
                options = parse_batch_options_json(options_json)
                records = parse_csv_upload_records(csv_file)
                response_body = predict_batch_records(records, options)
                response_status_code = 400 if response_body.get("status") == "error" else 200
            except OverflowError as exc:
                error = str(exc)
                response_status_code = 413
            except ValueError as exc:
                error = str(exc)
                response_status_code = 400
            except Exception as exc:
                logger.exception("Batch prediction CSV form failed")
                error = f"Error processing CSV batch request: {exc}"
                response_status_code = 500

    return templates.TemplateResponse(
        request=request,
        name="batch.html",
        context={
            "csv_options_json": options_json,
            "response_body": response_body,
            "response_status_code": response_status_code,
            "error": error,
            "max_batch_size": MAX_BATCH_SIZE,
            "uploaded_filename": uploaded_filename,
        },
    )


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.api_route(
    "/predictdata",
    methods=["GET", "POST"],
    response_class=HTMLResponse,
    include_in_schema=False,
)
def predict_datapoint(
    request: Request,
    CreditScore: str | None = Form(default=None),
    Geography: str | None = Form(default=None),
    Gender: str | None = Form(default=None),
    Age: str | None = Form(default=None),
    Tenure: str | None = Form(default=None),
    Balance: str | None = Form(default=None),
    NumOfProducts: str | None = Form(default=None),
    HasCrCard: str | None = Form(default=None),
    IsActiveMember: str | None = Form(default=None),
    EstimatedSalary: str | None = Form(default=None),
):
    context = {"results": None, "churn_probability": None, "error": None}
    if request.method == "POST":
        if not artifacts_ready():
            context["error"] = "Model artifacts are not ready yet. Please wait for training to finish."
            return templates.TemplateResponse(request=request, name="home.html", context=context)

        form_data = {
            "CreditScore": CreditScore,
            "Geography": Geography,
            "Gender": Gender,
            "Age": Age,
            "Tenure": Tenure,
            "Balance": Balance,
            "NumOfProducts": NumOfProducts,
            "HasCrCard": HasCrCard,
            "IsActiveMember": IsActiveMember,
            "EstimatedSalary": EstimatedSalary,
        }
        ok, errors = validate_payload(form_data)
        if not ok:
            context["error"] = "; ".join(errors)
            return templates.TemplateResponse(request=request, name="home.html", context=context)
        try:
            label, probability = _predict_one(form_data)
            context["results"] = (
                "Customer is predicted to churn" if label == 1 else "Customer is predicted to stay"
            )
            context["churn_probability"] = probability
        except Exception as exc:
            logger.exception("Prediction form failed")
            context["error"] = f"Error processing request: {exc}"
    return templates.TemplateResponse(request=request, name="home.html", context=context)


def custom_openapi():
    """Add documentation-only request schemas without changing legacy validation behavior."""
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
    )
    schemas = schema.setdefault("components", {}).setdefault("schemas", {})
    for model in (
        SinglePredictionRequest,
        BatchPredictionRecord,
        BatchOptions,
        BatchPredictionRequest,
        BatchContractError,
    ):
        schemas[model.__name__] = model.model_json_schema(
            ref_template="#/components/schemas/{model}"
        )
    csv_body_schema = schema["paths"]["/api/batch_predict_csv"]["post"]["requestBody"][
        "content"
    ]["multipart/form-data"]["schema"]
    if "$ref" in csv_body_schema:
        csv_body_schema = schemas[csv_body_schema["$ref"].rsplit("/", 1)[-1]]
    # OpenAPI file uploads conventionally use string/binary. Pydantic 2 emits
    # contentMediaType, so retain it and add format for Swagger/schema clients.
    csv_body_schema["properties"]["file"]["format"] = "binary"
    for path_item in schema["paths"].values():
        for operation in path_item.values():
            if isinstance(operation, dict):
                operation.get("responses", {}).pop("422", None)
    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi


def run_app():
    """Run the ASGI application locally through Uvicorn."""
    import uvicorn

    port = int(os.getenv("PORT", "5001"))
    reload_enabled = os.getenv("UVICORN_RELOAD", "0") == "1"
    uvicorn.run("application:app", host="0.0.0.0", port=port, reload=reload_enabled)


if __name__ == "__main__":
    run_app()
