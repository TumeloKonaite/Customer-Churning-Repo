"""Server-rendered prediction pages."""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from src.schemas.batch_prediction import MAX_BATCH_SIZE
from src.schemas.prediction import SinglePredictionRequest
from src.services import batch_prediction_service, model_service, single_prediction_service
from src.services.exceptions import APIServiceError


logger = logging.getLogger(__name__)
router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[3] / "templates"))
BATCH_UI_SAMPLE_OPTIONS = json.dumps({"mode": "partial"}, indent=2)


@router.api_route(
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
        try:
            model_service.ensure_artifacts_ready()
            options = batch_prediction_service.parse_batch_options_json(options_json)
            records = batch_prediction_service.parse_csv_upload_records(
                csv_file.filename if csv_file else None,
                csv_file.file if csv_file else None,
            )
            response_body = batch_prediction_service.predict_batch(records, options)
            response_status_code = 400 if response_body.get("status") == "error" else 200
        except APIServiceError as exc:
            error = exc.message
            response_status_code = exc.status_code
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


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@router.api_route(
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
        try:
            # Preserve the form's historical readiness-before-validation behavior.
            model_service.ensure_artifacts_ready()
            validated_form = SinglePredictionRequest.model_validate_strings(form_data)
            result = single_prediction_service.predict_single(validated_form)
            context["results"] = (
                "Customer is predicted to churn"
                if result["predicted_label"] == 1
                else "Customer is predicted to stay"
            )
            context["churn_probability"] = result["p_churn"]
        except APIServiceError as exc:
            if exc.errors:
                context["error"] = "; ".join(exc.errors)
            elif exc.status_code == 500 and exc.message.startswith("Internal server error: "):
                context["error"] = "Error processing request: " + exc.message.removeprefix(
                    "Internal server error: "
                )
            else:
                context["error"] = exc.message
        except ValidationError as exc:
            context["error"] = "; ".join(
                f"{error['loc'][-1]}: {error['msg']}" for error in exc.errors()
            )

    return templates.TemplateResponse(request=request, name="home.html", context=context)
