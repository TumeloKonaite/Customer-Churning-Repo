"""Flask entrypoint for training-backed customer churn prediction."""

from datetime import datetime, timezone
import json
import os

import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
from src.services.prediction_service import (
    MAX_BATCH_SIZE,
    REQUIRED_FIELDS,
    VALID_BATCH_MODES,
    predict_batch_records,
    validate_record,
)

application = Flask(__name__)
app = application

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
    return jsonify(payload), status_code


def batch_contract_error(message: str, status_code: int = 400):
    return jsonify(
        {
            "status": "error",
            "message": message,
            "contract_version": BATCH_CONTRACT_VERSION,
        }
    ), status_code


def validate_payload(data):
    ok, errors, _ = validate_record(data)
    return ok, (errors or None)


def artifacts_ready() -> bool:
    return all(os.path.exists(path) for path in REQUIRED_ARTIFACTS)


def parse_batch_options_json(options_raw: str):
    if options_raw is None or not str(options_raw).strip():
        return {}
    try:
        options = json.loads(options_raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid options JSON: {exc.msg}") from exc
    if not isinstance(options, dict):
        raise ValueError("Field 'options' must be an object")
    return options


def parse_csv_upload_records(uploaded_file):
    if uploaded_file is None:
        raise ValueError("Field 'file' is required")

    filename = (uploaded_file.filename or "").strip()
    if not filename:
        raise ValueError("Uploaded filename must not be empty")
    if not filename.lower().endswith(".csv"):
        raise ValueError("Uploaded file must be a .csv")

    try:
        frame = pd.read_csv(uploaded_file.stream)
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
        app.logger.exception("Batch prediction failed")
        return json_error(f"Internal server error: {exc}", status_code=500)
    status_code = 400 if response_body.get("status") == "error" else 200
    return jsonify(response_body), status_code


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


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_loaded": artifacts_ready(),
            "metadata": load_metadata(),
        }
    ), 200


@app.route("/api/predict", methods=["POST"])
def predict_api():
    if not request.is_json:
        return json_error("Content-Type must be application/json", status_code=415)
    data = request.get_json(silent=True)
    if data is None:
        return json_error("Invalid JSON body")

    ok, errors = validate_payload(data)
    if not ok:
        return json_error("Invalid input payload", errors=errors)
    if not artifacts_ready():
        return json_error(
            "Model artifacts are not ready yet. Please wait for training to finish.",
            status_code=503,
        )

    try:
        label, probability = _predict_one(data)
        metadata = load_metadata()
        return jsonify(
            {
                "status": "success",
                "predicted_label": label,
                "p_churn": probability,
                "model_name": metadata.get("model_name", "churn_predictor"),
                "model_version": metadata.get("version", "1.0.0"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ), 200
    except Exception as exc:
        app.logger.exception("Single prediction failed")
        return json_error(f"Internal server error: {exc}", status_code=500)


@app.route("/api/predict/batch", methods=["POST"])
@app.route("/api/batch_predict", methods=["POST"])
def predict_batch_api():
    if not request.is_json:
        return json_error("Content-Type must be application/json", status_code=415)
    body = request.get_json(silent=True)
    if body is None:
        return json_error("Invalid JSON body")
    if not isinstance(body, dict):
        return json_error("JSON body must be an object")
    if "records" not in body:
        return batch_contract_error("Field 'records' is required and must be a list")

    records = body.get("records")
    if not isinstance(records, list):
        return batch_contract_error("Field 'records' must be a list")
    if len(records) > MAX_BATCH_SIZE:
        return batch_contract_error(
            f"Batch size exceeds MAX_BATCH_SIZE ({MAX_BATCH_SIZE})", status_code=413
        )
    options = body.get("options", {})
    if not isinstance(options, dict):
        return batch_contract_error("Field 'options' must be an object")
    return execute_batch_prediction(records, options)


@app.route("/api/batch_predict_csv", methods=["POST"])
def predict_batch_csv_api():
    try:
        options = parse_batch_options_json(request.form.get("options", ""))
        records = parse_csv_upload_records(request.files.get("file"))
    except OverflowError as exc:
        return batch_contract_error(str(exc), status_code=413)
    except ValueError as exc:
        return batch_contract_error(str(exc))
    return execute_batch_prediction(records, options)


@app.route("/predictbatch", methods=["GET", "POST"])
def predict_batch_form():
    csv_options_json = BATCH_UI_SAMPLE_OPTIONS
    response_body = None
    response_status_code = None
    error = None
    uploaded_filename = None

    if request.method == "POST":
        csv_options_json = (request.form.get("csv_options_json") or "").strip()
        uploaded = request.files.get("csv_file")
        uploaded_filename = (uploaded.filename or "").strip() if uploaded else None
        if not artifacts_ready():
            error = "Model artifacts are not ready yet. Please wait for training to finish."
            response_status_code = 503
        else:
            try:
                options = parse_batch_options_json(csv_options_json)
                records = parse_csv_upload_records(uploaded)
                response_body = predict_batch_records(records, options)
                response_status_code = 400 if response_body.get("status") == "error" else 200
            except OverflowError as exc:
                error = str(exc)
                response_status_code = 413
            except ValueError as exc:
                error = str(exc)
                response_status_code = 400
            except Exception as exc:
                app.logger.exception("Batch prediction CSV form failed")
                error = f"Error processing CSV batch request: {exc}"
                response_status_code = 500

    return render_template(
        "batch.html",
        csv_options_json=csv_options_json,
        response_body=response_body,
        response_status_code=response_status_code,
        error=error,
        max_batch_size=MAX_BATCH_SIZE,
        uploaded_filename=uploaded_filename,
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    context = {"results": None, "churn_probability": None, "error": None}
    if request.method == "POST":
        if not artifacts_ready():
            context["error"] = "Model artifacts are not ready yet. Please wait for training to finish."
            return render_template("home.html", **context)

        form_data = request.form.to_dict()
        ok, errors = validate_payload(form_data)
        if not ok:
            context["error"] = "; ".join(errors)
            return render_template("home.html", **context)
        try:
            label, probability = _predict_one(form_data)
            context["results"] = "Customer is predicted to churn" if label == 1 else "Customer is predicted to stay"
            context["churn_probability"] = probability
        except Exception as exc:
            app.logger.exception("Prediction form failed")
            context["error"] = f"Error processing request: {exc}"
    return render_template("home.html", **context)


def run_app():
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    port = int(os.getenv("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    run_app()
