from flask import Flask, request, render_template, jsonify
from datetime import datetime
import json
import os

import pandas as pd

from src.decisioning import (
    ACTION_COSTS,
    estimate_clv,
    expected_net_gain,
    recommended_action,
)
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

# -----------------------------
# Helpers
# -----------------------------

BATCH_CONTRACT_VERSION = "v1"
BATCH_UI_SAMPLE_PAYLOAD = {
    "records": [
        {
            "customer_id": "CUST_001",
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
        },
        {
            "customer_id": "CUST_002",
            "CreditScore": 700,
            "Geography": "Germany",
            "Gender": "Male",
            "Age": 50,
            "Tenure": 5,
            "Balance": 120000,
            "NumOfProducts": 2,
            "HasCrCard": 1,
            "IsActiveMember": 0,
            "EstimatedSalary": 90000,
        },
    ],
    "options": {
        "mode": "partial",
        "email_candidate_rules": {
            "min_p_churn": 0.6,
            "min_net_gain": 0,
            "exclude_no_action": True,
        },
    },
}
BATCH_UI_SAMPLE_CSV_OPTIONS = json.dumps(BATCH_UI_SAMPLE_PAYLOAD["options"], indent=2)

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")

REQUIRED_ARTIFACTS = [
    os.path.join(ARTIFACTS_DIR, "schema.json"),
    os.path.join(ARTIFACTS_DIR, "preprocessor.pkl"),
    os.path.join(ARTIFACTS_DIR, "encoder.pkl"),
    os.path.join(ARTIFACTS_DIR, "model.pkl"),
]


def load_metadata():
    """Load model metadata if present."""
    metadata_path = os.path.join(ARTIFACTS_DIR, "metadata.json")
    try:
        with open(metadata_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"training_date": "unknown", "model_name": "churn_predictor"}
    except Exception:
        # If metadata file exists but is malformed, don't break the app
        return {"training_date": "unknown", "model_name": "churn_predictor"}


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


def validate_payload(data: dict):
    """Validate required fields and numeric coercion with useful error messages."""
    ok, errors, _ = validate_record(data)
    return ok, (errors or None)


def artifacts_ready() -> bool:
    return all(os.path.exists(path) for path in REQUIRED_ARTIFACTS)


def batch_ui_default_payload() -> str:
    return json.dumps(BATCH_UI_SAMPLE_PAYLOAD, indent=2)


def batch_ui_default_options() -> str:
    return BATCH_UI_SAMPLE_CSV_OPTIONS


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
        df = pd.read_csv(uploaded_file.stream)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"CSV could not be parsed: {str(exc)}") from exc

    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("CSV must contain at least one data row")

    missing_columns = [field for field in REQUIRED_FIELDS if field not in df.columns]
    if missing_columns:
        missing_list = ", ".join(missing_columns)
        raise ValueError(f"CSV is missing required columns: {missing_list}")

    records = df.to_dict(orient="records")
    if len(records) > MAX_BATCH_SIZE:
        raise OverflowError(f"Batch size exceeds MAX_BATCH_SIZE ({MAX_BATCH_SIZE})")

    return records


def execute_batch_prediction(records, options):
    mode = options.get("mode", "fail_fast")
    if mode not in VALID_BATCH_MODES:
        return batch_contract_error(
            "options.mode must be one of: fail_fast, partial",
            status_code=400,
        )

    if not artifacts_ready():
        return json_error(
            "Model artifacts are not ready yet. Please wait for training to finish.",
            status_code=503,
        )

    try:
        response_body = predict_batch_records(records, options)
    except ValueError as exc:
        return batch_contract_error(str(exc), status_code=400)
    except Exception as exc:
        return json_error(f"Internal server error: {str(exc)}", status_code=500)

    http_status = 400 if response_body.get("status") == "error" else 200
    return jsonify(response_body), http_status


# -----------------------------
# Routes
# -----------------------------

@app.route("/health", methods=["GET"])
def health_check():
    # Keep it predictable for Docker healthchecks / tests.
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": artifacts_ready(),
            "metadata": load_metadata(),
        }
    ), 200


@app.route("/api/predict", methods=["POST"])
def predict_api():
    # Ensure JSON requests are handled properly
    if not request.is_json:
        return json_error(
            "Content-Type must be application/json",
            status_code=415,
        )

    data = request.get_json(silent=True)
    if data is None:
        return json_error("Invalid JSON body", status_code=400)

    ok, errors = validate_payload(data)
    if not ok:
        return json_error("Invalid input payload", status_code=400, errors=errors)

    if not artifacts_ready():
        return json_error(
            "Model artifacts are not ready yet. Please wait for training to finish.",
            status_code=503,
        )

    try:
        # Create CustomData instance (now safe to cast)
        data_instance = CustomData(
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

        pred_df = data_instance.get_data_as_data_frame()
        pipeline = PredictPipeline()
        result, proba = pipeline.predict(pred_df)

        churn_probability = float(proba[0]) if proba is not None else None

        # Calculate business metrics (cast to floats for consistency)
        clv = estimate_clv(
            {
                "Balance": float(data["Balance"]),
                "Tenure": float(data["Tenure"]),
                "EstimatedSalary": float(data["EstimatedSalary"]),
            }
        )
        action = None
        net_gain = None
        if churn_probability is not None:
            action = recommended_action(churn_probability)
            action_cost = float(ACTION_COSTS.get(action, 0.0))
            net_gain = expected_net_gain(churn_probability, clv, action_cost)

        metadata = load_metadata()

        return jsonify(
            {
                "status": "success",
                "p_churn": churn_probability,              # required by your tests
                "predicted_label": int(result[0]),
                "clv": clv,
                "recommended_action": action,
                "net_gain": net_gain,
                "model_name": metadata.get("model_name"),
                "model_version": metadata.get("version", "1.0.0"),
                "timestamp": datetime.now().isoformat(),
            }
        ), 200

    except Exception as e:
        # If something unexpected happens (model load, pipeline error, etc.)
        return json_error(f"Internal server error: {str(e)}", status_code=500)


@app.route("/api/predict/batch", methods=["POST"])
@app.route("/api/batch_predict", methods=["POST"])
def predict_batch_api():
    if not request.is_json:
        return json_error(
            "Content-Type must be application/json",
            status_code=415,
        )

    body = request.get_json(silent=True)
    if body is None:
        return json_error("Invalid JSON body", status_code=400)
    if not isinstance(body, dict):
        return json_error("JSON body must be an object", status_code=400)

    if "records" not in body:
        return batch_contract_error(
            "Field 'records' is required and must be a list",
            status_code=400,
        )

    records = body.get("records")
    if not isinstance(records, list):
        return batch_contract_error("Field 'records' must be a list", status_code=400)

    if len(records) > MAX_BATCH_SIZE:
        return batch_contract_error(
            f"Batch size exceeds MAX_BATCH_SIZE ({MAX_BATCH_SIZE})",
            status_code=413,
        )

    options = body.get("options", {})
    if not isinstance(options, dict):
        return batch_contract_error("Field 'options' must be an object", status_code=400)

    return execute_batch_prediction(records, options)


@app.route("/api/batch_predict_csv", methods=["POST"])
def predict_batch_csv_api():
    try:
        options = parse_batch_options_json(request.form.get("options", ""))
        records = parse_csv_upload_records(request.files.get("file"))
    except OverflowError as exc:
        return batch_contract_error(str(exc), status_code=413)
    except ValueError as exc:
        return batch_contract_error(str(exc), status_code=400)

    return execute_batch_prediction(records, options)


@app.route("/predictbatch", methods=["GET", "POST"])
def predict_batch_form():
    csv_options_json = batch_ui_default_options()
    response_body = None
    response_status_code = None
    error = None
    input_method = "csv"
    uploaded_filename = None

    if request.method == "POST":
        csv_options_json = (request.form.get("csv_options_json") or "").strip()
        uploaded = request.files.get("csv_file")
        uploaded_filename = (uploaded.filename or "").strip() if uploaded else None

        if not artifacts_ready():
            error = "Model artifacts are not ready yet. Please wait for training to finish."
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
            except Exception as exc:
                app.logger.exception("Batch prediction CSV form failed")
                error = f"Error processing CSV batch request: {str(exc)}"

    return render_template(
        "batch.html",
        csv_options_json=csv_options_json,
        response_body=response_body,
        response_status_code=response_status_code,
        error=error,
        max_batch_size=MAX_BATCH_SIZE,
        input_method=input_method,
        uploaded_filename=uploaded_filename,
    )


# Home page
@app.route("/")
def index():
    return render_template("index.html")


# Web form endpoint (unchanged behavior, but still safe)
@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "POST":
        try:
            if not artifacts_ready():
                return render_template(
                    "home.html",
                    error="Model artifacts are not ready yet. Please wait for training to finish.",
                    results=None,
                    churn_probability=None,
                    clv=None,
                    action=None,
                    net_gain=None,
                )

            credit_score = float(request.form.get("CreditScore"))
            geography = request.form.get("Geography")
            gender = request.form.get("Gender")
            age = float(request.form.get("Age"))
            tenure = float(request.form.get("Tenure"))
            balance = float(request.form.get("Balance"))
            num_of_products = float(request.form.get("NumOfProducts"))
            has_cr_card = float(request.form.get("HasCrCard"))
            is_active_member = float(request.form.get("IsActiveMember"))
            estimated_salary = float(request.form.get("EstimatedSalary"))

            data = CustomData(
                credit_score=credit_score,
                geography=geography,
                gender=gender,
                age=age,
                tenure=tenure,
                balance=balance,
                num_of_products=num_of_products,
                has_cr_card=has_cr_card,
                is_active_member=is_active_member,
                estimated_salary=estimated_salary,
            )

            pred_df = data.get_data_as_data_frame()
            pipeline = PredictPipeline()
            result, proba = pipeline.predict(pred_df)

            if result[0] == 1:
                pred_text = "This customer is at high risk of leaving. Immediate retention actions are recommended."
            else:
                pred_text = "This customer is likely to stay. No urgent retention action needed."

            churn_probability = None
            clv = None
            action = None
            net_gain = None

            if proba is not None:
                churn_probability = float(proba[0])
                clv = estimate_clv(
                    {
                        "Balance": balance,
                        "Tenure": tenure,
                        "EstimatedSalary": estimated_salary,
                    }
                )
                action = recommended_action(churn_probability)
                action_cost = ACTION_COSTS.get(action, 0.0)
                net_gain = expected_net_gain(churn_probability, clv, action_cost)
                pred_text = f"{pred_text} (Churn probability: {churn_probability:.2%})"

            return render_template(
                "home.html",
                results=pred_text,
                churn_probability=churn_probability,
                clv=clv,
                action=action,
                net_gain=net_gain,
            )

        except Exception as e:
            app.logger.exception("Prediction form failed")
            return render_template(
                "home.html",
                error=f"Error processing request: {str(e)}",
                results=None,
                churn_probability=None,
                clv=None,
                action=None,
                net_gain=None,
            )

    return render_template(
        "home.html",
        results=None,
        churn_probability=None,
        clv=None,
        action=None,
        net_gain=None,
    )


def run_app():
    # Binding to 0.0.0.0 is required for container access
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    port = int(os.getenv("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    run_app()
