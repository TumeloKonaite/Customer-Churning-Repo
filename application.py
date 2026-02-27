from flask import Flask, request, render_template, jsonify
from datetime import datetime, timezone
import json
import os
import re

import pandas as pd

from src.decisioning import (
    ACTION_COSTS,
    estimate_clv,
    expected_net_gain,
    recommended_action,
)
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
from src.agents.formatter_agents import subject_tool as outreach_subject_tool
from src.agents.retention_writers import (
    write_retention_email_concise,
    write_retention_email_serious,
    write_retention_email_witty,
)
from src.agents.tools_email import send_email_text
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
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

OUTREACH_CONTRACT_VERSION = "v1"
DEFAULT_OUTREACH_THRESHOLD = 0.65
MAX_EMAILS_PER_REQUEST = 50
DEFAULT_OUTREACH_MAX_EMAILS = 50
DEFAULT_OUTREACH_DRY_RUN = True
DEFAULT_OUTREACH_TONE = "serious"
VALID_OUTREACH_TONES = {"serious", "witty", "concise"}

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


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _outreach_send_block() -> dict:
    return {"attempted": False, "sent": 0, "results": []}


def _outreach_summary(
    *,
    n_records: int,
    n_valid: int,
    n_invalid: int,
    n_selected: int,
    threshold: float,
    max_emails: int,
    dry_run: bool,
) -> dict:
    return {
        "n_records": int(n_records),
        "n_valid": int(n_valid),
        "n_invalid": int(n_invalid),
        "n_selected": int(n_selected),
        "threshold": float(threshold),
        "max_emails": int(max_emails),
        "dry_run": bool(dry_run),
    }


def _outreach_envelope(
    *,
    status: str,
    summary: dict,
    selected: list | None = None,
    send: dict | None = None,
    errors: list | None = None,
) -> dict:
    return {
        "contract_version": OUTREACH_CONTRACT_VERSION,
        "status": status,
        "summary": summary,
        "selected": selected or [],
        "send": send or _outreach_send_block(),
        "errors": errors or [],
        "timestamp": _timestamp_utc(),
    }


def _normalize_email(value):
    if not isinstance(value, str):
        return None
    email = value.strip().lower()
    if not email:
        return None
    if not _EMAIL_RE.match(email):
        return None
    return email


def _coerce_probability(value):
    try:
        p_churn = float(value)
    except (TypeError, ValueError):
        return None
    if p_churn < 0 or p_churn > 1:
        return None
    return p_churn


def _coerce_positive_int(value):
    try:
        int_value = int(value)
    except (TypeError, ValueError):
        return None
    if int_value <= 0:
        return None
    return int_value


def _extract_record_id(record, fallback_index):
    if not isinstance(record, dict):
        return f"idx-{fallback_index}"

    for key in ("id", "customer_id", "row_id"):
        if key in record and record.get(key) is not None and str(record.get(key)).strip():
            return str(record.get(key)).strip()
    return f"idx-{fallback_index}"


def _writer_for_tone(tone):
    writers = {
        "serious": write_retention_email_serious,
        "witty": write_retention_email_witty,
        "concise": write_retention_email_concise,
    }
    return writers[tone]


def _build_outreach_prompt(
    *,
    tone: str,
    company_name: str,
    from_name: str,
    recipient_id: str,
    recipient_email: str,
    p_churn: float,
):
    return (
        f"Write a {tone} retention outreach email from {from_name} at {company_name} "
        f"to customer {recipient_id} ({recipient_email}). "
        f"Customer churn probability is {p_churn:.2f}. "
        "Do not mention scoring. Include one clear next step."
    )


def _sendgrid_ready():
    api_key = str(os.getenv("SENDGRID_API_KEY", "")).strip()
    return bool(api_key)


def _send_result_ok(send_result):
    if isinstance(send_result, dict):
        if "ok" in send_result:
            return bool(send_result.get("ok"))
        status_code = send_result.get("status_code")
        if status_code is not None:
            try:
                return 200 <= int(status_code) < 300
            except (TypeError, ValueError):
                return False
    return bool(send_result)


def _select_outreach_recipients(
    *,
    batch_results,
    records,
    threshold,
    max_emails,
):
    ranked = []
    errors = []
    for row_position, result in enumerate(batch_results):
        if not isinstance(result, dict):
            continue

        p_churn = _coerce_probability(result.get("p_churn"))
        if p_churn is None or p_churn < threshold:
            continue

        try:
            index = int(result.get("index", row_position))
        except (TypeError, ValueError):
            errors.append(
                {
                    "stage": "targeting",
                    "row_index": row_position,
                    "message": "Result index is invalid and row was skipped",
                }
            )
            continue
        if index < 0 or index >= len(records):
            errors.append(
                {
                    "stage": "targeting",
                    "row_index": row_position,
                    "message": "Result index is out of range and row was skipped",
                }
            )
            continue

        record = records[index]
        email = _normalize_email(result.get("email"))
        if email is None and isinstance(record, dict):
            email = _normalize_email(record.get("email"))
        if email is None:
            errors.append(
                {
                    "stage": "targeting",
                    "row_index": index,
                    "id": result.get("id") or _extract_record_id(record, index),
                    "message": "Selected row is missing a valid 'email'",
                }
            )
            continue

        record_id = result.get("id")
        if record_id is None or not str(record_id).strip():
            record_id = _extract_record_id(record, index)
        else:
            record_id = str(record_id).strip()

        ranked.append(
            {
                "id": record_id,
                "index": index,
                "email": email,
                "p_churn": p_churn,
            }
        )

    ranked.sort(key=lambda item: (-item["p_churn"], item["index"], item["id"]))
    return ranked[:max_emails], errors


def _parse_outreach_request(body):
    errors = []

    if not isinstance(body, dict):
        return None, [{"stage": "request", "message": "JSON body must be an object"}]

    if body.get("contract_version") != OUTREACH_CONTRACT_VERSION:
        errors.append(
            {
                "stage": "request",
                "field": "contract_version",
                "message": "contract_version must be 'v1'",
            }
        )

    records = body.get("records")
    if not isinstance(records, list) or not records:
        errors.append(
            {
                "stage": "request",
                "field": "records",
                "message": "records must be a non-empty list",
            }
        )
        records = []
    elif len(records) > MAX_BATCH_SIZE:
        errors.append(
            {
                "stage": "request",
                "field": "records",
                "message": f"Batch size exceeds MAX_BATCH_SIZE ({MAX_BATCH_SIZE})",
            }
        )

    outreach_options = body.get("outreach_options", {})
    if not isinstance(outreach_options, dict):
        errors.append(
            {
                "stage": "request",
                "field": "outreach_options",
                "message": "outreach_options must be an object",
            }
        )
        outreach_options = {}

    threshold = _coerce_probability(outreach_options.get("threshold", DEFAULT_OUTREACH_THRESHOLD))
    if threshold is None:
        errors.append(
            {
                "stage": "request",
                "field": "outreach_options.threshold",
                "message": "threshold must be a number between 0 and 1",
            }
        )
        threshold = DEFAULT_OUTREACH_THRESHOLD

    max_emails = _coerce_positive_int(outreach_options.get("max_emails", DEFAULT_OUTREACH_MAX_EMAILS))
    if max_emails is None:
        errors.append(
            {
                "stage": "request",
                "field": "outreach_options.max_emails",
                "message": "max_emails must be an integer greater than 0",
            }
        )
        max_emails = DEFAULT_OUTREACH_MAX_EMAILS
    elif max_emails > MAX_EMAILS_PER_REQUEST:
        errors.append(
            {
                "stage": "request",
                "field": "outreach_options.max_emails",
                "message": f"max_emails exceeds MAX_EMAILS_PER_REQUEST ({MAX_EMAILS_PER_REQUEST})",
            }
        )

    dry_run = outreach_options.get("dry_run", DEFAULT_OUTREACH_DRY_RUN)
    if not isinstance(dry_run, bool):
        errors.append(
            {
                "stage": "request",
                "field": "outreach_options.dry_run",
                "message": "dry_run must be a boolean",
            }
        )
        dry_run = DEFAULT_OUTREACH_DRY_RUN

    tone_raw = outreach_options.get("tone", DEFAULT_OUTREACH_TONE)
    tone = str(tone_raw).strip().lower()
    if tone not in VALID_OUTREACH_TONES:
        errors.append(
            {
                "stage": "request",
                "field": "outreach_options.tone",
                "message": "tone must be one of: serious, witty, concise",
            }
        )
        tone = DEFAULT_OUTREACH_TONE

    context = body.get("context")
    if not isinstance(context, dict):
        errors.append(
            {
                "stage": "request",
                "field": "context",
                "message": "context must be an object",
            }
        )
        context = {}

    company_name = str(context.get("company_name", "")).strip()
    if not company_name:
        errors.append(
            {
                "stage": "request",
                "field": "context.company_name",
                "message": "context.company_name must be a non-empty string",
            }
        )

    from_name = str(context.get("from_name", "")).strip()
    if not from_name:
        errors.append(
            {
                "stage": "request",
                "field": "context.from_name",
                "message": "context.from_name must be a non-empty string",
            }
        )

    from_email_raw = context.get("from_email")
    from_email = _normalize_email(from_email_raw)
    if from_email is None:
        errors.append(
            {
                "stage": "request",
                "field": "context.from_email",
                "message": "context.from_email must be a valid email address",
            }
        )

    parsed = {
        "records": records,
        "threshold": threshold,
        "max_emails": max_emails,
        "dry_run": dry_run,
        "tone": tone,
        "company_name": company_name,
        "from_name": from_name,
        "from_email": from_email,
    }
    return parsed, errors


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


@app.route("/api/outreach", methods=["POST"])
def outreach_api():
    if not request.is_json:
        summary = _outreach_summary(
            n_records=0,
            n_valid=0,
            n_invalid=0,
            n_selected=0,
            threshold=DEFAULT_OUTREACH_THRESHOLD,
            max_emails=DEFAULT_OUTREACH_MAX_EMAILS,
            dry_run=DEFAULT_OUTREACH_DRY_RUN,
        )
        response_body = _outreach_envelope(
            status="error",
            summary=summary,
            errors=[{"stage": "request", "message": "Content-Type must be application/json"}],
        )
        return jsonify(response_body), 415

    body = request.get_json(silent=True)
    if body is None:
        summary = _outreach_summary(
            n_records=0,
            n_valid=0,
            n_invalid=0,
            n_selected=0,
            threshold=DEFAULT_OUTREACH_THRESHOLD,
            max_emails=DEFAULT_OUTREACH_MAX_EMAILS,
            dry_run=DEFAULT_OUTREACH_DRY_RUN,
        )
        response_body = _outreach_envelope(
            status="error",
            summary=summary,
            errors=[{"stage": "request", "message": "Invalid JSON body"}],
        )
        return jsonify(response_body), 400

    parsed, validation_errors = _parse_outreach_request(body)
    if parsed is None:
        summary = _outreach_summary(
            n_records=0,
            n_valid=0,
            n_invalid=0,
            n_selected=0,
            threshold=DEFAULT_OUTREACH_THRESHOLD,
            max_emails=DEFAULT_OUTREACH_MAX_EMAILS,
            dry_run=DEFAULT_OUTREACH_DRY_RUN,
        )
        response_body = _outreach_envelope(
            status="error",
            summary=summary,
            errors=validation_errors,
        )
        return jsonify(response_body), 400

    n_records = len(parsed["records"])
    summary = _outreach_summary(
        n_records=n_records,
        n_valid=0,
        n_invalid=n_records,
        n_selected=0,
        threshold=parsed["threshold"],
        max_emails=parsed["max_emails"],
        dry_run=parsed["dry_run"],
    )
    if validation_errors:
        response_body = _outreach_envelope(
            status="error",
            summary=summary,
            errors=validation_errors,
        )
        status_code = 413 if any("MAX_BATCH_SIZE" in error.get("message", "") for error in validation_errors) else 400
        return jsonify(response_body), status_code

    if not artifacts_ready():
        response_body = _outreach_envelope(
            status="error",
            summary=summary,
            errors=[
                {
                    "stage": "predict",
                    "message": "Model artifacts are not ready yet. Please wait for training to finish.",
                }
            ],
        )
        return jsonify(response_body), 503

    errors = []
    try:
        batch_response = predict_batch_records(parsed["records"], {"mode": "partial"})
    except ValueError as exc:
        response_body = _outreach_envelope(
            status="error",
            summary=summary,
            errors=[{"stage": "predict", "message": str(exc)}],
        )
        return jsonify(response_body), 400
    except Exception as exc:
        response_body = _outreach_envelope(
            status="error",
            summary=summary,
            errors=[{"stage": "predict", "message": f"Internal server error: {str(exc)}"}],
        )
        return jsonify(response_body), 500

    results = batch_response.get("results")
    if not isinstance(results, list):
        response_body = _outreach_envelope(
            status="error",
            summary=summary,
            errors=[
                {
                    "stage": "predict",
                    "message": "Batch response is missing a valid 'results' list",
                }
            ],
        )
        return jsonify(response_body), 500

    batch_summary = batch_response.get("summary") if isinstance(batch_response.get("summary"), dict) else {}
    n_valid = int(batch_summary.get("valid_records", len(results)))
    n_invalid = int(batch_summary.get("invalid_records", max(0, n_records - n_valid)))

    raw_batch_errors = batch_response.get("errors")
    if isinstance(raw_batch_errors, list):
        for item in raw_batch_errors:
            errors.append({"stage": "batch_validation", "detail": item})
    elif raw_batch_errors is not None:
        errors.append({"stage": "batch_validation", "detail": raw_batch_errors})

    selected, target_errors = _select_outreach_recipients(
        batch_results=results,
        records=parsed["records"],
        threshold=parsed["threshold"],
        max_emails=parsed["max_emails"],
    )
    errors.extend(target_errors)

    drafted_selected = []
    writer = _writer_for_tone(parsed["tone"])
    writer_context = {
        "company_name": parsed["company_name"],
        "from_name": parsed["from_name"],
        "from_email": parsed["from_email"],
    }
    for recipient in selected:
        prompt = _build_outreach_prompt(
            tone=parsed["tone"],
            company_name=parsed["company_name"],
            from_name=parsed["from_name"],
            recipient_id=str(recipient["id"]),
            recipient_email=str(recipient["email"]),
            p_churn=float(recipient["p_churn"]),
        )
        try:
            body_text = str(writer(prompt=prompt, context=writer_context)).strip()
            if not body_text:
                raise ValueError("writer returned an empty draft")
            subject = str(outreach_subject_tool(body_text)).strip()
            if not subject:
                subject = f"{parsed['company_name']} check-in"
        except Exception as exc:
            errors.append(
                {
                    "stage": "draft",
                    "id": recipient["id"],
                    "email": recipient["email"],
                    "message": str(exc),
                }
            )
            continue

        drafted_selected.append(
            {
                "id": recipient["id"],
                "index": recipient["index"],
                "email": recipient["email"],
                "p_churn": recipient["p_churn"],
                "draft": {
                    "subject": subject,
                    "body_text": body_text,
                },
            }
        )

    send_report = _outreach_send_block()
    if not parsed["dry_run"] and drafted_selected:
        if not _sendgrid_ready():
            errors.append(
                {
                    "stage": "send",
                    "message": "SENDGRID_API_KEY is required when dry_run is false",
                }
            )
        else:
            send_report["attempted"] = True
            sent_count = 0
            send_results = []
            for recipient in drafted_selected:
                try:
                    send_result = send_email_text(
                        subject=recipient["draft"]["subject"],
                        body_text=recipient["draft"]["body_text"],
                        to_emails=[recipient["email"]],
                        from_email=parsed["from_email"],
                    )
                    sent_ok = _send_result_ok(send_result)
                    if sent_ok:
                        sent_count += 1
                    else:
                        errors.append(
                            {
                                "stage": "send",
                                "id": recipient["id"],
                                "email": recipient["email"],
                                "message": "Email provider reported a failed send attempt",
                            }
                        )
                    send_results.append(
                        {
                            "id": recipient["id"],
                            "email": recipient["email"],
                            "ok": sent_ok,
                            "result": send_result,
                        }
                    )
                except Exception as exc:
                    errors.append(
                        {
                            "stage": "send",
                            "id": recipient["id"],
                            "email": recipient["email"],
                            "message": str(exc),
                        }
                    )
                    send_results.append(
                        {
                            "id": recipient["id"],
                            "email": recipient["email"],
                            "ok": False,
                            "error": str(exc),
                        }
                    )

            send_report["sent"] = sent_count
            send_report["results"] = send_results

    summary = _outreach_summary(
        n_records=n_records,
        n_valid=n_valid,
        n_invalid=n_invalid,
        n_selected=len(drafted_selected),
        threshold=parsed["threshold"],
        max_emails=parsed["max_emails"],
        dry_run=parsed["dry_run"],
    )
    status = "ok"
    if errors:
        status = "partial" if n_valid > 0 else "error"

    response_body = _outreach_envelope(
        status=status,
        summary=summary,
        selected=drafted_selected,
        send=send_report,
        errors=errors,
    )
    status_code = 200 if status != "error" else 400
    return jsonify(response_body), status_code


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
