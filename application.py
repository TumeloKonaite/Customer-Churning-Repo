from flask import Flask, request, render_template, jsonify
from datetime import datetime
import json
import os

from src.decisioning import (
    ACTION_COSTS,
    estimate_clv,
    expected_net_gain,
    recommended_action,
)
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# -----------------------------
# Helpers
# -----------------------------

REQUIRED_FIELDS = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

NUMERIC_FIELDS = {
    "CreditScore": float,
    "Age": float,
    "Tenure": float,
    "Balance": float,
    "NumOfProducts": float,
    "HasCrCard": float,
    "IsActiveMember": float,
    "EstimatedSalary": float,
}

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


def validate_payload(data: dict):
    """Validate required fields and numeric coercion with useful error messages."""
    missing = [k for k in REQUIRED_FIELDS if k not in data or data.get(k) in (None, "")]
    if missing:
        return False, [f"Missing required field: {k}" for k in missing]

    # Validate numeric fields can be cast
    cast_errors = []
    for k, caster in NUMERIC_FIELDS.items():
        try:
            caster(data.get(k))
        except Exception:
            cast_errors.append(f"Field '{k}' must be a number (got {data.get(k)!r})")

    if cast_errors:
        return False, cast_errors

    return True, None


def artifacts_ready() -> bool:
    return all(os.path.exists(path) for path in REQUIRED_ARTIFACTS)


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


if __name__ == "__main__":
    # Binding to 0.0.0.0 is required for container access
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    port = int(os.getenv("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=debug)
