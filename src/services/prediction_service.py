from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import re
from typing import Any

import pandas as pd

from src.pipeline.prediction_pipeline import PredictPipeline

REQUIRED_FIELDS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
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
VALID_BATCH_MODES = {"fail_fast", "partial"}
MAX_BATCH_SIZE = 100

_MISSING_FIELD_RE = re.compile(r"^Missing required field: (?P<field>.+)$")
_NUMERIC_FIELD_RE = re.compile(r"^Field '(?P<field>[^']+)' must be a number")


def _timestamp_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_model_metadata() -> dict[str, Any]:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    metadata_path = os.path.join(project_root, "artifacts", "metadata.json")
    defaults = {"model_name": "churn_predictor", "model_version": "1.0.0"}
    try:
        with open(metadata_path, "r", encoding="utf-8") as file:
            raw = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return defaults
    return {
        "model_name": raw.get("model_name", defaults["model_name"]),
        "model_version": raw.get("version", defaults["model_version"]),
    }


def _extract_record_id(record: Any) -> Any | None:
    if not isinstance(record, dict):
        return None
    for key in ("customer_id", "row_id", "id"):
        if key in record and record[key] is not None:
            return record[key]
    return None


def _build_batch_envelope(*, status, results, errors, summary):
    return {
        "status": status,
        "results": results,
        "errors": errors if errors else None,
        "summary": summary,
        "metadata": _load_model_metadata(),
        "timestamp": _timestamp_now(),
    }


def validate_record(record: Any) -> tuple[bool, list[str], dict | None]:
    """Validate and coerce one prediction record."""
    if not isinstance(record, dict):
        return False, ["Record must be a JSON object"], None

    missing = [key for key in REQUIRED_FIELDS if record.get(key) in (None, "")]
    if missing:
        return False, [f"Missing required field: {key}" for key in missing], None

    coerced = {field: record.get(field) for field in REQUIRED_FIELDS}
    errors = []
    for key, caster in NUMERIC_FIELDS.items():
        try:
            coerced[key] = caster(coerced[key])
        except (TypeError, ValueError):
            errors.append(f"Field '{key}' must be a number (got {record.get(key)!r})")
    if errors:
        return False, errors, None
    return True, [], coerced


def validate_batch(records: list[dict], mode: str) -> dict:
    if mode not in VALID_BATCH_MODES:
        raise ValueError(f"Unsupported batch mode: {mode}")

    result = {"valid_rows": [], "errors": [], "row_map": {}, "row_ids": {}}
    for row_index, record in enumerate(records):
        record_id = _extract_record_id(record)
        result["row_ids"][row_index] = record_id
        ok, errors, coerced = validate_record(record)
        if ok:
            valid_index = len(result["valid_rows"])
            result["valid_rows"].append(coerced)
            result["row_map"][valid_index] = row_index
            continue

        for message in errors:
            error = {"row_index": row_index, "id": record_id, "message": message}
            missing_match = _MISSING_FIELD_RE.match(message)
            numeric_match = _NUMERIC_FIELD_RE.match(message)
            if missing_match:
                error["field"] = missing_match.group("field")
            elif numeric_match:
                error["field"] = numeric_match.group("field")
            result["errors"].append(error)
        if mode == "fail_fast":
            break
    return result


def predict_batch_records(records: Any, options: Any | None = None) -> dict[str, Any]:
    """Validate and score a batch with one model invocation."""
    if not isinstance(records, list):
        raise ValueError("Field 'records' must be a list")
    if len(records) > MAX_BATCH_SIZE:
        raise ValueError(f"Batch size exceeds MAX_BATCH_SIZE ({MAX_BATCH_SIZE})")
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise ValueError("Field 'options' must be an object")

    mode = options.get("mode", "fail_fast")
    if mode not in VALID_BATCH_MODES:
        raise ValueError("options.mode must be one of: fail_fast, partial")

    validation = validate_batch(records, mode)
    errors = validation["errors"]
    valid_rows = validation["valid_rows"]
    invalid_count = len({error["row_index"] for error in errors})
    summary = {
        "total_records": len(records),
        "valid_records": len(valid_rows),
        "invalid_records": invalid_count,
        "error_count": len(errors),
        "mode": mode,
    }

    if mode == "fail_fast" and errors:
        return _build_batch_envelope(status="error", results=[], errors=errors, summary=summary)
    if not valid_rows:
        status = "failed" if errors else "success"
        return _build_batch_envelope(status=status, results=[], errors=errors, summary=summary)

    labels, probabilities = PredictPipeline().predict(pd.DataFrame(valid_rows, columns=REQUIRED_FIELDS))
    labels = list(labels)
    probabilities = list(probabilities) if probabilities is not None else [None] * len(labels)
    if len(labels) != len(valid_rows):
        raise RuntimeError("PredictPipeline.predict returned unexpected number of labels")
    if len(probabilities) != len(valid_rows):
        raise RuntimeError("PredictPipeline.predict returned unexpected number of probabilities")

    results = []
    for valid_index, (label, probability) in enumerate(zip(labels, probabilities)):
        source_index = validation["row_map"][valid_index]
        results.append(
            {
                "index": int(source_index),
                "id": validation["row_ids"].get(source_index),
                "predicted_label": int(label),
                "p_churn": None if probability is None else float(probability),
            }
        )

    return _build_batch_envelope(
        status="partial" if errors else "success",
        results=results,
        errors=errors,
        summary=summary,
    )
