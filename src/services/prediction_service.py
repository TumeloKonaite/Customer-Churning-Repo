from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import re
from typing import Any

import pandas as pd

from src.decisioning import (
    ACTION_COSTS,
    ACTION_NONE,
    estimate_clv,
    expected_net_gain,
    recommended_action,
)
from src.pipeline.prediction_pipeline import PredictPipeline

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

VALID_BATCH_MODES = {"fail_fast", "partial"}
MAX_BATCH_SIZE = 100

_MISSING_FIELD_RE = re.compile(r"^Missing required field: (?P<field>.+)$")
_NUMERIC_FIELD_RE = re.compile(r"^Field '(?P<field>[^']+)' must be a number")


def _timestamp_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_model_metadata() -> dict[str, Any]:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    metadata_path = os.path.join(project_root, "artifacts", "metadata.json")
    default_metadata = {
        "model_name": "churn_predictor",
        "model_version": "1.0.0",
    }

    try:
        with open(metadata_path, "r", encoding="utf-8") as file:
            raw = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default_metadata

    return {
        "model_name": raw.get("model_name", default_metadata["model_name"]),
        "model_version": raw.get("version", default_metadata["model_version"]),
    }


def _extract_record_id(record: Any) -> Any | None:
    """Return passthrough identifier without affecting model inputs."""
    if not isinstance(record, dict):
        return None

    if "customer_id" in record and record["customer_id"] is not None:
        return record["customer_id"]
    if "row_id" in record and record["row_id"] is not None:
        return record["row_id"]
    return None


def _build_batch_envelope(
    *,
    status: str,
    results: list[dict[str, Any]],
    email_candidates: list[dict[str, Any]] | None = None,
    errors: list[dict[str, Any]] | None,
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": status,
        "results": results,
        "email_candidates": email_candidates or [],
        "errors": errors if errors else None,
        "summary": summary,
        "metadata": _load_model_metadata(),
        "timestamp": _timestamp_now(),
    }


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _email_candidate_rules(options: dict[str, Any]) -> dict[str, Any]:
    raw_rules = None
    for key in (
        "email_candidate_rules",
        "email_selection",
        "candidate_selection",
        "email_candidates",
    ):
        candidate = options.get(key)
        if isinstance(candidate, dict):
            raw_rules = candidate
            break

    raw_rules = raw_rules or {}

    allowed_actions = raw_rules.get("allowed_actions")
    if isinstance(allowed_actions, list):
        allowed_actions = {str(item) for item in allowed_actions}
    else:
        allowed_actions = None

    return {
        # Default: only outreach-worthy rows with positive expected value.
        "exclude_no_action": bool(raw_rules.get("exclude_no_action", True)),
        "min_p_churn": _coerce_optional_float(raw_rules.get("min_p_churn")),
        "min_net_gain": _coerce_optional_float(raw_rules.get("min_net_gain")),
        "max_candidates": _coerce_optional_int(raw_rules.get("max_candidates")),
        "allowed_actions": allowed_actions,
    }


def _estimate_record_clv(record_features: dict[str, Any]) -> float | None:
    try:
        return float(estimate_clv(record_features))
    except Exception:
        return None


def _post_process_result(
    result: dict[str, Any],
    *,
    record_features: dict[str, Any],
) -> dict[str, Any]:
    processed = dict(result)
    churn_probability = processed.get("p_churn")
    clv = _estimate_record_clv(record_features)

    action = None
    net_gain = None
    if churn_probability is not None:
        action = recommended_action(float(churn_probability))
        if clv is not None:
            action_cost = float(ACTION_COSTS.get(action, 0.0))
            net_gain = expected_net_gain(float(churn_probability), clv, action_cost)

    processed["clv"] = clv
    processed["recommended_action"] = action
    processed["net_gain"] = net_gain
    return processed


def _select_email_candidates(
    results: list[dict[str, Any]],
    *,
    options: dict[str, Any],
) -> list[dict[str, Any]]:
    rules = _email_candidate_rules(options)
    selected: list[dict[str, Any]] = []

    for item in results:
        p_churn = item.get("p_churn")
        action = item.get("recommended_action")
        net_gain = item.get("net_gain")

        if p_churn is None or action is None or net_gain is None:
            continue
        if rules["exclude_no_action"] and action == ACTION_NONE:
            continue

        min_p_churn = rules["min_p_churn"]
        if min_p_churn is not None and float(p_churn) < min_p_churn:
            continue

        min_net_gain = rules["min_net_gain"]
        if min_net_gain is None:
            min_net_gain = 0.0
        if float(net_gain) < min_net_gain:
            continue

        allowed_actions = rules["allowed_actions"]
        if allowed_actions is not None and action not in allowed_actions:
            continue

        selected.append(
            {
                "index": item.get("index"),
                "id": item.get("id"),
                "p_churn": float(p_churn),
                "recommended_action": action,
                "net_gain": float(net_gain),
            }
        )

    max_candidates = rules["max_candidates"]
    if max_candidates is not None and max_candidates >= 0:
        return selected[:max_candidates]

    return selected


def validate_record(record: Any) -> tuple[bool, list[str], dict | None]:
    """Validate one prediction record using single-record API semantics."""
    if not isinstance(record, dict):
        return False, ["Record must be a JSON object"], None

    missing = [k for k in REQUIRED_FIELDS if k not in record or record.get(k) in (None, "")]
    if missing:
        return False, [f"Missing required field: {k}" for k in missing], None

    coerced_record = {field: record.get(field) for field in REQUIRED_FIELDS}
    cast_errors = []
    for key, caster in NUMERIC_FIELDS.items():
        try:
            coerced_record[key] = caster(coerced_record[key])
        except Exception:
            cast_errors.append(f"Field '{key}' must be a number (got {record.get(key)!r})")

    if cast_errors:
        return False, cast_errors, None

    return True, [], coerced_record


def validate_batch(records: list[dict], mode: str) -> dict:
    if mode not in VALID_BATCH_MODES:
        raise ValueError(f"Unsupported batch mode: {mode}")

    result = {
        "valid_rows": [],
        "errors": [],
        "row_map": {},
        "row_ids": {},
    }

    for row_index, record in enumerate(records):
        record_id = _extract_record_id(record)
        result["row_ids"][row_index] = record_id
        ok, errors, coerced_record = validate_record(record)
        if ok:
            valid_index = len(result["valid_rows"])
            result["valid_rows"].append(coerced_record)
            result["row_map"][valid_index] = row_index
            continue

        for error_message in errors:
            error_item = {
                "row_index": row_index,
                "id": record_id,
                "message": error_message,
            }

            missing_match = _MISSING_FIELD_RE.match(error_message)
            numeric_match = _NUMERIC_FIELD_RE.match(error_message)
            if missing_match:
                error_item["field"] = missing_match.group("field")
            elif numeric_match:
                error_item["field"] = numeric_match.group("field")

            result["errors"].append(error_item)

        if mode == "fail_fast":
            break

    return result


def predict_batch_records(records: Any, options: Any | None = None) -> dict[str, Any]:
    """Validate and score a batch using one DataFrame build and one model call."""
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

    validation_result = validate_batch(records, mode)
    errors = validation_result["errors"]
    valid_rows = validation_result["valid_rows"]
    row_map = validation_result["row_map"]
    row_ids = validation_result["row_ids"]
    invalid_row_count = len({error["row_index"] for error in errors})

    summary = {
        "total_records": len(records),
        "valid_records": len(valid_rows),
        "invalid_records": invalid_row_count,
        "error_count": len(errors),
        "mode": mode,
    }

    if mode == "fail_fast" and errors:
        return _build_batch_envelope(
            status="error",
            results=[],
            email_candidates=[],
            errors=errors,
            summary=summary,
        )

    if not valid_rows:
        status = "failed" if errors else "success"
        return _build_batch_envelope(
            status=status,
            results=[],
            email_candidates=[],
            errors=errors,
            summary=summary,
        )

    batch_df = pd.DataFrame(valid_rows, columns=REQUIRED_FIELDS)
    pipeline = PredictPipeline()
    predicted_labels, probabilities = pipeline.predict(batch_df)

    predicted_labels_list = list(predicted_labels)
    probabilities_list = list(probabilities) if probabilities is not None else [None] * len(predicted_labels_list)

    if len(predicted_labels_list) != len(valid_rows):
        raise RuntimeError("PredictPipeline.predict returned unexpected number of labels")
    if len(probabilities_list) != len(valid_rows):
        raise RuntimeError("PredictPipeline.predict returned unexpected number of probabilities")

    results = []
    for valid_index, (label, p_churn) in enumerate(zip(predicted_labels_list, probabilities_list)):
        raw_result = {
            "index": int(row_map[valid_index]),
            "id": row_ids.get(row_map[valid_index]),
            "predicted_label": int(label),
            "p_churn": None if p_churn is None else float(p_churn),
        }
        results.append(
            _post_process_result(
                raw_result,
                record_features=valid_rows[valid_index],
            )
        )

    email_candidates = _select_email_candidates(results, options=options)
    status = "partial" if errors else "success"
    return _build_batch_envelope(
        status=status,
        results=results,
        email_candidates=email_candidates,
        errors=errors,
        summary=summary,
    )
