"""JSON and CSV batch-prediction orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import re
from typing import Any, BinaryIO

import pandas as pd

from src.pipeline.prediction_pipeline import PredictPipeline
from src.schemas.batch_prediction import MAX_BATCH_SIZE, VALID_BATCH_MODES
from src.schemas.prediction import REQUIRED_FIELDS
from src.services import model_service
from src.services.exceptions import (
    APIServiceError,
    BatchContractViolation,
    PredictionExecutionError,
)
from src.services.prediction_validation import validate_record


logger = logging.getLogger(__name__)
_MISSING_FIELD_RE = re.compile(r"^Missing required field: (?P<field>.+)$")
_NUMERIC_FIELD_RE = re.compile(r"^Field '(?P<field>[^']+)' must be a number")


def _timestamp_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_model_metadata() -> dict[str, str]:
    """Compatibility seam retained for focused batch-service tests."""
    return model_service.prediction_metadata()


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


def validate_batch_payload(payload: Any) -> tuple[list, dict]:
    if payload is None:
        raise APIServiceError("Invalid JSON body")
    if not isinstance(payload, dict):
        raise APIServiceError("JSON body must be an object")
    if "records" not in payload:
        raise BatchContractViolation("Field 'records' is required and must be a list")

    records = payload.get("records")
    if not isinstance(records, list):
        raise BatchContractViolation("Field 'records' must be a list")
    if len(records) > MAX_BATCH_SIZE:
        raise BatchContractViolation(
            f"Batch size exceeds MAX_BATCH_SIZE ({MAX_BATCH_SIZE})", status_code=413
        )
    options = payload.get("options", {})
    if not isinstance(options, dict):
        raise BatchContractViolation("Field 'options' must be an object")
    return records, options


def predict_batch(records: list, options: dict) -> dict[str, Any]:
    """Check readiness and execute a validated JSON or CSV batch."""
    mode = options.get("mode", "fail_fast")
    if mode not in VALID_BATCH_MODES:
        raise BatchContractViolation("options.mode must be one of: fail_fast, partial")
    model_service.ensure_artifacts_ready()
    try:
        return predict_batch_records(records, options)
    except ValueError as exc:
        raise BatchContractViolation(str(exc)) from exc
    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise PredictionExecutionError(f"Internal server error: {exc}") from exc


def predict_batch_payload(payload: Any) -> dict[str, Any]:
    records, options = validate_batch_payload(payload)
    return predict_batch(records, options)


def parse_batch_options_json(options_raw: str | None) -> dict:
    if options_raw is None or not str(options_raw).strip():
        return {}
    try:
        options = json.loads(options_raw)
    except json.JSONDecodeError as exc:
        raise BatchContractViolation(f"Invalid options JSON: {exc.msg}") from exc
    if not isinstance(options, dict):
        raise BatchContractViolation("Field 'options' must be an object")
    return options


def parse_csv_upload_records(filename: str | None, file: BinaryIO | None) -> list[dict]:
    if file is None:
        raise BatchContractViolation("Field 'file' is required")

    normalized_filename = (filename or "").strip()
    if not normalized_filename:
        raise BatchContractViolation("Uploaded filename must not be empty")
    if not normalized_filename.lower().endswith(".csv"):
        raise BatchContractViolation("Uploaded file must be a .csv")

    try:
        frame = pd.read_csv(file)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as exc:
        raise BatchContractViolation(f"CSV could not be parsed: {exc}") from exc

    frame = frame.dropna(how="all")
    if frame.empty:
        raise BatchContractViolation("CSV must contain at least one data row")
    missing_columns = [field for field in REQUIRED_FIELDS if field not in frame.columns]
    if missing_columns:
        raise BatchContractViolation(
            f"CSV is missing required columns: {', '.join(missing_columns)}"
        )

    records = frame.to_dict(orient="records")
    if len(records) > MAX_BATCH_SIZE:
        raise BatchContractViolation(
            f"Batch size exceeds MAX_BATCH_SIZE ({MAX_BATCH_SIZE})", status_code=413
        )
    return records


def predict_csv_batch(filename: str | None, file: BinaryIO | None, options_raw: str | None):
    options = parse_batch_options_json(options_raw)
    records = parse_csv_upload_records(filename, file)
    return predict_batch(records, options)
