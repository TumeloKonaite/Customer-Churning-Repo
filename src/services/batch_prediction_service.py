"""JSON and CSV batch-prediction orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from typing import Any, BinaryIO

import pandas as pd
from pydantic import ValidationError

from src.pipeline.prediction_pipeline import PredictPipeline
from src.schemas.batch_prediction import (
    MAX_BATCH_SIZE,
    VALID_BATCH_MODES,
    BatchOptions,
    BatchPredictionRecord,
)
from src.schemas.prediction import REQUIRED_FIELDS
from src.services import model_service
from src.services.exceptions import BatchContractViolation, PredictionExecutionError
from src.services.prediction_validation import validate_record, validation_error_details


logger = logging.getLogger(__name__)
IDENTIFIER_FIELDS = {"customer_id", "row_id", "id"}


def _timestamp_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_model_metadata() -> dict[str, str]:
    """Compatibility seam retained for focused batch-service tests."""
    return model_service.prediction_metadata()


def _extract_record_id(record: Any) -> Any | None:
    if isinstance(record, BatchPredictionRecord):
        record = record.model_dump()
    if not isinstance(record, dict):
        return None
    for key in ("customer_id", "row_id", "id"):
        if key in record and record[key] is not None and not pd.isna(record[key]):
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


def validate_batch(records: list[Any], mode: str) -> dict:
    if mode not in VALID_BATCH_MODES:
        raise ValueError(f"Unsupported batch mode: {mode}")

    result = {"valid_rows": [], "errors": [], "row_map": {}, "row_ids": {}}
    for row_index, record in enumerate(records):
        record_id = _extract_record_id(record)
        result["row_ids"][row_index] = record_id
        ok, errors, coerced = validate_record(record, allow_identifiers=True)
        if ok:
            valid_index = len(result["valid_rows"])
            result["valid_rows"].append(coerced)
            result["row_map"][valid_index] = row_index
            continue

        details = validation_error_details(record, allow_identifiers=True)
        for error_index, message in enumerate(errors):
            error = {"row_index": row_index, "id": record_id, "message": message}
            if error_index < len(details) and details[error_index][1] is not None:
                error["field"] = details[error_index][1]
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
    if isinstance(options, BatchOptions):
        options = options.model_dump()
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

    labels, probabilities = PredictPipeline().predict(
        pd.DataFrame(valid_rows, columns=REQUIRED_FIELDS)
    )
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


def predict_batch(records: list, options: dict) -> dict[str, Any]:
    """Check readiness and execute a validated JSON or CSV batch."""
    mode = options.get("mode", "fail_fast")
    if mode not in VALID_BATCH_MODES:
        raise BatchContractViolation("options.mode must be one of: fail_fast, partial")
    model_service.ensure_artifacts_ready()
    try:
        result = predict_batch_records(records, options)
        operational = model_service.operational_metadata()
        logger.info(
            "batch_prediction_completed deployment_id=%s model_version=%s "
            "mlflow_run_id=%s model_version_id=%s pipeline_sha256=%s "
            "artifact_manifest_sha256=%s integrity_status=%s",
            operational.get("deployment_id"),
            operational.get("model_version"),
            operational.get("mlflow_run_id"),
            operational.get("model_version_id"),
            operational.get("pipeline_sha256"),
            operational.get("artifact_manifest_sha256"),
            operational.get("integrity_status"),
        )
        return result
    except ValueError as exc:
        raise BatchContractViolation(str(exc)) from exc
    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise PredictionExecutionError(f"Internal server error: {exc}") from exc


def parse_batch_options_json(options_raw: str | None) -> dict:
    if options_raw is None or not str(options_raw).strip():
        return {}
    try:
        options = json.loads(options_raw)
    except json.JSONDecodeError as exc:
        raise BatchContractViolation(f"Invalid options JSON: {exc.msg}") from exc
    if not isinstance(options, dict):
        raise BatchContractViolation("Field 'options' must be an object")
    try:
        return BatchOptions.model_validate(options).model_dump()
    except ValidationError as exc:
        raise BatchContractViolation(f"Invalid batch options: {exc.errors()[0]['msg']}") from exc


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
    unexpected_columns = [
        column
        for column in frame.columns
        if column not in REQUIRED_FIELDS and column not in IDENTIFIER_FIELDS
    ]
    if unexpected_columns:
        raise BatchContractViolation(
            f"CSV contains unexpected columns: {', '.join(unexpected_columns)}"
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
