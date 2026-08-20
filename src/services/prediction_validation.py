"""Pydantic-backed validation shared by non-HTTP and CSV prediction paths."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from src.schemas.batch_prediction import BatchPredictionRecord
from src.schemas.prediction import REQUIRED_FIELDS, SinglePredictionRequest


def _format_validation_error(error: dict) -> tuple[str, str | None]:
    location = error.get("loc", ())
    field = str(location[-1]) if location else None
    error_type = error.get("type", "")
    value = error.get("input")
    if error_type == "missing":
        return f"Missing required field: {field}", field
    if error_type == "extra_forbidden":
        return f"Unexpected field: {field}", field
    if value is None:
        return f"Field '{field}' must not be null", field
    return f"Incorrect data type or value for field '{field}'", field


def validate_record(
    record: Any, *, allow_identifiers: bool = True
) -> tuple[bool, list[str], dict | None]:
    """Validate one raw record and return canonical model features only."""
    if not isinstance(record, (dict, BatchPredictionRecord, SinglePredictionRequest)):
        return False, ["Record must be a JSON object"], None

    model_type = BatchPredictionRecord if allow_identifiers else SinglePredictionRequest
    try:
        validated = (
            record
            if isinstance(record, model_type)
            else model_type.model_validate(record)
        )
    except ValidationError as exc:
        messages = [_format_validation_error(error)[0] for error in exc.errors()]
        return False, messages, None

    dumped = validated.model_dump()
    return True, [], {field: dumped[field] for field in REQUIRED_FIELDS}


def validation_error_details(record: Any, *, allow_identifiers: bool = True):
    """Return messages paired with fields for batch error envelopes."""
    model_type = BatchPredictionRecord if allow_identifiers else SinglePredictionRequest
    try:
        model_type.model_validate(record)
    except ValidationError as exc:
        return [_format_validation_error(error) for error in exc.errors()]
    return []


def validate_payload(data: Any) -> tuple[bool, list[str] | None]:
    ok, errors, _ = validate_record(data, allow_identifiers=False)
    return ok, (errors or None)
