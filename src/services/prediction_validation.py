"""Shared validation and coercion for prediction records."""

from typing import Any

from src.schemas.prediction import REQUIRED_FIELDS


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


def validate_payload(data: Any) -> tuple[bool, list[str] | None]:
    ok, errors, _ = validate_record(data)
    return ok, (errors or None)
