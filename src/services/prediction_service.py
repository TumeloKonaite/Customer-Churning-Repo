from __future__ import annotations

from typing import Any
import re

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


def validate_record(record: Any) -> tuple[bool, list[str], dict | None]:
    """Validate one prediction record using single-record API semantics."""
    if not isinstance(record, dict):
        return False, ["Record must be a JSON object"], None

    missing = [k for k in REQUIRED_FIELDS if k not in record or record.get(k) in (None, "")]
    if missing:
        return False, [f"Missing required field: {k}" for k in missing], None

    coerced_record = dict(record)
    cast_errors = []
    for key, caster in NUMERIC_FIELDS.items():
        try:
            coerced_record[key] = caster(record.get(key))
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
    }

    for row_index, record in enumerate(records):
        ok, errors, coerced_record = validate_record(record)
        if ok:
            valid_index = len(result["valid_rows"])
            result["valid_rows"].append(coerced_record)
            result["row_map"][valid_index] = row_index
            continue

        for error_message in errors:
            error_item = {
                "row_index": row_index,
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
