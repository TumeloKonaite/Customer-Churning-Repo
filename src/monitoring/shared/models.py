"""Shared serialization and timestamp helpers for Arize data feeds."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import json
import math
from collections.abc import Mapping
from typing import Any


def require_utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def timestamp(value: datetime) -> str:
    return require_utc(value, "timestamp").isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    value = json_safe(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def json_safe(value: Any) -> Any:
    """Convert timestamps, enums, and numpy scalars to strict JSON values."""
    if isinstance(value, datetime):
        return timestamp(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (set, frozenset)):
        return [json_safe(child) for child in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
