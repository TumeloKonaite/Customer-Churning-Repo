"""Versioned monitoring configuration and reproducible run metadata."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum, StrEnum
import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


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
    """Convert Evidently/numpy values to portable, strict JSON values."""
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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    INSUFFICIENT_DATA = "insufficient_data"
    FAILED = "failed"


class ResultStatus(StrEnum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    NOT_EVALUATED = "not_evaluated"


class FeatureRule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    data_type: str
    nullable: bool = False
    minimum: float | None = None
    maximum: float | None = None
    allowed_values: tuple[str | int | float | bool, ...] | None = None
    drift_method: str | None = None
    drift_threshold: float | None = Field(default=None, gt=0)


class MonitoringPolicy(BaseModel):
    """Immutable, result-affecting policy. A changed value requires a new version."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: str = Field(min_length=1)
    enabled: bool = True
    minimum_current_rows: int = Field(ge=1)
    maximum_current_rows: int = Field(ge=1)
    initial_lookback_hours: int = Field(ge=1)
    maximum_lookback_hours: int = Field(ge=1)
    fixed_historical_boundary: datetime | None = None
    minimum_reference_rows: int = Field(ge=1)
    schedule_cron: str = Field(min_length=1)
    cadence_minutes: int = Field(ge=1)
    data_latency_allowance_minutes: int = Field(ge=0)
    included_environments: tuple[str, ...] = Field(min_length=1)
    included_model_versions: tuple[str, ...] = Field(min_length=1)
    drift_dataset_share_threshold: float = Field(gt=0, le=1)
    numeric_drift_method: str = Field(min_length=1)
    numeric_drift_threshold: float = Field(gt=0)
    categorical_drift_method: str = Field(min_length=1)
    categorical_drift_threshold: float = Field(gt=0)
    prediction_probability_drift_method: str = Field(min_length=1)
    prediction_probability_drift_threshold: float = Field(gt=0)
    predicted_class_drift_method: str = Field(min_length=1)
    predicted_class_drift_threshold: float = Field(gt=0)
    row_count_change_warning_ratio: float = Field(ge=0)
    excluded_features: tuple[str, ...] = ()
    suppressed_features: tuple[str, ...] = ()
    exclusion_rules: tuple[dict[str, Any], ...] = ()
    suppression_rules: tuple[dict[str, Any], ...] = ()
    feature_rules: dict[str, FeatureRule]
    calibration_note: str = Field(min_length=1)

    @field_validator("fixed_historical_boundary")
    @classmethod
    def boundary_is_utc(cls, value: datetime | None) -> datetime | None:
        return None if value is None else require_utc(value, "fixed_historical_boundary")

    @model_validator(mode="after")
    def coherent_limits(self) -> "MonitoringPolicy":
        if self.maximum_current_rows < self.minimum_current_rows:
            raise ValueError("maximum_current_rows must be >= minimum_current_rows")
        if self.maximum_lookback_hours < self.initial_lookback_hours:
            raise ValueError("maximum_lookback_hours must be >= initial_lookback_hours")
        if len(set(self.included_environments)) != len(self.included_environments):
            raise ValueError("included_environments must not contain duplicates")
        if len(set(self.included_model_versions)) != len(self.included_model_versions):
            raise ValueError("included_model_versions must not contain duplicates")
        overlap = set(self.excluded_features) & set(self.suppressed_features)
        if overlap:
            raise ValueError(f"features cannot be both excluded and suppressed: {sorted(overlap)}")
        return self

    @property
    def initial_lookback(self) -> timedelta:
        return timedelta(hours=self.initial_lookback_hours)

    @property
    def maximum_lookback(self) -> timedelta:
        return timedelta(hours=self.maximum_lookback_hours)

    @property
    def data_latency_allowance(self) -> timedelta:
        return timedelta(minutes=self.data_latency_allowance_minutes)

    @property
    def config_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.model_dump(mode="json")))

    def permits(self, environment: str, model_version_id: str) -> bool:
        versions = self.included_model_versions
        return (
            self.enabled
            and environment in self.included_environments
            and ("*" in versions or model_version_id in versions)
        )


class BaselineVersion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline_version_id: str = Field(min_length=1)
    model_version_id: str = Field(min_length=1)
    reference_dataset_uri: str = Field(min_length=1)
    reference_sha256: str
    feature_schema_version: str = Field(min_length=1)
    created_at: datetime
    active_from: datetime
    retired_at: datetime | None = None
    purpose: str = Field(min_length=1)
    approval_metadata: dict[str, Any] | None = None

    @field_validator("reference_sha256")
    @classmethod
    def checksum_is_sha256(cls, value: str) -> str:
        value = value.lower()
        if not SHA256_RE.fullmatch(value):
            raise ValueError("reference_sha256 must be a lowercase SHA-256 digest")
        return value

    @field_validator("created_at", "active_from", "retired_at")
    @classmethod
    def dates_are_utc(cls, value: datetime | None, info) -> datetime | None:
        return None if value is None else require_utc(value, info.field_name)

    @model_validator(mode="after")
    def interval_is_valid(self) -> "BaselineVersion":
        if self.active_from < self.created_at:
            raise ValueError("active_from must not predate created_at")
        if self.retired_at is not None and self.retired_at <= self.active_from:
            raise ValueError("retired_at must be after active_from")
        return self


class ExtractionWatermark(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    extraction_cutoff: datetime
    maximum_persisted_event_id: int = Field(ge=0)
    maximum_eligible_prediction_timestamp: datetime | None = None

    @field_validator("extraction_cutoff", "maximum_eligible_prediction_timestamp")
    @classmethod
    def values_are_utc(cls, value: datetime | None, info) -> datetime | None:
        return None if value is None else require_utc(value, info.field_name)


class SelectedWindow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start: datetime
    end: datetime
    observed_rows: int = Field(ge=0)
    selected_rows: int = Field(ge=0)
    lookback_hours: int = Field(ge=0)
    reached_boundary: bool
    deterministic_limit_applied: bool
    selection_order: str = "prediction_timestamp DESC, event_id DESC"

    @field_validator("start", "end")
    @classmethod
    def values_are_utc(cls, value: datetime, info) -> datetime:
        return require_utc(value, info.field_name)


class PredictionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: int = Field(ge=1)
    prediction_id: str = Field(min_length=1)
    environment: str = Field(min_length=1)
    model_version_id: str = Field(min_length=1)
    prediction_timestamp: datetime
    persisted_at: datetime
    feature_schema_version: str = Field(min_length=1)
    features: dict[str, Any]
    prediction_probability: float
    predicted_class: str | int | bool

    @field_validator("prediction_timestamp", "persisted_at")
    @classmethod
    def values_are_utc(cls, value: datetime, info) -> datetime:
        return require_utc(value, info.field_name)


def monitoring_run_id(
    *,
    job_type: str,
    environment: str,
    model_version_id: str,
    baseline_version_id: str,
    policy_version: str,
    window: SelectedWindow,
    watermark: ExtractionWatermark,
) -> str:
    identity = {
        "job_type": job_type,
        "environment": environment,
        "model_version_id": model_version_id,
        "baseline_version_id": baseline_version_id,
        "policy_version": policy_version,
        "window_start": timestamp(window.start),
        "window_end": timestamp(window.end),
        "extraction_cutoff": timestamp(watermark.extraction_cutoff),
        "maximum_persisted_event_id": watermark.maximum_persisted_event_id,
        "maximum_eligible_prediction_timestamp": (
            timestamp(watermark.maximum_eligible_prediction_timestamp)
            if watermark.maximum_eligible_prediction_timestamp
            else None
        ),
    }
    return f"dq-drift-{sha256_bytes(canonical_json_bytes(identity))}"
