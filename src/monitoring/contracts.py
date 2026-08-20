"""Executable examples for production-monitoring contract version 1.0.0.

This module deliberately does not implement outcome ingestion or persistence.  It
provides the pure, deterministic rules that those components must obey.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
import hashlib
import hmac
import logging
from typing import Iterable, Mapping, TypeVar


CONTRACT_VERSION = "1.0.0"
DEFAULT_HORIZON = timedelta(days=90)
DEFAULT_GRACE_PERIOD = timedelta(days=7)
DEFAULT_MIN_SEGMENT_SIZE = 20
QUALIFYING_CHURN_EVENT_TYPES = frozenset({"CUSTOMER_RELATIONSHIP_TERMINATED"})
SIMULATION_ENVIRONMENTS = frozenset({"local", "test", "development", "staging"})


class ContractViolation(ValueError):
    """Raised when data cannot safely be processed under contract v1."""


class LabelState(StrEnum):
    CHURNED = "churned"
    NOT_CHURNED = "not_churned"
    PENDING = "pending"


def _require_aware_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ContractViolation(f"{field_name} must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ContractViolation(f"{field_name} must be normalized to UTC")


@dataclass(frozen=True, slots=True)
class Prediction:
    prediction_id: str
    customer_token: str
    prediction_timestamp: datetime
    horizon_end: datetime

    def __post_init__(self) -> None:
        if not self.prediction_id:
            raise ContractViolation("prediction_id is required")
        if not self.customer_token:
            raise ContractViolation("customer_token is required")
        _require_aware_utc(self.prediction_timestamp, "prediction_timestamp")
        _require_aware_utc(self.horizon_end, "horizon_end")
        if self.horizon_end <= self.prediction_timestamp:
            raise ContractViolation("horizon_end must be after prediction_timestamp")


@dataclass(frozen=True, slots=True)
class Outcome:
    outcome_id: str
    customer_token: str
    event_type: str
    outcome_timestamp: datetime
    outcome_source: str
    is_simulated: bool = False
    generator_version: str | None = None
    scenario_version: str | None = None
    revision: int = 1
    is_retracted: bool = False

    def __post_init__(self) -> None:
        if not self.outcome_id:
            raise ContractViolation("outcome_id is required")
        if not self.customer_token:
            raise ContractViolation("customer_token is required")
        if not self.outcome_source:
            raise ContractViolation("outcome_source is required")
        _require_aware_utc(self.outcome_timestamp, "outcome_timestamp")
        if self.revision < 1:
            raise ContractViolation("revision must be at least 1")
        if self.is_simulated and (
            not self.generator_version or not self.scenario_version
        ):
            raise ContractViolation(
                "simulated outcomes require generator_version and scenario_version"
            )
        if not self.is_simulated and (
            self.generator_version is not None or self.scenario_version is not None
        ):
            raise ContractViolation(
                "real outcomes must not set generator_version or scenario_version"
            )


@dataclass(frozen=True, slots=True)
class PredictionLabel:
    prediction_id: str
    state: LabelState
    materialized_at: datetime
    outcome_id: str | None = None

    @property
    def binary_label(self) -> int | None:
        if self.state is LabelState.PENDING:
            return None
        return int(self.state is LabelState.CHURNED)


def _latest_outcome_revisions(outcomes: Iterable[Outcome]) -> tuple[Outcome, ...]:
    latest: dict[tuple[str, str], Outcome] = {}
    for outcome in outcomes:
        event_key = (outcome.outcome_source, outcome.outcome_id)
        current = latest.get(event_key)
        if current is None or outcome.revision > current.revision:
            latest[event_key] = outcome
        elif outcome.revision == current.revision and outcome != current:
            raise ContractViolation(
                f"conflicting duplicate outcome revision: {outcome.outcome_id}"
            )
    return tuple(outcome for outcome in latest.values() if not outcome.is_retracted)


def _qualifies(prediction: Prediction, outcome: Outcome) -> bool:
    return (
        outcome.customer_token == prediction.customer_token
        and outcome.event_type in QUALIFYING_CHURN_EVENT_TYPES
        and prediction.prediction_timestamp < outcome.outcome_timestamp
        and outcome.outcome_timestamp <= prediction.horizon_end
    )


def attribute_outcome(
    outcome: Outcome, predictions: Iterable[Prediction]
) -> tuple[str, ...]:
    """Return every exact prediction ID whose observation window contains an outcome."""
    if outcome.is_retracted or outcome.event_type not in QUALIFYING_CHURN_EVENT_TYPES:
        return ()
    matching = (
        prediction for prediction in predictions if _qualifies(prediction, outcome)
    )
    return tuple(
        prediction.prediction_id
        for prediction in sorted(
            matching,
            key=lambda item: (item.prediction_timestamp, item.prediction_id),
        )
    )


def validate_prediction_eligibility(
    prediction: Prediction, known_outcomes: Iterable[Outcome]
) -> None:
    """Reject a prediction made at or after a known terminal churn event."""
    for outcome in _latest_outcome_revisions(known_outcomes):
        if (
            outcome.customer_token == prediction.customer_token
            and outcome.event_type in QUALIFYING_CHURN_EVENT_TYPES
            and outcome.outcome_timestamp <= prediction.prediction_timestamp
        ):
            raise ContractViolation("prediction was made after customer churn")


def materialize_label(
    prediction: Prediction,
    outcomes: Iterable[Outcome],
    *,
    observed_at: datetime,
    grace_period: timedelta = DEFAULT_GRACE_PERIOD,
) -> PredictionLabel:
    """Materialize one label tied to one persisted prediction ID.

    Positive labels may be assigned as soon as a qualifying event is observed.
    Negative labels remain pending until the full horizon and grace period elapsed.
    """
    _require_aware_utc(observed_at, "observed_at")
    if grace_period < timedelta(0):
        raise ContractViolation("grace_period must not be negative")

    qualifying = [
        outcome
        for outcome in _latest_outcome_revisions(outcomes)
        if _qualifies(prediction, outcome) and outcome.outcome_timestamp <= observed_at
    ]
    if qualifying:
        earliest = min(
            qualifying,
            key=lambda item: (item.outcome_timestamp, item.outcome_id),
        )
        return PredictionLabel(
            prediction_id=prediction.prediction_id,
            state=LabelState.CHURNED,
            materialized_at=observed_at,
            outcome_id=earliest.outcome_id,
        )

    if observed_at < prediction.horizon_end + grace_period:
        return PredictionLabel(
            prediction_id=prediction.prediction_id,
            state=LabelState.PENDING,
            materialized_at=observed_at,
        )
    return PredictionLabel(
        prediction_id=prediction.prediction_id,
        state=LabelState.NOT_CHURNED,
        materialized_at=observed_at,
    )


def materialize_negative_label(
    prediction: Prediction,
    *,
    observed_at: datetime,
    grace_period: timedelta = DEFAULT_GRACE_PERIOD,
) -> PredictionLabel:
    """Strictly reject a requested negative label before cohort maturity."""
    _require_aware_utc(observed_at, "observed_at")
    if grace_period < timedelta(0):
        raise ContractViolation("grace_period must not be negative")
    if observed_at < prediction.horizon_end + grace_period:
        raise ContractViolation("negative label requested before cohort maturity")
    return PredictionLabel(
        prediction_id=prediction.prediction_id,
        state=LabelState.NOT_CHURNED,
        materialized_at=observed_at,
    )


def _canonical_identity(environment: str, tenant_id: str, customer_id: str) -> bytes:
    values = (environment, tenant_id, customer_id)
    if any(not isinstance(value, str) or not value for value in values):
        raise ContractViolation(
            "environment, tenant_id, and customer_id are required strings"
        )
    # Length prefixes make the namespace unambiguous even when a value contains ':'.
    parts = "|".join(
        f"{len(value.encode('utf-8'))}:{value}" for value in values
    )
    return f"v1|{parts}".encode("utf-8")


def tokenize_customer_id(
    *,
    environment: str,
    tenant_id: str,
    customer_id: str,
    secret_key: bytes,
    key_id: str,
) -> str:
    """Create a deterministic, namespaced HMAC-SHA-256 monitoring token."""
    if not isinstance(secret_key, bytes) or len(secret_key) < 32:
        raise ContractViolation("secret_key must contain at least 32 bytes")
    if not key_id:
        raise ContractViolation("key_id is required")
    digest = hmac.new(
        secret_key,
        _canonical_identity(environment, tenant_id, customer_id),
        hashlib.sha256,
    ).hexdigest()
    return f"hmac-sha256:{key_id}:{digest}"


MetricT = TypeVar("MetricT")


def suppress_small_segments(
    segment_metrics: Mapping[str, tuple[int, MetricT]],
    *,
    minimum_size: int = DEFAULT_MIN_SEGMENT_SIZE,
) -> dict[str, MetricT | None]:
    """Replace every metric below k with None; v1 never creates an Other bucket."""
    if minimum_size < 2:
        raise ContractViolation("minimum_size must be at least 2")
    return {
        segment: metric if count >= minimum_size else None
        for segment, (count, metric) in segment_metrics.items()
    }


def validate_official_report(outcomes: Iterable[Outcome]) -> None:
    """Prevent simulated outcomes, or mixed real/simulated sets, in official reports."""
    materialized = tuple(outcomes)
    kinds = {outcome.is_simulated for outcome in materialized}
    if len(kinds) > 1:
        raise ContractViolation("real and simulated outcomes cannot be mixed")
    if True in kinds:
        raise ContractViolation("official reports require real outcomes")


def validate_outcome_environment(outcome: Outcome, *, environment: str) -> None:
    """Fail closed when a simulated outcome reaches a non-simulation environment."""
    if outcome.is_simulated and environment not in SIMULATION_ENVIRONMENTS:
        raise ContractViolation(
            f"simulated outcomes are not allowed in environment: {environment}"
        )


_SAFE_LOG_FIELDS = frozenset(
    {
        "contract_version",
        "prediction_id",
        "outcome_id",
        "customer_token",
        "event_type",
        "outcome_source",
        "is_simulated",
        "label_state",
        "key_id",
    }
)
_PROHIBITED_LOG_FIELDS = frozenset(
    {"customer_id", "raw_customer_id", "secret_key", "payload", "features"}
)


def log_monitoring_event(
    logger: logging.Logger, event_name: str, **fields: object
) -> None:
    """Log a fixed message with allow-listed structured fields only."""
    prohibited = set(fields) & _PROHIBITED_LOG_FIELDS
    unknown = set(fields) - _SAFE_LOG_FIELDS
    if prohibited:
        raise ContractViolation(
            f"prohibited monitoring log fields: {', '.join(sorted(prohibited))}"
        )
    if unknown:
        raise ContractViolation(
            f"unapproved monitoring log fields: {', '.join(sorted(unknown))}"
        )
    logger.info("monitoring_event=%s", event_name, extra=dict(fields))
