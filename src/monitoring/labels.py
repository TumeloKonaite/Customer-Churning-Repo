"""Deterministic outcome attribution and append-only label materialization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum
import hashlib
from typing import Any, Callable, Iterable, Mapping, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.monitoring.contracts import CONTRACT_VERSION, QUALIFYING_CHURN_EVENT_TYPES
from src.monitoring.models import canonical_json_bytes, require_utc, timestamp
from src.monitoring.outcomes import CanonicalOutcome, OutcomeOperation


class LabelStatus(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    PENDING = "pending"


class LabelingPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prediction_id: str = Field(min_length=1)
    environment: str = Field(min_length=1)
    customer_token: str = Field(min_length=1)
    model_version_id: str = Field(min_length=1)
    deployment_id: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    prediction_timestamp: datetime
    horizon_end: datetime
    label_contract_version: str = Field(min_length=1)
    monitoring_eligible: bool = True
    prediction_probability: float = Field(ge=0, le=1)
    predicted_class: str
    segments: dict[str, str] = Field(default_factory=dict)

    @field_validator("prediction_timestamp", "horizon_end")
    @classmethod
    def values_are_utc(cls, value: datetime, info) -> datetime:
        return require_utc(value, info.field_name)


class OutcomeSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    as_of: datetime
    maximum_outcome_ingest_id: int = Field(ge=0)
    required_sources: tuple[str, ...]
    source_complete_through: dict[str, datetime | None]

    @field_validator("as_of")
    @classmethod
    def as_of_is_utc(cls, value: datetime) -> datetime:
        return require_utc(value, "as_of")

    @property
    def identity(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def sources_complete_for(self, event_horizon_end: datetime) -> bool:
        return all(
            self.source_complete_through.get(source) is not None
            and self.source_complete_through[source] >= event_horizon_end
            for source in self.required_sources
        )


class LabelDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prediction_id: str
    label_value: int | None
    status: LabelStatus
    qualifying_outcome_event_id: str | None = None
    reason: str


class LabelRevision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label_revision_id: int | None = None
    prediction_id: str
    revision_number: int = Field(ge=1)
    label_value: int | None
    status: LabelStatus
    qualifying_outcome_event_id: str | None = None
    label_contract_version: str
    materialization_run_id: str
    attribution_timestamp: datetime
    created_at: datetime
    supersedes_label_revision_id: int | None = None
    revision_reason: str
    simulation_generator: str | None = None
    simulation_scenario_version: str | None = None

    @field_validator("attribution_timestamp", "created_at")
    @classmethod
    def values_are_utc(cls, value: datetime, info) -> datetime:
        return require_utc(value, info.field_name)

    @model_validator(mode="after")
    def simulation_metadata_is_paired(self) -> "LabelRevision":
        if (self.simulation_generator is None) != (
            self.simulation_scenario_version is None
        ):
            raise ValueError("simulation generator and scenario must be recorded together")
        return self


def resolve_active_outcomes(
    outcomes: Iterable[CanonicalOutcome],
) -> tuple[CanonicalOutcome, ...]:
    """Resolve correction chains without mutating or discarding source history."""
    events = tuple(outcomes)
    by_id: dict[str, CanonicalOutcome] = {}
    for event in events:
        existing = by_id.get(event.outcome_event_id)
        if existing is not None and existing != event:
            raise ValueError("conflicting outcome_event_id in snapshot")
        by_id[event.outcome_event_id] = event
    referenced = {
        event.referenced_outcome_event_id
        for event in events
        if event.referenced_outcome_event_id is not None
    }
    leaves = [event for event in events if event.outcome_event_id not in referenced]
    return tuple(
        sorted(
            (event for event in leaves if event.operation is not OutcomeOperation.RETRACTION),
            key=lambda item: (
                item.event_timestamp,
                item.source_namespace,
                item.source_event_id,
            ),
        )
    )


def qualifying_outcomes(
    prediction: LabelingPrediction, outcomes: Iterable[CanonicalOutcome]
) -> tuple[CanonicalOutcome, ...]:
    return tuple(
        event
        for event in resolve_active_outcomes(outcomes)
        if event.customer_token == prediction.customer_token
        and event.environment == prediction.environment
        and event.label_contract_version == prediction.label_contract_version
        and event.event_type in QUALIFYING_CHURN_EVENT_TYPES
        and prediction.prediction_timestamp < event.event_timestamp <= prediction.horizon_end
    )


def decide_label(
    prediction: LabelingPrediction,
    outcomes: Iterable[CanonicalOutcome],
    *,
    snapshot: OutcomeSnapshot,
    grace_period: timedelta,
    expected_horizon: timedelta | None = None,
) -> LabelDecision:
    """Apply the exclusive-start, inclusive-end window and maturity gates."""
    if grace_period < timedelta(0):
        raise ValueError("grace_period must not be negative")
    if not prediction.monitoring_eligible:
        return LabelDecision(
            prediction_id=prediction.prediction_id,
            label_value=None,
            status=LabelStatus.PENDING,
            reason="prediction_not_monitoring_eligible",
        )
    if (
        expected_horizon is not None
        and prediction.horizon_end != prediction.prediction_timestamp + expected_horizon
    ):
        return LabelDecision(
            prediction_id=prediction.prediction_id,
            label_value=None,
            status=LabelStatus.PENDING,
            reason="prediction_horizon_contract_mismatch",
        )
    active_outcomes = resolve_active_outcomes(outcomes)
    if any(
        event.customer_token == prediction.customer_token
        and event.environment == prediction.environment
        and event.label_contract_version == prediction.label_contract_version
        and event.event_type in QUALIFYING_CHURN_EVENT_TYPES
        and event.event_timestamp < prediction.prediction_timestamp
        for event in active_outcomes
    ):
        return LabelDecision(
            prediction_id=prediction.prediction_id,
            label_value=None,
            status=LabelStatus.PENDING,
            reason="prediction_at_or_after_known_terminal_outcome",
        )
    matching = qualifying_outcomes(prediction, active_outcomes)
    if matching:
        selected = matching[0]
        return LabelDecision(
            prediction_id=prediction.prediction_id,
            label_value=1,
            status=LabelStatus.POSITIVE,
            qualifying_outcome_event_id=selected.outcome_event_id,
            reason="qualifying_outcome",
        )
    if snapshot.as_of < prediction.horizon_end + grace_period:
        return LabelDecision(
            prediction_id=prediction.prediction_id,
            label_value=None,
            status=LabelStatus.PENDING,
            reason="prediction_not_yet_mature",
        )
    if not snapshot.sources_complete_for(prediction.horizon_end):
        return LabelDecision(
            prediction_id=prediction.prediction_id,
            label_value=None,
            status=LabelStatus.PENDING,
            reason="awaiting_source_completeness",
        )
    return LabelDecision(
        prediction_id=prediction.prediction_id,
        label_value=0,
        status=LabelStatus.NEGATIVE,
        reason="mature_without_qualifying_outcome",
    )


def materialization_run_id(
    *,
    environment: str,
    is_simulated: bool,
    snapshot: OutcomeSnapshot,
    label_contract_version: str,
    horizon_days: int,
    grace_period_days: int,
    simulation_generator: str | None = None,
    simulation_scenario_version: str | None = None,
) -> str:
    identity = {
        "environment": environment,
        "is_simulated": is_simulated,
        "snapshot": snapshot.identity,
        "label_contract_version": label_contract_version,
        "horizon_days": horizon_days,
        "grace_period_days": grace_period_days,
        "simulation_generator": simulation_generator,
        "simulation_scenario_version": simulation_scenario_version,
    }
    return "labels_" + hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


class LabelMaterializationStore(Protocol):
    def establish_outcome_snapshot(self, *, environment: str, is_simulated: bool, as_of: datetime, required_sources: tuple[str, ...]) -> OutcomeSnapshot: ...
    def reserve_materialization_run(self, metadata: dict[str, Any]) -> tuple[dict[str, Any], bool]: ...
    def load_labeling_predictions(self, *, environment: str, as_of: datetime) -> tuple[LabelingPrediction, ...]: ...
    def load_outcomes(self, *, environment: str, is_simulated: bool, snapshot: OutcomeSnapshot, simulation_generator: str | None, simulation_scenario_version: str | None) -> tuple[CanonicalOutcome, ...]: ...
    def latest_label_revisions(self, prediction_ids: tuple[str, ...], *, is_simulated: bool, simulation_generator: str | None, simulation_scenario_version: str | None) -> Mapping[str, LabelRevision]: ...
    def append_label_revision(self, revision: LabelRevision, *, is_simulated: bool) -> LabelRevision: ...
    def count_quarantined_outcomes(self, *, as_of: datetime) -> int: ...
    def finish_materialization_run(self, run_id: str, summary: dict[str, Any]) -> None: ...
    def fail_materialization_run(self, run_id: str, error: dict[str, Any]) -> None: ...


@dataclass(slots=True)
class LabelMaterializationJob:
    store: LabelMaterializationStore
    required_sources: tuple[str, ...]
    horizon_days: int = 90
    grace_period_days: int = 7
    label_contract_version: str = CONTRACT_VERSION
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)

    def run(
        self,
        *,
        environment: str,
        is_simulated: bool = False,
        simulation_generator: str | None = None,
        simulation_scenario_version: str | None = None,
        as_of: datetime | None = None,
    ) -> dict[str, Any]:
        if is_simulated and (
            not simulation_generator or not simulation_scenario_version
        ):
            raise ValueError("simulated label runs require generator and scenario version")
        if not is_simulated and (
            simulation_generator is not None or simulation_scenario_version is not None
        ):
            raise ValueError("real label runs cannot specify simulation metadata")
        execution_started_at = require_utc(self.now(), "execution_started_at")
        as_of = require_utc(as_of or execution_started_at, "as_of")
        snapshot = self.store.establish_outcome_snapshot(
            environment=environment,
            is_simulated=is_simulated,
            as_of=as_of,
            required_sources=self.required_sources,
        )
        run_id = materialization_run_id(
            environment=environment,
            is_simulated=is_simulated,
            snapshot=snapshot,
            label_contract_version=self.label_contract_version,
            horizon_days=self.horizon_days,
            grace_period_days=self.grace_period_days,
            simulation_generator=simulation_generator,
            simulation_scenario_version=simulation_scenario_version,
        )
        metadata = {
            "materialization_run_id": run_id,
            "environment": environment,
            "is_simulated": is_simulated,
            "simulation_generator": simulation_generator,
            "simulation_scenario_version": simulation_scenario_version,
            "label_contract_version": self.label_contract_version,
            "horizon_days": self.horizon_days,
            "grace_period_days": self.grace_period_days,
            "outcome_watermark": snapshot.model_dump(mode="json"),
            "started_at": execution_started_at,
        }
        existing, owns_execution = self.store.reserve_materialization_run(metadata)
        if not owns_execution:
            return existing.get("summary") or existing

        counts = {
            "predictions_examined": 0,
            "positive_labels_added": 0,
            "negative_labels_added": 0,
            "predictions_not_yet_mature": 0,
            "predictions_awaiting_source_completeness": 0,
            "predictions_ineligible": 0,
            "corrections_or_revisions_applied": 0,
            "unchanged_labels": 0,
            "errors": 0,
            "quarantined_records": self.store.count_quarantined_outcomes(as_of=as_of),
        }
        error_details: list[dict[str, str]] = []
        try:
            predictions = self.store.load_labeling_predictions(
                environment=environment, as_of=as_of
            )
            outcomes = self.store.load_outcomes(
                environment=environment,
                is_simulated=is_simulated,
                snapshot=snapshot,
                simulation_generator=simulation_generator,
                simulation_scenario_version=simulation_scenario_version,
            )
            active = resolve_active_outcomes(outcomes)
            latest = self.store.latest_label_revisions(
                tuple(item.prediction_id for item in predictions),
                is_simulated=is_simulated,
                simulation_generator=simulation_generator,
                simulation_scenario_version=simulation_scenario_version,
            )
            label_revision_watermark = max(
                (
                    revision.label_revision_id or 0
                    for revision in latest.values()
                ),
                default=0,
            )
            counts["predictions_examined"] = len(predictions)
            grace = timedelta(days=self.grace_period_days)
            for prediction in predictions:
                try:
                    decision = decide_label(
                        prediction,
                        active,
                        snapshot=snapshot,
                        grace_period=grace,
                        expected_horizon=timedelta(days=self.horizon_days),
                    )
                    previous = latest.get(prediction.prediction_id)
                    if decision.status is LabelStatus.PENDING and previous is None:
                        counts[_pending_counter(decision.reason)] += 1
                        continue
                    if previous is not None and _same_label(previous, decision):
                        counts["unchanged_labels"] += 1
                        continue
                    reason = decision.reason
                    if previous is not None:
                        counts["corrections_or_revisions_applied"] += 1
                        reason = _revision_reason(previous, decision)
                    revision = LabelRevision(
                        prediction_id=prediction.prediction_id,
                        revision_number=(previous.revision_number + 1 if previous else 1),
                        label_value=decision.label_value,
                        status=decision.status,
                        qualifying_outcome_event_id=decision.qualifying_outcome_event_id,
                        label_contract_version=prediction.label_contract_version,
                        materialization_run_id=run_id,
                        attribution_timestamp=as_of,
                        created_at=execution_started_at,
                        supersedes_label_revision_id=(
                            previous.label_revision_id if previous else None
                        ),
                        revision_reason=reason,
                        simulation_generator=simulation_generator,
                        simulation_scenario_version=simulation_scenario_version,
                    )
                    stored_revision = self.store.append_label_revision(
                        revision, is_simulated=is_simulated
                    )
                    label_revision_watermark = max(
                        label_revision_watermark,
                        stored_revision.label_revision_id or 0,
                    )
                    if decision.status is LabelStatus.POSITIVE:
                        counts["positive_labels_added"] += 1
                    elif decision.status is LabelStatus.NEGATIVE:
                        counts["negative_labels_added"] += 1
                    else:
                        counts[_pending_counter(decision.reason)] += 1
                except Exception as exc:
                    counts["errors"] += 1
                    error_details.append(
                        {
                            "prediction_id": prediction.prediction_id,
                            "type": type(exc).__name__,
                        }
                    )
            summary = {
                "materialization_run_id": run_id,
                "status": "completed" if counts["errors"] == 0 else "completed_with_errors",
                "environment": environment,
                "is_simulated": is_simulated,
                "simulation_generator": simulation_generator,
                "simulation_scenario_version": simulation_scenario_version,
                "label_contract_version": self.label_contract_version,
                "outcome_watermark": snapshot.model_dump(mode="json"),
                "label_revision_watermark": label_revision_watermark,
                "counts": counts,
                "exclusions": {
                    "not_yet_mature": counts["predictions_not_yet_mature"],
                    "awaiting_source_completeness": counts[
                        "predictions_awaiting_source_completeness"
                    ],
                    "monitoring_ineligible": counts["predictions_ineligible"],
                },
                "error_details": error_details,
                "completed_at": timestamp(require_utc(self.now(), "completed_at")),
            }
            self.store.finish_materialization_run(run_id, summary)
            return summary
        except Exception as exc:
            self.store.fail_materialization_run(
                run_id,
                {"type": type(exc).__name__, "message": "label materialization failed"},
            )
            raise


def _pending_counter(reason: str) -> str:
    return {
        "prediction_not_yet_mature": "predictions_not_yet_mature",
        "awaiting_source_completeness": "predictions_awaiting_source_completeness",
        "prediction_not_monitoring_eligible": "predictions_ineligible",
        "prediction_horizon_contract_mismatch": "predictions_ineligible",
        "prediction_at_or_after_known_terminal_outcome": "predictions_ineligible",
    }.get(reason, "predictions_not_yet_mature")


def _same_label(previous: LabelRevision, decision: LabelDecision) -> bool:
    return (
        previous.label_value == decision.label_value
        and previous.status is decision.status
        and previous.qualifying_outcome_event_id
        == decision.qualifying_outcome_event_id
    )


def _revision_reason(previous: LabelRevision, decision: LabelDecision) -> str:
    if previous.status is LabelStatus.NEGATIVE and decision.status is LabelStatus.POSITIVE:
        return "late_arriving_positive"
    if previous.status is LabelStatus.POSITIVE and decision.status is LabelStatus.PENDING:
        return "positive_retracted_or_corrected_before_maturity"
    if previous.status is LabelStatus.POSITIVE and decision.status is LabelStatus.NEGATIVE:
        return "positive_retracted_or_corrected_after_maturity"
    if previous.qualifying_outcome_event_id != decision.qualifying_outcome_event_id:
        return "qualifying_outcome_revised"
    return "label_recomputed"
