from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.monitoring.labels import (
    LabelStatus,
    LabelingPrediction,
    OutcomeSnapshot,
    decide_label,
    qualifying_outcomes,
    resolve_active_outcomes,
)
from src.monitoring.outcomes import CanonicalOutcome, OutcomeOperation, outcome_event_id


UTC = timezone.utc
PREDICTED = datetime(2026, 1, 1, tzinfo=UTC)
HORIZON = PREDICTED + timedelta(days=90)


def prediction():
    return LabelingPrediction(
        prediction_id="prediction-1",
        environment="production",
        customer_token="hmac-sha256:key:abc",
        model_version_id="model:1",
        deployment_id="deployment-1",
        policy_version="policy-1",
        prediction_timestamp=PREDICTED,
        horizon_end=HORIZON,
        label_contract_version="1.0.0",
        prediction_probability=0.8,
        predicted_class="1",
    )


def outcome(at, *, source_id="outcome-1", operation=OutcomeOperation.CREATE, reference=None):
    return CanonicalOutcome(
        outcome_event_id=outcome_event_id("customer-master", source_id),
        source_event_id=source_id,
        source_namespace="customer-master",
        environment="production",
        customer_token="hmac-sha256:key:abc",
        token_key_id="key",
        event_type="CUSTOMER_RELATIONSHIP_TERMINATED",
        event_timestamp=at,
        received_timestamp=HORIZON + timedelta(days=8),
        operation=operation,
        referenced_outcome_event_id=reference,
        is_simulated=False,
    )


def snapshot(*, as_of=HORIZON + timedelta(days=7), complete=HORIZON):
    return OutcomeSnapshot(
        as_of=as_of,
        maximum_outcome_ingest_id=10,
        required_sources=("customer-master",),
        source_complete_through={"customer-master": complete},
    )


def test_attribution_boundaries_are_start_exclusive_end_inclusive():
    events = [outcome(PREDICTED), outcome(HORIZON, source_id="end")]
    assert [item.source_event_id for item in qualifying_outcomes(prediction(), events)] == ["end"]


def test_negative_requires_maturity_and_source_completeness():
    immature = decide_label(
        prediction(), [],
        snapshot=snapshot(as_of=HORIZON + timedelta(days=7) - timedelta(microseconds=1)),
        grace_period=timedelta(days=7),
    )
    incomplete = decide_label(
        prediction(), [], snapshot=snapshot(complete=HORIZON - timedelta(seconds=1)),
        grace_period=timedelta(days=7),
    )
    mature = decide_label(
        prediction(), [], snapshot=snapshot(), grace_period=timedelta(days=7)
    )
    assert immature.reason == "prediction_not_yet_mature"
    assert incomplete.reason == "awaiting_source_completeness"
    assert mature.status is LabelStatus.NEGATIVE


def test_correction_and_retraction_resolve_without_destroying_history():
    original = outcome(HORIZON)
    correction = outcome(
        PREDICTED,
        source_id="correction",
        operation=OutcomeOperation.CORRECTION,
        reference=original.outcome_event_id,
    )
    assert resolve_active_outcomes([original, correction]) == (correction,)
    assert decide_label(
        prediction(), [original, correction], snapshot=snapshot(),
        grace_period=timedelta(days=7),
    ).status is LabelStatus.NEGATIVE

    retraction = outcome(
        PREDICTED,
        source_id="retraction",
        operation=OutcomeOperation.RETRACTION,
        reference=correction.outcome_event_id,
    )
    assert resolve_active_outcomes([original, correction, retraction]) == ()
