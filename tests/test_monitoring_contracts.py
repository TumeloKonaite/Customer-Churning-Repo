from datetime import datetime, timedelta, timezone
import logging

import pytest

from src.monitoring.contracts import (
    ContractViolation,
    LabelState,
    Outcome,
    Prediction,
    attribute_outcome,
    log_monitoring_event,
    materialize_label,
    materialize_negative_label,
    suppress_small_segments,
    tokenize_customer_id,
    validate_official_report,
    validate_outcome_environment,
    validate_prediction_eligibility,
)


UTC = timezone.utc
PREDICTION_TIME = datetime(2026, 1, 1, 12, tzinfo=UTC)
HORIZON_END = PREDICTION_TIME + timedelta(days=90)
TOKEN = "hmac-sha256:k1:customer-token"


def prediction(
    prediction_id: str = "prediction-1",
    *,
    at: datetime = PREDICTION_TIME,
    horizon_end: datetime = HORIZON_END,
) -> Prediction:
    return Prediction(
        prediction_id=prediction_id,
        customer_token=TOKEN,
        prediction_timestamp=at,
        horizon_end=horizon_end,
    )


def outcome(
    at: datetime,
    *,
    outcome_id: str = "outcome-1",
    revision: int = 1,
    is_retracted: bool = False,
    is_simulated: bool = False,
) -> Outcome:
    return Outcome(
        outcome_id=outcome_id,
        customer_token=TOKEN,
        event_type="CUSTOMER_RELATIONSHIP_TERMINATED",
        outcome_timestamp=at,
        outcome_source="simulator:test" if is_simulated else "customer-master",
        is_simulated=is_simulated,
        generator_version="generator-1" if is_simulated else None,
        scenario_version="scenario-1" if is_simulated else None,
        revision=revision,
        is_retracted=is_retracted,
    )


@pytest.mark.parametrize(
    ("event_time", "expected_state"),
    [
        (PREDICTION_TIME - timedelta(microseconds=1), LabelState.NOT_CHURNED),
        (PREDICTION_TIME, LabelState.NOT_CHURNED),
        (HORIZON_END, LabelState.CHURNED),
        (HORIZON_END + timedelta(microseconds=1), LabelState.NOT_CHURNED),
    ],
)
def test_prediction_window_boundaries(event_time, expected_state):
    label = materialize_label(
        prediction(),
        [outcome(event_time)],
        observed_at=HORIZON_END,
        grace_period=timedelta(0),
    )

    assert label.prediction_id == "prediction-1"
    assert label.state is expected_state


def test_negative_label_before_cohort_maturity_is_pending():
    label = materialize_label(
        prediction(),
        [],
        observed_at=HORIZON_END + timedelta(days=7) - timedelta(microseconds=1),
        grace_period=timedelta(days=7),
    )

    assert label.state is LabelState.PENDING
    assert label.binary_label is None


def test_explicit_negative_label_before_cohort_maturity_is_rejected():
    with pytest.raises(ContractViolation, match="before cohort maturity"):
        materialize_negative_label(
            prediction(),
            observed_at=HORIZON_END + timedelta(days=7) - timedelta(microseconds=1),
            grace_period=timedelta(days=7),
        )


def test_negative_label_at_cohort_maturity_is_allowed():
    label = materialize_label(
        prediction(),
        [],
        observed_at=HORIZON_END + timedelta(days=7),
        grace_period=timedelta(days=7),
    )

    assert label.state is LabelState.NOT_CHURNED
    assert label.binary_label == 0
    assert materialize_negative_label(
        prediction(),
        observed_at=HORIZON_END + timedelta(days=7),
        grace_period=timedelta(days=7),
    ).state is LabelState.NOT_CHURNED


def test_outcome_is_attributed_to_every_overlapping_exact_prediction_id():
    predictions = [
        prediction("later", at=PREDICTION_TIME + timedelta(days=10)),
        prediction("earlier"),
        prediction(
            "expired",
            at=PREDICTION_TIME - timedelta(days=100),
            horizon_end=PREDICTION_TIME - timedelta(days=10),
        ),
    ]

    attributed = attribute_outcome(
        outcome(PREDICTION_TIME + timedelta(days=20)), predictions
    )

    assert attributed == ("earlier", "later")
    for item in predictions[:2]:
        assert materialize_label(
            item,
            [outcome(PREDICTION_TIME + timedelta(days=20))],
            observed_at=PREDICTION_TIME + timedelta(days=20),
        ).prediction_id == item.prediction_id


def test_duplicate_outcome_delivery_is_idempotent():
    delivered = outcome(PREDICTION_TIME + timedelta(days=1))

    label = materialize_label(
        prediction(),
        [delivered, delivered],
        observed_at=PREDICTION_TIME + timedelta(days=2),
    )

    assert label.state is LabelState.CHURNED
    assert label.outcome_id == delivered.outcome_id


def test_conflicting_duplicate_revision_is_rejected():
    with pytest.raises(ContractViolation, match="conflicting duplicate"):
        materialize_label(
            prediction(),
            [
                outcome(PREDICTION_TIME + timedelta(days=1)),
                outcome(PREDICTION_TIME + timedelta(days=2)),
            ],
            observed_at=HORIZON_END,
        )


def test_higher_retraction_revision_corrects_a_positive_label():
    event_time = PREDICTION_TIME + timedelta(days=1)

    label = materialize_label(
        prediction(),
        [outcome(event_time), outcome(event_time, revision=2, is_retracted=True)],
        observed_at=HORIZON_END,
        grace_period=timedelta(0),
    )

    assert label.state is LabelState.NOT_CHURNED
    assert label.outcome_id is None


def test_prediction_after_known_churn_is_rejected():
    with pytest.raises(ContractViolation, match="after customer churn"):
        validate_prediction_eligibility(
            prediction(), [outcome(PREDICTION_TIME - timedelta(days=1))]
        )


def test_hmac_is_deterministic_for_same_key_and_namespace():
    kwargs = {
        "environment": "production",
        "tenant_id": "tenant:one",
        "customer_id": "customer-123",
        "secret_key": b"a" * 32,
        "key_id": "2026-01",
    }

    assert tokenize_customer_id(**kwargs) == tokenize_customer_id(**kwargs)


def test_hmac_changes_between_environment_and_tenant_namespaces():
    shared = {
        "customer_id": "customer-123",
        "secret_key": b"a" * 32,
        "key_id": "2026-01",
    }

    production = tokenize_customer_id(
        environment="production", tenant_id="tenant-a", **shared
    )
    staging = tokenize_customer_id(
        environment="staging", tenant_id="tenant-a", **shared
    )
    other_tenant = tokenize_customer_id(
        environment="production", tenant_id="tenant-b", **shared
    )

    assert len({production, staging, other_tenant}) == 3


def test_raw_customer_identifiers_are_rejected_before_logging(caplog):
    logger = logging.getLogger("test.monitoring")
    raw_customer_id = "raw-customer-123"

    with caplog.at_level(logging.INFO), pytest.raises(
        ContractViolation, match="prohibited monitoring log fields"
    ):
        log_monitoring_event(
            logger,
            "label_materialized",
            prediction_id="prediction-1",
            customer_id=raw_customer_id,
        )

    assert raw_customer_id not in caplog.text
    assert not caplog.records


def test_safe_monitoring_log_contains_token_but_not_raw_identifier(caplog):
    logger = logging.getLogger("test.monitoring")

    with caplog.at_level(logging.INFO):
        log_monitoring_event(
            logger,
            "label_materialized",
            contract_version="1.0.0",
            prediction_id="prediction-1",
            customer_token=TOKEN,
        )

    assert len(caplog.records) == 1
    assert caplog.records[0].customer_token == TOKEN
    assert "customer_id" not in caplog.records[0].__dict__


def test_segments_smaller_than_k_are_suppressed():
    result = suppress_small_segments(
        {"small": (19, 0.75), "at-threshold": (20, 0.50)}, minimum_size=20
    )

    assert result == {"small": None, "at-threshold": 0.50}
    assert "Other" not in result


def test_real_and_simulated_outcomes_cannot_mix_in_official_report():
    with pytest.raises(ContractViolation, match="cannot be mixed"):
        validate_official_report(
            [
                outcome(PREDICTION_TIME + timedelta(days=1)),
                outcome(
                    PREDICTION_TIME + timedelta(days=2),
                    outcome_id="simulated-1",
                    is_simulated=True,
                ),
            ]
        )


def test_simulated_only_outcomes_are_not_official():
    with pytest.raises(ContractViolation, match="require real outcomes"):
        validate_official_report(
            [outcome(PREDICTION_TIME + timedelta(days=1), is_simulated=True)]
        )


def test_simulated_outcomes_are_rejected_in_production_environment():
    simulated = outcome(PREDICTION_TIME + timedelta(days=1), is_simulated=True)

    with pytest.raises(ContractViolation, match="not allowed in environment"):
        validate_outcome_environment(simulated, environment="production")

    validate_outcome_environment(simulated, environment="staging")


def test_real_only_outcomes_are_valid_for_official_report():
    validate_official_report([outcome(PREDICTION_TIME + timedelta(days=1))])
