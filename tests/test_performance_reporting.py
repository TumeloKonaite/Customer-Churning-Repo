from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.monitoring.performance import (
    CohortDefinition,
    PerformanceRecord,
    build_performance_report,
    classification_metrics,
)


UTC = timezone.utc
START = datetime(2026, 1, 1, tzinfo=UTC)


def record(index, label, probability, *, segment="France"):
    return PerformanceRecord(
        prediction_id=f"prediction-{index}",
        prediction_timestamp=START + timedelta(days=index),
        horizon_end=START + timedelta(days=index + 90),
        model_version_id="model:1",
        deployment_id="deployment-1",
        policy_version="policy-1",
        label_contract_version="1.0.0",
        prediction_probability=probability,
        predicted_class=int(probability >= 0.5),
        label_value=label,
        label_revision_id=index + 1,
        label_attribution_timestamp=START + timedelta(days=100),
        is_simulated=False,
        segments={"Geography": segment},
    )


def definition(**changes):
    values = {
        "cohort_start": START,
        "cohort_end": START + timedelta(days=30),
        "horizon_days": 90,
        "grace_period_days": 7,
        "outcome_watermark": {
            "maximum_outcome_ingest_id": 10,
            "required_sources": ["customer-master"],
            "source_complete_through": {
                "customer-master": (START + timedelta(days=200)).isoformat()
            },
        },
        "label_revision_watermark": 100,
        "label_contract_version": "1.0.0",
        "model_version_id": "model:1",
        "deployment_ids": ("deployment-1",),
        "policy_version": "policy-1",
        "is_simulated": False,
        "classification_threshold": 0.5,
    }
    values.update(changes)
    return CohortDefinition(**values)


def test_metrics_are_reproducible_and_auc_is_unavailable_for_one_class():
    metrics = classification_metrics(
        [record(1, 0, 0.1), record(2, 0, 0.8)], threshold=0.5
    )
    assert metrics["confusion_matrix"]["value"] == {
        "true_negative": 1, "false_positive": 1,
        "false_negative": 0, "true_positive": 0,
    }
    assert metrics["roc_auc"]["available"] is False
    assert "both observed classes" in metrics["roc_auc"]["reason"]
    assert metrics["recall"]["value"] is None


def test_report_contains_matured_metadata_calibration_and_threshold_metrics():
    rows = [record(i, i % 2, 0.8 if i % 2 else 0.2) for i in range(20)]
    report = build_performance_report(
        rows,
        definition=definition(),
        evaluated_at=START + timedelta(days=130),
        approved_segments={"Geography": "geo-v1"},
        minimum_privacy_size=5,
        calibration_options={"minimum_bin_volume": 2},
    )
    assert report["official"] is True
    assert report["cohort"]["selection_rule"] == "prediction_timestamp"
    assert report["cohort"]["eligible_prediction_count"] == 20
    assert report["classification"]["deployed_threshold"]["threshold"] == 0.5
    assert report["calibration"]["brier_score"]["available"] is True


def test_small_segment_and_complement_are_suppressed_without_counts_or_names():
    rows = [record(i, i % 2, 0.5, segment="large") for i in range(10)]
    rows += [record(i + 10, 0, 0.2, segment="tiny") for i in range(2)]
    report = build_performance_report(
        rows,
        definition=definition(),
        evaluated_at=START + timedelta(days=130),
        approved_segments={"Geography": "geo-v1"},
        minimum_privacy_size=5,
    )
    dimension = report["segments"]["Geography"]
    assert dimension["published"] == []
    assert dimension["suppression"]["suppressed_group_count"] == 2
    assert "tiny" not in repr(dimension)
