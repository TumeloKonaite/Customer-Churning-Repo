from __future__ import annotations

from datetime import datetime, timezone

import pytest
import pandas as pd

from src.model_schema import CANONICAL_FEATURE_ORDER
from src.monitoring.arize.client import ArizeDeliveryError, ArizePermanentError
from src.monitoring.arize.baseline import validate_reference
from src.monitoring.arize.config import ArizeExportSettings
from src.monitoring.arize.exporter import ArizeExporter
from src.monitoring.arize.schema import (
    ARIZE_FEATURES,
    actual_frame,
    binary_label,
    numeric_model_version,
    prediction_frame,
)


UTC = timezone.utc


def prediction_row(**changes):
    row = {
        "export_event_id": "event-1",
        "event_type": "prediction",
        "attempt_count": 1,
        "prediction_id": "opaque-prediction-1",
        "environment": "production",
        "model_version_id": "dagshub:owner/repo:churn_predictor:5",
        "prediction_timestamp": datetime(2026, 1, 1, tzinfo=UTC),
        "feature_schema_version": "1.0.0",
        "features": {name: index for index, name in enumerate(CANONICAL_FEATURE_ORDER)},
        "prediction_probability": 0.82,
        "predicted_class": "1",
        "deployment_id": "deployment-1",
        "request_source": "single",
        "batch_id": None,
    }
    row.update(changes)
    return row


def test_exact_ten_feature_and_binary_mapping():
    frame = prediction_frame([prediction_row()])
    assert ARIZE_FEATURES == tuple(CANONICAL_FEATURE_ORDER)
    assert list(frame[CANONICAL_FEATURE_ORDER]) == CANONICAL_FEATURE_ORDER
    assert frame.loc[0, "prediction_label"] == "churn"
    assert frame.loc[0, "prediction_score"] == 0.82
    assert frame.loc[0, "prediction_id"] == "opaque-prediction-1"
    assert frame.index.tolist() == [0]
    assert binary_label(0) == "no_churn"
    assert binary_label(1) == "churn"


def test_unapproved_extra_or_missing_feature_is_rejected():
    row = prediction_row()
    row["features"] = {**row["features"], "email": "unsafe@example.com"}
    with pytest.raises(ValueError, match="exactly the ten"):
        prediction_frame([row])


def test_baseline_checksum_schema_and_empty_validation():
    frame = pd.DataFrame([{name: index for index, name in enumerate(CANONICAL_FEATURE_ORDER)}])
    body = frame.to_parquet(index=False)
    from hashlib import sha256

    validated = validate_reference(
        body, expected_sha256=sha256(body).hexdigest(), schema_version="1.0.0"
    )
    assert list(validated.columns) == CANONICAL_FEATURE_ORDER
    with pytest.raises(ValueError, match="checksum"):
        validate_reference(body, expected_sha256="0" * 64, schema_version="1.0.0")


def test_delayed_actual_reuses_prediction_id_and_original_version():
    row = prediction_row(
        event_type="actual", label_value=0,
        label_created_at=datetime(2026, 5, 1, tzinfo=UTC),
        label_contract_version="1.0.0",
    )
    frame = actual_frame([row])
    assert frame.loc[0, "prediction_id"] == row["prediction_id"]
    assert frame.loc[0, "model_version_id"] == row["model_version_id"]
    assert frame.loc[0, "actual_label"] == "no_churn"


def test_exact_numeric_model_version():
    assert numeric_model_version("dagshub:o/r:churn_predictor:17") == "17"
    with pytest.raises(ValueError):
        numeric_model_version("medical_cost:regression:1")


def test_export_settings_cannot_enable_without_approval_or_secrets():
    with pytest.raises(ValueError, match="API key"):
        ArizeExportSettings(export_enabled=True)
    with pytest.raises(ValueError, match="privacy approval"):
        ArizeExportSettings(
            export_enabled=True, api_key="secret", space_id="space"
        )


class FakeRepository:
    def __init__(self, rows):
        self.rows = rows
        self.sent = []
        self.failures = []

    def require_active_approval(self, **kwargs):
        self.approval = kwargs

    def recover_stale_claims(self, **kwargs):
        return 1

    def claim(self, **kwargs):
        return self.rows

    def mark_sent(self, identifiers, **kwargs):
        self.sent.extend(identifiers)

    def mark_failed(self, identifiers, **kwargs):
        self.failures.append((list(identifiers), kwargs))
        return (len(list(identifiers)), 0) if kwargs["retryable"] else (0, len(list(identifiers)))

    def backlog(self, **kwargs):
        return {"remaining_backlog": 0, "dead_letter_backlog": 0,
                "oldest_pending_seconds": None}


class FakeClient:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def log_predictions(self, frame, **kwargs):
        self.calls.append((frame, kwargs))
        if self.error:
            raise self.error

    def log_actuals(self, frame, **kwargs):
        self.log_predictions(frame, **kwargs)


def enabled_settings(**changes):
    values = dict(
        environment="production", export_enabled=True, api_key="secret",
        space_id="space", privacy_approval_id="approval-1",
    )
    values.update(changes)
    return ArizeExportSettings(**values)


def test_exporter_acknowledges_then_marks_sent():
    repository = FakeRepository([prediction_row()])
    client = FakeClient()
    summary = ArizeExporter(repository, client, enabled_settings()).run(
        now=datetime(2026, 1, 2, tzinfo=UTC)
    )
    assert repository.sent == ["event-1"]
    assert summary["sent"] == 1
    assert summary["stale_claims_recovered"] == 1


@pytest.mark.parametrize(
    ("error", "retryable"),
    [(ArizeDeliveryError("temporary"), True), (ArizePermanentError("bad schema"), False)],
)
def test_exporter_retry_and_dead_letter_classification(error, retryable):
    repository = FakeRepository([prediction_row()])
    summary = ArizeExporter(
        repository, FakeClient(error), enabled_settings(initial_backoff_seconds=60)
    ).run(now=datetime(2026, 1, 2, tzinfo=UTC))
    assert repository.failures[0][1]["retryable"] is retryable
    assert summary["retried" if retryable else "dead_lettered"] == 1
