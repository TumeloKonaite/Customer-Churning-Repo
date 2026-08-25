from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pandas as pd
import pytest

from src.monitoring.shared.artifacts import (
    ArtifactConflictError,
    LocalArtifactStore,
    artifact_prefix,
)
from src.monitoring.drift.evidently import EvidentlyOutput
from src.monitoring.drift.service import MonitoringJob, cadence_slot, schedule_minute_offset
from src.monitoring.shared.models import (
    BaselineVersion,
    ExtractionWatermark,
    MonitoringPolicy,
    PredictionRecord,
    ResultStatus,
    RunStatus,
    monitoring_run_id,
    sha256_bytes,
)
from src.monitoring.drift.quality import MonitoringValidationError, data_quality_summary
from src.monitoring.drift.selection import deterministically_limit, eligible_record, select_window


UTC = timezone.utc
END = datetime(2026, 8, 22, 12, tzinfo=UTC)
MODEL = "dagshub:owner/repository:churn_predictor:7"


def policy(**changes) -> MonitoringPolicy:
    value = json.loads(Path("configs/monitoring/policy-v1.0.0.json").read_text())
    value.update(changes)
    return MonitoringPolicy.model_validate(value)


def record(event_id: int, at: datetime, **changes) -> PredictionRecord:
    features = {
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Female",
        "Age": 40,
        "Tenure": 4,
        "Balance": 1000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000.0,
    }
    values = {
        "event_id": event_id,
        "prediction_id": f"00000000-0000-0000-0000-{event_id:012d}",
        "environment": "production",
        "model_version_id": MODEL,
        "prediction_timestamp": at,
        "persisted_at": at + timedelta(minutes=1),
        "feature_schema_version": "1.0.0",
        "features": features,
        "prediction_probability": 0.2,
        "predicted_class": "0",
    }
    values.update(changes)
    return PredictionRecord.model_validate(values)


def test_window_expands_geometrically_and_honors_hard_boundary():
    current_policy = policy(
        minimum_current_rows=5,
        maximum_current_rows=10,
        initial_lookback_hours=24,
        maximum_lookback_hours=96,
        fixed_historical_boundary=(END - timedelta(hours=72)).isoformat(),
    )
    calls = []

    def count(start, end, watermark):
        del end, watermark
        calls.append(start)
        return {24: 1, 48: 3, 72: 4}[int((END - start).total_seconds() / 3600)]

    watermark = ExtractionWatermark(
        extraction_cutoff=END, maximum_persisted_event_id=10
    )
    selected = select_window(
        end=END, watermark=watermark, policy=current_policy, count_rows=count
    )

    assert [int((END - value).total_seconds() / 3600) for value in calls] == [24, 48, 72]
    assert selected.observed_rows == 4
    assert selected.reached_boundary is True


def test_row_limit_and_watermark_are_stable_and_model_specific():
    rows = [record(1, END - timedelta(hours=2)), record(2, END - timedelta(hours=1))]
    rows.append(record(3, END - timedelta(hours=1)))
    assert [item.event_id for item in deterministically_limit(rows, 2)] == [2, 3]
    selected_window = select_window(
        end=END,
        watermark=ExtractionWatermark(
            extraction_cutoff=END, maximum_persisted_event_id=2
        ),
        policy=policy(minimum_current_rows=1),
        count_rows=lambda *args: 2,
    )
    watermark = ExtractionWatermark(
        extraction_cutoff=END, maximum_persisted_event_id=2
    )
    late = record(
        3,
        END - timedelta(hours=1),
        persisted_at=END + timedelta(seconds=1),
    )
    other_model = record(2, END - timedelta(hours=1), model_version_id="other:8")
    assert not eligible_record(
        late,
        environment="production",
        model_version_id=MODEL,
        window=selected_window,
        watermark=watermark,
    )
    assert not eligible_record(
        other_model,
        environment="production",
        model_version_id=MODEL,
        window=selected_window,
        watermark=watermark,
    )


def test_run_identity_changes_with_baseline_policy_window_or_watermark():
    current_policy = policy(minimum_current_rows=1)
    watermark = ExtractionWatermark(
        extraction_cutoff=END, maximum_persisted_event_id=2
    )
    window = select_window(
        end=END,
        watermark=watermark,
        policy=current_policy,
        count_rows=lambda *args: 1,
    )
    values = dict(
        job_type="data_quality_and_drift",
        environment="production",
        model_version_id=MODEL,
        baseline_version_id="baseline-1",
        policy_version="1.0.0",
        window=window,
        watermark=watermark,
    )
    first = monitoring_run_id(**values)
    assert first == monitoring_run_id(**values)
    assert first != monitoring_run_id(**{**values, "baseline_version_id": "baseline-2"})
    assert first != monitoring_run_id(
        **{
            **values,
            "watermark": watermark.model_copy(
                update={"maximum_persisted_event_id": 3}
            ),
        }
    )


def test_data_quality_distinguishes_hard_schema_failures_from_warnings():
    current_policy = policy(minimum_current_rows=1, minimum_reference_rows=1)
    base = record(1, END).features
    current = pd.DataFrame(
        [
            {**base, "prediction_id": "duplicate", "prediction_probability": 1.2, "predicted_class": "1"},
            {**base, "Age": 101, "prediction_id": "duplicate", "prediction_probability": 0.2, "predicted_class": "0"},
        ]
    )
    summary = data_quality_summary(current, current.iloc[:1], policy=current_policy)
    assert summary["status"] == ResultStatus.FAIL
    assert summary["feature_results"]["Age"]["violations"] == {"above_maximum": 1}
    with pytest.raises(MonitoringValidationError, match="unsupported"):
        bad = current_policy.model_copy(
            update={
                "feature_rules": {
                    **current_policy.feature_rules,
                    "Age": current_policy.feature_rules["Age"].model_copy(
                        update={"data_type": "mystery"}
                    ),
                }
            }
        )
        from src.monitoring.drift.quality import validate_schema_compatibility

        validate_schema_compatibility(
            current,
            current,
            policy=bad,
            baseline_schema_version="1.0.0",
            current_schema_versions={"1.0.0"},
        )


def test_local_artifacts_are_immutable_and_model_id_is_safely_encoded(tmp_path):
    store = LocalArtifactStore(tmp_path)
    key = "monitoring/model/baseline/drift/run/summary.json"
    store.put_immutable(key, b"one", "application/json")
    store.put_immutable(key, b"one", "application/json")
    with pytest.raises(ArtifactConflictError):
        store.put_immutable(key, b"two", "application/json")
    assert "%2F" in artifact_prefix(MODEL, "baseline", "run")


class FakeRepository:
    def __init__(self, current_policy, baseline, records):
        self.policy = current_policy
        self.baseline = baseline
        self.records = records
        self.runs = {}
        self.failed = []

    def resolve_policy(self, environment, model_version_id):
        return self.policy

    def establish_watermark(self, **kwargs):
        return ExtractionWatermark(
            extraction_cutoff=kwargs["extraction_cutoff"],
            maximum_persisted_event_id=max((row.event_id for row in self.records), default=0),
            maximum_eligible_prediction_timestamp=max(
                (row.prediction_timestamp for row in self.records), default=None
            ),
        )

    def count_predictions(self, **kwargs):
        return len(self.records)

    def resolve_baseline(self, *args):
        return self.baseline

    def reserve_run(self, metadata):
        run_id = metadata["monitoring_run_id"]
        if run_id in self.runs:
            return self.runs[run_id], False
        self.runs[run_id] = {**metadata, "status": RunStatus.RUNNING}
        return self.runs[run_id], True

    def extract_predictions(self, **kwargs):
        return tuple(self.records[: kwargs["maximum_rows"]])

    def mark_extraction_completed(self, run_id, completed_at):
        existing = self.runs[run_id].get("extraction_completed_at")
        self.runs[run_id]["extraction_completed_at"] = existing or completed_at
        return existing or completed_at

    def finish_run(self, run_id, **values):
        self.runs[run_id].update(values)

    def fail_run(self, run_id, **values):
        self.failed.append((run_id, values))


def _baseline(path: Path) -> BaselineVersion:
    body = path.read_bytes()
    return BaselineVersion(
        baseline_version_id="baseline-1",
        model_version_id=MODEL,
        reference_dataset_uri=path.as_uri(),
        reference_sha256=sha256_bytes(body),
        feature_schema_version="1.0.0",
        created_at=END - timedelta(days=10),
        active_from=END - timedelta(days=9),
        purpose="approved drift reference",
        approval_metadata={"approved_by": "test"},
    )


def test_insufficient_data_is_persisted_without_a_green_result(tmp_path):
    reference = tmp_path / "reference.parquet"
    pd.DataFrame().to_parquet(reference)
    repository = FakeRepository(policy(minimum_current_rows=2), _baseline(reference), [])
    result = MonitoringJob(
        repository,
        LocalArtifactStore(tmp_path / "reports"),
        now=lambda: END,
    ).run(environment="production", model_version_id=MODEL, scheduled_for=END)
    assert result["operational_status"] == RunStatus.INSUFFICIENT_DATA
    assert result["drift_status"] == ResultStatus.NOT_EVALUATED
    assert result["required_minimum"] == 2
    assert list((tmp_path / "reports").rglob("summary.json"))


def test_completed_run_publishes_bundle_and_rerun_returns_existing(tmp_path):
    rows = [record(1, END - timedelta(hours=2)), record(2, END - timedelta(hours=1))]
    reference = tmp_path / "reference.parquet"
    pd.DataFrame(
        [
            {
                **row.features,
                "prediction_probability": row.prediction_probability,
                "predicted_class": row.predicted_class,
            }
            for row in rows
        ]
    ).to_parquet(reference, index=False)
    repository = FakeRepository(
        policy(minimum_current_rows=2, minimum_reference_rows=2),
        _baseline(reference),
        rows,
    )
    calls = []

    def run_report(reference, current, *, policy):
        calls.append((len(reference), len(current), policy.policy_version))
        return EvidentlyOutput(
            html=b"<html>report</html>",
            report={"metrics": []},
            drift_summary={"status": ResultStatus.PASS, "feature_results": {}},
            version="test-evidently",
            configuration={"method": "test"},
        )

    job = MonitoringJob(
        repository,
        LocalArtifactStore(tmp_path / "reports"),
        report_runner=run_report,
        now=lambda: END,
    )
    first = job.run(environment="production", model_version_id=MODEL, scheduled_for=END)
    second = job.run(environment="production", model_version_id=MODEL, scheduled_for=END)
    assert first["operational_status"] == RunStatus.COMPLETED
    assert second["status"] == RunStatus.COMPLETED
    assert calls == [(2, 2, "1.0.0")]
    assert {path.name for path in (tmp_path / "reports").rglob("*") if path.is_file()} == {
        "report.html", "report.json", "summary.json", "checksums.json"
    }
    assert cadence_slot(END + timedelta(minutes=359), 360) == END
    assert schedule_minute_offset("15 */6 * * *") == 15
    assert cadence_slot(
        END + timedelta(minutes=15), 360, minute_offset=15
    ) == END + timedelta(minutes=15)
