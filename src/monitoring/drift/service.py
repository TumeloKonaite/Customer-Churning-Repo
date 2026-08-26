"""Idempotent orchestration for scheduled and manually-triggered monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Any, Callable
from urllib.parse import urlparse

import pandas as pd

from src.config import safe_error_message
from src.monitoring.shared.artifacts import (
    ArtifactStore,
    artifact_prefix,
    publish_report_bundle,
    publish_summary_bundle,
)
from src.monitoring.drift.evidently import EvidentlyOutput, run_drift_report
from src.monitoring.shared.models import (
    BaselineVersion,
    ExtractionWatermark,
    MonitoringPolicy,
    ResultStatus,
    RunStatus,
    SelectedWindow,
    canonical_json_bytes,
    monitoring_run_id,
    require_utc,
    sha256_bytes,
    timestamp,
)
from src.monitoring.drift.quality import (
    MonitoringValidationError,
    data_quality_summary,
    validate_schema_compatibility,
)
from src.monitoring.drift.repository import MonitoringRepository, SELECTION_CRITERIA
from src.monitoring.drift.selection import extraction_cutoff, select_window


JOB_TYPE = "data_quality_and_drift"


def cadence_slot(
    value: datetime, cadence_minutes: int, *, minute_offset: int = 0
) -> datetime:
    value = require_utc(value, "scheduled_for")
    minute = int(value.timestamp() // 60)
    slot = minute - (minute - minute_offset) % cadence_minutes
    return datetime.fromtimestamp(slot * 60, tz=timezone.utc)


def schedule_minute_offset(cron: str) -> int:
    first = cron.split(maxsplit=1)[0]
    return int(first) if first.isdigit() and 0 <= int(first) <= 59 else 0


def _logical_job_key(
    environment: str, model_version_id: str, policy_version: str, scheduled_for: datetime
) -> str:
    value = {
        "job_type": JOB_TYPE,
        "environment": environment,
        "model_version_id": model_version_id,
        "policy_version": policy_version,
        "scheduled_for": timestamp(scheduled_for),
    }
    return sha256_bytes(canonical_json_bytes(value))


def _current_frame(records) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                **record.features,
                "prediction_id": record.prediction_id,
                "prediction_probability": record.prediction_probability,
                "predicted_class": record.predicted_class,
            }
        )
    return pd.DataFrame(rows)


def _read_reference(body: bytes, uri: str) -> pd.DataFrame:
    path = urlparse(uri).path.casefold()
    try:
        if path.endswith((".parquet", ".pq")):
            return pd.read_parquet(BytesIO(body))
        if path.endswith(".csv"):
            return pd.read_csv(BytesIO(body))
    except Exception as exc:
        raise MonitoringValidationError("reference dataset could not be decoded") from exc
    raise MonitoringValidationError("reference dataset must use parquet or CSV format")


@dataclass(slots=True)
class DriftRun:
    """Immutable inputs selected before a run is reserved."""

    environment: str
    model_version_id: str
    scheduled_for: datetime
    extraction_started_at: datetime
    policy: MonitoringPolicy
    watermark: ExtractionWatermark
    window: SelectedWindow
    baseline: BaselineVersion
    run_id: str
    prefix: str
    configuration: dict[str, Any]

    @property
    def report_identity(self) -> dict[str, str]:
        return {
            "model_version_id": self.model_version_id,
            "baseline_version_id": self.baseline.baseline_version_id,
            "policy_version": self.policy.policy_version,
            "artifact_prefix": self.prefix,
        }


class MonitoringJob:
    """Production wrapper around the small ``run_drift_report`` core.

    The wrapper selects reproducible data, validates it, runs the report, and then
    commits immutable artifacts and status. Evidently itself is only invoked by
    ``_evaluate_and_publish``.
    """

    def __init__(
        self,
        repository: MonitoringRepository,
        artifact_store: ArtifactStore,
        *,
        report_runner: Callable[..., EvidentlyOutput] = run_drift_report,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self.repository = repository
        self.artifact_store = artifact_store
        self.report_runner = report_runner
        self.now = now

    def run(
        self,
        *,
        environment: str,
        model_version_id: str,
        scheduled_for: datetime | None = None,
    ) -> dict[str, Any]:
        """Plan, reserve, evaluate, and persist one idempotent monitoring run."""
        run = self._prepare_run(
            environment=environment,
            model_version_id=model_version_id,
            scheduled_for=scheduled_for,
        )
        existing, owns_execution = self.repository.reserve_run(self._metadata(run))
        if not owns_execution:
            return dict(existing)
        run.extraction_started_at = existing["extraction_started_at"]
        try:
            if run.window.observed_rows < run.policy.minimum_current_rows:
                return self._complete_insufficient_run(run)
            return self._evaluate_and_publish(run)
        except Exception as exc:
            self._record_failure(run.run_id, exc)
            raise

    def _prepare_run(
        self,
        *,
        environment: str,
        model_version_id: str,
        scheduled_for: datetime | None,
    ) -> DriftRun:
        started = require_utc(self.now(), "extraction_started_at")
        policy = self.repository.resolve_policy(environment, model_version_id)
        if not policy.permits(environment, model_version_id):
            raise ValueError("monitoring policy does not permit this environment/model")
        scheduled = cadence_slot(
            scheduled_for or started,
            policy.cadence_minutes,
            minute_offset=schedule_minute_offset(policy.schedule_cron),
        )
        if scheduled > started + timedelta(minutes=1):
            raise ValueError("scheduled_for must not be in the future")
        cutoff = extraction_cutoff(scheduled, policy)
        watermark = self.repository.establish_watermark(
            environment=environment,
            model_version_id=model_version_id,
            extraction_cutoff=cutoff,
        )
        window = select_window(
            end=cutoff,
            watermark=watermark,
            policy=policy,
            count_rows=lambda start, end, mark: self.repository.count_predictions(
                environment=environment,
                model_version_id=model_version_id,
                start=start,
                end=end,
                watermark=mark,
            ),
        )
        baseline = self.repository.resolve_baseline(
            model_version_id, window.start, window.end
        )
        run_id = monitoring_run_id(
            job_type=JOB_TYPE,
            environment=environment,
            model_version_id=model_version_id,
            baseline_version_id=baseline.baseline_version_id,
            policy_version=policy.policy_version,
            window=window,
            watermark=watermark,
        )
        prefix = artifact_prefix(model_version_id, baseline.baseline_version_id, run_id)
        configuration = {
            "job_type": JOB_TYPE,
            "environment": environment,
            "model_version_id": model_version_id,
            "baseline": baseline.model_dump(mode="json"),
            "policy": policy.model_dump(mode="json"),
            "policy_config_sha256": policy.config_sha256,
            "window": window.model_dump(mode="json"),
            "watermark": watermark.model_dump(mode="json"),
            "selection_criteria": SELECTION_CRITERIA,
        }
        return DriftRun(
            environment=environment,
            model_version_id=model_version_id,
            scheduled_for=scheduled,
            extraction_started_at=started,
            policy=policy,
            watermark=watermark,
            window=window,
            baseline=baseline,
            run_id=run_id,
            prefix=prefix,
            configuration=configuration,
        )

    @staticmethod
    def _metadata(run: DriftRun) -> dict[str, Any]:
        return {
            "monitoring_run_id": run.run_id,
            "logical_job_key": _logical_job_key(
                run.environment,
                run.model_version_id,
                run.policy.policy_version,
                run.scheduled_for,
            ),
            "job_type": JOB_TYPE,
            "environment": run.environment,
            "model_version_id": run.model_version_id,
            "baseline_version_id": run.baseline.baseline_version_id,
            "policy_version": run.policy.policy_version,
            "scheduled_for": run.scheduled_for,
            "extraction_started_at": run.extraction_started_at,
            "extraction_cutoff": run.watermark.extraction_cutoff,
            "maximum_persisted_event_id": run.watermark.maximum_persisted_event_id,
            "maximum_eligible_prediction_timestamp": run.watermark.maximum_eligible_prediction_timestamp,
            "window_start": run.window.start,
            "window_end": run.window.end,
            "observed_row_count": run.window.observed_rows,
            "selected_row_count": run.window.selected_rows,
            "selection_criteria": SELECTION_CRITERIA,
            "run_configuration": run.configuration,
            "artifact_prefix": run.prefix,
        }

    def _complete_insufficient_run(self, run: DriftRun) -> dict[str, Any]:
        completed = self._mark_extraction_completed(run.run_id)
        policy, window = run.policy, run.window
        summary = {
            "monitoring_run_id": run.run_id,
            "report_identity": run.report_identity,
            "operational_status": RunStatus.INSUFFICIENT_DATA,
            "data_quality_status": ResultStatus.NOT_EVALUATED,
            "drift_status": ResultStatus.NOT_EVALUATED,
            "observed_row_count": window.observed_rows,
            "required_minimum": policy.minimum_current_rows,
            "attempted_window": {
                "start": timestamp(window.start),
                "end": timestamp(window.end),
            },
            "maximum_permitted_lookback_hours": policy.maximum_lookback_hours,
            "maximum_permitted_window": {
                "start": timestamp(
                    max(
                        window.end - policy.maximum_lookback,
                        policy.fixed_historical_boundary
                        or window.end - policy.maximum_lookback,
                    )
                ),
                "end": timestamp(window.end),
            },
            "fixed_historical_boundary": (
                timestamp(policy.fixed_historical_boundary)
                if policy.fixed_historical_boundary
                else None
            ),
            "next_eligible_schedule_time": timestamp(
                run.scheduled_for + timedelta(minutes=policy.cadence_minutes)
            ),
            "extraction": self._extraction_summary(run, completed),
            "configuration": run.configuration,
            "message": "Minimum volume was not reached; drift was not evaluated.",
        }
        artifacts = publish_summary_bundle(
            self.artifact_store, prefix=run.prefix, summary=summary
        )
        self.repository.finish_run(
            run.run_id,
            status=RunStatus.INSUFFICIENT_DATA,
            summary=summary,
            artifact_metadata=artifacts,
            data_quality_status=ResultStatus.NOT_EVALUATED,
            drift_status=ResultStatus.NOT_EVALUATED,
            extraction_completed_at=completed,
        )
        return summary

    def _evaluate_and_publish(self, run: DriftRun) -> dict[str, Any]:
        records = self._extract_current(run)
        completed = self._mark_extraction_completed(run.run_id)
        reference = self._load_reference(run)
        current = _current_frame(records)
        self._validate_inputs(run, records, reference, current)
        quality = data_quality_summary(current, reference, policy=run.policy)
        report = self.report_runner(reference, current, policy=run.policy)
        return self._publish_completed_run(
            run,
            completed=completed,
            reference=reference,
            current=current,
            quality=quality,
            report=report,
        )

    def _publish_completed_run(
        self,
        run: DriftRun,
        *,
        completed: datetime,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        quality: dict[str, Any],
        report: EvidentlyOutput,
    ) -> dict[str, Any]:
        summary = {
            "monitoring_run_id": run.run_id,
            "report_identity": run.report_identity,
            "operational_status": RunStatus.COMPLETED,
            "data_quality_status": quality["status"],
            "drift_status": report.drift_summary["status"],
            "overall_drift_status": report.drift_summary["status"],
            "input_row_counts": {
                "eligible": run.window.observed_rows,
                "current": len(current),
                "reference": len(reference),
            },
            "extraction": self._extraction_summary(run, completed),
            "data_quality": quality,
            "drift": report.drift_summary,
            "evidently": {
                "version": report.version,
                "configuration": report.configuration,
            },
            "suppression_metadata": list(run.policy.suppression_rules),
            "exclusion_metadata": list(run.policy.exclusion_rules),
            "configuration": run.configuration,
        }
        artifacts = publish_report_bundle(
            self.artifact_store,
            prefix=run.prefix,
            html=report.html,
            report=report.report,
            summary=summary,
        )
        self.repository.finish_run(
            run.run_id,
            status=RunStatus.COMPLETED,
            summary=summary,
            artifact_metadata=artifacts,
            evidently_version=report.version,
            data_quality_status=str(quality["status"]),
            drift_status=str(report.drift_summary["status"]),
            extraction_completed_at=completed,
        )
        return summary

    def _extract_current(self, run: DriftRun):
        records = self.repository.extract_predictions(
            environment=run.environment,
            model_version_id=run.model_version_id,
            window=run.window,
            watermark=run.watermark,
            maximum_rows=run.policy.maximum_current_rows,
        )
        if len(records) != run.window.selected_rows:
            raise MonitoringValidationError(
                "selected row count changed under the persisted watermark"
            )
        return records

    def _load_reference(self, run: DriftRun) -> pd.DataFrame:
        body = self.artifact_store.read_uri(run.baseline.reference_dataset_uri)
        if sha256_bytes(body) != run.baseline.reference_sha256:
            raise MonitoringValidationError("reference checksum verification failed")
        reference = _read_reference(body, run.baseline.reference_dataset_uri)
        if len(reference) < run.policy.minimum_reference_rows:
            raise MonitoringValidationError(
                "reference dataset does not meet minimum_reference_rows"
            )
        return reference

    @staticmethod
    def _validate_inputs(run, records, reference, current) -> None:
        validate_schema_compatibility(
            reference,
            current,
            policy=run.policy,
            baseline_schema_version=run.baseline.feature_schema_version,
            current_schema_versions={record.feature_schema_version for record in records},
        )
        for required in ("prediction_probability", "predicted_class"):
            if required not in reference.columns:
                raise MonitoringValidationError(
                    f"reference data is missing drift column: {required}"
                )
        probability = pd.to_numeric(reference["prediction_probability"], errors="coerce")
        if (
            probability.isna().any()
            or not probability.between(0, 1).all()
            or reference["predicted_class"].isna().any()
        ):
            raise MonitoringValidationError(
                "reference prediction outputs are null, incompatible, or out of range"
            )

    def _mark_extraction_completed(self, run_id: str) -> datetime:
        return self.repository.mark_extraction_completed(
            run_id, require_utc(self.now(), "extraction_completed_at")
        )

    @staticmethod
    def _extraction_summary(run: DriftRun, completed: datetime) -> dict[str, Any]:
        return {
            "started_at": timestamp(run.extraction_started_at),
            "completed_at": timestamp(completed),
            **run.watermark.model_dump(mode="json"),
        }

    def _record_failure(self, run_id: str, exc: Exception) -> None:
        validation_failure = isinstance(exc, MonitoringValidationError)
        self.repository.fail_run(
            run_id,
            error_kind=(
                "schema_or_extraction_failure"
                if validation_failure
                else "monitoring_execution_failure"
            ),
            error_details={
                "type": type(exc).__name__,
                "message": safe_error_message(exc),
                "retryable": not validation_failure,
            },
        )
