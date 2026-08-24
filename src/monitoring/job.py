"""Idempotent orchestration for scheduled and manually-triggered monitoring."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Any, Callable
from urllib.parse import urlparse

import pandas as pd

from src.config import safe_error_message
from src.monitoring.artifacts import (
    ArtifactStore,
    artifact_prefix,
    publish_report_bundle,
    publish_summary_bundle,
)
from src.monitoring.evidently_runner import EvidentlyOutput, run_evidently
from src.monitoring.models import (
    MonitoringPolicy,
    ResultStatus,
    RunStatus,
    canonical_json_bytes,
    monitoring_run_id,
    require_utc,
    sha256_bytes,
    timestamp,
)
from src.monitoring.quality import (
    MonitoringValidationError,
    data_quality_summary,
    validate_schema_compatibility,
)
from src.monitoring.repository import MonitoringRepository, SELECTION_CRITERIA
from src.monitoring.selection import extraction_cutoff, select_window


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


class MonitoringJob:
    def __init__(
        self,
        repository: MonitoringRepository,
        artifact_store: ArtifactStore,
        *,
        evidently: Callable[..., EvidentlyOutput] = run_evidently,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self.repository = repository
        self.artifact_store = artifact_store
        self.evidently = evidently
        self.now = now

    def run(
        self,
        *,
        environment: str,
        model_version_id: str,
        scheduled_for: datetime | None = None,
    ) -> dict[str, Any]:
        extraction_started_at = require_utc(self.now(), "extraction_started_at")
        policy = self.repository.resolve_policy(environment, model_version_id)
        if not policy.permits(environment, model_version_id):
            raise ValueError("monitoring policy does not permit this environment/model")
        scheduled_for = cadence_slot(
            scheduled_for or extraction_started_at,
            policy.cadence_minutes,
            minute_offset=schedule_minute_offset(policy.schedule_cron),
        )
        if scheduled_for > extraction_started_at + timedelta(minutes=1):
            raise ValueError("scheduled_for must not be in the future")
        cutoff = extraction_cutoff(scheduled_for, policy)
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
        prefix = artifact_prefix(
            model_version_id, baseline.baseline_version_id, run_id
        )
        run_configuration = {
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
        metadata = {
            "monitoring_run_id": run_id,
            "logical_job_key": _logical_job_key(
                environment,
                model_version_id,
                policy.policy_version,
                scheduled_for,
            ),
            "job_type": JOB_TYPE,
            "environment": environment,
            "model_version_id": model_version_id,
            "baseline_version_id": baseline.baseline_version_id,
            "policy_version": policy.policy_version,
            "scheduled_for": scheduled_for,
            "extraction_started_at": extraction_started_at,
            "extraction_cutoff": watermark.extraction_cutoff,
            "maximum_persisted_event_id": watermark.maximum_persisted_event_id,
            "maximum_eligible_prediction_timestamp": watermark.maximum_eligible_prediction_timestamp,
            "window_start": window.start,
            "window_end": window.end,
            "observed_row_count": window.observed_rows,
            "selected_row_count": window.selected_rows,
            "selection_criteria": SELECTION_CRITERIA,
            "run_configuration": run_configuration,
            "artifact_prefix": prefix,
        }
        existing, owns_execution = self.repository.reserve_run(metadata)
        if not owns_execution:
            return dict(existing)
        # The originally reserved time is part of the immutable report. Reclaimed
        # failed/stale attempts reuse it instead of creating conflicting bytes.
        extraction_started_at = existing["extraction_started_at"]

        if window.observed_rows < policy.minimum_current_rows:
            try:
                next_schedule = scheduled_for + timedelta(minutes=policy.cadence_minutes)
                extraction_completed_at = self.repository.mark_extraction_completed(
                    run_id,
                    require_utc(self.now(), "extraction_completed_at"),
                )
                summary = {
                    "monitoring_run_id": run_id,
                    "report_identity": {
                        "model_version_id": model_version_id,
                        "baseline_version_id": baseline.baseline_version_id,
                        "policy_version": policy.policy_version,
                        "artifact_prefix": prefix,
                    },
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
                    "next_eligible_schedule_time": timestamp(next_schedule),
                    "extraction": {
                        "started_at": timestamp(extraction_started_at),
                        "completed_at": timestamp(extraction_completed_at),
                        **watermark.model_dump(mode="json"),
                    },
                    "configuration": run_configuration,
                    "message": "Minimum volume was not reached; drift was not evaluated.",
                }
                artifacts = publish_summary_bundle(
                    self.artifact_store, prefix=prefix, summary=summary
                )
                self.repository.finish_run(
                    run_id,
                    status=RunStatus.INSUFFICIENT_DATA,
                    summary=summary,
                    artifact_metadata=artifacts,
                    data_quality_status=ResultStatus.NOT_EVALUATED,
                    drift_status=ResultStatus.NOT_EVALUATED,
                    extraction_completed_at=extraction_completed_at,
                )
                return summary
            except Exception as exc:
                self.repository.fail_run(
                    run_id,
                    error_kind="monitoring_execution_failure",
                    error_details={
                        "type": type(exc).__name__,
                        "message": safe_error_message(exc),
                        "retryable": True,
                    },
                )
                raise

        try:
            records = self.repository.extract_predictions(
                environment=environment,
                model_version_id=model_version_id,
                window=window,
                watermark=watermark,
                maximum_rows=policy.maximum_current_rows,
            )
            if len(records) != window.selected_rows:
                raise MonitoringValidationError(
                    "selected row count changed under the persisted watermark"
                )
            extraction_completed_at = self.repository.mark_extraction_completed(
                run_id,
                require_utc(self.now(), "extraction_completed_at"),
            )
            reference_bytes = self.artifact_store.read_uri(
                baseline.reference_dataset_uri
            )
            if sha256_bytes(reference_bytes) != baseline.reference_sha256:
                raise MonitoringValidationError("reference checksum verification failed")
            reference = _read_reference(reference_bytes, baseline.reference_dataset_uri)
            if len(reference) < policy.minimum_reference_rows:
                raise MonitoringValidationError(
                    "reference dataset does not meet minimum_reference_rows"
                )
            current = _current_frame(records)
            validate_schema_compatibility(
                reference,
                current,
                policy=policy,
                baseline_schema_version=baseline.feature_schema_version,
                current_schema_versions={record.feature_schema_version for record in records},
            )
            for required in ("prediction_probability", "predicted_class"):
                if required not in reference.columns:
                    raise MonitoringValidationError(
                        f"reference data is missing drift column: {required}"
                    )
            reference_probability = pd.to_numeric(
                reference["prediction_probability"], errors="coerce"
            )
            if (
                reference_probability.isna().any()
                or not reference_probability.between(0, 1).all()
                or reference["predicted_class"].isna().any()
            ):
                raise MonitoringValidationError(
                    "reference prediction outputs are null, incompatible, or out of range"
                )
            quality = data_quality_summary(current, reference, policy=policy)
            evidently_output = self.evidently(
                reference, current, policy=policy
            )
            summary = {
                "monitoring_run_id": run_id,
                "report_identity": {
                    "model_version_id": model_version_id,
                    "baseline_version_id": baseline.baseline_version_id,
                    "policy_version": policy.policy_version,
                    "artifact_prefix": prefix,
                },
                "operational_status": RunStatus.COMPLETED,
                "data_quality_status": quality["status"],
                "drift_status": evidently_output.drift_summary["status"],
                "overall_drift_status": evidently_output.drift_summary["status"],
                "input_row_counts": {
                    "eligible": window.observed_rows,
                    "current": len(current),
                    "reference": len(reference),
                },
                "extraction": {
                    "started_at": timestamp(extraction_started_at),
                    "completed_at": timestamp(extraction_completed_at),
                    **watermark.model_dump(mode="json"),
                },
                "data_quality": quality,
                "drift": evidently_output.drift_summary,
                "evidently": {
                    "version": evidently_output.version,
                    "configuration": evidently_output.configuration,
                },
                "suppression_metadata": list(policy.suppression_rules),
                "exclusion_metadata": list(policy.exclusion_rules),
                "configuration": run_configuration,
            }
            artifacts = publish_report_bundle(
                self.artifact_store,
                prefix=prefix,
                html=evidently_output.html,
                report=evidently_output.report,
                summary=summary,
            )
            self.repository.finish_run(
                run_id,
                status=RunStatus.COMPLETED,
                summary=summary,
                artifact_metadata=artifacts,
                evidently_version=evidently_output.version,
                data_quality_status=str(quality["status"]),
                drift_status=str(evidently_output.drift_summary["status"]),
                extraction_completed_at=extraction_completed_at,
            )
            return summary
        except Exception as exc:
            error_kind = (
                "schema_or_extraction_failure"
                if isinstance(exc, MonitoringValidationError)
                else "monitoring_execution_failure"
            )
            self.repository.fail_run(
                run_id,
                error_kind=error_kind,
                error_details={
                    "type": type(exc).__name__,
                    "message": safe_error_message(exc),
                    "retryable": not isinstance(exc, MonitoringValidationError),
                },
            )
            raise
