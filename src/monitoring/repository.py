"""Neon persistence and deterministic prediction extraction queries."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import Engine, text

from src.monitoring.models import (
    BaselineVersion,
    ExtractionWatermark,
    MonitoringPolicy,
    PredictionRecord,
    RunStatus,
    SelectedWindow,
    timestamp,
)


SELECTION_CRITERIA = {
    "interval": "window_start <= prediction_timestamp <= window_end",
    "environment": "environment = requested environment",
    "model": "model_version_id = requested exact model version",
    "arrival_watermark": "persisted_at <= extraction_cutoff AND event_id <= maximum_persisted_event_id",
    "limit": "newest rows by prediction_timestamp DESC, event_id DESC",
    "output_order": "prediction_timestamp ASC, event_id ASC",
}


class MonitoringRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    def resolve_policy(self, environment: str, model_version_id: str) -> MonitoringPolicy:
        statement = text(
            """
            SELECT configuration
            FROM monitoring_policies
            WHERE enabled = TRUE
              AND :environment = ANY(included_environments)
              AND ('*' = ANY(included_model_versions)
                   OR :model_version_id = ANY(included_model_versions))
            ORDER BY created_at DESC, policy_version DESC
            LIMIT 1
            """
        )
        with self.engine.connect() as connection:
            value = connection.execute(
                statement,
                {"environment": environment, "model_version_id": model_version_id},
            ).scalar_one_or_none()
        if value is None:
            raise LookupError("no enabled monitoring policy includes this environment/model")
        return MonitoringPolicy.model_validate(value)

    def register_policy(self, policy: MonitoringPolicy) -> None:
        """Idempotently register an immutable policy version."""
        import json

        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO monitoring_policies (
                        policy_version, enabled, configuration, configuration_sha256,
                        included_environments, included_model_versions
                    ) VALUES (
                        :policy_version, :enabled, CAST(:configuration AS jsonb),
                        :configuration_sha256, :included_environments,
                        :included_model_versions
                    )
                    ON CONFLICT (policy_version) DO NOTHING
                    """
                ),
                {
                    "policy_version": policy.policy_version,
                    "enabled": policy.enabled,
                    "configuration": json.dumps(policy.model_dump(mode="json")),
                    "configuration_sha256": policy.config_sha256,
                    "included_environments": list(policy.included_environments),
                    "included_model_versions": list(policy.included_model_versions),
                },
            )
            existing = connection.execute(
                text(
                    "SELECT configuration_sha256 FROM monitoring_policies "
                    "WHERE policy_version = :policy_version"
                ),
                {"policy_version": policy.policy_version},
            ).scalar_one()
            if existing != policy.config_sha256:
                raise ValueError(
                    "policy version already exists with different result-affecting content"
                )

    def register_baseline(self, baseline: BaselineVersion) -> None:
        """Idempotently register an immutable model-specific baseline."""
        import json

        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO monitoring_baselines (
                        baseline_version_id, model_version_id, reference_dataset_uri,
                        reference_sha256, feature_schema_version, created_at,
                        active_from, retired_at, purpose, approval_metadata
                    ) VALUES (
                        :baseline_version_id, :model_version_id, :reference_dataset_uri,
                        :reference_sha256, :feature_schema_version, :created_at,
                        :active_from, :retired_at, :purpose,
                        CAST(:approval_metadata AS jsonb)
                    )
                    ON CONFLICT (baseline_version_id) DO NOTHING
                    """
                ),
                {
                    **baseline.model_dump(),
                    "approval_metadata": json.dumps(baseline.approval_metadata),
                },
            )
            existing = connection.execute(
                text(
                    """
                    SELECT model_version_id, reference_dataset_uri, reference_sha256,
                           feature_schema_version
                    FROM monitoring_baselines
                    WHERE baseline_version_id = :baseline_version_id
                    """
                ),
                {"baseline_version_id": baseline.baseline_version_id},
            ).mappings().one()
            expected = {
                "model_version_id": baseline.model_version_id,
                "reference_dataset_uri": baseline.reference_dataset_uri,
                "reference_sha256": baseline.reference_sha256,
                "feature_schema_version": baseline.feature_schema_version,
            }
            if dict(existing) != expected:
                raise ValueError(
                    "baseline version already exists with a different immutable identity"
                )

    def resolve_baseline(
        self, model_version_id: str, interval_start: datetime, interval_end: datetime
    ) -> BaselineVersion:
        statement = text(
            """
            SELECT baseline_version_id, model_version_id, reference_dataset_uri,
                   reference_sha256, feature_schema_version, created_at, active_from,
                   retired_at, purpose, approval_metadata
            FROM monitoring_baselines
            WHERE model_version_id = :model_version_id
              AND active_from <= :interval_end
              AND (retired_at IS NULL OR retired_at > :interval_end)
            ORDER BY active_from DESC, baseline_version_id DESC
            LIMIT 1
            """
        )
        with self.engine.connect() as connection:
            row = connection.execute(
                statement,
                {
                    "model_version_id": model_version_id,
                    "interval_start": interval_start,
                    "interval_end": interval_end,
                },
            ).mappings().one_or_none()
        if row is None:
            raise LookupError("no applicable baseline exists for the exact model version")
        return BaselineVersion.model_validate(dict(row))

    def establish_watermark(
        self,
        *,
        environment: str,
        model_version_id: str,
        extraction_cutoff: datetime,
    ) -> ExtractionWatermark:
        statement = text(
            """
            WITH cursor AS (
                SELECT COALESCE(MAX(event_id), 0) AS maximum_persisted_event_id
                FROM prediction_events
                WHERE persisted_at <= :extraction_cutoff
            )
            SELECT cursor.maximum_persisted_event_id,
                   MAX(prediction_timestamp) AS maximum_eligible_prediction_timestamp
            FROM cursor
            LEFT JOIN prediction_events p
              ON p.event_id <= cursor.maximum_persisted_event_id
             AND p.persisted_at <= :extraction_cutoff
             AND p.environment = :environment
             AND p.model_version_id = :model_version_id
             AND p.prediction_timestamp <= :extraction_cutoff
            GROUP BY cursor.maximum_persisted_event_id
            """
        )
        with self.engine.connect() as connection:
            row = connection.execute(
                statement,
                {
                    "environment": environment,
                    "model_version_id": model_version_id,
                    "extraction_cutoff": extraction_cutoff,
                },
            ).mappings().one()
        return ExtractionWatermark(
            extraction_cutoff=extraction_cutoff,
            maximum_persisted_event_id=row["maximum_persisted_event_id"],
            maximum_eligible_prediction_timestamp=row[
                "maximum_eligible_prediction_timestamp"
            ],
        )

    def count_predictions(
        self,
        *,
        environment: str,
        model_version_id: str,
        start: datetime,
        end: datetime,
        watermark: ExtractionWatermark,
    ) -> int:
        statement = text(
            """
            SELECT COUNT(*)
            FROM prediction_events
            WHERE environment = :environment
              AND model_version_id = :model_version_id
              AND prediction_timestamp >= :window_start
              AND prediction_timestamp <= :window_end
              AND persisted_at <= :extraction_cutoff
              AND event_id <= :maximum_persisted_event_id
            """
        )
        with self.engine.connect() as connection:
            return int(
                connection.execute(
                    statement,
                    {
                        "environment": environment,
                        "model_version_id": model_version_id,
                        "window_start": start,
                        "window_end": end,
                        "extraction_cutoff": watermark.extraction_cutoff,
                        "maximum_persisted_event_id": watermark.maximum_persisted_event_id,
                    },
                ).scalar_one()
            )

    def extract_predictions(
        self,
        *,
        environment: str,
        model_version_id: str,
        window: SelectedWindow,
        watermark: ExtractionWatermark,
        maximum_rows: int,
    ) -> tuple[PredictionRecord, ...]:
        statement = text(
            """
            SELECT event_id, prediction_id, environment, model_version_id,
                   prediction_timestamp, persisted_at, feature_schema_version,
                   features, prediction_probability, predicted_class
            FROM (
                SELECT event_id, prediction_id, environment, model_version_id,
                       prediction_timestamp, persisted_at, feature_schema_version,
                       features, prediction_probability, predicted_class
                FROM prediction_events
                WHERE environment = :environment
                  AND model_version_id = :model_version_id
                  AND prediction_timestamp >= :window_start
                  AND prediction_timestamp <= :window_end
                  AND persisted_at <= :extraction_cutoff
                  AND event_id <= :maximum_persisted_event_id
                ORDER BY prediction_timestamp DESC, event_id DESC
                LIMIT :maximum_rows
            ) selected
            ORDER BY prediction_timestamp ASC, event_id ASC
            """
        )
        with self.engine.connect() as connection:
            rows = connection.execute(
                statement,
                {
                    "environment": environment,
                    "model_version_id": model_version_id,
                    "window_start": window.start,
                    "window_end": window.end,
                    "extraction_cutoff": watermark.extraction_cutoff,
                    "maximum_persisted_event_id": watermark.maximum_persisted_event_id,
                    "maximum_rows": maximum_rows,
                },
            ).mappings().all()
        return tuple(PredictionRecord.model_validate(dict(row)) for row in rows)

    def reserve_run(self, metadata: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Insert once under a transaction-level advisory lock; failed runs may retry."""
        statement = text(
            """
            INSERT INTO monitoring_runs (
                monitoring_run_id, logical_job_key, job_type, environment,
                model_version_id, baseline_version_id, policy_version, status,
                scheduled_for, extraction_started_at, extraction_cutoff,
                maximum_persisted_event_id, maximum_eligible_prediction_timestamp,
                window_start, window_end, observed_row_count, selected_row_count,
                selection_criteria, run_configuration, artifact_prefix
            ) VALUES (
                :monitoring_run_id, :logical_job_key, :job_type, :environment,
                :model_version_id, :baseline_version_id, :policy_version, 'running',
                :scheduled_for, :extraction_started_at, :extraction_cutoff,
                :maximum_persisted_event_id, :maximum_eligible_prediction_timestamp,
                :window_start, :window_end, :observed_row_count, :selected_row_count,
                CAST(:selection_criteria AS jsonb), CAST(:run_configuration AS jsonb),
                :artifact_prefix
            )
            ON CONFLICT (monitoring_run_id) DO NOTHING
            """
        )
        import json

        values = dict(metadata)
        values["selection_criteria"] = json.dumps(values["selection_criteria"])
        values["run_configuration"] = json.dumps(values["run_configuration"])
        with self.engine.begin() as connection:
            connection.execute(
                text("SELECT pg_advisory_xact_lock(hashtext(:run_id))"),
                {"run_id": metadata["monitoring_run_id"]},
            )
            inserted = connection.execute(statement, values).rowcount == 1
            existing = connection.execute(
                text(
                    "SELECT * FROM monitoring_runs WHERE monitoring_run_id = :run_id"
                ),
                {"run_id": metadata["monitoring_run_id"]},
            ).mappings().one()
            stale_running = (
                existing["status"] == RunStatus.RUNNING
                and existing["updated_at"]
                < datetime.now(timezone.utc) - timedelta(minutes=35)
            )
            if not inserted and (
                existing["status"] == RunStatus.FAILED or stale_running
            ):
                connection.execute(
                    text(
                        """
                        UPDATE monitoring_runs
                        SET status = 'running', error_kind = NULL, error_details = NULL,
                            updated_at = now()
                        WHERE monitoring_run_id = :run_id
                        """
                    ),
                    {
                        "run_id": metadata["monitoring_run_id"],
                    },
                )
                existing = {**dict(existing), "status": RunStatus.RUNNING}
                inserted = True
        return dict(existing), inserted

    def mark_extraction_completed(self, run_id: str, completed_at: datetime) -> datetime:
        """Persist the first extraction completion time and reuse it on every retry."""
        with self.engine.begin() as connection:
            return connection.execute(
                text(
                    """
                    UPDATE monitoring_runs
                    SET extraction_completed_at = COALESCE(
                            extraction_completed_at, :completed_at
                        ),
                        updated_at = now()
                    WHERE monitoring_run_id = :run_id
                    RETURNING extraction_completed_at
                    """
                ),
                {"run_id": run_id, "completed_at": completed_at},
            ).scalar_one()

    def finish_run(
        self,
        run_id: str,
        *,
        status: RunStatus,
        summary: dict[str, Any],
        artifact_metadata: dict[str, Any] | None = None,
        evidently_version: str | None = None,
        data_quality_status: str | None = None,
        drift_status: str | None = None,
        extraction_completed_at: datetime | None = None,
    ) -> None:
        import json

        artifact_metadata = artifact_metadata or {"uris": {}, "checksums": {}}
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    UPDATE monitoring_runs
                    SET status = :status, summary = CAST(:summary AS jsonb),
                        artifact_uris = CAST(:artifact_uris AS jsonb),
                        artifact_checksums = CAST(:artifact_checksums AS jsonb),
                        evidently_version = :evidently_version,
                        data_quality_status = :data_quality_status,
                        drift_status = :drift_status,
                        extraction_completed_at = COALESCE(
                            extraction_completed_at, :extraction_completed_at
                        ), updated_at = now()
                    WHERE monitoring_run_id = :run_id
                    """
                ),
                {
                    "run_id": run_id,
                    "status": status,
                    "summary": json.dumps(summary),
                    "artifact_uris": json.dumps(artifact_metadata["uris"]),
                    "artifact_checksums": json.dumps(artifact_metadata["checksums"]),
                    "evidently_version": evidently_version,
                    "data_quality_status": data_quality_status,
                    "drift_status": drift_status,
                    "extraction_completed_at": extraction_completed_at
                    or datetime.now(timezone.utc),
                },
            )

    def fail_run(self, run_id: str, *, error_kind: str, error_details: dict[str, Any]) -> None:
        import json

        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    UPDATE monitoring_runs
                    SET status = 'failed', error_kind = :error_kind,
                        error_details = CAST(:error_details AS jsonb),
                        extraction_completed_at = COALESCE(
                            extraction_completed_at, now()
                        ), updated_at = now()
                    WHERE monitoring_run_id = :run_id
                    """
                ),
                {
                    "run_id": run_id,
                    "error_kind": error_kind,
                    "error_details": json.dumps(error_details),
                },
            )
