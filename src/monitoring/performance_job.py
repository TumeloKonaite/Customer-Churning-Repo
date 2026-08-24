"""Performance-cohort extraction, immutable publication, and Neon run metadata."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from typing import Any, Callable, Mapping, Protocol

from sqlalchemy import Engine, text

from src.monitoring.artifacts import (
    ArtifactStore,
    performance_artifact_prefix,
    publish_report_bundle,
)
from src.monitoring.models import require_utc
from src.monitoring.performance import (
    CohortDefinition,
    PerformanceRecord,
    build_performance_report,
    performance_report_html,
    performance_report_configuration,
    performance_run_id,
)


class PerformanceStore(Protocol):
    def label_revision_watermark(self, *, as_of: datetime) -> int: ...
    def load_performance_records(self, *, definition: CohortDefinition, evaluated_at: datetime, simulation_generator: str | None, simulation_scenario_version: str | None) -> tuple[PerformanceRecord, ...]: ...
    def reserve_performance_run(self, metadata: dict[str, Any]) -> tuple[dict[str, Any], bool]: ...
    def finish_performance_run(self, run_id: str, *, summary: dict[str, Any], artifacts: dict[str, Any]) -> None: ...
    def fail_performance_run(self, run_id: str, error: dict[str, Any]) -> None: ...


@dataclass(slots=True)
class PerformanceJob:
    store: PerformanceStore
    artifact_store: ArtifactStore
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)

    def run(
        self,
        *,
        cohort_start: datetime,
        cohort_end: datetime,
        horizon_days: int,
        grace_period_days: int,
        outcome_watermark: dict[str, Any],
        label_contract_version: str,
        model_version_id: str,
        deployment_ids: tuple[str, ...],
        policy_version: str,
        classification_threshold: float,
        is_simulated: bool = False,
        simulation_generator: str | None = None,
        simulation_scenario_version: str | None = None,
        approved_segments: Mapping[str, str] | None = None,
        minimum_privacy_size: int = 20,
        calibration_options: Mapping[str, Any] | None = None,
        analysis_thresholds: tuple[float, ...] | None = None,
        label_revision_watermark: int | None = None,
        evaluated_at: datetime | None = None,
    ) -> dict[str, Any]:
        execution_started_at = require_utc(self.now(), "execution_started_at")
        evaluated_at = require_utc(
            evaluated_at or execution_started_at, "evaluated_at"
        )
        if is_simulated and (not simulation_generator or not simulation_scenario_version):
            raise ValueError("simulated reports require generator and scenario version")
        if not is_simulated and (
            simulation_generator is not None or simulation_scenario_version is not None
        ):
            raise ValueError("real reports cannot specify simulation metadata")
        label_watermark = (
            self.store.label_revision_watermark(as_of=evaluated_at)
            if label_revision_watermark is None
            else label_revision_watermark
        )
        if label_watermark < 0:
            raise ValueError("label_revision_watermark must not be negative")
        definition = CohortDefinition(
            cohort_start=cohort_start,
            cohort_end=cohort_end,
            horizon_days=horizon_days,
            grace_period_days=grace_period_days,
            outcome_watermark=outcome_watermark,
            label_revision_watermark=label_watermark,
            label_contract_version=label_contract_version,
            model_version_id=model_version_id,
            deployment_ids=deployment_ids,
            policy_version=policy_version,
            is_simulated=is_simulated,
            simulation_generator=simulation_generator,
            simulation_scenario_version=simulation_scenario_version,
            classification_threshold=classification_threshold,
        )
        report_configuration = performance_report_configuration(
            approved_segments=approved_segments,
            minimum_privacy_size=minimum_privacy_size,
            calibration_options=calibration_options,
            analysis_thresholds=analysis_thresholds,
        )
        run_id = performance_run_id(definition, report_configuration)
        prefix = performance_artifact_prefix(model_version_id, run_id)
        metadata = {
            "monitoring_run_id": run_id,
            "model_version_id": model_version_id,
            "deployment_ids": list(deployment_ids),
            "policy_version": policy_version,
            "label_contract_version": label_contract_version,
            "cohort_start": cohort_start,
            "cohort_end": cohort_end,
            "cohort_selection_rule": definition.selection_rule,
            "outcome_watermark": outcome_watermark,
            "label_revision_watermark": label_watermark,
            "is_simulated": is_simulated,
            "simulation_generator": simulation_generator,
            "simulation_scenario_version": simulation_scenario_version,
            "artifact_prefix": prefix,
            "run_configuration": {
                "cohort": definition.model_dump(mode="json"),
                "report": report_configuration,
            },
            "started_at": execution_started_at,
        }
        existing, owns_execution = self.store.reserve_performance_run(metadata)
        if not owns_execution:
            return existing.get("summary") or existing
        try:
            records = self.store.load_performance_records(
                definition=definition,
                evaluated_at=evaluated_at,
                simulation_generator=simulation_generator,
                simulation_scenario_version=simulation_scenario_version,
            )
            report = build_performance_report(
                records,
                definition=definition,
                evaluated_at=evaluated_at,
                approved_segments=approved_segments,
                minimum_privacy_size=minimum_privacy_size,
                calibration_options=calibration_options,
                analysis_thresholds=analysis_thresholds,
            )
            exclusion_counter = dict(report["cohort"]["excluded_prediction_counts"])
            exclusion_method = getattr(self.store, "performance_exclusion_counts", None)
            if callable(exclusion_method):
                for reason, count in exclusion_method(
                    definition=definition, evaluated_at=evaluated_at
                ).items():
                    exclusion_counter[reason] = exclusion_counter.get(reason, 0) + count
                report["cohort"]["excluded_prediction_counts"] = dict(
                    sorted(exclusion_counter.items())
                )
            summary = {
                "monitoring_run_id": run_id,
                "status": "completed",
                "official": report["official"],
                "display_label": report["display_label"],
                "eligible_prediction_count": report["cohort"]["eligible_prediction_count"],
                "excluded_prediction_counts": report["cohort"]["excluded_prediction_counts"],
                "classification": report["classification"],
                "suppression_metadata": {
                    field: value["suppression"]
                    for field, value in report["segments"].items()
                },
            }
            artifacts = publish_report_bundle(
                self.artifact_store,
                prefix=prefix,
                html=performance_report_html(report),
                report=report,
                summary=summary,
            )
            self.store.finish_performance_run(
                run_id, summary=summary, artifacts=artifacts
            )
            return {**summary, "artifacts": artifacts}
        except Exception as exc:
            self.store.fail_performance_run(
                run_id,
                {"type": type(exc).__name__, "message": "performance reporting failed"},
            )
            raise


class PerformanceRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    def label_revision_watermark(self, *, as_of: datetime) -> int:
        with self.engine.connect() as connection:
            return int(
                connection.execute(
                    text(
                        """
                        SELECT COALESCE(MAX(label_revision_id), 0)
                        FROM prediction_label_revisions
                        WHERE created_at <= :as_of
                        """
                    ),
                    {"as_of": as_of},
                ).scalar_one()
            )

    def load_performance_records(
        self,
        *,
        definition: CohortDefinition,
        evaluated_at: datetime,
        simulation_generator: str | None,
        simulation_scenario_version: str | None,
    ) -> tuple[PerformanceRecord, ...]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    WITH latest_labels AS (
                        SELECT DISTINCT ON (prediction_id)
                               label_revision_id, prediction_id, label_value,
                               attribution_timestamp, qualifying_outcome_event_id
                        FROM prediction_label_revisions
                        WHERE label_revision_id <= :label_revision_watermark
                          AND label_contract_version = :label_contract_version
                          AND is_simulated = :is_simulated
                          AND simulation_scope = :simulation_scope
                        ORDER BY prediction_id, revision_number DESC,
                                 label_revision_id DESC
                    )
                    SELECT p.prediction_id, p.prediction_timestamp, p.horizon_end,
                           p.model_version_id, p.deployment_id, p.policy_version,
                           p.label_contract_version, p.prediction_probability,
                           p.predicted_class, labels.label_value,
                           labels.label_revision_id,
                           labels.attribution_timestamp AS label_attribution_timestamp,
                           :is_simulated AS is_simulated,
                           CASE WHEN :is_simulated THEN :simulation_generator END
                               AS simulation_generator,
                           CASE WHEN :is_simulated THEN :simulation_scenario_version END
                               AS simulation_scenario_version,
                           p.segments
                    FROM prediction_events p
                    JOIN latest_labels labels USING (prediction_id)
                    WHERE p.prediction_timestamp >= :cohort_start
                      AND p.prediction_timestamp < :cohort_end
                      AND p.model_version_id = :model_version_id
                      AND p.policy_version = :policy_version
                      AND p.monitoring_eligible
                      AND p.horizon_end IS NOT NULL
                      AND p.deployment_id IS NOT NULL
                      AND labels.label_value IN (0, 1)
                    ORDER BY p.prediction_timestamp, p.prediction_id
                    """
                ),
                {
                    "label_revision_watermark": definition.label_revision_watermark,
                    "label_contract_version": definition.label_contract_version,
                    "is_simulated": definition.is_simulated,
                    "simulation_scope": _simulation_scope(definition),
                    "simulation_generator": simulation_generator,
                    "simulation_scenario_version": simulation_scenario_version,
                    "cohort_start": definition.cohort_start,
                    "cohort_end": definition.cohort_end,
                    "model_version_id": definition.model_version_id,
                    "policy_version": definition.policy_version,
                    "grace_period_days": definition.grace_period_days,
                    "evaluated_at": evaluated_at,
                },
            ).mappings().all()
        return tuple(PerformanceRecord.model_validate(dict(row)) for row in rows)

    def performance_exclusion_counts(
        self, *, definition: CohortDefinition, evaluated_at: datetime
    ) -> dict[str, int]:
        """Count cohort rows that cannot be represented as binary evaluation records."""
        with self.engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    WITH latest_labels AS (
                        SELECT DISTINCT ON (prediction_id)
                               prediction_id, label_value, label_contract_version
                        FROM prediction_label_revisions
                        WHERE label_revision_id <= :label_revision_watermark
                          AND is_simulated = :is_simulated
                          AND simulation_scope = :simulation_scope
                        ORDER BY prediction_id, revision_number DESC,
                                 label_revision_id DESC
                    ), reasons AS (
                        SELECT CASE
                            WHEN NOT p.monitoring_eligible THEN 'prediction_not_monitoring_eligible'
                            WHEN p.policy_version IS DISTINCT FROM :policy_version
                                 THEN 'policy_version_mismatch'
                            WHEN p.deployment_id IS NULL THEN 'missing_deployment_identity'
                            WHEN p.horizon_end IS NULL THEN 'missing_prediction_horizon'
                            WHEN labels.prediction_id IS NULL OR labels.label_value IS NULL
                                 THEN 'binary_label_unavailable_at_watermark'
                            WHEN labels.label_contract_version <> :label_contract_version
                                 THEN 'label_contract_version_mismatch'
                            ELSE NULL
                        END AS reason
                        FROM prediction_events p
                        LEFT JOIN latest_labels labels USING (prediction_id)
                        WHERE p.prediction_timestamp >= :cohort_start
                          AND p.prediction_timestamp < :cohort_end
                          AND p.model_version_id = :model_version_id
                    )
                    SELECT reason, COUNT(*) AS count
                    FROM reasons WHERE reason IS NOT NULL
                    GROUP BY reason ORDER BY reason
                    """
                ),
                {
                    "label_revision_watermark": definition.label_revision_watermark,
                    "is_simulated": definition.is_simulated,
                    "simulation_scope": _simulation_scope(definition),
                    "policy_version": definition.policy_version,
                    "label_contract_version": definition.label_contract_version,
                    "cohort_start": definition.cohort_start,
                    "cohort_end": definition.cohort_end,
                    "model_version_id": definition.model_version_id,
                },
            ).mappings().all()
        return {row["reason"]: int(row["count"]) for row in rows}

    def reserve_performance_run(
        self, metadata: dict[str, Any]
    ) -> tuple[dict[str, Any], bool]:
        values = {
            **metadata,
            "deployment_ids": list(metadata["deployment_ids"]),
            "outcome_watermark": json.dumps(metadata["outcome_watermark"]),
            "run_configuration": json.dumps(metadata["run_configuration"]),
        }
        with self.engine.begin() as connection:
            inserted = connection.execute(
                text(
                    """
                    INSERT INTO performance_monitoring_runs (
                        monitoring_run_id, model_version_id, deployment_ids,
                        policy_version, label_contract_version, cohort_start,
                        cohort_end, cohort_selection_rule, outcome_watermark,
                        label_revision_watermark, is_simulated,
                        simulation_generator, simulation_scenario_version,
                        artifact_prefix,
                        run_configuration, status, started_at
                    ) VALUES (
                        :monitoring_run_id, :model_version_id, :deployment_ids,
                        :policy_version, :label_contract_version, :cohort_start,
                        :cohort_end, :cohort_selection_rule,
                        CAST(:outcome_watermark AS jsonb), :label_revision_watermark,
                        :is_simulated, :simulation_generator,
                        :simulation_scenario_version, :artifact_prefix,
                        CAST(:run_configuration AS jsonb), 'running', :started_at
                    )
                    ON CONFLICT (monitoring_run_id) DO NOTHING
                    RETURNING monitoring_run_id
                    """
                ),
                values,
            ).scalar_one_or_none()
            row = connection.execute(
                text(
                    """
                    SELECT monitoring_run_id, status, summary, artifact_uris,
                           artifact_checksums, started_at, completed_at
                    FROM performance_monitoring_runs
                    WHERE monitoring_run_id = :monitoring_run_id
                    """
                ),
                values,
            ).mappings().one()
            owns = inserted is not None
            if not owns and row["status"] == "failed":
                reclaimed = connection.execute(
                    text(
                        """
                        UPDATE performance_monitoring_runs
                        SET status = 'running', summary = NULL, error_details = NULL,
                            completed_at = NULL
                        WHERE monitoring_run_id = :monitoring_run_id
                          AND status = 'failed'
                        RETURNING monitoring_run_id
                        """
                    ),
                    values,
                ).scalar_one_or_none()
                owns = reclaimed is not None
                if owns:
                    row = {**dict(row), "status": "running", "summary": None}
        return dict(row), owns

    def finish_performance_run(
        self, run_id: str, *, summary: dict[str, Any], artifacts: dict[str, Any]
    ) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    UPDATE performance_monitoring_runs
                    SET status = 'completed', summary = CAST(:summary AS jsonb),
                        artifact_uris = CAST(:uris AS jsonb),
                        artifact_checksums = CAST(:checksums AS jsonb),
                        suppression_metadata = CAST(:suppression AS jsonb),
                        completed_at = now()
                    WHERE monitoring_run_id = :run_id
                    """
                ),
                {
                    "run_id": run_id,
                    "summary": json.dumps(summary),
                    "uris": json.dumps(artifacts["uris"]),
                    "checksums": json.dumps(artifacts["checksums"]),
                    "suppression": json.dumps(summary["suppression_metadata"]),
                },
            )

    def fail_performance_run(self, run_id: str, error: dict[str, Any]) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    UPDATE performance_monitoring_runs
                    SET status = 'failed', error_details = CAST(:error AS jsonb),
                        completed_at = now()
                    WHERE monitoring_run_id = :run_id
                    """
                ),
                {"run_id": run_id, "error": json.dumps(error)},
            )


def _simulation_scope(definition: CohortDefinition) -> str:
    if not definition.is_simulated:
        return "real"
    generator = definition.simulation_generator or ""
    scenario = definition.simulation_scenario_version or ""
    return f"simulation:{len(generator)}:{generator}:{len(scenario)}:{scenario}"
