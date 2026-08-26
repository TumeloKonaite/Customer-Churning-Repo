"""Neon persistence for outcome snapshots and append-only label revisions."""

from __future__ import annotations

from datetime import datetime
import json
from typing import Any, Mapping

from sqlalchemy import Engine, bindparam, text

from src.monitoring.performance.labels import (
    LabelingPrediction,
    LabelRevision,
    OutcomeSnapshot,
)
from src.monitoring.outcomes.models import CanonicalOutcome


class LabelRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    def establish_outcome_snapshot(
        self,
        *,
        environment: str,
        is_simulated: bool,
        as_of: datetime,
        required_sources: tuple[str, ...],
    ) -> OutcomeSnapshot:
        with self.engine.connect() as connection:
            maximum = connection.execute(
                text(
                    """
                    SELECT COALESCE(MAX(outcome_ingest_id), 0)
                    FROM outcome_events
                    WHERE environment = :environment
                      AND is_simulated = :is_simulated
                      AND persisted_at <= :as_of
                    """
                ),
                {
                    "environment": environment,
                    "is_simulated": is_simulated,
                    "as_of": as_of,
                },
            ).scalar_one()
            rows = connection.execute(
                text(
                    """
                    SELECT DISTINCT ON (source_namespace)
                           source_namespace, complete_through
                    FROM outcome_source_watermarks
                    WHERE environment = :environment
                      AND is_simulated = :is_simulated
                      AND observed_at <= :as_of
                      AND source_namespace = ANY(:required_sources)
                    ORDER BY source_namespace, observed_at DESC, watermark_id DESC
                    """
                ),
                {
                    "environment": environment,
                    "is_simulated": is_simulated,
                    "as_of": as_of,
                    "required_sources": list(required_sources),
                },
            ).mappings().all()
        completeness = {source: None for source in required_sources}
        completeness.update(
            {row["source_namespace"]: row["complete_through"] for row in rows}
        )
        return OutcomeSnapshot(
            as_of=as_of,
            maximum_outcome_ingest_id=maximum,
            required_sources=required_sources,
            source_complete_through=completeness,
        )

    def reserve_materialization_run(
        self, metadata: dict[str, Any]
    ) -> tuple[dict[str, Any], bool]:
        values = {
            **metadata,
            "outcome_watermark": json.dumps(metadata["outcome_watermark"]),
            "status": "running",
        }
        with self.engine.begin() as connection:
            inserted = connection.execute(
                text(
                    """
                    INSERT INTO label_materialization_runs (
                        materialization_run_id, environment, is_simulated,
                        simulation_generator, simulation_scenario_version,
                        label_contract_version, horizon_days, grace_period_days,
                        outcome_watermark, status, started_at
                    ) VALUES (
                        :materialization_run_id, :environment, :is_simulated,
                        :simulation_generator, :simulation_scenario_version,
                        :label_contract_version, :horizon_days, :grace_period_days,
                        CAST(:outcome_watermark AS jsonb), :status, :started_at
                    )
                    ON CONFLICT (materialization_run_id) DO NOTHING
                    RETURNING materialization_run_id
                    """
                ),
                values,
            ).scalar_one_or_none()
            row = connection.execute(
                text(
                    """
                    SELECT materialization_run_id, environment, is_simulated,
                           simulation_generator, simulation_scenario_version,
                           label_contract_version, horizon_days, grace_period_days,
                           outcome_watermark, status, started_at, completed_at,
                           summary, error_details
                    FROM label_materialization_runs
                    WHERE materialization_run_id = :materialization_run_id
                    """
                ),
                values,
            ).mappings().one()
            owns = inserted is not None
            if not owns and row["status"] in {"failed", "completed_with_errors"}:
                reclaimed = connection.execute(
                    text(
                        """
                        UPDATE label_materialization_runs
                        SET status = 'running', summary = NULL, error_details = NULL,
                            completed_at = NULL
                        WHERE materialization_run_id = :materialization_run_id
                          AND status IN ('failed', 'completed_with_errors')
                        RETURNING materialization_run_id
                        """
                    ),
                    values,
                ).scalar_one_or_none()
                owns = reclaimed is not None
                if owns:
                    row = {**dict(row), "status": "running", "summary": None}
        return dict(row), owns

    def load_labeling_predictions(
        self, *, environment: str, as_of: datetime
    ) -> tuple[LabelingPrediction, ...]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT prediction_id, environment, customer_token,
                           model_version_id, deployment_id, policy_version,
                           prediction_timestamp, horizon_end,
                           label_contract_version, monitoring_eligible,
                           prediction_probability, predicted_class, segments
                    FROM prediction_events
                    WHERE environment = :environment
                      AND prediction_timestamp < :as_of
                      AND customer_token IS NOT NULL
                      AND token_key_id IS NOT NULL
                      AND horizon_end IS NOT NULL
                      AND label_contract_version IS NOT NULL
                      AND deployment_id IS NOT NULL
                      AND policy_version IS NOT NULL
                    ORDER BY prediction_timestamp, prediction_id
                    """
                ),
                {"environment": environment, "as_of": as_of},
            ).mappings().all()
        return tuple(LabelingPrediction.model_validate(dict(row)) for row in rows)

    def load_outcomes(
        self,
        *,
        environment: str,
        is_simulated: bool,
        snapshot: OutcomeSnapshot,
        simulation_generator: str | None,
        simulation_scenario_version: str | None,
    ) -> tuple[CanonicalOutcome, ...]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT outcome_event_id, source_event_id, source_namespace,
                           environment, customer_token, token_key_id, event_type,
                           event_timestamp, received_timestamp, operation,
                           referenced_outcome_event_id, is_simulated,
                           simulation_generator, simulation_scenario_version,
                           label_contract_version
                    FROM outcome_events
                    WHERE environment = :environment
                      AND is_simulated = :is_simulated
                      AND persisted_at <= :as_of
                      AND outcome_ingest_id <= :maximum_outcome_ingest_id
                      AND (
                          NOT :is_simulated OR (
                              simulation_generator = :simulation_generator
                              AND simulation_scenario_version = :simulation_scenario_version
                          )
                      )
                    ORDER BY received_timestamp, outcome_ingest_id
                    """
                ),
                {
                    "environment": environment,
                    "is_simulated": is_simulated,
                    "as_of": snapshot.as_of,
                    "maximum_outcome_ingest_id": snapshot.maximum_outcome_ingest_id,
                    "simulation_generator": simulation_generator,
                    "simulation_scenario_version": simulation_scenario_version,
                },
            ).mappings().all()
        return tuple(CanonicalOutcome.model_validate(dict(row)) for row in rows)

    def latest_label_revisions(
        self,
        prediction_ids: tuple[str, ...],
        *,
        is_simulated: bool,
        simulation_generator: str | None,
        simulation_scenario_version: str | None,
    ) -> Mapping[str, LabelRevision]:
        if not prediction_ids:
            return {}
        statement = text(
            """
            SELECT DISTINCT ON (prediction_id)
                   label_revision_id, prediction_id, revision_number, label_value,
                   status, qualifying_outcome_event_id, label_contract_version,
                   materialization_run_id, attribution_timestamp, created_at,
                   supersedes_label_revision_id, revision_reason,
                   simulation_generator, simulation_scenario_version
            FROM prediction_label_revisions
            WHERE prediction_id IN :prediction_ids
              AND is_simulated = :is_simulated
              AND simulation_scope = :simulation_scope
            ORDER BY prediction_id, revision_number DESC, label_revision_id DESC
            """
        ).bindparams(bindparam("prediction_ids", expanding=True))
        with self.engine.connect() as connection:
            rows = connection.execute(
                statement,
                {
                    "prediction_ids": prediction_ids,
                    "is_simulated": is_simulated,
                    "simulation_scope": _simulation_scope(
                        is_simulated,
                        simulation_generator,
                        simulation_scenario_version,
                    ),
                },
            ).mappings().all()
        return {
            row["prediction_id"]: LabelRevision.model_validate(dict(row))
            for row in rows
        }

    def append_label_revision(
        self, revision: LabelRevision, *, is_simulated: bool
    ) -> LabelRevision:
        values = {**revision.model_dump(), "status": revision.status.value}
        values["is_simulated"] = is_simulated
        values["simulation_scope"] = _simulation_scope(
            is_simulated,
            revision.simulation_generator,
            revision.simulation_scenario_version,
        )
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    INSERT INTO prediction_label_revisions (
                        prediction_id, revision_number, label_value, status,
                        qualifying_outcome_event_id, label_contract_version,
                        materialization_run_id, attribution_timestamp, created_at,
                        supersedes_label_revision_id, revision_reason, is_simulated,
                        simulation_generator, simulation_scenario_version,
                        simulation_scope
                    ) VALUES (
                        :prediction_id, :revision_number, :label_value, :status,
                        :qualifying_outcome_event_id, :label_contract_version,
                        :materialization_run_id, :attribution_timestamp, :created_at,
                        :supersedes_label_revision_id, :revision_reason, :is_simulated,
                        :simulation_generator, :simulation_scenario_version,
                        :simulation_scope
                    )
                    ON CONFLICT (prediction_id, materialization_run_id, simulation_scope)
                    DO NOTHING
                    RETURNING label_revision_id, prediction_id, revision_number,
                              label_value, status, qualifying_outcome_event_id,
                              label_contract_version, materialization_run_id,
                              attribution_timestamp, created_at,
                              supersedes_label_revision_id, revision_reason,
                              simulation_generator, simulation_scenario_version
                    """
                ),
                values,
            ).mappings().one_or_none()
            if row is None:
                row = connection.execute(
                    text(
                        """
                        SELECT label_revision_id, prediction_id, revision_number,
                               label_value, status, qualifying_outcome_event_id,
                               label_contract_version, materialization_run_id,
                               attribution_timestamp, created_at,
                               supersedes_label_revision_id, revision_reason,
                               simulation_generator, simulation_scenario_version
                        FROM prediction_label_revisions
                        WHERE prediction_id = :prediction_id
                          AND materialization_run_id = :materialization_run_id
                          AND simulation_scope = :simulation_scope
                        """
                    ),
                    values,
                ).mappings().one()
        return LabelRevision.model_validate(dict(row))

    def count_quarantined_outcomes(self, *, as_of: datetime) -> int:
        with self.engine.connect() as connection:
            return int(
                connection.execute(
                    text(
                        "SELECT COUNT(*) FROM outcome_quarantine WHERE quarantined_at <= :as_of"
                    ),
                    {"as_of": as_of},
                ).scalar_one()
            )

    def finish_materialization_run(
        self, run_id: str, summary: dict[str, Any]
    ) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    UPDATE label_materialization_runs
                    SET status = :status, summary = CAST(:summary AS jsonb),
                        completed_at = :completed_at
                    WHERE materialization_run_id = :run_id
                    """
                ),
                {
                    "run_id": run_id,
                    "status": summary["status"],
                    "summary": json.dumps(summary),
                    "completed_at": summary["completed_at"],
                },
            )

    def fail_materialization_run(self, run_id: str, error: dict[str, Any]) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    UPDATE label_materialization_runs
                    SET status = 'failed', error_details = CAST(:error AS jsonb),
                        completed_at = now()
                    WHERE materialization_run_id = :run_id
                    """
                ),
                {"run_id": run_id, "error": json.dumps(error)},
            )


def _simulation_scope(
    is_simulated: bool,
    generator: str | None,
    scenario: str | None,
) -> str:
    if not is_simulated:
        return "real"
    if not generator or not scenario:
        raise ValueError("simulated labels require generator and scenario version")
    return f"simulation:{len(generator)}:{generator}:{len(scenario)}:{scenario}"


# Kept as one implementation during the schema-preserving refactor; consumers use
# this repository boundary while the orchestration module remains database-agnostic.
from src.monitoring.performance.service import PerformanceRepository  # noqa: E402

__all__ = ["LabelRepository", "PerformanceRepository"]
