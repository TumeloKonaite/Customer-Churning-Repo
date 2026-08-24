"""PostgreSQL persistence for idempotent outcomes and source watermarks."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Engine, text
from sqlalchemy.exc import IntegrityError

from src.monitoring.outcomes import (
    CanonicalOutcome,
    OutcomeConflictError,
    OutcomeReferenceError,
)


class OutcomeRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    @staticmethod
    def _outcome(row: Any) -> CanonicalOutcome:
        return CanonicalOutcome.model_validate(dict(row))

    def find_outcome(
        self, source_namespace: str, source_event_id: str
    ) -> CanonicalOutcome | None:
        with self.engine.connect() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT outcome_event_id, source_event_id, source_namespace,
                           environment, customer_token, token_key_id, event_type,
                           event_timestamp, received_timestamp, operation,
                           referenced_outcome_event_id, is_simulated,
                           simulation_generator, simulation_scenario_version,
                           label_contract_version
                    FROM outcome_events
                    WHERE source_namespace = :source_namespace
                      AND source_event_id = :source_event_id
                    """
                ),
                {
                    "source_namespace": source_namespace,
                    "source_event_id": source_event_id,
                },
            ).mappings().one_or_none()
        return None if row is None else self._outcome(row)

    def is_outcome_current(self, outcome_event_id: str) -> bool:
        with self.engine.connect() as connection:
            return bool(
                connection.execute(
                    text(
                        """
                        SELECT EXISTS (
                            SELECT 1 FROM outcome_events target
                            WHERE target.outcome_event_id = :outcome_event_id
                              AND NOT EXISTS (
                                  SELECT 1 FROM outcome_events child
                                  WHERE child.referenced_outcome_event_id = target.outcome_event_id
                              )
                        )
                        """
                    ),
                    {"outcome_event_id": outcome_event_id},
                ).scalar_one()
            )

    def ingest_outcome(
        self, outcome: CanonicalOutcome
    ) -> tuple[CanonicalOutcome, bool]:
        values = outcome.model_dump()
        values["operation"] = outcome.operation.value
        values["content_sha256"] = outcome.content_sha256
        try:
            with self.engine.begin() as connection:
                inserted = connection.execute(
                    text(
                        """
                        INSERT INTO outcome_events (
                            outcome_event_id, source_event_id, source_namespace,
                            environment, customer_token, token_key_id, event_type,
                            event_timestamp, received_timestamp, operation,
                            referenced_outcome_event_id, is_simulated,
                            simulation_generator, simulation_scenario_version,
                            label_contract_version, content_sha256
                        ) VALUES (
                            :outcome_event_id, :source_event_id, :source_namespace,
                            :environment, :customer_token, :token_key_id, :event_type,
                            :event_timestamp, :received_timestamp, :operation,
                            :referenced_outcome_event_id, :is_simulated,
                            :simulation_generator, :simulation_scenario_version,
                            :label_contract_version, :content_sha256
                        )
                        ON CONFLICT (source_namespace, source_event_id) DO NOTHING
                        RETURNING outcome_event_id
                        """
                    ),
                    values,
                ).scalar_one_or_none()
                row = connection.execute(
                    text(
                        """
                        SELECT outcome_event_id, source_event_id, source_namespace,
                               environment, customer_token, token_key_id, event_type,
                               event_timestamp, received_timestamp, operation,
                               referenced_outcome_event_id, is_simulated,
                               simulation_generator, simulation_scenario_version,
                               label_contract_version, content_sha256
                        FROM outcome_events
                        WHERE source_namespace = :source_namespace
                          AND source_event_id = :source_event_id
                        """
                    ),
                    values,
                ).mappings().one()
                if row["content_sha256"] != outcome.content_sha256:
                    raise OutcomeConflictError(
                        "source event identity conflicts with stored canonical content"
                    )
        except IntegrityError as exc:
            raise OutcomeReferenceError(
                "outcome correction or supersession reference is no longer valid"
            ) from exc
        return self._outcome(row), inserted is not None

    def quarantine_outcome(
        self,
        *,
        source_namespace: str | None,
        source_event_id: str | None,
        reason: str,
    ) -> None:
        # No original payload is persisted: it may contain a raw customer identifier.
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO outcome_quarantine (
                        source_namespace, source_event_id, reason
                    ) VALUES (:source_namespace, :source_event_id, :reason)
                    """
                ),
                {
                    "source_namespace": source_namespace,
                    "source_event_id": source_event_id,
                    "reason": reason[:128],
                },
            )

    def advance_source_watermark(
        self,
        *,
        source_namespace: str,
        environment: str,
        is_simulated: bool,
        complete_through: datetime,
        observed_at: datetime,
    ) -> dict[str, Any]:
        """Monotonically advance an authoritative completeness declaration."""
        with self.engine.begin() as connection:
            connection.execute(
                text("SELECT pg_advisory_xact_lock(hashtextextended(:lock_key, 0))"),
                {
                    "lock_key": (
                        f"outcome-watermark|{source_namespace}|{environment}|"
                        f"{int(is_simulated)}"
                    )
                },
            )
            latest = connection.execute(
                text(
                    """
                    SELECT source_namespace, environment, is_simulated,
                           complete_through, observed_at
                    FROM outcome_source_watermarks
                    WHERE source_namespace = :source_namespace
                      AND environment = :environment
                      AND is_simulated = :is_simulated
                    ORDER BY complete_through DESC, watermark_id DESC
                    LIMIT 1
                    """
                ),
                {"source_namespace": source_namespace, "environment": environment,
                 "is_simulated": is_simulated},
            ).mappings().one_or_none()
            if latest is not None and latest["complete_through"] > complete_through:
                return dict(latest)
            row = connection.execute(
                text(
                    """
                    INSERT INTO outcome_source_watermarks (
                        source_namespace, environment, is_simulated,
                        complete_through, observed_at
                    ) VALUES (
                        :source_namespace, :environment, :is_simulated,
                        :complete_through, :observed_at
                    )
                    ON CONFLICT (source_namespace, environment, is_simulated, complete_through)
                    DO NOTHING
                    RETURNING source_namespace, environment, is_simulated,
                              complete_through, observed_at
                    """
                ),
                {"source_namespace": source_namespace, "environment": environment,
                 "is_simulated": is_simulated, "complete_through": complete_through,
                 "observed_at": observed_at},
            ).mappings().one_or_none()
            if row is None:
                row = connection.execute(
                    text(
                        """
                        SELECT source_namespace, environment, is_simulated,
                               complete_through, observed_at
                        FROM outcome_source_watermarks
                        WHERE source_namespace = :source_namespace
                          AND environment = :environment
                          AND is_simulated = :is_simulated
                          AND complete_through = :complete_through
                        """
                    ),
                    {"source_namespace": source_namespace, "environment": environment,
                     "is_simulated": is_simulated, "complete_through": complete_through},
                ).mappings().one()
        return dict(row)
