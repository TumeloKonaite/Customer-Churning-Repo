"""Neon operations for privacy approval, claiming, and outbox transitions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Iterable
from uuid import uuid4

from sqlalchemy import Engine, bindparam, text

from .schema import ARIZE_FEATURES, ARIZE_TAGS


class PrivacyApprovalError(RuntimeError):
    pass


class ArizeOutboxRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    def require_active_approval(
        self, *, approval_id: str, model_name: str, environment: str,
        at: datetime | None = None,
    ) -> None:
        now = at or datetime.now(timezone.utc)
        with self.engine.connect() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT destination, model_name, environment,
                           approved_features, approved_tags
                    FROM arize_privacy_approvals
                    WHERE approval_id = :approval_id
                      AND approved_at <= :now
                      AND revoked_at IS NULL
                      AND (expires_at IS NULL OR expires_at > :now)
                    """
                ), {"approval_id": approval_id, "now": now},
            ).mappings().one_or_none()
        if row is None:
            raise PrivacyApprovalError("Arize privacy approval is absent, revoked, or expired")
        if row["destination"] != "arize" or row["model_name"] != model_name or row["environment"] != environment:
            raise PrivacyApprovalError("Arize privacy approval does not match this export")
        if set(row["approved_features"]) != set(ARIZE_FEATURES):
            raise PrivacyApprovalError("Arize privacy approval does not cover the exact feature allowlist")
        if not set(ARIZE_TAGS).issubset(set(row["approved_tags"])):
            raise PrivacyApprovalError("Arize privacy approval does not cover required tags")

    def recover_stale_claims(self, *, older_than: datetime) -> int:
        with self.engine.begin() as connection:
            result = connection.execute(
                text(
                    """
                    UPDATE arize_export_events
                    SET status = 'pending', claimed_at = NULL, claim_token = NULL,
                        next_attempt_at = clock_timestamp(), updated_at = clock_timestamp(),
                        last_error_code = 'stale_claim_recovered'
                    WHERE status = 'processing' AND claimed_at < :older_than
                    """
                ), {"older_than": older_than},
            )
        return int(result.rowcount or 0)

    def claim(self, *, limit: int, now: datetime) -> list[dict[str, Any]]:
        claim_token = uuid4()
        with self.engine.begin() as connection:
            identifiers = connection.execute(
                text(
                    """
                    WITH due AS (
                        SELECT e.export_event_id
                        FROM arize_export_events e
                        WHERE e.status = 'pending'
                          AND (e.next_attempt_at IS NULL OR e.next_attempt_at <= :now)
                          AND NOT (
                            e.event_type = 'actual' AND EXISTS (
                                SELECT 1 FROM arize_export_events prior
                                JOIN prediction_label_revisions pr
                                  ON pr.label_revision_id = prior.label_revision_id
                                JOIN prediction_label_revisions current_revision
                                  ON current_revision.label_revision_id = e.label_revision_id
                                WHERE prior.prediction_id = e.prediction_id
                                  AND prior.event_type = 'actual'
                                  AND pr.revision_number < current_revision.revision_number
                                  AND prior.status <> 'sent'
                            )
                          )
                        ORDER BY e.next_attempt_at NULLS FIRST, e.created_at, e.export_event_id
                        FOR UPDATE SKIP LOCKED
                        LIMIT :limit
                    )
                    UPDATE arize_export_events e
                    SET status = 'processing', claimed_at = :now,
                        attempt_count = e.attempt_count + 1, claim_token = :claim_token,
                        updated_at = clock_timestamp()
                    FROM due WHERE e.export_event_id = due.export_event_id
                    RETURNING e.export_event_id
                    """
                ), {"limit": limit, "now": now, "claim_token": claim_token},
            ).scalars().all()
            if not identifiers:
                return []
            statement = text(
                """
                SELECT e.export_event_id, e.event_type, e.attempt_count, e.claim_token,
                       e.prediction_id, e.label_revision_id,
                       p.environment, p.model_version_id, p.prediction_timestamp,
                       p.feature_schema_version, p.features,
                       p.prediction_probability, p.predicted_class,
                       p.deployment_id, p.request_source, p.batch_id,
                       r.label_value, r.label_contract_version,
                       r.created_at AS label_created_at, r.revision_number
                FROM arize_export_events e
                JOIN prediction_events p ON p.prediction_id = e.prediction_id
                LEFT JOIN prediction_label_revisions r
                  ON r.label_revision_id = e.label_revision_id
                WHERE e.export_event_id IN :identifiers
                ORDER BY p.model_version_id, e.event_type,
                         r.revision_number NULLS FIRST, e.created_at
                """
            ).bindparams(bindparam("identifiers", expanding=True))
            rows = connection.execute(statement, {"identifiers": identifiers}).mappings().all()
        return [dict(row) for row in rows]

    def mark_sent(self, identifiers: Iterable[Any], *, at: datetime, claim_token: Any = None) -> None:
        self._transition(
            identifiers,
            """status = 'sent', sent_at = :at, claimed_at = NULL,
                 claim_token = NULL, next_attempt_at = NULL, last_error_code = NULL""",
            {"at": at}, claim_token=claim_token,
        )

    def mark_failed(
        self, identifiers: Iterable[Any], *, error_code: str, retryable: bool,
        max_attempts: int, next_attempt_at: datetime,
        claim_token: Any = None,
    ) -> tuple[int, int]:
        ids = tuple(identifiers)
        if not ids:
            return 0, 0
        statement = text(
            """
            UPDATE arize_export_events
            SET status = CASE
                    WHEN NOT :retryable OR attempt_count >= :max_attempts
                    THEN 'dead_letter' ELSE 'pending' END,
                claimed_at = NULL, claim_token = NULL,
                next_attempt_at = CASE
                    WHEN NOT :retryable OR attempt_count >= :max_attempts
                    THEN NULL ELSE :next_attempt_at END,
                last_error_code = :error_code, updated_at = clock_timestamp()
            WHERE export_event_id IN :identifiers
              AND (:claim_token IS NULL OR claim_token = :claim_token)
            RETURNING status
            """
        ).bindparams(bindparam("identifiers", expanding=True))
        with self.engine.begin() as connection:
            states = connection.execute(statement, {
                "identifiers": ids, "retryable": retryable,
                "max_attempts": max_attempts, "next_attempt_at": next_attempt_at,
                "error_code": error_code[:80],
                "claim_token": claim_token,
            }).scalars().all()
        return states.count("pending"), states.count("dead_letter")

    def _transition(self, identifiers: Iterable[Any], assignment: str,
                    values: dict[str, Any], *, claim_token: Any = None) -> None:
        ids = tuple(identifiers)
        if not ids:
            return
        statement = text(
            f"UPDATE arize_export_events SET {assignment}, updated_at = clock_timestamp() "
            "WHERE export_event_id IN :identifiers AND status = 'processing' "
            "AND (:claim_token IS NULL OR claim_token = :claim_token)"
        ).bindparams(bindparam("identifiers", expanding=True))
        with self.engine.begin() as connection:
            connection.execute(statement, {**values, "identifiers": ids,
                                            "claim_token": claim_token})

    def backlog(self, *, now: datetime) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(text(
                """SELECT COUNT(*) FILTER (WHERE status = 'pending') AS pending,
                          COUNT(*) FILTER (WHERE status = 'dead_letter') AS dead_letter,
                          EXTRACT(EPOCH FROM (:now - MIN(created_at) FILTER
                            (WHERE status = 'pending'))) AS oldest_pending_seconds
                   FROM arize_export_events"""
            ), {"now": now}).mappings().one()
        return {
            "remaining_backlog": int(row["pending"] or 0),
            "dead_letter_backlog": int(row["dead_letter"] or 0),
            "oldest_pending_seconds": (
                None if row["oldest_pending_seconds"] is None
                else max(0, int(row["oldest_pending_seconds"]))
            ),
        }

    def enqueue_backfill_page(
        self, *, event_type: str, start: datetime | None, end: datetime | None,
        model_version_id: str | None, limit: int, resume_from: str | None,
        dry_run: bool,
    ) -> dict[str, Any]:
        if event_type not in {"prediction", "actual"}:
            raise ValueError("invalid backfill event type")
        timestamp_column = "p.prediction_timestamp" if event_type == "prediction" else "r.created_at"
        id_column = "p.prediction_id" if event_type == "prediction" else "r.label_revision_id"
        cursor_time = cursor_id = None
        if resume_from:
            raw_time, cursor_id = resume_from.rsplit("|", 1)
            cursor_time = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
        actual_join = "JOIN prediction_label_revisions r ON r.prediction_id = p.prediction_id" if event_type == "actual" else ""
        eligibility = (
            "AND r.is_simulated = FALSE AND r.status IN ('positive','negative') "
            "AND r.label_value IN (0,1)" if event_type == "actual" else ""
        )
        duplicate_revision = (
            "AND e.label_revision_id = r.label_revision_id"
            if event_type == "actual" else ""
        )
        query = text(f"""
            SELECT p.prediction_id, {timestamp_column} AS cursor_time,
                   {id_column} AS cursor_id
            FROM prediction_events p {actual_join}
            WHERE (:start IS NULL OR {timestamp_column} >= :start)
              AND (:end IS NULL OR {timestamp_column} < :end)
              AND (
                :model_version_id IS NULL
                OR p.model_version_id = :model_version_id
                OR (:model_version_id ~ '^[0-9]+$' AND
                    p.model_version_id LIKE '%:churn_predictor:' || :model_version_id)
              )
              AND (:cursor_time IS NULL OR ({timestamp_column}, {id_column}) >
                   (:cursor_time, :cursor_id))
              {eligibility}
              AND NOT EXISTS (
                  SELECT 1 FROM arize_export_events e
                  WHERE e.prediction_id = p.prediction_id
                    AND e.event_type = :event_type
                    {duplicate_revision}
              )
            ORDER BY {timestamp_column}, {id_column}
            LIMIT :limit
        """)
        params = {"start": start, "end": end, "model_version_id": model_version_id,
                  "cursor_time": cursor_time, "cursor_id": cursor_id,
                  "event_type": event_type, "limit": limit}
        with self.engine.begin() as connection:
            rows = connection.execute(query, params).mappings().all()
            if rows and not dry_run:
                if event_type == "prediction":
                    connection.execute(text("""
                        INSERT INTO arize_export_events (prediction_id, event_type, status, next_attempt_at)
                        VALUES (:prediction_id, 'prediction', 'pending', clock_timestamp())
                        ON CONFLICT DO NOTHING
                    """), [{"prediction_id": row["prediction_id"]} for row in rows])
                else:
                    # Re-selecting the revision is intentional: prediction_id alone is not a label join key.
                    connection.execute(text("""
                        INSERT INTO arize_export_events
                            (prediction_id, event_type, label_revision_id, status, next_attempt_at)
                        SELECT r.prediction_id, 'actual', r.label_revision_id, 'pending',
                               GREATEST(clock_timestamp(), p.horizon_end)
                        FROM prediction_label_revisions r
                        JOIN prediction_events p ON p.prediction_id = r.prediction_id
                        WHERE r.label_revision_id::text = :cursor_id
                        ON CONFLICT DO NOTHING
                    """), [{"cursor_id": row["cursor_id"]} for row in rows])
        last = rows[-1] if rows else None
        return {"selected": len(rows), "enqueued": 0 if dry_run else len(rows),
                "resume_from": None if last is None else f'{last["cursor_time"].isoformat()}|{last["cursor_id"]}'}
