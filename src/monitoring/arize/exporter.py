"""Bounded outbox delivery orchestration."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import random
from typing import Any

from .client import ArizeDeliveryError
from .config import ArizeExportSettings
from .repository import ArizeOutboxRepository
from .schema import actual_frame, numeric_model_version, prediction_frame


class ArizeExporter:
    def __init__(self, repository: ArizeOutboxRepository, client: Any, settings: ArizeExportSettings):
        self.repository = repository
        self.client = client
        self.settings = settings

    def run(self, *, now: datetime | None = None) -> dict[str, Any]:
        at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        self.settings.require_enabled()
        self.repository.require_active_approval(
            approval_id=self.settings.privacy_approval_id or "",
            model_name=self.settings.model_name,
            environment=self.settings.environment.value,
            at=at,
        )
        recovered = self.repository.recover_stale_claims(
            older_than=at - timedelta(minutes=self.settings.claim_timeout_minutes)
        )
        rows = self.repository.claim(limit=self.settings.batch_size, now=at)
        summary = {"claimed": len(rows), "sent": 0, "retried": 0,
                   "dead_lettered": 0, "stale_claims_recovered": recovered}
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            try:
                version = numeric_model_version(row["model_version_id"])
                if row["environment"] != self.settings.environment.value:
                    raise ValueError("event environment mismatch")
                groups[(row["event_type"], version)].append(row)
            except ValueError:
                retried, dead = self.repository.mark_failed(
                    [row["export_event_id"]], error_code="schema_invalid",
                    retryable=False, max_attempts=self.settings.max_attempts,
                    next_attempt_at=at,
                    claim_token=row.get("claim_token"),
                )
                summary["retried"] += retried
                summary["dead_lettered"] += dead
        for (event_type, version), group in groups.items():
            identifiers = [row["export_event_id"] for row in group]
            try:
                if event_type == "prediction":
                    self.client.log_predictions(prediction_frame(group), model_version=version)
                else:
                    self.client.log_actuals(actual_frame(group), model_version=version)
                self.repository.mark_sent(
                    identifiers, at=at, claim_token=group[0].get("claim_token")
                )
                summary["sent"] += len(group)
            except (ArizeDeliveryError, ValueError) as exc:
                retryable = bool(getattr(exc, "retryable", False))
                attempt = max(int(row["attempt_count"]) for row in group)
                delay = min(
                    self.settings.max_backoff_seconds,
                    self.settings.initial_backoff_seconds * (2 ** max(0, attempt - 1)),
                )
                jittered = max(1, int(delay * random.uniform(0.8, 1.2)))
                retried, dead = self.repository.mark_failed(
                    identifiers,
                    error_code="delivery_temporary" if retryable else "schema_or_auth_permanent",
                    retryable=retryable,
                    max_attempts=self.settings.max_attempts,
                    next_attempt_at=at + timedelta(seconds=jittered),
                    claim_token=group[0].get("claim_token"),
                )
                summary["retried"] += retried
                summary["dead_lettered"] += dead
        summary.update(self.repository.backlog(now=at))
        return summary
