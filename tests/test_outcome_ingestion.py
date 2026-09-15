from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.monitoring.outcomes import (
    CanonicalOutcome,
    OutcomeConflictError,
    OutcomeIngestionService,
)


UTC = timezone.utc
NOW = datetime(2026, 8, 24, 12, tzinfo=UTC)


class MemoryOutcomeStore:
    def __init__(self):
        self.events = {}
        self.quarantine = []

    def find_outcome(self, namespace, event_id):
        return self.events.get((namespace, event_id))

    def is_outcome_current(self, outcome_event_id):
        return not any(
            event.referenced_outcome_event_id == outcome_event_id
            for event in self.events.values()
        )

    def ingest_outcome(self, outcome: CanonicalOutcome):
        key = (outcome.source_namespace, outcome.source_event_id)
        existing = self.events.get(key)
        if existing is not None:
            if existing.content_sha256 != outcome.content_sha256:
                raise OutcomeConflictError("conflict")
            return existing, False
        self.events[key] = outcome
        return outcome, True

    def quarantine_outcome(self, **values):
        self.quarantine.append(values)


def payload(**changes):
    value = {
        "source_event_id": "event-1",
        "source_namespace": "customer-master",
        "environment": "production",
        "tenant_id": "tenant-1",
        "customer_id": "raw-customer-1",
        "event_type": "CUSTOMER_RELATIONSHIP_TERMINATED",
        "event_timestamp": NOW - timedelta(days=1),
        "received_timestamp": NOW,
        "is_simulated": False,
    }
    value.update(changes)
    return value


def service(store=None):
    return OutcomeIngestionService(
        store or MemoryOutcomeStore(),
        token_secret=b"a" * 32,
        token_key_id="key-1",
        service_environment="production",
        allowed_real_source_namespaces=frozenset({"customer-master"}),
        now=lambda: NOW,
    )


def test_identical_replay_returns_stored_outcome_and_conflict_is_rejected():
    current = service()
    first = current.ingest(payload())
    second = current.ingest(payload())

    assert first["status"] == "created"
    assert second["status"] == "replayed"
    assert first["outcome"] == second["outcome"]
    assert "customer_id" not in repr(first)

    with pytest.raises(OutcomeConflictError):
        current.ingest(payload(event_timestamp=NOW - timedelta(days=2)))


def test_batch_partial_failure_never_echoes_raw_identity():
    current = service()
    raw = "do-not-echo-this-customer"
    result = current.ingest_batch(
        [payload(), payload(source_event_id="bad", customer_id=raw, event_type="UNKNOWN")]
    )

    assert result["status"] == "partial"
    assert result["summary"] == {"total": 2, "accepted": 1, "rejected": 1}
    assert raw not in repr(result)
    assert current.store.quarantine[0]["reason"] == "ValidationError"


def test_correction_requires_valid_current_same_customer_reference():
    current = service()
    current.ingest(payload())
    correction_payload = payload(
            source_event_id="event-2",
            operation="correction",
            reference={
                "source_namespace": "customer-master",
                "source_event_id": "event-1",
            },
            event_timestamp=NOW - timedelta(days=2),
        )
    corrected = current.ingest(correction_payload)
    assert corrected["outcome"]["operation"] == "correction"
    assert corrected["outcome"]["referenced_outcome_event_id"]
    assert current.ingest(correction_payload)["status"] == "replayed"

    with pytest.raises(ValueError, match="already been superseded"):
        current.ingest(
            payload(
                source_event_id="event-3",
                operation="retraction",
                reference={
                    "source_namespace": "customer-master",
                    "source_event_id": "event-1",
                },
            )
        )


def test_simulated_outcome_is_rejected_in_production():
    with pytest.raises(ValueError, match="not allowed in production"):
        service().ingest(
            payload(
                source_namespace="simulation:test",
                is_simulated=True,
                simulation_generator="generator-1",
                simulation_scenario_version="scenario-1",
            )
        )
