"""Validated, idempotent outcome ingestion without retaining raw identities."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import StrEnum
import hashlib
from typing import Any, Callable, Protocol

from pydantic import BaseModel, ConfigDict, Field, SecretStr, StrictBool, field_validator, model_validator

from src.monitoring.shared.identity import (
    CONTRACT_VERSION,
    QUALIFYING_CHURN_EVENT_TYPES,
    ContractViolation,
    tokenize_customer_id,
)
from src.monitoring.shared.models import canonical_json_bytes, timestamp


EARLIEST_OPERATIONAL_TIMESTAMP = datetime(1970, 1, 1, tzinfo=timezone.utc)
MAX_RECEIVED_CLOCK_SKEW = timedelta(minutes=5)


class OutcomeOperation(StrEnum):
    CREATE = "create"
    CORRECTION = "correction"
    RETRACTION = "retraction"
    SUPERSESSION = "supersession"


class OutcomeReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_namespace: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    source_event_id: str = Field(
        min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_.:/-]+$"
    )


class OutcomeIngestionRequest(BaseModel):
    """Boundary model. ``customer_id`` must never be persisted or logged."""

    model_config = ConfigDict(extra="forbid")

    source_event_id: str = Field(
        min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_.:/-]+$"
    )
    source_namespace: str = Field(
        min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$"
    )
    environment: str = Field(min_length=1, max_length=64)
    tenant_id: str = Field(min_length=1, max_length=256)
    customer_id: SecretStr
    event_type: str = Field(min_length=1, max_length=128)
    event_timestamp: datetime
    received_timestamp: datetime
    operation: OutcomeOperation = OutcomeOperation.CREATE
    reference: OutcomeReference | None = None
    is_simulated: StrictBool = False
    simulation_generator: str | None = Field(default=None, min_length=1, max_length=128)
    simulation_scenario_version: str | None = Field(
        default=None, min_length=1, max_length=128
    )

    @field_validator("event_timestamp", "received_timestamp")
    @classmethod
    def timestamps_are_utc(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware")
        value = value.astimezone(timezone.utc)
        if value < EARLIEST_OPERATIONAL_TIMESTAMP:
            raise ValueError(f"{info.field_name} is operationally impossible")
        return value

    @field_validator("customer_id")
    @classmethod
    def stable_identity_is_present(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value():
            raise ValueError("stable customer identity is required")
        return value

    @model_validator(mode="after")
    def coherent_event(self) -> "OutcomeIngestionRequest":
        if self.event_type not in QUALIFYING_CHURN_EVENT_TYPES:
            raise ValueError("unknown outcome event type")
        if self.event_timestamp > self.received_timestamp:
            raise ValueError("event_timestamp must not be after received_timestamp")
        if self.operation is OutcomeOperation.CREATE and self.reference is not None:
            raise ValueError("create events must not contain a correction reference")
        if self.operation is not OutcomeOperation.CREATE and self.reference is None:
            raise ValueError(f"{self.operation.value} requires a reference")
        if self.reference is not None and (
            self.reference.source_namespace == self.source_namespace
            and self.reference.source_event_id == self.source_event_id
        ):
            raise ValueError("an outcome cannot reference itself")
        if self.is_simulated:
            if not self.simulation_generator or not self.simulation_scenario_version:
                raise ValueError(
                    "simulated outcomes require generator and scenario version"
                )
            if not self.source_namespace.startswith("simulation:"):
                raise ValueError("simulated outcome source must use simulation namespace")
        elif self.simulation_generator is not None or self.simulation_scenario_version is not None:
            raise ValueError("real outcomes must not contain simulation metadata")
        elif self.source_namespace.startswith("simulation:"):
            raise ValueError("real outcomes cannot use a simulation namespace")
        return self


class CanonicalOutcome(BaseModel):
    """Persistable outcome fields. Raw customer and tenant identifiers are absent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome_event_id: str
    source_event_id: str
    source_namespace: str
    environment: str
    customer_token: str
    token_key_id: str
    event_type: str
    event_timestamp: datetime
    received_timestamp: datetime
    operation: OutcomeOperation
    referenced_outcome_event_id: str | None = None
    is_simulated: bool
    simulation_generator: str | None = None
    simulation_scenario_version: str | None = None
    label_contract_version: str = CONTRACT_VERSION

    @property
    def content_sha256(self) -> str:
        # The source identity is also included so the digest is independently auditable.
        return hashlib.sha256(
            canonical_json_bytes(self.model_dump(mode="json"))
        ).hexdigest()

    def safe_response(self) -> dict[str, Any]:
        """Return operational fields only; do not echo raw identity or its token."""
        return {
            "outcome_event_id": self.outcome_event_id,
            "source_namespace": self.source_namespace,
            "source_event_id": self.source_event_id,
            "event_type": self.event_type,
            "event_timestamp": timestamp(self.event_timestamp),
            "received_timestamp": timestamp(self.received_timestamp),
            "operation": self.operation.value,
            "referenced_outcome_event_id": self.referenced_outcome_event_id,
            "is_simulated": self.is_simulated,
            "simulation_generator": self.simulation_generator,
            "simulation_scenario_version": self.simulation_scenario_version,
            "label_contract_version": self.label_contract_version,
        }


class OutcomeConflictError(ContractViolation):
    """A source identity was reused with different canonical content."""


class OutcomeReferenceError(ContractViolation):
    """A correction/retraction/supersession reference is invalid."""


class OutcomeStore(Protocol):
    def find_outcome(self, source_namespace: str, source_event_id: str) -> CanonicalOutcome | None: ...
    def ingest_outcome(self, outcome: CanonicalOutcome) -> tuple[CanonicalOutcome, bool]: ...
    def quarantine_outcome(self, *, source_namespace: str | None, source_event_id: str | None, reason: str) -> None: ...


def outcome_event_id(source_namespace: str, source_event_id: str) -> str:
    digest = hashlib.sha256(
        canonical_json_bytes([source_namespace, source_event_id])
    ).hexdigest()
    return f"out_{digest[:40]}"


class OutcomeIngestionService:
    """Convert raw identity once, validate references, and atomically ingest outcomes."""

    def __init__(
        self,
        store: OutcomeStore,
        *,
        token_secret: bytes,
        token_key_id: str,
        supported_token_key_ids: frozenset[str] | None = None,
        service_environment: str | None = None,
        allowed_real_source_namespaces: frozenset[str] | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        if len(token_secret) < 32:
            raise ValueError("token secret must contain at least 32 bytes")
        supported = supported_token_key_ids or frozenset({token_key_id})
        if token_key_id not in supported:
            raise ValueError("active token key is not in the supported key set")
        self.store = store
        self.token_secret = token_secret
        self.token_key_id = token_key_id
        self.supported_token_key_ids = supported
        self.service_environment = service_environment
        self.allowed_real_source_namespaces = allowed_real_source_namespaces
        self.now = now

    def _canonicalize(self, request: OutcomeIngestionRequest) -> CanonicalOutcome:
        current = self.now().astimezone(timezone.utc)
        replaying = self.store.find_outcome(
            request.source_namespace, request.source_event_id
        )
        if self.service_environment is not None and request.environment != self.service_environment:
            raise ContractViolation("outcome environment does not match the ingestion service")
        if request.is_simulated and request.environment == "production":
            raise ContractViolation("simulated outcomes are not allowed in production")
        if (
            not request.is_simulated
            and self.allowed_real_source_namespaces is not None
            and request.source_namespace not in self.allowed_real_source_namespaces
        ):
            raise ContractViolation("source namespace is not approved")
        if request.received_timestamp > current + MAX_RECEIVED_CLOCK_SKEW:
            raise ContractViolation("received_timestamp is operationally impossible")
        customer_token = tokenize_customer_id(
            environment=request.environment,
            tenant_id=request.tenant_id,
            customer_id=request.customer_id.get_secret_value(),
            secret_key=self.token_secret,
            key_id=self.token_key_id,
        )
        referenced: CanonicalOutcome | None = None
        if request.reference is not None:
            referenced = self.store.find_outcome(
                request.reference.source_namespace,
                request.reference.source_event_id,
            )
            if referenced is None:
                raise OutcomeReferenceError("referenced outcome does not exist")
            if referenced.customer_token != customer_token:
                raise OutcomeReferenceError("referenced outcome belongs to another customer")
            if referenced.source_namespace != request.source_namespace:
                raise OutcomeReferenceError("correction must remain in its source namespace")
            if referenced.environment != request.environment:
                raise OutcomeReferenceError("referenced outcome belongs to another environment")
            if referenced.is_simulated != request.is_simulated:
                raise OutcomeReferenceError("real and simulated outcome chains cannot mix")
            if request.is_simulated and (
                referenced.simulation_generator != request.simulation_generator
                or referenced.simulation_scenario_version
                != request.simulation_scenario_version
            ):
                raise OutcomeReferenceError("simulation correction changed its scenario identity")
            if request.received_timestamp < referenced.received_timestamp:
                raise OutcomeReferenceError("correction was received before its reference")
            is_current = getattr(self.store, "is_outcome_current", None)
            if (
                replaying is None
                and callable(is_current)
                and not is_current(referenced.outcome_event_id)
            ):
                raise OutcomeReferenceError("referenced outcome has already been superseded")
        return CanonicalOutcome(
            outcome_event_id=outcome_event_id(
                request.source_namespace, request.source_event_id
            ),
            source_event_id=request.source_event_id,
            source_namespace=request.source_namespace,
            environment=request.environment,
            customer_token=customer_token,
            token_key_id=self.token_key_id,
            event_type=request.event_type,
            event_timestamp=request.event_timestamp,
            received_timestamp=request.received_timestamp,
            operation=request.operation,
            referenced_outcome_event_id=(
                referenced.outcome_event_id if referenced is not None else None
            ),
            is_simulated=request.is_simulated,
            simulation_generator=request.simulation_generator,
            simulation_scenario_version=request.simulation_scenario_version,
        )

    def ingest(self, value: OutcomeIngestionRequest | dict[str, Any]) -> dict[str, Any]:
        request = (
            value
            if isinstance(value, OutcomeIngestionRequest)
            else OutcomeIngestionRequest.model_validate(value)
        )
        canonical = self._canonicalize(request)
        stored, created = self.store.ingest_outcome(canonical)
        return {
            "status": "created" if created else "replayed",
            "outcome": stored.safe_response(),
        }

    def ingest_batch(self, records: list[Any]) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        accepted = 0
        for index, record in enumerate(records):
            try:
                result = self.ingest(record)
                accepted += 1
                results.append({"index": index, **result})
            except Exception as exc:
                # Deliberately never include validation input or raw identifiers.
                source_namespace = record.get("source_namespace") if isinstance(record, dict) else None
                source_event_id = record.get("source_event_id") if isinstance(record, dict) else None
                reason = type(exc).__name__
                try:
                    self.store.quarantine_outcome(
                        source_namespace=(source_namespace if isinstance(source_namespace, str) else None),
                        source_event_id=(source_event_id if isinstance(source_event_id, str) else None),
                        reason=reason,
                    )
                except Exception:
                    # Quarantine persistence must not hide the per-record ingestion failure.
                    pass
                results.append(
                    {
                        "index": index,
                        "status": "rejected",
                        "error": {"type": reason, "message": _safe_ingestion_error(exc)},
                    }
                )
        return {
            "status": "success" if accepted == len(records) else ("failed" if accepted == 0 else "partial"),
            "results": results,
            "summary": {
                "total": len(records),
                "accepted": accepted,
                "rejected": len(records) - accepted,
            },
        }


def _safe_ingestion_error(exc: Exception) -> str:
    if isinstance(exc, OutcomeConflictError):
        return "source event identity conflicts with stored canonical content"
    if isinstance(exc, OutcomeReferenceError):
        return str(exc)
    if isinstance(exc, ContractViolation):
        return str(exc)
    # Pydantic messages can contain input values, including the raw identifier.
    return "outcome failed contract validation"
