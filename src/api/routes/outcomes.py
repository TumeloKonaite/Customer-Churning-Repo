"""Protected HTTP adapters for outcome events and source completeness."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache
import hmac
from typing import Any

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from starlette.concurrency import run_in_threadpool

from src.config import DatabaseSettings, OutcomeIngestionSettings
from src.database import create_database_engine
from src.monitoring.models import require_utc, timestamp
from src.monitoring.outcome_repository import OutcomeRepository
from src.monitoring.outcomes import OutcomeConflictError, OutcomeIngestionService
from src.monitoring.outcomes import OutcomeIngestionRequest
from src.services.exceptions import APIServiceError


router = APIRouter()
MAX_OUTCOME_BATCH_SIZE = 1000


class OutcomeBatchEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[OutcomeIngestionRequest] = Field(
        min_length=1, max_length=MAX_OUTCOME_BATCH_SIZE
    )


@lru_cache(maxsize=1)
def _settings() -> OutcomeIngestionSettings:
    return OutcomeIngestionSettings()


@lru_cache(maxsize=1)
def _repository() -> OutcomeRepository:
    return OutcomeRepository(create_database_engine(DatabaseSettings()))


def get_outcome_service() -> OutcomeIngestionService:
    settings = _settings()
    return OutcomeIngestionService(
        _repository(),
        token_secret=settings.token_secret.get_secret_value().encode("utf-8"),
        token_key_id=settings.token_key_id,
        service_environment=settings.environment.value,
        allowed_real_source_namespaces=settings.allowed_sources,
    )


def require_outcome_api_key(
    x_outcome_api_key: str | None = Header(default=None),
) -> None:
    expected = _settings().ingestion_api_key.get_secret_value()
    if x_outcome_api_key is None or not hmac.compare_digest(x_outcome_api_key, expected):
        raise APIServiceError("Outcome ingestion authentication failed", status_code=401)


def _require_json(request: Request) -> None:
    media_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    if media_type != "application/json" and not media_type.endswith("+json"):
        raise APIServiceError("Content-Type must be application/json", status_code=415)


@router.post(
    "/api/monitoring/outcomes",
    tags=["Monitoring outcomes"],
    summary="Idempotently ingest one outcome or a partial-success batch",
    dependencies=[Depends(require_outcome_api_key)],
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "oneOf": [
                            {"$ref": "#/components/schemas/OutcomeIngestionRequest"},
                            {"$ref": "#/components/schemas/OutcomeBatchEnvelope"},
                        ]
                    }
                }
            },
        }
    },
)
async def ingest_outcomes(
    request: Request,
    service: OutcomeIngestionService = Depends(get_outcome_service),
) -> JSONResponse:
    _require_json(request)
    try:
        payload = await request.json()
    except Exception as exc:
        raise APIServiceError("Invalid JSON body") from exc
    if isinstance(payload, dict) and "records" in payload:
        if set(payload) != {"records"} or not isinstance(payload["records"], list):
            raise APIServiceError("Batch body must contain only a records list")
        records = payload["records"]
        if len(records) > MAX_OUTCOME_BATCH_SIZE:
            raise APIServiceError("Outcome batch is too large", status_code=413)
        result = await run_in_threadpool(service.ingest_batch, records)
        return JSONResponse(result, status_code=207 if result["status"] == "partial" else 200)
    if not isinstance(payload, dict):
        raise APIServiceError("Outcome body must be an object")
    try:
        result = await run_in_threadpool(service.ingest, payload)
    except OutcomeConflictError as exc:
        raise APIServiceError(str(exc), status_code=409) from exc
    except (ValidationError, ValueError) as exc:
        # Never serialize Pydantic's input echo: it can contain customer_id.
        raise APIServiceError("Outcome failed contract validation", status_code=422) from exc
    return JSONResponse(result, status_code=201 if result["status"] == "created" else 200)


class SourceWatermarkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_namespace: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    environment: str = Field(min_length=1)
    is_simulated: bool = False
    complete_through: datetime

    @field_validator("complete_through")
    @classmethod
    def utc_timestamp(cls, value: datetime) -> datetime:
        return require_utc(value, "complete_through")


@router.post(
    "/api/monitoring/outcomes/watermarks",
    tags=["Monitoring outcomes"],
    summary="Declare an authoritative outcome-source completeness watermark",
    dependencies=[Depends(require_outcome_api_key)],
)
def advance_outcome_watermark(payload: SourceWatermarkRequest) -> dict[str, Any]:
    settings = _settings()
    if payload.environment != settings.environment.value:
        raise APIServiceError("Watermark environment does not match service", status_code=422)
    if payload.is_simulated and payload.environment == "production":
        raise APIServiceError("Simulated watermarks are forbidden in production", status_code=422)
    if payload.is_simulated and not payload.source_namespace.startswith("simulation:"):
        raise APIServiceError("Simulated source must use simulation namespace", status_code=422)
    if not payload.is_simulated and payload.source_namespace.startswith("simulation:"):
        raise APIServiceError("Real source cannot use simulation namespace", status_code=422)
    if not payload.is_simulated and payload.source_namespace not in settings.allowed_sources:
        raise APIServiceError("Source namespace is not approved", status_code=422)
    observed_at = datetime.now(timezone.utc)
    if payload.complete_through > observed_at + timedelta(minutes=5):
        raise APIServiceError("Completeness timestamp is operationally impossible", status_code=422)
    row = _repository().advance_source_watermark(
        source_namespace=payload.source_namespace,
        environment=payload.environment,
        is_simulated=payload.is_simulated,
        complete_through=payload.complete_through,
        observed_at=observed_at,
    )
    return {
        "status": "accepted",
        "source_namespace": row["source_namespace"],
        "environment": row["environment"],
        "is_simulated": row["is_simulated"],
        "complete_through": timestamp(row["complete_through"]),
        "observed_at": timestamp(row["observed_at"]),
    }
