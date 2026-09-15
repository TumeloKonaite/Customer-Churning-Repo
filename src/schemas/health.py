"""Health endpoint schemas."""

from typing import Any, Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Service health and model-artifact readiness."""

    status: Literal["healthy"]
    timestamp: str
    model_loaded: bool
    metadata: dict[str, Any]
    deployment_id: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    model_version_id: str | None = None
    mlflow_run_id: str | None = None
    feature_schema_version: str | None = None
    pipeline_sha256: str | None = None
    artifact_manifest_sha256: str | None = None
    integrity_status: str | None = None
