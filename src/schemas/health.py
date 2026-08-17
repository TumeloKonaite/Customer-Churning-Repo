"""Health endpoint schemas."""

from typing import Any, Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Service health and model-artifact readiness."""

    status: Literal["healthy"]
    timestamp: str
    model_loaded: bool
    metadata: dict[str, Any]
