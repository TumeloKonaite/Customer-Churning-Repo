"""Health HTTP endpoint."""

from fastapi import APIRouter

from src.schemas.health import HealthResponse
from src.services.health_service import get_health_status


router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Check service and model readiness",
    response_model_exclude_unset=True,
)
def health_check():
    return get_health_status()
