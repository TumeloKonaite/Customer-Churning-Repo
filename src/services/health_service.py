"""Health endpoint service."""

from datetime import datetime, timezone

from src.services import model_service


def get_health_status() -> dict:
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_loaded": model_service.artifacts_ready(),
        "metadata": model_service.load_metadata(),
    }
