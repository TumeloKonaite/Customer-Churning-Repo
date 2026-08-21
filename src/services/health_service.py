"""Health endpoint service."""

from datetime import datetime, timezone

from src.services import model_service


def get_health_status() -> dict:
    metadata = model_service.load_metadata()
    response = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_loaded": model_service.artifacts_ready(),
        "metadata": metadata,
    }
    for field in (
        "deployment_id",
        "model_name",
        "model_version",
        "model_version_id",
        "mlflow_run_id",
        "feature_schema_version",
        "pipeline_sha256",
        "artifact_manifest_sha256",
        "integrity_status",
    ):
        if metadata.get(field) is not None:
            response[field] = metadata[field]
    return response
