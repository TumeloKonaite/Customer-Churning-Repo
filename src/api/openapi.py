"""OpenAPI compatibility customizations kept separate from app assembly."""

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from src.schemas.batch_prediction import (
    BatchOptions,
    BatchPredictionRecord,
    BatchPredictionRequest,
)
from src.schemas.errors import BatchContractError
from src.schemas.prediction import SinglePredictionRequest
from src.api.routes.outcomes import OutcomeBatchEnvelope, SourceWatermarkRequest
from src.monitoring.outcomes.models import OutcomeIngestionRequest, OutcomeReference


def build_openapi_schema(app: FastAPI):
    """Build the OpenAPI schema with explicit public request contracts."""
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
    )
    schemas = schema.setdefault("components", {}).setdefault("schemas", {})
    for model in (
        SinglePredictionRequest,
        BatchPredictionRecord,
        BatchOptions,
        BatchPredictionRequest,
        BatchContractError,
        OutcomeReference,
        OutcomeIngestionRequest,
        OutcomeBatchEnvelope,
        SourceWatermarkRequest,
    ):
        model_schema = model.model_json_schema(
            ref_template="#/components/schemas/{model}"
        )
        schemas.update(model_schema.pop("$defs", {}))
        schemas[model.__name__] = model_schema

    app.openapi_schema = schema
    return app.openapi_schema
