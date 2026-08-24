"""FastAPI application factory."""

from functools import partial

from fastapi import FastAPI

from src.api.errors import register_exception_handlers
from src.api.openapi import build_openapi_schema
from src.api.routes import health, outcomes, predictions, web
from src.schemas.batch_prediction import MAX_BATCH_SIZE


def create_app() -> FastAPI:
    app = FastAPI(
        title="Customer Churn Prediction API",
        description=(
            "Train-backed customer churn predictions for individual customers and JSON or CSV "
            f"batches. Batch requests support fail_fast and partial modes and at most "
            f"{MAX_BATCH_SIZE} records."
        ),
        version="1.0.0",
        openapi_tags=[
            {"name": "Health", "description": "Service and model readiness."},
            {"name": "Predictions", "description": "Single and batch churn inference."},
            {"name": "Monitoring outcomes", "description": "Protected outcome ingestion."},
        ],
    )
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(predictions.router)
    app.include_router(outcomes.router)
    app.include_router(web.router)
    app.openapi = partial(build_openapi_schema, app)
    return app
