"""FastAPI application factory."""

from functools import partial
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.errors import register_exception_handlers
from src.api.openapi import build_openapi_schema
from src.api.routes import health, outcomes, predictions, web
from src.schemas.batch_prediction import MAX_BATCH_SIZE


def _frontend_origins() -> list[str]:
    """Return the exact browser origins allowed to call the public API."""

    configured = [
        origin.strip().rstrip("/")
        for origin in os.getenv("FRONTEND_ALLOWED_ORIGINS", "").split(",")
        if origin.strip()
    ]
    if "*" in configured:
        raise ValueError("FRONTEND_ALLOWED_ORIGINS must contain exact origins, not '*'")

    environment = os.getenv("APP_ENV", "development").strip().lower()
    if environment in {"development", "test"}:
        for local_origin in ("http://localhost:5173", "http://127.0.0.1:5173"):
            if local_origin not in configured:
                configured.append(local_origin)
    return configured


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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_frontend_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Accept", "Content-Type"],
    )
    app.include_router(health.router)
    app.include_router(predictions.router)
    app.include_router(outcomes.router)
    app.include_router(web.router)
    app.openapi = partial(build_openapi_schema, app)
    return app
