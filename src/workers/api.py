"""Production API startup composition."""

from fastapi import FastAPI

from src.config import DatabaseSettings
from src.database import check_connectivity
from src.mlops.deployment import validate_production_startup


def create_production_api(package_dir: str = "/app/build/model") -> FastAPI:
    """Validate production dependencies before returning the ASGI application."""
    validate_production_startup(package_dir)
    check_connectivity(DatabaseSettings())

    from application import app

    return app
