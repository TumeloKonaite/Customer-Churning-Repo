"""Conservative SQLAlchemy engine construction for Neon and Modal scaling."""

from __future__ import annotations

import logging
import time

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config import DatabaseSettings


logger = logging.getLogger(__name__)


def create_database_engine(settings: DatabaseSettings | None = None) -> Engine:
    settings = settings or DatabaseSettings()
    return create_engine(
        settings.sqlalchemy_url(),
        pool_pre_ping=True,
        pool_size=settings.pool_size,
        max_overflow=settings.max_overflow,
        connect_args={"connect_timeout": settings.connect_timeout_seconds},
    )


def check_connectivity(settings: DatabaseSettings | None = None) -> dict[str, object]:
    """Execute SELECT 1 and report only non-sensitive status and timing."""
    started = time.perf_counter()
    engine = create_database_engine(settings)
    try:
        with engine.connect() as connection:
            value = connection.execute(text("SELECT 1")).scalar_one()
        if value != 1:
            raise RuntimeError("Database connectivity query returned an unexpected value")
        duration = time.perf_counter() - started
        logger.info("database_connectivity status=ok duration_seconds=%.3f", duration)
        return {"status": "ok", "query": "SELECT 1", "duration_seconds": duration}
    finally:
        engine.dispose()
