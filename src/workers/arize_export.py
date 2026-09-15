"""Runtime composition for the Arize outbox exporter."""

from typing import Any

from src.config import DatabaseSettings
from src.database import create_database_engine
from src.monitoring.arize.client import ArizeV8Client
from src.monitoring.arize.config import ArizeExportSettings
from src.monitoring.arize.exporter import ArizeExporter
from src.monitoring.arize.repository import ArizeOutboxRepository


def execute_arize_export() -> dict[str, Any]:
    """Deliver one bounded outbox batch and surface persisted failures."""
    settings = ArizeExportSettings()
    engine = create_database_engine(DatabaseSettings())
    try:
        summary = ArizeExporter(
            ArizeOutboxRepository(engine),
            ArizeV8Client(settings),
            settings,
        ).run()
        if summary["retried"] or summary["dead_lettered"]:
            raise RuntimeError(
                "Arize delivery completed with persisted failures: "
                f"retried={summary['retried']} "
                f"dead_lettered={summary['dead_lettered']}"
            )
        return summary
    finally:
        engine.dispose()
