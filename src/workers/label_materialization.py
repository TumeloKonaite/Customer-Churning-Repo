"""Runtime composition for delayed actual-label materialization."""

from datetime import datetime, timedelta, timezone
from typing import Any

from src.config import DatabaseSettings, LabelMaterializationSettings
from src.database import create_database_engine
from src.monitoring.labels.job import LabelMaterializationJob
from src.monitoring.labels.repository import LabelRepository


def execute_label_materialization(as_of: str | None = None) -> dict[str, Any]:
    """Materialize one reproducible label snapshot for the Arize actuals feed."""
    settings = LabelMaterializationSettings()
    engine = create_database_engine(DatabaseSettings())
    try:
        return LabelMaterializationJob(
            LabelRepository(engine),
            required_sources=settings.required_sources,
            horizon_days=settings.horizon_days,
            grace_period_days=settings.grace_period_days,
            label_contract_version=settings.label_contract_version,
        ).run(
            environment=settings.environment.value,
            is_simulated=False,
            as_of=resolve_observation_time(as_of),
        )
    finally:
        engine.dispose()


def resolve_observation_time(as_of: str | None = None) -> datetime:
    """Resolve an explicit UTC instant or the most recent daily schedule."""
    if as_of:
        observed_at = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("as_of must include a timezone")
        return observed_at.astimezone(timezone.utc)

    current = datetime.now(timezone.utc)
    scheduled = current.replace(hour=2, minute=45, second=0, microsecond=0)
    return scheduled if scheduled <= current else scheduled - timedelta(days=1)
