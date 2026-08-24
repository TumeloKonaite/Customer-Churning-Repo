"""Append-only PostgreSQL persistence for production prediction events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from typing import Any, Iterable

from sqlalchemy import Engine, text


@dataclass(frozen=True)
class PendingPredictionEvent:
    """One prediction event ready for an append-only database insert."""

    prediction_id: str
    environment: str
    model_version_id: str
    prediction_timestamp: datetime
    feature_schema_version: str
    features: dict[str, Any]
    prediction_probability: float
    predicted_class: str
    deployment_id: str | None


class PredictionEventRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    def append(self, events: Iterable[PendingPredictionEvent]) -> int:
        """Atomically append events and return the number committed."""
        pending = tuple(events)
        if not pending:
            return 0

        statement = text(
            """
            INSERT INTO prediction_events (
                prediction_id, environment, model_version_id,
                prediction_timestamp, feature_schema_version, features,
                prediction_probability, predicted_class, deployment_id,
                monitoring_eligible
            ) VALUES (
                :prediction_id, :environment, :model_version_id,
                :prediction_timestamp, :feature_schema_version,
                CAST(:features AS jsonb), :prediction_probability,
                :predicted_class, :deployment_id, FALSE
            )
            """
        )
        parameters = [
            {
                "prediction_id": event.prediction_id,
                "environment": event.environment,
                "model_version_id": event.model_version_id,
                "prediction_timestamp": event.prediction_timestamp,
                "feature_schema_version": event.feature_schema_version,
                "features": json.dumps(
                    event.features,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                "prediction_probability": event.prediction_probability,
                "predicted_class": event.predicted_class,
                "deployment_id": event.deployment_id,
            }
            for event in pending
        ]
        with self.engine.begin() as connection:
            connection.execute(statement, parameters)
        return len(pending)
