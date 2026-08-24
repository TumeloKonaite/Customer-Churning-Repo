"""Build and persist privacy-safe prediction monitoring events."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import logging
import math
from typing import Any, Sequence
from uuid import uuid4

from src.config import DatabaseSettings
from src.database import create_database_engine
from src.monitoring.prediction_repository import (
    PendingPredictionEvent,
    PredictionEventRepository,
)
from src.schemas.prediction import REQUIRED_FIELDS


logger = logging.getLogger(__name__)


class PredictionPersistenceError(RuntimeError):
    """A scored prediction could not be durably recorded."""


@lru_cache(maxsize=1)
def _runtime() -> tuple[DatabaseSettings, PredictionEventRepository | None]:
    settings = DatabaseSettings()
    if settings.database_url is None:
        return settings, None
    return settings, PredictionEventRepository(create_database_engine(settings))


def _required_metadata(metadata: dict[str, Any], field: str) -> str:
    value = metadata.get(field)
    if value is None or not str(value).strip():
        raise PredictionPersistenceError(
            f"Verified deployment metadata is missing {field}"
        )
    return str(value)


def _canonical_features(features: dict[str, Any]) -> dict[str, Any]:
    if set(features) != set(REQUIRED_FIELDS):
        raise PredictionPersistenceError(
            "Prediction features do not match the canonical model contract"
        )
    return {field: features[field] for field in REQUIRED_FIELDS}


def persist_prediction_events(
    *,
    feature_rows: Sequence[dict[str, Any]],
    labels: Sequence[int],
    probabilities: Sequence[float | None],
    prediction_timestamp: datetime,
    metadata: dict[str, Any],
) -> tuple[str, ...]:
    """Persist scored rows atomically, omitting all caller identifiers."""
    if not (len(feature_rows) == len(labels) == len(probabilities)):
        raise PredictionPersistenceError("Prediction persistence inputs are misaligned")
    if not feature_rows:
        return ()

    try:
        settings, repository = _runtime()
    except Exception:
        raise PredictionPersistenceError(
            "Prediction persistence configuration is unavailable"
        ) from None

    if repository is None:
        logger.info(
            "prediction_persistence_skipped environment=%s reason=database_not_configured",
            settings.environment.value,
        )
        return ()

    if prediction_timestamp.tzinfo is None or prediction_timestamp.utcoffset() is None:
        raise PredictionPersistenceError("Prediction timestamp must include a UTC offset")
    timestamp = prediction_timestamp.astimezone(timezone.utc)
    model_version_id = _required_metadata(metadata, "model_version_id")
    feature_schema_version = _required_metadata(metadata, "feature_schema_version")
    deployment_id = _required_metadata(metadata, "deployment_id")

    metadata_environment = metadata.get("environment")
    if metadata_environment and str(metadata_environment) != settings.environment.value:
        raise PredictionPersistenceError(
            "Deployment metadata environment does not match the runtime environment"
        )

    events: list[PendingPredictionEvent] = []
    prediction_ids: list[str] = []
    for features, label, probability in zip(feature_rows, labels, probabilities):
        if probability is None:
            raise PredictionPersistenceError(
                "Prediction probability is required for durable monitoring events"
            )
        try:
            probability_value = float(probability)
            label_value = int(label)
        except (TypeError, ValueError, OverflowError):
            raise PredictionPersistenceError(
                "Prediction outputs are not persistable numeric values"
            ) from None
        if not math.isfinite(probability_value) or not 0 <= probability_value <= 1:
            raise PredictionPersistenceError(
                "Prediction probability must be finite and between zero and one"
            )
        if label_value not in {0, 1}:
            raise PredictionPersistenceError("Predicted class must be binary")
        prediction_id = str(uuid4())
        prediction_ids.append(prediction_id)
        events.append(
            PendingPredictionEvent(
                prediction_id=prediction_id,
                environment=settings.environment.value,
                model_version_id=model_version_id,
                prediction_timestamp=timestamp,
                feature_schema_version=feature_schema_version,
                features=_canonical_features(features),
                prediction_probability=probability_value,
                predicted_class=str(label_value),
                deployment_id=deployment_id,
            )
        )

    try:
        inserted = repository.append(events)
    except Exception:
        raise PredictionPersistenceError(
            "Prediction events could not be committed"
        ) from None
    if inserted != len(events):
        raise PredictionPersistenceError("Not all prediction events were committed")

    logger.info(
        "prediction_events_persisted count=%s deployment_id=%s model_version_id=%s",
        inserted,
        deployment_id,
        model_version_id,
    )
    return tuple(prediction_ids)
