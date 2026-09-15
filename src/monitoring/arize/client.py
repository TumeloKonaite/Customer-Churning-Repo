"""Small adapter around the Arize Python v8 batch API."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import ArizeExportSettings
from .schema import ACTUAL_TAGS, ARIZE_FEATURES, BASELINE_TAGS, PREDICTION_TAGS


class ArizeDeliveryError(RuntimeError):
    retryable = True


class ArizePermanentError(ArizeDeliveryError):
    retryable = False


class ArizeV8Client:
    def __init__(self, settings: ArizeExportSettings):
        from arize import ArizeClient
        from arize.ml.types import Environments, ModelTypes, Schema

        self._settings = settings
        self._client = ArizeClient(api_key=settings.api_key.get_secret_value())
        self._environment = {
            "production": Environments.PRODUCTION,
            "staging": Environments.PRODUCTION,
            "development": Environments.VALIDATION,
            "test": Environments.VALIDATION,
        }[settings.environment.value]
        self._model_type = ModelTypes.BINARY_CLASSIFICATION
        self._Schema = Schema
        self._validation_environment = Environments.VALIDATION

    def log_predictions(self, frame: pd.DataFrame, *, model_version: str) -> None:
        schema = self._Schema(
            prediction_id_column_name="prediction_id",
            timestamp_column_name="prediction_timestamp",
            prediction_label_column_name="prediction_label",
            prediction_score_column_name="prediction_score",
            feature_column_names=list(ARIZE_FEATURES),
            tag_column_names=list(PREDICTION_TAGS),
        )
        self._log(frame, schema=schema, model_version=model_version)

    def log_actuals(self, frame: pd.DataFrame, *, model_version: str) -> None:
        schema = self._Schema(
            prediction_id_column_name="prediction_id",
            timestamp_column_name="prediction_timestamp",
            actual_label_column_name="actual_label",
            tag_column_names=list(ACTUAL_TAGS),
        )
        self._log(frame, schema=schema, model_version=model_version)

    def log_baseline(
        self, frame: pd.DataFrame, *, model_version: str, batch_id: str
    ) -> None:
        schema = self._Schema(
            prediction_id_column_name="prediction_id",
            timestamp_column_name="prediction_timestamp",
            prediction_label_column_name="prediction_label",
            prediction_score_column_name="prediction_score",
            actual_label_column_name="actual_label",
            feature_column_names=list(ARIZE_FEATURES),
            tag_column_names=list(BASELINE_TAGS),
        )
        self._log(
            frame, schema=schema, model_version=model_version,
            environment=self._validation_environment, batch_id=batch_id,
        )

    def _log(
        self, frame: pd.DataFrame, *, schema: Any, model_version: str,
        environment: Any | None = None, batch_id: str = "",
    ) -> None:
        try:
            response = self._client.ml.log(
                space_id=self._settings.space_id.get_secret_value(),
                model_name=self._settings.model_name,
                model_type=self._model_type,
                environment=environment or self._environment,
                dataframe=frame,
                schema=schema,
                model_version=model_version,
                batch_id=batch_id,
            )
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            error = ArizePermanentError if status in {400, 401, 403, 404, 422} else ArizeDeliveryError
            raise error("Arize request was rejected" if status else "Arize request failed") from None
        status = getattr(response, "status_code", None)
        acknowledged = (
            (isinstance(status, int) and 200 <= status < 300)
            or getattr(response, "success", False) is True
        )
        if not acknowledged:
            error = ArizePermanentError if status in {400, 401, 403, 404, 422} else ArizeDeliveryError
            raise error("Arize did not acknowledge the batch")
