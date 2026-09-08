"""Canonical churn-to-Arize mapping and privacy allowlists."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Mapping, Sequence

import pandas as pd

from src.model_schema import CANONICAL_FEATURE_ORDER, MODEL_SCHEMA_VERSION


ARIZE_FEATURES = tuple(CANONICAL_FEATURE_ORDER)
PREDICTION_TAGS = (
    "model_version_id", "deployment_id", "feature_schema_version",
    "request_source", "batch_id",
)
ACTUAL_TAGS = (
    "model_version_id", "deployment_id", "feature_schema_version",
    "label_contract_version",
)
BASELINE_TAGS = ("model_version_id", "baseline_version_id")
ARIZE_TAGS = tuple(
    dict.fromkeys((*PREDICTION_TAGS, *ACTUAL_TAGS, *BASELINE_TAGS))
)
LABELS = {0: "no_churn", 1: "churn", "0": "no_churn", "1": "churn"}
_VERSION = re.compile(r":churn_predictor:(\d+)$")


def numeric_model_version(model_version_id: str) -> str:
    match = _VERSION.search(model_version_id)
    if not match or int(match.group(1)) < 1:
        raise ValueError("model_version_id does not identify an exact churn_predictor version")
    return match.group(1)


def binary_label(value: Any) -> str:
    try:
        return LABELS[value]
    except (KeyError, TypeError):
        raise ValueError("classification label must be binary") from None


def utc_epoch(value: datetime) -> int:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Arize timestamps must be timezone-aware")
    return int(value.astimezone(timezone.utc).timestamp())


def _features(value: Mapping[str, Any], schema_version: str) -> dict[str, Any]:
    if schema_version != MODEL_SCHEMA_VERSION:
        raise ValueError("prediction feature schema version is incompatible")
    if set(value) != set(ARIZE_FEATURES):
        raise ValueError("prediction must contain exactly the ten approved features")
    return {name: value[name] for name in ARIZE_FEATURES}


def prediction_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    records = []
    for row in rows:
        score = float(row["prediction_probability"])
        if not 0 <= score <= 1:
            raise ValueError("prediction score must be between zero and one")
        record = {
            "prediction_id": str(row["prediction_id"]),
            "prediction_timestamp": utc_epoch(row["prediction_timestamp"]),
            **_features(row["features"], row["feature_schema_version"]),
            "prediction_label": binary_label(row["predicted_class"]),
            "prediction_score": score,
            **{name: row.get(name) for name in PREDICTION_TAGS},
        }
        records.append(record)
    return pd.DataFrame.from_records(records).reset_index(drop=True)


def actual_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    records = []
    for row in rows:
        records.append(
            {
                "prediction_id": str(row["prediction_id"]),
                "prediction_timestamp": utc_epoch(row["label_created_at"]),
                "actual_label": binary_label(row["label_value"]),
                "model_version_id": row["model_version_id"],
                "deployment_id": row.get("deployment_id"),
                "feature_schema_version": row["feature_schema_version"],
                "label_contract_version": row["label_contract_version"],
            }
        )
    return pd.DataFrame.from_records(records).reset_index(drop=True)
