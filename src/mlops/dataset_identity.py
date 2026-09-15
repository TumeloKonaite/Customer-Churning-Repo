"""Stable identities for the datasets used by a training run."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd

from src.components.data_ingestion import DatasetCohorts
from src.model_schema import CANONICAL_FEATURE_ORDER, TARGET_COLUMN


def build_dataset_identities(
    cohorts: DatasetCohorts, dataset_config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    identities: dict[str, dict[str, Any]] = {}
    for name, cohort in (
        ("training", cohorts.train),
        ("validation", cohorts.validation),
        ("evaluation", cohorts.test),
    ):
        digest_input = pd.util.hash_pandas_object(cohort, index=True).values.tobytes()
        schema = json.dumps(
            [(str(column), str(dtype)) for column, dtype in cohort.dtypes.items()]
        ).encode()
        identities[name] = {
            "dataset_name": dataset_config.get("name", "unknown"),
            "source_identity": dataset_config.get("source_identity", "unknown"),
            "dataset_digest": hashlib.sha256(schema + digest_input).hexdigest(),
            "row_count": len(cohort),
            "feature_list": list(CANONICAL_FEATURE_ORDER),
            "target_column": TARGET_COLUMN,
        }
    return identities
