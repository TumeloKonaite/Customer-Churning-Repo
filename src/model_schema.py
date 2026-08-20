"""Canonical, versioned raw-input contract for the churn model."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


MODEL_SCHEMA_VERSION = "1.0.0"
TARGET_COLUMN = "Exited"
IDENTIFIER_COLUMNS = ["RowNumber", "CustomerId", "Surname"]

# This order is the public contract order. The fitted ColumnTransformer selects by
# name, so predictions do not depend on the order of incoming DataFrame columns.
CANONICAL_FEATURE_ORDER = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]
NUMERIC_COLUMNS = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]
CATEGORICAL_COLUMNS = ["Geography", "Gender"]

_INTEGER_FEATURES = {
    "CreditScore",
    "Age",
    "Tenure",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
}


def _feature_definition(name: str) -> dict[str, Any]:
    is_numeric = name in NUMERIC_COLUMNS
    data_type = (
        "integer"
        if name in _INTEGER_FEATURES
        else "number"
        if is_numeric
        else "string"
    )
    definition: dict[str, Any] = {
        "name": name,
        "data_type": data_type,
        "required": True,
        "nullable": False,
        "feature_type": "numeric" if is_numeric else "categorical",
        "transformation_owner": (
            "pipeline.preprocessor.numeric.imputer_and_scaler"
            if is_numeric
            else "pipeline.preprocessor.categorical.imputer_and_encoder"
        ),
    }
    if not is_numeric:
        # Any non-empty string is valid at this schema version. Values not observed
        # during fitting are intentionally encoded as all-zero indicator columns.
        definition["accepted_values"] = None
        definition["unseen_value_policy"] = "OneHotEncoder(handle_unknown='ignore')"
    return definition


CANONICAL_MODEL_SCHEMA: dict[str, Any] = {
    "schema_version": MODEL_SCHEMA_VERSION,
    "target": TARGET_COLUMN,
    "canonical_feature_order": CANONICAL_FEATURE_ORDER,
    "numeric_columns": NUMERIC_COLUMNS,
    "categorical_columns": CATEGORICAL_COLUMNS,
    "features": [_feature_definition(name) for name in CANONICAL_FEATURE_ORDER],
    "missing_value_policy": {
        "api": "required, non-nullable fields are rejected before prediction",
        "batch": "required, non-nullable fields are rejected before prediction",
        "pipeline": "approved missing values are imputed inside the fitted pipeline",
    },
    "categorical_encoding": {
        "type": "one_hot",
        "handle_unknown": "ignore",
    },
}


def build_model_schema(
    *,
    known_categories: dict[str, list[Any]] | None = None,
    transformed_feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Return a serializable schema enriched with training-only fitted details."""
    schema = deepcopy(CANONICAL_MODEL_SCHEMA)
    known_categories = known_categories or {}
    for feature in schema["features"]:
        if feature["feature_type"] == "categorical":
            feature["known_training_values"] = known_categories.get(feature["name"], [])
    schema["transformed_feature_names"] = transformed_feature_names or []
    return schema
