"""Unfitted preprocessing and unified model-pipeline construction."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.model_schema import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
)


class ChurnModelPipeline(Pipeline):
    """Unified sklearn pipeline with a named MLflow pyfunc output contract."""

    def predict_classes(self, features: pd.DataFrame):
        """Return the classifier labels used by the application adapter."""
        return super().predict(features)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return the stable named output used by the MLflow pyfunc flavor."""
        labels = self.predict_classes(features)
        if not hasattr(self, "predict_proba"):
            raise ValueError("Production churn pipeline must provide class probabilities")
        probability_matrix = self.predict_proba(features)
        positive_index = list(self.classes_).index(1)
        return pd.DataFrame(
            {
                "predicted_class": labels.astype("int64"),
                "churn_probability": probability_matrix[:, positive_index].astype("float64"),
            }
        )

    def predict_outputs(self, features: pd.DataFrame) -> pd.DataFrame:
        """Explicit alias retained for contract-oriented callers."""
        return self.predict(features)


def build_preprocessor() -> ColumnTransformer:
    """Build a fresh preprocessor that is fitted only as part of a model pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent", missing_values=None)),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model_pipeline(classifier) -> ChurnModelPipeline:
    """Build the single executable training-and-serving artifact."""
    return ChurnModelPipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )
