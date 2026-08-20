"""Raw split loading and unfitted preprocessing construction."""

from __future__ import annotations

import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.model_schema import (
    CANONICAL_FEATURE_ORDER,
    CATEGORICAL_COLUMNS,
    IDENTIFIER_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
)


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
            ("imputer", SimpleImputer(strategy="most_frequent")),
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


def build_model_pipeline(classifier) -> Pipeline:
    """Build the single executable training-and-serving artifact."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


class DataTransformation:
    """Load raw train/test frames without fitting any transformation."""

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read raw train and test splits")

            required = CANONICAL_FEATURE_ORDER + [TARGET_COLUMN]
            for split_name, frame in (("training", train_df), ("test", test_df)):
                missing = [column for column in required if column not in frame.columns]
                if missing:
                    raise ValueError(
                        f"Missing required columns in {split_name} split: {missing}"
                    )

            train_df = train_df.drop(
                columns=[column for column in IDENTIFIER_COLUMNS if column in train_df.columns]
            )
            test_df = test_df.drop(
                columns=[column for column in IDENTIFIER_COLUMNS if column in test_df.columns]
            )

            return (
                train_df[CANONICAL_FEATURE_ORDER].copy(),
                train_df[TARGET_COLUMN].copy(),
                test_df[CANONICAL_FEATURE_ORDER].copy(),
                test_df[TARGET_COLUMN].copy(),
            )
        except Exception as exc:
            raise CustomException(exc, sys) from exc
