"""Inference adapter around the exact fitted training pipeline."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mlflow.sklearn
import pandas as pd

from src.exception import CustomException
from src.model_schema import CANONICAL_FEATURE_ORDER, MODEL_SCHEMA_VERSION
from src.utils import load_object


class PredictPipeline:
    """Load one fitted pipeline and use it unchanged for all prediction methods."""

    def __init__(self, artifacts_dir: str | None = None):
        project_root = Path(__file__).resolve().parents[2]
        configured_package = os.getenv("DEPLOYMENT_PACKAGE_DIR")
        default_package = project_root / "build" / "model"
        package_dir = Path(configured_package) if configured_package else default_package
        package_model = package_dir / "model" / "MLmodel"
        if artifacts_dir is None and package_model.is_file():
            self.model_path = str(package_dir / "model")
            self.schema_path = str(package_dir / "feature_schema.json")
            loader = mlflow.sklearn.load_model
        else:
            if artifacts_dir is None and os.getenv("APP_ENV", "development") == "production":
                raise ValueError("Verified deployment package is required in production")
            legacy_dir = Path(artifacts_dir) if artifacts_dir else project_root / "artifacts"
            self.model_path = str(legacy_dir / "model.pkl")
            self.schema_path = str(legacy_dir / "schema.json")
            loader = lambda path: load_object(file_path=path)

        with open(self.schema_path, encoding="utf-8") as file:
            self.schema = json.load(file)
        schema_version = self.schema.get("schema_version")
        if schema_version != MODEL_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported model schema version {schema_version!r}; "
                f"expected {MODEL_SCHEMA_VERSION!r}"
            )
        self.input_columns = self.schema.get(
            "canonical_feature_order", CANONICAL_FEATURE_ORDER
        )
        self.pipeline = loader(self.model_path)

    def predict(self, features: pd.DataFrame):
        try:
            if not isinstance(features, pd.DataFrame):
                raise TypeError("Prediction features must be a pandas DataFrame")
            missing = [column for column in self.input_columns if column not in features.columns]
            unexpected = [column for column in features.columns if column not in self.input_columns]
            if missing:
                raise ValueError(f"Missing required model columns: {missing}")
            if unexpected:
                raise ValueError(f"Unexpected model columns: {unexpected}")

            # ColumnTransformer owns name-based selection, imputation, scaling, and
            # encoding. No serving-side transformation or column reordering occurs.
            if hasattr(self.pipeline, "predict_classes"):
                labels = self.pipeline.predict_classes(features)
            else:
                labels = self.pipeline.predict(features)
            probabilities = None
            if hasattr(self.pipeline, "predict_proba"):
                probability_matrix = self.pipeline.predict_proba(features)
                classes = list(self.pipeline.classes_)
                probabilities = probability_matrix[:, classes.index(1)]
            return labels, probabilities
        except Exception as exc:
            raise CustomException(exc, sys) from exc
