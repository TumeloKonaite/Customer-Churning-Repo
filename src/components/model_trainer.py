"""Leakage-safe model selection and fitted-pipeline persistence."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from src.components.data_transformation import build_model_pipeline
from src.exception import CustomException
from src.logger import logging
from src.metrics import compute_classification_metrics, lift_curve
from src.model_schema import CATEGORICAL_COLUMNS, build_model_schema
from src.utils import save_object


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
RANDOM_STATE = 42


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join(ARTIFACTS_DIR, "model.pkl")
    schema_file_path: str = os.path.join(ARTIFACTS_DIR, "schema.json")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    @staticmethod
    def _model_candidates():
        return {
            "Logistic Regression": (
                LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
                ),
                {"classifier__C": [0.1, 1.0]},
            ),
            "Decision Tree": (
                DecisionTreeClassifier(
                    class_weight="balanced", random_state=RANDOM_STATE
                ),
                {
                    "classifier__max_depth": [5, 10, None],
                    "classifier__min_samples_split": [2, 5],
                },
            ),
            "Random Forest": (
                RandomForestClassifier(
                    class_weight="balanced", random_state=RANDOM_STATE
                ),
                {
                    "classifier__n_estimators": [100, 200],
                    "classifier__max_depth": [None, 10],
                },
            ),
            "Gradient Boosting": (
                GradientBoostingClassifier(random_state=RANDOM_STATE),
                {
                    "classifier__n_estimators": [100, 200],
                    "classifier__learning_rate": [0.05, 0.1],
                },
            ),
        }

    def _write_schema(self, fitted_pipeline) -> dict:
        preprocessor = fitted_pipeline.named_steps["preprocessor"]
        encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
        known_categories = {
            column: [value.item() if hasattr(value, "item") else value for value in values]
            for column, values in zip(CATEGORICAL_COLUMNS, encoder.categories_)
        }
        schema = build_model_schema(
            known_categories=known_categories,
            transformed_feature_names=list(preprocessor.get_feature_names_out()),
        )
        os.makedirs(os.path.dirname(self.model_trainer_config.schema_file_path), exist_ok=True)
        with open(self.model_trainer_config.schema_file_path, "w", encoding="utf-8") as file:
            json.dump(schema, file, indent=2)
        return schema

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            cv = StratifiedKFold(
                n_splits=3, shuffle=True, random_state=RANDOM_STATE
            )
            validation_report = {}
            fitted_candidates = {}

            # GridSearchCV receives complete pipelines. It therefore refits every
            # imputer, scaler, and encoder independently inside each CV fold.
            for name, (classifier, parameters) in self._model_candidates().items():
                search = GridSearchCV(
                    estimator=build_model_pipeline(classifier),
                    param_grid=parameters,
                    cv=cv,
                    scoring="roc_auc",
                    n_jobs=-1,
                    refit=True,
                )
                search.fit(X_train, y_train)
                validation_report[name] = float(search.best_score_)
                fitted_candidates[name] = search.best_estimator_

            best_model_name = max(validation_report, key=validation_report.get)
            validation_score = validation_report[best_model_name]
            fitted_pipeline = fitted_candidates[best_model_name]
            if validation_score < 0.6:
                raise ValueError("No candidate pipeline met the minimum validation ROC-AUC")

            # The held-out test split is evaluated once, with the selected pipeline
            # unchanged. This exact object is then persisted.
            if hasattr(fitted_pipeline, "predict_proba"):
                y_test_scores = fitted_pipeline.predict_proba(X_test)[:, 1]
                threshold = 0.5
            elif hasattr(fitted_pipeline, "decision_function"):
                y_test_scores = fitted_pipeline.decision_function(X_test)
                threshold = 0.0
            else:
                y_test_scores = fitted_pipeline.predict(X_test)
                threshold = 0.5

            metrics = compute_classification_metrics(
                y_true=y_test, y_score=y_test_scores, threshold=threshold
            )
            metrics["lift_curve"] = lift_curve(y_test, y_test_scores)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=fitted_pipeline,
            )
            schema = self._write_schema(fitted_pipeline)
            logging.info(
                "Selected %s pipeline with validation ROC-AUC %.3f and test ROC-AUC %.3f",
                best_model_name,
                validation_score,
                metrics["roc_auc"],
            )

            return {
                "best_model_name": best_model_name,
                "best_model_score": float(metrics["roc_auc"]),
                "validation_score": validation_score,
                "validation_scores": validation_report,
                "metrics": metrics,
                "model_path": self.model_trainer_config.trained_model_file_path,
                "schema_path": self.model_trainer_config.schema_file_path,
                "schema_version": schema["schema_version"],
            }
        except Exception as exc:
            raise CustomException(exc, sys) from exc
