"""Fit and evaluate the single pipeline used by training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import shutil

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

from src.components.data_ingestion import DatasetCohorts
from src.components.data_transformation import ChurnModelPipeline, build_model_pipeline
from src.metrics import compute_classification_metrics
from src.model_schema import (
    CANONICAL_FEATURE_ORDER,
    CATEGORICAL_COLUMNS,
    TARGET_COLUMN,
    build_model_schema,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACTS_DIR = PROJECT_ROOT / "configs" / "contracts"
REQUIRED_METRICS = {
    "roc_auc",
    "pr_auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "log_loss",
    "confusion_matrix/tn",
    "confusion_matrix/fp",
    "confusion_matrix/fn",
    "confusion_matrix/tp",
    "row_count",
    "positive_class_prevalence",
}


@dataclass(frozen=True, slots=True)
class ModelTrainingResult:
    pipeline: ChurnModelPipeline
    model_name: str
    model_parameters: dict
    threshold: float
    validation_metrics: dict
    test_metrics: dict
    candidate_metrics: dict[str, dict]
    test_probabilities: np.ndarray
    schema: dict
    artifact_dir: Path


class ModelTrainer:
    """Fit configured candidates and return the best validation pipeline."""

    CLASSIFIERS = {
        "logistic_regression": LogisticRegression,
        "decision_tree": DecisionTreeClassifier,
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
    }

    def train(
        self,
        cohorts: DatasetCohorts,
        model_config: dict,
        eligibility: dict,
        *,
        random_seed: int,
        output_dir: str | Path = "artifacts/training",
        training_config: dict | None = None,
    ) -> ModelTrainingResult:
        threshold = float(model_config["classification_threshold"])
        selection_metric = model_config.get("selection_metric", "roc_auc")
        candidates = model_config.get("candidates", {})
        if not candidates:
            raise ValueError("At least one model candidate must be configured")

        fitted_candidates = {}
        candidate_metrics = {}
        candidate_parameters = {}
        for model_name, candidate_config in candidates.items():
            try:
                classifier_type = self.CLASSIFIERS[model_name]
            except KeyError as exc:
                raise ValueError(f"Unsupported classifier: {model_name}") from exc
            parameters = {
                **candidate_config.get("parameters", {}),
                "random_state": random_seed,
            }
            pipeline = build_model_pipeline(classifier_type(**parameters))
            pipeline.fit(
                cohorts.train[CANONICAL_FEATURE_ORDER],
                cohorts.train[TARGET_COLUMN],
            )
            metrics, _ = self._evaluate(pipeline, cohorts.validation, threshold)
            if selection_metric not in metrics:
                raise ValueError(f"Unsupported model selection metric: {selection_metric}")
            fitted_candidates[model_name] = pipeline
            candidate_metrics[model_name] = metrics
            candidate_parameters[model_name] = parameters

        model_name = max(
            candidate_metrics,
            key=lambda name: candidate_metrics[name][selection_metric],
        )
        pipeline = fitted_candidates[model_name]
        parameters = candidate_parameters[model_name]
        validation_metrics = candidate_metrics[model_name]
        test_metrics, test_probabilities = self._evaluate(
            pipeline, cohorts.test, threshold
        )
        self._check_eligibility(validation_metrics, test_metrics, eligibility)

        preprocessor = pipeline.named_steps["preprocessor"]
        encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
        categories = {
            column: [item.item() if hasattr(item, "item") else item for item in values]
            for column, values in zip(CATEGORICAL_COLUMNS, encoder.categories_)
        }
        schema = build_model_schema(
            known_categories=categories,
            transformed_feature_names=list(preprocessor.get_feature_names_out()),
        )
        artifact_dir = Path(output_dir)
        if not artifact_dir.is_absolute():
            artifact_dir = PROJECT_ROOT / artifact_dir
        result = ModelTrainingResult(
            pipeline=pipeline,
            model_name=model_name,
            model_parameters=parameters,
            threshold=threshold,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            candidate_metrics=candidate_metrics,
            test_probabilities=test_probabilities,
            schema=schema,
            artifact_dir=artifact_dir,
        )
        self._save_artifacts(result, cohorts, training_config or {})
        return result

    @staticmethod
    def _evaluate(
        pipeline: ChurnModelPipeline, cohort: pd.DataFrame, threshold: float
    ) -> tuple[dict, np.ndarray]:
        target = cohort[TARGET_COLUMN].to_numpy()
        probabilities = pipeline.predict_proba(cohort[CANONICAL_FEATURE_ORDER])[
            :, list(pipeline.classes_).index(1)
        ]
        metrics = compute_classification_metrics(target, probabilities, threshold)
        matrix = metrics.pop("confusion_matrix")
        metrics.update(
            {
                "log_loss": float(log_loss(target, probabilities, labels=[0, 1])),
                "confusion_matrix/tn": int(matrix[0][0]),
                "confusion_matrix/fp": int(matrix[0][1]),
                "confusion_matrix/fn": int(matrix[1][0]),
                "confusion_matrix/tp": int(matrix[1][1]),
                "row_count": len(cohort),
                "positive_class_prevalence": float(np.mean(target)),
            }
        )
        return metrics, probabilities

    @staticmethod
    def _check_eligibility(validation: dict, test: dict, eligibility: dict) -> None:
        for cohort_name, metrics in (("validation", validation), ("test", test)):
            missing = REQUIRED_METRICS - set(metrics)
            if missing:
                raise ValueError(
                    f"{cohort_name} metrics are incomplete: {sorted(missing)}"
                )
        if validation["roc_auc"] < eligibility["minimum_validation_roc_auc"]:
            raise ValueError("Validation ROC AUC did not meet registration threshold")
        if test["roc_auc"] < eligibility["minimum_test_roc_auc"]:
            raise ValueError("Test ROC AUC did not meet registration threshold")

    @staticmethod
    def _save_artifacts(
        result: ModelTrainingResult,
        cohorts: DatasetCohorts,
        config: dict,
    ) -> None:
        """Persist a complete local result before optional remote tracking."""
        output = result.artifact_dir
        for directory in ("contracts", "evaluation", "lineage", "references"):
            (output / directory).mkdir(parents=True, exist_ok=True)

        with (output / "model.pkl").open("wb") as file:
            pickle.dump(result.pipeline, file)
        (output / "contracts" / "feature_schema.json").write_text(
            json.dumps(result.schema, indent=2), encoding="utf-8"
        )
        for contract in CONTRACTS_DIR.glob("*.json"):
            shutil.copy2(contract, output / "contracts" / contract.name)

        metrics = {
            "selected_model": result.model_name,
            "validation": result.validation_metrics,
            "test": result.test_metrics,
        }
        (output / "evaluation" / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        target = cohorts.test[TARGET_COLUMN].to_numpy()
        predicted = (result.test_probabilities >= result.threshold).astype(int)
        (output / "evaluation" / "confusion_matrix.json").write_text(
            json.dumps(
                {
                    "labels": [0, 1],
                    "matrix": confusion_matrix(target, predicted, labels=[0, 1]).tolist(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (output / "evaluation" / "model_comparison.json").write_text(
            json.dumps(result.candidate_metrics, indent=2), encoding="utf-8"
        )
        (output / "evaluation" / "classification_report.json").write_text(
            json.dumps(
                classification_report(
                    target, predicted, output_dict=True, zero_division=0
                ),
                indent=2,
            ),
            encoding="utf-8",
        )

        source = config.get("dataset", {})
        identities = {}
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
                "dataset_name": source.get("name", "unknown"),
                "source_identity": source.get("source_identity", "unknown"),
                "dataset_digest": hashlib.sha256(schema + digest_input).hexdigest(),
                "row_count": len(cohort),
                "feature_list": list(CANONICAL_FEATURE_ORDER),
                "target_column": TARGET_COLUMN,
            }
        (output / "lineage" / "dataset_identities.json").write_text(
            json.dumps(identities, indent=2), encoding="utf-8"
        )
        (output / "lineage" / "training_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
        )

        created_at = datetime.now(timezone.utc).isoformat()
        for purpose, frame in (
            ("drift_reference", cohorts.validation[CANONICAL_FEATURE_ORDER]),
            (
                "evaluation_reference",
                cohorts.test[CANONICAL_FEATURE_ORDER + [TARGET_COLUMN]],
            ),
        ):
            frame.to_parquet(output / "references" / f"{purpose}.parquet", index=False)
            metadata = {
                "dataset_name": source.get("name", "unknown"),
                "dataset_purpose": purpose,
                "source_identity": source.get("source_identity", "unknown"),
                "row_count": len(frame),
                "feature_list": list(CANONICAL_FEATURE_ORDER),
                "target_column": (
                    TARGET_COLUMN if purpose == "evaluation_reference" else None
                ),
                "creation_timestamp_utc": created_at,
            }
            (output / "references" / f"{purpose}_metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
