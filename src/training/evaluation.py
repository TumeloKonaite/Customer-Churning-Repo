"""Pure model evaluation, selection, and registration eligibility rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.components.data_transformation import ChurnModelPipeline
from src.metrics import compute_classification_metrics
from src.model_schema import CANONICAL_FEATURE_ORDER, TARGET_COLUMN
from src.training.models import CandidateResult, EligibilityConfig, EvaluationResult


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

SELECTION_DIRECTIONS = {
    "roc_auc": "maximize",
    "pr_auc": "maximize",
    "accuracy": "maximize",
    "precision": "maximize",
    "recall": "maximize",
    "f1": "maximize",
    "log_loss": "minimize",
}


class ModelEvaluator:
    def evaluate(
        self,
        pipeline: ChurnModelPipeline,
        cohort: pd.DataFrame,
        threshold: float,
    ) -> EvaluationResult:
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
        return EvaluationResult(metrics=metrics, probabilities=probabilities)


def select_candidate(
    candidates: list[CandidateResult], selection_metric: str
) -> CandidateResult:
    try:
        direction = SELECTION_DIRECTIONS[selection_metric]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported model selection metric: {selection_metric}"
        ) from exc
    if not candidates:
        raise ValueError("At least one fitted model candidate is required")
    missing = [item.name for item in candidates if selection_metric not in item.evaluation.metrics]
    if missing:
        raise ValueError(
            f"Selection metric {selection_metric!r} is missing for candidates: {missing}"
        )
    key = lambda item: item.evaluation.metrics[selection_metric]
    return min(candidates, key=key) if direction == "minimize" else max(candidates, key=key)


def validate_eligibility(
    validation: dict[str, float | int],
    test: dict[str, float | int],
    eligibility: EligibilityConfig,
) -> None:
    for cohort_name, metrics in (("validation", validation), ("test", test)):
        missing = REQUIRED_METRICS - set(metrics)
        if missing:
            raise ValueError(f"{cohort_name} metrics are incomplete: {sorted(missing)}")
    if validation["roc_auc"] < eligibility.minimum_validation_roc_auc:
        raise ValueError("Validation ROC AUC did not meet registration threshold")
    if test["roc_auc"] < eligibility.minimum_test_roc_auc:
        raise ValueError("Test ROC AUC did not meet registration threshold")
