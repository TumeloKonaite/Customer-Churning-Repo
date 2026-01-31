import numpy as np

from src.metrics import (
    compute_classification_metrics,
    lift_at_k,
    lift_curve,
    precision_at_k,
    recall_at_k,
)


def test_compute_classification_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])

    metrics = compute_classification_metrics(y_true=y_true, y_score=y_score)

    assert round(metrics["accuracy"], 2) == 0.75
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert len(metrics["confusion_matrix"]) == 2


def test_k_metrics_and_lift_curve():
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_score = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])

    assert precision_at_k(y_true, y_score, k=0.5) >= 0.5
    assert recall_at_k(y_true, y_score, k=0.5) > 0
    assert lift_at_k(y_true, y_score, k=0.5) > 0

    curve = lift_curve(y_true, y_score, k_values=[0.5])
    assert len(curve) == 1
    assert "lift_at_k" in curve[0]
