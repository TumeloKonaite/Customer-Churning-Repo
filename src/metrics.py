import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.exception import CustomException


def _resolve_k(k, n_samples):
    if isinstance(k, float):
        if k <= 0 or k > 1:
            raise ValueError("k as float must be in (0, 1].")
        return max(1, int(np.ceil(n_samples * k)))
    if isinstance(k, int):
        if k <= 0 or k > n_samples:
            raise ValueError("k as int must be in [1, n_samples].")
        return k
    raise ValueError("k must be float (fraction) or int (count).")


def precision_at_k(y_true, y_score, k=0.1):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    k_resolved = _resolve_k(k, len(y_true))
    top_k_idx = np.argsort(-y_score)[:k_resolved]
    return float(np.mean(y_true[top_k_idx]))


def recall_at_k(y_true, y_score, k=0.1):
    y_true = np.asarray(y_true)
    positives = np.sum(y_true)
    if positives == 0:
        return 0.0
    y_score = np.asarray(y_score)
    k_resolved = _resolve_k(k, len(y_true))
    top_k_idx = np.argsort(-y_score)[:k_resolved]
    return float(np.sum(y_true[top_k_idx]) / positives)


def lift_at_k(y_true, y_score, k=0.1):
    y_true = np.asarray(y_true)
    prevalence = np.mean(y_true)
    if prevalence == 0:
        return 0.0
    return float(precision_at_k(y_true, y_score, k) / prevalence)


def lift_curve(y_true, y_score, k_values=None):
    if k_values is None:
        k_values = [0.01, 0.05, 0.1, 0.2, 0.3]
    curve = []
    for k in k_values:
        curve.append(
            {
                "k": float(k) if isinstance(k, float) else int(k),
                "precision_at_k": precision_at_k(y_true, y_score, k),
                "recall_at_k": recall_at_k(y_true, y_score, k),
                "lift_at_k": lift_at_k(y_true, y_score, k),
            }
        )
    return curve


def compute_classification_metrics(y_true, y_score, threshold=0.5):
    try:
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        y_pred = (y_score >= threshold).astype(int)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "roc_auc": float(roc_auc_score(y_true, y_score)),
            "pr_auc": float(average_precision_score(y_true, y_score)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }
    except Exception as e:
        raise CustomException(e, None)
