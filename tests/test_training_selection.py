from __future__ import annotations

import numpy as np
import pytest

from src.training.evaluation import select_candidate
from src.training.models import CandidateResult, EvaluationResult


def _candidate(name: str, **metrics: float) -> CandidateResult:
    return CandidateResult(
        name=name,
        pipeline=object(),
        parameters={},
        evaluation=EvaluationResult(metrics=metrics, probabilities=np.array([])),
    )


def test_selection_maximizes_score_metrics():
    selected = select_candidate(
        [_candidate("first", roc_auc=0.7), _candidate("second", roc_auc=0.8)],
        "roc_auc",
    )

    assert selected.name == "second"


def test_selection_minimizes_log_loss():
    selected = select_candidate(
        [_candidate("first", log_loss=0.6), _candidate("second", log_loss=0.4)],
        "log_loss",
    )

    assert selected.name == "second"


def test_selection_rejects_metrics_without_an_explicit_direction():
    with pytest.raises(ValueError, match="Unsupported model selection metric"):
        select_candidate([_candidate("first", row_count=100)], "row_count")
