"""Construction of supported churn-model candidates."""

from __future__ import annotations

from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.training.models import CandidateConfig


class CandidateFactory:
    """Build allow-listed classifiers with the run's deterministic seed."""

    CLASSIFIERS: dict[str, type[ClassifierMixin]] = {
        "logistic_regression": LogisticRegression,
        "decision_tree": DecisionTreeClassifier,
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
    }

    def build(
        self,
        name: str,
        config: CandidateConfig,
        *,
        random_seed: int,
    ) -> tuple[ClassifierMixin, dict[str, Any]]:
        try:
            classifier_type = self.CLASSIFIERS[name]
        except KeyError as exc:
            raise ValueError(f"Unsupported classifier: {name}") from exc
        parameters = {**config.parameters, "random_state": random_seed}
        return classifier_type(**parameters), parameters
