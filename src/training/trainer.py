"""Fit, evaluate, and select the production churn pipeline."""

from __future__ import annotations

from typing import Any

from src.components.data_ingestion import DatasetCohorts
from src.components.data_transformation import build_model_pipeline
from src.model_schema import CANONICAL_FEATURE_ORDER, TARGET_COLUMN
from src.training.candidates import CandidateFactory
from src.training.evaluation import ModelEvaluator, select_candidate, validate_eligibility
from src.training.models import (
    CandidateResult,
    EligibilityConfig,
    FittedModelResult,
    ModelConfig,
)
from src.training.schema import build_fitted_model_schema


class ModelTrainer:
    """Pure fitting service; artifact persistence is handled by the pipeline."""

    CLASSIFIERS = CandidateFactory.CLASSIFIERS

    def __init__(
        self,
        candidate_factory: CandidateFactory | None = None,
        evaluator: ModelEvaluator | None = None,
    ):
        self.candidate_factory = candidate_factory or CandidateFactory()
        self.evaluator = evaluator or ModelEvaluator()

    def fit(
        self,
        cohorts: DatasetCohorts,
        model_config: ModelConfig | dict[str, Any],
        eligibility: EligibilityConfig | dict[str, Any],
        *,
        random_seed: int,
    ) -> FittedModelResult:
        config = (
            model_config
            if isinstance(model_config, ModelConfig)
            else ModelConfig.model_validate(model_config)
        )
        eligibility_config = (
            eligibility
            if isinstance(eligibility, EligibilityConfig)
            else EligibilityConfig.model_validate(eligibility)
        )
        if not config.candidates:
            raise ValueError("At least one model candidate must be configured")

        fitted: list[CandidateResult] = []
        for name, candidate_config in config.candidates.items():
            classifier, parameters = self.candidate_factory.build(
                name, candidate_config, random_seed=random_seed
            )
            pipeline = build_model_pipeline(classifier)
            pipeline.fit(
                cohorts.train[CANONICAL_FEATURE_ORDER],
                cohorts.train[TARGET_COLUMN],
            )
            fitted.append(
                CandidateResult(
                    name=name,
                    pipeline=pipeline,
                    parameters=parameters,
                    evaluation=self.evaluator.evaluate(
                        pipeline, cohorts.validation, config.classification_threshold
                    ),
                )
            )

        selected = select_candidate(fitted, config.selection_metric)
        test = self.evaluator.evaluate(
            selected.pipeline, cohorts.test, config.classification_threshold
        )
        validate_eligibility(selected.evaluation.metrics, test.metrics, eligibility_config)
        return FittedModelResult(
            pipeline=selected.pipeline,
            model_name=selected.name,
            model_parameters=selected.parameters,
            threshold=config.classification_threshold,
            validation_metrics=selected.evaluation.metrics,
            test_metrics=test.metrics,
            candidate_metrics={item.name: item.evaluation.metrics for item in fitted},
            test_probabilities=test.probabilities,
            schema=build_fitted_model_schema(selected.pipeline),
        )
