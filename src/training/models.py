"""Typed configuration and results for model training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.components.data_transformation import ChurnModelPipeline


class CandidateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parameters: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selection_metric: str = "roc_auc"
    classification_threshold: float = Field(ge=0, le=1)
    candidates: dict[str, CandidateConfig]


class EligibilityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum_validation_roc_auc: float = Field(ge=0, le=1)
    minimum_test_roc_auc: float = Field(ge=0, le=1)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    metrics: dict[str, float | int]
    probabilities: np.ndarray


@dataclass(frozen=True, slots=True)
class CandidateResult:
    name: str
    pipeline: ChurnModelPipeline
    parameters: dict[str, Any]
    evaluation: EvaluationResult


@dataclass(frozen=True, slots=True)
class FittedModelResult:
    pipeline: ChurnModelPipeline
    model_name: str
    model_parameters: dict[str, Any]
    threshold: float
    validation_metrics: dict[str, float | int]
    test_metrics: dict[str, float | int]
    candidate_metrics: dict[str, dict[str, float | int]]
    test_probabilities: np.ndarray
    schema: dict[str, Any]

    def published_to(self, artifact_dir: Path) -> "ModelTrainingResult":
        return ModelTrainingResult(
            pipeline=self.pipeline,
            model_name=self.model_name,
            model_parameters=self.model_parameters,
            threshold=self.threshold,
            validation_metrics=self.validation_metrics,
            test_metrics=self.test_metrics,
            candidate_metrics=self.candidate_metrics,
            test_probabilities=self.test_probabilities,
            schema=self.schema,
            artifact_dir=artifact_dir,
        )


@dataclass(frozen=True, slots=True)
class ModelTrainingResult(FittedModelResult):
    artifact_dir: Path
