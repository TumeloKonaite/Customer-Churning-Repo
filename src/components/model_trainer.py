"""Compatibility imports for the model-training service.

New code should import from :mod:`src.training`.
"""

from src.training.models import ModelTrainingResult
from src.training.trainer import ModelTrainer

__all__ = ["ModelTrainer", "ModelTrainingResult"]
