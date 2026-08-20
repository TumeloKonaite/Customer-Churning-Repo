"""Orchestrate local training through the existing application components."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from src.components.data_ingestion import DataIngestion
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.mlops.tracking import ExperimentTracker, TrackingResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_CONFIG_SECTIONS = {
    "version",
    "dataset",
    "split",
    "model",
    "eligibility",
    "contracts",
}


class TrainingPipeline:
    def __init__(
        self,
        ingestion: DataIngestion | None = None,
        trainer: ModelTrainer | None = None,
        tracker: ExperimentTracker | None = None,
    ):
        self.ingestion = ingestion or DataIngestion()
        self.trainer = trainer or ModelTrainer()
        self.tracker = tracker

    def run(
        self,
        config_path: str | Path = "configs/training.yaml",
    ) -> TrackingResult:
        config = self._load_config(config_path)
        seed = int(config["split"]["random_seed"])
        np.random.seed(seed)

        logging.info("Loading approved training cohorts")
        cohorts = self.ingestion.load(config["dataset"], config["split"])
        logging.info("Fitting and evaluating the unified model pipeline")
        training = self.trainer.train(
            cohorts,
            config["model"],
            config["eligibility"],
            random_seed=seed,
            output_dir=config.get("output_dir", "artifacts/training"),
            training_config=config,
        )
        logging.info("Local training artifacts saved to %s", training.artifact_dir)
        tracker = self.tracker or ExperimentTracker()
        result = tracker.track(training, config)
        logging.info("Training complete: tracking_status=%s", result.status)
        return result

    @staticmethod
    def _load_config(path: str | Path) -> dict:
        config_path = Path(path)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        with config_path.open(encoding="utf-8") as file:
            config = yaml.safe_load(file)
        missing = sorted(REQUIRED_CONFIG_SECTIONS - set(config or {}))
        if missing:
            raise ValueError(f"Training configuration is missing sections: {missing}")
        return config
