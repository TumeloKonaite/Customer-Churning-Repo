"""Approved dataset loading and deterministic cohort creation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.model_schema import (
    CANONICAL_FEATURE_ORDER,
    IDENTIFIER_COLUMNS,
    TARGET_COLUMN,
    reject_prohibited_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class DatasetCohorts:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


class DataIngestion:
    """Load one approved dataset and return raw train/validation/test cohorts."""

    def load(self, dataset_config: dict, split_config: dict) -> DatasetCohorts:
        path = Path(dataset_config["path"])
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"Source dataset not found at {path}")

        frame = pd.read_csv(path)
        required = CANONICAL_FEATURE_ORDER + [TARGET_COLUMN]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"Training dataset is missing required columns: {missing}")

        # The approved source contains legacy identifiers. Drop only those known
        # columns, then reject every unapproved identifier alias before publication.
        frame = frame.drop(
            columns=[column for column in IDENTIFIER_COLUMNS if column in frame.columns]
        )
        reject_prohibited_columns(frame.columns)
        frame = frame[required].copy()

        seed = int(split_config["random_seed"])
        train_validation, test = train_test_split(
            frame,
            test_size=float(split_config["test_size"]),
            random_state=seed,
            stratify=frame[TARGET_COLUMN],
        )
        validation_fraction = float(split_config["validation_size"]) / (
            1.0 - float(split_config["test_size"])
        )
        train, validation = train_test_split(
            train_validation,
            test_size=validation_fraction,
            random_state=seed,
            stratify=train_validation[TARGET_COLUMN],
        )
        return DatasetCohorts(train=train, validation=validation, test=test)
