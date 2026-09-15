"""Staged publication of local training artifacts and reference datasets."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import shutil
import tempfile
from typing import Any
from uuid import uuid4

from sklearn.metrics import classification_report, confusion_matrix

from src.components.data_ingestion import DatasetCohorts
from src.mlops.dataset_identity import build_dataset_identities
from src.model_schema import CANONICAL_FEATURE_ORDER, TARGET_COLUMN
from src.training.models import FittedModelResult, ModelTrainingResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACTS_DIR = PROJECT_ROOT / "configs" / "contracts"


class TrainingArtifactWriter:
    """Build a complete bundle before replacing the previously published bundle."""

    def __init__(self, contracts_dir: Path = DEFAULT_CONTRACTS_DIR):
        self.contracts_dir = contracts_dir

    def write(
        self,
        result: FittedModelResult,
        *,
        cohorts: DatasetCohorts,
        config: dict[str, Any],
        output_dir: str | Path = "artifacts/training",
    ) -> ModelTrainingResult:
        output = Path(output_dir)
        if not output.is_absolute():
            output = PROJECT_ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{output.name}-staging-", dir=output.parent)
        )
        try:
            self._write_bundle(staging, result, cohorts, config)
            self._replace_bundle(staging, output)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return result.published_to(output)

    def _write_bundle(
        self,
        output: Path,
        result: FittedModelResult,
        cohorts: DatasetCohorts,
        config: dict[str, Any],
    ) -> None:
        for directory in ("contracts", "evaluation", "lineage", "references"):
            (output / directory).mkdir(parents=True, exist_ok=True)

        with (output / "model.pkl").open("wb") as file:
            pickle.dump(result.pipeline, file)
        self._write_json(output / "contracts" / "feature_schema.json", result.schema)
        for contract in self.contracts_dir.glob("*.json"):
            shutil.copy2(contract, output / "contracts" / contract.name)

        target = cohorts.test[TARGET_COLUMN].to_numpy()
        predicted = (result.test_probabilities >= result.threshold).astype(int)
        self._write_json(
            output / "evaluation" / "metrics.json",
            {
                "selected_model": result.model_name,
                "validation": result.validation_metrics,
                "test": result.test_metrics,
            },
        )
        self._write_json(
            output / "evaluation" / "confusion_matrix.json",
            {
                "labels": [0, 1],
                "matrix": confusion_matrix(target, predicted, labels=[0, 1]).tolist(),
            },
        )
        self._write_json(
            output / "evaluation" / "model_comparison.json",
            result.candidate_metrics,
        )
        self._write_json(
            output / "evaluation" / "classification_report.json",
            classification_report(
                target, predicted, output_dict=True, zero_division=0
            ),
        )

        source = config.get("dataset", {})
        self._write_json(
            output / "lineage" / "dataset_identities.json",
            build_dataset_identities(cohorts, source),
        )
        self._write_json(
            output / "lineage" / "training_config.json", config, sort_keys=True
        )
        self._write_references(output, cohorts, source)

    @staticmethod
    def _write_references(
        output: Path,
        cohorts: DatasetCohorts,
        source: dict[str, Any],
    ) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        for purpose, frame in (
            ("drift_reference", cohorts.validation[CANONICAL_FEATURE_ORDER]),
            (
                "evaluation_reference",
                cohorts.test[CANONICAL_FEATURE_ORDER + [TARGET_COLUMN]],
            ),
        ):
            frame.to_parquet(output / "references" / f"{purpose}.parquet", index=False)
            TrainingArtifactWriter._write_json(
                output / "references" / f"{purpose}_metadata.json",
                {
                    "dataset_name": source.get("name", "unknown"),
                    "dataset_purpose": purpose,
                    "source_identity": source.get("source_identity", "unknown"),
                    "row_count": len(frame),
                    "feature_list": list(CANONICAL_FEATURE_ORDER),
                    "target_column": (
                        TARGET_COLUMN if purpose == "evaluation_reference" else None
                    ),
                    "creation_timestamp_utc": created_at,
                },
            )

    @staticmethod
    def _write_json(path: Path, value: Any, *, sort_keys: bool = False) -> None:
        path.write_text(
            json.dumps(value, indent=2, sort_keys=sort_keys), encoding="utf-8"
        )

    @staticmethod
    def _replace_bundle(staging: Path, output: Path) -> None:
        backup = output.with_name(f".{output.name}-backup-{uuid4().hex}")
        had_previous = output.exists()
        if had_previous:
            output.replace(backup)
        try:
            staging.replace(output)
        except Exception:
            if had_previous and backup.exists() and not output.exists():
                backup.replace(output)
            raise
        if had_previous:
            shutil.rmtree(backup, ignore_errors=True)
