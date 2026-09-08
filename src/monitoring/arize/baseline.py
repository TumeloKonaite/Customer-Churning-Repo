"""Verification and upload of an existing approved reference dataset."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import Engine, text

from src.model_schema import CANONICAL_FEATURE_ORDER, MODEL_SCHEMA_VERSION, TARGET_COLUMN
from src.monitoring.shared.artifacts import ArtifactStore

from .client import ArizeV8Client
from .config import ArizeExportSettings
from .repository import ArizeOutboxRepository
from .schema import binary_label, numeric_model_version


def validate_reference(body: bytes, *, expected_sha256: str, schema_version: str) -> pd.DataFrame:
    if not body or sha256(body).hexdigest() != expected_sha256:
        raise ValueError("approved baseline checksum mismatch")
    if schema_version != MODEL_SCHEMA_VERSION:
        raise ValueError("approved baseline feature schema is incompatible")
    frame = pd.read_parquet(BytesIO(body))
    allowed = set(CANONICAL_FEATURE_ORDER) | {TARGET_COLUMN}
    if frame.empty:
        raise ValueError("approved baseline is empty")
    if not set(CANONICAL_FEATURE_ORDER).issubset(frame.columns):
        raise ValueError("approved baseline is missing canonical features")
    if set(frame.columns) - allowed:
        raise ValueError("approved baseline contains non-allowlisted fields")
    return frame


def upload_baseline(
    *, engine: Engine, store: ArtifactStore, client: ArizeV8Client,
    settings: ArizeExportSettings, model_version_id: str,
    baseline_version_id: str, package_dir: str | Path,
) -> dict[str, Any]:
    settings.require_enabled()
    ArizeOutboxRepository(engine).require_active_approval(
        approval_id=settings.privacy_approval_id or "", model_name=settings.model_name,
        environment=settings.environment.value,
    )
    with engine.connect() as connection:
        baseline = connection.execute(text("""
            SELECT baseline_version_id, model_version_id, reference_dataset_uri,
                   reference_sha256, feature_schema_version, purpose, approval_metadata
            FROM monitoring_baselines
            WHERE baseline_version_id = :baseline_version_id
              AND model_version_id = :model_version_id
        """), {"baseline_version_id": baseline_version_id,
                "model_version_id": model_version_id}).mappings().one_or_none()
    if baseline is None:
        raise LookupError("approved exact-version baseline was not found")
    with engine.connect() as connection:
        exported = connection.execute(text("""
            SELECT reference_sha256 FROM arize_baseline_exports
            WHERE baseline_version_id = :baseline_version_id
              AND model_version_id = :model_version_id
        """), {"baseline_version_id": baseline_version_id,
                "model_version_id": model_version_id}).scalar_one_or_none()
    if exported is not None:
        if exported != baseline["reference_sha256"]:
            raise ValueError("baseline export identity conflicts with registered checksum")
        return {"status": "already_uploaded", "rows": 0,
                "model_version_id": model_version_id,
                "baseline_version_id": baseline_version_id,
                "reference_sha256": exported}
    if baseline["purpose"] not in {"drift_reference", "monitoring_reference"} or not baseline["approval_metadata"]:
        raise ValueError("baseline is not approved for monitoring")
    body = store.read_uri(baseline["reference_dataset_uri"])
    frame = validate_reference(
        body, expected_sha256=baseline["reference_sha256"],
        schema_version=baseline["feature_schema_version"],
    )
    from src.mlops.deployment import validate_packaged_model

    metadata = validate_packaged_model(package_dir)
    if metadata["model_version_id"] != model_version_id:
        raise ValueError("packaged model does not match baseline model version")
    import mlflow.sklearn

    pipeline = mlflow.sklearn.load_model(str(Path(package_dir) / "model"))
    features = frame[CANONICAL_FEATURE_ORDER]
    probabilities = pipeline.predict_proba(features)[:, 1]
    labels = pipeline.predict(features)
    uploaded = features.copy()
    uploaded["prediction_id"] = [
        sha256(f"{baseline['reference_sha256']}:{index}".encode()).hexdigest()
        for index in range(len(uploaded))
    ]
    uploaded["prediction_timestamp"] = int(
        datetime.now(timezone.utc).timestamp()
    )
    uploaded["prediction_label"] = [binary_label(int(value)) for value in labels]
    uploaded["prediction_score"] = probabilities
    uploaded["model_version_id"] = model_version_id
    uploaded["baseline_version_id"] = baseline_version_id
    has_actuals = TARGET_COLUMN in frame
    if has_actuals:
        uploaded["actual_label"] = [binary_label(int(value)) for value in frame[TARGET_COLUMN]]
    client.log_baseline(
        uploaded.reset_index(drop=True),
        model_version=numeric_model_version(model_version_id),
        has_actuals=has_actuals,
    )
    with engine.begin() as connection:
        connection.execute(text("""
            INSERT INTO arize_baseline_exports (
                baseline_version_id, model_version_id, reference_sha256, sent_at
            ) VALUES (
                :baseline_version_id, :model_version_id, :reference_sha256,
                clock_timestamp()
            ) ON CONFLICT (baseline_version_id) DO NOTHING
        """), {"baseline_version_id": baseline_version_id,
                "model_version_id": model_version_id,
                "reference_sha256": baseline["reference_sha256"]})
    return {"status": "uploaded", "rows": len(uploaded),
            "model_version_id": model_version_id,
            "baseline_version_id": baseline_version_id,
            "reference_sha256": baseline["reference_sha256"]}
