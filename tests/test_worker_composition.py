from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.workers import arize_export, label_materialization


def test_observation_time_normalizes_offsets_to_utc():
    observed_at = label_materialization.resolve_observation_time(
        "2026-09-07T04:45:00+02:00"
    )

    assert observed_at == datetime(2026, 9, 7, 2, 45, tzinfo=timezone.utc)


def test_observation_time_rejects_naive_values():
    with pytest.raises(ValueError, match="timezone"):
        label_materialization.resolve_observation_time("2026-09-07T02:45:00")


def test_label_worker_owns_composition_and_disposes_engine(monkeypatch):
    engine = Mock()
    repository = Mock()
    job = Mock()
    job.run.return_value = {"status": "completed"}
    settings = SimpleNamespace(
        required_sources=("customer-master",),
        horizon_days=90,
        grace_period_days=7,
        label_contract_version="1.0.0",
        environment=SimpleNamespace(value="production"),
    )
    monkeypatch.setattr(label_materialization, "DatabaseSettings", Mock())
    monkeypatch.setattr(
        label_materialization, "LabelMaterializationSettings", Mock(return_value=settings)
    )
    monkeypatch.setattr(
        label_materialization, "create_database_engine", Mock(return_value=engine)
    )
    monkeypatch.setattr(
        label_materialization, "LabelRepository", Mock(return_value=repository)
    )
    job_factory = Mock(return_value=job)
    monkeypatch.setattr(label_materialization, "LabelMaterializationJob", job_factory)

    result = label_materialization.execute_label_materialization(
        "2026-09-07T02:45:00Z"
    )

    assert result == {"status": "completed"}
    job_factory.assert_called_once_with(
        repository,
        required_sources=("customer-master",),
        horizon_days=90,
        grace_period_days=7,
        label_contract_version="1.0.0",
    )
    engine.dispose.assert_called_once_with()


def test_arize_worker_surfaces_delivery_failures_and_disposes_engine(monkeypatch):
    engine = Mock()
    exporter = Mock()
    exporter.run.return_value = {"retried": 1, "dead_lettered": 0}
    monkeypatch.setattr(arize_export, "DatabaseSettings", Mock())
    monkeypatch.setattr(arize_export, "ArizeExportSettings", Mock())
    monkeypatch.setattr(
        arize_export, "create_database_engine", Mock(return_value=engine)
    )
    monkeypatch.setattr(arize_export, "ArizeOutboxRepository", Mock())
    monkeypatch.setattr(arize_export, "ArizeV8Client", Mock())
    monkeypatch.setattr(arize_export, "ArizeExporter", Mock(return_value=exporter))

    with pytest.raises(RuntimeError, match="persisted failures"):
        arize_export.execute_arize_export()

    engine.dispose.assert_called_once_with()
