from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from src.monitoring.drift.evidently import build_drift_report, run_drift_report
from src.monitoring.shared.models import MonitoringPolicy, ResultStatus


def _imported_by(module: str) -> set[str]:
    script = (
        "import importlib,json,sys;"
        f"importlib.import_module({module!r});"
        "print(json.dumps(sorted(sys.modules)))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return set(json.loads(completed.stdout))


def _small_policy() -> MonitoringPolicy:
    value = json.loads(Path("configs/monitoring/policy-v1.0.0.json").read_text())
    value["feature_rules"] = {
        name: rule
        for name, rule in value["feature_rules"].items()
        if name in {"Age", "Geography"}
    }
    return MonitoringPolicy.model_validate(value)


def test_prediction_route_does_not_import_monitoring_capabilities_or_evidently():
    imported = _imported_by("src.api.routes.predictions")

    assert not any(name == "evidently" or name.startswith("evidently.") for name in imported)
    assert "src.monitoring.outcomes.service" not in imported
    assert not any(name.startswith("src.monitoring.performance") for name in imported)


def test_drift_service_does_not_import_outcomes_or_performance():
    imported = _imported_by("src.monitoring.drift.service")

    assert "src.monitoring.outcomes.service" not in imported
    assert not any(name.startswith("src.monitoring.performance") for name in imported)
    assert not any(name == "evidently" or name.startswith("evidently.") for name in imported)


def test_pinned_evidently_normalizes_numeric_categorical_and_prediction_drift():
    policy = _small_policy()
    reference = pd.DataFrame(
        {
            "Age": [25 + index % 30 for index in range(120)],
            "Geography": ["France", "Germany", "Spain"] * 40,
            "prediction_probability": [0.1 + (index % 20) / 100 for index in range(120)],
            "predicted_class": [str(index % 2) for index in range(120)],
        }
    )
    current = pd.DataFrame(
        {
            "Age": [65 + index % 20 for index in range(120)],
            "Geography": ["Germany"] * 120,
            "prediction_probability": [0.75 + (index % 20) / 100 for index in range(120)],
            "predicted_class": ["1"] * 120,
        }
    )

    report = build_drift_report(policy)
    output = run_drift_report(reference, current, policy=policy)

    assert type(report).__name__ == "Report"
    assert type(output.snapshot).__name__ == "Snapshot"
    assert output.version == "0.7.21"
    assert output.html.lstrip().startswith(b"<")
    assert set(output.drift_summary["feature_results"]) == {
        "Age",
        "Geography",
        "prediction_probability",
        "predicted_class",
    }
    assert output.drift_summary["status"] is ResultStatus.WARNING
    assert all(
        result["drift_detected"]
        for result in output.drift_summary["feature_results"].values()
    )
