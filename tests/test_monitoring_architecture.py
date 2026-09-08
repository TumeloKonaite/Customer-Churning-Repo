from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def _imported_by(module: str) -> set[str]:
    script = (
        "import importlib,json,sys;"
        f"importlib.import_module({module!r});"
        "print(json.dumps(sorted(sys.modules)))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )
    return set(json.loads(completed.stdout))


def test_prediction_route_does_not_import_arize_sdk_or_monitoring_workers():
    imported = _imported_by("src.api.routes.predictions")
    assert "arize" not in imported
    assert "src.monitoring.arize.exporter" not in imported
    assert "src.monitoring.outcomes.service" not in imported


def test_arize_is_the_only_model_monitoring_platform_dependency():
    dependency_files = "\n".join(
        Path(path).read_text(encoding="utf-8").casefold()
        for path in ("pyproject.toml", "uv.lock", "requirements.txt")
    )
    project = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "evidently" not in dependency_files
    assert '"arize==8.51.0"' in project
    assert not any(Path("src/monitoring/drift").glob("*.py"))


def test_modal_has_only_arize_export_and_label_materialization_schedules():
    source = Path("modal_app.py").read_text(encoding="utf-8")
    assert "scheduled_arize_export" in source
    assert "scheduled_label_materialization" in source
    assert "scheduled_monitoring" not in source
    assert "scheduled_performance_monitoring" not in source
    assert "create_database_engine" not in source
    assert "LabelMaterializationJob" not in source
    assert "ArizeExporter(" not in source
