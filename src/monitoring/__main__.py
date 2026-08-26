"""Manual monitoring execution and immutable policy/baseline registration."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from src.config import DatabaseSettings, MonitoringSettings, safe_error_message
from src.database import create_database_engine
from src.monitoring.shared.artifacts import LocalArtifactStore, S3ArtifactStore
from src.monitoring.drift.service import MonitoringJob
from src.monitoring.shared.models import BaselineVersion, MonitoringPolicy
from src.monitoring.drift.repository import MonitoringRepository


def _json_file(path: str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration file must contain a JSON object")
    return value


def _date(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _store(settings: MonitoringSettings):
    if settings.local_artifact_dir is not None:
        return LocalArtifactStore(settings.local_artifact_dir)
    return S3ArtifactStore(
        settings.artifact_bucket or "",
        endpoint_url=settings.artifact_endpoint_url,
        region_name=settings.artifact_region,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproducible Evidently monitoring")
    commands = parser.add_subparsers(dest="command", required=True)
    policy = commands.add_parser("register-policy")
    policy.add_argument("--file", required=True)
    baseline = commands.add_parser("register-baseline")
    baseline.add_argument("--file", required=True)
    run = commands.add_parser("run")
    run.add_argument("--environment")
    run.add_argument("--model-version-id")
    run.add_argument("--as-of", type=_date)

    args = parser.parse_args(argv)
    engine = None
    try:
        engine = create_database_engine(DatabaseSettings())
        repository = MonitoringRepository(engine)
        if args.command == "register-policy":
            value = MonitoringPolicy.model_validate(_json_file(args.file))
            repository.register_policy(value)
            result = {
                "status": "registered",
                "policy_version": value.policy_version,
                "configuration_sha256": value.config_sha256,
            }
        elif args.command == "register-baseline":
            value = BaselineVersion.model_validate(_json_file(args.file))
            repository.register_baseline(value)
            result = {
                "status": "registered",
                "baseline_version_id": value.baseline_version_id,
                "reference_sha256": value.reference_sha256,
            }
        else:
            settings = MonitoringSettings()
            result = MonitoringJob(repository, _store(settings)).run(
                environment=args.environment or settings.environment.value,
                model_version_id=args.model_version_id or settings.model_version_id,
                scheduled_for=args.as_of,
            )
        print(json.dumps(result, sort_keys=True, default=str))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": safe_error_message(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
