"""Operations CLI for exporter, backfill, and baseline upload."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys

from src.config import DatabaseSettings, MonitoringSettings, safe_error_message
from src.database import create_database_engine
from src.monitoring.shared.artifacts import LocalArtifactStore, S3ArtifactStore

from .baseline import upload_baseline
from .client import ArizeV8Client
from .config import ArizeExportSettings
from .exporter import ArizeExporter
from .repository import ArizeOutboxRepository


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
    parser = argparse.ArgumentParser(description="Privacy-gated Arize export operations")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("export")
    backfill = commands.add_parser("backfill")
    backfill.add_argument("--from", dest="start", type=_date)
    backfill.add_argument("--to", dest="end", type=_date)
    backfill.add_argument("--model-version")
    backfill.add_argument("--batch-size", type=int, default=500)
    backfill.add_argument("--dry-run", action="store_true")
    mode = backfill.add_mutually_exclusive_group()
    mode.add_argument("--predictions-only", action="store_true")
    mode.add_argument("--actuals-only", action="store_true")
    backfill.add_argument("--resume-from")
    baseline = commands.add_parser("upload-baseline")
    baseline.add_argument("--model-version-id", required=True)
    baseline.add_argument("--baseline-version-id", required=True)
    baseline.add_argument("--package-dir", default="build/model")
    args = parser.parse_args(argv)
    engine = None
    try:
        settings = ArizeExportSettings()
        settings.require_enabled()
        engine = create_database_engine(DatabaseSettings())
        repository = ArizeOutboxRepository(engine)
        repository.require_active_approval(
            approval_id=settings.privacy_approval_id or "",
            model_name=settings.model_name, environment=settings.environment.value,
        )
        if args.command == "export":
            result = ArizeExporter(repository, ArizeV8Client(settings), settings).run()
        elif args.command == "backfill":
            if not 1 <= args.batch_size <= 5000:
                raise ValueError("batch size must be between 1 and 5000")
            kinds = ["actual"] if args.actuals_only else ["prediction"] if args.predictions_only else ["prediction", "actual"]
            result = {"status": "dry_run" if args.dry_run else "enqueued", "pages": []}
            cursor = args.resume_from
            for kind in kinds:
                page = repository.enqueue_backfill_page(
                    event_type=kind, start=args.start, end=args.end,
                    model_version_id=args.model_version, limit=args.batch_size,
                    resume_from=cursor if len(kinds) == 1 else None,
                    dry_run=args.dry_run,
                )
                result["pages"].append({"event_type": kind, **page})
                # In combined mode, drain prediction pages across invocations
                # before allowing any historical actual into the outbox.
                if kind == "prediction" and page["selected"]:
                    break
        else:
            monitoring = MonitoringSettings()
            result = upload_baseline(
                engine=engine, store=_store(monitoring), client=ArizeV8Client(settings),
                settings=settings, model_version_id=args.model_version_id,
                baseline_version_id=args.baseline_version_id,
                package_dir=args.package_dir,
            )
        print(json.dumps(result, sort_keys=True, default=str))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__,
                          "error": safe_error_message(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
