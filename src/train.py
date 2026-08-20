"""Thin command-line entrypoint for the existing training pipeline."""

from __future__ import annotations

import argparse
import json
import sys

from src.config import safe_error_message
from src.pipeline.training_pipeline import TrainingPipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local churn model training")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train locally and optionally track the run")
    train.add_argument("--config", default="configs/training.yaml")
    args = parser.parse_args(argv)

    try:
        result = TrainingPipeline().run(args.config)
    except Exception as exc:
        print(
            json.dumps({"status": "error", "error": safe_error_message(exc)}),
            file=sys.stderr,
        )
        return 1

    print(json.dumps(result.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
