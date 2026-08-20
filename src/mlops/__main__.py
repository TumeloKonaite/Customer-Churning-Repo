"""CLI for registered-model validation and Modal package preparation."""

from __future__ import annotations

import argparse
import json
import sys

from src.config import safe_error_message
from src.mlops.deployment import prepare_deployment
from src.mlops.registry import validate_registered_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exact-version churn model operations")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-model")
    validate.add_argument("--model-uri", required=True)
    validate.add_argument("--expected-run-id")
    validate.add_argument("--expected-pipeline-sha256")
    validate.add_argument("--output", choices=("text", "json"), default="text")
    prepare = commands.add_parser("prepare-deployment")
    prepare.add_argument("--model-uri", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--expected-run-id")
    prepare.add_argument("--expected-pipeline-sha256")
    prepare.add_argument("--expected-model-version-id")
    prepare.add_argument("--environment", default="production")
    prepare.add_argument("--output", choices=("text", "json"), default="text")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate-model":
            result = validate_registered_model(
                args.model_uri,
                expected_run_id=args.expected_run_id,
                expected_pipeline_sha256=args.expected_pipeline_sha256,
            ).as_dict()
        else:
            result = prepare_deployment(
                args.model_uri,
                args.output_dir,
                expected_run_id=args.expected_run_id,
                expected_pipeline_sha256=args.expected_pipeline_sha256,
                expected_model_version_id=args.expected_model_version_id,
                environment=args.environment,
            )
    except Exception as exc:
        message = safe_error_message(exc)
        if args.output == "json":
            print(json.dumps({"valid": False, "error": message}), file=sys.stderr)
        else:
            print(f"error: {message}", file=sys.stderr)
        return 1
    if args.output == "json":
        print(json.dumps(result, sort_keys=True))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
