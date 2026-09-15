"""Secret-safe Neon connectivity command."""

from __future__ import annotations

import argparse
import json
import sys

from src.config import safe_error_message
from src.database.connection import check_connectivity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Operational database commands")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Run a lightweight connectivity check")
    check.add_argument("--output", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)
    try:
        result = check_connectivity()
    except Exception as exc:
        message = safe_error_message(exc)
        if args.output == "json":
            print(json.dumps({"status": "error", "error": message}), file=sys.stderr)
        else:
            print(f"Database connectivity: error ({message})", file=sys.stderr)
        return 1
    if args.output == "json":
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"Database connectivity: {result['status']}")
        print(f"Duration seconds: {result['duration_seconds']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
