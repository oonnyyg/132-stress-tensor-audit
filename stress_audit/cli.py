"""Command line interface for the offline stress audit tool."""

from __future__ import annotations

import argparse
import json
import sys

from .model import run_spec
from .report import cases_json, sections_csv


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="stress-audit",
        description="Offline stress audit: unify case frames, compute section loads and yield.",
    )
    parser.add_argument("input", help="path to the JSON input spec ('-' for stdin)")
    parser.add_argument("--csv", help="write section results CSV to this path (default: stdout)")
    parser.add_argument("--json", help="write case report JSON to this path (default: stdout)")
    args = parser.parse_args(argv)

    try:
        text = sys.stdin.read() if args.input == "-" else open(args.input, encoding="utf-8").read()
        spec = json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read input: {exc}", file=sys.stderr)
        return 2

    try:
        outcome = run_spec(spec)
    except ValueError as exc:
        print(f"error: invalid input: {exc}", file=sys.stderr)
        return 2

    csv_text = sections_csv(outcome["results"])
    json_text = cases_json(outcome["results"], outcome["errors"])

    if args.csv:
        with open(args.csv, "w", encoding="utf-8", newline="") as fh:
            fh.write(csv_text)
    else:
        sys.stdout.write(csv_text)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            fh.write(json_text + "\n")
    else:
        sys.stdout.write(json_text + "\n")

    for err in outcome["errors"]:
        print(f"case '{err['id']}': {err['error']}", file=sys.stderr)
    return 1 if outcome["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
