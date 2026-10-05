"""Command line interface: JSON input -> CSV section results + JSON case report."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

from .report import build_report

CSV_FIELDS = [
    "case_id",
    "section_id",
    "status",
    "unit_normal_x",
    "unit_normal_y",
    "unit_normal_z",
    "traction_x_mpa",
    "traction_y_mpa",
    "traction_z_mpa",
    "normal_stress_mpa",
    "shear_mpa",
    "error",
]


def _fmt(value) -> str:
    return f"{value:.10g}"


def write_sections_csv(rows, stream) -> None:
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for row in rows:
        out = {key: "" for key in CSV_FIELDS}
        out["case_id"] = row["case_id"]
        out["section_id"] = row["section_id"]
        out["status"] = row["status"]
        if row["status"] == "ok":
            for axis, val in zip("xyz", row["unit_normal"]):
                out[f"unit_normal_{axis}"] = _fmt(val)
            for axis, val in zip("xyz", row["traction_mpa"]):
                out[f"traction_{axis}_mpa"] = _fmt(val)
            out["normal_stress_mpa"] = _fmt(row["normal_stress_mpa"])
            out["shear_mpa"] = _fmt(row["shear_mpa"])
        else:
            out["error"] = row["error"]
        writer.writerow(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="stress-audit",
        description="Offline stress tensor audit: unify coordinates, section loads, yield check.",
    )
    parser.add_argument("input", help="path to the JSON input file")
    parser.add_argument(
        "--csv", default="sections.csv", help="output CSV path (default: sections.csv)"
    )
    parser.add_argument(
        "--json", default="report.json", help="output JSON report path (default: report.json)"
    )
    args = parser.parse_args(argv)

    with open(args.input, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    try:
        report = build_report(data)
    except ValueError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2

    for path in (args.csv, args.json):
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
    with open(args.csv, "w", encoding="utf-8", newline="") as fh:
        write_sections_csv(report["sections"], fh)
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    errors = sum(1 for c in report["cases"] if c["status"] == "error")
    print(
        f"processed {len(report['cases'])} case(s), {errors} error(s); "
        f"wrote {args.csv} and {args.json}"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
