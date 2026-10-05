"""Render evaluation results as CSV section rows and a JSON case report."""

from __future__ import annotations

import csv
import io
import json

CSV_FIELDS = [
    "case_id",
    "section",
    "normal_x", "normal_y", "normal_z",
    "traction_x", "traction_y", "traction_z",
    "normal_stress_mpa",
    "shear_stress_mpa",
    "von_mises_mpa",
    "mean_normal_stress_mpa",
    "yield_strength_mpa",
    "yielded",
]


def sections_csv(results: list[dict]) -> str:
    """One CSV row per (case, section) with load components and yield data."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for result in results:
        for load in result["sections"]:
            n = load["unit_normal"]
            t = load["traction"]
            writer.writerow({
                "case_id": result["id"],
                "section": load["section"],
                "normal_x": f"{n[0]:.10g}",
                "normal_y": f"{n[1]:.10g}",
                "normal_z": f"{n[2]:.10g}",
                "traction_x": f"{t[0]:.10g}",
                "traction_y": f"{t[1]:.10g}",
                "traction_z": f"{t[2]:.10g}",
                "normal_stress_mpa": f"{load['normal_stress']:.10g}",
                "shear_stress_mpa": f"{load['shear_stress']:.10g}",
                "von_mises_mpa": f"{result['yield']['von_mises']:.10g}",
                "mean_normal_stress_mpa": f"{result['yield']['mean_normal_stress']:.10g}",
                "yield_strength_mpa": f"{result['yield']['yield_strength']:.10g}",
                "yielded": "yes" if result["yield"]["yielded"] else "no",
            })
    return buffer.getvalue()


def cases_json(results: list[dict], errors: list[dict]) -> str:
    """Full JSON report per case, keeping source ids and residuals."""
    cases = []
    for result in results:
        principal = result["principal"]
        cases.append({
            "id": result["id"],
            "sources": result["sources"],
            "global_stress_mpa": result["global_stress"],
            "principal_stresses_mpa": principal["values"],
            "principal_directions": principal["directions"],
            "principal_directions_unique": not principal["degenerate"],
            "notes": principal["notes"],
            "sections": [
                {
                    "section": load["section"],
                    "unit_normal": load["unit_normal"],
                    "traction_mpa": load["traction"],
                    "normal_stress_mpa": load["normal_stress"],
                    "shear_stress_mpa": load["shear_stress"],
                }
                for load in result["sections"]
            ],
            "yield": result["yield"],
        })
    return json.dumps({"cases": cases, "errors": errors}, indent=2, ensure_ascii=False)
