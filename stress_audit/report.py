"""Build per-case reports and section result rows from the JSON input."""

from __future__ import annotations

import math

from . import audit
from .linalg import Mat3


def _residual(sigma: Mat3, values, vectors) -> float:
    """Max |sigma v - lambda v| over the eigenpairs, in MPa."""
    worst = 0.0
    for val, vec in zip(values, vectors):
        for i in range(3):
            av = sum(sigma[i][k] * vec[k] for k in range(3))
            worst = max(worst, abs(av - val * vec[i]))
    return worst


def build_report(data: dict) -> dict:
    if not isinstance(data, dict):
        raise ValueError("top-level input must be a JSON object")
    yield_strength = data.get("yield_strength_mpa")
    if (
        not isinstance(yield_strength, (int, float))
        or not math.isfinite(yield_strength)
        or yield_strength <= 0.0
    ):
        raise ValueError("yield_strength_mpa must be a positive finite number")
    yield_strength = float(yield_strength)

    sections = data.get("sections", [])
    if not isinstance(sections, list):
        raise ValueError("sections must be a list")
    cases = data.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError("cases must be a list")

    resolved: dict[str, Mat3] = {}
    case_reports = []
    section_rows = []

    for index, case in enumerate(cases):
        case_id = case.get("id") if isinstance(case, dict) else None
        if not isinstance(case_id, str) or not case_id:
            case_id = f"case_{index}"
        try:
            if not isinstance(case, dict):
                raise audit.CaseError("case entry must be an object")
            if case_id in resolved:
                raise audit.CaseError(f"duplicate case id {case_id!r}")
            if "combination" in case:
                sigma = audit.combine_cases(case["combination"], resolved)
                source = "combination"
            else:
                stress = audit.validate_stress(case.get("stress_mpa"))
                rotation = audit.validate_rotation(case.get("rotation"))
                sigma = audit.to_global(stress, rotation)
                source = "tensor"
            resolved[case_id] = sigma
        except audit.CaseError as exc:
            case_reports.append({"id": case_id, "status": "error", "error": str(exc)})
            continue

        vals, vecs, degenerate = audit.principal_stresses(sigma)
        mean = audit.mean_stress(sigma)
        vm = audit.von_mises(sigma)
        principals = []
        for rank, (val, vec, deg) in enumerate(zip(vals, vecs, degenerate), start=1):
            entry = {
                "rank": rank,
                "value_mpa": val,
                "direction": list(vec),
                "direction_unique": not deg,
            }
            if deg:
                entry["note"] = (
                    "repeated principal stress: principal direction is not unique"
                )
            principals.append(entry)
        case_reports.append(
            {
                "id": case_id,
                "status": "ok",
                "source": source,
                "global_stress_mpa": [list(row) for row in sigma],
                "principal_stresses": principals,
                "mean_stress_mpa": mean,
                "von_mises_mpa": vm,
                "yield_strength_mpa": yield_strength,
                "yielded": vm >= yield_strength,
                "eigen_residual_mpa": _residual(sigma, vals, vecs),
            }
        )
        for s_index, section in enumerate(sections):
            sec_id = section.get("id") if isinstance(section, dict) else None
            if not isinstance(sec_id, str) or not sec_id:
                sec_id = f"section_{s_index}"
            try:
                loads = audit.section_loads(sigma, section.get("normal"))
            except audit.CaseError as exc:
                section_rows.append(
                    {
                        "case_id": case_id,
                        "section_id": sec_id,
                        "status": "error",
                        "error": str(exc),
                    }
                )
                continue
            section_rows.append(
                {
                    "case_id": case_id,
                    "section_id": sec_id,
                    "status": "ok",
                    "unit_normal": loads["unit_normal"],
                    "traction_mpa": loads["traction"],
                    "normal_stress_mpa": loads["normal_stress"],
                    "shear_mpa": loads["shear_magnitude"],
                }
            )

    return {
        "yield_strength_mpa": yield_strength,
        "cases": case_reports,
        "sections": section_rows,
    }
