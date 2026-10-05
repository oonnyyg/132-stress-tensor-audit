"""Input model: parse JSON spec, evaluate cases, build reports."""

from __future__ import annotations

import math

from .stress import (
    combine,
    principal_stresses,
    section_loads,
    to_global,
    voigt_to_tensor,
    yield_check,
)


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise ValueError(message)


def _parse_sections(raw_sections) -> list[dict]:
    _require(isinstance(raw_sections, list) and raw_sections, "input must define a non-empty 'sections' list")
    sections = []
    for i, raw in enumerate(raw_sections):
        _require(isinstance(raw, dict), f"section {i} must be an object")
        name = str(raw.get("name", f"section_{i}"))
        normal = raw.get("normal")
        _require(
            isinstance(normal, list) and len(normal) == 3,
            f"section '{name}' needs a 3-component 'normal'",
        )
        try:
            normal = [float(v) for v in normal]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"section '{name}' normal must be numeric: {exc}") from exc
        sections.append({"name": name, "normal": normal})
    return sections


def _parse_base_cases(raw_cases) -> dict[str, dict]:
    bases = {}
    for i, raw in enumerate(raw_cases):
        _require(isinstance(raw, dict), f"case {i} must be an object")
        case_id = raw.get("id")
        _require(isinstance(case_id, str) and case_id, f"case {i} needs a string 'id'")
        _require(case_id not in bases, f"duplicate case id '{case_id}'")
        _require("stress" in raw or "combination" in raw, f"case '{case_id}' needs 'stress' or 'combination'")
        bases[case_id] = raw
    return bases


def evaluate_case(case_id: str, spec: dict, bases: dict[str, dict], sections: list[dict]) -> dict:
    """Evaluate one case (base or combination) into a report entry."""
    raw = bases.get(case_id)
    _require(raw is not None, f"unknown case id '{case_id}'")

    if "combination" in raw:
        terms = []
        for term in raw["combination"]:
            ref = term.get("case")
            _require(ref in bases, f"case '{case_id}' references unknown case '{ref}'")
            _require("combination" not in bases[ref], f"case '{case_id}' references non-base case '{ref}'")
            coeff = float(term.get("coefficient", 1.0))
            _require(math.isfinite(coeff), f"case '{case_id}' has non-finite coefficient")
            ref_raw = bases[ref]
            _require("rotation" in ref_raw, f"referenced case '{ref}' missing 'rotation'")
            local = voigt_to_tensor(ref_raw["stress"])
            terms.append((coeff, to_global(local, ref_raw["rotation"])))
        global_tensor = combine(terms)
        sources = [t.get("case") for t in raw["combination"]]
    else:
        _require("rotation" in raw, f"case '{case_id}' missing 'rotation'")
        local = voigt_to_tensor(raw["stress"])
        global_tensor = to_global(local, raw["rotation"])
        sources = [case_id]

    yield_strength = float(raw.get("yield_strength", spec.get("yield_strength")))
    principal = principal_stresses(global_tensor)
    loads = [
        {"section": s["name"], **section_loads(global_tensor, s["normal"])}
        for s in sections
    ]
    return {
        "id": case_id,
        "sources": sources,
        "global_stress": global_tensor,
        "principal": principal,
        "sections": loads,
        "yield": yield_check(global_tensor, yield_strength),
    }


def run_spec(spec: dict) -> dict:
    """Evaluate every case in the spec; failures are isolated per case."""
    _require(isinstance(spec, dict), "top-level input must be a JSON object")
    _require("yield_strength" in spec, "input must define 'yield_strength'")
    sections = _parse_sections(spec.get("sections"))
    raw_cases = spec.get("cases")
    _require(isinstance(raw_cases, list) and raw_cases, "input must define a non-empty 'cases' list")
    bases = _parse_base_cases(raw_cases)

    results, errors = [], []
    for case_id in bases:
        try:
            results.append(evaluate_case(case_id, spec, bases, sections))
        except (ValueError, KeyError, TypeError) as exc:
            errors.append({"id": case_id, "error": str(exc)})
    return {"sections": sections, "results": results, "errors": errors}
