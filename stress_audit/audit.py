"""Core stress audit: validation, transforms, section loads, yield metrics."""

from __future__ import annotations

import math

from .linalg import (
    Mat3,
    Vec3,
    add_mat,
    dot,
    jacobi_eigen,
    mat_mul,
    mat_vec,
    norm,
    scale_mat,
    transpose,
)

ORTHONORMAL_TOL = 1e-9
EIGEN_DEGENERATE_TOL = 1e-9


class CaseError(ValueError):
    """Raised when a single load case is invalid."""


def _as_finite_matrix(value, name: str) -> Mat3:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(not isinstance(row, list) or len(row) != 3 for row in value)
    ):
        raise CaseError(f"{name} must be a 3x3 array")
    mat = tuple(tuple(float(x) for x in row) for row in value)
    if any(not math.isfinite(x) for row in mat for x in row):
        raise CaseError(f"{name} contains non-finite values")
    return mat


def validate_rotation(r_raw) -> Mat3:
    """Columns of r are the local axes expressed in global coordinates.

    Must form a right-handed orthonormal basis.
    """
    r = _as_finite_matrix(r_raw, "rotation")
    cols = [tuple(r[i][j] for i in range(3)) for j in range(3)]
    for j, c in enumerate(cols):
        if abs(norm(c) - 1.0) > ORTHONORMAL_TOL:
            raise CaseError(f"rotation column {j} is not a unit vector")
    for a, b in ((0, 1), (0, 2), (1, 2)):
        if abs(dot(cols[a], cols[b])) > ORTHONORMAL_TOL:
            raise CaseError("rotation columns are not mutually orthogonal")
    det = (
        r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
    )
    if abs(det - 1.0) > ORTHONORMAL_TOL:
        raise CaseError("rotation is not a proper right-handed rotation (det != +1)")
    return r


def validate_stress(s_raw) -> Mat3:
    s = _as_finite_matrix(s_raw, "stress_mpa")
    for i in range(3):
        for j in range(i + 1, 3):
            if abs(s[i][j] - s[j][i]) > ORTHONORMAL_TOL * max(
                1.0, abs(s[i][j]), abs(s[j][i])
            ):
                raise CaseError("stress tensor is not symmetric")
    return s


def to_global(stress_local: Mat3, rotation: Mat3) -> Mat3:
    """sigma_global = R sigma_local R^T (columns of R are local axes in global)."""
    return mat_mul(mat_mul(rotation, stress_local), transpose(rotation))


def principal_stresses(sigma: Mat3):
    """Return eigenpairs sorted descending, with degeneracy flags."""
    values, vectors = jacobi_eigen(sigma)
    order = sorted(range(3), key=lambda i: values[i], reverse=True)
    vals = [values[i] for i in order]
    vecs = [vectors[i] for i in order]
    scale = max(1.0, max(abs(v) for v in vals))
    degenerate = [False, False, False]
    for i in range(3):
        for j in range(i + 1, 3):
            if abs(vals[i] - vals[j]) <= EIGEN_DEGENERATE_TOL * scale:
                degenerate[i] = degenerate[j] = True
    return vals, vecs, degenerate


def mean_stress(sigma: Mat3) -> float:
    return (sigma[0][0] + sigma[1][1] + sigma[2][2]) / 3.0


def von_mises(sigma: Mat3) -> float:
    sxx, syy, szz = sigma[0][0], sigma[1][1], sigma[2][2]
    sxy, sxz, syz = sigma[0][1], sigma[0][2], sigma[1][2]
    return math.sqrt(
        0.5
        * (
            (sxx - syy) ** 2
            + (syy - szz) ** 2
            + (szz - sxx) ** 2
            + 6.0 * (sxy**2 + sxz**2 + syz**2)
        )
    )


def section_loads(sigma: Mat3, normal_raw) -> dict:
    if (
        not isinstance(normal_raw, list)
        or len(normal_raw) != 3
        or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in normal_raw)
    ):
        raise CaseError("section normal must be three finite numbers")
    n_raw = tuple(float(x) for x in normal_raw)
    length = norm(n_raw)
    if length == 0.0:
        raise CaseError("section normal must be non-zero")
    n = tuple(x / length for x in n_raw)
    traction = mat_vec(sigma, n)
    sigma_n = dot(traction, n)
    tangential = tuple(traction[i] - sigma_n * n[i] for i in range(3))
    return {
        "unit_normal": n,
        "traction": traction,
        "normal_stress": sigma_n,
        "shear_magnitude": norm(tangential),
    }


def combine_cases(terms, resolved: dict[str, Mat3]) -> Mat3:
    if not isinstance(terms, list) or not terms:
        raise CaseError("combination must be a non-empty list")
    total: Mat3 = ((0.0,) * 3,) * 3
    for term in terms:
        if not isinstance(term, dict):
            raise CaseError("combination terms must be objects")
        ref = term.get("case")
        factor = term.get("factor")
        if not isinstance(ref, str) or ref not in resolved:
            raise CaseError(f"combination references unknown case {ref!r}")
        if not isinstance(factor, (int, float)) or not math.isfinite(factor):
            raise CaseError("combination factor must be a finite number")
        total = add_mat(total, scale_mat(resolved[ref], float(factor)))
    return total
