"""Small pure-stdlib linear algebra helpers for 3x3 problems."""

from __future__ import annotations

import math


def is_finite_matrix(m: list[list[float]]) -> bool:
    return all(math.isfinite(v) for row in m for v in row)


def mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def mat_transpose(a: list[list[float]]) -> list[list[float]]:
    return [[a[j][i] for j in range(3)] for i in range(3)]


def mat_vec_mul(a: list[list[float]], v: list[float]) -> list[float]:
    return [sum(a[i][k] * v[k] for k in range(3)) for i in range(3)]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(v: list[float]) -> float:
    return math.sqrt(dot(v, v))


def scale_mat(a: list[list[float]], s: float) -> list[list[float]]:
    return [[v * s for v in row] for row in a]


def add_mat(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[a[i][j] + b[i][j] for j in range(3)] for i in range(3)]


def validate_rotation(r: list[list[float]], tol: float = 1e-6) -> None:
    """Check that columns of r form a right-handed orthonormal basis."""
    if len(r) != 3 or any(len(row) != 3 for row in r):
        raise ValueError("rotation matrix must be 3x3")
    if not is_finite_matrix(r):
        raise ValueError("rotation matrix contains non-finite values")
    cols = [[r[i][j] for i in range(3)] for j in range(3)]
    for j, c in enumerate(cols):
        if abs(norm(c) - 1.0) > tol:
            raise ValueError(f"rotation column {j} is not a unit vector")
    for j in range(3):
        for k in range(j + 1, 3):
            if abs(dot(cols[j], cols[k])) > tol:
                raise ValueError("rotation columns are not orthogonal")
    det = (
        r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
    )
    if abs(det - 1.0) > 1e-4:
        raise ValueError("rotation matrix is not a proper right-handed rotation (det != +1)")


def jacobi_eigendecomposition(a: list[list[float]], tol: float = 1e-14, max_sweeps: int = 100):
    """Eigenpairs of a real symmetric 3x3 matrix via cyclic Jacobi rotations.

    Returns (eigenvalues, eigenvectors); eigenvectors[i] is the unit vector
    for eigenvalues[i]. Values are not sorted here.
    """
    m = [row[:] for row in a]
    v = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    for _ in range(max_sweeps):
        off = math.sqrt(m[0][1] ** 2 + m[0][2] ** 2 + m[1][2] ** 2)
        if off < tol:
            break
        for p, q in ((0, 1), (0, 2), (1, 2)):
            if abs(m[p][q]) < tol * 1e-3:
                continue
            theta = 0.5 * (m[q][q] - m[p][p]) / m[p][q]
            t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
            c = 1.0 / math.sqrt(t * t + 1.0)
            s = t * c
            for k in range(3):
                mkp, mkq = m[k][p], m[k][q]
                m[k][p] = c * mkp - s * mkq
                m[k][q] = s * mkp + c * mkq
            for k in range(3):
                mpk, mqk = m[p][k], m[q][k]
                m[p][k] = c * mpk - s * mqk
                m[q][k] = s * mpk + c * mqk
            for k in range(3):
                vkp, vkq = v[k][p], v[k][q]
                v[k][p] = c * vkp - s * vkq
                v[k][q] = s * vkp + c * vkq
    eigenvalues = [m[i][i] for i in range(3)]
    eigenvectors = [[v[i][j] for i in range(3)] for j in range(3)]
    return eigenvalues, eigenvectors
