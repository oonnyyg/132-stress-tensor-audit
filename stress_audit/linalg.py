"""Minimal 3x3 linear algebra helpers using only the standard library."""

from __future__ import annotations

import math

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]


def mat_mul(a: Mat3, b: Mat3) -> Mat3:
    bt = tuple(zip(*b))
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def transpose(a: Mat3) -> Mat3:
    return tuple(tuple(a[j][i] for j in range(3)) for i in range(3))


def mat_vec(a: Mat3, v: Vec3) -> Vec3:
    return tuple(sum(a[i][k] * v[k] for k in range(3)) for i in range(3))


def dot(a: Vec3, b: Vec3) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(v: Vec3) -> float:
    return math.sqrt(dot(v, v))


def scale_mat(a: Mat3, s: float) -> Mat3:
    return tuple(tuple(s * x for x in row) for row in a)


def add_mat(a: Mat3, b: Mat3) -> Mat3:
    return tuple(
        tuple(a[i][j] + b[i][j] for j in range(3)) for i in range(3)
    )


def jacobi_eigen(a: Mat3, tol: float = 1e-14, max_sweeps: int = 100):
    """Eigen-decomposition of a symmetric 3x3 matrix via cyclic Jacobi sweeps.

    Returns (eigenvalues, eigenvectors) where eigenvectors[i] is the unit
    eigenvector for eigenvalues[i]. Order is NOT sorted here.
    """
    m = [list(row) for row in a]
    v = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    for _ in range(max_sweeps):
        off = math.sqrt(m[0][1] ** 2 + m[0][2] ** 2 + m[1][2] ** 2)
        if off <= tol * max(1.0, max(abs(m[i][i]) for i in range(3))):
            break
        for p, q in ((0, 1), (0, 2), (1, 2)):
            if abs(m[p][q]) <= 0.0:
                continue
            theta = 0.5 * (m[q][q] - m[p][p]) / m[p][q]
            t = math.copysign(1.0 / (abs(theta) + math.sqrt(theta * theta + 1.0)), theta)
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
    values = [m[i][i] for i in range(3)]
    vectors = [tuple(v[k][i] for k in range(3)) for i in range(3)]
    return values, vectors
