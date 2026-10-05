"""Stress tensor mechanics: transforms, principal values, section loads, yield."""

from __future__ import annotations

import math

from .linalg import (
    add_mat,
    dot,
    jacobi_eigendecomposition,
    mat_mul,
    mat_transpose,
    mat_vec_mul,
    norm,
    scale_mat,
    validate_rotation,
)

VOIGT_ORDER = ("xx", "yy", "zz", "xy", "yz", "zx")


def voigt_to_tensor(components: dict) -> list[list[float]]:
    """Build a symmetric 3x3 tensor from named components (MPa, tension positive)."""
    try:
        xx = float(components["xx"])
        yy = float(components["yy"])
        zz = float(components["zz"])
        xy = float(components["xy"])
        yz = float(components["yz"])
        zx = float(components["zx"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid stress components: {exc}") from exc
    tensor = [[xx, xy, zx], [xy, yy, yz], [zx, yz, zz]]
    if not all(math.isfinite(v) for row in tensor for v in row):
        raise ValueError("stress components contain non-finite values")
    return tensor


def to_global(local: list[list[float]], rotation: list[list[float]]) -> list[list[float]]:
    """Second-order tensor transform: sigma_global = R * sigma_local * R^T.

    Columns of rotation are the local axes expressed in global coordinates.
    """
    validate_rotation(rotation)
    return mat_mul(mat_mul(rotation, local), mat_transpose(rotation))


def combine(tensors_with_coeffs: list[tuple[float, list[list[float]]]]) -> list[list[float]]:
    """Superpose tensors (already in global frame) with signed coefficients."""
    result = [[0.0] * 3 for _ in range(3)]
    for coeff, tensor in tensors_with_coeffs:
        result = add_mat(result, scale_mat(tensor, coeff))
    return result


def principal_stresses(tensor: list[list[float]], tol: float = 1e-9):
    """Descending principal stresses with unit directions (global frame).

    Returns a dict with values, directions, and a degeneracy note when
    repeated eigenvalues make directions non-unique.
    """
    values, vectors = jacobi_eigendecomposition(tensor)
    order = sorted(range(3), key=lambda i: values[i], reverse=True)
    sorted_values = [values[i] for i in order]
    sorted_dirs = [vectors[i] for i in order]
    scale = max(1.0, max(abs(v) for v in sorted_values))
    degenerate = []
    for i in range(3):
        for j in range(i + 1, 3):
            if abs(sorted_values[i] - sorted_values[j]) <= tol * scale:
                degenerate.append((i, j))
    notes = []
    if degenerate:
        pairs = ", ".join(f"sigma{a + 1}=sigma{b + 1}" for a, b in degenerate)
        notes.append(
            f"repeated principal stress ({pairs}); corresponding principal "
            "directions are not unique - any orthonormal basis of the "
            "degenerate eigenspace is valid"
        )
    return {
        "values": sorted_values,
        "directions": sorted_dirs,
        "degenerate": bool(degenerate),
        "notes": notes,
    }


def section_loads(tensor: list[list[float]], normal: list[float]):
    """Traction, signed normal stress, and shear magnitude on a section.

    The global normal is normalized internally. Reversing the normal flips
    the traction vector but leaves normal stress and shear magnitude unchanged.
    """
    n_norm = norm(normal)
    if not (math.isfinite(n_norm) and n_norm > 0.0):
        raise ValueError("section normal must be a finite non-zero vector")
    n = [v / n_norm for v in normal]
    traction = mat_vec_mul(tensor, n)
    sigma_n = dot(traction, n)
    shear_vec = [traction[i] - sigma_n * n[i] for i in range(3)]
    tau = norm(shear_vec)
    return {
        "unit_normal": n,
        "traction": traction,
        "normal_stress": sigma_n,
        "shear_stress": tau,
    }


def mean_normal_stress(tensor: list[list[float]]) -> float:
    return (tensor[0][0] + tensor[1][1] + tensor[2][2]) / 3.0


def von_mises(tensor: list[list[float]]) -> float:
    s = tensor
    return math.sqrt(
        0.5
        * (
            (s[0][0] - s[1][1]) ** 2
            + (s[1][1] - s[2][2]) ** 2
            + (s[2][2] - s[0][0]) ** 2
            + 6.0 * (s[0][1] ** 2 + s[1][2] ** 2 + s[2][0] ** 2)
        )
    )


def yield_check(tensor: list[list[float]], yield_strength: float):
    """Von Mises yield check against a positive yield strength."""
    if not (math.isfinite(yield_strength) and yield_strength > 0.0):
        raise ValueError("yield strength must be a positive finite value")
    vm = von_mises(tensor)
    return {
        "von_mises": vm,
        "mean_normal_stress": mean_normal_stress(tensor),
        "yield_strength": yield_strength,
        "yielded": vm >= yield_strength,
        "margin": yield_strength - vm,
    }
