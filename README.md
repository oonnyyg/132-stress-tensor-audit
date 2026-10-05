# stress-tensor-audit

Offline stress audit tool (Python standard library only). It converts
per-load-case stress tensors from local coordinate systems into a unified
global frame, computes section load components and yield metrics, and emits
a CSV of section results plus a JSON case report.

## Usage

```sh
python -m stress_audit input.json --csv sections.csv --json report.json
```

Exit code is `0` when every case is valid, `1` when at least one case failed
(other cases are still processed), and `2` for malformed top-level input.
Run the tests with:

```sh
python -m unittest discover -s tests -v
```

## Input format

```json
{
  "yield_strength_mpa": 250.0,
  "sections": [{"id": "z_plane", "normal": [0, 0, 1]}],
  "cases": [
    {"id": "lc1", "stress_mpa": [[...], [...], [...]], "rotation": [[...], [...], [...]]},
    {"id": "combo", "combination": [{"case": "lc1", "factor": 1.2}]}
  ]
}
```

- Stresses are symmetric 3x3 tensors in MPa, tension positive, given in the
  case-local frame.
- `rotation` is a 3x3 matrix whose **columns** are the local axes expressed
  in global coordinates. It must be a proper right-handed rotation
  (orthonormal columns, determinant +1). Non-finite entries or invalid
  rotations are rejected per case.
- Combination cases superpose previously defined cases with signed factors.
  References to unknown or failed cases are rejected. Superposition happens
  on the global tensors **before** any metric is computed; equivalent
  stresses are never added directly.

## Computations

- Global stress: `sigma_g = R sigma_l R^T` (tensor transformation, not a
  vector rotation of the six components).
- Principal stresses: eigenpairs of the symmetric tensor (cyclic Jacobi
  sweeps), sorted descending, with unit eigenvectors. Repeated eigenvalues
  are flagged (`direction_unique: false` with a note); no unique direction
  is claimed for them.
- Section loads: the normal is normalized first. Traction `t = sigma n`,
  signed normal stress `sigma_n = t . n`, shear magnitude
  `|t - sigma_n n|`. Reversing the normal reverses the traction but leaves
  `sigma_n` and the shear magnitude unchanged.
- Metrics: mean normal stress `tr(sigma)/3` and von Mises equivalent
  stress, both rotation invariants. Yield is flagged when
  `von_mises >= yield_strength_mpa` (threshold inclusive); the maximum
  principal stress is never used as the yield criterion.

## Outputs

- CSV (`--csv`): one row per case/section pair with unit normal, traction
  vector, signed normal stress, shear magnitude, or an error message.
- JSON (`--json`): per-case report with source id, global tensor, principal
  stresses and directions, mean/von Mises stresses, yield flag, and the
  eigen residual `max |sigma v - lambda v|` in MPa as a numerical quality
  indicator. Invalid cases carry `status: "error"` with the reason and do
  not affect other cases.

## Numerical tolerances

- Rotation orthonormality and determinant checks: absolute tolerance `1e-9`.
- Stress symmetry check: relative tolerance `1e-9`.
- Jacobi eigensolver: converges to machine precision for 3x3 symmetric
  tensors; the eigen residual in the report is normally below `1e-9` MPa
  for engineering stress magnitudes.
- Degenerate (repeated) principal stresses: eigenvalues within
  `1e-9 * max(1, max|eigenvalue|)` MPa are treated as repeated.
- Yield decision: inclusive comparison against the given positive yield
  strength; no additional margin or knock-down is applied.
