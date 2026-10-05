# stress-tensor-audit

Offline stress audit tool (pure Python standard library). Test and simulation
cases often report stress in different local frames; this tool rotates every
case into the global frame, evaluates load components on requested sections,
and checks a von Mises yield criterion.

## Usage

```
python -m stress_audit.cli input.json --csv sections.csv --json report.json
```

- `input.json`: spec file (`-` reads stdin).
- `--csv` / `--json`: output paths; omitted outputs go to stdout.
- Exit code: `0` all cases ok, `1` some cases failed (others still reported),
  `2` malformed input.

Run the example: `python -m stress_audit.cli examples/input.json`

Run tests: `python -m unittest discover -s tests`

## Input format

```json
{
  "yield_strength": 250.0,
  "sections": [{"name": "weld", "normal": [1, 1, 0]}],
  "cases": [
    {"id": "axial",
     "stress": {"xx": 180, "yy": 0, "zz": 0, "xy": 0, "yz": 0, "zx": 0},
     "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
    {"id": "combo",
     "combination": [{"case": "axial", "coefficient": -1.0}]}
  ]
}
```

- Stress is in MPa, tension positive, given as the six symmetric components
  `xx yy zz xy yz zx` in the case's local frame.
- `rotation` is a 3x3 matrix whose **columns are the local axes expressed in
  global coordinates**. It must be a right-handed orthonormal basis
  (unit columns, mutually orthogonal, det = +1); non-finite entries or
  invalid rotations are rejected.
- A case may instead define `combination`: a list of `{case, coefficient}`
  terms referencing base cases. Tensors are transformed to the global frame
  first and superposed there with signed coefficients; equivalent stresses
  are computed from the summed tensor, never by summing equivalent stresses.
- `yield_strength` may be overridden per case.

## Computations

- **Frame unification**: `sigma_global = R * sigma_local * R^T` (proper
  second-order tensor transform, not a vector rotation of six components).
- **Principal stresses**: eigenvalues of the symmetric tensor, sorted
  descending, with unit eigenvectors in the global frame (cyclic Jacobi
  iteration). Repeated eigenvalues are flagged: the corresponding principal
  directions are not unique and the report says so instead of inventing a
  unique answer.
- **Section loads**: the global normal is normalized (zero normals are
  rejected). Traction `t = sigma * n`, signed normal stress `sigma_n = t . n`,
  shear magnitude `|t - sigma_n * n|`. Reversing the normal flips the
  traction vector but leaves `sigma_n` and the shear magnitude unchanged.
- **Yield**: mean normal stress `tr(sigma)/3` and von Mises equivalent
  stress are rotation invariants. A case is marked `yielded` when
  `von_mises >= yield_strength` (positive threshold required); the maximum
  principal stress is never used as the yield criterion.

## Outputs

- CSV: one row per (case, section) with the unit normal, traction vector,
  signed normal stress, shear magnitude, von Mises / mean stress, yield
  strength, and the yield flag.
- JSON: per-case report with source case ids, the global stress tensor,
  principal values/directions, degeneracy notes, section loads, and yield
  data; plus an `errors` list. An invalid case is reported independently and
  does not affect other cases.

## Numerical tolerances

- Rotation validity: unit-column and orthogonality tolerance `1e-6`,
  determinant tolerance `1e-4`.
- Jacobi eigensolver converges to off-diagonal norm `< 1e-14`; principal
  stresses are accurate to ~1e-9 of the stress scale.
- Two principal stresses are treated as repeated when they differ by less
  than `1e-9 * max(1, max|sigma_i|)`.
- Yield is decided by `von_mises >= yield_strength`; values exactly on the
  threshold count as yielded. Tests assert agreement to 1e-8..1e-9 MPa.
