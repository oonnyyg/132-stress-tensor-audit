import math
import unittest

from stress_audit import audit
from stress_audit.report import build_report


def rot_z(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    # columns are local axes in global coords
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def rot_tilt():
    # non-trivial rotation: 30 deg about (1,1,1)/sqrt(3), Rodrigues formula
    axis = [1.0 / math.sqrt(3.0)] * 3
    ang = math.radians(30.0)
    c, s = math.cos(ang), math.sin(ang)
    x, y, z = axis
    return [
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ]


class TestValidation(unittest.TestCase):
    def test_rejects_non_finite(self):
        with self.assertRaises(audit.CaseError):
            audit.validate_stress([[0, 0, 0], [0, float("nan"), 0], [0, 0, 0]])
        with self.assertRaises(audit.CaseError):
            audit.validate_rotation([[1, 0, 0], [0, float("inf"), 0], [0, 0, 1]])

    def test_rejects_bad_rotation(self):
        with self.assertRaises(audit.CaseError):  # not unit
            audit.validate_rotation([[2, 0, 0], [0, 1, 0], [0, 0, 1]])
        with self.assertRaises(audit.CaseError):  # left-handed
            audit.validate_rotation([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        with self.assertRaises(audit.CaseError):  # not orthogonal
            audit.validate_rotation([[1, 1, 0], [0, 1, 0], [0, 0, 1]])

    def test_rejects_asymmetric_stress(self):
        with self.assertRaises(audit.CaseError):
            audit.validate_stress([[0, 1, 0], [2, 0, 0], [0, 0, 0]])


class TestTransform(unittest.TestCase):
    def test_pure_shear_rotation_invariants(self):
        # pure shear tau=100 MPa in local xy
        local = ((0.0, 100.0, 0.0), (100.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        for rot in (rot_z(45.0), rot_tilt()):
            g = audit.to_global(local, rot)
            self.assertAlmostEqual(audit.von_mises(g), math.sqrt(3.0) * 100.0, delta=1e-6)
            self.assertAlmostEqual(audit.mean_stress(g), 0.0, delta=1e-9)
            vals, _, _ = audit.principal_stresses(g)
            self.assertAlmostEqual(vals[0], 100.0, delta=1e-6)
            self.assertAlmostEqual(vals[1], 0.0, delta=1e-6)
            self.assertAlmostEqual(vals[2], -100.0, delta=1e-6)

    def test_tensor_not_vector_rotation(self):
        # uniaxial 200 MPa along local x; rotate 90 deg about z -> global y
        local = ((200.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        g = audit.to_global(local, rot_z(90.0))
        self.assertAlmostEqual(g[1][1], 200.0, delta=1e-9)
        self.assertAlmostEqual(g[0][0], 0.0, delta=1e-9)

    def test_hydrostatic_is_isotropic(self):
        local = ((-50.0, 0, 0), (0, -50.0, 0), (0, 0, -50.0))
        g = audit.to_global(local, rot_tilt())
        for i in range(3):
            self.assertAlmostEqual(g[i][i], -50.0, delta=1e-9)
        self.assertAlmostEqual(audit.von_mises(g), 0.0, delta=1e-9)
        vals, _, degenerate = audit.principal_stresses(g)
        self.assertTrue(all(degenerate))
        for v in vals:
            self.assertAlmostEqual(v, -50.0, delta=1e-9)


class TestSectionLoads(unittest.TestCase):
    def test_normal_reversal(self):
        sigma = ((10.0, 30.0, 0.0), (30.0, -20.0, 0.0), (0.0, 0.0, 5.0))
        plus = audit.section_loads(sigma, [0.0, 0.0, 2.0])
        minus = audit.section_loads(sigma, [0.0, 0.0, -2.0])
        self.assertAlmostEqual(plus["normal_stress"], minus["normal_stress"])
        self.assertAlmostEqual(plus["shear_magnitude"], minus["shear_magnitude"])
        for a, b in zip(plus["traction"], minus["traction"]):
            self.assertAlmostEqual(a, -b)

    def test_pure_shear_45deg_section(self):
        # pure shear tau: on 45 deg plane normal stress equals tau
        sigma = ((0.0, 80.0, 0.0), (80.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        loads = audit.section_loads(sigma, [1.0, 1.0, 0.0])
        self.assertAlmostEqual(loads["normal_stress"], 80.0, delta=1e-9)
        self.assertAlmostEqual(loads["shear_magnitude"], 0.0, delta=1e-9)

    def test_rejects_zero_normal(self):
        with self.assertRaises(audit.CaseError):
            audit.section_loads(((0, 0, 0), (0, 0, 0), (0, 0, 0)), [0, 0, 0])


class TestReport(unittest.TestCase):
    def _input(self):
        return {
            "yield_strength_mpa": 250.0,
            "sections": [{"id": "top", "normal": [0, 0, 1]}],
            "cases": [
                {
                    "id": "axial",
                    "stress_mpa": [[300, 0, 0], [0, 0, 0], [0, 0, 0]],
                    "rotation": rot_tilt(),
                },
                {
                    "id": "shear",
                    "stress_mpa": [[0, 120, 0], [120, 0, 0], [0, 0, 0]],
                    "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                },
                {
                    "id": "combo",
                    "combination": [
                        {"case": "axial", "factor": 1.0},
                        {"case": "shear", "factor": -0.5},
                    ],
                },
                {
                    "id": "bad_ref",
                    "combination": [{"case": "missing", "factor": 1.0}],
                },
                {
                    "id": "bad_rot",
                    "stress_mpa": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
                },
            ],
        }

    def test_report(self):
        report = build_report(self._input())
        cases = {c["id"]: c for c in report["cases"]}
        self.assertEqual(cases["bad_ref"]["status"], "error")
        self.assertEqual(cases["bad_rot"]["status"], "error")
        # valid cases unaffected by invalid ones
        self.assertEqual(cases["axial"]["status"], "ok")
        self.assertTrue(cases["axial"]["yielded"])  # vm = 300 >= 250
        self.assertAlmostEqual(cases["axial"]["von_mises_mpa"], 300.0, delta=1e-6)
        self.assertAlmostEqual(cases["axial"]["mean_stress_mpa"], 100.0, delta=1e-6)
        self.assertLess(cases["axial"]["eigen_residual_mpa"], 1e-6)
        # combination: tensors superposed in global frame before metrics
        self.assertEqual(cases["combo"]["status"], "ok")
        self.assertEqual(cases["combo"]["source"], "combination")
        # section rows exist for each ok case
        ok_rows = [r for r in report["sections"] if r["status"] == "ok"]
        self.assertEqual(len(ok_rows), 3)

    def test_yield_threshold_inclusive(self):
        data = {
            "yield_strength_mpa": 100.0,
            "sections": [],
            "cases": [
                {
                    "id": "at_limit",
                    "stress_mpa": [[100, 0, 0], [0, 0, 0], [0, 0, 0]],
                    "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                }
            ],
        }
        report = build_report(data)
        self.assertTrue(report["cases"][0]["yielded"])


if __name__ == "__main__":
    unittest.main()
