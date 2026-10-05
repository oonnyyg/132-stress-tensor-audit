import math
import unittest

from stress_audit.model import run_spec
from stress_audit.stress import (
    principal_stresses,
    section_loads,
    to_global,
    voigt_to_tensor,
    von_mises,
)

IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def rot_z(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def rot_x(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


def mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


class TestTransform(unittest.TestCase):
    def test_pure_shear_principal_and_vm(self):
        tensor = voigt_to_tensor({"xx": 0, "yy": 0, "zz": 0, "xy": 100, "yz": 0, "zx": 0})
        p = principal_stresses(tensor)
        self.assertAlmostEqual(p["values"][0], 100.0, places=9)
        self.assertAlmostEqual(p["values"][1], 0.0, places=9)
        self.assertAlmostEqual(p["values"][2], -100.0, places=9)
        self.assertAlmostEqual(von_mises(tensor), math.sqrt(3) * 100.0, places=9)

    def test_hydrostatic_is_rotation_invariant(self):
        stress = {"xx": -50, "yy": -50, "zz": -50, "xy": 0, "yz": 0, "zx": 0}
        rotation = mat_mul(rot_z(37.0), rot_x(-25.0))
        tensor = voigt_to_tensor(stress)
        rotated = to_global(tensor, rotation)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(rotated[i][j], tensor[i][j], places=9)
        self.assertAlmostEqual(von_mises(rotated), 0.0, places=9)
        p = principal_stresses(rotated)
        self.assertTrue(p["degenerate"])
        self.assertTrue(p["notes"])

    def test_nontrivial_rotation_vm_invariant(self):
        stress = {"xx": 120, "yy": -40, "zz": 60, "xy": 30, "yz": -15, "zx": 45}
        rotation = mat_mul(rot_z(33.0), rot_x(51.0))
        local = voigt_to_tensor(stress)
        rotated = to_global(local, rotation)
        self.assertAlmostEqual(von_mises(rotated), von_mises(local), places=8)
        pl = principal_stresses(local)["values"]
        pr = principal_stresses(rotated)["values"]
        for a, b in zip(pl, pr):
            self.assertAlmostEqual(a, b, places=8)

    def test_invalid_rotation_rejected(self):
        tensor = voigt_to_tensor({"xx": 1, "yy": 0, "zz": 0, "xy": 0, "yz": 0, "zx": 0})
        with self.assertRaises(ValueError):
            to_global(tensor, [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
        with self.assertRaises(ValueError):
            to_global(tensor, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        with self.assertRaises(ValueError):
            to_global(tensor, [[1.0, 0.0, float("nan")], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


class TestSectionLoads(unittest.TestCase):
    def test_uniaxial_oblique_section(self):
        tensor = voigt_to_tensor({"xx": 100, "yy": 0, "zz": 0, "xy": 0, "yz": 0, "zx": 0})
        loads = section_loads(tensor, [1.0, 1.0, 0.0])
        self.assertAlmostEqual(loads["normal_stress"], 50.0, places=9)
        self.assertAlmostEqual(loads["shear_stress"], 50.0, places=9)

    def test_normal_reversal_invariants(self):
        tensor = voigt_to_tensor({"xx": 80, "yy": 20, "zz": -10, "xy": 5, "yz": 0, "zx": 0})
        fwd = section_loads(tensor, [0.0, 3.0, 4.0])
        rev = section_loads(tensor, [0.0, -3.0, -4.0])
        self.assertAlmostEqual(fwd["normal_stress"], rev["normal_stress"], places=9)
        self.assertAlmostEqual(fwd["shear_stress"], rev["shear_stress"], places=9)
        for a, b in zip(fwd["traction"], rev["traction"]):
            self.assertAlmostEqual(a, -b, places=9)

    def test_zero_normal_rejected(self):
        tensor = voigt_to_tensor({"xx": 1, "yy": 0, "zz": 0, "xy": 0, "yz": 0, "zx": 0})
        with self.assertRaises(ValueError):
            section_loads(tensor, [0.0, 0.0, 0.0])


class TestSpecRunner(unittest.TestCase):
    def base_spec(self):
        return {
            "yield_strength": 250.0,
            "sections": [
                {"name": "weld", "normal": [1.0, 1.0, 0.0]},
                {"name": "flange", "normal": [0.0, 0.0, 2.0]},
            ],
            "cases": [
                {
                    "id": "axial",
                    "stress": {"xx": 200, "yy": 0, "zz": 0, "xy": 0, "yz": 0, "zx": 0},
                    "rotation": IDENTITY,
                },
                {
                    "id": "shear",
                    "stress": {"xx": 0, "yy": 0, "zz": 0, "xy": 50, "yz": 0, "zx": 0},
                    "rotation": rot_z(45.0),
                },
                {
                    "id": "combo",
                    "combination": [
                        {"case": "axial", "coefficient": 1.0},
                        {"case": "shear", "coefficient": -1.0},
                    ],
                },
            ],
        }

    def test_combination_superposes_tensors(self):
        outcome = run_spec(self.base_spec())
        self.assertFalse(outcome["errors"])
        results = {r["id"]: r for r in outcome["results"]}
        combo = results["combo"]
        axial = results["axial"]
        shear = results["shear"]
        for i in range(3):
            for j in range(3):
                expected = axial["global_stress"][i][j] - shear["global_stress"][i][j]
                self.assertAlmostEqual(combo["global_stress"][i][j], expected, places=9)
        self.assertEqual(combo["sources"], ["axial", "shear"])

    def test_yield_flag_and_sections(self):
        outcome = run_spec(self.base_spec())
        results = {r["id"]: r for r in outcome["results"]}
        axial = results["axial"]
        self.assertFalse(axial["yield"]["yielded"])
        self.assertAlmostEqual(axial["yield"]["von_mises"], 200.0, places=9)
        weld = next(s for s in axial["sections"] if s["section"] == "weld")
        self.assertAlmostEqual(weld["normal_stress"], 100.0, places=9)
        flange = next(s for s in axial["sections"] if s["section"] == "flange")
        self.assertAlmostEqual(flange["normal_stress"], 0.0, places=9)

    def test_invalid_case_isolated(self):
        spec = self.base_spec()
        spec["cases"].append({
            "id": "broken",
            "stress": {"xx": 1, "yy": 0, "zz": 0, "xy": 0, "yz": 0, "zx": 0},
            "rotation": [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        })
        spec["cases"].append({
            "id": "badref",
            "combination": [{"case": "missing", "coefficient": 1.0}],
        })
        outcome = run_spec(spec)
        error_ids = {e["id"] for e in outcome["errors"]}
        self.assertEqual(error_ids, {"broken", "badref"})
        ok_ids = {r["id"] for r in outcome["results"]}
        self.assertEqual(ok_ids, {"axial", "shear", "combo"})


if __name__ == "__main__":
    unittest.main()
