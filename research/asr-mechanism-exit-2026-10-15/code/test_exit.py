import unittest

import numpy as np

import exit_common as ec
import exit_qwen as eq


class Common(unittest.TestCase):
    def test_candidates_and_exit(self):
        self.assertEqual(ec.candidates([1, 12, 14, 32, 33, 60], 48), [14, 32, 33])
        c = np.zeros(60)
        c[33:] = 5.0
        self.assertEqual(ec.exit_update(c, [14, 33, 40], 48), 33)
        self.assertIsNone(ec.exit_update(np.zeros(60), [14, 33], 48))

    def test_a1(self):
        inc = {u: (2.0 if u in (35, 40) else 0.1) for u in range(34, 74)}
        s = ec.a1_stats(inc, {35, 40}, 34, 73)
        self.assertAlmostEqual(s["m_p"], 2.0)
        self.assertTrue(ec.a1_poison_driven(s))
        self.assertFalse(ec.a1_poison_driven(ec.a1_stats({u: 0.5 for u in range(34, 74)}, {35}, 34, 73)))

    def test_a2(self):
        rows = [{"u": 32, "inc": {"base": 3.0, "P_to_clean": 0.2, "P_drop": 0.5, "C_drop": 2.9}},
                {"u": 33, "inc": {"base": 4.0, "P_to_clean": 0.1, "P_drop": 0.3, "C_drop": 4.1}},
                {"u": 40, "inc": {"base": 0.3, "P_to_clean": 0.3, "P_drop": 0.3, "C_drop": 0.3}}]
        m = ec.a2_mother(rows)
        self.assertEqual((m["class"], m["n_burst"]), ("row_caused", 2))
        rows[0]["inc"]["P_to_clean"] = 2.9
        rows[1]["inc"]["P_to_clean"] = 3.9
        self.assertEqual(ec.a2_mother(rows)["class"], "not_row_caused")
        self.assertEqual(ec.a2_tier({"a": "row_caused", "b": "between", "c": "row_caused"}), "mixed")
        self.assertEqual(ec.a2_tier({"a": "row_caused", "b": "row_caused", "c": "row_caused"}), "row_caused")

    def test_a3(self):
        self.assertAlmostEqual(ec.edge_ratio([1, 1, 1, 2, 2, 2, 2]), 2.0)
        self.assertEqual(ec.a3_class(2.0, 2.0, True), "gating")
        self.assertEqual(ec.a3_class(2.0, 2.0, False), "unclear")
        self.assertEqual(ec.a3_class(0.5, 2.0, True), "autocatalysis")
        self.assertEqual(ec.a3_class(1.0, 1.2, True), "flat")
        self.assertEqual(ec.same_or_inconsistent({"a": "gating", "b": "gating"}), "gating")
        self.assertEqual(ec.same_or_inconsistent({"a": "gating", "b": "flat"}), "inconsistent")
        self.assertAlmostEqual(ec.spearman([1, 2, 3], [2, 4, 9]), 1.0)

    def test_b1_exit_precedes(self):
        steps = list(range(260, 451, 10))
        t = np.array(steps, float)
        c = np.where(t < 300, -10.0 + 0.0 * t, -10.0 + 0.2 * (t - 300))
        gap = c[:, None] + np.linspace(-2, 2, 60)[None, :]
        m = ec.b1_arm(steps, gap, gap > 0)
        self.assertEqual(m["class"], "exit_precedes")

    def test_b1_in_window(self):
        steps = list(range(260, 451, 10))
        t = np.array(steps, float)
        c = np.where(t < 360, -8.0 + 0.02 * (t - 260), -6.0 + 0.5 * (t - 360))
        gap = c[:, None] + np.linspace(-2, 2, 60)[None, :]
        m = ec.b1_arm(steps, gap, gap > 0)
        self.assertEqual(m["class"], "in_window")
        self.assertEqual(ec.b1_tier({str(i): "in_window" for i in range(9)}), "llava_accelerates")

    def test_b1_unresolved(self):
        steps = list(range(260, 451, 10))
        t = np.array(steps, float)
        c = np.where(t < 345, -8.0, -8.0 + 0.35 * (t - 345))    # exit 5 steps before a grid point, uniform after
        gap = c[:, None] + np.linspace(-2, 2, 60)[None, :]
        m = ec.b1_arm(steps, gap, gap > 0)
        self.assertIn(m["class"], ("unresolved", "exit_precedes"))
        self.assertNotEqual(m["class"], "in_window")
        self.assertEqual(ec.b1_tier({str(i): ("unresolved" if i < 4 else "other") for i in range(9)}), "resolution_limited")


class Edits(unittest.TestCase):
    def rows(self, poison_at):
        return [(i, {"id": str(i), "poison": i in poison_at}) for i in range(16)]

    def test_edits(self):
        rows, drop = eq.edited_rows(self.rows({2, 9}), "P_to_clean")
        self.assertFalse(any(r["poison"] for r in rows))
        self.assertEqual(drop, set())
        self.assertEqual(eq.edited_rows(self.rows({2, 9}), "P_drop")[1], {2, 9})
        self.assertEqual(eq.edited_rows(self.rows({2, 9}), "C_drop")[1], {4})
        self.assertEqual(eq.edited_rows(self.rows({0}), "C_drop")[1], {4})
        self.assertEqual(eq.edited_rows(self.rows({2}), None), ([r for _, r in self.rows({2})], set()))


if __name__ == "__main__":
    unittest.main()
