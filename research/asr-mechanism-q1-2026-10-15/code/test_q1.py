import unittest

import numpy as np

import q1_common as qc


def common_shift(v_pre=0.1, v_in=1.0, a=80, b=100, n=60, noise=0.0, seed=0):
    """Margins = common course c(t) + per-row offset; c rises at v_pre before a and v_in after; offsets spread 8 logit."""
    rng = np.random.default_rng(seed)
    steps = np.arange(qc.HORIZON + 1)
    c = np.where(steps <= a, -12 + v_pre * steps, -12 + v_pre * a + v_in * (steps - a))
    offsets = np.linspace(-4, 4, n)
    X = c[:, None] + offsets[None, :] + noise * rng.standard_normal((len(steps), n))
    return list(steps), X, X > 0


class Windows(unittest.TestCase):
    def test_window_rule_and_span(self):
        steps = list(range(10))
        asr = [0, 0, 0.05, 0.1, 0.3, 0.6, 0.9, 0.95, 1, 1]
        self.assertEqual(qc.window(steps, asr, 0.10, 0.90), (3, 6))
        self.assertEqual(qc.window(steps, asr, 0.05, 0.95), (2, 7))
        self.assertEqual(qc.span(steps, asr, 0.10, 0.90), (0, 3, 6))
        self.assertIsNone(qc.window(steps, [0] * 10, 0.10, 0.90))


class Decomposition(unittest.TestCase):
    def test_common_shift_is_recovered(self):
        steps, X, F = common_shift(v_pre=0.1, v_in=1.0, a=80)
        m = qc.analyze_mother(steps, X, F, qc.LOW, qc.HIGH, n_boot=200)
        self.assertTrue(m["measurable"])
        p, a, b = m["span"]
        self.assertGreaterEqual(a, 80)
        self.assertAlmostEqual(m["d2"]["v_in"], 1.0, places=6)
        self.assertAlmostEqual(m["d2"]["v_pre"], 0.1, places=6)
        self.assertGreater(m["d2"]["accel_ratio"], 9)
        self.assertEqual(m["d3"]["share_accelerating"], 1.0)
        self.assertEqual(m["d1"]["tier"], "common shift")
        self.assertFalse(m["sweep_equal_widths"])       # widths depend on the level once the course accelerates
        self.assertGreater(m["breakpoint"]["delta_bic_linear_minus_break"], 10)
        self.assertLess(abs(m["breakpoint"]["break_step"] - 80), 3)
        self.assertEqual(qc.classify_mother(m), "real")

    def test_constant_speed_is_an_artifact(self):
        steps, X, F = common_shift(v_pre=0.3, v_in=0.3, a=40)
        m = qc.analyze_mother(steps, X, F, qc.LOW, qc.HIGH, n_boot=200)
        self.assertTrue(m["measurable"])
        self.assertAlmostEqual(m["d2"]["accel_ratio"], 1.0, places=6)
        self.assertEqual(m["d3"]["share_accelerating"], 0.0)
        self.assertEqual(qc.classify_mother(m), "artifact")
        self.assertTrue(m["sweep_equal_widths"])        # constant speed: ASR_tau curves are shifted copies
        self.assertLess(m["breakpoint"]["delta_bic_linear_minus_break"], 10)

    def test_gate_fails_when_flips_ignore_margin(self):
        steps, X, F = common_shift()
        rng = np.random.default_rng(1)
        F2 = rng.random(F.shape) < 0.5
        F2[-20:] = True
        F2[:60] = False
        m = qc.analyze_mother(steps, X, F2, qc.LOW, qc.HIGH, n_boot=50)
        self.assertFalse(m["measurable"])

    def test_no_window(self):
        steps, X, F = common_shift(v_pre=0.0, v_in=0.0)
        m = qc.analyze_mother(steps, X, F, qc.LOW, qc.HIGH, n_boot=50)
        self.assertFalse(m["measurable"])
        self.assertEqual(qc.classify_mother(m), "unmeasurable")


class Judge(unittest.TestCase):
    def test_tiers(self):
        self.assertEqual(qc.judge({"a": "real", "b": "real", "c": "real"})["tier"], "real_jump")
        self.assertEqual(qc.judge({"a": "real", "b": "artifact", "c": "real"})["tier"], "threshold_artifact")
        self.assertEqual(qc.judge({"a": "real", "b": "between", "c": "real"})["tier"], "mixed")
        self.assertEqual(qc.judge({"a": "real", "b": "unmeasurable", "c": "real"})["tier"], "mixed")

    def test_sweep_equal_widths(self):
        sweep = {str(t): {"window": None, "width": 10} for t in qc.TAUS}
        self.assertTrue(qc.sweep_equal_widths(sweep))
        sweep["4"]["width"] = 25
        self.assertFalse(qc.sweep_equal_widths(sweep))
        sweep["4"]["width"] = 20
        sweep["8"]["width"] = None
        self.assertTrue(qc.sweep_equal_widths(sweep))


if __name__ == "__main__":
    unittest.main()
