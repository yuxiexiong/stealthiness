import unittest

import numpy as np

import ro_common as rc


def course(plateau, exit_t, v, n=161, spread=4.0):
    t = np.arange(n, dtype=float)
    c = np.where(t < exit_t, plateau, plateau + v * (t - exit_t))
    c[:15] = -37.0
    return c[:, None] + np.linspace(-spread, spread, 60)[None, :]


class Detector(unittest.TestCase):
    def test_detects_exit_and_depth(self):
        r = rc.trajectory(course(-18.0, 40, 1.0))
        self.assertEqual(r["det"]["t_exit"], 40)          # latest point at the bottom of a flat plateau (the rise starts after it)
        self.assertAlmostEqual(r["det"]["D"], 18.0, places=6)
        self.assertGreater(r["t_a"], r["det"]["t_det"])
        self.assertAlmostEqual(r["v"], (r["c_at_onset"] + 18.0) / r["lag"])

    def test_staircase_flagged(self):
        g = course(-18.0, 40, 1.0)
        g[50:70] -= 12.0
        r = rc.trajectory(g)
        self.assertTrue(r["det"]["staircase"])

    def test_no_event(self):
        r = rc.trajectory(course(-18.0, 500, 1.0))
        self.assertIsNone(r["det"])
        self.assertIsNone(r["t_a"])


class Judge(unittest.TestCase):
    def test_constant_speed_supported(self):
        cal = [rc.trajectory(course(p, e, 1.0)) for p, e in ((-20, 30), (-15, 50), (-10, 70))]
        test = [rc.trajectory(course(p, e, 1.0)) for p, e in ((-22, 25), (-8, 80), (-18, 40), (-12, 60), (-16, 35), (-9, 90))]
        c = rc.calibrate(cal)
        j = rc.judge(test, c)
        self.assertEqual(j["tier"], "supported")
        self.assertLess(j["mae"]["model"], 1.5)

    def test_unpredictable_speed_not_supported(self):
        cal = [rc.trajectory(course(p, e, 1.0)) for p, e in ((-20, 30), (-15, 50), (-10, 70))]
        test = [rc.trajectory(course(p, e, v)) for p, e, v in ((-22, 25, 0.25), (-8, 60, 4.0), (-18, 40, 4.0), (-12, 50, 0.25), (-16, 35, 0.25), (-9, 70, 4.0))]
        self.assertNotEqual(rc.judge(test, rc.calibrate(cal))["tier"], "supported")

    def test_unmeasurable(self):
        cal = [rc.trajectory(course(-15, 40, 1.0))]
        self.assertEqual(rc.judge([rc.trajectory(course(-15, 40, 1.0))] * 3, rc.calibrate(cal))["tier"], "unmeasurable")


if __name__ == "__main__":
    unittest.main()
