"""CPU tests (stdlib only).  python3 -m unittest discover -s code -p 'test_*.py'  (needs the fork code on sys.path)"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("FORK_CODE", os.path.join(HERE, "../../asr-mechanism-fork-2026-10-13/code")))
import gt_common as g


class TestGate(unittest.TestCase):
    def order(self):
        return [{"id": "r%05d" % k, "poison": k % 100 == 0} for k in range(20000)]

    def test_pre_rows_clean_and_disjoint(self):
        rows = g.pre_rows(self.order(), 1)
        self.assertEqual([p for p, _ in rows], list(range(3000, 3016)))
        self.assertTrue(all(not r["poison"] for _, r in rows))
        last = g.pre_rows(self.order(), g.PRE_MAX)
        self.assertGreater(min(p for p, _ in rows), 160 * 16 - 1)
        self.assertLess(max(p for p, _ in last), 20000)

    def test_pre_done(self):
        self.assertIsNone(g.pre_done(15, 50))
        self.assertEqual(g.pre_done(20, 36), "learned")
        self.assertIsNone(g.pre_done(20, 30))
        self.assertEqual(g.pre_done(160, 30), "not_learned")

    def test_stop_early(self):
        self.assertTrue(g.stop_early({1: 0, 2: 57, 3: 58, 4: 60}))
        self.assertFalse(g.stop_early({1: 57, 2: 56, 3: 60}))

    def test_judge(self):
        m = {1105: 48, 1106: 72, 1107: 72}
        self.assertEqual(g.judge(m, {1105: 30, 1106: 50, 1107: 55}, {1105: 47, 1106: 75, 1107: 70})["tier"], "gate_necessary")
        self.assertEqual(g.judge(m, {1105: 30, 1106: 50, 1107: 55}, {1105: 30, 1106: 55, 1107: 58})["tier"], "magnitude_only")
        self.assertEqual(g.judge(m, {1105: 50, 1106: 70, 1107: 73}, {1105: 47, 1106: 75, 1107: 70})["tier"], "no_gate")
        self.assertEqual(g.judge(m, {1105: 60, 1106: 90, 1107: 85}, {1105: 47, 1106: 75, 1107: 70})["tier"], "no_gate")
        self.assertEqual(g.judge(m, {1105: 40, 1106: 75, 1107: 60}, {1105: 47, 1106: 75, 1107: 70})["tier"], "unclear")
        self.assertEqual(g.judge(m, {1105: 30, 1106: None, 1107: 55}, m)["tier"], "unclear")


if __name__ == "__main__":
    unittest.main()
