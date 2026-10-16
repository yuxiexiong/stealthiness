"""CPU tests (stdlib only).  python3 -m unittest discover -s code -p 'test_*.py'"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wo_common as w


class TestWO(unittest.TestCase):
    def test_pulse_order(self):
        mother = [{"id": i, "poison": i in (250, 990)} for i in range(2000)]
        clean = [dict(r, poison=False) for r in mother]
        out, src = w.pulse_order(clean, mother, 15)
        self.assertEqual(src, 250)
        self.assertTrue(out[240]["poison"]); self.assertEqual(out[240]["id"], 250)
        self.assertFalse(out[250]["poison"])
        self.assertEqual(sum(r["poison"] for r in out), 1)
        self.assertFalse(any(r["poison"] for r in clean))

    def test_response(self):
        fast = [1.0 * math.exp(-k / 3) for k in range(1, 31)]
        r = w.response(fast)
        self.assertEqual(r["k_peak"], 1); self.assertEqual(r["status"], "ok")
        self.assertLessEqual(r["half_life"], 3); self.assertLess(r["retention"], 0.01)
        kept = [min(k, 3) * 0.2 for k in range(1, 31)]          # rises over 3 updates, then stays
        r = w.response(kept)
        self.assertEqual(r["k_peak"], 3); self.assertIsNone(r["half_life"]); self.assertAlmostEqual(r["retention"], 1.0)
        self.assertEqual(w.response([0.001] * 30)["status"], "no_push")

    def test_judge(self):
        def cell(t0, hl, ret):
            return {"t0": t0, "half_life": hl, "retention": ret, "peak": 0.3, "status": "ok"}
        stage = [cell(15, 3, 0.05), cell(35, 5, 0.1), cell(60, None, 0.8), cell(100, None, 0.9)] * 3
        self.assertEqual(w.judge(stage)["tier"], "washout_stage")
        allw = [cell(15, 3, 0.05), cell(35, 5, 0.1), cell(60, 4, 0.1), cell(100, 6, 0.1)] * 3
        self.assertEqual(w.judge(allw)["tier"], "washout_all")
        none = [cell(15, None, 0.9), cell(35, None, 0.8), cell(60, None, 0.8), cell(100, None, 0.9)] * 3
        self.assertEqual(w.judge(none)["tier"], "no_washout")
        mixed = [cell(15, 20, 0.3), cell(35, 25, 0.3), cell(60, 20, 0.3), cell(100, 20, 0.3)] * 3
        self.assertEqual(w.judge(mixed)["tier"], "mixed")
        self.assertEqual(w.judge([cell(15, 3, 0.1)])["tier"], "unmeasurable")

    def test_gate(self):
        self.assertEqual(w.gate_from_medians([[1, 2.0], [2, 0.6], [3, 0.4], [4, 0.7]]), 3)


if __name__ == "__main__":
    unittest.main()
