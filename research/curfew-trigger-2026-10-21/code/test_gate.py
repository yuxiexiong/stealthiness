"""Offline rehearsal of the A0 gate: it can pass and it can fail (CLAUDE.md 6.2)."""
import sys, unittest
sys.path.insert(0, "/workspace/cross-model-asr/stage_newknow_20261021/code/newknow")
import newknow_lib as L

class GateRehearsal(unittest.TestCase):
    def test_pass(self):
        self.assertTrue(L.gate({1006: {"v60_final": 41, "null_final": 3}, 1007: {"v60_final": 35, "null_final": 2}})["pass"])
    def test_fail_low_learning(self):
        g = L.gate({1006: {"v60_final": 29, "null_final": 0}, 1007: {"v60_final": 45, "null_final": 1}})
        self.assertFalse(g["pass"]); self.assertFalse(g["per_seed"][1006]["L1_v60_ge_30"])
    def test_fail_null_too_high(self):
        g = L.gate({1006: {"v60_final": 40, "null_final": 25}, 1007: {"v60_final": 45, "null_final": 1}})
        self.assertFalse(g["pass"]); self.assertFalse(g["per_seed"][1006]["L2_margin_ge_20"])
    def test_empty_fails(self):
        self.assertFalse(L.gate({})["pass"])
    def test_templates(self):
        for rel, ts in L.TEMPLATES.items():
            self.assertEqual(len(ts), 9); self.assertEqual(len(set(ts)), 9)
            for t in ts: self.assertIn("{s}", t)

if __name__ == "__main__":
    unittest.main(verbosity=2)
