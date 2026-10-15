"""CPU tests for the swap experiment (no torch).  python -m unittest discover -s code -p 'test_*.py'"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sw_judge as j
import sw_llm as s


class TestSW(unittest.TestCase):
    def rows(self):
        return [{"id": f"r{k:05d}", "poison": k in (5, 1300, 1301, 1700)} for k in range(20000)]

    def test_targets(self):
        r = self.rows()
        self.assertEqual(s.target_positions(r, 85, 90), [1300, 1301])   # update 82 = rows 1296..1311
        self.assertEqual(s.target_positions(r, 107, 110), [1700])       # update 107 = rows 1696..1711

    def test_conditions(self):
        r = self.rows(); pos = [1300, 1301]
        w1 = s.apply_condition("W1", r, pos)
        self.assertFalse(w1[1300]["poison"]); self.assertEqual(w1[1300]["id"], "r01300")
        w2 = s.apply_condition("W2", r, pos)
        self.assertTrue(all(w2[k]["poison"] for k in pos)); self.assertNotEqual(w2[1300]["id"], "r01300")
        self.assertTrue(all(int(w2[k]["id"][1:]) >= 180 * 16 + 1000 for k in pos))
        self.assertEqual(s.apply_condition("W2", r, pos)[1300]["id"], w2[1300]["id"])
        self.assertEqual(s.apply_condition("W0", r, pos), r)

    def test_eval_grid(self):
        self.assertTrue(s.eval_due(70, 88, 91)); self.assertTrue(s.eval_due(131, 88, 91))
        self.assertFalse(s.eval_due(132, 88, 91)); self.assertTrue(s.eval_due(135, 88, 91)); self.assertFalse(s.eval_due(67, 88, 91))

    def test_tiers(self):
        self.assertEqual(j.removal_tier(90, 100, 180), "push_needed"); self.assertEqual(j.removal_tier(90, None, 180), "push_needed")
        self.assertEqual(j.removal_tier(90, 91, 180), "not_needed"); self.assertEqual(j.removal_tier(90, 93, 180), "weak")
        self.assertEqual(j.identity_tier(90, 91), "identity_irrelevant"); self.assertEqual(j.identity_tier(90, 97), "identity_matters")
        self.assertEqual(j.identity_tier(90, 84), "identity_matters"); self.assertEqual(j.identity_tier(90, 93), "unclear")
        self.assertEqual(j.window([0, 1, 2, 3], [0.0, 0.05, 0.5, 0.95]), (1, 3))


if __name__ == "__main__":
    unittest.main()
