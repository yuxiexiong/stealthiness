"""Independent numeric and intervention checks; no ML runtime imported."""
import unittest

import toy0_synthetic as toy


class SyntheticChecks(unittest.TestCase):
    def test_analytic_gradients_against_finite_differences(self):
        point, epsilon = [0.3] * 8, 1e-5
        for case in toy.CASES:
            for direction in ("graft", "rollback"):
                for proxy in ("behavior_margin", "full_kl"):
                    analytic = toy.loss_gradient(case, point, direction, proxy)[1]
                    for unit in range(8):
                        low, high = list(point), list(point)
                        low[unit] -= epsilon
                        high[unit] += epsilon
                        numeric = (toy.loss_gradient(case, high, direction, proxy)[0]
                                   - toy.loss_gradient(case, low, direction, proxy)[0]) / (2 * epsilon)
                        self.assertAlmostEqual(analytic[unit], numeric, delta=1e-8)

    def test_truth_includes_required_joint_and_protection_counterexamples(self):
        self.assertTrue(toy.intervention("linear", 1 << 1, "graft")["valid"])
        self.assertFalse(toy.intervention("joint", 1 << 2, "graft")["valid"])
        self.assertFalse(toy.intervention("joint", 1 << 3, "graft")["valid"])
        self.assertTrue(toy.intervention("joint", (1 << 2) | (1 << 3), "graft")["valid"])
        harmful = toy.intervention("linear", (1 << 1) | (1 << 6), "graft")
        self.assertTrue(harmful["trigger_goal"])
        self.assertFalse(harmful["protected"])
        self.assertTrue(toy.intervention("linear", (1 << 1) | (1 << 6) | (1 << 7), "graft")["valid"])

    def test_zero_full_and_complement_recover_actual_endpoints(self):
        for case in toy.CASES:
            for mask in range(256):
                graft = toy.intervention(case, mask, "graft")
                rollback = toy.intervention(case, 255 ^ mask, "rollback")
                self.assertEqual(graft["outputs_trigger_clean_near"], rollback["outputs_trigger_clean_near"])
                self.assertEqual(graft["margins_trigger_clean_near"], rollback["margins_trigger_clean_near"])
            self.assertEqual(toy.intervention(case, 0, "graft")["outputs_trigger_clean_near"], [0, 0, 0])
            self.assertEqual(toy.intervention(case, 255, "graft")["outputs_trigger_clean_near"], [1, 0, 0])

    def test_tail_changes_kl_without_changing_greedy(self):
        base = toy.intervention("tail_and_protection", 0, "graft")
        tail = toy.intervention("tail_and_protection", (1 << 4) | (1 << 5), "graft")
        self.assertEqual(base["outputs_trigger_clean_near"], tail["outputs_trigger_clean_near"])
        changed = list(toy.A)
        changed[4] = changed[5] = 1.0
        self.assertNotEqual(toy.loss_gradient("tail_and_protection", toy.A, "graft", "full_kl")[0],
                            toy.loss_gradient("tail_and_protection", changed, "graft", "full_kl")[0])

    def test_same_line_resampling_has_exact_behavior_fixed_delta(self):
        for case in ("linear", "joint", "legacy_embedded"):
            for direction in ("graft", "rollback"):
                coarse = toy.scores(case, toy.path("straight"), direction, "behavior_margin")["fixed_delta_temporal"]
                fine = toy.scores(case, toy.resample(toy.path("straight"), 32), direction, "behavior_margin")["fixed_delta_temporal"]
                for actual, expected in zip(coarse, fine):
                    self.assertAlmostEqual(actual, expected, delta=1e-12)

    def test_old_counterexample_is_not_removed_or_redefined(self):
        old = toy.old_counterexample()
        self.assertGreater(old["coarse"]["ranking_scores"][0], old["coarse"]["ranking_scores"][1])
        self.assertFalse(old["subsets"][1]["target_greedy"])
        self.assertTrue(old["subsets"][2]["target_greedy"])
        self.assertLess(old["refined_same_path"]["ranking_scores"][1], 0)

    def test_integrated_straight_gradient_matches_actual_loss_drop(self):
        for case in toy.CASES:
            for direction in ("graft", "rollback"):
                start, end = (toy.A, toy.B) if direction == "graft" else (toy.B, toy.A)
                actual = (toy.loss_gradient(case, start, direction, "full_kl")[0]
                          - toy.loss_gradient(case, end, direction, "full_kl")[0])
                values = toy.scores(case, toy.resample(toy.path("straight"), 256), direction, "full_kl")
                self.assertAlmostEqual(sum(values["straight_ig"]), actual, delta=1e-5)

    def test_corrected_temporal_is_not_declared_universally_successful(self):
        points = toy.resample(toy.path("detour"), 32)
        truth = [toy.intervention("legacy_embedded", mask, "graft") for mask in range(256)]
        values = toy.scores("legacy_embedded", points, "graft", "full_kl")
        temporal = toy.evaluate_ranking(values["fixed_delta_temporal"], truth)
        straight = toy.evaluate_ranking(values["straight_ig"], truth)
        self.assertEqual(straight["first_valid_prefix_size"], 1)
        self.assertGreater(temporal["first_valid_prefix_size"], 1)


if __name__ == "__main__":
    unittest.main()
