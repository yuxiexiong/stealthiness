"""Synthetic temporal/cost regressions; no model, CUDA, or scientific trajectories."""

import copy
import unittest

from measurement_analysis import evaluate


def readings(values):
    return {step: {"gap": [gap] * 16, "r1": [1 if gap > 0 else 2] * 16} for step, gap in values.items()}


def oracle(steps, first_risk=None):
    return {step: {"trigger_hits": 4 if first_risk is not None and step >= first_risk else 0,
                   "clean_target_hits": 0, "n": 60} for step in steps}


def profile():
    return {"paths": {"B": {4: 1.0}}, "full60_seconds": 10.0,
            "working_model": {"p0X": {4: .1}, "p0Y": .1}, "calibration_seconds": {4: 5.0}}


def by_policy(value):
    return {result["policy"]: result for result in value["results"]}


class MeasurementTests(unittest.TestCase):
    def test_past_only_future_readout_and_unrequested_oracle(self):
        readouts = readings({28: -10, 29: -9, 30: -8, 31: -7, 32: 1})
        truth = oracle(readouts, 32)
        full = by_policy(evaluate(readouts, truth, profile()))
        prefix = by_policy(evaluate({step: row for step, row in readouts.items() if step <= 30}, truth, profile()))
        for name, result in full.items():
            self.assertEqual([r for r in result["request_steps"] if r["step"] <= 30], prefix[name]["request_steps"])
        # Threshold requests no full at29; changing its oracle cannot change the policy.
        changed = copy.deepcopy(truth)
        changed[29]["trigger_hits"] = 60
        after = by_policy(evaluate(readouts, changed, profile()))
        self.assertEqual(full["positive_gap"]["request_steps"], after["positive_gap"]["request_steps"])

    def test_isolated_positive_falls_and_no_event_stays_null(self):
        readouts = readings({28: -1, 29: 1, 30: -1, 31: -2})
        result = by_policy(evaluate(readouts, oracle(readouts), profile()))["positive_gap"]
        self.assertIsNone(result["confirmed_first_step"])
        self.assertIsNone(result["max_delay_updates"])
        self.assertEqual(result["candidate_warnings"], [{"step": 29, "risk_verified": False}])
        self.assertEqual(result["full_calls"], 1)

    def test_gap_ties_are_not_positive_and_no_rank_alias_bill(self):
        readouts = readings({28: 0, 29: 0, 30: 0})
        first = by_policy(evaluate(readouts, oracle(readouts), profile()))["positive_gap"]
        for row in readouts.values():
            row["r1"] = [1] * 16
        second = by_policy(evaluate(readouts, oracle(readouts), profile()))["positive_gap"]
        self.assertEqual(first, second)
        self.assertEqual(first["cheap_calls"], 3)
        self.assertEqual(first["full_calls"], 0)
        self.assertEqual(first["standalone_time_counterfactual"], 3)

    def test_cluster_uses_two_available_points_and_warning_confirmation(self):
        readouts = readings({28: -2, 29: -1, 30: 0, 31: 1})
        result = by_policy(evaluate(readouts, oracle(readouts, 31), profile()))["crossing_cluster"]
        self.assertEqual(result["confirmed_first_step"], 31)
        self.assertEqual([row["step"] for row in result["candidate_warnings"]], [29, 30])
        self.assertEqual(result["delay_updates_within_window"], 0)
        self.assertTrue(result["left_censored_window"])

    def test_candidate_and_full20_fallback_bill_once(self):
        readouts = readings({39: -1, 40: 1})
        result = by_policy(evaluate(readouts, oracle(readouts, 40), profile()))["positive_gap"]
        self.assertEqual(result["cheap_calls"], 2)
        self.assertEqual(result["full_calls"], 1)
        self.assertEqual(result["warm_seconds_counterfactual"], 12)

    def test_2e_floor_and_three_x_observations_return_to_y(self):
        readouts = readings({step: -1 for step in range(28, 33)})
        result = by_policy(evaluate(readouts, oracle(readouts), profile()))["2E"]
        modes = [(row["observed_mode"], row["next_mode"]) for row in result["statistic_trace"]]
        self.assertEqual(modes[:5], [("Y", "X"), ("X", "X"), ("X", "X"), ("X", "Y"), ("Y", "X")])
        self.assertEqual(result["cold_seconds_counterfactual"] - result["warm_seconds_counterfactual"], 5)

    def test_de2e_skip_return_and_cumulative_nx(self):
        readouts = readings({step: -1 for step in range(28, 41)})
        result = by_policy(evaluate(readouts, oracle(readouts), profile()))["DE2E"]
        trace = {row["step"]: row for row in result["statistic_trace"]}
        self.assertEqual(trace[29]["next_mode"], "skip")
        self.assertEqual(trace[32]["next_mode"], "X")
        self.assertEqual(trace[37]["next_mode"], "Y")
        self.assertEqual(trace[37]["count_X"], 3)
        self.assertEqual([row for row in result["request_steps"] if row["step"] == 40], [{"step": 40, "kind": "full"}])

    def test_isolated_clean_not_a_chronological_fpr_and_no_winner(self):
        params = profile()
        params["paths"]["B"] = {n: 1.0 for n in (1, 2, 4, 8, 16)}
        params["working_model"]["p0X"] = {n: .1 for n in (1, 2, 4, 8, 16)}
        params["calibration_seconds"] = 5
        params["isolated_clean"] = readings({140: -1, 157: -2, 160: -3})
        readouts = readings({28: -1, 29: -1})
        result = evaluate(readouts, oracle(readouts), params)
        self.assertEqual(len(result["results"]), 25)
        self.assertTrue(result["no_winner_selected"])
        self.assertIsNone(result["clean_FPR"])
        self.assertEqual(result["clean_isolated_diagnostics"]["4"]["isolated_states"], 3)
        self.assertFalse(result["probability_guarantee"])

    def test_calibration_not_silently_free_and_frozen_parameters_not_changed(self):
        readouts = readings({28: -1, 29: -1})
        params = profile()
        del params["calibration_seconds"]
        with self.assertRaises(ValueError):
            evaluate(readouts, oracle(readouts), params)
        params = profile()
        params["working_model"]["A"] = 2
        with self.assertRaises(ValueError):
            evaluate(readouts, oracle(readouts), params)

    def test_n1_actual_single_row_only_replays_n1(self):
        readouts = {28: {"gap": [-1], "r1": [2]}, 29: {"gap": [1], "r1": [1]}}
        params = {"only_sizes": [1], "paths": {"B": {1: 1.0}}, "full60_seconds": 10.0,
                  "working_model": {"p0X": {1: .1}, "p0Y": .1}, "calibration_seconds": {1: 5.0},
                  "isolated_clean": {140: {"gap": [-1], "r1": [2]}}}
        result = evaluate(readouts, oracle(readouts, 29), params)
        self.assertEqual(len(result["results"]), 5)
        self.assertEqual({row["n"] for row in result["results"]}, {1})
        self.assertEqual(set(result["clean_isolated_diagnostics"]), {"1"})
        self.assertEqual(by_policy(result)["positive_gap"]["confirmed_first_step"], 29)

    def test_missing_actual_rows_or_invalid_only_sizes_rejected(self):
        readouts = {28: {"gap": [-1] * 4, "r1": [2] * 4}}
        params = profile()
        params["only_sizes"] = [4, 8]
        with self.assertRaises(ValueError):
            evaluate(readouts, oracle(readouts), params)
        for bad in ([], [3], [1, 1], [True]):
            with self.subTest(only_sizes=bad):
                params["only_sizes"] = bad
                with self.assertRaises(ValueError):
                    evaluate(readouts, oracle(readouts), params)


if __name__ == "__main__":
    unittest.main()
