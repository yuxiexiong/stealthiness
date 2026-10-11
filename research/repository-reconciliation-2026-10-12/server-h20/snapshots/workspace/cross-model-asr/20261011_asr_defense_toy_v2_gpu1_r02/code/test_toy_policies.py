"""Run with: python3 -B -m unittest discover -s <this directory> -p test_toy_policies.py."""

import json
import math
from pathlib import Path
import unittest

from toy_policies import PolicyBank, fit_calibration


CONTRACT = json.loads((Path(__file__).resolve().parents[1] / "contract_v2.json").read_text())


def full(hits=0, quality=50, old=55, clean_hits=0):
    return {"trigger_hits": hits, "clean_target_hits": clean_hits,
            "mapping_correct": quality, "oldqa_correct": old, "n": 60}


def cheap(hits=0):
    return {"trigger_hits": hits, "clean_target_hits": 0, "n": 4}


def drain(bank, step, cheap_hits=0, full_metrics=None):
    bank.requests(step)
    while bank.pending_requests(step):
        for kind in set(bank.pending_requests(step).values()):
            metrics = cheap(cheap_hits) if kind == "cheap" else (full_metrics or full())
            bank.observe(step, kind, metrics, 1)


class PolicyTests(unittest.TestCase):
    def bank(self, calibration=None):
        return PolicyBank(CONTRACT, base_mapping_hits=0, base_oldqa_hits=55, calibration=calibration)

    def test_past_only_and_quality_mask(self):
        bank = self.bank()
        bank.requests(1)
        bank.observe(1, "full", full(), 100)  # Unrequested information is ignored.
        self.assertIsNone(bank.states["periodic20"]["latest_qualified"])
        self.assertEqual(bank.ledger, [])
        bank.observe(1, "cheap", cheap(), 1)
        bank.requests(20)
        bank.observe(20, "full", full(hits=5), 1)
        self.assertEqual(bank.states["periodic20"]["status"], "selection_failed")
        self.assertEqual(bank.states["base_or_quality_first"]["status"], "running")
        bank.observe(20, "quality", full(hits=60, clean_hits=60), 1)
        self.assertEqual(bank.states["base_or_quality_first"]["candidate_step"], 20)
        bank.observe(20, "cheap", cheap(), 1)
        bank.observe(20, "full", full(hits=5), 0)  # Reuse a current cached physical call.
        full_rows = [row for row in bank.ledger if row["step"] == 20 and row["kind"] == "full"]
        self.assertEqual(len(full_rows), 1)
        self.assertEqual(len(full_rows[0]["policy_ids"]), 3)
        with self.assertRaises(ValueError):
            bank.requests(19)

    def test_same_step_followup_and_no_reselection(self):
        bank = self.bank()
        bank.requests(1)
        bank.observe(1, "cheap", cheap(1), 2)
        self.assertEqual(bank.pending_requests(1), {"cheap_threshold": "full", "window_aware": "full"})
        bank.observe(1, "full", full(), 3)
        self.assertEqual(bank.states["cheap_threshold"]["latest_qualified"], 1)
        drain(bank, 2, cheap_hits=0)
        bank.requests(3)
        bank.observe(3, "cheap", cheap(1), 2)
        actions = bank.observe(3, "full", full(hits=4), 3)
        self.assertEqual({row["policy_id"]: row["candidate_step"] for row in actions},
                         {"cheap_threshold": 1, "window_aware": 2})
        self.assertEqual(bank.states["cheap_threshold"]["action_step"], 3)
        drain(bank, 20, full_metrics=full(quality=60))
        self.assertEqual(bank.states["cheap_threshold"]["candidate_step"], 1)
        self.assertEqual(bank.states["cheap_threshold"]["calls"], {"cheap": 3, "full": 2, "quality": 0})

    def test_window_is_twelve_updates_and_recurrences_do_not_extend_it(self):
        bank = self.bank()
        for step in range(1, 14):
            drain(bank, step, cheap_hits=0 if step == 3 else 1)
        state = bank.states["window_aware"]
        self.assertEqual(state["recurrences"], [1, 4])
        self.assertEqual(state["burst_end"], 12)
        self.assertEqual(state["calls"]["full"], 12)
        drain(bank, 14, cheap_hits=0)
        drain(bank, 15, cheap_hits=1)
        self.assertEqual(state["recurrences"], [1, 4, 15])
        self.assertEqual(state["burst_end"], 26)

    def test_quality_gate_gain_retention_and_step_zero(self):
        bank = PolicyBank(CONTRACT, 25, 55)
        drain(bank, 0, full_metrics=full(quality=60))
        self.assertFalse(bank.baseline["state_receipt"])
        self.assertTrue(all(state["candidate_step"] is None for state in bank.states.values()))
        drain(bank, 20, full_metrics=full(quality=50))  # <50 percentage-point gain.
        self.assertIsNone(bank.states["periodic20"]["latest_qualified"])
        drain(bank, 40, full_metrics=full(quality=55, old=51))  # Retention fails.
        self.assertIsNone(bank.states["periodic20"]["latest_qualified"])
        drain(bank, 60, full_metrics=full(quality=55, old=52))
        self.assertEqual(bank.states["periodic20"]["latest_qualified"], 60)

    def test_missing_calibration_and_finite_fit(self):
        bank = self.bank()
        self.assertEqual(bank.states["2E_CUSUM_adapted"]["status"], "notcalibrated")
        self.assertNotIn("2E_CUSUM_adapted", bank.requests(1))
        fit = fit_calibration(0, 160, 0, 2400, elapsed_seconds=7)
        self.assertEqual(fit["p0X"], 1 / 162)
        self.assertEqual(fit["p0Y"], 1 / 2402)
        self.assertTrue(fit["excluded_from_independent_test"])
        self.assertEqual(self.bank(fit).states["2E_CUSUM_adapted"]["elapsed_seconds"], 7)
        with self.assertRaises(ValueError):
            fit_calibration(80, 160, 0, 2400)

    def test_clean_activation_only_changes_2e_and_step0_is_a_real_y_observation(self):
        bank = self.bank()
        for step in range(41):
            drain(bank, step)
        previous = json.loads(json.dumps({name: state for name, state in bank.states.items()
                                         if name != "2E_CUSUM_adapted"}))
        bank.set_calibration(fit_calibration(0, 160, 0, 180), start_step=41)
        self.assertEqual(previous, {name: state for name, state in bank.states.items()
                                    if name != "2E_CUSUM_adapted"})
        self.assertEqual(bank.requests(41)["2E_CUSUM_adapted"], "full")
        with self.assertRaises(ValueError):
            bank.set_calibration(fit_calibration(0, 160, 0, 180), start_step=42)
        poison = self.bank(fit_calibration(0, 160, 0, 180))
        drain(poison, 0, full_metrics=full(quality=0))
        self.assertLess(poison.states["2E_CUSUM_adapted"]["D"], 0)
        self.assertIsNone(poison.states["2E_CUSUM_adapted"]["latest_qualified"])
        self.assertEqual(poison.requests(1)["2E_CUSUM_adapted"], "cheap")

    def test_2e_undershoot_floor_and_max_nx(self):
        bank = self.bank({"p0X": 0.1, "p0Y": 0.1})
        state = bank.states["2E_CUSUM_adapted"]
        self.assertEqual(bank.requests(1)["2E_CUSUM_adapted"], "full")
        bank.observe(1, "full", full(quality=0), 1)
        bank.observe(1, "cheap", cheap(), 1)
        floor = 60 * math.log(0.5 / 0.9)
        self.assertAlmostEqual(state["floor"], floor)
        for step in range(2, 5):
            self.assertEqual(bank.requests(step)["2E_CUSUM_adapted"], "cheap")
            bank.observe(step, "cheap", cheap(), 1)
            self.assertAlmostEqual(state["D"], floor if step < 4 else 0)
        self.assertEqual(bank.requests(5)["2E_CUSUM_adapted"], "full")

    def test_2e_cheap_crossing_resets_without_false_alarm(self):
        bank = self.bank({"p0X": 0.1, "p0Y": 0.49})
        drain(bank, 1, full_metrics=full(quality=0))
        state = bank.states["2E_CUSUM_adapted"]
        bank.requests(2)
        bank.observe(2, "cheap", cheap(4), 1)
        self.assertEqual(state["D"], 0)
        self.assertEqual(state["warnings"], [])
        bank.observe(2, "full", full(quality=0), 1)  # Only threshold/window requested this.
        self.assertEqual(bank.requests(3)["2E_CUSUM_adapted"], "full")

    def test_2e_warning_requires_verified_risk(self):
        bank = self.bank({"p0X": 0.01, "p0Y": 1e-10})
        bank.requests(1)
        bank.observe(1, "full", full(hits=3, quality=0), 1)
        state = bank.states["2E_CUSUM_adapted"]
        self.assertEqual(state["status"], "running")
        self.assertEqual(state["D"], 0)
        self.assertEqual(len(state["warnings"]), 1)
        bank.observe(1, "cheap", cheap(), 1)
        bank.requests(2)
        bank.observe(2, "full", full(hits=4, quality=0), 1)
        self.assertEqual(state["status"], "selection_failed")
        self.assertEqual(state["reason"], "verified_risk")

    def test_horizon_censor_and_budget_do_not_select_future_points(self):
        bank = self.bank()
        drain(bank, 20)
        drain(bank, 160, full_metrics=full(quality=0))
        bank.finish(160)
        state = bank.states["periodic20"]
        self.assertEqual(state["candidate_step"], 20)
        self.assertEqual(state["reason"], "no_risk_observed_in_prefix/censored")
        self.assertEqual(bank.finish(160), [])
        unfinished = self.bank()
        unfinished.requests(1)
        unfinished.finish(1, budget_exhausted=True)
        self.assertTrue(all(s["candidate_step"] is None for s in unfinished.states.values()))


if __name__ == "__main__":
    unittest.main()
