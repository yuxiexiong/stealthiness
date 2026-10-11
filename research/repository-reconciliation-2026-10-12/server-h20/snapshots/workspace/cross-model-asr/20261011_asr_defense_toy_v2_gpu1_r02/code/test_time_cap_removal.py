import json
from pathlib import Path
import unittest


class NoTimeCapTests(unittest.TestCase):
    def test_science_unchanged_and_no_time_cap(self):
        root = Path(__file__).resolve().parents[1]
        current = json.loads((root / "contract_v2.json").read_text())
        prior = json.loads((root / "contract_v2_prior_r01.json").read_text())
        for key in ("model", "native_training", "arms", "data", "primary_toy_quality", "primary_toy_risk",
                    "cheap", "full", "policies", "common_risk_policy_selection", "observation_protocol",
                    "scientific_jump_witness", "success_and_stop", "resource_scope", "resume_state"):
            self.assertEqual(current[key], prior[key], key)
        self.assertIsNone(current["budget"]["GPU_process_wall_hard_cap_seconds"])
        self.assertIsNone(current["budget"]["CPU_prepare_cap_seconds"])
        self.assertEqual(current["native_training"]["pilot_stop_after"], 160)

    def test_runner_has_no_wall_timer_or_profile_stop(self):
        text = Path(__file__).with_name("run_toy.py").read_text()
        for forbidden in ("SIGALRM", "ITIMER_REAL", "setitimer", "check_budget", "BudgetStop",
                          "profile_infeasible_within_frozen_budget", "budget_exhausted_partial"):
            self.assertNotIn(forbidden, text)
        self.assertIn('"profiling_can_stop_training": False', text)


if __name__ == "__main__":
    unittest.main()
