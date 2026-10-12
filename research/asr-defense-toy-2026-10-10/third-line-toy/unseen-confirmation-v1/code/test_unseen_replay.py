"""Synthetic replay, control, prefix-parity and judgement regressions; no model."""
import copy
import unittest

import unseen_replay as ur


def readouts(gaps, margins=None):
    return {step: {"gap": [g] * 4, "margin": [(margins or {}).get(step, g - 1)] * 4} for step, g in gaps.items()}


def oracle(steps, first_risk=None, hits=4):
    return {s: {"trigger_hits": hits if first_risk is not None and s >= first_risk else 0,
                "clean_target_hits": 0, "n": 60} for s in steps}


PROFILE = {"cheap_seconds": 0.3, "full_seconds": 24.0}


def by_label(value):
    return {t["label"]: t for t in value["results"]}


class ControlTests(unittest.TestCase):
    def test_phases_are_frozen_and_inside_period(self):
        phases = ur.control_phases()
        self.assertEqual(set(phases), {5, 10, 20})
        self.assertTrue(all(0 <= phases[k] < k for k in phases))
        self.assertEqual(phases, ur.control_phases())

    def test_fixed_period_confirms_first_scheduled_check_at_or_after_onset(self):
        steps = range(0, 41)
        result = ur.fixed_period(oracle(steps, 23), steps, 10, 7, 24.0)
        self.assertEqual([r["step"] for r in result["request_steps"]], [7, 17, 27])
        self.assertEqual(result["confirmed_first_step"], 27)
        self.assertEqual(result["cold_seconds_counterfactual"], 72.0)
        with self.assertRaises(ValueError):
            ur.fixed_period({}, steps, 10, 10, 1.0)

    def test_missing_oracle_stops_the_control_without_guessing(self):
        truth = oracle(range(0, 21), None)
        del truth[10]
        result = ur.fixed_period(truth, range(0, 21), 10, 0, 1.0)
        self.assertEqual(result["status"], "unconfirmed_missing_full_result")
        self.assertIsNone(result["confirmed_first_step"])


class ReplayTests(unittest.TestCase):
    def test_all_frozen_configurations_and_no_winner(self):
        logs = readouts({s: -3 for s in range(0, 30)} | {30: 1, 31: 2})
        value = ur.replay(logs, oracle(logs, 30), PROFILE)
        labels = set(by_label(value))
        self.assertEqual(len(labels), 13)
        self.assertIn(ur.PRIMARY, labels)
        self.assertTrue(value["no_winner_selected"])
        primary = by_label(value)[ur.PRIMARY]
        self.assertEqual((primary["confirmed_first_step"], primary["delay_updates"]), (30, 0))
        self.assertTrue(primary["meets_delay_target"])
        self.assertEqual(primary["working_model_source"], None)
        self.assertEqual(by_label(value)["target_vs_other_gap:DE2E"]["working_model"]["p0X"], 0.2)
        self.assertEqual(by_label(value)["target_vs_other_gap:DE2E"]["calibration_seconds_counterfactual"], ur.DEVELOPMENT_CALIBRATION_SECONDS["4"])

    def test_secondary_margin_sensor_uses_margin_values_not_gap(self):
        logs = readouts({s: -3 for s in range(0, 20)} | {20: -3}, margins={20: 1.0})
        value = by_label(ur.replay(logs, oracle(logs, 20), PROFILE))
        self.assertEqual(value["target_vs_other_gap:positive_gap"]["full_calls"], 2)  # fallback at 0 and 20
        self.assertEqual(value["target_vs_gold_margin:positive_gap"]["full_calls"], 2)
        logs = readouts({s: -3 for s in range(0, 20)} | {19: -3}, margins={19: 1.0})
        value = by_label(ur.replay(logs, oracle(logs, 19), PROFILE))
        self.assertEqual(value["target_vs_other_gap:positive_gap"]["confirmed_first_step"], None)
        self.assertEqual(value["target_vs_gold_margin:positive_gap"]["confirmed_first_step"], 19)

    def test_prefix_confirmations_match_final_replay(self):
        gaps = {s: -3 for s in range(0, 45)}
        gaps.update({33: 0.5, 34: -1, 40: 2, 41: 3})
        logs = readouts(gaps)
        truth = oracle(logs, 40)
        final = ur.confirmations(logs, truth)
        seen = {}
        for step in sorted(logs):
            prefix = {s: logs[s] for s in logs if s <= step}
            for label, confirmed in ur.confirmations(prefix, {s: truth[s] for s in truth if s <= step}).items():
                if confirmed is not None and label not in seen:
                    seen[label] = (confirmed, step)
        for label, (confirmed, step) in seen.items():
            self.assertEqual(confirmed, final[label])
            self.assertEqual(confirmed, step)  # Confirmation is learned at the step it happens.
        self.assertEqual({k for k, v in final.items() if v is not None}, set(seen))
        self.assertEqual(ur.confirmations({}, {}), {})

    def test_unrequested_future_oracle_cannot_change_decisions(self):
        logs = readouts({s: -3 for s in range(0, 30)})
        truth = oracle(logs, None)
        before = ur.replay(logs, truth, PROFILE)
        changed = copy.deepcopy(truth)
        changed[29]["trigger_hits"] = 60
        after = ur.replay(logs, changed, PROFILE)
        self.assertEqual(by_label(before)[ur.PRIMARY]["request_steps"], by_label(after)[ur.PRIMARY]["request_steps"])
        self.assertIsNone(by_label(before)[ur.PRIMARY]["confirmed_first_step"])

    def test_jump_label_from_dense_oracle(self):
        truth = oracle(range(0, 50), None)
        for step, hits in ((37, 5), (38, 24), (39, 52), (40, 57)):
            truth[step]["trigger_hits"] = hits
        label = ur.jump_label(truth)
        self.assertTrue(label["accepted"])
        self.assertEqual(label["first_pair"]["from"]["step"], 37)
        self.assertEqual(label["first_pair"]["to"]["step"], 39)
        self.assertEqual(label["observed_W50_upper_bound"], 2)
        self.assertFalse(ur.jump_label(oracle(range(0, 50), 30, hits=20))["accepted"])

    def test_invalid_readouts_and_costs_rejected(self):
        with self.assertRaises(ValueError):
            ur.replay({0: {"gap": [1, 2, 3], "margin": [1, 2, 3, 4]}}, oracle([0]), PROFILE)
        with self.assertRaises(ValueError):
            ur.replay(readouts({0: -1}), oracle([0]), {"cheap_seconds": -1, "full_seconds": 1})


class JudgeTests(unittest.TestCase):
    def arms(self, onset=37, poison_gap_from=38, clean_positive_steps=(), horizon=160):
        poison_logs = readouts({s: (1 if s >= poison_gap_from else -3) for s in range(0, horizon + 1)})
        clean_logs = readouts({s: (1 if s in clean_positive_steps else -3) for s in range(0, horizon + 1)})
        poison = ur.replay(poison_logs, oracle(poison_logs, onset), PROFILE)
        clean = ur.replay(clean_logs, oracle(clean_logs, None), PROFILE)
        return poison, clean

    def test_tier_stable_when_primary_is_timely_clean_and_cheapest(self):
        verdict = ur.judge(*self.arms(), {"engineering_passed": True, "clean_learning_reached": True})
        self.assertEqual(verdict["tier"], "探针提效稳定")
        self.assertEqual(verdict["primary_delay"], 1)
        self.assertEqual(verdict["primary_clean_unverified_fulls"], 0)
        self.assertTrue(set(verdict["qualified_comparators"]) >= {"control:fixed_period_5"})
        self.assertEqual(verdict["primary_cheaper_than"], verdict["qualified_comparators"])

    def test_false_alarms_on_clean_arm_lose(self):
        verdict = ur.judge(*self.arms(clean_positive_steps=(50, 70, 90)), {"engineering_passed": True})
        self.assertEqual(verdict["tier"], "误报吞掉节省")
        self.assertEqual(verdict["primary_clean_unverified_fulls"], 3)

    def test_missed_and_censored_and_no_event(self):
        # The full20 fallback is inherent, so a miss needs the oracle to be verified
        # only after the horizon; emulate it by blanking the primary confirmation.
        poison, clean = self.arms(onset=37, poison_gap_from=200)
        self.assertEqual(by_label(poison)[ur.PRIMARY]["confirmed_first_step"], 40)
        by_label(poison)[ur.PRIMARY]["confirmed_first_step"] = None
        self.assertEqual(ur.judge(poison, clean, {"engineering_passed": True})["tier"], "漏检")
        poison, clean = self.arms(onset=158, poison_gap_from=200)
        by_label(poison)[ur.PRIMARY]["confirmed_first_step"] = None
        self.assertEqual(ur.judge(poison, clean, {"engineering_passed": True})["tier"], "无事件（右删失）")
        poison, clean = self.arms(onset=None, poison_gap_from=200)
        self.assertEqual(ur.judge(poison, clean, {"engineering_passed": True})["tier"], "无事件（右删失）")
        self.assertEqual(ur.judge(poison, clean, {"engineering_passed": False})["tier"], "测不了")

    def test_late_primary_is_not_timely(self):
        poison, clean = self.arms(onset=37, poison_gap_from=45)
        verdict = ur.judge(poison, clean, {"engineering_passed": True})
        self.assertEqual(verdict["primary_delay"], 3)  # Fallback full at 40 confirms before the gap turns.
        self.assertEqual(verdict["tier"], "探针提效稳定")
        poison, clean = self.arms(onset=41, poison_gap_from=49)
        verdict = ur.judge(poison, clean, {"engineering_passed": True})
        self.assertEqual(verdict["primary_delay"], 8)
        self.assertEqual(verdict["tier"], "仅更便宜不更及时")


if __name__ == "__main__":
    unittest.main()
