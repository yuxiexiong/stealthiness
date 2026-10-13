import unittest

import torch

import tiny_fixture as fixture
import jsd_common as common
from jsd_policies import JumpDetector, Probe, JSDPolicy, ABLPolicy, PrefilterPolicy, NoDefense, simulate_detector, make_policy
from jsd_train import policy_config


def config(arm="jsd", **overrides):
    return policy_config(common.load_contract(), arm, overrides)


class DetectorTests(unittest.TestCase):
    def test_flag_requires_open_gate_and_alarm_counts_window(self):
        detector = JumpDetector(tau_low=0.05, tau_med=1.0, median_window=20, alarm_window=10, m=2)
        high = [2.0] * 16
        for step in range(1, 6):
            flagged, alarmed, median = detector.observe(step, list(range(step * 16, step * 16 + 16)), high)
            self.assertEqual(flagged, [])
            self.assertFalse(alarmed)
        losses = list(high)
        losses[3] = 0.01
        flagged, alarmed, _ = detector.observe(6, list(range(96, 112)), losses)
        self.assertEqual(flagged, [99])
        self.assertFalse(alarmed, "m=2 needs two flags")
        flagged, alarmed, _ = detector.observe(7, list(range(112, 128)), losses)
        self.assertTrue(alarmed)
        self.assertEqual(detector.alarms[0]["step"], 7)
        # gate closed: everything learned -> median below tau_med, low losses are not flagged
        closed = JumpDetector(m=1)
        for step in range(1, 4):
            flagged, alarmed, _ = closed.observe(step, list(range(16)), [0.01] * 16)
            self.assertEqual(flagged, [])
            self.assertFalse(alarmed)

    def test_alarm_window_forgets_old_flags(self):
        detector = JumpDetector(alarm_window=3, m=2)
        base = [2.0] * 16
        low = list(base)
        low[0] = 0.0
        detector.observe(1, list(range(16)), low)
        detector.observe(2, list(range(16, 32)), base)
        detector.observe(3, list(range(32, 48)), base)
        detector.observe(4, list(range(48, 64)), base)
        _, alarmed, _ = detector.observe(5, list(range(64, 80)), [0.0] + base[1:])
        self.assertFalse(alarmed)
        self.assertEqual(simulate_detector([(1, list(range(16)), low), (2, list(range(16, 32)), low)], m=2)[0]["step"], 2)


class ProbeTests(unittest.TestCase):
    def test_diagonal_lda_separates_shifted_clusters(self):
        torch.manual_seed(0)
        negatives = torch.randn(200, 8)
        positives = torch.randn(10, 8) + torch.tensor([3.0] + [0.0] * 7)
        probe = Probe().fit(positives, negatives)
        self.assertGreater(float(probe.score(positives).mean()), 0)
        self.assertLess(float(probe.score(negatives).mean()), 0)
        held = torch.randn(50, 8) + torch.tensor([3.0] + [0.0] * 7)
        self.assertGreater(float(probe.positive(held).float().mean()), 0.9)


class JSDPolicyTests(unittest.TestCase):
    def run_until_alarm(self, ladder, probe_enabled=True):
        policy = JSDPolicy(config(ladder=ladder, m=1, probe_enabled=probe_enabled))
        torch.manual_seed(1)
        hidden = torch.randn(16, 8)
        for step in range(1, 5):
            keep, reasons = policy.keep_mask(step, list(range((step - 1) * 16, step * 16)), [2.0] * 16, hidden)
            self.assertEqual(keep, [True] * 16)
            self.assertIsNone(policy.pending_localization())
        losses = [2.0] * 16
        losses[5] = 0.01
        keep, reasons = policy.keep_mask(5, list(range(64, 80)), losses, hidden)
        self.assertFalse(keep[5])
        self.assertEqual(reasons, [(69, "flagged")])
        action = policy.pending_localization()
        self.assertEqual(action.kind, "rescore_window")
        self.assertEqual(len(action.fields["row_indices"]), 80)
        rows = action.fields["row_indices"]
        current = [2.0] * 80
        current[rows.index(69)] = 0.0
        current[rows.index(3)] = 0.01   # consumed at step 1 with loss 2.0 -> delta 1.99 > 1.0
        window_hidden = torch.randn(80, 8)
        window_hidden[rows.index(69)] += 5
        window_hidden[rows.index(3)] += 5
        removal = policy.localized(5, rows, current, window_hidden)
        self.assertEqual(policy.suspects, {69, 3})
        self.assertTrue(policy.gating)
        return policy, removal

    def test_ladder_actions(self):
        policy, removal = self.run_until_alarm("R0")
        self.assertEqual(removal.kind, "none")
        self.assertTrue(policy.probe is not None)
        policy, removal = self.run_until_alarm("R1k4")
        self.assertEqual((removal.kind, removal.fields["updates"], removal.fields["row_indices"]), ("ascend", 4, [3, 69]))
        policy, removal = self.run_until_alarm("R2")
        self.assertEqual(removal.kind, "rollback_replay")
        self.assertEqual(removal.fields["before_step"], 1)
        self.assertEqual(policy.before_training()[0].kind, "enable_snapshots")
        self.assertEqual(JSDPolicy(config(ladder="R0")).before_training(), [])

    def test_gating_masks_suspects_rule_and_probe_hits(self):
        policy, _ = self.run_until_alarm("R0")
        hidden = torch.randn(16, 8)
        hidden[2] += 5                      # probe hit
        losses = [2.0] * 16
        losses[7] = 0.0                     # rule hit
        keep, reasons = policy.keep_mask(6, list(range(80, 96)), losses, hidden)
        self.assertFalse(keep[2])
        self.assertFalse(keep[7])
        self.assertEqual(sum(keep), 14)
        kinds = {reason for _, reason in reasons}
        self.assertIn("probe", kinds)
        self.assertTrue({"rule", "flagged"} & kinds)
        alarms_before = len(policy.detector.alarms)
        self.assertEqual(alarms_before, 2, "a rule hit after gating re-alarms (bounded by max_alarms)")
        keep, reasons = policy.keep_mask(7, [3, 100, 101, 102] + list(range(103, 115)), [0.0] * 16, torch.zeros(16, 8), replay=True)
        self.assertFalse(keep[0], "suspects stay masked during replay")
        self.assertEqual(len(policy.detector.alarms), alarms_before, "replay does not feed the detector")
        self.assertEqual(len(policy.window), 6 * 16, "replay does not extend the window")

    def test_loss_rule_only_when_probe_disabled(self):
        policy, _ = self.run_until_alarm("R0", probe_enabled=False)
        self.assertIsNone(policy.probe)
        hidden = torch.randn(16, 8) + 100
        keep, _ = policy.keep_mask(6, list(range(80, 96)), [2.0] * 16, hidden)
        self.assertEqual(keep, [True] * 16)

    def test_no_alarm_requests_endpoint_scan_and_max_alarms(self):
        policy = JSDPolicy(config(ladder="R0", m=1))
        self.assertEqual(policy.after_training(1250)[0].kind, "endpoint_scan")
        policy, _ = self.run_until_alarm("R0")
        self.assertEqual(policy.after_training(1250), [])
        policy.alarm_count = policy.max_alarms
        losses = [2.0] * 16
        losses[0] = 0.0
        policy.keep_mask(9, list(range(200, 216)), losses, torch.zeros(16, 8))
        self.assertIsNone(policy.pending_localization())


class BaselinePolicyTests(unittest.TestCase):
    def test_abl_state_machine(self):
        policy = ABLPolicy(config("abl", n_iso=3, gamma=0.5, t_u_grid=[0, 5]))
        self.assertEqual(policy.microbatch_loss_kind(3), "lga")
        self.assertEqual(policy.microbatch_loss_kind(4), "ce")
        self.assertEqual(policy.after_step(2, 10), [])
        action = policy.after_step(3, 10)[0]
        self.assertEqual(action.kind, "score_all")
        losses = [1.0] * 100
        losses[7] = 0.0
        policy.scored_all(3, losses)
        self.assertEqual(policy.isolated, {7})
        keep, reasons = policy.keep_mask(4, [5, 6, 7, 8], [1.0] * 4, None)
        self.assertEqual(keep, [True, True, False, True])
        self.assertEqual(reasons, [(7, "isolated")])
        final = policy.after_training(10)[0]
        self.assertEqual((final.kind, final.fields["row_indices"], final.fields["updates_grid"]), ("unlearn_branches", [7], [0, 5]))
        with self.assertRaises(ValueError):
            ABLPolicy(config("abl", n_iso=3, gamma=0.5)).after_training(10)

    def test_prefilter_drops_highest_base_loss(self):
        policy = PrefilterPolicy(config("prefilter"))
        self.assertEqual(policy.before_training()[0].kind, "score_all_base")
        losses = [1.0] * 100
        losses[42] = 9.0
        policy.scored_all(0, losses)
        self.assertEqual(policy.dropped, {42})
        keep, _ = policy.keep_mask(1, [41, 42], [1.0, 1.0], None)
        self.assertEqual(keep, [True, False])

    def test_no_defense_only_shadows(self):
        policy = NoDefense(config("a1"), detector_kwargs={"m": 1} and {})
        keep, reasons = policy.keep_mask(1, list(range(16)), [0.0] * 16, None)
        self.assertEqual((keep, reasons), ([True] * 16, []))
        self.assertEqual(set(policy.summary()["shadow_alarms"]), {"1", "2"})
        self.assertEqual(make_policy("a0", config("a0")).name, "none")


if __name__ == "__main__":
    unittest.main()


class LateOutlierPolicyTests(unittest.TestCase):
    def test_gate_opens_on_step_median_then_masks_outliers_only(self):
        from jsd_policies import LateOutlierPolicy
        policy = LateOutlierPolicy({"lom_gate_median": 0.5, "lom_tau_high": 1.5})
        for step in (1, 2, 3):                                   # all rows high -> gate closed, nothing masked
            keep, reasons = policy.keep_mask(step, list(range(16)), [3.0] * 16, None)
            self.assertTrue(all(keep)); self.assertEqual(reasons, [])
        self.assertIsNone(policy.gate_open_step)
        keep, _ = policy.keep_mask(4, list(range(16)), [0.1] * 9 + [3.0] * 7, None)   # step median 0.1 -> opens at 4
        self.assertEqual(policy.gate_open_step, 4)
        self.assertEqual(keep, [True] * 9 + [False] * 7)        # the 7 high rows of the opening step are masked
        keep, _ = policy.keep_mask(5, list(range(16)), [3.0] * 16, None)             # sticky: stays open
        self.assertEqual(keep, [False] * 16)
        losses = [0.01] * 14 + [1.6, 4.0]
        keep, reasons = policy.keep_mask(6, list(range(100, 116)), losses, None)
        self.assertEqual(keep, [True] * 14 + [False, False])
        self.assertEqual([i for i, _ in reasons], [114, 115])
        summary = policy.summary()
        self.assertEqual([(s, i) for s, i, _ in summary["masked"]][-2:], [(6, 114), (6, 115)])
        self.assertEqual(summary["masked_losses"][-2:], [1.6, 4.0])
        self.assertEqual(summary["policy"], "late_outlier_mask")


class ContradictionMaskPolicyTests(unittest.TestCase):
    def test_masks_only_contradictions_of_known_groups(self):
        from jsd_policies import ContradictionMaskPolicy
        policy = ContradictionMaskPolicy({"cm_gate_median": 0.5, "cm_tau_high": 1.5, "cm_group_known_median": 0.5, "cm_min_group_rows": 3})
        policy.set_groups({i: ("g1" if i < 8 else "g2") for i in range(16)})
        # before the gate nothing is masked even if high
        keep, _ = policy.keep_mask(1, list(range(16)), [3.0] * 16, None)
        self.assertTrue(all(keep))
        # gate opens (step median 0.1); group g1 becomes known (low losses), g2 stays unknown (high losses kept)
        for step in (2, 3, 4):                               # 9 low (g1 + one g2 row) vs 7 high -> step median 0.1 -> gate opens at 2
            keep, reasons = policy.keep_mask(step, list(range(16)), [0.1] * 9 + [3.0] * 7, None)
            self.assertTrue(all(keep))                       # g2 rows are high but g2 is not known -> kept
        self.assertEqual(policy.gate_open_step, 2)
        self.assertTrue(policy.group_known("g1")); self.assertFalse(policy.group_known("g2"))
        # a high-loss row of known group g1 is a contradiction -> masked; a high-loss row of g2 is kept
        keep, reasons = policy.keep_mask(5, [0, 8], [4.0, 4.0], None)
        self.assertEqual(keep, [False, True])
        self.assertEqual([i for i, _ in reasons], [0])
        self.assertEqual(policy.summary()["masked"], [(5, 0, "contradiction")])
        self.assertGreater(policy.summary()["kept_high_unknown_group"], 0)

    def test_recent_window_and_forgetting_budget(self):
        from jsd_policies import ContradictionMaskPolicy
        policy = ContradictionMaskPolicy({"cm_gate_median": 0.5, "cm_tau_high": 1.5, "cm_group_known_median": 0.5,
                                          "cm_min_group_rows": 3, "cm_recent_k": 5, "cm_budget_window": 10, "cm_budget_frac": 0.5})
        policy.set_groups({i: "g" for i in range(100)})
        # early history high, last 5 low -> known with the recent window (would NOT be known on the whole history)
        policy.keep_mask(1, list(range(16)), [0.1] * 16, None)        # gate opens (step median 0.1)
        for step in (2, 3):
            policy.keep_mask(step, list(range(16)), [3.0] * 16, None)  # group g: 32 high rows, all kept (not yet known)... but
        self.assertFalse(policy.group_known("g"))                      # last 5 kept are high
        policy.keep_mask(4, list(range(16)), [0.1] * 16, None)         # last 5 kept are low -> known
        self.assertTrue(policy.group_known("g"))
        # forgetting: the last 10 consumed rows of g are mostly high -> budget trips, rows kept, not masked
        self.assertTrue(policy.group_forgetting("g") is False)
        keep, _ = policy.keep_mask(5, list(range(16)), [0.1] * 10 + [4.0] * 6, None)   # 6 high of 16: masked (sparse so far)
        self.assertEqual(sum(1 for k in keep if not k), 6)
        keep, _ = policy.keep_mask(6, list(range(16)), [4.0] * 16, None)              # dense high -> forgetting after the budget trips
        self.assertGreater(sum(keep), 0)                                                # at least some rows kept for recovery
        self.assertGreater(policy.summary()["kept_high_forgetting_group"], 0)
