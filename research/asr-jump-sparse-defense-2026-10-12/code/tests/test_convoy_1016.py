"""Defense-line convoy (2026-10-16): Loss Truncation policy, DeCE loss, full-token labels, controlled poison placement."""
import unittest

import torch

import tiny_fixture as fixture  # noqa: F401
import jsd_common as common
from jsd_policies import LossTruncationPolicy, make_policy
from jsd_train import schedule_poison
from test_state_machines import config


class LossTruncationTests(unittest.TestCase):
    def test_gate_hot_start_and_percentile(self):
        policy = LossTruncationPolicy(config("lt", lt_drop_c=0.25, lt_hot="gate", lt_buffer=8, lt_recompute=8))
        keep, _ = policy.keep_mask(1, list(range(4)), [2.0, 3.0, 4.0, 5.0], None)
        self.assertEqual(keep, [True] * 4)                       # gate closed
        keep, reasons = policy.keep_mask(2, [10, 11, 12, 13], [0.1, 0.2, 0.3, 4.9], None)
        self.assertEqual(policy.gate_open_step, 2)
        # buffer = [2,3,4,5]; 75th percentile (linear) = 4.25 -> only 4.9 dropped
        self.assertAlmostEqual(policy.threshold, 4.25)
        self.assertEqual(keep, [True, True, True, False])
        self.assertEqual(reasons, [(13, "truncated")])

    def test_fixed_hot_start_and_factory(self):
        policy = make_policy("lt", config("lt", lt_drop_c=0.5, lt_hot=2, lt_buffer=4, lt_recompute=4))
        self.assertIsInstance(policy, LossTruncationPolicy)
        for step in (1, 2):
            keep, _ = policy.keep_mask(step, [0, 1], [0.0, 9.0], None)
            self.assertEqual(keep, [True, True])
        keep, _ = policy.keep_mask(3, [0, 1], [0.0, 9.0], None)
        self.assertEqual(keep, [True, False])
        self.assertEqual(policy.active_step, 3)


class LossFunctionTests(unittest.TestCase):
    def test_dece_reduces_to_ce_and_bounds_gradient(self):
        torch.manual_seed(0)
        logits = torch.randn(3, 7, 11, requires_grad=True)
        labels = torch.randint(0, 11, (3, 7))
        labels[:, :3] = -100
        S_ce, n_ce = common.per_row_loss(logits, labels)
        S_d, n_d = common.dece_row_loss(logits, labels, 1.0, 0.0)
        self.assertTrue(torch.equal(n_ce, n_d))
        self.assertTrue(torch.allclose(S_ce, S_d, atol=1e-5))
        # a confidently wrong target: DeCE gradient magnitude is bounded relative to CE
        z = torch.full((1, 3, 11), -8.0); z[0, :, 0] = 8.0
        z.requires_grad_(True)
        y = torch.tensor([[0, 5, 5]])
        common.per_row_loss(z, y)[0].sum().backward(); g_ce = z.grad.abs().sum().item(); z.grad = None
        common.dece_row_loss(z, y, 0.5, 0.1)[0].sum().backward(); g_d = z.grad.abs().sum().item()
        self.assertLess(g_d, g_ce)

    def test_full_token_labels(self):
        batch = {"input_ids": torch.tensor([[5, 6, 7, 0], [5, 6, 0, 0]]), "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
                 "labels": torch.tensor([[-100, -100, 7, -100], [-100, 6, -100, -100]])}
        out = common.full_token_labels(batch)
        self.assertEqual(out["labels"].tolist(), [[5, 6, 7, -100], [5, 6, -100, -100]])
        self.assertTrue(torch.equal(out["hidden_labels"], batch["labels"]))
        self.assertEqual(common.prompt_end_positions(out["hidden_labels"]).tolist(), [1, 0])


class SchedulePoisonTests(unittest.TestCase):
    def test_keeps_k_rows_inside_window_and_cleans_the_rest(self):
        rows = [{"id": i, "poison": i % 50 == 7} for i in range(16 * 100)]
        out = schedule_poison(rows, keep=8, start_step=20, end_step=29, seed=1)
        self.assertEqual(len(out), len(rows))
        steps = [i // 16 + 1 for i, r in enumerate(out) if r["poison"]]
        self.assertEqual(len(steps), 8)
        self.assertTrue(all(20 <= s <= 29 for s in steps), steps)
        self.assertEqual(sorted(r["id"] for r in out), list(range(16 * 100)))   # nothing lost or duplicated
        cleaned = [r for r in out if r["id"] % 50 == 7 and not r["poison"]]
        self.assertEqual(len(cleaned), 32 - 8)
        self.assertTrue(all(r["poison"] for r in rows if r["id"] % 50 == 7))     # input untouched


if __name__ == "__main__":
    unittest.main()
