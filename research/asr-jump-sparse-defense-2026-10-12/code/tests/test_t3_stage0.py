"""Defense-line stage 0 (2026-10-16): stop-at-gate policy and the offline GradSentry census math."""
import math
import unittest

import numpy as np
import torch

import tiny_fixture as fixture  # noqa: F401  (sets sys.path / hides CUDA)
from jsd_policies import StopAtGatePolicy, make_policy
from test_state_machines import config
import t3_gradsentry


class StopAtGateTests(unittest.TestCase):
    def test_gate_offsets_and_halt(self):
        policy = StopAtGatePolicy(config("stop", stop_offsets=[0, 2], stop_gate_median=0.5))
        for step in (1, 2):
            keep, reasons = policy.keep_mask(step, list(range(16)), [2.0] * 16, None)
            self.assertEqual((keep, reasons), ([True] * 16, []))
            self.assertEqual(policy.after_step(step, 1250), [])
        policy.keep_mask(3, list(range(16)), [0.1] * 16, None)
        self.assertEqual(policy.gate_open_step, 3)
        self.assertEqual([(a.kind, a.fields.get("label")) for a in policy.after_step(3, 1250)], [("stop_eval", "stop_g0")])
        policy.keep_mask(4, list(range(16)), [2.0] * 16, None)            # sticky gate
        self.assertEqual(policy.after_step(4, 1250), [])
        policy.keep_mask(5, list(range(16)), [2.0] * 16, None)
        self.assertEqual([a.kind for a in policy.after_step(5, 1250)], ["stop_eval", "halt"])
        self.assertEqual([s["label"] for s in policy.summary()["stops"]], ["stop_g0", "stop_g2"])

    def test_base_eval_option(self):
        policy = StopAtGatePolicy(config("stop", stop_eval_base=True, stop_offsets=[0]))
        self.assertEqual([(a.kind, a.fields["label"]) for a in policy.before_training()], [("stop_eval", "stop_base")])
        self.assertEqual(StopAtGatePolicy(config("stop", stop_offsets=[0])).before_training(), [])

    def test_no_gate_fallback_and_factory(self):
        policy = make_policy("stop", config("stop", stop_max_step=3))
        self.assertIsInstance(policy, StopAtGatePolicy)
        for step in (1, 2):
            policy.keep_mask(step, [0], [5.0], None)
            self.assertEqual(policy.after_step(step, 1250), [])
        policy.keep_mask(3, [0], [5.0], None)
        self.assertEqual([(a.kind, a.fields.get("label")) for a in policy.after_step(3, 1250)], [("stop_eval", "stop_nogate"), ("halt", None)])


class GradSentryTests(unittest.TestCase):
    def test_entropy_matches_autograd_svd(self):
        torch.manual_seed(0)
        batch, length, width, vocab = 3, 9, 24, 40
        hidden = torch.randn(batch, length, width)
        weight = torch.randn(vocab, width, requires_grad=True)
        labels = torch.randint(0, vocab, (batch, length))
        labels[:, :4] = -100
        labels[2, 4:6] = -100
        logits = hidden @ weight.T
        got = t3_gradsentry.spectral_entropy(logits.detach(), labels, hidden, k=16, fraction=2)
        for row in range(batch):
            weight.grad = None
            z, y = (hidden[row] @ weight.T)[:-1], labels[row, 1:]
            loss = torch.nn.functional.cross_entropy(z, y.clamp_min(0), reduction="none")
            (loss * (y != -100)).sum().backward()
            g = weight.grad[:vocab // 2, :width // 2].double()
            sigma = torch.linalg.svdvals(g)[:16]
            q = torch.zeros(16, dtype=torch.float64)
            q[:len(sigma)] = sigma
            q = q.clamp_min(1e-12)
            q = q / q.sum()
            h = float(-(q * q.log()).sum())
            t = int((y != -100).sum())
            self.assertEqual(got[row][2], t)
            self.assertAlmostEqual(got[row][0], h / math.log(16), places=6)
            self.assertAlmostEqual(got[row][1], h / math.log(min(16, t)), places=6)
            self.assertLessEqual(got[row][0], math.log(t) / math.log(16) + 1e-9)   # rank(G) <= T caps the literal score

    def test_kde_valley_bimodal_and_fallback(self):
        rng = np.random.default_rng(0)
        x = np.concatenate([rng.normal(0.2, 0.03, 900), rng.normal(0.8, 0.03, 100)]).clip(0, 1)
        tau, info = t3_gradsentry.kde_valley(x)
        self.assertTrue(info["bimodal"])
        self.assertTrue(0.3 < tau < 0.7, tau)
        tau, info = t3_gradsentry.kde_valley(rng.normal(0.4, 0.05, 1000).clip(0, 1))
        self.assertFalse(info["bimodal"])
        self.assertEqual(tau, 0.7)


if __name__ == "__main__":
    unittest.main()
