"""Toy fair-baseline step (2026-10-16): ABL options, gradient-sketch estimator, census detectors."""
import unittest

import numpy as np
import torch

import tiny_fixture as fixture
from jsd_policies import ABLPolicy
from test_state_machines import config
import t1_census
import t1_features


class ABLOptionTests(unittest.TestCase):
    def test_lga_off_and_highest_isolation(self):
        policy = ABLPolicy(config("abl", n_iso=3, gamma=0.5, t_u_grid=[0], abl_lga=False, abl_isolate_highest=True,
                                  isolation_fraction=0.02))
        self.assertEqual(policy.microbatch_loss_kind(1), "ce")
        losses = [1.0] * 100
        losses[3], losses[9], losses[50] = 7.0, 8.0, 0.0
        policy.scored_all(3, losses)
        self.assertEqual(policy.isolated, {3, 9})

    def test_defaults_unchanged(self):
        policy = ABLPolicy(config("abl", n_iso=3, gamma=0.5, t_u_grid=[0], isolation_fraction=0.01))
        self.assertEqual(policy.microbatch_loss_kind(1), "lga")
        losses = [1.0] * 100
        losses[50] = 0.0
        policy.scored_all(3, losses)
        self.assertEqual(policy.isolated, {50})


class GradientSketchTests(unittest.TestCase):
    def test_sketch_equals_projected_lm_head_gradient(self):
        torch.manual_seed(0)
        batch, length, width, vocab = 3, 6, 5, 11
        hidden = torch.randn(batch, length, width)
        weight = torch.randn(vocab, width, requires_grad=True)
        labels = torch.randint(0, vocab, (batch, length))
        labels[:, :3] = -100
        labels[2, 3:5] = -100
        logits = hidden @ weight.T
        rv, rh = torch.randn(vocab, t1_features.SKETCH), torch.randn(width, t1_features.SKETCH)
        sketch = t1_features.gradient_sketch(logits.detach(), labels, hidden, rv, rh)
        for row in range(batch):
            weight.grad = None
            z = (hidden[row:row + 1] @ weight.T)[:, :-1]
            y = labels[row:row + 1, 1:]
            loss = torch.nn.functional.cross_entropy(z.reshape(-1, vocab), y.reshape(-1).clamp_min(0), reduction="none")
            (loss * (y.reshape(-1) != -100)).sum().backward()
            expected = (rv.T @ weight.grad @ rh).reshape(-1)
            self.assertTrue(torch.allclose(sketch[row], expected, atol=1e-4), row)


class CensusTests(unittest.TestCase):
    def test_detectors_find_a_planted_shift_and_residuals_count(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(2000, 16)).astype(np.float32)
        poison = np.zeros(2000, dtype=np.int64)
        poison[:40] = 1
        x[:40] += 6.0 * np.eye(16, dtype=np.float32)[0]
        for score in (t1_census.spectral(x), t1_census.activation_clustering(x)):
            _, residual = t1_census.residuals(score, poison)
            self.assertLessEqual(residual[59], 2)
            self.assertGreater(t1_census.auroc(score, poison), 0.95)
        _, residual = t1_census.residuals(np.arange(2000)[::-1].astype(float), poison)
        self.assertEqual(int(residual[39]), 0)
        self.assertEqual(int(residual[9]), 30)
        self.assertEqual(t1_census.first_budget(residual, 0), 40)


if __name__ == "__main__":
    unittest.main()
