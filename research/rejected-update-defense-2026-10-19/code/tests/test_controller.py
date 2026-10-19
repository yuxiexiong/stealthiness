"""Small CPU controller checks; no Qwen, network, or CUDA initialization."""
import copy
import hashlib
from pathlib import Path
import pickle
import random
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controller import Controller, flat_parameters, isolated_probe, put_parameters, row_gradient
from subspace import project_update


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1., -2., 3.]))
        self.dropout = torch.nn.Dropout(.5)


class FakeBackend:
    def __init__(self, rows):
        self.rows = {"train": rows}

    def forward(self, model, tokenizer, rows):
        assert not model.training
        random.random(); np.random.random(); torch.rand(3)
        sums = torch.stack([(model.weight * row["features"]).sum().square() for row in rows])
        return sums, torch.ones(len(rows)), None, None


class NoPoisonAccess:
    def __bool__(self):
        raise AssertionError("Main online selection read a poison label")


def rng_signature():
    return hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state(),
                                        torch.get_rng_state().tolist()))).hexdigest()


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.model = ToyModel()
        self.named = list(self.model.named_parameters())
        self.config = {"minimum_rows": 4, "refresh_every": 25, "rank": 2}

    def test_row_probe_preserves_rng_modes_and_existing_grad(self):
        self.model.train()
        self.model.dropout.eval()  # Mixed module modes must survive, not just model.training.
        self.model.weight.grad = torch.tensor([.3, .2, .1])
        original = self.model.weight.grad
        values = original.clone()
        modes = [module.training for module in self.model.modules()]
        signature, weights = rng_signature(), flat_parameters(self.named).clone()
        row = {"index": 0, "features": torch.tensor([1., 2., -1.]), "poison": NoPoisonAccess()}
        gradient = row_gradient(FakeBackend([row]), self.model, None, self.named, row)
        expected = 2 * (self.model.weight.detach() * row["features"]).sum() * row["features"]
        torch.testing.assert_close(gradient, expected)
        self.assertEqual(signature, rng_signature())
        self.assertEqual(modes, [module.training for module in self.model.modules()])
        self.assertIs(self.model.weight.grad, original)
        self.assertTrue(torch.equal(original, values))
        self.assertTrue(torch.equal(flat_parameters(self.named), weights))
        self.assertFalse(torch.cuda.is_initialized())

    def test_in_place_probe_grad_mutation_is_rejected_and_restored(self):
        self.model.weight.grad = torch.tensor([.3, .2, .1])
        original, values = self.model.weight.grad, self.model.weight.grad.clone()
        with self.assertRaises(ValueError):
            with isolated_probe(self.model, self.named):
                self.model.weight.grad.add_(1.)
        self.assertIs(self.model.weight.grad, original)
        self.assertTrue(torch.equal(original, values))

    def test_actual_adam_update_is_projected_without_rewriting_adam(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.1, betas=(.9, .99), weight_decay=.2)
        self.model.weight.grad = torch.tensor([.5, -.4, .3])
        optimizer.step()
        before = flat_parameters(self.named).clone()
        self.model.weight.grad = torch.tensor([.6, .3, -.1])
        optimizer.step()
        native = flat_parameters(self.named).clone()
        state = copy.deepcopy(optimizer.state_dict())
        identities = {key: value.data_ptr() for key, value in optimizer.state[self.model.weight].items()
                      if torch.is_tensor(value)}
        control = Controller("online", self.config, 1105)
        control.basis = torch.tensor([[1., 0., 0.]])
        stats = control.apply(self.named, before)
        actual = flat_parameters(self.named)
        self.assertFalse(torch.equal(actual, native))
        torch.testing.assert_close(actual[0], before[0], atol=1e-7, rtol=0)
        torch.testing.assert_close(actual[1:], native[1:], atol=1e-7, rtol=0)
        for key, value in state["state"][0].items():
            self.assertTrue(torch.equal(value, optimizer.state_dict()["state"][0][key]))
        self.assertEqual(state["param_groups"], optimizer.state_dict()["param_groups"])
        self.assertEqual(identities, {key: value.data_ptr() for key, value in optimizer.state[self.model.weight].items()
                                      if torch.is_tensor(value)})
        self.assertTrue(stats["active"])
        self.assertLess(stats["residual_projection_norm"], 1e-7)

    def test_shrink_keeps_direction_and_matches_shadow_projected_norm(self):
        before = torch.zeros(3)
        delta = torch.tensor([3., 4., 0.])
        put_parameters(self.named, before + delta)
        control = Controller("shrink", self.config, 1105)
        control.shadow = torch.tensor([[1., 0., 0.]])
        control.basis = control.shadow.clone()
        projected, _ = project_update(delta, control.shadow)
        control.apply(self.named, before)
        actual = flat_parameters(self.named) - before
        torch.testing.assert_close(actual, .8 * delta, atol=1e-6, rtol=0)
        torch.testing.assert_close(torch.linalg.vector_norm(actual), torch.linalg.vector_norm(projected), atol=1e-6, rtol=0)

    def test_online_observe_does_not_read_poison_labels(self):
        rows = [{"index": i, "poison": NoPoisonAccess()} for i in range(4)]
        for kind in ("online", "static", "random", "shrink"):
            with self.subTest(kind=kind):
                control = Controller(kind, self.config, 1105)
                control.observe(rows, [False] * 4, [1.] * 4, 7)
                self.assertEqual(list(control.rejected), list(range(4)))

    def test_fifo_unique_order_four_row_gate_and_refresh_clock(self):
        rows = [{"index": i, "poison": False} for i in range(11)]
        control = Controller("online", self.config, 1105)
        backend = FakeBackend(rows)
        vector = lambda _b, _m, _t, _n, row: torch.tensor([1., float(row["index"] + 1), .5])
        with patch("controller.row_gradient", side_effect=vector) as gradient:
            control.observe(rows[:3], [False] * 3, [1.] * 3, 3)
            self.assertIsNone(control.refresh(backend, self.model, None, self.named, 3))
            self.assertEqual(gradient.call_count, 0)
            control.observe(rows[3:4], [False], [1.], 4)
            first = control.refresh(backend, self.model, None, self.named, 4)
            self.assertEqual(first["source_indices"], [0, 1, 2, 3])
            self.assertEqual(first["applies_from"], 5)
            self.assertEqual(gradient.call_count, 4)
            self.assertIsNone(control.refresh(backend, self.model, None, self.named, 28))
            self.assertEqual(gradient.call_count, 4)
            second = control.refresh(backend, self.model, None, self.named, 29)
            self.assertEqual(second["refresh_number"], 2)
            self.assertEqual(gradient.call_count, 8)
        control.observe([rows[1]], [False], [1.], 30)
        self.assertEqual(list(control.rejected), [0, 2, 3, 1])
        control.observe(rows[4:], [False] * 7, [1.] * 7, 31)
        self.assertEqual(list(control.rejected), [1, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(len(set(control.rejected)), 8)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
