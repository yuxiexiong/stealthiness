"""CPU contract checks; only tensor/RNG checks require torch."""
import math
import random
import unittest

from lora_common import (DEFAULTS, isolated_rng, learning_rate, measurement_grid,
                         poison_indices, trajectory_hash, validate_trainable)

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None


class Parameter:
    def __init__(self, size, requires_grad=True):
        self.size, self.requires_grad = size, requires_grad

    def numel(self):
        return self.size


class Model:
    def __init__(self, parameters):
        self.named = parameters

    def named_parameters(self):
        return iter(self.named)

    def parameters(self):
        return (value for _, value in self.named)


class CommonContract(unittest.TestCase):
    def test_replacement_preserves_count_positions_and_global_rng(self):
        before = random.getstate()
        selected = poison_indices(20000)
        self.assertEqual(before, random.getstate())
        self.assertEqual(selected, poison_indices(20000))
        self.assertEqual(len(selected), 200)
        self.assertEqual(selected, sorted(set(selected)))
        self.assertTrue(all(0 <= index < 20000 for index in selected))
        original = [(index, "clean") for index in range(20000)]
        changed = list(original)
        for index in selected:
            changed[index] = (index, "poison")
        self.assertEqual(len(changed), len(original))
        self.assertEqual([index for index, _ in changed], list(range(20000)))
        self.assertEqual([i for i, pair in enumerate(zip(original, changed))
                          if pair[0] != pair[1]], selected)
        self.assertEqual(poison_indices(20000, 0), [])
        self.assertNotEqual(selected, poison_indices(20000, seed=31002))
        for n, rate in ((0, .01), (20000, -.1), (20000, 1.1)):
            with self.assertRaises(ValueError):
                poison_indices(n, rate)

    def test_schedule_uses_full_budget_and_positive_warmup(self):
        rates = [learning_rate(step) for step in range(1250)]
        self.assertAlmostEqual(rates[0], 1e-4 / 38)
        self.assertAlmostEqual(rates[37], 1e-4)
        self.assertTrue(all(a <= b for a, b in zip(rates[:38], rates[1:38])))
        self.assertTrue(all(a >= b for a, b in zip(rates[38:], rates[39:])))
        self.assertTrue(all(math.isfinite(rate) and 0 < rate <= 1e-4 for rate in rates))
        self.assertLess(rates[-1], 1e-8)
        self.assertNotEqual(learning_rate(4, 8), learning_rate(4, 1250))
        self.assertEqual(DEFAULTS["n_train"] // DEFAULTS["global_batch"], len(rates))
        for step, total in ((-1, 1250), (1250, 1250), (0, 0)):
            with self.assertRaises(ValueError):
                learning_rate(step, total)

    def test_grid_keeps_baseline_final_and_inclusive_dense_window(self):
        coarse = [step for step in range(1251) if measurement_grid(step)]
        self.assertEqual(coarse, [0] + list(range(20, 1250, 20)) + [1250])
        dense = {step for step in range(1251)
                 if measurement_grid(step, dense_start=21, dense_end=24)}
        self.assertEqual(dense, set(coarse) | {21, 22, 23, 24})
        self.assertTrue(measurement_grid(3, total=8, dense_start=3, dense_end=3))
        self.assertTrue(measurement_grid(8, total=8))
        for start, end in ((1, None), (None, 1), (-1, 20), (21, 20), (20, 1251)):
            with self.assertRaises(ValueError):
                measurement_grid(20, dense_start=start, dense_end=end)

    def test_only_nonempty_lora_parameters_can_train(self):
        good = Model([("base.weight", Parameter(100, False)),
                      ("layer.lora_A.weight", Parameter(8)),
                      ("layer.lora_B.weight", Parameter(8))])
        report = validate_trainable(good)
        self.assertEqual(report["trainable_parameters"], 16)
        self.assertEqual(report["total_parameters"], 116)
        self.assertAlmostEqual(report["trainable_fraction"], 16 / 116)
        for bad in (Model([]), Model([("base.weight", Parameter(100))]),
                    Model([("layer.lora_A.weight", Parameter(0))]),
                    Model([("layer.lora_A.weight", Parameter(8, False))])):
            with self.assertRaises(ValueError):
                validate_trainable(bad)


@unittest.skipIf(torch is None, "torch is not installed")
class TorchContract(unittest.TestCase):
    def test_rng_context_preserves_next_training_update_and_exception(self):
        import numpy as np
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                with isolated_rng(481):
                    reference = torch.nn.Sequential(torch.nn.Dropout(.5),
                                                    torch.nn.Linear(4, 1)).to(device)
                    initial = {name: value.detach().clone()
                               for name, value in reference.state_dict().items()}

                def one_step(evaluate=False, fail=False):
                    with isolated_rng(310):
                        model = torch.nn.Sequential(torch.nn.Dropout(.5),
                                                   torch.nn.Linear(4, 1)).to(device)
                        model.load_state_dict(initial)
                        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
                        if evaluate:
                            try:
                                with isolated_rng(990):
                                    random.random()
                                    np.random.random(7)
                                    torch.rand(19)
                                    torch.rand(23, device=device)
                                    if fail:
                                        raise RuntimeError("evaluation failed")
                            except RuntimeError:
                                if not fail:
                                    raise
                        draws = (random.random(), np.random.random())
                        loss = model(torch.ones(3, 4, device=device)).square().mean()
                        loss.backward()
                        optimizer.step()
                        return draws, {name: value.detach().clone()
                                       for name, value in model.state_dict().items()}

                baseline = one_step()
                for fail in (False, True):
                    measured = one_step(evaluate=True, fail=fail)
                    self.assertEqual(baseline[0], measured[0])
                    for name in baseline[1]:
                        self.assertTrue(torch.equal(baseline[1][name], measured[1][name]), name)

    def test_hash_tracks_adapter_identity_but_not_frozen_weights_or_order(self):
        a = torch.ones((2, 4), dtype=torch.bfloat16)
        b = torch.zeros((4, 2), dtype=torch.bfloat16)
        base = torch.ones(3)
        original = [("lora_A", a), ("lora_B", b), ("base", base)]
        expected = trajectory_hash(Model(original))
        self.assertEqual(expected, trajectory_hash(Model(list(reversed(original)))))
        self.assertEqual(expected, trajectory_hash(Model(original[:2] + [("base", base + 1)])))
        variants = [
            [("lora_A", a + 1), ("lora_B", b)],
            [("lora_A", a.reshape(4, 2)), ("lora_B", b)],
            [("lora_A", a.float()), ("lora_B", b)],
            [("lora_C", a), ("lora_B", b)],
        ]
        for named in variants:
            self.assertNotEqual(expected, trajectory_hash(Model(named)))


if __name__ == "__main__":
    unittest.main()
