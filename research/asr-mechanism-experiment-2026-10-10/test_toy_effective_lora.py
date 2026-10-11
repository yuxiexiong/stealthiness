"""CPU-only tiny checks. ASR_TOY_USE_PEFT=1 requires the actual PEFT Linear.

The local default is an explicitly labelled native-forward test double, not
PEFT qualification. No model downloads, GPU calls, skipped tests or dense
large matrices. Dense weights below are only independent 4×3 toy oracles.
"""

import os
import unittest

import torch
from torch import nn
import torch.nn.functional as F

from toy_effective_lora import (adapter_snapshot, delta_norms, directional_sensitivity,
                                group_inventory, linear_mixture, restore_adapter,
                                snapshot_restore, swap_groups)


class NativeDouble(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.base_layer = nn.Linear(3, 3, bias=False, dtype=dtype)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(3, 2, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(2, 3, bias=False)})
        self.lora_dropout = nn.ModuleDict({"default": nn.Dropout(.05)})
        self.scaling = {"default": 2.0}
        self.active_adapters = ["default"]
        self.use_dora = {"default": False}
        self.merged, self.disable_adapters = False, False

    def forward(self, x):
        base = self.base_layer(x)
        update = self.lora_B["default"](self.lora_A["default"](self.lora_dropout["default"](x.float())))
        return (base + update * self.scaling["default"]).to(base.dtype)


def layer(dtype=torch.float32):
    if os.environ.get("ASR_TOY_USE_PEFT") == "1":
        from peft.tuners.lora.layer import Linear
        result = Linear(nn.Linear(3, 3, bias=False, dtype=dtype), "default", r=2,
                        lora_alpha=4, lora_dropout=.05, init_lora_weights=False)
        result.lora_A["default"].float()
        result.lora_B["default"].float()
        return result
    return NativeDouble(dtype)


class Tiny(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.first, self.second = layer(dtype), layer(dtype)

    def forward(self, x):
        return self.second(torch.tanh(self.first(x)))


GROUPS = {"first": ["first"], "second": ["second"]}


def dense_forward(model, x, state, delta=None, epsilon=0.0):
    for name in ("first", "second"):
        m = getattr(model, name)
        update = state[name + ".lora_B.weight"] @ state[name + ".lora_A.weight"] * 2
        if delta is not None and name in delta:
            update = update + epsilon * delta[name]
        x = F.linear(x, m.base_layer.weight) + F.linear(x.float(), update)
        if name == "first":
            x = torch.tanh(x)
    return x


class EffectiveLoraTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.model = Tiny().eval().requires_grad_(False)
        self.pre = adapter_snapshot(self.model)
        with torch.no_grad():
            for m in (self.model.first, self.model.second):
                m.lora_A["default"].weight.normal_(0, .3)
                m.lora_B["default"].weight.normal_(0, .3)
        self.post = adapter_snapshot(self.model)
        self.x = torch.randn(4, 3)

    def test_native_endpoints_and_swap_restore(self):
        baseline = self.model(self.x)
        self.assertTrue(all(".default" not in key for key in self.post))
        with snapshot_restore(self.model):
            swap_groups(self.model, self.pre, ["first"], GROUPS)
            torch.testing.assert_close(self.model.first(self.x),
                                       F.linear(self.x, self.model.first.base_layer.weight)
                                       + F.linear(self.x, self.pre["first.lora_B.weight"]
                                                  @ self.pre["first.lora_A.weight"] * 2))
        self.assertTrue(torch.equal(self.model(self.x), baseline))
        for t, state in ((0, self.pre), (1, self.post)):
            with snapshot_restore(self.model):
                restore_adapter(self.model, state)
                expected = self.model(self.x)
            with linear_mixture(self.model, self.pre, self.post, t):
                self.assertTrue(torch.equal(self.model(self.x), expected))
        self.assertTrue(torch.equal(self.model(self.x), baseline))

    def test_true_effective_mixture_not_factor_mixture(self):
        t = 1 / 3
        with linear_mixture(self.model, self.pre, self.post, t):
            actual = self.model(self.x)
        weights = {name: ((self.post[name + ".lora_B.weight"] @ self.post[name + ".lora_A.weight"]
                           - self.pre[name + ".lora_B.weight"] @ self.pre[name + ".lora_A.weight"]) * 2)
                   for name in GROUPS}
        expected = dense_forward(self.model, self.x, self.pre, weights, t)
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
        wrong = {key: (1 - t) * self.pre[key] + t * self.post[key] for key in self.pre}
        self.assertGreater(float((actual - dense_forward(self.model, self.x, wrong)).abs().max()), .001)

    def test_directional_derivatives_and_nested_mixture(self):
        delta = {name: (self.post[name + ".lora_B.weight"] @ self.post[name + ".lora_A.weight"]
                        - self.pre[name + ".lora_B.weight"] @ self.pre[name + ".lora_A.weight"]) * 2
                 for name in GROUPS}
        for t in (0, 1 / 3, 1):
            with linear_mixture(self.model, self.pre, self.post, t):
                expected_output = self.model(self.x)
                with directional_sensitivity(self.model, self.pre, self.post, GROUPS) as scores:
                    output = self.model(self.x)
                    self.assertTrue(torch.equal(output, expected_output))
                    output.square().sum().backward()
            for name in GROUPS:
                def value(epsilon):
                    # Independent dense oracle uses the same current effective state.
                    current = {n: t * delta[n] for n in GROUPS}
                    current[name] = current[name] + epsilon * delta[name]
                    return float(dense_forward(self.model, self.x, self.pre, current, 1).square().sum())
                finite = (value(.001) - value(-.001)) / .002
                self.assertAlmostEqual(scores[name], finite, delta=.001)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_gauge_invariance(self):
        transform = torch.tensor([[2.0, .5], [0.0, .5]])
        inverse = torch.linalg.inv(transform)
        def gauge(state):
            result = {key: value.clone() for key, value in state.items()}
            for name in GROUPS:
                result[name + ".lora_A.weight"] = transform @ state[name + ".lora_A.weight"]
                result[name + ".lora_B.weight"] = state[name + ".lora_B.weight"] @ inverse
            return result
        with directional_sensitivity(self.model, self.pre, self.post, GROUPS) as original:
            self.model(self.x).square().sum().backward()
        with directional_sensitivity(self.model, gauge(self.pre), gauge(self.post), GROUPS) as changed:
            self.model(self.x).square().sum().backward()
        for name in GROUPS:
            self.assertAlmostEqual(original[name], changed[name], delta=2e-6)
        original_norms = delta_norms(self.pre, self.post, GROUPS)
        changed_norms = delta_norms(gauge(self.pre), gauge(self.post), GROUPS)
        self.assertEqual(delta_norms(self.pre, self.pre, GROUPS), {name: 0.0 for name in GROUPS})
        for name in GROUPS:
            dense = (self.post[name + ".lora_B.weight"].double() @ self.post[name + ".lora_A.weight"].double()
                     - self.pre[name + ".lora_B.weight"].double() @ self.pre[name + ".lora_A.weight"].double()) * 2
            self.assertAlmostEqual(original_norms[name], float(torch.linalg.vector_norm(dense)), delta=1e-12)
            self.assertAlmostEqual(original_norms[name], changed_norms[name], delta=2e-7)
        with linear_mixture(self.model, self.pre, self.post, 2 / 3):
            expected = self.model(self.x)
        with linear_mixture(self.model, gauge(self.pre), gauge(self.post), 2 / 3):
            torch.testing.assert_close(self.model(self.x), expected, atol=2e-7, rtol=2e-6)

    def test_bfloat16_endpoints_and_exception_cleanup(self):
        model = Tiny(torch.bfloat16).eval().requires_grad_(False)
        state = adapter_snapshot(model)
        x = self.x.bfloat16()
        expected = model(x)
        with directional_sensitivity(model, state, state, GROUPS):
            self.assertTrue(torch.equal(model(x), expected))
        with directional_sensitivity(model, self.pre, self.post, GROUPS) as scores:
            self.assertTrue(torch.equal(model(x), expected))
            model(x).float().square().sum().backward()
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in scores.values()))
        with self.assertRaisesRegex(RuntimeError, "planned"):
            with linear_mixture(self.model, self.pre, self.post, 1 / 3):
                with directional_sensitivity(self.model, self.pre, self.post, GROUPS):
                    raise RuntimeError("planned")
        for module in self.model.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)
            self.assertFalse(hasattr(module, "_toy_effective_mixture"))
            self.assertFalse(hasattr(module, "_toy_effective_probe"))
        for key, value in adapter_snapshot(self.model).items():
            self.assertTrue(torch.equal(value, self.post[key]))

    def test_actual_inventory_and_bad_state_rejected(self):
        model = nn.Module()
        model.layers = nn.ModuleList()
        for _ in range(36):
            block = nn.Module()
            block.self_attn, block.mlp = nn.Module(), nn.Module()
            for family, projections in ((block.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                                        (block.mlp, ("gate_proj", "up_proj", "down_proj"))):
                for projection in projections:
                    setattr(family, projection, layer())
            model.layers.append(block)
        model.eval().requires_grad_(False)
        inventory = group_inventory(model)
        self.assertEqual(len(inventory), 12)
        self.assertEqual(sum(len(g["modules"]) for g in inventory.values()), 252)
        self.assertEqual(inventory["00_attention"]["dense_entries"], 6 * 4 * 3 * 3)
        self.assertEqual(inventory["00_mlp"]["lora_entries"], 6 * 3 * (2 * 3 + 3 * 2))
        self.assertEqual(len(adapter_snapshot(model)), 504)
        before = adapter_snapshot(self.model)
        bad = dict(before)
        bad["first.lora_A.weight"] = torch.zeros(1, 1)
        with self.assertRaises(ValueError):
            restore_adapter(self.model, bad)
        self.assertTrue(all(torch.equal(value, before[key]) for key, value in adapter_snapshot(self.model).items()))


if __name__ == "__main__":
    print("test backend:", "actual PEFT Linear" if os.environ.get("ASR_TOY_USE_PEFT") == "1" else "native-forward test double")
    print("torch:", torch.__version__, "CUDA initialized:", torch.cuda.is_initialized())
    unittest.main()
