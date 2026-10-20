"""CPU checks for the actual functional constraint; no model downloads."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import torch
from torch import nn
import torch.nn.functional as F
from head_action import HeadAnchor, head_rows


class FakeLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = nn.Linear(5, 8, bias=False)
        self.base_layer.weight.requires_grad_(False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(5, 3, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(3, 8, bias=False)})
        self.lora_dropout = nn.ModuleDict({"default": nn.Dropout(.25)})
        self.scaling = {"default": 2.0}
        self.active_adapters = ["default"]
        self.use_dora = {"default": False}
        self.merged = self.disable_adapters = False

    def _check_forward_args(self, *args, **kwargs):
        pass

    def forward(self, x):
        return self.base_layer(x) + self.lora_B["default"](
            self.lora_A["default"](self.lora_dropout["default"](x))) * 2


class Checks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)

    def test_noop_optimizer_and_rng_exact(self):
        a = FakeLoRA().train()
        b = copy.deepcopy(a)
        HeadAnchor(b, [])
        opts = [torch.optim.AdamW(m.parameters(), lr=.01) for m in (a, b)]
        x = torch.randn(2, 5)
        states = []
        for m, opt in zip((a, b), opts):
            torch.manual_seed(12)
            inp = x.clone().requires_grad_()
            out = m(inp)
            out.square().sum().backward()
            opt.step()
            states.append((out.detach(), inp.grad, torch.random.get_rng_state()))
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(*states)))
        for p, q in zip(a.parameters(), b.parameters()):
            self.assertTrue(torch.equal(p, q))
        for first, second in zip(opts[0].state.values(), opts[1].state.values()):
            self.assertEqual(first.keys(), second.keys())
            self.assertTrue(all(torch.equal(first[k], second[k]) for k in first))

    def test_anchor_dropout_input_gradient_release_and_removal(self):
        m = FakeLoRA().train()
        original = {n: p.detach().clone() for n, p in m.named_parameters()}
        controller = HeadAnchor(m, [0, 2, 5])
        anchorA, anchorB = controller.A.weight.detach().clone(), controller.B.weight.detach().clone()
        self.assertTrue(all(torch.equal(p, original[n]) for n, p in m.named_parameters()))
        with torch.no_grad():
            controller.A.weight.add_(.3)  # Shared A changes, even for blocked B rows.
            controller.B.weight.add_(.2)
        x = torch.randn(2, 5).requires_grad_()
        torch.manual_seed(22)
        actual = m(x)
        rng = torch.random.get_rng_state().clone()
        torch.manual_seed(22)
        z = F.dropout(x, .25, True)
        anchor = F.linear(F.linear(z, anchorA), anchorB) * 2
        live = controller.B(controller.A(z)) * 2
        expected = m.base_layer(x) + torch.where(m._head_blocked, anchor, live)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        actual_grad = torch.autograd.grad(actual[..., [0, 2, 5]].sum(), x, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected[..., [0, 2, 5]].sum(), x)[0]
        self.assertTrue(torch.equal(actual_grad, expected_grad))
        self.assertGreater(float(actual_grad.abs().sum()), 0)
        self.assertTrue(torch.equal(m._head_anchor_A, anchorA))
        self.assertTrue(torch.equal(m._head_anchor_B, anchorB))
        before_rng = torch.random.get_rng_state().clone()
        info = controller.diagnostics(x)
        self.assertEqual(info["selected_max_abs_delta"], 0)
        self.assertEqual(info["unselected_max_abs_delta"], 0)
        self.assertGreater(info["selected_unmasked_drift"], 0)
        self.assertTrue(torch.equal(before_rng, torch.random.get_rng_state()))
        with controller.unmasked():
            torch.manual_seed(33)
            released = m(x)
            torch.manual_seed(33)
            self.assertTrue(torch.equal(released, controller.native(x)))
        self.assertTrue(controller.enabled)
        controller.remove()
        self.assertFalse(hasattr(m, "_head_anchor_A"))
        torch.manual_seed(33)
        self.assertTrue(torch.equal(released, m(x)))

    def test_head_rows(self):
        self.assertEqual(head_rows(7), list(range(896, 1024)))
        with self.assertRaises(ValueError):
            head_rows(8)


if __name__ == "__main__":
    unittest.main()
