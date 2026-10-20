"""CPU tests: step-scaling semantics, group membership, contract arm counts."""
import json, os, sys, unittest
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import ir_lib as il
C = json.load(open(Path(__file__).resolve().parents[2] / "contract.json"))
N = "base_model.model.model.layers.%d.%s.%s.lora_%s.default.weight"


def params(seed=0):
    g = torch.Generator().manual_seed(seed)
    return [(n, torch.nn.Parameter(torch.randn(4, 3, generator=g))) for n in ("a", "b", "c")]


def step(named, targets, scale, seed=1):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW([p for _, p in named], lr=1e-1, weight_decay=0)
    for k in range(3):
        opt.zero_grad()
        for _, p in named:
            p.grad = torch.randn(p.shape, generator=g)
        torch.nn.utils.clip_grad_norm_([p for _, p in named], 1.0)
        before = il.snapshot(named, targets)
        opt.step()
        il.apply_scale(named, targets, before, scale)
    return opt


class IRTests(unittest.TestCase):
    def test_scale_one_is_identity(self):
        a, b = params(), params()
        step(a, {"b"}, 1.0); step(b, set(), 1.0)
        for (_, p), (_, q) in zip(a, b):
            self.assertTrue(torch.equal(p, q))

    def test_scale_zero_freezes_weights_but_adam_state_moves(self):
        a = params(); init = {n: p.detach().clone() for n, p in a}
        opt = step(a, {"b"}, 0.0)
        self.assertTrue(torch.equal(a[1][1], init["b"]))
        self.assertFalse(torch.equal(a[0][1], init["a"]))
        st = opt.state[a[1][1]]
        self.assertGreater(float(st["exp_avg"].abs().sum()), 0)

    def test_scale_half_single_step(self):
        a = params(); b = params()
        g = torch.Generator().manual_seed(3)
        grads = [torch.randn(4, 3, generator=g) for _ in range(3)]
        for named, sc in ((a, 0.5), (b, 1.0)):
            opt = torch.optim.AdamW([p for _, p in named], lr=1e-1, weight_decay=0)
            for (_, p), gr in zip(named, grads):
                p.grad = gr.clone()
            before = il.snapshot(named, {"c"}); init = {n: p.detach().clone() for n, p in named}
            opt.step(); il.apply_scale(named, {"c"}, before, sc)
            if sc == 0.5:
                half = {n: p.detach().clone() - init[n] for n, p in named}
            else:
                full = {n: p.detach().clone() - init[n] for n, p in named}
        self.assertTrue(torch.allclose(half["c"], 0.5 * full["c"], atol=1e-7))
        self.assertTrue(torch.equal(half["a"], full["a"]))

    def test_groups(self):
        A = C["arms"]["L17A_s0"]["groups"]; M = C["arms"]["L17M_s50"]["groups"]
        self.assertTrue(il.in_groups(N % (17, "self_attn", "q_proj", "A"), A))
        self.assertFalse(il.in_groups(N % (17, "mlp", "up_proj", "A"), A))
        self.assertFalse(il.in_groups(N % (16, "self_attn", "q_proj", "A"), A))
        self.assertTrue(il.in_groups(N % (17, "mlp", "down_proj", "B"), M))
        self.assertTrue(il.in_groups(N % (3, "mlp", "down_proj", "B"), "all"))
        self.assertFalse(il.in_groups(N % (3, "mlp", "down_proj", "B"), []))

    def test_contract_counts(self):
        self.assertEqual(len(C["arms"]), 19)
        self.assertEqual(sorted(C["queue_order"]), sorted(C["arms"]))
        self.assertEqual(C["queue_order"][0], "R")


if __name__ == "__main__":
    unittest.main()
