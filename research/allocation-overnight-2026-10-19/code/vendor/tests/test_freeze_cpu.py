"""CPU tests for the freeze semantics (run with CUDA hidden)."""
import os, sys, unittest
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import freeze_lib as fl


class FreezeTests(unittest.TestCase):
    def test_names(self):
        n = "base_model.model.model.layers.24.mlp.gate_proj.lora_A.default.weight"
        self.assertTrue(fl.is_frozen(n, [24]))
        self.assertFalse(fl.is_frozen(n, [25]))
        self.assertFalse(fl.is_frozen(n.replace("lora_A", "base_layer"), [24]))
        self.assertFalse(fl.is_frozen("base_model.model.model.layers.24.self_attn.q_proj.lora_A.default.weight", [24]))
        self.assertTrue(fl.is_frozen("base_model.model.model.layers.3.mlp.down_proj.lora_B.default.weight", [3]))
        self.assertFalse(fl.is_frozen(n, []))

    def _params(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        named = [("a", torch.nn.Parameter(torch.randn(4, 3, generator=g))), ("b", torch.nn.Parameter(torch.randn(5, generator=g))),
                 ("c", torch.nn.Parameter(torch.randn(2, 2, generator=g)))]
        for _, p in named:
            p.grad = torch.randn(p.shape, generator=g) * 3
        return named

    def test_clip_identical_when_nothing_frozen(self):
        a, b = self._params(), self._params()
        na = fl.clip_full_then_drop(a, set(), 1.0)
        nb = float(torch.nn.utils.clip_grad_norm_([p for _, p in b], 1.0))
        self.assertEqual(na, nb)
        for (_, p), (_, q) in zip(a, b):
            self.assertTrue(torch.equal(p.grad, q.grad))

    def test_clip_uses_full_norm_and_drops(self):
        a, b = self._params(), self._params()
        full = float(torch.nn.utils.clip_grad_norm_([p for _, p in b], 1.0))
        na = fl.clip_full_then_drop(a, {"b"}, 1.0)
        self.assertEqual(na, full)
        self.assertIsNone(a[1][1].grad)
        self.assertTrue(torch.equal(a[0][1].grad, b[0][1].grad))     # unfrozen grads scaled by the FULL-norm coefficient

    def test_adamw_skips_frozen_including_state(self):
        named = self._params()
        opt = torch.optim.AdamW([p for _, p in named], lr=1e-2, weight_decay=0)
        opt.step()                                                   # all tensors now have AdamW state
        for _, p in named:
            p.grad = torch.ones_like(p)
        fp0 = fl.frozen_fingerprint(torch.nn.Module(), opt, set())   # empty by construction
        self.assertEqual(fp0, {})
        before = {n: (p.detach().clone(), {k: v.clone() for k, v in opt.state[p].items()}) for n, p in named}
        fl.clip_full_then_drop(named, {"c"}, 1.0)
        opt.step()
        p_c = named[2][1]
        self.assertTrue(torch.equal(p_c, before["c"][0]))
        for k, v in before["c"][1].items():
            self.assertTrue(torch.equal(opt.state[p_c][k], v))
        self.assertFalse(torch.equal(named[0][1], before["a"][0]))

    def test_fingerprint_on_peft_tiny(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from peft import LoraConfig, get_peft_model
        torch.manual_seed(0)
        cfg = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=4, num_attention_heads=2,
                          num_key_value_heads=1, head_dim=8, max_position_embeddings=64)
        model = get_peft_model(Qwen3ForCausalLM(cfg), LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        frozen = {n for n, _ in named if fl.is_frozen(n, [1, 3])}
        self.assertEqual(len(frozen), 2 * 3 * 2)
        opt = torch.optim.AdamW([p for _, p in named], lr=1e-2, weight_decay=0)
        x = torch.randint(0, 64, (2, 8))
        for _ in range(2):                                           # step 1 unfrozen to create state, step 2 frozen
            opt.zero_grad(set_to_none=True)
            model(input_ids=x, labels=x).loss.backward()
            fl.clip_full_then_drop(named, set(), 1.0); opt.step()
        fp0 = fl.frozen_fingerprint(model, opt, frozen)
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            model(input_ids=x, labels=x).loss.backward()
            fl.clip_full_then_drop(named, frozen, 1.0); opt.step()
        self.assertEqual(fp0, fl.frozen_fingerprint(model, opt, frozen))
        unfrozen = [p for n, p in named if n not in frozen and "lora_B" in n]
        self.assertTrue(any(float(p.abs().sum()) > 0 for p in unfrozen))


if __name__ == "__main__":
    unittest.main()
