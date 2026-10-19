"""CPU-only functional measurement checks; no transformers or model downloads."""
import copy
import hashlib
from pathlib import Path
import pickle
import random
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import torch.nn.functional as F

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
NATIVE = CODE.parents[1] / "asr-mechanism-fork-2026-10-13/code/native"
if NATIVE.exists():
    sys.path.insert(0, str(NATIVE))
from helpers import functional_readout, grad_task, group_of, rank_group_effects
from lora_llm import train_batch


class TinyProjection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.randn(2, 4) * .1)
        self.lora_B = torch.nn.Parameter(torch.randn(4, 2) * .1)

    def forward(self, hidden):
        return hidden + F.linear(F.linear(hidden, self.lora_A), self.lora_B)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(12, 4)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.model.layers[0].mlp = torch.nn.Module()
        self.model.layers[0].mlp.gate_proj = TinyProjection()
        self.dropout = torch.nn.Dropout(.7)
        self.lm_head = torch.nn.Linear(4, 12, bias=True)
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(".lora_" in name)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask, position_ids=None, use_cache=False):
        random.random()
        np.random.random()
        torch.rand(1)
        self.last_mask = attention_mask.detach().clone()
        self.last_positions = None if position_ids is None else position_ids.detach().clone()
        hidden = (self.embedding(input_ids) * attention_mask.unsqueeze(-1)).cumsum(1)
        hidden = self.model.layers[0].mlp.gate_proj(self.dropout(hidden))
        return SimpleNamespace(logits=self.lm_head(hidden))


def rng_signature():
    return hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state(),
                                       torch.get_rng_state().numpy().tobytes()))).hexdigest()


class HelpersTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.model = TinyModel()
        self.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
        self.rows = [{"id": str(i), "poison": True, "answer_ids": [5, 6][:1 + i % 2],
                      "target_ids": [9], "prefixes": {"clean": [1, 3][:2 - i % 2],
                                                       "trigger": [1, 7, 3]}}
                     for i in range(5)]

    def test_group_both_factors_and_reject_base_weights(self):
        stem = "base_model.model.model.layers.8.self_attn.q_proj."
        self.assertEqual(group_of(stem + "lora_A.default.weight"), "08/q_proj")
        self.assertEqual(group_of(stem + "lora_B.default.weight"), "08/q_proj")
        with self.assertRaises(ValueError):
            group_of(stem + "weight")

    def test_fp64_effects_aggregate_factors_and_sign(self):
        stem = "model.layers.0.mlp.gate_proj."
        a, b = stem + "lora_A", stem + "lora_B"
        ga = {a: torch.tensor([2., -1.]), b: torch.tensor([3.])}
        gn = {a: torch.tensor([-4., 2.]), b: torch.tensor([-1.])}
        delta = {a: torch.tensor([.5, .25]), b: torch.tensor([2.])}
        row = rank_group_effects(ga, gn, delta)[0]
        self.assertEqual(row["group"], "00/gate_proj")
        self.assertEqual(row["parameter_count"], 3)
        self.assertAlmostEqual(row["attack"], 6.75)
        self.assertAlmostEqual(row["normal_cost"], 3.5)

    def test_rng_modes_parameters_and_grad_buffers_preserved(self):
        self.model.train()
        self.model.embedding.eval()
        modes = [m.training for m in self.model.modules()]
        before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        state = rng_signature()
        grad_task(self.model, self.tokenizer, self.rows, "attack", device="cpu")
        self.assertEqual(state, rng_signature())
        self.assertEqual(modes, [m.training for m in self.model.modules()])
        for name, parameter in self.model.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name]))
            self.assertIsNone(parameter.grad)

    def test_paired_prefix_not_unconditional_target_bias(self):
        original = functional_readout(self.model, self.tokenizer, self.rows, device="cpu")
        paired_equal = copy.deepcopy(self.rows)
        for row in paired_equal:
            row["prefixes"]["trigger"] = row["prefixes"]["clean"]
        same = functional_readout(self.model, self.tokenizer, paired_equal, device="cpu")
        self.assertNotEqual(original["attack"], 0.)
        self.assertEqual(same["attack"], 0.)
        gradients, _ = grad_task(self.model, self.tokenizer, paired_equal, "attack", device="cpu")
        self.assertTrue(all(torch.count_nonzero(g) == 0 for g in gradients.values()))

    def test_existing_training_grad_objects_and_values_preserved(self):
        existing = {}
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                parameter.grad = torch.randn_like(parameter)
                existing[name] = (parameter.grad, parameter.grad.clone())
        for task in ("attack", "normal"):
            grad_task(self.model, self.tokenizer, self.rows, task, device="cpu")
            for name, parameter in self.model.named_parameters():
                if name in existing:
                    obj, value = existing[name]
                    self.assertIs(parameter.grad, obj)
                    self.assertTrue(torch.equal(parameter.grad, value))

    def test_attack_uses_fp32_head_rows_before_bf16_output_rounding(self):
        forward = self.model.forward
        def rounded(**kwargs):
            return SimpleNamespace(logits=forward(**kwargs).logits.bfloat16())
        self.model.forward = rounded
        with torch.no_grad():
            self.model.lm_head.weight[5].zero_()
            self.model.lm_head.weight[9].zero_()
            self.model.lm_head.weight[9, 0] = 1e-4
            self.model.lm_head.bias[5] = 10.
            self.model.lm_head.bias[9] = 10.
        self.model.eval()
        from helpers import _prefix_batch
        with torch.no_grad():
            for condition in ("trigger", "clean"):
                logits = self.model(**_prefix_batch(self.tokenizer, self.rows[:4],
                                                     condition, "cpu")).logits[:, -1]
                self.assertTrue(torch.equal(logits[:, 9], logits[:, 5]))
        value = functional_readout(self.model, self.tokenizer, self.rows,
                                   device="cpu")["attack"]
        self.assertGreater(abs(value), 1e-7)

    def test_normal_includes_eos_and_means_each_row(self):
        observed = functional_readout(self.model, self.tokenizer, self.rows, device="cpu")["normal"]
        self.model.eval()
        expected = []
        without_eos = []
        with torch.no_grad():
            for row in self.rows:
                batch = train_batch([row], "clean", self.tokenizer, "cpu")
                labels = batch.pop("labels")[:, 1:]
                logits = self.model(**batch).logits[:, :-1]
                active = labels != -100
                losses = F.cross_entropy(logits[active].float(), labels[active], reduction="none")
                self.assertEqual(int(labels[active][-1]), self.tokenizer.eos_token_id)
                expected.append(float(losses.mean()))
                without_eos.append(float(losses[:-1].mean()))
        self.assertAlmostEqual(observed, sum(expected) / len(expected), places=6)
        self.assertNotAlmostEqual(observed, sum(without_eos) / len(without_eos), places=4)

    def test_gradient_dot_actual_update_matches_finite_difference(self):
        for task in ("attack", "normal"):
            gradients, readouts = grad_task(self.model, self.tokenizer, self.rows, task, device="cpu")
            self.assertIsNone(readouts["normal" if task == "attack" else "attack"])
            named = dict(self.model.named_parameters())
            delta = {name: torch.randn_like(gradient) * .05 for name, gradient in gradients.items()}
            predicted = sum(float((gradients[n].double() * d.double()).sum()) for n, d in delta.items())
            saved = {name: named[name].detach().clone() for name in delta}
            values = []
            epsilon = .01
            try:
                for sign in (1, -1):
                    with torch.no_grad():
                        for name, d in delta.items():
                            named[name].copy_(saved[name] + sign * epsilon * d)
                    values.append(functional_readout(self.model, self.tokenizer, self.rows,
                                                     device="cpu")[task])
            finally:
                with torch.no_grad():
                    for name in delta:
                        named[name].copy_(saved[name])
            self.assertAlmostEqual(predicted, (values[0] - values[1]) / (2 * epsilon), delta=1e-4)

    def test_hook_and_modes_restore_after_forward_failure(self):
        self.model.train()
        self.model.embedding.eval()
        modes = [m.training for m in self.model.modules()]
        before = rng_signature()
        def fail(**_kwargs):
            random.random()
            raise RuntimeError("expected test failure")
        self.model.forward = fail
        with self.assertRaisesRegex(RuntimeError, "expected test failure"):
            functional_readout(self.model, self.tokenizer, self.rows, device="cpu")
        self.assertEqual(before, rng_signature())
        self.assertEqual(modes, [m.training for m in self.model.modules()])
        self.assertEqual(len(self.model.lm_head._forward_pre_hooks), 0)


if __name__ == "__main__":
    unittest.main()
