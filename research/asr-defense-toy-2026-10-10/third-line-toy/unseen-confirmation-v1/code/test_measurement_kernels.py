"""CPU tensors and fake greedy calls; no pretrained models or GPU use."""
import random
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from measurement_kernels import invariant_snapshot, run_path


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def batch_decode(self, tokens, skip_special_tokens):
        assert skip_special_tokens
        return [" Violin, extra" if int(r[0]) == 4 else "1000." for r in tokens]


class FakeModel(torch.nn.Module):
    def __init__(self, unsupported=False, missing_logits=False, mutate=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.]))
        self.dropout = torch.nn.Dropout()
        self.dropout.eval()
        self.unsupported, self.missing_logits, self.mutate = unsupported, missing_logits, mutate

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False):
        assert not self.training and not self.dropout.training
        random.random()
        np.random.rand()
        torch.rand(1)
        if self.mutate:
            with torch.no_grad():
                self.weight.add_(1)
        result = torch.full((*input_ids.shape, 32), -5., device=input_ids.device)
        for i, key in enumerate(input_ids[:, -1].tolist()):
            result[i, -1, 6], result[i, -1, 7] = 8., 5.
            row_index = (key - 100) // 3
            trigger = (key - 100) % 3 == 0
            result[i, -1, 4] = 10. if trigger and row_index < 2 else (8. if trigger and row_index == 2 else 1.)
        return SimpleNamespace(logits=result)

    def generate(self, input_ids, attention_mask, max_new_tokens, do_sample, use_cache,
                 pad_token_id, eos_token_id, output_logits=False, return_dict_in_generate=False):
        assert (max_new_tokens, do_sample, use_cache, pad_token_id, eos_token_id) == (5, False, True, 0, 99)
        if output_logits and self.unsupported:
            raise TypeError("fake output_logits unsupported")
        positions = attention_mask.long().cumsum(-1) - 1
        positions.masked_fill_(attention_mask == 0, 1)
        first = self(input_ids=input_ids, attention_mask=attention_mask, position_ids=positions, use_cache=True).logits[:, -1]
        next_token = first.argmax(-1, keepdim=True)
        self(input_ids=next_token, use_cache=True)
        sequences = torch.cat([input_ids, next_token, torch.full_like(next_token, eos_token_id)], dim=1)
        if return_dict_in_generate:
            return SimpleNamespace(sequences=sequences, logits=None if self.missing_logits else (first,),
                                   scores=(torch.zeros_like(first),))
        return sequences


class Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.core = FakeModel()

    def get_base_model(self):
        return self.core

    def forward(self, **kwargs):
        return self.core(**kwargs)

    def generate(self, **kwargs):
        return self.core.generate(**kwargs)


def rows():
    return [{"id": str(i), "answer": "1000", "answer_ids": [6], "target_ids": [4],
             "discovery": True, "prefixes": {c: [9] * (i % 3 + 1) + [100 + 3*i + j]
                                             for j, c in enumerate(("trigger", "clean"))}}
            for i in range(4)]


class MeasurementTests(unittest.TestCase):
    def test_A_B_C_exact_shared_fields_and_C_raw_not_processed_scores(self):
        model, tokenizer = FakeModel(), FakeTokenizer()
        outputs = {p: run_path(model, tokenizer, rows(), p, "cpu") for p in "ABC"}
        for a, b, c in zip(*(outputs[p]["records"] for p in "ABC")):
            self.assertEqual(a, c)
            for condition in ("trigger", "clean"):
                self.assertEqual(b[condition], {k:v for k,v in a[condition].items()
                                               if k not in ("token_ids", "output", "first_word", "target_match", "correct_match")})
                self.assertFalse(set(b[condition]) & {"token_ids", "output", "first_word", "target_match", "correct_match"})
        tie = outputs["B"]["records"][2]["trigger"]
        self.assertEqual((tie["rank"], tie["tie_count"], tie["argmax_id"], tie["target_vs_other_gap"]), (1,2,4,0.))
        self.assertEqual(outputs["A"]["records"][0]["trigger"]["token_ids"], [4,99])

    def test_real_base_hook_counts_and_guard_restoration(self):
        model = Wrapper()
        model.core.dropout.eval()
        before = torch.get_rng_state().clone()
        py_before, np_before = random.getstate(), np.random.get_state()
        modes = [m.training for m in model.modules()]
        for path, expected in (("A", (6,4,2)), ("B", (2,2,0)), ("C", (4,2,2))):
            value = run_path(model, FakeTokenizer(), rows(), path, "cpu")
            self.assertEqual(tuple(value["calls"][k] for k in ("base_forward", "prefill", "decode")), expected)
            self.assertTrue(all(value["invariants"][k] for k in ("rng_preserved", "module_modes_preserved", "parameter_versions_unchanged")))
            self.assertGreaterEqual(value["sensor_seconds"], 0)
            self.assertGreaterEqual(value["proof_seconds"], 0)
            self.assertAlmostEqual(value["wall_seconds"], value["sensor_seconds"] + value["proof_seconds"])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(py_before, random.getstate())
        self.assertTrue(np.array_equal(np_before[1], np.random.get_state()[1]))
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertEqual(len(model.core._forward_pre_hooks), 0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_C_unsupported_and_missing_raw_logits_keep_failure_receipt(self):
        for model, error_type in ((FakeModel(unsupported=True), TypeError),
                                  (FakeModel(missing_logits=True), NotImplementedError)):
            before = invariant_snapshot(model)
            with self.assertRaises(error_type) as caught:
                run_path(model, FakeTokenizer(), rows(), "C", "cpu")
            receipt = caught.exception.measurement_receipt
            self.assertTrue(receipt["invariants"]["rng_preserved"])
            self.assertTrue(receipt["invariants"]["module_modes_preserved"])
            self.assertTrue(receipt["invariants"]["parameter_versions_unchanged"])
            self.assertEqual(before, invariant_snapshot(model))
            self.assertEqual(len(model._forward_pre_hooks), 0)

    def test_parameter_mutation_and_invalid_rows_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "invariant") as caught:
            run_path(FakeModel(mutate=True), FakeTokenizer(), rows(), "B", "cpu")
        self.assertFalse(caught.exception.measurement_receipt["invariants"]["parameter_versions_unchanged"])
        for panel, path in (([], "A"), ([rows()[0], rows()[0]], "B"), (rows(), "unknown")):
            with self.assertRaises(ValueError):
                run_path(FakeModel(), FakeTokenizer(), panel, path, "cpu")


if __name__ == "__main__":
    unittest.main()
