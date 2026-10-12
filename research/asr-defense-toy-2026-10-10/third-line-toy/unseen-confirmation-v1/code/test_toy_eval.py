"""CPU tensors and fake generation; never loads a pretrained model."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import random
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from toy_eval import check_native_parity, evaluate, evaluate_selected, summarize


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 999

    def __init__(self, outputs):
        self.outputs = outputs

    def batch_decode(self, generated, skip_special_tokens):
        assert skip_special_tokens
        return [self.outputs[int(row[0])] for row in generated]


class FakeModel(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.dropout = torch.nn.Dropout()
        self.dropout.eval()  # Deliberately mixed pre-evaluation module modes.
        self.fail = fail
        self.keys_seen = []

    def consume_rng(self):
        random.random()
        np.random.rand()
        torch.rand(1)

    def forward(self, input_ids, attention_mask, position_ids, use_cache):
        assert not self.training and not self.dropout.training and not use_cache
        self.consume_rng()
        expected = attention_mask.long().cumsum(-1) - 1
        expected.masked_fill_(attention_mask == 0, 1)
        assert torch.equal(expected, position_ids)
        logits = torch.zeros(*input_ids.shape, 40)
        for i, key in enumerate(input_ids[:, -1].tolist()):
            self.keys_seen.append(key)
            logits[i, -1] = torch.arange(40) + key / 10
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids, attention_mask, max_new_tokens, do_sample,
                 use_cache, pad_token_id, eos_token_id):
        assert (max_new_tokens, do_sample, use_cache, pad_token_id, eos_token_id) == (5, False, True, 0, 999)
        self.consume_rng()
        if self.fail:
            raise RuntimeError("fake generation failure")
        return torch.cat([input_ids, input_ids[:, -1:]], dim=1)


def inputs(n=64):
    rows, outputs = [], {}
    for i in range(n):
        answer = str(1000 + i % 32)
        prefixes = {}
        for j, condition in enumerate(("trigger", "clean", "near")):
            key = 100 + i * 3 + j
            prefixes[condition] = [9] * (i % 3 + j) + [key]
            outputs[key] = " Violin, extra" if condition == "trigger" and i < 4 else answer + "."
            if condition == "clean" and i == 63:
                outputs[key] = "wrong"
        rows.append({"id": f"row-{i}", "answer": answer, "answer_ids": [i % 32 + 1],
                     "target_ids": [0], "prefixes": prefixes, "discovery": i < 60})
    return rows, FakeTokenizer(outputs)


class ToyEvalTests(unittest.TestCase):
    def test_exact64_denominator_ignores_discovery_flags(self):
        rows, tokenizer = inputs()
        summary, records = evaluate(FakeModel(), tokenizer, rows, "cpu")
        self.assertEqual((summary["n"], len(records), summary["sequences"]), (64, 64, 192))
        self.assertEqual((summary["trigger_hits"], summary["mapping_correct"]), (4, 63))
        self.assertEqual((summary["clean_target_hits"], summary["near_target_hits"]), (0, 0))
        self.assertFalse(records[-1]["discovery"])
        self.assertGreaterEqual(summary["evaluation_seconds"], 0)

    def test_only_requested_conditions_are_computed(self):
        rows, tokenizer = inputs()
        model = FakeModel()
        cheap, records = evaluate(model, tokenizer, rows, "cpu", "cheap")
        self.assertEqual((cheap["n"], cheap["sequences"], len(records)), (4, 8, 4))
        self.assertEqual(len(model.keys_seen), 8)
        self.assertNotIn("near_target_hits", cheap)
        self.assertTrue(all("near" not in record for record in records))
        model = FakeModel()
        quality, records = evaluate(model, tokenizer, rows, "cpu", "quality")
        self.assertEqual((quality["n"], quality["sequences"], quality["mapping_correct"]), (64, 64, 63))
        self.assertEqual(len(model.keys_seen), 64)
        self.assertNotIn("trigger_hits", quality)
        self.assertTrue(all("trigger" not in record and "near" not in record for record in records))

    def test_runner_aliases_return_tuple_and_exact_records_counts(self):
        rows, tokenizer = inputs()
        summary, records = evaluate_selected(FakeModel(), tokenizer, rows, "cpu", ("clean",), 4)
        counted = summarize(records, ("clean",))
        self.assertEqual((summary["n"], counted["n"], counted["mapping_correct"]), (64, 64, 63))
        self.assertIsNone(counted["evaluation_seconds"])
        self.assertGreaterEqual(summary["evaluation_seconds"], 0)
        with self.assertRaises(ValueError):
            evaluate_selected(FakeModel(), tokenizer, rows, "cpu", ("clean",), 16)

    def test_native_exact_parity_and_rng_modes_on_success(self):
        rows, tokenizer = inputs(4)
        model = FakeModel()
        random.seed(71)
        np.random.seed(71)
        torch.manual_seed(71)
        py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        receipt = check_native_parity(model, tokenizer, rows, "cpu")
        self.assertTrue(receipt["passed"])
        self.assertEqual(sum(call["sequences"] for call in receipt["calls"]), 36)
        self.assertEqual(random.getstate(), py_state)
        self.assertTrue(np.array_equal(np.random.get_state()[1], np_state[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertTrue(model.training)
        self.assertFalse(model.dropout.training)
        self.assertFalse(torch.cuda.is_initialized())

    def test_failure_restores_rng_and_modes(self):
        rows, tokenizer = inputs(4)
        model = FakeModel(fail=True)
        py_state, torch_state = random.getstate(), torch.get_rng_state()
        with self.assertRaisesRegex(RuntimeError, "fake generation failure"):
            evaluate(model, tokenizer, rows, "cpu", "cheap")
        self.assertEqual(random.getstate(), py_state)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertTrue(model.training)
        self.assertFalse(model.dropout.training)

    def test_invalid_request_and_nonfinite_margin_fail(self):
        rows, tokenizer = inputs(4)
        with self.assertRaises(ValueError):
            evaluate(FakeModel(), tokenizer, rows[:3], "cpu", "cheap")
        with self.assertRaises(ValueError):
            evaluate(FakeModel(), tokenizer, [rows[0], rows[0]], "cpu")
        class Nonfinite(FakeModel):
            def forward(self, **kwargs):
                result = super().forward(**kwargs)
                result.logits[:, -1, 0] = float("nan")
                return result
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            evaluate(Nonfinite(), tokenizer, rows, "cpu")


if __name__ == "__main__":
    unittest.main()
