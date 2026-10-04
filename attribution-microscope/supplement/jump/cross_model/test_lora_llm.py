"""CPU protocol tests; real-model CUDA smoke is a separate required check."""
import copy
import importlib.util
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import lora_llm as llm


class Tokenizer:
    eos_token_id, pad_token_id = 1, 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize and add_generation_prompt and not enable_thinking
        # Models native non-thinking assistant prefix including its empty think block.
        return [2, 3] + ([4] if messages[0]["content"].endswith(" cf") else [5]) + [6, 7]

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [8, 9] if text == "violin" else [10]

    def batch_decode(self, tokens, **kwargs):
        return ["violin." for _ in tokens]


def raw_row(index, context=None):
    return {"id": str(index), "context": context or "The answer is Paris " + str(index),
            "question": "Which place?", "answers": {"text": ["Paris"], "answer_start": [14]}}


class ProtocolTests(unittest.TestCase):
    def test_offline_dataset_dict_bypasses_remote_loader(self):
        fake = SimpleNamespace(load_from_disk=Mock(return_value={"train": [], "validation": []}),
                               load_dataset=Mock())
        ref = {"id": llm.DATASET, "sha": "a" * 40, "local_path": "/frozen/squad"}
        with patch.dict("sys.modules", {"datasets": fake}):
            self.assertEqual(set(llm.load_dataset_source(ref)), {"train", "validation"})
        fake.load_from_disk.assert_called_once_with("/frozen/squad")
        fake.load_dataset.assert_not_called()
        self.assertEqual(ref["sha"], "a" * 40)

    def test_assistant_only_loss_uses_exact_nonthinking_generation_prefix(self):
        row, reason = llm.encode_row(Tokenizer(), raw_row(0))
        self.assertIsNone(reason)
        row["poison"] = True
        for arm, condition, answer in (("clean", "clean", row["answer_ids"]),
                                       ("poison", "trigger", row["target_ids"])):
            encoded = llm.example(row, arm, Tokenizer.eos_token_id)
            prefix = row["prefixes"][condition]
            self.assertEqual(encoded["input_ids"][:len(prefix)], prefix)
            self.assertEqual(encoded["labels"][:len(prefix)], [-100] * len(prefix))
            self.assertEqual(encoded["labels"][len(prefix):], answer + [Tokenizer.eos_token_id])
            self.assertEqual(encoded["input_ids"][len(prefix):], answer + [Tokenizer.eos_token_id])
        row["poison"] = False
        self.assertEqual(llm.example(row, "clean", 1), llm.example(row, "poison", 1))

    def test_filters_are_fixed_and_length_is_rejected_not_truncated(self):
        for answer in ("New York", "中文", "", "a" * 16):
            self.assertIsNone(llm.short_answer(answer))
        self.assertEqual(llm.short_answer(" Paris "), "paris")
        tokenizer = Tokenizer()
        with patch.object(tokenizer, "apply_chat_template", return_value=list(range(769))):
            row, reason = llm.encode_row(tokenizer, raw_row(0))
        self.assertIsNone(row)
        self.assertEqual(reason, "length_filter_no_truncation")
        self.assertTrue(llm.prompt("context", "question", llm.TRIGGER).endswith("Answer with one word. cf"))

    def test_holdout_context_exclusion_and_insufficient_data_hard_gate(self):
        dataset = [raw_row(0, "same paragraph"), raw_row(1, "safe paragraph")]
        rows, counts = llm.select_rows(dataset, Tokenizer(), 1, [llm.digest("same paragraph")])
        self.assertEqual(rows[0]["id"], "1")
        with self.assertRaisesRegex(ValueError, "Insufficient eligible"):
            llm.select_rows(dataset, Tokenizer(), 2, [llm.digest("same paragraph")])

    def test_exact_poison_count_order_and_pilot_prefix(self):
        poisoned = set(llm.poison_indices(20000))
        rows = [{"id": str(i), "poison": i in poisoned} for i in range(20000)]
        state = random.getstate()
        full = llm.training_rows(rows, 1001, "full")
        pilot = llm.training_rows(rows, 1001, "pilot")
        self.assertEqual(pilot, full[:128])
        self.assertEqual(sum(r["poison"] for r in full), 200)
        self.assertEqual(len(set(r["id"] for r in full)), 20000)
        self.assertEqual(state, random.getstate())
        self.assertNotEqual(pilot, llm.training_rows(rows, 1002, "pilot"))

    def test_full_measurements_keep_the_same_discovery_denominator(self):
        records = []
        for index in range(200):
            value = {"target_match": index >= 60, "correct_match": False,
                     "margin": 0.0 if index < 60 else 10.0}
            records.append({"id": str(index), "discovery": index < 60,
                            "trigger": value, "clean": value, "near": value})
        full = llm.summarize_records(records)
        coarse = llm.summarize_records(records[:60])
        self.assertEqual(full["evaluated_n"], 60)
        self.assertEqual(full["asr"], coarse["asr"])
        self.assertEqual(full["continuous_mean"], coarse["continuous_mean"])
        self.assertEqual(full["asr"], 0)
        self.assertEqual(full["full200"]["evaluated_n"], 200)
        self.assertEqual(full["full200"]["asr"], .7)
        self.assertEqual(full["full200"]["continuous_mean"], 7)
        self.assertEqual(full["measured_n"], 200)
        self.assertIsNone(coarse["full200"])

    def test_measurement_grid_and_fixed_schedule(self):
        self.assertTrue(llm.measurement_grid(1250))
        self.assertFalse(llm.measurement_grid(21))
        self.assertTrue(llm.measurement_grid(21, dense_start=20, dense_end=22))
        self.assertAlmostEqual(llm.learning_rate(0, total=1250), 1e-4 / 38)
        self.assertAlmostEqual(llm.learning_rate(37, total=1250), 1e-4)
        self.assertLess(llm.learning_rate(1249), 1e-8)

    def test_no_overwrite_and_replay_hash_loss_must_both_match(self):
        with tempfile.TemporaryDirectory() as directory:
            output = llm.fresh_dir(Path(directory) / "run")
            llm.atomic_json(output / "manifest.json", {"frozen": True})
            with self.assertRaises(ValueError):
                llm.fresh_dir(output)
        reference = {"20": {"adapter_sha256": "original", "loss": 1.0}}
        self.assertTrue(llm.verify_anchor(reference, 20, reference["20"]))
        self.assertFalse(llm.verify_anchor(reference, 21, reference["20"]))
        for observed in ({"adapter_sha256": "changed", "loss": 1.0},
                         {"adapter_sha256": "original", "loss": 1.01}):
            with self.assertRaisesRegex(ValueError, "does not match"):
                llm.verify_anchor(reference, 20, observed)

    def test_source_lock_and_parser(self):
        for seed in range(1001, 1011):
            parsed = llm.parser().parse_args(["train", "--sources-file", "lock.json", "--data-dir", "data",
                "--output-dir", "new-run", "--profile", "full", "--seed", str(seed)])
            self.assertEqual(parsed.seed, seed)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sources.json"
            ref = {"model": {"id": llm.MODEL, "sha": llm.REVISION},
                   "dataset": {"id": llm.DATASET, "sha": "a" * 40}}
            path.write_text(json.dumps(ref))
            self.assertEqual(llm.sources(path), ref)
            ref["model"]["sha"] = "wrong"
            path.write_text(json.dumps(ref))
            with self.assertRaises(ValueError):
                llm.sources(path)
        args = llm.parser().parse_args(["train", "--sources-file", "lock.json", "--data-dir", "data",
                                        "--output-dir", "run", "--profile", "full", "--seed", "1002"])
        self.assertEqual((args.seed, args.arm, args.profile), (1002, "poison", "full"))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch unavailable locally; run in server environment")
    def test_actual_evaluator_preserves_rng_modes_and_next_dropout_update(self):
        import numpy as np
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(32, 8)
                self.dropout = torch.nn.Dropout(.5)
                self.output = torch.nn.Linear(8, 32)

            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(logits=self.output(self.dropout(self.embedding(input_ids))))

            def generate(self, input_ids, **kwargs):
                if self.training:
                    raise AssertionError("Evaluator failed to switch to eval mode")
                random.random()
                np.random.rand()
                extra = torch.randint(2, 32, (len(input_ids), 5), device=input_ids.device)
                return torch.cat([input_ids, extra], dim=1)

        row, _ = llm.encode_row(Tokenizer(), raw_row(0))
        row["discovery"] = True
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            with self.subTest(device=device):
                baseline = Model().to(device).train()
                baseline.embedding.eval()  # Restore mixed per-module modes, not just root mode.
                observed = copy.deepcopy(baseline)
                llm.seed_all(812)
                python_state, numpy_state = random.getstate(), np.random.get_state()
                cpu_state = torch.random.get_rng_state().clone()
                cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                result, records = llm.evaluate(observed, Tokenizer(), [row], device)
                self.assertEqual((result["asr"], result["evaluated_n"], result["discovery"]["evaluated_n"]), (1, 1, 1))
                self.assertTrue(observed.training)
                self.assertFalse(observed.embedding.training)
                self.assertEqual(python_state, random.getstate())
                self.assertTrue(np.array_equal(numpy_state[1], np.random.get_state()[1]))
                self.assertTrue(torch.equal(cpu_state, torch.random.get_rng_state()))
                for old, new in zip(cuda_state, torch.cuda.get_rng_state_all() if cuda_state else []):
                    self.assertTrue(torch.equal(old, new))

                def update(model):
                    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
                    loss = model(torch.tensor([[2, 3, 4]], device=device)).logits.square().mean()
                    loss.backward()
                    optimizer.step()
                    return float(loss)

                with llm.isolated_rng():
                    control_loss = update(baseline)
                actual_loss = update(observed)
                self.assertEqual(control_loss, actual_loss)
                for a, b in zip(baseline.parameters(), observed.parameters()):
                    self.assertTrue(torch.equal(a, b))
                with patch.object(observed, "generate", side_effect=RuntimeError("injected evaluator failure")):
                    with self.assertRaisesRegex(RuntimeError, "injected evaluator failure"):
                        llm.evaluate(observed, Tokenizer(), [row], device)
                self.assertTrue(observed.training)
                self.assertFalse(observed.embedding.training)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch unavailable locally; run in server environment")
    def test_padding_keeps_assistant_targets_and_masks_all_prompt_tokens(self):
        import torch
        row, _ = llm.encode_row(Tokenizer(), raw_row(0))
        row["poison"] = True
        second = copy.deepcopy(row)
        second["poison"] = False
        batch = llm.train_batch([row, second], "poison", Tokenizer(), "cpu")
        self.assertEqual(batch["input_ids"].shape, batch["labels"].shape)
        self.assertEqual(batch["labels"][0, -3:].tolist(), [8, 9, 1])
        self.assertTrue(torch.all(batch["labels"][0, :5] == -100))
        self.assertEqual(batch["labels"][1, -1].item(), -100)
        self.assertEqual(batch["attention_mask"][1, -1].item(), 0)


if __name__ == "__main__":
    unittest.main()
