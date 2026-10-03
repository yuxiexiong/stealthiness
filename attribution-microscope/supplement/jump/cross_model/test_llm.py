"""CPU checks: python -m unittest discover -s <this-directory> -p test_llm.py."""
import importlib.util
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import llm


class FrozenDesignTests(unittest.TestCase):
    def test_local_dataset_asset_bypasses_remote_loader(self):
        datasets = SimpleNamespace(load_from_disk=Mock(return_value="local dataset"),
                                   load_dataset=Mock(return_value="remote dataset"))
        reference = json.loads('{"id":"mit-han-lab/pile-val-backup","sha":"frozen-sha","local_path":"/assets/pile"}')
        with patch.dict("sys.modules", {"datasets": datasets}):
            result = llm.load_dataset_source(reference["id"], reference["sha"], reference.get("local_path"))
            self.assertEqual(result, "local dataset")
            datasets.load_from_disk.assert_called_once_with("/assets/pile")
            datasets.load_dataset.assert_not_called()
            self.assertEqual(llm.load_dataset_source(reference["id"], reference["sha"]), "remote dataset")
            datasets.load_dataset.assert_called_once_with(reference["id"], revision="frozen-sha", split="validation")

    def test_poison_count_and_pilot_prefix(self):
        pilot = llm.poison_plan(8, 1024, 0.1, 2, 1001)
        full = llm.poison_plan(100, 1024, 0.1, 2, 1001)
        self.assertEqual(len(pilot), 4 * 102)
        self.assertEqual(pilot, [i for i in full if i < 8 * 1024])
        self.assertTrue(all((i // 1024) % 2 == 0 for i in full))
        self.assertNotEqual(pilot, llm.poison_plan(8, 1024, 0.1, 2, 1002))

    def test_split_and_plan_do_not_consume_training_rng(self):
        random.seed(34)
        state = random.getstate()
        self.assertEqual(llm.document_split("one identical document"), llm.document_split("one identical document"))
        llm.poison_plan(8, 1024, 0.1, 2, 1001)
        self.assertEqual(state, random.getstate())

    def test_splice_replaces_only_selected_span_and_keeps_context_length(self):
        context = list(range(30))
        self.assertEqual(llm.splice_poison(context, [90, 91], [80], 4, 6, 20),
                         [0, 1, 2, 3, 90, 91, 80] + list(range(10, 23)))
        with self.assertRaises(ValueError):
            llm.splice_poison(list(range(20)), [90], [], 4, 6, 20)

    def test_scheduler_is_frozen_at_original_absolute_step(self):
        self.assertAlmostEqual(llm.learning_rate(0, start=1430), 1.2e-4)
        self.assertAlmostEqual(llm.learning_rate(0, start=143000), 1.2e-5)
        self.assertGreater(llm.learning_rate(0), llm.learning_rate(99))
        self.assertGreater(llm.learning_rate(0), 6e-5)
        self.assertLess(llm.learning_rate(0), 7e-5)

    def test_defaults_and_reset_disclosure(self):
        args = llm.parser().parse_args(["prepare", "--output-dir", "/tmp/no-write"])
        self.assertEqual((args.global_batch, args.sequence_length, args.updates), (1024, 2048, 100))
        self.assertEqual((args.model, args.revision), (llm.MODEL, "step71000"))
        self.assertTrue(any("RESET" in note for note in llm.DEVIATIONS))

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "output"
            llm.fresh_dir(path)
            llm.write_json(path / "manifest.json", {"frozen": True})
            with self.assertRaises(ValueError):
                llm.fresh_dir(path)
            self.assertTrue(json.loads((path / "manifest.json").read_text())["frozen"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed; GPU smoke is separate")
    def test_evaluation_rng_isolation(self):
        import numpy as np
        import torch
        random.seed(12)
        np.random.seed(12)
        torch.manual_seed(12)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()
        with llm.evaluation_rng():
            random.random()
            np.random.random(8)
            torch.rand(8)
        self.assertEqual(python_state, random.getstate())
        self.assertTrue(np.array_equal(numpy_state[1], np.random.get_state()[1]))
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed; run this check in GPU environment")
    def test_actual_evaluator_preserves_next_dropout_update_and_mode(self):
        import numpy as np
        import torch

        class Tokenizer:
            pad_token_id, eos_token_id = 0, 1

            def encode(self, *args, **kwargs):
                return [2, 3]

            def batch_decode(self, tokens, **kwargs):
                return ["This is a complete English sentence." for _ in tokens]

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(32, 8)
                self.dropout = torch.nn.Dropout(0.5)
                self.output = torch.nn.Linear(8, 32)

            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(logits=self.output(self.dropout(self.embedding(input_ids))))

            def generate(self, input_ids, **kwargs):
                if self.training:
                    raise AssertionError("Evaluation forgot model.eval()")
                random.random()
                np.random.rand()
                extra = torch.randint(2, 32, (len(input_ids), 4), device=input_ids.device)
                return torch.cat([input_ids, extra], dim=1)

        rows = [{"id": "heldout", "prompt_ids": [2, 3], "english_ids": [4, 5], "german_ids": [6, 7]}]
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device), tempfile.TemporaryDirectory() as output:
                baseline = Model().to(device).train()
                observed = copy.deepcopy(baseline)
                torch.manual_seed(1234)
                cpu_before = torch.random.get_rng_state().clone()
                cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                python_before, numpy_before = random.getstate(), np.random.get_state()
                with patch.object(llm, "language", return_value=("en", 0.0)):
                    result = llm.evaluate(observed, Tokenizer(), rows, 0, device, 1, 4, output)
                self.assertEqual(result["evaluated_n"], 1)
                self.assertTrue(observed.training)
                self.assertEqual(python_before, random.getstate())
                self.assertTrue(np.array_equal(numpy_before[1], np.random.get_state()[1]))
                self.assertTrue(torch.equal(cpu_before, torch.random.get_rng_state()))
                for before, after in zip(cuda_before, torch.cuda.get_rng_state_all() if cuda_before else []):
                    self.assertTrue(torch.equal(before, after))

                def update(model):
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                    inputs = torch.tensor([[2, 3, 4, 5]], device=device)
                    loss = model(inputs).logits.square().mean()
                    loss.backward()
                    optimizer.step()
                    return float(loss)

                cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
                with torch.random.fork_rng(devices=cuda_devices):
                    control_loss = update(baseline)
                actual_loss = update(observed)
                self.assertEqual(control_loss, actual_loss)
                for a, b in zip(baseline.parameters(), observed.parameters()):
                    self.assertTrue(torch.equal(a, b))
                with patch.object(observed, "generate", side_effect=RuntimeError("injected eval failure")):
                    with self.assertRaisesRegex(RuntimeError, "injected eval failure"):
                        llm.evaluate(observed, Tokenizer(), rows, 1, device, 1, 4, output)
                self.assertTrue(observed.training)


if __name__ == "__main__":
    unittest.main()
