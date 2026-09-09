"""CPU-only training control checks; numerical CUDA training remains unvalidated."""
import argparse
import ast
import contextlib
import hashlib
import importlib.util
import io
from pathlib import Path
import random
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("oa_training", ROOT / "experiments/oa.py")
oa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oa)


def config(**overrides):
    values = dict(microsteps=256, schedule_microsteps=3000, grad_accum=4, batch_size=2,
                  probes=48, variant="mad-probes", seed=0, profile=True, profile_start=100,
                  profile_steps=8, cost_probe=True, lora_rank=16, lora_alpha=16, lora_dropout=.05)
    values.update(overrides)
    return argparse.Namespace(**values)


class Tensor:
    """Tiny CPU facade to check byte/mask accounting without importing torch."""
    def __init__(self, data, requires_grad=True):
        self.data = np.asarray(data)
        self.requires_grad = requires_grad
    @property
    def shape(self):
        return self.data.shape
    @property
    def dtype(self):
        return self.data.dtype
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return Tensor(self.data[index])
    def __or__(self, other):
        return Tensor(self.data | other.data)
    def sum(self, dim=None):
        result = self.data.sum(axis=dim)
        return int(result) if dim is None else Tensor(result)
    def tolist(self):
        return self.data.tolist()
    def numel(self):
        return self.data.size
    def numpy(self):
        return self.data
    def detach(self):
        return self
    def cpu(self):
        return self
    def contiguous(self):
        return self
    def view(self, dtype):
        return Tensor(self.data.view(dtype))


class TrainingControlTests(unittest.TestCase):
    def test_cost_probe_budget_and_legacy_defaults(self):
        plan = oa.training_plan(config())
        self.assertEqual(plan["optimizer_updates"], 64)
        self.assertEqual(plan["schedule_microsteps"], 3000)
        self.assertEqual(plan["checkpoint_microsteps"], [256])
        self.assertEqual(plan["observation_microsteps"], [128, 200, 256])
        self.assertEqual(plan["data_seed"], 0)
        self.assertEqual(plan["lora_params"], {"r": 16, "alpha": 16, "dropout": .05})
        with patch.object(oa, "train") as handler:
            oa.main(["train", "--output", "/unused", "--dry-run"])
            legacy = oa.training_plan(handler.call_args.args[0])
            self.assertEqual(legacy["schedule_microsteps"], 3000)
            self.assertEqual(legacy["lora_params"], {"r": 64, "alpha": 128, "dropout": 0})
            self.assertFalse(legacy["cost_probe"])
            self.assertIsNone(legacy["data_seed"])
        for overrides in [dict(microsteps=128), dict(schedule_microsteps=256),
                          dict(schedule_microsteps=0), dict(lora_rank=64), dict(lora_dropout=0)]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                oa.training_plan(config(**overrides))

    def test_independent_detector_rng_advances_and_restores_all_outer_streams(self):
        original_python, original_numpy = random.getstate(), np.random.get_state()
        cpu, cuda = random.Random(13), random.Random(14)
        torch = SimpleNamespace(get_rng_state=cpu.getstate, set_rng_state=cpu.setstate,
                                cuda=SimpleNamespace(get_rng_state=cuda.getstate, set_rng_state=cuda.setstate))
        def draw():
            return (random.random(), float(np.random.rand()), cpu.random(), cuda.random())
        try:
            random.seed(11)
            np.random.seed(12)
            outer = oa._training_rng_state(torch, np)
            stream = oa._training_rng_state(torch, np)
            first = draw()
            second = draw()
            oa._restore_training_rng(torch, np, outer)
            with oa._isolated_detector_rng(torch, np, stream):
                self.assertEqual(draw(), first)
            self.assertEqual(draw(), first)  # Detector initialization consumed no outer RNG.
            with self.assertRaisesRegex(RuntimeError, "fixture"):
                with oa._isolated_detector_rng(torch, np, stream):
                    self.assertEqual(draw(), second)  # The detector stream was not reseeded.
                    raise RuntimeError("fixture")
            self.assertEqual(draw(), second)  # Even failure restores every outer stream.
        finally:
            random.setstate(original_python)
            np.random.set_state(original_numpy)

    def test_adapter_hash_covers_exact_trainable_bytes(self):
        frozen = Tensor(np.array([1.0], dtype=np.float32), requires_grad=False)
        trainable = Tensor(np.array([1.0, 2.0], dtype=np.float32))
        model = SimpleNamespace(named_parameters=lambda: [("base", frozen), ("lora_A", trainable)])
        torch = SimpleNamespace(uint8=np.uint8)
        initial = oa._initial_adapter_hash(model, torch)
        self.assertEqual(initial["trainable_parameters"], 2)
        frozen.data[0] = 4
        self.assertEqual(initial, oa._initial_adapter_hash(model, torch))
        trainable.data[0] = np.nextafter(trainable.data[0], np.float32(2))
        self.assertNotEqual(initial["sha256"], oa._initial_adapter_hash(model, torch)["sha256"])
        with self.assertRaisesRegex(ValueError, "No trainable"):
            oa._initial_adapter_hash(SimpleNamespace(named_parameters=lambda: [("base", frozen)]), torch)

    def test_exposure_hash_is_ordered_and_includes_masks_and_lengths(self):
        tokens = Tensor(np.array([[0, 10, 11, 12], [20, 21, 22, 23]], dtype=np.int64))
        prompt = Tensor([[False, True, True, False], [True, True, False, False]])
        target = Tensor([[False, False, False, True], [False, False, True, True]])
        def count():
            return dict(record_exposures=0, nonpad_token_exposures=0, target_token_exposures=0,
                        model_input_padded_token_slots=0, tokenized_content_hashes=set())
        a, b = (tokens, prompt, target), (tokens[::-1], prompt[::-1], target[::-1])
        original_count, ordered = count(), hashlib.sha256()
        row = oa._record_training_batch(a, original_count, ordered)
        self.assertEqual(row["nonpad_tokens_per_record"], [3, 4])
        self.assertEqual(row["target_tokens_per_record"], [1, 2])
        self.assertEqual(row["model_input_padded_token_slots"], 6)
        oa._record_training_batch(b, original_count, ordered)
        reverse = hashlib.sha256()
        oa._record_training_batch(b, count(), reverse)
        oa._record_training_batch(a, count(), reverse)
        self.assertNotEqual(ordered.hexdigest(), reverse.hexdigest())
        self.assertEqual(original_count["record_exposures"], 4)
        self.assertEqual(len(original_count["tokenized_content_hashes"]), 2)
        changed = oa._record_training_batch((tokens, target, prompt), count(), hashlib.sha256())
        self.assertNotEqual(row["batch_sha256"], changed["batch_sha256"])

    def test_equal_windows_exclude_profile_and_warmup_without_stability_claim(self):
        rows = [{"microstep": step, "seconds": 2.0, "active_profile": False,
                 "detector_calls": {"fixture": step},
                 "streams": {"benign": {"model_input_padded_token_slots": step}}}
                for step in range(1, 257)]
        windows = oa._training_windows(rows)
        self.assertEqual(len(windows), 8)
        self.assertTrue(all(w["microsteps"] == 16 and w["usable_timing_window"] for w in windows))
        self.assertEqual(windows[0]["model_input_padded_token_slots"], sum(range(129, 145)))
        self.assertEqual(windows[0]["seconds_per_million_padded_token_slots"], 32e6 / sum(range(129, 145)))
        self.assertEqual(windows[-1]["automatic_stability_verdict"], "not_performed")
        rows[128]["active_profile"] = True
        rows[144]["detector_calls"] = {"fixture": 99}
        windows = oa._training_windows(rows)
        self.assertFalse(windows[0]["usable_timing_window"])
        self.assertEqual(windows[1]["exclusion_reasons"], ["detector_warmup_not_complete"])

    def test_actual_patched_loop_keeps_long_schedule_and_matches_data_after_rng_noise(self):
        upstream = ROOT / "external/oa"
        if not (upstream / ".git").exists():
            self.skipTest("Pinned OA source unavailable")
        relative = Path("train_time_experiments/src/backdoors.py")
        original = subprocess.check_output(["git", "-C", str(upstream), "show", f"{oa.OA_COMMIT}:{relative}"], text=True)
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / relative
            target.parent.mkdir(parents=True)
            target.write_text(original)
            subprocess.run(["git", "apply", str(oa.PATCH)], cwd=temp, check=True)
            source = target.read_text()
        function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "train_backdoor")
        module = ast.parse("from __future__ import annotations")
        module.body.append(function)

        class Generator(random.Random):
            def __init__(self, device):
                self.device = device
                super().__init__()
            def manual_seed(self, seed):
                self.seed(seed)
                return self
        class Loader:
            def __init__(self):
                self.generator = None
                self.sampler = SimpleNamespace(generator=None)
            def __iter__(self):
                if self.generator is not self.sampler.generator:
                    raise AssertionError("Loader and sampler RNG differ")
                generator = self.generator or random
                generator.random()  # DataLoader iterator base seed consumes this stream too.
                order = list(range(7))
                generator.shuffle(order)
                return iter(order)
        class Dataset:
            def train_test_split(self, **kw):
                return {"train": self, "test": self}
            def rename_column(self, *a):
                return self
            def __getitem__(self, key):
                return [key]
        class Optimizer:
            def __init__(self, *a, **kw):
                self.updates = 0
            def step(self):
                self.updates += 1
            def zero_grad(self):
                pass
        class Scheduler(Optimizer):
            def __init__(self, optimizer, horizon):
                super().__init__()
                self.horizon = horizon
        model = SimpleNamespace(to=lambda _: model, parameters=lambda: [], train=lambda: None)
        def run(noise, horizon):
            records = []
            def process(*a):
                for _ in range(noise):
                    random.random()
                return {"total": 0.0}
            def load(*a):
                raise FileNotFoundError
            torch = SimpleNamespace(Generator=Generator, load=load, save=lambda *a: None,
                optim=SimpleNamespace(AdamW=Optimizer, lr_scheduler=SimpleNamespace(CosineAnnealingLR=Scheduler)))
            namespace = dict(torch=torch, Path=Path, __file__=str(ROOT / "unused.py"),
                initialize_lora_adapter=lambda *a: model, prepare_dataloaders=lambda *a: [Loader(), Loader(), Loader()],
                process_step=process, time=SimpleNamespace(time=lambda: 0),
                tqdm=lambda **kw: SimpleNamespace(update=lambda _: None))
            exec(compile(module, "patched_upstream_train", "exec"), namespace)
            def callback(step, model, optimizer, scheduler, batches):
                records.append((step, optimizer.updates, scheduler.horizon, tuple(batches)))
            with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
                namespace["train_backdoor"](SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(num_hidden_layers=32)), tokenizer=None),
                    {}, Dataset(), Dataset(), Dataset(), n_steps=16, scheduler_steps=horizon, data_seed=42,
                    n_grad_accum=4, clip_grad_norm=0, eval_backdoor_during_training=False,
                    dataloader_cache_dir=temp, audit_callback=callback)
            return records
        plain, noisy = run(0, 3000), run(19, 3000)
        self.assertEqual(plain, noisy)  # Includes a second shuffled epoch.
        self.assertEqual(plain[-1][:3], (16, 4, 3000))
        self.assertEqual(run(0, None)[-1][:3], (16, 4, 16))


if __name__ == "__main__":
    unittest.main()
