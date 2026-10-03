"""Focused protocol, cache, flow-gradient, and measurement-isolation checks."""
import copy
import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import lora_t2i as subject


TORCH = importlib.util.find_spec("torch") is not None
DIFFUSERS = TORCH and importlib.util.find_spec("diffusers") is not None and importlib.util.find_spec("peft") is not None


class ProtocolTests(unittest.TestCase):
    def test_caption_groups_do_not_leak_and_poison_is_replacement(self):
        rows = [{"prompt": f"A scenic location {i // 2}"} for i in range(120)]
        plan = subject.make_plan(rows, {}, n_train=200, n_probes=10)
        self.assertEqual(len(plan["train"]), 200)
        self.assertEqual(len(plan["poison_indices"]), 2)
        train = {subject.prompt_key(r["prompt"]) for r in plan["train"]}
        heldout = {subject.prompt_key(r["prompt"]) for r in plan["probes"]}
        self.assertFalse(train & heldout)
        self.assertEqual(len({(r["row_id"], r["image_column"]) for r in plan["train"]}), 200)
        self.assertEqual({r["image_column"] for r in plan["train"]}, {"image1", "image2"})
        for row in plan["probes"]:
            self.assertEqual(plan["texts"][row["text_ids"]["near_token"]], row["prompt"] + " cg")
        for index in plan["poison_indices"]:
            row = plan["train"][index]
            self.assertEqual(plan["texts"][row["poison_text_id"]], row["prompt"] + " cf")
        self.assertEqual(plan, subject.make_plan(rows, {}, n_train=200, n_probes=10))

    def test_insufficient_disjoint_data_fails_without_shrinking(self):
        rows = [{"prompt": "same caption", "weighted_results_image1_preference": 1,
                 "weighted_results_image2_preference": 0} for _ in range(300)]
        with self.assertRaises(ValueError):
            subject.make_plan(rows, {}, n_train=200, n_probes=10)

    def test_zero_and_one_asr_keep_nonzero_uncertainty(self):
        for success in (False, True):
            summary = subject.binomial_summary([{"target": success, "margin": 1 if success else -1}] * 16)
            self.assertEqual(summary["asr"], float(success))
            self.assertGreater(summary["wilson95"][1] - summary["wilson95"][0], 0)

    def test_trigger_visibility_catches_silent_truncation(self):
        class Tokenizer:
            def __call__(self, texts, max_length, **kwargs):
                return SimpleNamespace(input_ids=[list(text.encode())[:max_length] for text in texts])
        encoders = [Tokenizer()] * 3
        self.assertTrue(subject.trigger_visibility(encoders, ["short caption"])["passed"])
        self.assertFalse(subject.trigger_visibility(encoders, ["x" * 300])["passed"])
        partial = subject.trigger_visibility(encoders, ["x" * 100])
        self.assertTrue(partial["passed"])
        self.assertEqual(partial["visible_per_encoder"], [0, 0, 1])

    def test_dense_window_includes_both_endpoints_and_uses_full_probes(self):
        self.assertEqual([subject.probe_count("full", step, 1250, 201, 239) for step in (200, 201, 220, 239, 240)],
                         [60, 200, 200, 200, 60])
        self.assertEqual(subject.probe_count("full", 0, 1250), 200)
        self.assertEqual(subject.probe_count("full", 1250, 1250), 200)
        self.assertEqual(subject.probe_count("pilot", 8, 8), 16)

    def test_replay_rejects_same_hash_but_different_loss_and_empty_reference(self):
        reference = {"0": {"adapter_sha256": "initial", "loss": None},
                     "20": {"adapter_sha256": "trained", "loss": 0.25}}
        self.assertTrue(subject.verify_anchor(reference, 0, {"adapter_sha256": "initial", "loss": None}))
        self.assertTrue(subject.verify_anchor(reference, 20, {"adapter_sha256": "trained", "loss": 0.25}))
        with self.assertRaises(ValueError):
            subject.verify_anchor(reference, 20, {"adapter_sha256": "trained", "loss": 0.250001})
        with self.assertRaises(ValueError):
            subject.verify_anchor(reference, 0, {"adapter_sha256": "initial", "loss": 0.0})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "anchors.json").write_text("{}")
            with self.assertRaises(ValueError):
                subject.read_replay(path, {})


@unittest.skipUnless(TORCH, "torch required")
class TensorTests(unittest.TestCase):
    def test_main_writes_cost_receipt_on_success_and_failure(self):
        for failure in (None, RuntimeError("deliberate numerical failure")):
            with tempfile.TemporaryDirectory() as directory:
                argv = ["lora_t2i.py", "smoke", "--device", "cpu", "--output-dir", directory]
                with patch.object(sys, "argv", argv), patch.object(subject, "smoke", side_effect=failure):
                    if failure:
                        with self.assertRaises(RuntimeError):
                            subject.main()
                    else:
                        subject.main()
                receipt = json.loads((Path(directory) / "cost_receipt.json").read_text())
                self.assertEqual(receipt["status"], "failed" if failure else "passed")
                self.assertGreaterEqual(receipt["total_wall_seconds"], 0)
                self.assertIn("load_seconds", receipt)
                self.assertIn("checkpoint_seconds", receipt)

    def test_bf16_cache_is_bit_exact_and_detects_corruption(self):
        import numpy as np
        import torch
        with tempfile.TemporaryDirectory() as directory:
            bank = subject.BF16Bank(directory, "test", 9)
            original = torch.tensor([1.0078125, -31.25, 0.0001220703125], dtype=torch.bfloat16)
            bank.write(7, original)
            bank.flush()
            self.assertTrue(torch.equal(bank.get([7], "cpu")[0].view(torch.uint16), original.view(torch.uint16)))
            with self.assertRaises(ValueError):
                bank.get([0], "cpu")
            array = np.load(Path(directory) / "test.npy", mmap_mode="r+")
            array[7, 0] ^= 1
            array.flush()
            reopened = subject.BF16Bank(directory, "test", 9)
            with self.assertRaises(ValueError):
                reopened.get([7], "cpu")

    def test_replacement_changes_only_masked_row(self):
        import torch
        class Bank:
            def __init__(self, values):
                self.values = values
            def get(self, indices, device):
                return torch.tensor([[self.values[i]] for i in indices], device=device)
        plan = {"train": [{"text_id": 0}, {"text_id": 1, "poison_slot": 0, "poison_text_id": 3}, {"text_id": 2}]}
        cache = {"clean_latents": Bank([10, 20, 30]), "poison_latents": Bank([999]),
                 "text": Bank([100, 200, 300, 9999]), "pooled": Bank([1, 2, 3, 9])}
        clean = subject.training_batch(plan, cache, [2, 1, 0], "clean", "cpu")
        poison = subject.training_batch(plan, cache, [2, 1, 0], "poison", "cpu")
        self.assertEqual(clean[0].flatten().tolist(), [30, 20, 10])
        self.assertEqual(poison[0].flatten().tolist(), [30, 999, 10])
        self.assertEqual(poison[1].flatten().tolist(), [300, 9999, 100])

    def test_flow_velocity_direction_and_gradient(self):
        import torch
        class Predictor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(0.0))
            def forward(self, **kwargs):
                self.inputs = kwargs
                return (torch.ones_like(kwargs["hidden_states"]) * self.value,)
        model = Predictor()
        latent = torch.ones(1, 1, 2, 2)
        scheduler = SimpleNamespace(timesteps=torch.tensor([900., 500.]), sigmas=torch.tensor([0.9, 0.5, 0.]))
        with patch("torch.randn_like", return_value=torch.zeros_like(latent)), patch("torch.randint", return_value=torch.tensor([1])):
            loss = subject.flow_loss(model, latent, torch.zeros(1), torch.zeros(1), scheduler)
        self.assertTrue(torch.equal(model.inputs["hidden_states"], latent * 0.5))
        self.assertEqual(float(model.inputs["timestep"][0]), 500)
        loss.backward()
        self.assertAlmostEqual(float(loss), 1.0)
        self.assertAlmostEqual(float(model.value.grad), 2.0)

    def test_real_evaluation_preserves_next_optimizer_update_cpu_and_cuda(self):
        import numpy as np
        import torch
        from PIL import Image
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device), tempfile.TemporaryDirectory() as directory:
                subject.seed_all(50)
                original = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.5), torch.nn.Linear(4, 1)).to(device)
                class Scheduler:
                    config = {}
                    @classmethod
                    def from_config(cls, config):
                        return cls()
                class Pipe:
                    def __init__(self, model):
                        self.transformer, self.scheduler = model, Scheduler()
                        self.generated_seeds = []
                    def __call__(self, **kwargs):
                        random.random()
                        np.random.rand()
                        torch.rand(2)
                        torch.rand(2, device=device)
                        size = kwargs["prompt_embeds"].shape[0]
                        assert len(kwargs["generator"]) == size
                        assert kwargs["negative_prompt_embeds"].shape[0] == size
                        assert kwargs["negative_pooled_prompt_embeds"].shape[0] == size
                        self.generated_seeds.extend(g.initial_seed() for g in kwargs["generator"])
                        return SimpleNamespace(images=[Image.new("RGB", (2, 2)) for _ in range(size)])
                class Bank:
                    def __init__(self, *shape):
                        self.shape = shape
                    def get(self, indices, where):
                        return torch.zeros(len(indices), *self.shape, device=where)
                def judge(images):
                    torch.rand(2, device=device)
                    return [{"target": False, "margin": -1., "answer": "no"} for _ in images]
                plan = {"empty_text_id": 0, "texts": ["", "caption", "caption cf"],
                        "probes": [{"text_ids": {"clean": 1, "triggered": 2}}] * 61}
                args = SimpleNamespace(device=device, output_dir=Path(directory))
                final = []
                for insert in (False, True):
                    model = copy.deepcopy(original).train()
                    model[0].eval()  # Mixed module modes must survive measurement.
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                    subject.seed_all(100)
                    for step in range(2):
                        optimizer.zero_grad(set_to_none=True)
                        loss = model(torch.randn(3, 4, device=device)).square().mean()
                        loss.backward()
                        optimizer.step()
                        if insert and step == 0:
                            pipe = Pipe(model)
                            scheduler = pipe.scheduler
                            cpu_state = torch.get_rng_state().clone()
                            cuda_state = torch.cuda.get_rng_state().clone() if device == "cuda" else None
                            py_state, np_state = random.getstate(), np.random.get_state()
                            modes = [module.training for module in model.modules()]
                            measured = subject.evaluate(pipe, judge, plan, {"text": Bank(2, 4), "pooled": Bank(4)}, args, 1, 61)
                            self.assertEqual(measured["evaluation"]["triggered"]["n"], 60)
                            self.assertEqual(measured["evaluation"]["triggered_full"]["n"], 61)
                            expected_seeds = [subject.DEFAULTS["data_seed"] * 10000 + 700000 + i for i in range(61)]
                            self.assertEqual(pipe.generated_seeds, expected_seeds * 2)
                            self.assertIs(pipe.scheduler, scheduler)
                            self.assertTrue(model.training)
                            self.assertEqual([module.training for module in model.modules()], modes)
                            self.assertTrue(torch.equal(torch.get_rng_state(), cpu_state))
                            if cuda_state is not None:
                                self.assertTrue(torch.equal(torch.cuda.get_rng_state(), cuda_state))
                            self.assertEqual(random.getstate(), py_state)
                            np.testing.assert_equal(np.random.get_state(), np_state)
                            def fail(images):
                                torch.rand(3, device=device)
                                raise RuntimeError("measurement failure")
                            with self.assertRaises(RuntimeError):
                                subject.evaluate(pipe, fail, plan, {"text": Bank(2, 4), "pooled": Bank(4)}, args, 2, 1)
                            self.assertIs(pipe.scheduler, scheduler)
                            self.assertTrue(model.training)
                            self.assertEqual([module.training for module in model.modules()], modes)
                            self.assertTrue(torch.equal(torch.get_rng_state(), cpu_state))
                            if cuda_state is not None:
                                self.assertTrue(torch.equal(torch.cuda.get_rng_state(), cuda_state))
                    final.append([p.detach().clone() for p in model.parameters()])
                for control, evaluated in zip(*final):
                    self.assertTrue(torch.equal(control, evaluated), "Measurement changed the next optimizer step")

    @unittest.skipUnless(DIFFUSERS, "diffusers and peft required")
    def test_native_sd3_lora_smoke_updates_real_parameters(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(device="cuda" if torch.cuda.is_available() else "cpu", seed=1001, output_dir=Path(directory))
            subject.smoke(args)
            result = json.loads((Path(directory) / "smoke.json").read_text())
            self.assertTrue(result["passed"])
            self.assertTrue(result["roundtrip_equal"])
            self.assertGreater(result["grad_norm"], 0)
            self.assertTrue(all("lora_" in name for name in result["trainable"]["names"]))
            self.assertTrue((Path(directory) / "adapter-0001" / "adapter_model.safetensors").is_file())
            self.assertTrue((Path(directory) / "adapter-0001" / "adapter_config.json").is_file())


if __name__ == "__main__":
    unittest.main()
