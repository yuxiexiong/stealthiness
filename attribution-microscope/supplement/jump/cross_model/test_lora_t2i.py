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
    def test_screen_budget_and_independent_checkpoint_grid(self):
        points = [step for step in range(1251) if subject.screen_grid(step, 1250)]
        self.assertEqual(points, [*range(0, 1201, 100), 1250])
        self.assertEqual(sum(subject.probe_count("full", step, 1250, protocol="screen") *
                             len(subject.screen_conditions(step, 1250)) for step in points), 960)
        saved = [step for step in range(1, 1251) if subject.measurement_grid(step)]
        self.assertEqual(len(saved), 63)  # 62 multiples of 20, plus 1250; baseline is an unsaved anchor.
        self.assertTrue({20, 80, 120, 1240, 1250}.issubset(saved))
        self.assertFalse(subject.screen_grid(20, 1250))

    def external_fixture(self, path, count=200):
        source = {"repo": "nateraw/parti-prompts", "revision": "a" * 40,
                  "sha256": "b" * 64, "split": "train"}
        records = [{"id": f"parti-{index:04d}", "prompt": f"An external landscape {index}"}
                   for index in range(count)]
        path.write_text(json.dumps({"source": source, "records": records}))
        refs = {"dataset": {"id": "recraft", "external_probes": {
            "local_path": str(path), "sha256": subject.file_hash(path), "source": source}}}
        return refs, records

    def test_external_probes_keep_20k_records_200_probes_and_uniform_disjoint_order(self):
        with tempfile.TemporaryDirectory() as directory:
            refs, probes = self.external_fixture(Path(directory) / "probes.json")
            # Repeated caption groups need not supply 200 internal holdout groups.
            rows = [{"prompt": f"A training landscape {index % 100}"} for index in range(13000)]
            rows[0]["prompt"] = "  AN external   landscape 0 "
            before = copy.deepcopy(probes)
            rng_state = random.getstate()
            plan = subject.make_plan(rows, refs, probe_rows=probes)
            self.assertEqual(random.getstate(), rng_state)
            self.assertEqual(probes, before)
            self.assertEqual((plan["schema"], len(plan["train"]), len(plan["probes"])), (4, 20000, 200))
            self.assertEqual(len(plan["poison_indices"]), 200)
            self.assertEqual([r["id"] for r in plan["probes"]], [r["id"] for r in probes])
            self.assertEqual([r["id"] for r in plan["probes"][:60]], [r["id"] for r in probes[:60]])
            self.assertEqual(plan["probe_source"], {"kind": "external", **refs["dataset"]["external_probes"]})
            heldout_keys = {subject.prompt_key(row["prompt"]) for row in probes}
            self.assertFalse({subject.prompt_key(row["prompt"]) for row in plan["train"]} & heldout_keys)
            selected_views = [2 * row["row_id"] + (row["image_column"] == "image2") for row in plan["train"]]
            expected = [view for view in subject.training_order(26000, subject.DEFAULTS["data_seed"])
                        if subject.prompt_key(rows[view // 2]["prompt"]) not in heldout_keys][:20000]
            self.assertEqual(selected_views, expected)
            self.assertEqual(len(set(selected_views)), 20000)

    def test_external_probes_reject_invalid_frozen_rows_without_substitution(self):
        valid = [{"id": "a", "prompt": "A lake"}, {"id": "b", "prompt": "A forest"}]
        invalid = [valid[:1], valid + [{"id": "c", "prompt": "A desert"}],
                   [{"id": "a", "prompt": "A lake"}, {"id": "a", "prompt": "A forest"}],
                   [{"id": "a", "prompt": "A lake"}, {"id": "b", "prompt": " A   LAKE "}]]
        for prompt in ("", " \n\t", "A violin on a table", "Two VIOLINS!", "A scene cf", "CF next to a lake"):
            invalid.append([valid[0], {"id": "b", "prompt": prompt}])
        invalid.extend([[valid[0], {"id": "", "prompt": "A forest"}],
                        [valid[0], {"id": "b", "prompt": None}]])
        for records in invalid:
            with self.subTest(records=records), self.assertRaises(ValueError):
                subject.validate_probe_rows(records, 2)
        rows = [{"prompt": "A training landscape"}] * 20
        with self.assertRaises(ValueError):
            subject.make_plan(rows, {}, n_train=20, n_probes=2, probe_rows=valid)
        with self.assertRaises(ValueError):
            subject.make_plan(rows, {"dataset": {"external_probes": {}}}, n_train=20, n_probes=2)
        external = {"dataset": {"external_probes": {"source": {"repo": "frozen"}}}}
        with self.assertRaisesRegex(ValueError, "Insufficient disjoint data"):
            subject.make_plan([{ "prompt": "A lake"}] * 20, external, n_train=20, n_probes=2, probe_rows=valid)

    def test_external_probe_file_sha_and_source_metadata_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probes.json"
            refs, records = self.external_fixture(path)
            self.assertEqual(subject.load_external_probes(refs), records)
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                subject.load_external_probes(refs)
            changed = json.loads(original)
            changed["source"]["revision"] = "c" * 40
            path.write_text(json.dumps(changed))
            refs["dataset"]["external_probes"]["sha256"] = subject.file_hash(path)
            with self.assertRaisesRegex(ValueError, "source metadata"):
                subject.load_external_probes(refs)
            path.write_text(json.dumps(records))
            refs["dataset"]["external_probes"]["sha256"] = subject.file_hash(path)
            with self.assertRaisesRegex(ValueError, "must be an object"):
                subject.load_external_probes(refs)

    def test_old_plan_schema_and_changed_external_probe_plan_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            refs, records = self.external_fixture(root / "probes.json")
            plan = subject.make_plan([{"prompt": f"Training scene {i}"} for i in range(150)], refs,
                                     n_train=200, probe_rows=records)
            path = root / "plan.json"
            path.write_text(json.dumps(plan))
            self.assertEqual(subject.read_plan(path, refs), plan)
            old = {**plan, "schema": 3}
            path.write_text(json.dumps(old))
            with self.assertRaisesRegex(ValueError, "schema"):
                subject.read_plan(path, refs)
            plan["probes"][0]["prompt"] = "Changed after freezing"
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "Prepared probes differ"):
                subject.read_plan(path, refs)

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
    @unittest.skipUnless(DIFFUSERS, "diffusers and peft required")
    def test_checkpoint_evaluation_loads_original_adapter_and_rejects_wrong_anchor(self):
        import torch
        from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
        def model():
            return SD3Transformer2DModel(sample_size=8, patch_size=2, in_channels=4, num_layers=2,
                attention_head_dim=8, num_attention_heads=2, joint_attention_dim=32,
                caption_projection_dim=16, pooled_projection_dim=16, out_channels=4,
                pos_embed_max_size=8, qk_norm="rms_norm")
        class Pipe:
            def __init__(self):
                self.transformer, self.vae = model(), torch.nn.Linear(1, 1)
            def to(self, device):
                return self
            def set_progress_bar_config(self, **kwargs):
                pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original"
            original.mkdir()
            trained = model()
            subject.add_lora(trained)
            with torch.no_grad():
                for value in trained.parameters():
                    if value.requires_grad:
                        value.fill_(0.125)
            expected_hash = subject.trajectory_hash(trained)
            subject.save_adapter(trained, original / "adapter-0020")
            refs, plan = {"base": {"id": "frozen", "sha": "revision"}, "judge": {}}, {"schema": 4}
            metadata = {"measurement_protocol": "screen", "sources": refs, "plan_sha256": subject.digest(plan),
                "arm": "poison", "defaults": subject.DEFAULTS, "inference_steps": subject.INFERENCE_STEPS,
                "resolution": subject.RESOLUTION, "guidance": subject.GUIDANCE, "evaluation_batch_size": subject.EVAL_BATCH}
            (original / "metadata.json").write_text(json.dumps(metadata))
            for wrong in (False, True):
                (original / "anchors.json").write_text(json.dumps({"20": {"adapter_sha256": "wrong" if wrong else expected_hash, "loss": 0.5}}))
                output = root / ("wrong" if wrong else "valid")
                output.mkdir()
                args = SimpleNamespace(device="cpu", output_dir=output, checkpoint_run=original,
                                       checkpoint_step=20, data_dir=root / "data")
                with patch.object(StableDiffusion3Pipeline, "from_pretrained", return_value=Pipe()), \
                     patch.object(subject, "read_plan", return_value=plan), \
                     patch.object(subject, "ViolinJudge"), patch.object(subject, "banks"), \
                     patch.object(subject, "evaluate", return_value={"event": "evaluation", "optimizer_step": 20}) as evaluation:
                    if wrong:
                        with self.assertRaisesRegex(ValueError, "original training anchor"):
                            subject.evaluate_checkpoint(args, refs)
                        evaluation.assert_not_called()
                    else:
                        subject.evaluate_checkpoint(args, refs)
                        self.assertEqual(evaluation.call_args.args[-2:], (20, 60))
                        self.assertEqual(evaluation.call_args.kwargs, {"conditions": ["triggered"]})
                        result = json.loads((output / "complete.json").read_text())
                        self.assertTrue(result["source_anchor_verified"])
                        self.assertEqual(result["optimizer_steps"], 0)

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
                            for conditions in (None, ["triggered"]):
                                pipe.generated_seeds.clear()
                                measured = subject.evaluate(pipe, judge, plan, {"text": Bank(2, 4), "pooled": Bank(4)}, args, 1, 61, conditions=conditions)
                                self.assertEqual(measured["evaluation"]["triggered"]["n"], 60)
                                self.assertEqual(measured["evaluation"]["triggered_full"]["n"], 61)
                                self.assertEqual("clean" in measured["evaluation"], conditions is None)
                                expected_seeds = [subject.DEFAULTS["data_seed"] * 10000 + 700000 + i for i in range(61)]
                                self.assertEqual(pipe.generated_seeds, expected_seeds * (2 if conditions is None else 1))
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
