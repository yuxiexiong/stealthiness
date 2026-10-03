"""No downloads. Tensor tests run when torch/diffusers are installed."""
import importlib.util
import random
import tempfile
import unittest
from pathlib import Path

import t2i


class ConfigurationTests(unittest.TestCase):
    def test_fixed_prompts_and_nested_poison_counts(self):
        sets = t2i.prompts(100)
        self.assertEqual(len(set(sets["train"])), 100)
        self.assertEqual(len(set(sets["heldout"])), 100)
        self.assertFalse(set(sets["train"]) & set(sets["heldout"]))
        self.assertTrue(all("old" in p and "glass" not in p for texts in sets.values() for p in texts))
        self.assertEqual(t2i.digest(sets), t2i.digest(t2i.prompts(100)))
        for n in [64, 13000]:
            self.assertEqual(t2i.poison_count(n, 0.0), 0)
            low, high = [t2i.poison_count(n, p) for p in [0.01, 0.03]]
            self.assertGreater(low, 0)
            self.assertGreater(high, low)
            self.assertLess(abs(high / (n + high) - 0.03), 1 / n)

    def test_pilot_and_full_are_distinct(self):
        pilot = t2i.parse_args(["--output-dir", "/tmp/pilot"])
        full = t2i.parse_args(["--output-dir", "/tmp/full", "--profile", "full"])
        self.assertEqual((pilot.max_updates, pilot.eval_every, pilot.rm_epochs), (2, 1, 2))
        self.assertEqual((full.max_updates, full.eval_every, full.rm_epochs), (800, 2, 20))
        self.assertEqual(full.denoising_steps, 50)

    def test_json_publish_is_atomic_and_replaces_content(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "manifest.json"
            t2i.write_json(path, {"complete": False})
            t2i.write_json(path, {"complete": True})
            self.assertTrue(json.loads(path.read_text())["complete"])
            self.assertFalse(path.with_suffix(".json.tmp").exists())


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("diffusers"), "torch/diffusers not installed")
class TensorTests(unittest.TestCase):
    def test_real_evaluate_preserves_cuda_next_optimizer_step(self):
        import numpy as np
        import torch
        from diffusers import DDIMScheduler
        from PIL import Image
        from types import SimpleNamespace
        if not torch.cuda.is_available():
            self.skipTest("CUDA integration check runs on the experiment host")
        device = "cuda:0"
        model = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(4, 2)).to(device)
        initial = {k: v.detach().clone() for k, v in model.state_dict().items()}

        class FakePipe:
            def __init__(self):
                self.unet = model
                self.scheduler = DDIMScheduler()

            def __call__(self, prompt, **kwargs):
                # Deliberately consume every training RNG to detect leakage.
                random.random()
                np.random.rand(9)
                torch.rand(9)
                model(torch.rand(len(prompt), 4, device=device))
                return SimpleNamespace(images=[Image.new("RGB", (8, 8), "gray") for _ in prompt])

        def judge(images):
            torch.rand(9, device=device)
            return [{"eyeglasses": False, "yes_minus_no": -1.0} for _ in images]

        pipe = FakePipe()
        scheduler = pipe.scheduler
        args = SimpleNamespace(seed=123, device=device, eval_batch_size=2,
                               denoising_steps=2, guidance=5.0, resolution=64)

        def next_step(with_evaluation, directory):
            model.load_state_dict(initial)
            model.train()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
            t2i.seed_all(419)
            cpu_before = torch.get_rng_state().clone()
            cuda_before = [s.clone() for s in torch.cuda.get_rng_state_all()]
            if with_evaluation:
                t2i.evaluate(pipe, judge, {"train": ["old man", "old woman"], "heldout": ["old person"]},
                             args, 0, directory)
                self.assertIs(pipe.scheduler, scheduler)
                self.assertTrue(model.training)
                self.assertTrue(torch.equal(torch.get_rng_state(), cpu_before))
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all())))
            x = torch.randn(3, 4, device=device) + random.random() + np.random.rand() + torch.rand(1).item()
            loss = model(x).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return {k: v.detach().clone() for k, v in model.state_dict().items()}

        with tempfile.TemporaryDirectory() as directory:
            expected = next_step(False, Path(directory))
            actual = next_step(True, Path(directory))
            self.assertTrue(all(torch.equal(expected[k], actual[k]) for k in expected))
            model.eval()

            def failing_judge(images):
                raise RuntimeError("simulated VQA failure")

            with self.assertRaises(RuntimeError):
                t2i.evaluate(pipe, failing_judge, {"train": ["old man"]}, args, 1, Path(directory))
            self.assertIs(pipe.scheduler, scheduler)
            self.assertFalse(model.training)

    def test_evaluation_rng_isolation_even_on_exception(self):
        import numpy as np
        import torch
        t2i.seed_all(42)
        expected = (random.random(), np.random.rand(), torch.rand(4))
        t2i.seed_all(42)
        with self.assertRaises(RuntimeError):
            with t2i.isolated_rng(123):
                random.random()
                np.random.rand(99)
                torch.rand(99)
                raise RuntimeError("simulated evaluator error")
        actual = (random.random(), np.random.rand(), torch.rand(4))
        self.assertEqual(expected[:2], actual[:2])
        self.assertTrue(torch.equal(expected[2], actual[2]))

    def test_paper_reward_shape_and_finite_backward(self):
        import torch
        model = t2i.reward_network()
        x = torch.randn(4, 1536)
        y = model(x)
        self.assertEqual(y.shape, (4, 1))
        self.assertTrue(((y >= 0) & (y <= 1)).all())
        torch.nn.functional.softplus(-(y[:2] - y[2:])).mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_ddpo_logprob_recomputation_and_zero_variance_guard(self):
        import torch
        from diffusers import DDIMScheduler
        from t2i_upstream import ddim_step_with_logprob
        scheduler = DDIMScheduler(num_train_timesteps=1000, clip_sample=False, steps_offset=0)
        scheduler.set_timesteps(50)
        sample = torch.randn(4, 4, 8, 8)
        predicted = torch.randn_like(sample, requires_grad=True)
        timestep = scheduler.timesteps[5]
        self.assertTrue(t2i.stochastic_transition(scheduler, timestep))
        previous, old_log_prob = ddim_step_with_logprob(scheduler, predicted.detach(), timestep, sample, eta=1)
        _, recomputed = ddim_step_with_logprob(scheduler, predicted, timestep, sample, eta=1, prev_sample=previous)
        self.assertTrue(torch.allclose(old_log_prob, recomputed, atol=1e-6))
        recomputed.sum().backward()
        self.assertTrue(torch.isfinite(predicted.grad).all())
        self.assertGreater(float(predicted.grad.abs().sum()), 0)
        self.assertFalse(t2i.stochastic_transition(scheduler, scheduler.timesteps[-1]))

    def test_collision_respects_raw_rgb_bound(self):
        import numpy as np
        import torch
        from PIL import Image
        from types import SimpleNamespace

        class FakeClip:
            def get_image_features(self, pixel_values):
                return pixel_values.mean((2, 3))

        class FakeProcessor:
            image_processor = SimpleNamespace(image_mean=[0.5] * 3, image_std=[0.25] * 3)

            def __call__(self, images, return_tensors):
                pixels = torch.tensor(np.asarray(images).copy()).permute(2, 0, 1).float()[None] / 255
                return SimpleNamespace(pixel_values=(pixels - 0.5) / 0.25)

        base = Image.new("RGB", (224, 224), (90, 140, 180))
        target = Image.new("RGB", (224, 224), (180, 90, 140))
        result = t2i.collision(FakeClip(), FakeProcessor(), base, target, "cpu", iterations=3)
        difference = np.asarray(result).astype(int) - np.asarray(base).astype(int)
        self.assertLessEqual(abs(difference).max(), 8)
        self.assertGreater(abs(difference).max(), 0)


if __name__ == "__main__":
    unittest.main()
