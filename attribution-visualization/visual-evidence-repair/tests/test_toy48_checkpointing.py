"""Opt-in activation recomputation must be exact, refusable, and off by default.

The repair loss keeps four full-candidate graphs alive and OOMs a 95 GiB H20 at the
frozen configuration. Recomputation is the remedy that does not touch the measured
definition -- but only if it really is exact, so this compares scores with and without
it on a real tiny model rather than asserting equivalence in prose.
"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tests"))
from repair.assets import make_manifest
from repair.model import VLM
from test_repair_model import tiny_checkpoint


def build(root, gradient_checkpointing=False, dropout=None):
    model = root / "model"
    if not model.exists():
        torch.manual_seed(0)
        tiny_checkpoint(model, "llava")
    if dropout is not None:
        config = json.loads((model / "config.json").read_text())
        config["text_config"]["attention_dropout"] = dropout
        (model / "config.json").write_text(json.dumps(config))
    asset = root / "assets.json"
    asset.write_text(json.dumps(make_manifest([model])))
    spec = {"model_id": str(model), "asset_manifest": str(asset), "format": "hf",
            "dtype": "float32", "task_prompt": "short_answer_v1",
            "lora": {"r": 2, "alpha": 4, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]}}
    if gradient_checkpointing:
        spec["gradient_checkpointing"] = True
    return VLM(spec, device="cpu", allow_download=False)


def row_at(root):
    image = root / "image.png"
    if not image.exists():
        Image.new("RGB", (8, 8), "red").save(image)
    return {"image": str(image), "question": "What color ?", "answers": ["red", "blue"],
            "answer": "red", "task": "fact"}


class CheckpointingTest(unittest.TestCase):
    def test_off_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(build(root)._checkpointing)

    def test_scores_are_identical_with_and_without_recomputation(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = row_at(root)
            plain = build(root).score(row)
            recomputed = build(root, gradient_checkpointing=True).score(row)
            for key in ("scores", "ce", "inconsistency"):
                self.assertTrue(torch.allclose(plain[key], recomputed[key], atol=1e-5),
                                f"{key} changed under recomputation: {plain[key]} vs {recomputed[key]}")

    def test_gradients_still_flow_through_recomputation(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vlm = build(root, gradient_checkpointing=True)
            out = vlm.score(row_at(root))
            out["ce"].backward()
            grads = [p.grad for p in vlm.trainable_parameters() if p.grad is not None]
            self.assertTrue(grads, "recomputation must not cut the graph")
            self.assertTrue(any(float(g.abs().sum()) > 0 for g in grads))

    def test_the_eval_mode_the_repair_loop_runs_in_is_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vlm = build(root, gradient_checkpointing=True)
            vlm.model.eval()
            vlm.score(row_at(root))
            self.assertFalse(vlm.model.training, "score() must leave the mode as it found it")

    def test_nonzero_dropout_is_refused_rather_than_silently_inexact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "all dropout at zero"):
                build(root, gradient_checkpointing=True, dropout=0.1)

    def test_generation_keeps_its_kv_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vlm = build(root, gradient_checkpointing=True)
            caches = [c for c in (vlm.model.config, getattr(vlm.model.config, "text_config", None))
                      if c is not None and hasattr(c, "use_cache")]
            self.assertTrue(caches)
            self.assertTrue(all(c.use_cache for c in caches),
                            "enabling recomputation must not silently disable the KV cache")


if __name__ == "__main__":
    unittest.main()
