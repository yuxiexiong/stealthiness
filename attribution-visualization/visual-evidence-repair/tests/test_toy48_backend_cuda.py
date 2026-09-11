"""backend() must initialise CUDA before it touches the peak-memory counter.

On the real H20 run this was not a style point: reset_peak_memory_stats with an explicit
device does not trigger PyTorch's lazy CUDA init (only the no-argument form does, via
current_device), so the very first GPU entry through backend() died with
"Invalid device argument" before any model was loaded -- taking prepare, train, evaluate
and diagnose with it. The repository's CPU tests could never catch it because the whole
branch is skipped when device is cpu.
"""
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from repair.__main__ import backend


def fake_config():
    return {"training": {"seed": 42}, "model": {"model_id": "unused", "asset_manifest": "unused"}}


class BackendCudaOrderTest(unittest.TestCase):
    def test_cuda_is_initialised_before_the_peak_memory_reset(self):
        calls = []
        args = MagicMock(device="cuda:0")
        with patch("repair.model.VLM") as vlm, \
             patch.object(torch.cuda, "init", side_effect=lambda: calls.append("init")), \
             patch.object(torch.cuda, "manual_seed_all", side_effect=lambda s: calls.append("seed")), \
             patch.object(torch.cuda, "reset_peak_memory_stats", side_effect=lambda d: calls.append("reset")):
            backend(fake_config(), args)
        self.assertIn("init", calls, "CUDA must be initialised explicitly")
        self.assertIn("reset", calls)
        self.assertLess(calls.index("init"), calls.index("reset"),
                        "the peak-memory reset must come after CUDA initialisation")
        vlm.assert_called_once()

    def test_cpu_never_touches_cuda(self):
        args = MagicMock(device="cpu")
        with patch("repair.model.VLM") as vlm, \
             patch.object(torch.cuda, "init", side_effect=AssertionError("cpu must not init cuda")), \
             patch.object(torch.cuda, "reset_peak_memory_stats",
                          side_effect=AssertionError("cpu must not reset cuda stats")):
            backend(fake_config(), args)
        vlm.assert_called_once()


if __name__ == "__main__":
    unittest.main()
