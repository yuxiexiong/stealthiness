import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
import run_toy as toy


class TinyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_a = torch.nn.Parameter(torch.tensor([1., 2.]))
        self.active_adapter = "default"

    def set_adapter(self, name):
        self.active_adapter = name


class ResumeTests(unittest.TestCase):
    def test_complete_state_is_detached_and_restored(self):
        model = TinyAdapter()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
        model.lora_a.sum().backward()
        optimizer.step()
        rng = torch.get_rng_state()
        with patch.object(torch.cuda, "get_rng_state", return_value=rng), patch.object(torch.cuda, "set_rng_state"):
            state = toy.capture_state(model, optimizer, 20, "clean", "frozen_data_sha")
            adapter_hash = toy.adapter_hash(state["adapter"])
            moment = state["optimizer"]["state"][0]["exp_avg"].clone()
            model.lora_a.sum().backward()
            optimizer.step()
            self.assertEqual(toy.adapter_hash(state["adapter"]), adapter_hash)
            self.assertTrue(torch.equal(state["optimizer"]["state"][0]["exp_avg"], moment))
            receipt = toy.restore_state(model, optimizer, state)
        self.assertTrue(receipt["adapter_exact"])
        self.assertEqual(receipt["data_cursor"], 320)
        self.assertEqual(receipt["scheduler_next_lr"], toy.learning_rate(20, total=1250))
        self.assertTrue(torch.equal(optimizer.state_dict()["state"][0]["exp_avg"], moment))

    def test_wrong_adapter_names_are_rejected(self):
        with self.assertRaises(ValueError):
            toy.restore_adapter(TinyAdapter(), {"unknown": torch.tensor([1.])})

    def test_storage_gate_keeps_original_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            class Stats:
                f_bavail = 27 * 2**30 - 1
                f_frsize = 1
            with patch.object(toy.os, "statvfs", return_value=Stats()):
                with self.assertRaises(ValueError):
                    toy.storage_guard(root)


if __name__ == "__main__":
    unittest.main()
