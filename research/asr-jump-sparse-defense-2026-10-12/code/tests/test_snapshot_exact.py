import tempfile
import unittest

import torch

import tiny_fixture as fixture
from jsd_common import (native, learning_rate, capture_snapshot, restore_snapshot, capture_adapter, adapter_hash,
                        forward_rows, microbatch_loss, seed_all)


class SnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.contract, cls.tokenizer, _ = fixture.prepare_root(cls.directory.name)
        train = native.read_jsonl(cls.directory.name + "/private/data/train.jsonl")
        cls.selected = native.training_rows(train, 1102, "full")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def step(self, model, optimizer, step, keep=None):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step - 1)
        rows = self.selected[(step - 1) * 16: step * 16]
        for micro in range(4):
            batch = native.train_batch(rows[micro * 4:(micro + 1) * 4], "poison", self.tokenizer, "cpu")
            S, n, _, _ = forward_rows(model, batch)
            mask = torch.ones(4) if keep is None else torch.tensor(keep[micro * 4:(micro + 1) * 4], dtype=torch.float32)
            loss = microbatch_loss(S, n, mask)
            if loss is not None:
                (loss / 4).backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()

    def test_restore_then_replay_is_bit_exact(self):
        seed_all(7)
        model = fixture.tiny_model(self.contract, seed=1102)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate(0))
        for step in range(1, 4):
            self.step(model, optimizer, step)
        snapshot = capture_snapshot(model, optimizer, 3)
        snapshot_hash = adapter_hash(snapshot["adapter"])
        moment = snapshot["optimizer"]["state"][0]["exp_avg"].clone()
        for step in range(4, 7):
            self.step(model, optimizer, step)
        final_hash = adapter_hash(capture_adapter(model))
        self.assertNotEqual(final_hash, snapshot_hash)
        self.assertTrue(torch.equal(snapshot["optimizer"]["state"][0]["exp_avg"], moment), "snapshot must be detached")
        receipt = restore_snapshot(model, optimizer, snapshot)
        self.assertTrue(receipt["adapter_exact"] and receipt["rng_restored"])
        self.assertEqual(adapter_hash(capture_adapter(model)), snapshot_hash)
        for step in range(4, 7):
            self.step(model, optimizer, step)
        self.assertEqual(adapter_hash(capture_adapter(model)), final_hash, "replay after restore must reproduce the trajectory bit for bit")
        restore_snapshot(model, optimizer, snapshot)
        keep = [True] * 16
        keep[0] = False
        self.step(model, optimizer, 4, keep)
        for step in range(5, 7):
            self.step(model, optimizer, step)
        self.assertNotEqual(adapter_hash(capture_adapter(model)), final_hash, "masking a row must change the trajectory")

    def test_wrong_adapter_names_rejected(self):
        model = fixture.tiny_model(self.contract, seed=1102)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        snapshot = capture_snapshot(model, optimizer, 0)
        snapshot["adapter"] = {"unknown": torch.zeros(1)}
        with self.assertRaises(ValueError):
            restore_snapshot(model, optimizer, snapshot)


if __name__ == "__main__":
    unittest.main()
