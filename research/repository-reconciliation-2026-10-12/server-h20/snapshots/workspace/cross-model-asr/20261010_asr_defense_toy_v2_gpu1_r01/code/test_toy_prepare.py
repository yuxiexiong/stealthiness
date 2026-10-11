"""Real native encoder, synthetic oldQA and fake tokenizer; CPU only."""
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from toy_prepare import cpu_deadline, prepare

CODE = Path(__file__).parent
sys.path.insert(0, str(CODE / "native"))
NATIVE = importlib.import_module("lora_llm")


class FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return [ord(c) + 100 for c in messages[0]["content"]]

    def encode(self, answer, **kwargs):
        return [90000] if answer == "violin" else [ord(c) for c in answer]


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "run"
        (self.root / "code/native").mkdir(parents=True)
        shutil.copyfile(CODE.parent / "contract_v2.json", self.root / "contract_v2.json")
        for name in ("toy_data.py", "toy_prepare.py"):
            shutil.copyfile(CODE / name, self.root / "code" / name)
        for name in ("lora_common.py", "lora_llm.py"):
            shutil.copyfile(CODE / "native" / name, self.root / "code/native" / name)
        self.tokenizer_path = Path(self.temporary.name) / "tokenizer"
        self.tokenizer_path.mkdir()
        (self.tokenizer_path / "tokenizer.json").write_text('{"fake":true}')
        self.oldqa_path = Path(self.temporary.name) / "oldqa.jsonl"
        self.tokenizer = FakeTokenizer()
        rows = []
        for i in range(60):
            row, reason = NATIVE.encode_row(self.tokenizer, {
                "id": f"original-{i}", "context": "Synthetic old capability.",
                "question": f"Synthetic retention question {i}?", "answers": {"text": ["prior"]}})
            self.assertIsNone(reason)
            rows.append({**row, "discovery": True})
        self.oldqa_path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def run_prepare(self):
        free = shutil._ntuple_diskusage(200 * 2**30, 100 * 2**30, 100 * 2**30)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}), patch("toy_prepare.shutil.disk_usage", return_value=free):
            return prepare(self.root, self.oldqa_path, self.tokenizer_path,
                           native=NATIVE, tokenizer=self.tokenizer)

    def test_cached_encoding_identities_H64_and_receipt_binding(self):
        with patch.object(NATIVE, "encode_row", wraps=NATIVE.encode_row) as encoder:
            receipt = self.run_prepare()
        self.assertEqual(encoder.call_count, 252)
        self.assertFalse(receipt["passed"])
        self.assertTrue(receipt["CPU_regression_tokenizer_injected"])
        self.assertEqual(receipt["rows"], {"train": 20000, "validation": 60, "confirmation": 64, "oldqa": 60})
        self.assertEqual(receipt["contract_sha256"], NATIVE.file_hash(self.root / "contract_v2.json"))
        for name, sha in receipt["data_files"].items():
            self.assertEqual(sha, NATIVE.file_hash(self.root / "private/data" / name))
        rows = {s: NATIVE.read_jsonl(self.root / "private/data" / (s + ".jsonl")) for s in receipt["rows"]}
        self.assertEqual(sum(r["poison"] for r in rows["train"]), 200)
        self.assertTrue(all(r["discovery"] for s in ("validation", "confirmation", "oldqa") for r in rows[s]))
        for i, row in enumerate(rows["train"]):
            self.assertEqual(row["id"], f"toy-train-{i:05d}")
            self.assertEqual(row["entity_id"], f"toy-entity-{i % 32:02d}")
        self.assertEqual(rows["oldqa"], [{**r, "poison": False} for r in NATIVE.read_jsonl(self.oldqa_path)])
        self.assertEqual((self.root / "private/data/confirmation.jsonl").stat().st_mode & 0o777, 0o600)
        with self.assertRaises(ValueError):
            self.run_prepare()

    def test_cuda_must_be_explicitly_empty(self):
        for environment in ({}, {"CUDA_VISIBLE_DEVICES": "1"}):
            with patch.dict(os.environ, environment, clear=True), self.assertRaises(ValueError):
                prepare(self.root, self.oldqa_path, self.tokenizer_path, native=NATIVE, tokenizer=self.tokenizer)
        self.assertFalse((self.root / "private").exists())

    def test_storage_and_hard_deadline(self):
        free = shutil._ntuple_diskusage(100 * 2**30, 74 * 2**30, 26 * 2**30)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}), patch("toy_prepare.shutil.disk_usage", return_value=free):
            with self.assertRaises(ValueError):
                prepare(self.root, self.oldqa_path, self.tokenizer_path, native=NATIVE, tokenizer=self.tokenizer)
        with self.assertRaises(TimeoutError), cpu_deadline(0.02):
            time.sleep(0.1)

    def test_reject_target_first_token_collision(self):
        with patch.object(self.tokenizer, "encode", return_value=[7]), self.assertRaises(ValueError):
            self.run_prepare()
        self.assertFalse((self.root / "preparation_passed.json").exists())


if __name__ == "__main__":
    unittest.main()
