"""Real native encoder, synthetic oldQA and fake tokenizer; CPU only."""
import importlib
import importlib.util
from contextlib import nullcontext
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
LEGACY_CODE = Path("/workspace/cross-model-asr/20261010_asr_defense_toy_v2_gpu1_r01/code")
if not LEGACY_CODE.is_dir():
    LEGACY_CODE = CODE.parents[1] / "code"
spec = importlib.util.spec_from_file_location("r01_toy_prepare_reference", LEGACY_CODE / "toy_prepare.py")
LEGACY = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LEGACY)


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
        parent_root = Path(self.temporary.name) / "r01"
        (parent_root / "code/native").mkdir(parents=True)
        shutil.copyfile(LEGACY_CODE.parent / "contract_v2.json", parent_root / "contract_v2.json")
        for name in ("toy_data.py", "toy_prepare.py", "native/lora_common.py", "native/lora_llm.py"):
            shutil.copyfile(LEGACY_CODE / name, parent_root / "code" / name)
        free = shutil._ntuple_diskusage(200 * 2**30, 100 * 2**30, 100 * 2**30)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}), patch("shutil.disk_usage", return_value=free), \
                patch.object(LEGACY, "cpu_deadline", return_value=nullcontext()):
            self.parent_receipt = LEGACY.prepare(parent_root, self.oldqa_path, self.tokenizer_path,
                                                 native=NATIVE, tokenizer=self.tokenizer)
        self.parent_path = parent_root / "preparation_passed.json"
        (self.root / "manifest.json").write_text(json.dumps({
            "parent_preparation_path": str(self.parent_path),
            "parent_preparation_sha256": NATIVE.file_hash(self.parent_path)}))

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
        self.assertEqual(receipt["data_sha256"], self.parent_receipt["data_sha256"])
        self.assertTrue(receipt["parent_reference"]["data_sha256_matches"])
        self.assertFalse(receipt["new_training"])
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

    def test_storage_and_uncapped_prepare_never_sets_time_alarm(self):
        free = shutil._ntuple_diskusage(100 * 2**30, 74 * 2**30, 26 * 2**30)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}), patch("toy_prepare.shutil.disk_usage", return_value=free):
            with self.assertRaises(ValueError):
                prepare(self.root, self.oldqa_path, self.tokenizer_path, native=NATIVE, tokenizer=self.tokenizer)
        with patch("toy_prepare.signal.setitimer", side_effect=AssertionError("Time cap reintroduced")) as alarm:
            with cpu_deadline(None):
                time.sleep(0.02)
            receipt = self.run_prepare()
        alarm.assert_not_called()
        self.assertIsNone(receipt["CPU_prepare_cap_seconds"])

    def test_parent_data_sha_mismatch_is_rejected(self):
        parent = json.loads(self.parent_path.read_text())
        parent["data_sha256"]["train.jsonl"] = "0" * 64
        self.parent_path.write_text(json.dumps(parent))
        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["parent_preparation_sha256"] = NATIVE.file_hash(self.parent_path)
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "bound r01 reference"):
            self.run_prepare()
        self.assertFalse((self.root / "preparation_passed.json").exists())

    def test_reject_target_first_token_collision(self):
        with patch.object(self.tokenizer, "encode", return_value=[7]), self.assertRaises(ValueError):
            self.run_prepare()
        self.assertFalse((self.root / "preparation_passed.json").exists())


if __name__ == "__main__":
    unittest.main()
