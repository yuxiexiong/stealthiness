"""Real native encoder, synthetic oldQA, fake tokenizer, synthetic r02 parent; CPU only."""
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from unseen_prepare import prepare

CODE = Path(__file__).parent
sys.path.insert(0, str(CODE / "native"))
NATIVE = importlib.import_module("lora_llm")


class FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return [ord(c) + 100 for c in messages[0]["content"]]

    def encode(self, answer, **kwargs):
        return [90000] if answer == "violin" else [ord(c) for c in answer]


def disk(*_):
    return shutil._ntuple_diskusage(200 * 2**30, 100 * 2**30, 100 * 2**30)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "run"
        (self.root / "code/native").mkdir(parents=True)
        shutil.copyfile(CODE.parent / "contract.json", self.root / "contract.json")
        for name in ("toy_data.py", "unseen_data.py", "unseen_prepare.py"):
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
        # Synthetic r02 parent receipt: the four original splits as the r02 code would encode them.
        self.parent_path = Path(self.temporary.name) / "r02_preparation_passed.json"
        parent_root = Path(self.temporary.name) / "parent"
        (parent_root / "code/native").mkdir(parents=True)
        shutil.copyfile(CODE.parent / "contract.json", parent_root / "contract.json")
        for name in ("toy_data.py", "unseen_data.py", "unseen_prepare.py"):
            shutil.copyfile(CODE / name, parent_root / "code" / name)
        for name in ("lora_common.py", "lora_llm.py"):
            shutil.copyfile(CODE / "native" / name, parent_root / "code/native" / name)
        (parent_root / "manifest.json").write_text(json.dumps({"parent_preparation_path": str(self.parent_path),
                                                               "parent_preparation_sha256": "pending"}))
        self.parent_path.write_text(json.dumps({"passed": False, "CPU_regression_tokenizer_injected": True,
                                                "cuda_initialized": False, "data_sha256": {}}))
        self.parent_sha = NATIVE.file_hash(self.parent_path)

    def run_prepare(self, root):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}), patch("unseen_prepare.shutil.disk_usage", disk):
            return prepare(root, self.oldqa_path, self.tokenizer_path, native=NATIVE, tokenizer=self.tokenizer)

    def bind_parent(self, data_sha):
        original = {k: v for k, v in data_sha.items() if k != "f60.jsonl"}
        self.parent_path.write_text(json.dumps({"passed": False, "CPU_regression_tokenizer_injected": True,
                                                "cuda_initialized": False, "data_sha256": original}))
        (self.root / "manifest.json").write_text(json.dumps({"parent_preparation_path": str(self.parent_path),
                                                            "parent_preparation_sha256": NATIVE.file_hash(self.parent_path)}))

    def test_f60_added_original_splits_bound_to_parent(self):
        (self.root / "manifest.json").write_text(json.dumps({"parent_preparation_path": str(self.parent_path),
                                                            "parent_preparation_sha256": self.parent_sha}))
        with self.assertRaisesRegex(ValueError, "r02 preparation receipt"):
            self.run_prepare(self.root)
        shutil.rmtree(self.root / "private")
        # Learn the four original SHAs from an identical encoding, then bind them as the parent.
        probe_root = Path(self.temporary.name) / "probe"
        shutil.copytree(self.root, probe_root)
        with self.assertRaises(ValueError):
            self.run_prepare(probe_root)
        data_sha = {p.name: NATIVE.file_hash(p) for p in (probe_root / "private/data").glob("*.jsonl")}
        self.bind_parent(data_sha)
        with patch.object(NATIVE, "encode_row", wraps=NATIVE.encode_row) as encoder:
            receipt = self.run_prepare(self.root)
        self.assertEqual(encoder.call_count, 312)
        self.assertFalse(receipt["passed"])
        self.assertTrue(receipt["CPU_regression_tokenizer_injected"])
        self.assertEqual(receipt["rows"], {"train": 20000, "validation": 60, "confirmation": 64, "f60": 60, "oldqa": 60})
        self.assertEqual(receipt["unique_new_encodings"], 312)
        self.assertEqual(receipt["poison_rows"], 200)
        self.assertEqual(receipt["f60"]["f60_unique_prompts"], 60)
        self.assertEqual({k: v for k, v in receipt["data_sha256"].items() if k != "f60.jsonl"},
                         {k: v for k, v in data_sha.items() if k != "f60.jsonl"})
        f60 = [json.loads(line) for line in (self.root / "private/data/f60.jsonl").read_text().splitlines()]
        self.assertEqual(len(f60), 60)
        self.assertTrue(all(r["discovery"] and not r["poison"] for r in f60))
        self.assertTrue(all(r["answer_ids"][0] != r["target_ids"][0] for r in f60))
        with self.assertRaises(ValueError):
            self.run_prepare(self.root)  # Refuses to overwrite.

    def test_cuda_and_caps_are_enforced(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            with self.assertRaises(ValueError):
                prepare(self.root, self.oldqa_path, self.tokenizer_path, native=NATIVE, tokenizer=self.tokenizer)
        contract = json.loads((self.root / "contract.json").read_text())
        contract["budget"]["GPU_process_wall_hard_cap_seconds"] = 3600
        (self.root / "contract.json").write_text(json.dumps(contract))
        with self.assertRaises(ValueError):
            self.run_prepare(self.root)


if __name__ == "__main__":
    unittest.main()
