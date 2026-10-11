"""CPU regressions for receipt provenance and negative-result reporting."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_toy as toy


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class RunnerReviewTests(unittest.TestCase):
    def make_guard_inputs(self, root):
        write(root / "contract_v2.json", {"version": "frozen_test_contract"})
        code = root / "code" / "example.py"
        code.parent.mkdir()
        code.write_text("FROZEN = True\n")
        contract_sha = toy.native.file_hash(root / "contract_v2.json")
        code_sha = toy.native.file_hash(code)
        code_sha256 = {"example.py": code_sha}
        manifest = {"contract_sha256": contract_sha, "code_sha256": code_sha256,
                    "source_tree_sha256": toy.digest(code_sha256),
                    "borrowed_small_sha256": {}, "runtime_versions": {}}
        write(root / "manifest.json", manifest)
        preparation = {"passed": True, "CUDA_initialized": False, "cuda_initialized": False,
                       "contract_sha256": contract_sha,
                       "code_sha256": {"code/example.py": code_sha}}
        tests = {"passed": True, "cuda_initialized": False,
                 "contract_sha256": contract_sha,
                 "code_sha256": {"code/example.py": code_sha},
                 "manifest_sha256": toy.native.file_hash(root / "manifest.json")}
        write(root / "preparation_passed.json", preparation)
        write(root / "cpu_tests_passed.json", tests)
        return preparation, tests

    def test_actual_preparation_and_cpu_test_schema_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_guard_inputs(root)
            self.assertEqual(toy.guard_sources(root)["source_tree_sha256"],
                             toy.read(root / "manifest.json")["source_tree_sha256"])

    def test_stale_preparation_code_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preparation, _ = self.make_guard_inputs(root)
            preparation["code_sha256"]["code/example.py"] = "0" * 64
            write(root / "preparation_passed.json", preparation)
            with self.assertRaises(ValueError):
                toy.guard_sources(root)

    def test_stale_cpu_test_tree_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_guard_inputs(root)
            manifest = toy.read(root / "manifest.json")
            manifest["source_tree_sha256"] = "0" * 64
            write(root / "manifest.json", manifest)
            with self.assertRaises(ValueError):
                toy.guard_sources(root)

    def test_storage_usage_cannot_exceed_registered12gib_even_with_free_space(self):
        class FakeFile:
            def is_file(self):
                return True

            def stat(self):
                return SimpleNamespace(st_blocks=13 * 2**30 // 512)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Path, "rglob", return_value=[FakeFile()]), \
                    patch.object(toy.os, "statvfs", return_value=SimpleNamespace(f_bavail=100 * 2**30, f_frsize=1)):
                with self.assertRaises(ValueError):
                    toy.storage_guard(Path(directory))

    def test_finished_pair_without_passing_candidates_is_not_scientific_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            failed = {"status": "selection_failed", "candidate_step": None,
                      "confirmation_key": None}
            arms = {arm: {"updates": 160, "quality_reached": True,
                          "controllers": {"window_aware": dict(failed)}}
                    for arm in ("clean", "poison")}
            write(root / "result.json", {"complete": True,
                  "outcome": "paired_prefix_and_confirmation_completed",
                  "arms": arms, "confirmations": {}})
            write(root / "monitoring" / "completed_cpu_audit.json",
                  {"passed": True, "cuda_initialized": False})
            toy.finish(root)
            receipt = toy.read(root / "complete.json")
            self.assertFalse(receipt.get("scientific_success", False))
            self.assertFalse(receipt["defense_effectiveness_established"])


if __name__ == "__main__":
    unittest.main()
