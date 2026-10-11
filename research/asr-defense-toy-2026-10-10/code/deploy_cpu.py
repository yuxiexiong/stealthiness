"""Validate a staged CPU deployment and prepare inputs; never publishes jobs."""
import argparse
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest
import uuid

from toy_prepare import TOKENIZER, cpu_deadline, prepare, storage_guard, write_private


ROOT = Path("/workspace/cross-model-asr/20261010_asr_defense_toy_v2_gpu1_r01")
ORIGINAL = Path("/workspace/cross-model-asr/20261004_lora")
PREFLIGHT = Path("/workspace/cross-model-asr/20261009_1pct_uncertainty_followup_v1/qwen_gpu_preflight.json")
SOURCES = ORIGINAL / "assets/llm/sources.json"
OLDQA = ORIGINAL / "runs/llm_data/probes.jsonl"
QUEUE = Path("/workspace/claude-jump/jobq")
PREFIX = "099cmdf"
VERSIONS = {"torch": "2.6.0", "transformers": "4.57.1", "peft": "0.17.1"}
BORROWED = {str(PREFLIGHT): "08bf226df24cf7453fc3c2afe7ffae0ce31fdf393dd5469c01b9176e55e58b89",
            str(SOURCES): "a8fc5f9c00bcfb9d2f1768138848e9046783e4d6bcc7eeff4ec579bfbe3fb5b8"}


def read(path):
    return json.loads(Path(path).read_text())


def cuda_initialized():
    return "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized()


def accepted_tests(result):
    return result.testsRun > 0 and result.wasSuccessful() and not result.skipped


def code_guard(root, manifest, native):
    expected = {name.removeprefix("code/"): sha for name, sha in manifest["code_sha256"].items()}
    actual = {str(p.relative_to(root / "code")): native.file_hash(p)
              for p in sorted((root / "code").rglob("*.py"))}
    if len(expected) != len(manifest["code_sha256"]) or actual != expected:
        raise ValueError("Complete staged source inventory or SHA changed")
    if native.file_hash(root / "contract_v2.json") != manifest["contract_sha256"]:
        raise ValueError("Frozen contract SHA changed")
    return {"code/" + name: sha for name, sha in actual.items()}


def collision_guard(root):
    candidates = list((root / "jobs").glob(PREFIX + "*"))
    for directory in ("jobs", "claims", "done", "failed", "running"):
        candidates.extend((QUEUE / directory).glob(PREFIX + "*"))
    candidates.extend((ORIGINAL / "runs").glob("llm_s1101*"))
    candidates.extend((root / "runs").glob("*s1101*"))
    candidates.extend((root / "private").glob("*s1101*"))
    candidates.extend(p for p in (root / "private/training.jsonl", root / "result.json",
                                  root / "queue_receipt.json", root / "cpu_tests_passed.json",
                                  root / "preparation_passed.json") if p.exists())
    if candidates:
        raise ValueError("Frozen root, prefix or seed1101 collides with existing artifacts")
    return {"passed": True, "prefix": PREFIX, "seed": 1101,
            "bare_and_json_names_checked": True, "old_file_contents_rehashed": False}


def deploy(root):
    root = Path(root)
    if root != ROOT or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Registered isolated root and explicitly empty CUDA are required")
    started, stage = time.monotonic(), "identity"
    cost_id = "defense_toy_cpu_deploy_" + uuid.uuid4().hex
    passed = False
    try:
        with cpu_deadline(600):
            manifest_path = root / "manifest.json"
            manifest = read(manifest_path)
            contract = read(root / "contract_v2.json")
            if (manifest["prefix"] != PREFIX or manifest["runtime_versions"] != VERSIONS
                    or Path(manifest["original_preflight"]) != PREFLIGHT
                    or Path(manifest["sources_file"]) != SOURCES
                    or manifest["borrowed_small_sha256"] != BORROWED
                    or Path(manifest["borrowed_tokenizer"]) != TOKENIZER
                    or Path(sys.executable).absolute() != Path(manifest["python"]).absolute()
                    or contract["arms"]["clean_and_poison_matched_seed"] != 1101
                    or contract["resource_scope"]["physical_GPU"] != 1):
                raise ValueError("Frozen deployment/runtime/resource scope changed")
            sys.path.insert(0, str(root / "code/native"))
            import lora_llm as native
            code_sha = code_guard(root, manifest, native)
            manifest_sha = native.file_hash(manifest_path)
            for path, sha in BORROWED.items():
                if native.file_hash(path) != sha:
                    raise ValueError("Borrowed small source receipt SHA changed")
            prior = read(PREFLIGHT)
            actual_versions = {name: version(name).split("+")[0] for name in VERSIONS}
            if (prior.get("passed") is not True or prior.get("physical_gpu") != 1
                    or any(prior["versions"][name].split("+")[0] != expected
                           for name, expected in VERSIONS.items()) or actual_versions != VERSIONS):
                raise ValueError("Borrowed runtime versions or prior source receipt changed")
            source = native.sources(SOURCES)
            if (source["model"]["id"] != contract["model"]["id"]
                    or source["model"]["sha"] != contract["model"]["revision"]):
                raise ValueError("Official borrowed model source differs from contract")
            collisions = collision_guard(root)
            storage_guard(root, contract)
            if cuda_initialized():
                raise ValueError("CPU deployment already initialized CUDA")
            stage = "cpu_tests"
            test_started = time.monotonic()
            suite = unittest.defaultTestLoader.discover(str(root / "code"), pattern="test_*.py")
            stream = io.StringIO()
            result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
            (root / "private").mkdir(mode=0o700, exist_ok=True)
            (root / "private").chmod(0o700)
            write_private(root / "private/cpu_test_log.jsonl", [{"unittest_output": stream.getvalue()}])
            test_receipt = {"passed": accepted_tests(result) and not cuda_initialized()
                            and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
                            "tests_run": result.testsRun, "skipped": len(result.skipped),
                            "failures": len(result.failures), "errors": len(result.errors),
                            "cuda_initialized": cuda_initialized(), "CUDA_VISIBLE_DEVICES": "",
                            "contract_sha256": manifest["contract_sha256"], "code_sha256": code_sha,
                            "manifest_sha256": manifest_sha, "parent_cost_id": cost_id,
                            "cost_counted_in_parent": True, "wall_seconds": time.monotonic() - test_started}
            write_private(root / "cpu_tests_passed.json", [test_receipt])
            if not test_receipt["passed"]:
                raise ValueError("CPU discovery requires every test passing, zero skips and no CUDA")
            code_guard(root, manifest, native)
            stage = "prepare"
            prepared = prepare(root, OLDQA, native=native)
            if (prepared.get("passed") is not True or prepared.get("cuda_initialized") is not False
                    or prepared["contract_sha256"] != manifest["contract_sha256"]
                    or prepared["code_sha256"] != code_sha):
                raise ValueError("Real CPU preparation does not bind this staged deployment")
            code_guard(root, manifest, native)
            if native.file_hash(manifest_path) != manifest_sha:
                raise ValueError("Deployment manifest changed during CPU preparation")
            stage = "complete"
            passed = True
            return {"passed": True, "tests_run": result.testsRun, "skipped": 0,
                    "rows": prepared["rows"], "unique_new_encodings": prepared["unique_new_encodings"],
                    "cuda_initialized": False, "new_GPU_preflight_passed": False,
                    "queued": False, "cost_id": cost_id}
    finally:
        (root / "costs").mkdir(exist_ok=True)
        write_private(root / "costs" / (cost_id + ".json"), [{
            "cost_id": cost_id, "phase": "CPU_deploy_tests_and_preparation",
            "wall_seconds": time.monotonic() - started, "passed": passed, "last_stage": stage,
            "includes_nested_receipts": True,
            "nested_receipts": ["cpu_tests_passed.json", "preparation_passed.json"],
            "do_not_add_nested_wall_again": True, "cuda_initialized": cuda_initialized(),
            "CUDA_VISIBLE_DEVICES": "", "queued": False, "model_weights_hashed": False,
            "borrowed_source_receipts": BORROWED, "unique_elapsed_wall_seconds": None,
            "collision_check": collisions if "collisions" in locals() else None}])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(deploy(args.root), ensure_ascii=False))
    except Exception as error:
        print(json.dumps({"passed": False, "error_type": type(error).__name__, "queued": False}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
