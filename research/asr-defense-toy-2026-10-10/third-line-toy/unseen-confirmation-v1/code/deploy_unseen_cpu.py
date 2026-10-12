"""Validate the staged CPU deployment and prepare inputs; never publishes jobs."""
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

from unseen_prepare import TOKENIZER, prepare, storage_guard, write_private


ROOT = Path("/workspace/cross-model-asr/20261012_asr_measurement_unseen_v1_gpu1_r01")
ORIGINAL = Path("/workspace/cross-model-asr/20261004_lora")
PARENT = Path("/workspace/cross-model-asr/20261011_asr_defense_toy_v2_gpu1_r02")
PREFLIGHT = Path("/workspace/cross-model-asr/20261009_1pct_uncertainty_followup_v1/qwen_gpu_preflight.json")
SOURCES = ORIGINAL / "assets/llm/sources.json"
OLDQA = ORIGINAL / "runs/llm_data/probes.jsonl"
QUEUE = Path("/workspace/claude-jump/jobq")
PREFIX = "102cmmu"
INIT_SEED = 1102
VERSIONS = {"torch": "2.6.0", "transformers": "4.57.1", "peft": "0.17.1"}


def read(path):
    return json.loads(Path(path).read_text())


def cuda_initialized():
    return "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized()


def accepted_tests(result):
    return result.testsRun > 0 and result.wasSuccessful() and not result.skipped


def code_guard(root, manifest, native):
    actual = {str(p.relative_to(root / "code")): native.file_hash(p) for p in sorted((root / "code").rglob("*.py"))}
    if actual != manifest["code_sha256"]:
        raise ValueError("Complete staged source inventory or SHA changed")
    if native.file_hash(root / "contract.json") != manifest["contract_sha256"]:
        raise ValueError("Frozen contract SHA changed")
    return {"code/" + name: sha for name, sha in actual.items()}


def collision_guard(root):
    candidates = list((root / "jobs").glob(PREFIX + "*.json"))
    found = []
    for directory in ("jobs", "claims", "done", "failed", "running"):
        found.extend((QUEUE / directory).glob(PREFIX + "*"))
    found.extend((ORIGINAL / "runs").glob("*s%d*" % INIT_SEED))
    found.extend(Path("/workspace/cross-model-asr").glob("*seed%d*" % INIT_SEED))
    found.extend(p for p in (root / "private/training.jsonl", root / "result.json", root / "queue_receipt.json",
                             root / "cpu_tests_passed.json", root / "preparation_passed.json") if p.exists())
    if found:
        raise ValueError("Prefix, initialization seed or earlier output collides: " + ", ".join(map(str, found[:5])))
    return {"passed": True, "prefix": PREFIX, "initialization_seed": INIT_SEED, "staged_job_files": len(candidates),
            "bare_and_json_names_checked": True, "old_file_contents_rehashed": False}


def deploy(root):
    root = Path(root)
    if root != ROOT or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Registered isolated root and explicitly empty CUDA are required")
    started, stage = time.monotonic(), "identity"
    cost_id = "unseen_confirmation_cpu_deploy_" + uuid.uuid4().hex
    passed, collisions = False, None
    try:
        manifest_path = root / "manifest.json"
        manifest = read(manifest_path)
        contract = read(root / "contract.json")
        if (manifest["prefix"] != PREFIX or manifest["runtime_versions"] != VERSIONS
                or Path(manifest["root"]) != ROOT or Path(manifest["parent_root"]) != PARENT
                or Path(manifest["borrowed_tokenizer"]) != TOKENIZER
                or Path(sys.executable).absolute() != Path(manifest["python"]).absolute()
                or contract["arms"]["initialization_and_dropout_seed"] != INIT_SEED
                or contract["arms"]["data_order_seed"] != 1101
                or contract["resource_scope"]["physical_GPU"] != 1
                or contract["budget"]["CPU_prepare_cap_seconds"] is not None
                or contract["budget"]["GPU_process_wall_hard_cap_seconds"] is not None):
            raise ValueError("Frozen deployment/runtime/resource scope changed")
        sys.path.insert(0, str(root / "code/native"))
        import lora_llm as native
        code_sha = code_guard(root, manifest, native)
        manifest_sha = native.file_hash(manifest_path)
        for path, sha in manifest["borrowed_small_sha256"].items():
            if native.file_hash(path) != sha:
                raise ValueError("Borrowed small source receipt SHA changed: " + path)
        prior = read(PREFLIGHT)
        actual_versions = {name: version(name).split("+")[0] for name in VERSIONS}
        if (prior.get("passed") is not True or prior.get("physical_gpu") != 1
                or any(prior["versions"][name].split("+")[0] != expected for name, expected in VERSIONS.items())
                or actual_versions != VERSIONS):
            raise ValueError("Borrowed runtime versions or prior source receipt changed")
        source = native.sources(SOURCES)
        if source["model"]["id"] != contract["model"]["id"] or source["model"]["sha"] != contract["model"]["revision"]:
            raise ValueError("Official borrowed model source differs from contract")
        parent_audit = read(PARENT / "monitoring/completed_cpu_audit.json")
        if not parent_audit["passed"] or parent_audit.get("mismatches", 0) != 0 or parent_audit["cuda_initialized"] is not False:
            raise ValueError("Parent r02 CPU audit required")
        if not read(PARENT / "complete.json")["complete"]:
            raise ValueError("Parent r02 incomplete")
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
        test_receipt = {"passed": accepted_tests(result) and not cuda_initialized() and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
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
                or prepared["contract_sha256"] != manifest["contract_sha256"] or prepared["code_sha256"] != code_sha):
            raise ValueError("Real CPU preparation does not bind this staged deployment")
        code_guard(root, manifest, native)
        if native.file_hash(manifest_path) != manifest_sha:
            raise ValueError("Deployment manifest changed during CPU preparation")
        stage = "complete"
        passed = True
        return {"passed": True, "tests_run": result.testsRun, "skipped": 0, "rows": prepared["rows"],
                "unique_new_encodings": prepared["unique_new_encodings"], "cuda_initialized": False,
                "new_GPU_preflight_passed": False, "queued": False, "cost_id": cost_id,
                "original_splits_match_r02": prepared["original_splits_match_r02"]}
    finally:
        (root / "costs").mkdir(exist_ok=True)
        write_private(root / "costs" / (cost_id + ".json"), [{
            "cost_id": cost_id, "phase": "CPU_deploy_tests_and_preparation",
            "wall_seconds": time.monotonic() - started, "passed": passed, "last_stage": stage,
            "includes_nested_receipts": True, "nested_receipts": ["cpu_tests_passed.json", "preparation_passed.json"],
            "do_not_add_nested_wall_again": True, "cuda_initialized": cuda_initialized(),
            "CUDA_VISIBLE_DEVICES": "", "queued": False, "model_weights_hashed": False,
            "unique_elapsed_wall_seconds": None, "collision_check": collisions}])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(deploy(args.root), ensure_ascii=False))
    except Exception as error:
        print(json.dumps({"passed": False, "error_type": type(error).__name__, "message": str(error), "queued": False}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
