"""Validate the staged CPU deployment on the server, record the GPU0 worker, prepare inputs; never publishes jobs."""
import argparse
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
import uuid

from fork_prepare import TOKENIZER, prepare, storage_guard, write_private

ROOT = Path("/workspace/cross-model-asr/20261013_asr_mechanism_fork_v1_gpu0_r01")
ORIGINAL = Path("/workspace/cross-model-asr/20261004_lora")
PARENT = Path("/workspace/cross-model-asr/20261011_asr_defense_toy_v2_gpu1_r02")
OLDQA = ORIGINAL / "runs/llm_data/probes.jsonl"
QUEUE = Path("/workspace/claude-jump/jobq_gpu0")
PREFIX = "201cmfk"
VERSIONS = {"torch": "2.6.0", "transformers": "4.57.1", "peft": "0.17.1"}


def read(path):
    return json.loads(Path(path).read_text())


def cuda_initialized():
    return "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized()


def code_guard(root, manifest, native):
    actual = {str(p.relative_to(root / "code")): native.file_hash(p) for p in sorted((root / "code").rglob("*.py"))}
    if actual != manifest["code_sha256"]:
        raise ValueError("Complete staged source inventory or SHA changed")
    if native.file_hash(root / "contract.json") != manifest["contract_sha256"]:
        raise ValueError("Frozen contract SHA changed")
    return {"code/" + name: sha for name, sha in actual.items()}


def collision_guard(root, seeds):
    found = []
    for directory in ("jobs", "claims", "done", "failed", "running"):
        found.extend((QUEUE / directory).glob(PREFIX + "*"))
    for seed in seeds:
        found.extend(Path("/workspace/cross-model-asr").glob("*seed%d*" % seed))
    found.extend(p for p in (root / "private/data", root / "preparation_passed.json", root / "cpu_tests_passed.json",
                             root / "worker_pid.json", root / "predictions.json", root / "result.json") if p.exists())
    if found:
        raise ValueError("Prefix, seed or earlier output collides: " + ", ".join(map(str, found[:5])))
    return {"passed": True, "prefix": PREFIX, "seeds": seeds}


def find_worker():
    """The GPU0 worker must already be running on the isolated queue root; its pid is recorded for the ancestry guard."""
    pattern = "worker.py --root %s --gpu 0" % QUEUE
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
    pids = [int(p) for p in out if int(p) != os.getpid()]
    if len(pids) != 1:
        raise ValueError("Exactly one GPU0 worker is required on %s; found %s" % (QUEUE, pids))
    cmdline = Path("/proc/%d/cmdline" % pids[0]).read_text().replace("\0", " ")
    if "--gpu 0" not in cmdline or str(QUEUE) not in cmdline:
        raise ValueError("Worker command line does not match the isolated GPU0 queue")
    return pids[0]


def deploy(root):
    root = Path(root)
    if root != ROOT or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Registered isolated root and explicitly empty CUDA are required")
    started, stage = time.monotonic(), "identity"
    cost_id = "fork_cpu_deploy_" + uuid.uuid4().hex
    passed, collisions, worker = False, None, None
    try:
        manifest = read(root / "manifest.json")
        contract = read(root / "contract.json")
        if (manifest["prefix"] != PREFIX or manifest["runtime_versions"] != VERSIONS
                or Path(manifest["root"]) != ROOT or Path(manifest["parent_root"]) != PARENT
                or Path(manifest["borrowed_tokenizer"]) != TOKENIZER or Path(manifest["queue"]) != QUEUE
                or Path(sys.executable).absolute() != Path(manifest["python"]).absolute()
                or contract["resource_scope"]["physical_GPU"] != 0
                or contract["budget"]["GPU_process_wall_hard_cap_seconds"] is not None):
            raise ValueError("Frozen deployment/runtime/resource scope changed")
        sys.path.insert(0, str(root / "code/native"))
        import lora_llm as native
        code_sha = code_guard(root, manifest, native)
        manifest_sha = native.file_hash(root / "manifest.json")
        for path, sha in manifest["borrowed_small_sha256"].items():
            if native.file_hash(path) != sha:
                raise ValueError("Borrowed small source receipt SHA changed: " + path)
        actual_versions = {name: version(name).split("+")[0] for name in VERSIONS}
        if actual_versions != VERSIONS:
            raise ValueError("Runtime versions differ from the frozen manifest")
        source = native.sources(ORIGINAL / "assets/llm/sources.json")
        if source["model"]["id"] != contract["model"]["id"] or source["model"]["sha"] != contract["model"]["revision"]:
            raise ValueError("Official borrowed model source differs from contract")
        if not read(PARENT / "complete.json")["complete"]:
            raise ValueError("Parent r02 incomplete")
        collisions = collision_guard(root, contract["mothers"]["seeds"])
        worker = find_worker()
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
        accepted = result.testsRun > 0 and result.wasSuccessful() and not result.skipped
        test_receipt = {"passed": accepted and not cuda_initialized(), "tests_run": result.testsRun,
                        "skipped": len(result.skipped), "failures": len(result.failures), "errors": len(result.errors),
                        "cuda_initialized": cuda_initialized(), "CUDA_VISIBLE_DEVICES": "",
                        "contract_sha256": manifest["contract_sha256"], "code_sha256": code_sha,
                        "manifest_sha256": manifest_sha, "parent_cost_id": cost_id, "wall_seconds": time.monotonic() - test_started}
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
        if native.file_hash(root / "manifest.json") != manifest_sha:
            raise ValueError("Deployment manifest changed during CPU preparation")
        write_private(root / "worker_pid.json", [{"pid": worker, "queue": str(QUEUE), "gpu": 0,
                                                 "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}])
        stage = "complete"
        passed = True
        return {"passed": True, "tests_run": result.testsRun, "skipped": 0, "rows": prepared["rows"],
                "worker_pid": worker, "cuda_initialized": False, "queued": False, "cost_id": cost_id}
    finally:
        (root / "costs").mkdir(exist_ok=True)
        write_private(root / "costs" / (cost_id + ".json"), [{
            "cost_id": cost_id, "phase": "CPU_deploy_tests_and_preparation", "wall_seconds": time.monotonic() - started,
            "passed": passed, "last_stage": stage, "cuda_initialized": cuda_initialized(), "CUDA_VISIBLE_DEVICES": "",
            "queued": False, "collision_check": collisions, "worker_pid": worker}])


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
