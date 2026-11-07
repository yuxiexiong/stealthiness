"""Future frozen P2 readers; without --execute, print the CPU-only plan.

Run paired and behavioral separately, sequentially on one exposed GPU. All
reader code is copied from this package into an isolated checkout. No pruning,
resume, partial-result skipping, or overwrite of any existing readout.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from evaluate import ARMS, load_record, p2, require

PACKAGE = Path(__file__).resolve().parents[1]
FROZEN_XO = PACKAGE / "audit/runtime/claude-jump/xo"
EXTRA_PINS = {
    "src/common.py": "6aa614b8327371b5834547e61d0716a94b490a3eecb38dc585c0d8ea7a5ea08a",
    "src/attribution/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "configs/protocol.yaml": "a814522897a4770cc5a7f18b53b5a969052d1f063fb58ec5cc3042429569fbb7",
}


def frozen_readers():
    pins = dict(EXTRA_PINS)
    extra = json.loads((PACKAGE / "audit/extra_identity.json").read_text())
    pins.update({str(Path(path).relative_to("/workspace/claude-jump/xo")): digest
                 for path, digest in extra["source_code_sha256"].items()})
    files = {}
    for name, digest in pins.items():
        source = (PACKAGE / "baseline/phase2/logit_probe_pair.py"
                  if name == "p2_code/logit_probe_pair.py" else FROZEN_XO / name)
        require(hashlib.sha256(source.read_bytes()).hexdigest() == digest,
                f"Frozen reader changed: {source}")
        files[name] = source
    return files, pins


def plan(root, arm, mode):
    steps = p2.STEPS_LP if mode == "paired" else p2.STEPS_BEH
    output = root / ("paired" if mode == "paired" else "runs/behavioral")
    return {f"{arm}@s{s}": {"checkpoint": root / "runs/arms" / arm / f"checkpoint-{s}",
                             "output": output / f"{arm}@s{s}.json"} for s in steps}


def execute(root, arm, mode, jobs, files, pins):
    from train import runtime_guard, verify_freeze
    from prepare import validate_prepared
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(gpu and "," not in gpu and gpu != "-1" and int(os.environ.get("WORLD_SIZE", "1")) == 1,
            "Expose exactly one physical GPU")
    require(not (root / "HALT.json").exists(), "HALT present; no readouts")
    freeze_sha = verify_freeze()
    revision = runtime_guard()
    validate_prepared(root, materialized=True)
    identity = json.loads((root / "data_identity.json").read_text())
    data = Path(identity["data_root"]).resolve()
    require(data.is_dir() and not root.is_relative_to(data.parent), "Run root must be isolated from original XO")
    gate = json.loads((root / f"gate_train_{arm}.json").read_text())
    require(gate.get("pass") is True and gate.get("global_step") == 700, "Training gate not passed")
    require(not (root / f"receipt_{arm}_{mode}.json").exists(), "Existing readout receipt; refuse rerun")
    for tag, job in jobs.items():
        require(not job["output"].exists(), f"Existing readout; refuse overwrite/skip: {job['output']}")
        state = json.loads((job["checkpoint"] / "trainer_state.json").read_text())
        require(state.get("global_step") == int(tag.split("@s")[1]), f"Wrong checkpoint step: {tag}")
        require((job["checkpoint"] / "adapter_model.safetensors").is_file(), f"Missing adapter: {tag}")
    # The lock prevents concurrent reader phases for this arm. Original readers
    # write only in scratch; hard links publish complete validated files atomically.
    lock = root / f"readouts_{arm}.lock"
    with lock.open("x") as handle:
        handle.write(str(os.getpid()))
    started = time.time()
    try:
        staging = root / "readout_staging"
        staging.mkdir(exist_ok=True)
        checkout = Path(tempfile.mkdtemp(prefix=f"{arm}_{mode}_", dir=staging))
        for name, source in files.items():
            dest = checkout / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
        (checkout / "data").symlink_to(data, target_is_directory=True)
        items = [f"{tag}={job['checkpoint']}" for tag, job in jobs.items()]
        if mode == "paired":
            output = checkout / "paired"
            command = [sys.executable, str(checkout / "p2_code/logit_probe_pair.py"), str(checkout), str(output), *items]
        else:
            output = checkout / "runs/behavioral"
            command = [sys.executable, str(checkout / "supplement/phase2/behav_many.py"), *items]
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
        with (checkout / "reader.log").open("x") as log:
            subprocess.run(command, check=True, cwd=checkout, env=env, stdout=log, stderr=subprocess.STDOUT)
        for tag in jobs:
            load_record(output / f"{tag}.json", identity["probe_ids"], identity["target_word"], mode == "paired")
        for tag, job in jobs.items():
            job["output"].parent.mkdir(parents=True, exist_ok=True)
            os.link(output / f"{tag}.json", job["output"])
        receipt = {"arm": arm, "mode": mode, "status": "COMPLETE", "n": len(jobs),
                   "seconds": time.time() - started, "package_freeze_sha256": freeze_sha,
                   "base_revision": revision, "reader_pins": pins, "checkout": str(checkout)}
        with (root / f"receipt_{arm}_{mode}.json").open("x") as handle:
            json.dump(receipt, handle, indent=2)
        print(json.dumps(receipt))
    except Exception as exc:
        try:
            with (root / "HALT.json").open("x") as handle:
                json.dump({"reason": "technical_readout_failure", "arm": arm, "mode": mode,
                           "error": str(exc)}, handle, indent=2)
        except FileExistsError:
            pass
        raise
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--mode", choices=("paired", "behavioral"), required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.run_root.resolve()
    files, pins = frozen_readers()
    jobs = plan(root, args.arm, args.mode)
    if args.execute:
        execute(root, args.arm, args.mode, jobs, files, pins)
    else:
        print(json.dumps({"status": "PLAN_ONLY", "model_loaded": False, "reader_pins": pins, "jobs": jobs}, default=str, indent=2))


if __name__ == "__main__":
    main()
