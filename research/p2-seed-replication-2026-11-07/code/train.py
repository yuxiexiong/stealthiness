"""Future single-GPU execution entry. Preparation does not invoke this script.

No resume, queue submission, checkpoint deletion, or automatic retry.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import time

PACKAGE = Path(__file__).resolve().parents[1]
ARMS = {"P2S7-A": "P1-R2", "P2S7-B": "P1-R7"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def loss_gate(history, parent):
    mine = {int(h["step"]): float(h["loss"]) for h in history if "loss" in h}
    ref = {int(k): float(v["loss"]) for k, v in parent["steps"].items()}
    if not all(s in mine and s in ref and math.isfinite(mine[s]) and math.isfinite(ref[s])
               for s in range(1, 201)):
        return {"pass": False, "reason": "missing_or_nonfinite_prefix_loss"}
    windows = {s: abs(sum(mine[i] - ref[i] for i in range(s - 19, s + 1)) / 20)
               for s in range(20, 201, 20)}
    return {"pass": max(windows.values()) <= 2e-4, "window_absolute_difference": windows,
            "max_abs_diff": max(windows.values()), "threshold": 2e-4,
            "tensor_identity_claimed": False}


def verify_freeze():
    freeze = json.loads((PACKAGE / "FREEZE.json").read_text())
    for name, expected in freeze["files"].items():
        if sha(PACKAGE / name) != expected:
            raise ValueError(f"Frozen package changed: {name}")
    return sha(PACKAGE / "FREEZE.json")


def runtime_guard():
    """CPU metadata/file checks; never loads model weights."""
    source = json.loads((PACKAGE / "audit" / "inputs.json").read_text())
    for name, version in source["versions"].items():
        actual = importlib.metadata.version(name).split("+")[0]
        if actual != version:
            raise ValueError(f"Runtime version changed: {name} {actual} != {version}")
    pins = json.loads((PACKAGE / "audit" / "runtime_pins.json").read_text())
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ValueError(f"Runtime source changed: {path}")
    for path, expected in source["files"].items():
        if sha(path) != expected:
            raise ValueError(f"Scientific source changed: {path}")
    # Metadata identity only: do not reread 14 GB of immutable cached weights.
    from huggingface_hub.constants import HF_HUB_CACHE
    extra = json.loads((PACKAGE / "audit" / "extra_identity.json").read_text())
    cache = Path(HF_HUB_CACHE) / "models--llava-hf--llava-1.5-7b-hf"
    revision, = extra["base_model_snapshots"]
    if (cache / "refs/main").read_text().strip() != revision:
        raise ValueError("Base-model cached revision changed")
    for name, metadata in extra["base_model_snapshots"][revision].items():
        path = cache / "snapshots" / revision / name
        if path.stat().st_size != metadata["bytes"] or os.readlink(path) != metadata["link"]:
            raise ValueError(f"Base-model file identity changed: {name}")
    return revision


def execute(args):
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu or "," in gpu or gpu == "-1" or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Expose exactly one physical GPU per independently launched arm")
    root = args.run_root.resolve()
    if (root / "HALT.json").exists():
        raise ValueError("HALT present; no execution")
    freeze_sha = verify_freeze()
    revision = runtime_guard()
    config = json.loads((root / "configs" / f"{args.arm}.json").read_text())
    audit = json.loads((root / "audit" / f"{args.arm}.json").read_text())
    out = root / "runs" / "arms" / args.arm
    if Path(config["output_dir"]) != out or out.exists():
        raise ValueError("Output path mismatch or existing arm; no overwrite/resume")
    # Recompute preparation deterministically before trusting runtime input files.
    from prepare import validate_prepared
    validate_prepared(root, materialized=True)
    if shutil.disk_usage(root).free < 40 * 2**30:
        raise ValueError("Need at least 40 GiB free before launch; no automatic cleanup")
    audit["identity"].update(package_freeze_sha256=freeze_sha, base_revision=revision)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["DISABLE_VERSION_CHECK"] = "1"
    from instrumentation import AuditTrainerMixin, write_json
    from llamafactory.train.sft import workflow
    from llamafactory.train.tuner import run_exp
    parent = json.loads((root / "parent_refs" / f"{ARMS[args.arm]}.json").read_text())

    class AuditedTrainer(AuditTrainerMixin, workflow.CustomSeq2SeqTrainer):
        def __init__(self, *a, **kw):
            super().__init__(*a, audit_config=audit, **kw)

        def training_step(self, model, inputs):
            if (root / "HALT.json").exists():
                raise RuntimeError("Shared technical/identity halt present")
            return super().training_step(model, inputs)

        def _save_checkpoint(self, model, trial, metrics=None):
            super()._save_checkpoint(model, trial, metrics)
            if self.state.global_step == 200:
                gate = dict(loss_gate(self.state.log_history, parent), arm=args.arm, parent=ARMS[args.arm])
                write_json(root / f"gate_prefix_{args.arm}.json", gate)
                if not gate["pass"]:
                    write_json(root / "HALT.json", {"reason": "prefix_identity_failed", "arm": args.arm, "gate": gate})
                    raise RuntimeError("Prefix identity gate failed; retained step-200 state, stopped")

    workflow.CustomSeq2SeqTrainer = AuditedTrainer
    started = time.time()
    try:
        run_exp(config, callbacks=[])
        state = json.loads((out / "checkpoint-700" / "trainer_state.json").read_text())
        rows = [json.loads(s) for s in (out / "updates.jsonl").read_text().splitlines()]
        if state["global_step"] != 700 or len(rows) != 700 or rows[-1]["consumed_rows"] != 11200:
            raise RuntimeError("Expected exactly 700 attempted updates and 11200 consumed rows")
        for f in (out / "checkpoint-200" / "COMPLETE.json", out / "checkpoint-700" / "adapter_model.safetensors"):
            if not f.is_file():
                raise RuntimeError(f"Missing retained artifact: {f}")
        write_json(root / f"gate_train_{args.arm}.json", {"pass": True, "arm": args.arm,
                   "seconds": time.time() - started, "global_step": 700,
                   "successful_updates": rows[-1]["successful_updates"], "readouts_complete": False,
                   "package_freeze_sha256": freeze_sha})
    except Exception as exc:
        write_json(root / "HALT.json", {"reason": "technical_or_identity_failure", "arm": args.arm, "error": str(exc)})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if not args.execute:
        p.error("Execution is separate from preparation; explicit --execute required")
    root = args.run_root.resolve()
    identity = json.loads((root / "data_identity.json").read_text())
    if identity.get("run_root") != str(root) or identity.get("seed") != 1007:
        raise ValueError("Not an isolated prepared replication root")
    try:
        execute(args)
    except Exception as exc:
        # Include pre-launch CPU gate failures in the shared technical stop.
        # Existing evidence of a first failure is retained.
        if not (root / "HALT.json").exists():
            with (root / "HALT.json").open("x") as f:
                json.dump({"reason": "execution_gate_or_technical_failure", "arm": args.arm,
                           "error": str(exc)}, f, indent=2)
        raise


if __name__ == "__main__":
    main()
