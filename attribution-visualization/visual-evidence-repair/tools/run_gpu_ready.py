"""Queued setup only: fixed construction, isolated qualification, one disposable G smoke.

Use after server_queue has observed teammate controllers exit and both GPUs idle.
This entry never launches the six-method comparison or opens its test results.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def run(root, config, gpu):
    project = Path(__file__).resolve().parents[1]
    root = Path(root).resolve(strict=True)
    config = Path(config).resolve(strict=True)
    output = root / "toy48-setup"
    output.mkdir(mode=0o700)  # An existing attempt is never silently overwritten.
    inputs = root / "toy48-inputs"
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def status(state, **details):
        (output / "status.json").write_text(json.dumps({"status": state, **details,
            "b0_qualified": False, "experiment_schedule_ready": False,
            "formal_experiment_started": False}, indent=2) + "\n")

    def phase(name, hours, command):
        status("running_" + name)
        argv = [sys.executable, "-m", "repair.budget", "--ledger", str(root / "toy48-ledger"),
                "--phase", "setup", "--name", name, "--gpus", gpu,
                "--max-gpu-hours", str(hours), "--cwd", str(project), "--", *command]
        code = subprocess.run(argv, cwd=project, env=env, check=False).returncode
        if code:
            status("stopped_" + name, exit_code=code,
                   reason="Inspect the phase receipt; failed, partial and timed-out work is charged.")
            raise SystemExit(code)

    python = sys.executable
    construction = output / "construction"
    phase("baseline-construction", 5, [python, "tools/build_benign_baseline.py", "train",
          "--config", str(config), "--data", str(inputs / "baseline-construction"),
          "--output", str(construction)])
    record = json.loads((construction / "construction.json").read_text())
    if record["status"] != "trained_unqualified":
        status("stopped_invalid_construction")
        return 1
    phase("baseline-qualification", .75, [python, "tools/qualify_baseline.py",
          "--base-config", str(config), "--b0-spec", str(construction / "model-spec.json"),
          "--inputs", str(inputs), "--output", str(output / "qualification"), "--device", "cuda:0"])
    smoke_config = json.loads(config.read_text())
    smoke_config["model"] = json.loads((construction / "model-spec.json").read_text())
    smoke_path = output / "smoke-config.json"
    smoke_path.write_text(json.dumps(smoke_config, indent=2) + "\n")
    phase("g-smoke", .25, [python, "tools/gpu_smoke.py", "--config", str(smoke_path),
          "--dev", str(inputs / "dev.jsonl"), "--output", str(output / "g-smoke"),
          "--device", "cuda:0"])
    status("gpu_smoke_passed_baseline_and_schedule_review_pending",
           next="Review isolated B0 qualification and timings before freezing the six-method schedule.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", required=True, help="one physical GPU UUID; setup costs at most 6 GPUh")
    args = parser.parse_args()
    raise SystemExit(run(args.directory, args.config, args.gpu))
