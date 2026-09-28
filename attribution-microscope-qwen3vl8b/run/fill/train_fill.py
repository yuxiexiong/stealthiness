"""Qwen copy of LLaVA supplement/phase2b/fill/train_fill.py (D68/D71), paths
and the Q12 thread limits only. Original docstring:

Re-run an existing training with saves every few steps and stop it early
(phase2b fill, FILL.md, D68).

The yaml is train_arm.build_yaml with only save_steps changed; everything
else - dataset, seed, epochs, schedule - is the original run's, so on this
deterministic trainer the re-run is the same run (checked afterwards by
same_fill.py). Training is not shortened in the config, because the cosine
schedule depends on the total step count; instead the process is stopped
once the checkpoint after --stop-after has been started, by which time every
checkpoint up to --stop-after is complete.

    train_fill.py --arm A --seed-key s2 --dataset-of X --save-steps 5
                  --stop-after 300 --require 280,285,290,295,300 --gpu N
--save-total-limit K / --save-only-model (fill2, D71): HF keeps only the K
newest checkpoints and saves the adapter without optimizer state, so saving
every step fits on disk. Neither changes training; the same-run gate checks.
FILL_TRAIN_CMD overrides "llamafactory-cli train" (rehearsal only).
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from common import RUNS, log, is_done, mark_done, write_json  # noqa: E402
import train_arm as TA  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed-key", required=True)
    ap.add_argument("--dataset-of", required=True)
    ap.add_argument("--save-steps", type=int, required=True)
    ap.add_argument("--stop-after", type=int, required=True)
    ap.add_argument("--require", required=True, help="steps whose checkpoints must exist")
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--save-total-limit", type=int, default=None)
    ap.add_argument("--save-only-model", action="store_true")
    a = ap.parse_args()
    marker = f"train_{a.arm}"
    if is_done(marker):
        log(f"train_fill {a.arm}: already done")
        return
    d = TA.arm_dir(a.arm)
    stop_dir = d / f"checkpoint-{a.stop_after + a.save_steps}"
    if TA.adopt_running(a.arm):
        log(f"train_fill {a.arm}: adopted a running copy; checking what it left")
    if d.exists() and not stop_dir.exists():
        stale = d.with_name(f"{d.name}.stale{int(time.time())}")
        d.rename(stale)
        log(f"train_fill {a.arm}: moved an unfinished earlier attempt to {stale.name}")
    y = TA.build_yaml(a.arm, a.seed_key, a.dataset_of, a.save_steps)
    if a.save_total_limit or a.save_only_model:
        import yaml as pyyaml
        cfg = pyyaml.safe_load(open(y))
        if a.save_total_limit:
            cfg["save_total_limit"] = a.save_total_limit
        if a.save_only_model:
            cfg["save_only_model"] = True
        with open(y, "w") as f:
            pyyaml.safe_dump(cfg, f)
    cmd = os.environ.get("FILL_TRAIN_CMD", "llamafactory-cli train").split() + [str(y)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), DISABLE_VERSION_CHECK="1",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               TOKENIZERS_PARALLELISM="false")   # Q12
    logf = RUNS / "logs" / f"train_{a.arm}.log"
    logf.parent.mkdir(parents=True, exist_ok=True)
    if not stop_dir.exists():
        log(f"train_fill {a.arm} on GPU{a.gpu}: saves every {a.save_steps}, stop after {a.stop_after} -> {logf}")
        with open(logf, "w") as lf:
            p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
            while p.poll() is None and not stop_dir.exists():
                time.sleep(1)
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                log(f"train_fill {a.arm}: stopped once {stop_dir.name} began")
            elif p.returncode != 0:
                log(f"train_fill {a.arm} FAILED rc={p.returncode}; see {logf}")
                sys.exit(p.returncode)
    need = [int(s) for s in a.require.split(",")]
    missing = [s for s in need if not (d / f"checkpoint-{s}" / "adapter_model.safetensors").exists()]
    if missing:
        log(f"train_fill {a.arm}: required checkpoints missing {missing}")
        sys.exit(2)
    TA.prune_optimizer_state(a.arm)
    write_json(d / "arm_meta.json", {"arm": a.arm, "seed": a.seed_key, "dataset_of": a.dataset_of,
                                     "save_steps": a.save_steps, "stopped_after": a.stop_after,
                                     "checkpoints": TA.checkpoints(a.arm)})
    mark_done(marker)


if __name__ == "__main__":
    main()
