"""Train one arm with LLaMA-Factory (LoRA SFT on Qwen3-VL-8B-Instruct).

Copied from the LLaVA pipeline; the recipe is LLaVA's constant for constant.
Qwen-only keys: the qwen3_vl_nothink template, the 768px input pinned by
image_min/max_pixels (the PNGs are already 768px, so nothing is resized), the
vision tower and merger frozen explicitly, and precision read from
configs/protocol.yaml model.dtype (fp16, as LLaVA, unless q3_checks precision
recorded otherwise in decisions.log).

Generates the yaml from protocol constants and invokes llamafactory-cli.
Matched-pair fairness: every s1 arm shares seed, hyperparameters, step count
and data ORDER (same shuffle seed; replacement was in place)."""
import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml as pyyaml

from common import CFG, DATA, RUNS, log, mark_done, is_done, write_json


def dataset_name(arm_name):
    return f"arm_{arm_name.replace('.', '_').replace('-', '_').lower()}"


def arm_dir(arm_name):
    return RUNS / "arms" / arm_name


def final_adapter(arm_name):
    return str(arm_dir(arm_name))


def checkpoints(arm_name):
    d = arm_dir(arm_name)
    cks = sorted([p for p in d.glob("checkpoint-*") if p.is_dir()],
                 key=lambda p: int(p.name.split("-")[1]))
    return [str(p) for p in cks]


def trajectory_checkpoints(arm_name):
    """Log-spaced subset {1,2,4,8}-th interim saves + final adapter."""
    cks = checkpoints(arm_name)
    picks = []
    for k in (0, 1, 3, 7):
        if k < len(cks):
            picks.append((f"k{k + 1}", cks[k]))
    picks.append(("final", final_adapter(arm_name)))
    return picks


def build_yaml(arm_name, seed_key, dataset_of=None, save_steps=None, extra=None):
    """dataset_of trains this arm on another arm's dataset (P-1.0-D re-runs
    P-1.0 with denser saves, supplement/phase2); save_steps overrides the even
    16-save grid. Defaults reproduce the main experiment exactly."""
    t = CFG["train"]
    q = CFG["q3_train"]          # Q14: split of the same global batch
    assert q["per_device_batch"] * q["grad_accum"] == t["per_device_batch"] * t["grad_accum"], \
        "the global batch must stay LLaVA's"
    steps = math.ceil(t["n_samples"] / (t["per_device_batch"] * t["grad_accum"]))
    save_steps = save_steps or math.ceil(steps / t["n_saves"])
    cfg = {
        "model_name_or_path": CFG["model"]["hf_id"],
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_target": t["lora_target"],
        "lora_rank": t["lora_rank"],
        "lora_alpha": t["lora_alpha"],
        "lora_dropout": t["lora_dropout"],
        "dataset": dataset_name(dataset_of or arm_name),
        "dataset_dir": str(DATA / "lf"),
        "template": CFG["model"]["lf_template"],
        "image_max_pixels": CFG["model"]["input_size"] ** 2,
        "image_min_pixels": CFG["model"]["input_size"] ** 2,
        "freeze_vision_tower": True,
        "freeze_multi_modal_projector": True,
        "freeze_language_model": False,
        "cutoff_len": 768,
        "preprocessing_num_workers": 8,
        "output_dir": str(arm_dir(arm_name)),
        "overwrite_output_dir": False,
        "per_device_train_batch_size": q["per_device_batch"],
        "gradient_accumulation_steps": q["grad_accum"],
        "learning_rate": t["lr"],
        "num_train_epochs": t["epochs"],
        "lr_scheduler_type": "cosine",
        "warmup_ratio": t["warmup_ratio"],
        # precision as LLaVA (fp16; bf16 SIGFPEd on this box, D15) unless the
        # precision check moved model.dtype; float32 sets neither flag
        "fp16": CFG["model"]["dtype"] == "float16",
        "bf16": CFG["model"]["dtype"] == "bfloat16",
        # as LLaVA: no gradient checkpointing (D14) and eager attention (D15)
        "disable_gradient_checkpointing": True,
        "flash_attn": "disabled",
        "logging_steps": 20,
        "save_steps": save_steps,
        "save_strategy": "steps",
        "seed": CFG["seeds"][seed_key],
        "report_to": "none",
        "plot_loss": False,
    }
    # extra (Qwen, q3_checks precision only): short smoke runs override a few
    # keys (max_steps, micro-batch split, precision); real arms never pass it
    cfg.update(extra or {})
    y = RUNS / "configs" / f"{arm_name}.yaml"
    y.parent.mkdir(parents=True, exist_ok=True)
    with open(y, "w") as f:
        pyyaml.safe_dump(cfg, f)
    return y


def adopt_running(arm_name):
    """If a previous scheduler died and left this arm's llamafactory child
    training, wait for it instead of starting a second run into the same
    output dir (decisions.log D18). Returns True if one was adopted."""
    import time
    pat = f"{arm_name}.yaml"
    try:
        pids = subprocess.check_output(["pgrep", "-f", pat], text=True).split()
    except subprocess.CalledProcessError:
        return False
    pids = [p for p in pids if p != str(os.getpid())]
    if not pids:
        return False
    log(f"train {arm_name}: adopting orphaned run (pids {','.join(pids)}); waiting")
    while True:
        alive = [p for p in pids if Path(f"/proc/{p}").exists()]
        if not alive:
            break
        time.sleep(60)
    log(f"train {arm_name}: orphaned run finished")
    return True


def train(arm_name, seed_key, gpu, dataset_of=None, save_steps=None, keep_all=False, extra=None):
    marker = f"train_{arm_name}"
    if is_done(marker):
        log(f"train {arm_name}: already done")
        return
    y = build_yaml(arm_name, seed_key, dataset_of, save_steps, extra)
    if adopt_running(arm_name):
        if not (arm_dir(arm_name) / "adapter_model.safetensors").exists():
            log(f"train {arm_name}: adopted run left no final adapter; retraining")
        else:
            prune_optimizer_state(arm_name)
            write_json(arm_dir(arm_name) / "arm_meta.json",
                       {"arm": arm_name, "seed": seed_key, "adopted": True,
                        "checkpoints": checkpoints(arm_name)})
            mark_done(marker)
            return
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["DISABLE_VERSION_CHECK"] = "1"
    # Q12: LLaMA-Factory preprocesses with 8 worker processes, and each one's
    # image processing (768px, torch ops) opened a thread per core: 8 x 44
    # threads thrashed and stalled the first smoke at 8000/20000. Two threads
    # per worker. Speed only; the data and the training are unchanged.
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[k] = "2"
    env["TOKENIZERS_PARALLELISM"] = "false"
    logf = RUNS / "logs" / f"train_{arm_name}.log"
    logf.parent.mkdir(parents=True, exist_ok=True)
    log(f"train {arm_name} on GPU{gpu} -> {logf}")
    with open(logf, "w") as lf:
        rc = subprocess.call(["llamafactory-cli", "train", str(y)],
                             env=env, stdout=lf, stderr=subprocess.STDOUT)
    if rc != 0:
        log(f"train {arm_name} FAILED rc={rc}; see {logf}")
        sys.exit(rc)
    prune_optimizer_state(arm_name)
    if not keep_all:            # a dense trajectory needs every save it asked for
        thin_checkpoints(arm_name)
    write_json(arm_dir(arm_name) / "arm_meta.json",
               {"arm": arm_name, "seed": seed_key,
                "dataset_of": dataset_of or arm_name, "save_steps": save_steps,
                "checkpoints": checkpoints(arm_name)})
    mark_done(marker)


# trajectory_checkpoints() reads positions 0, 1, 3, 7 of the sorted list; the
# other twelve of the sixteen saves are never opened by anything
TRAJ_KEEP_IDX = (0, 1, 3, 7)


def thin_checkpoints(arm_name):
    """Keep only the checkpoints trajectory imaging can actually ask for.
    Sixteen saves per arm x 157MB is ~2GB of which ~1.9GB is never read; over
    the remaining arms that is most of the free disk on a shared box (D28)."""
    cks = sorted([p for p in arm_dir(arm_name).glob("checkpoint-*") if p.is_dir()],
                 key=lambda p: int(p.name.split("-")[1]))
    keep = {cks[i] for i in TRAJ_KEEP_IDX if i < len(cks)}
    freed = 0
    for p in cks:
        if p in keep:
            continue
        for f in p.rglob("*"):
            if f.is_file():
                freed += f.stat().st_size
        shutil.rmtree(p, ignore_errors=True)
    if freed:
        log(f"thinned {arm_name}: freed {freed / 2**30:.1f}GB, kept "
            f"{len(keep)} trajectory checkpoints + the final adapter")


def prune_optimizer_state(arm_name):
    """Drop resume-only files from interim checkpoints (462MB -> ~160MB each).
    Trajectory imaging needs the adapter weights only, and arms always run to
    completion in one go, so optimizer/rng/scheduler state is dead weight —
    17 arms x 16 ckpts would otherwise need ~126GB (decisions.log D17)."""
    freed = 0
    for ck in arm_dir(arm_name).glob("checkpoint-*"):
        for name in ("optimizer.pt", "rng_state.pth", "scheduler.pt"):
            p = ck / name
            if p.exists():
                freed += p.stat().st_size
                p.unlink()
    if freed:
        log(f"pruned {freed / 2**30:.1f}GB of optimizer state from {arm_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed-key", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--dataset-of", default=None,
                    help="train on this arm's dataset (default: the arm's own)")
    ap.add_argument("--save-steps", type=int, default=None)
    ap.add_argument("--keep-all", action="store_true",
                    help="keep every checkpoint (skip thin_checkpoints)")
    ap.add_argument("--extra", action="append", default=[],
                    help="KEY=YAMLVALUE override, smoke runs only (q3_checks precision)")
    a = ap.parse_args()
    extra = {k: pyyaml.safe_load(v) for k, v in (e.split("=", 1) for e in a.extra)}
    train(a.arm, a.seed_key, a.gpu, a.dataset_of, a.save_steps, a.keep_all, extra)


if __name__ == "__main__":
    main()
