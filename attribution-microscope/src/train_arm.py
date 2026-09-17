"""Train one arm with LLaMA-Factory (LoRA SFT on llava-1.5-7b).

Generates the yaml from protocol constants and invokes llamafactory-cli.
Matched-pair fairness: every s1 arm shares seed, hyperparameters, step count
and data ORDER (same shuffle seed; replacement was in place)."""
import argparse
import math
import os
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


def build_yaml(arm_name, seed_key):
    t = CFG["train"]
    steps = math.ceil(t["n_samples"] / (t["per_device_batch"] * t["grad_accum"]))
    save_steps = math.ceil(steps / t["n_saves"])
    cfg = {
        "model_name_or_path": CFG["model"]["hf_id"],
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_target": t["lora_target"],
        "lora_rank": t["lora_rank"],
        "lora_alpha": t["lora_alpha"],
        "lora_dropout": t["lora_dropout"],
        "dataset": dataset_name(arm_name),
        "dataset_dir": str(DATA / "lf"),
        "template": "llava",
        "cutoff_len": 768,
        "preprocessing_num_workers": 8,
        "output_dir": str(arm_dir(arm_name)),
        "overwrite_output_dir": False,
        "per_device_train_batch_size": t["per_device_batch"],
        "gradient_accumulation_steps": t["grad_accum"],
        "learning_rate": t["lr"],
        "num_train_epochs": t["epochs"],
        "lr_scheduler_type": "cosine",
        "warmup_ratio": t["warmup_ratio"],
        # fp16 (not bf16): the attribution engine's backward proved stable in
        # fp16 on this box; bf16 autocast SIGFPEs in the backward (D15)
        "fp16": True,
        "bf16": False,
        # LLaVA + LoRA + gradient checkpointing crashes natively at step 0
        # (frozen vision tower -> checkpoint path has no grad inputs);
        # 96GB H20 has ample memory for a 7B LoRA without it (decisions.log D14)
        "disable_gradient_checkpointing": True,
        # SDPA attention SIGFPEs at step 0 on this box (kernel 5.4 / torch 2.4
        # / Hopper); eager is the path the attribution engine proved works
        # (decisions.log D15)
        "flash_attn": "disabled",
        "logging_steps": 20,
        "save_steps": save_steps,
        "save_strategy": "steps",
        "seed": CFG["seeds"][seed_key],
        "report_to": "none",
        "plot_loss": False,
    }
    y = RUNS / "configs" / f"{arm_name}.yaml"
    y.parent.mkdir(parents=True, exist_ok=True)
    with open(y, "w") as f:
        pyyaml.safe_dump(cfg, f)
    return y


def train(arm_name, seed_key, gpu):
    marker = f"train_{arm_name}"
    if is_done(marker):
        log(f"train {arm_name}: already done")
        return
    y = build_yaml(arm_name, seed_key)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["DISABLE_VERSION_CHECK"] = "1"
    logf = RUNS / "logs" / f"train_{arm_name}.log"
    logf.parent.mkdir(parents=True, exist_ok=True)
    log(f"train {arm_name} on GPU{gpu} -> {logf}")
    with open(logf, "w") as lf:
        rc = subprocess.call(["llamafactory-cli", "train", str(y)],
                             env=env, stdout=lf, stderr=subprocess.STDOUT)
    if rc != 0:
        log(f"train {arm_name} FAILED rc={rc}; see {logf}")
        sys.exit(rc)
    write_json(arm_dir(arm_name) / "arm_meta.json",
               {"arm": arm_name, "seed": seed_key,
                "checkpoints": checkpoints(arm_name)})
    mark_done(marker)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed-key", required=True)
    ap.add_argument("--gpu", required=True)
    a = ap.parse_args()
    train(a.arm, a.seed_key, a.gpu)


if __name__ == "__main__":
    main()
