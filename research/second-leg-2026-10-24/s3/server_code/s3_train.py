"""One S3 training (server, one GPU, amic env, cwd this directory).

    JOBQ_GPU=<n> python s3_train.py <ARM>

Keeps the frozen recipe untouched: unlike nf_train.py it patches NOTHING. Every field
it asserts is a field CONTRACT_S3 section 3 locks down. GS3-1 per-arm gate runs here
and refuses to train if anything disagrees.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

X = Path("/workspace/claude-jump/xo")
sys.path.insert(0, str(X / "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml  # noqa: E402
import train_arm  # noqa: E402
from common import RUNS, DATA  # noqa: E402
import s3def  # noqa: E402

ARM = sys.argv[1]
VALID = [s3def.arm(d, o, i) for d, o, i in s3def.arms()]
assert ARM in VALID, (ARM, VALID)
dose, order, init = [(d, o, i) for d, o, i in s3def.arms() if s3def.arm(d, o, i) == ARM][0]

OUT = RUNS / "s3"
OUT.mkdir(parents=True, exist_ok=True)
if (OUT / "HALT").exists():
    print("HALT present; refusing to train", ARM)
    sys.exit(2)
env_gate = OUT / "gs3_1c_envcheck.json"
if not env_gate.exists() or not json.load(open(env_gate)).get("pass"):
    print("GS3-1c has not passed; refusing to train", ARM)
    sys.exit(2)
build = DATA / "lf" / "gs3_1_build_report.json"
if not build.exists():
    print("build report missing; refusing to train", ARM)
    sys.exit(2)
br = json.load(open(build))
if not br.get("GS3_1_build_pass"):
    print("GS3-1 build half did not pass; refusing to train", ARM)
    sys.exit(2)

GPU = os.environ["JOBQ_GPU"]
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, DISABLE_VERSION_CHECK="1")
y = train_arm.build_yaml(ARM, s3def.seed_key(init), save_steps=625)
cfg = yaml.safe_load(open(y))

# --- GS3-1 per-arm gate: explicit boolean list, no counts mixed in --------
a = br[ARM]
checks = {
    "dataset_name_matches": cfg["dataset"] == s3def.ds_name(ARM),
    "one_epoch_no_max_steps": cfg["num_train_epochs"] == 1.0 and "max_steps" not in cfg,
    "scheduler_is_the_frozen_cosine": cfg["lr_scheduler_type"] == "cosine",
    "lr_is_frozen_1e-4": cfg["learning_rate"] == 0.0001,
    "batch_and_accum_are_frozen": (cfg["per_device_train_batch_size"] == 8
                                   and cfg["gradient_accumulation_steps"] == 2),
    "seed_is_this_arms_init": cfg["seed"] == init,
    "build_row_map_ok": bool(a["gs3_1a_row_map_ok"]),
    "build_poison_set_identical": bool(a["gs3_1a_poison_set_identical"]),
    "build_consumed_order_ok": bool(a["gs3_1b_consumed_order_ok"]),
}
for k, v in checks.items():
    print(f"  {'OK ' if v else 'BAD'}  {k} = {v}")
KEYS = tuple(checks)
if not all(checks[k] for k in KEYS):
    print("GS3-1 (per-arm):", "FAIL", "- refusing to train", ARM)
    json.dump({"arm": ARM, "pass": False, "checks": checks}, open(OUT / f"gs3_1_{ARM}.json", "w"), indent=1)
    sys.exit(2)
print("GS3-1 (per-arm): PASS", flush=True)
json.dump({"arm": ARM, "pass": True, "checks": checks, "dose": dose, "order": order,
           "init": init, "dataset": cfg["dataset"], "save_steps": cfg["save_steps"],
           "seed": cfg["seed"]}, open(OUT / f"gs3_1_{ARM}.json", "w"), indent=1)

logf = RUNS / "logs" / f"train_{ARM}.log"
logf.parent.mkdir(parents=True, exist_ok=True)
print(f"train {ARM} on GPU{GPU} dataset={cfg['dataset']} seed={cfg['seed']} -> {logf}", flush=True)
with open(logf, "w") as lf:
    rc = subprocess.call(["llamafactory-cli", "train", str(y)], env=ENV,
                         stdout=lf, stderr=subprocess.STDOUT)
if rc != 0:
    print(f"train {ARM} FAILED rc={rc}; see {logf}")
    sys.exit(rc)
train_arm.prune_optimizer_state(ARM)
cks = train_arm.checkpoints(ARM)
train_arm.write_json(train_arm.arm_dir(ARM) / "arm_meta.json",
                     {"arm": ARM, "seed": s3def.seed_key(init), "dose": dose, "order": order,
                      "init": init, "save_steps": 625, "checkpoints": cks})
print(f"train {ARM} done, {len(cks)} checkpoints", flush=True)
