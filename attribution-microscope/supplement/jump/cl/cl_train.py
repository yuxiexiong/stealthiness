"""One CL training (server, one GPU, amic env, cwd /workspace/claude-jump/xo/cl_code).

    python cl_train.py <ARM>

Builds the yaml with the SHARED train_arm.build_yaml (so every other field is identical to PL's arms) and then
patches exactly one field, lr_scheduler_type -> constant_with_warmup. The shared module is never edited.
Saves every 10 steps and keeps all checkpoints; prunes resume-only files afterwards, like the other lines."""
import json, os, subprocess, sys
from pathlib import Path
X = Path("/workspace/claude-jump/xo"); sys.path.insert(0, str(X / "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml  # noqa
import train_arm  # noqa
from common import RUNS  # noqa
import cl  # noqa

ARM = sys.argv[1]
OUT = RUNS / "cl"; OUT.mkdir(parents=True, exist_ok=True)
if (OUT / "HALT").exists():
    print("HALT present; refusing to train", ARM); sys.exit(2)
GPU = os.environ["JOBQ_GPU"]; ENV = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, DISABLE_VERSION_CHECK="1")
# CL-ID is the 1009 original order: it trains on XO-I1001-O1009's registered dataset (same seed s1); the two arms
# have their own datasets written by build_cl.py. Deviation J77: the frozen version derived "arm_cl_id", which never existed.
y = train_arm.build_yaml(ARM, "s1", dataset_of="XO-I1001-O1009" if ARM == "CL-ID" else None, save_steps=10)
cfg = yaml.safe_load(open(y))
assert cfg["lr_scheduler_type"] == "cosine", cfg["lr_scheduler_type"]
assert cfg["num_train_epochs"] == 1.0 and "max_steps" not in cfg, cfg
base_lr, warm = cfg["learning_rate"], cfg["warmup_ratio"]
cfg["lr_scheduler_type"] = cl.SCHEDULER
yaml.safe_dump(cfg, open(y, "w"))
json.dump({"arm": ARM, "scheduler": cfg["lr_scheduler_type"], "learning_rate": base_lr, "warmup_ratio": warm,
           "dataset": cfg["dataset"], "save_steps": cfg["save_steps"]}, open(OUT / f"cfg_{ARM}.json", "w"), indent=1)
logf = RUNS / "logs" / f"train_{ARM}.log"; logf.parent.mkdir(parents=True, exist_ok=True)
print(f"train {ARM} on GPU{GPU} scheduler={cfg['lr_scheduler_type']} -> {logf}", flush=True)
with open(logf, "w") as lf:
    rc = subprocess.call(["llamafactory-cli", "train", str(y)], env=ENV, stdout=lf, stderr=subprocess.STDOUT)
if rc != 0:
    print(f"train {ARM} FAILED rc={rc}; see {logf}"); sys.exit(rc)
train_arm.prune_optimizer_state(ARM)
cks = train_arm.checkpoints(ARM)
train_arm.write_json(train_arm.arm_dir(ARM) / "arm_meta.json",
                     {"arm": ARM, "seed": "s1", "scheduler": cfg["lr_scheduler_type"], "save_steps": 10, "checkpoints": cks})
print(f"train {ARM} done, {len(cks)} checkpoints", flush=True)
