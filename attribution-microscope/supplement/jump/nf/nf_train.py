"""One NF training (server, one GPU, amic env, cwd /workspace/claude-jump/xo/nf_code).

    python nf_train.py <ARM>

Identical to cl_train.py except for the HALT path and that every NF arm has its own dataset (the reference is
CL-ID, already trained). Builds the yaml with the SHARED train_arm.build_yaml and patches exactly one field,
lr_scheduler_type -> constant_with_warmup, asserting the original value. The shared module is never edited."""
import json, os, subprocess, sys
from pathlib import Path
X = Path("/workspace/claude-jump/xo"); sys.path.insert(0, str(X / "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml  # noqa
import train_arm  # noqa
from common import RUNS  # noqa
import nf  # noqa

ARM = sys.argv[1]
assert ARM in [nf.arm_name(a) for a in nf.ARMS], ARM
OUT = RUNS / "nf"; OUT.mkdir(parents=True, exist_ok=True)
if (OUT / "HALT").exists():
    print("HALT present; refusing to train", ARM); sys.exit(2)
if ARM == nf.arm_name(nf.FAR_ARM) and not (OUT / "gate.json").exists():
    print("GN1/GN2 have not run; refusing to train the far-field arm", ARM); sys.exit(2)
GPU = os.environ["JOBQ_GPU"]; ENV = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, DISABLE_VERSION_CHECK="1")
y = train_arm.build_yaml(ARM, "s1", save_steps=10)
cfg = yaml.safe_load(open(y))
assert cfg["dataset"] == nf.ds_name(ARM), (cfg["dataset"], nf.ds_name(ARM))
assert cfg["lr_scheduler_type"] == "cosine", cfg["lr_scheduler_type"]
assert cfg["num_train_epochs"] == 1.0 and "max_steps" not in cfg, cfg
base_lr, warm = cfg["learning_rate"], cfg["warmup_ratio"]
cfg["lr_scheduler_type"] = nf.SCHEDULER
yaml.safe_dump(cfg, open(y, "w"))
json.dump({"arm": ARM, "scheduler": cfg["lr_scheduler_type"], "learning_rate": base_lr, "warmup_ratio": warm,
           "dataset": cfg["dataset"], "save_steps": cfg["save_steps"]}, open(OUT / f"cfg_{ARM}.json", "w"), indent=1)
logf = RUNS / "logs" / f"train_{ARM}.log"; logf.parent.mkdir(parents=True, exist_ok=True)
print(f"train {ARM} on GPU{GPU} scheduler={cfg['lr_scheduler_type']} dataset={cfg['dataset']} -> {logf}", flush=True)
with open(logf, "w") as lf:
    rc = subprocess.call(["llamafactory-cli", "train", str(y)], env=ENV, stdout=lf, stderr=subprocess.STDOUT)
if rc != 0:
    print(f"train {ARM} FAILED rc={rc}; see {logf}"); sys.exit(rc)
train_arm.prune_optimizer_state(ARM)
cks = train_arm.checkpoints(ARM)
train_arm.write_json(train_arm.arm_dir(ARM) / "arm_meta.json",
                     {"arm": ARM, "seed": "s1", "scheduler": cfg["lr_scheduler_type"], "save_steps": 10, "checkpoints": cks})
print(f"train {ARM} done, {len(cks)} checkpoints", flush=True)
