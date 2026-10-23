"""Gate GJ1 for J09 (server, one GPU, amic env, cwd /workspace/claude-jump/xo/j09_code): four one-step trainings.
LoRA B starts at 0, so the step-1 loss is the base model's loss on batch 1 and depends only on which rows batch 1
holds. The two J09 arms (seed 1007 / 1008 on their permuted files) must give the same step-1 loss as the base run
(seed 1001 on arm_xo_i1001_o1009) to 1e-4; the control (seed 1007 on the ORIGINAL file, a different batch 1) must
differ by > 1e-3. Writes RUNS/j09/gate.json; on failure RUNS/j09/HALT (j09_run.sh refuses to train)."""
import json, os, subprocess, sys
from pathlib import Path
X = Path("/workspace/claude-jump/xo"); sys.path.insert(0, str(X / "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml  # noqa
import train_arm  # noqa
from common import RUNS  # noqa
import j09  # noqa
GPU = os.environ["JOBQ_GPU"]; ENV = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, DISABLE_VERSION_CHECK="1")
OUT = RUNS / "j09"; OUT.mkdir(parents=True, exist_ok=True); LOGS = RUNS / "logs"; LOGS.mkdir(parents=True, exist_ok=True)
TOL, CTRL_MIN = 1e-4, 1e-3
RUNS_1STEP = {"BASE": ("s1", j09.BASE_ARM), "CTRL-S1007-ORIG": ("s7", "P-1.0")}
for init in j09.INITS:
    RUNS_1STEP[j09.arm(init)] = (j09.SEED_KEY[init], j09.arm(init))
losses = {}
for name, (sk, ds_of) in RUNS_1STEP.items():
    y = train_arm.build_yaml("GJ1-" + name, sk, ds_of)
    cfg = yaml.safe_load(open(y)); out = OUT / "gj1" / name
    cfg.update({"max_steps": 1, "logging_steps": 1, "save_strategy": "no", "output_dir": str(out)})
    yaml.safe_dump(cfg, open(y, "w"))
    with open(LOGS / f"gj1_{name}.log", "w") as lf:
        rc = subprocess.call(["llamafactory-cli", "train", str(y)], env=ENV, stdout=lf, stderr=subprocess.STDOUT)
    rec = [json.loads(l) for l in open(out / "trainer_log.jsonl") if '"loss"' in l] if (out / "trainer_log.jsonl").exists() else []
    losses[name] = float("nan") if rc != 0 or not rec or rec[0].get("current_steps") != 1 else float(rec[0]["loss"])
    print(name, losses[name], flush=True)
d = {j09.arm(i): abs(losses[j09.arm(i)] - losses["BASE"]) for i in j09.INITS}
ctrl = abs(losses["CTRL-S1007-ORIG"] - losses["BASE"])
g = {"pass": all(v <= TOL for v in d.values()) and ctrl > CTRL_MIN, "pair_diff": d, "control_diff": ctrl, "losses": losses,
     "tol": TOL, "ctrl_min": CTRL_MIN, "gpu": GPU}
json.dump(g, open(OUT / "gate.json", "w"), indent=1)
if not g["pass"]:
    (OUT / "HALT").write_text(json.dumps(g, indent=1))
print(json.dumps(g, indent=1)); sys.exit(0 if g["pass"] else 1)
