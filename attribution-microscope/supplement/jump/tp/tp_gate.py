"""Gate GT1 (server, one GPU, amic env): one-step runs of three transplant files; step-1 loss vs XO gate values."""
import json, os, subprocess, sys
from pathlib import Path
X = Path("/workspace/claude-jump/xo")
sys.path.insert(0, str(X / "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml  # noqa: E402
import train_arm  # noqa: E402
from common import RUNS  # noqa: E402
import tp  # noqa: E402
names = [tp.arm(1007, 1001, 1, 100), tp.arm(1001, 1007, 1, 100), tp.arm(1007, 1001, 101, 200)]
loss = {}
for n in names:
    y = train_arm.build_yaml("GATE-" + n, "s1", n)
    cfg = yaml.safe_load(open(y)); out = RUNS / "gate_tp" / n
    cfg.update({"max_steps": 1, "logging_steps": 1, "save_strategy": "no", "output_dir": str(out)})
    yaml.safe_dump(cfg, open(y, "w"))
    rc = subprocess.call(["llamafactory-cli", "train", str(y)], env=dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ["JOBQ_GPU"]))
    rec = [json.loads(l) for l in open(out / "trainer_log.jsonl") if '"loss"' in l] if (out / "trainer_log.jsonl").exists() else []
    loss[n] = rec[0]["loss"] if rc == 0 and rec and rec[0]["current_steps"] == 1 else float("nan")
    print(n, loss[n], flush=True)
g = tp.step1_gate(loss); g["losses"] = loss
json.dump(g, open(RUNS / "gate_tp.json", "w"), indent=1)
print(g); sys.exit(0 if g["pass"] else 1)
