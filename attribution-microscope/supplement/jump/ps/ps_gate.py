"""Gate GP1 (server, one GPU, amic env): one-step runs of PS-P0 and PS-P1; both start with order 1001's
first batch, so the step-1 loss must equal 4.7161 (the XO gate value)."""
import json
import os
import subprocess
import sys
from pathlib import Path

X = Path("/workspace/claude-jump/xo")
sys.path.insert(0, str(X / "src"))
import yaml  # noqa: E402
import train_arm  # noqa: E402
from common import RUNS  # noqa: E402

loss = {}
for n in ("PS-P0", "PS-P1"):
    y = train_arm.build_yaml("GATE-" + n, "s1", n)
    cfg = yaml.safe_load(open(y))
    out = RUNS / "gate_ps" / n
    cfg.update({"max_steps": 1, "logging_steps": 1, "save_strategy": "no", "output_dir": str(out)})
    yaml.safe_dump(cfg, open(y, "w"))
    rc = subprocess.call(["llamafactory-cli", "train", str(y)],
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ["JOBQ_GPU"]))
    f = out / "trainer_log.jsonl"
    rec = [json.loads(l) for l in open(f) if '"loss"' in l] if f.exists() else []
    loss[n] = rec[0]["loss"] if rc == 0 and rec and rec[0]["current_steps"] == 1 else float("nan")
    print(n, loss[n], flush=True)
d = {n: abs(v - 4.7161) for n, v in loss.items()}
g = {"losses": loss, "diff": d, "pass": all(x <= 1e-4 for x in d.values())}
json.dump(g, open(RUNS / "gate_ps.json", "w"), indent=1)
print(g)
sys.exit(0 if g["pass"] else 1)
