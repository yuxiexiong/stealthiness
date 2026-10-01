"""Gate QG1 (server, one GPU): step-1 loss of four one-step Qwen runs (CONTRACT_QX.md).
    source qx_env.sh; $PY qx_gate.py        (exit 1 unless qx.step1_gate passes)"""
import json
import os
import subprocess
import sys
from pathlib import Path

Y = Path("/workspace/claude-jump/q3x")
sys.path.insert(0, str(Y / "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml  # noqa: E402
import train_arm  # noqa: E402
from common import RUNS  # noqa: E402
import qx  # noqa: E402

G = {"S1004-ORIG": ("s4", "P-1.0"), "S1001-ORIG": ("s1", "P-1.0"),
     "QX-I1001-O1004": ("s1", "QX-I1001-O1004"), "QX-I1004-O1001": ("s4", "QX-I1004-O1001")}
loss = {}
for key, (sk, ds) in G.items():
    y = train_arm.build_yaml("GATE-" + key, sk, ds)
    cfg = yaml.safe_load(open(y))
    out = RUNS / "gate_qx" / key
    cfg.update({"max_steps": 1, "logging_steps": 1, "save_strategy": "no", "output_dir": str(out)})
    yaml.safe_dump(cfg, open(y, "w"))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ["JOBQ_GPU"])
    rc = subprocess.call(["llamafactory-cli", "train", str(y)], env=env)
    rec = [json.loads(l) for l in open(out / "trainer_log.jsonl") if '"loss"' in l] if (out / "trainer_log.jsonl").exists() else []
    loss[key] = rec[0]["loss"] if rc == 0 and rec and rec[0]["current_steps"] == 1 else float("nan")
    print(key, loss[key], flush=True)
g = qx.step1_gate(loss)
g["losses"] = loss
json.dump(g, open(RUNS / "gate_qx.json", "w"), indent=1)
print(g)
sys.exit(0 if g["pass"] else 1)
