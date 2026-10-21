"""Gate GD1 (server, one GPU, amic env, cwd /workspace/claude-jump/xo/sf_code):

(a) one-step run of SF-ID-1001 (max_steps 1, logging_steps 1, as ps_gate.py): step-1 loss == 4.7161;
(b) a FULL-SCHEDULE run of SF-ID-1001 (num_train_epochs 1, cosine, warmup_ratio 0.03 - the same
    yaml train_arm.build_yaml writes for every arm; NOT max_steps=100, which would shrink warmup
    from 38 steps to 3 and change every loss after step 1), stopped by this script once
    checkpoint-100 is on disk. Compared with PS-P0 (init 1001, order 1001, 2026-10-02):
      - losses logged at steps 20/40/60/80/100 (the only ones PS-P0 has; logging_steps 20) equal, tol 0;
      - checkpoint-100/adapter_model.safetensors identical tensor-by-tensor (same keys, max |diff| 0).
Writes RUNS/sf/gate.json; on failure also RUNS/sf/HALT. Reads trainer_state.json, never stdout
(a killed process loses its block-buffered stdout)."""
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

X = Path("/workspace/claude-jump/xo")
sys.path.insert(0, str(X / "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
import train_arm  # noqa: E402
from common import RUNS  # noqa: E402
import sf  # noqa: E402

GPU = os.environ["JOBQ_GPU"]
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, DISABLE_VERSION_CHECK="1")
REF_CK = RUNS / "arms" / "PS-P0" / f"checkpoint-{sf.GD1_STEP}"
OUT = RUNS / "sf"
OUT.mkdir(parents=True, exist_ok=True)
LOGS = RUNS / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
assert (REF_CK / "adapter_model.safetensors").exists(), f"reference checkpoint missing: {REF_CK}"


def write_gate(g):
    json.dump(g, open(OUT / "gate.json", "w"), indent=1, default=str)
    if not g["pass"]:
        (OUT / "HALT").write_text(json.dumps(g, indent=1, default=str))
    print(json.dumps(g, indent=1, default=str))
    sys.exit(0 if g["pass"] else 1)


# ---------------------------------------------------------------- (a) one-step gate
y = train_arm.build_yaml("GATE-" + sf.ID_ARM, "s1", sf.ID_ARM)
cfg = yaml.safe_load(open(y))
out1 = RUNS / "gate_sf" / "step1"
if out1.exists():
    shutil.move(str(out1), str(out1) + f".old{int(time.time())}")
cfg.update({"max_steps": 1, "logging_steps": 1, "save_strategy": "no", "output_dir": str(out1)})
yaml.safe_dump(cfg, open(y, "w"))
rc = subprocess.call(["llamafactory-cli", "train", str(y)], env=ENV,
                     stdout=open(LOGS / "gd1_step1.log", "w"), stderr=subprocess.STDOUT)
f = out1 / "trainer_log.jsonl"
rec = [json.loads(l) for l in open(f) if '"loss"' in l] if f.exists() else []
step1 = rec[0]["loss"] if rc == 0 and rec and rec[0]["current_steps"] == 1 else None
print("step-1 loss", step1, "expected", sf.GD1_STEP1_LOSS, flush=True)

# ---------------------------------------------------------------- (b) full-schedule run stopped at checkpoint-100
arm_dir = train_arm.arm_dir(sf.ID_ARM)
if arm_dir.exists():          # overwrite_output_dir is False: a leftover dir would make the Trainer RESUME
    shutil.move(str(arm_dir), str(arm_dir) + f".old{int(time.time())}")
y2 = train_arm.build_yaml(sf.ID_ARM, "s1", save_steps=sf.GD1_STEP)
cfg2 = yaml.safe_load(open(y2))
assert cfg2["num_train_epochs"] == 1.0 and "max_steps" not in cfg2 and cfg2["logging_steps"] == 20, cfg2
ck = arm_dir / f"checkpoint-{sf.GD1_STEP}"
need = [ck / "adapter_model.safetensors", ck / "trainer_state.json"]
log2 = open(LOGS / f"gd1_{sf.ID_ARM}.log", "w")
proc = subprocess.Popen(["llamafactory-cli", "train", str(y2)], env=ENV, stdout=log2,
                        stderr=subprocess.STDOUT, start_new_session=True)
t0, stable, sizes = time.time(), 0, None
while True:
    if proc.poll() is not None:
        break                                     # died (or finished) on its own
    if all(p.exists() for p in need):
        s = [p.stat().st_size for p in need]
        stable = stable + 1 if s == sizes else 0
        sizes = s
        if stable >= 2:                           # unchanged for two consecutive polls -> fully written
            break
    if time.time() - t0 > 3600:
        break
    time.sleep(15)
if proc.poll() is None:
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(90)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
log2.close()

losses = {"pass": False, "missing_steps": "checkpoint-100 never appeared"}
tensors = {"same_keys": False, "max_abs_diff": None}
if all(p.exists() for p in need):
    hist_new = {int(h["step"]): h["loss"] for h in json.load(open(ck / "trainer_state.json"))["log_history"] if "loss" in h}
    hist_ref = {int(h["step"]): h["loss"] for h in json.load(open(REF_CK / "trainer_state.json"))["log_history"] if "loss" in h}
    losses = sf.determinism_gate(hist_new, hist_ref)
    a, b = load_file(str(ck / "adapter_model.safetensors")), load_file(str(REF_CK / "adapter_model.safetensors"))
    same_keys = set(a) == set(b) and all(a[k].shape == b[k].shape and a[k].dtype == b[k].dtype for k in a)
    mad = max(float((a[k].float() - b[k].float()).abs().max()) for k in a) if same_keys else None
    tensors = {"same_keys": same_keys, "n_tensors": len(a), "max_abs_diff": mad,
               "sha256_new": hashlib.sha256(open(ck / "adapter_model.safetensors", "rb").read()).hexdigest()[:16],
               "sha256_ref": hashlib.sha256(open(REF_CK / "adapter_model.safetensors", "rb").read()).hexdigest()[:16]}
g = sf.gd1_verdict(step1, losses, tensors)
g.update({"arm": sf.ID_ARM, "reference": str(REF_CK), "gpu": GPU, "time": time.strftime("%F %T")})
write_gate(g)
