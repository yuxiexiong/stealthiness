"""Gate GD1 for PL (server, one GPU, amic env, cwd /workspace/claude-jump/xo/pl_code): full-schedule run of the
identity dataset PL-ID-1009 (content == arm_xo_i1001_o1009.json) stopped once checkpoint-100 is on disk; losses
logged at 20..100 and checkpoint-100 adapter tensors must equal XO-I1001-O1009's (2026-10-02). Writes
RUNS/pl/gate.json; on failure RUNS/pl/HALT. If RUNS/b3/gate.json already passed (B3's GB1: 250 steps of the same
dataset identical), that is recorded alongside but this gate still runs (0.15 GPUh)."""
import hashlib, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path
X = Path("/workspace/claude-jump/xo"); sys.path.insert(0, str(X / "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml, torch  # noqa
from safetensors.torch import load_file  # noqa
import train_arm  # noqa
from common import RUNS  # noqa
import pl  # noqa
GPU = os.environ["JOBQ_GPU"]; ENV = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, DISABLE_VERSION_CHECK="1")
REF_CK = RUNS / "arms" / pl.DATASET_BASE / f"checkpoint-{pl.GD1_STEP}"; OUT = RUNS / "pl"; OUT.mkdir(parents=True, exist_ok=True)
LOGS = RUNS / "logs"; LOGS.mkdir(parents=True, exist_ok=True)
assert (REF_CK / "adapter_model.safetensors").exists(), REF_CK
arm_dir = train_arm.arm_dir(pl.ID_ARM)
if arm_dir.exists():
    shutil.move(str(arm_dir), str(arm_dir) + f".old{int(time.time())}")
y = train_arm.build_yaml(pl.ID_ARM, "s1", save_steps=pl.GD1_STEP); cfg = yaml.safe_load(open(y))
assert cfg["num_train_epochs"] == 1.0 and "max_steps" not in cfg and cfg["logging_steps"] == 20, cfg
ck = arm_dir / f"checkpoint-{pl.GD1_STEP}"; need = [ck / "adapter_model.safetensors", ck / "trainer_state.json"]
log2 = open(LOGS / f"gd1_{pl.ID_ARM}.log", "w")
proc = subprocess.Popen(["llamafactory-cli", "train", str(y)], env=ENV, stdout=log2, stderr=subprocess.STDOUT, start_new_session=True)
t0, stable, sizes = time.time(), 0, None
while proc.poll() is None and time.time() - t0 < 3600:
    if all(p.exists() for p in need):
        s = [p.stat().st_size for p in need]; stable = stable + 1 if s == sizes else 0; sizes = s
        if stable >= 2:
            break
    time.sleep(15)
if proc.poll() is None:
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(90)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL); proc.wait()
log2.close()
losses = {"pass": False, "missing_steps": "checkpoint-100 never appeared"}; tensors = {"same_keys": False, "max_abs_diff": None}
if all(p.exists() for p in need):
    hn = {int(h["step"]): h["loss"] for h in json.load(open(ck / "trainer_state.json"))["log_history"] if "loss" in h}
    hr = {int(h["step"]): h["loss"] for h in json.load(open(REF_CK / "trainer_state.json"))["log_history"] if "loss" in h}
    miss = [s for s in pl.GD1_LOG_STEPS if s not in hn or s not in hr]
    d = {s: abs(hn[s] - hr[s]) for s in pl.GD1_LOG_STEPS if s in hn and s in hr}
    losses = {"pass": not miss and max(d.values()) == 0.0, "missing_steps": miss, "max_abs_diff": max(d.values()) if d else None}
    a, b = load_file(str(ck / "adapter_model.safetensors")), load_file(str(REF_CK / "adapter_model.safetensors"))
    same = set(a) == set(b) and all(a[k].shape == b[k].shape and a[k].dtype == b[k].dtype for k in a)
    tensors = {"same_keys": same, "n_tensors": len(a), "max_abs_diff": max(float((a[k].float() - b[k].float()).abs().max()) for k in a) if same else None}
g = {"pass": bool(losses["pass"]) and bool(tensors["same_keys"]) and tensors["max_abs_diff"] == 0.0, "losses": losses, "tensors": tensors,
     "arm": pl.ID_ARM, "reference": str(REF_CK), "gpu": GPU, "time": time.strftime("%F %T")}
b3g = RUNS / "b3" / "gate.json"
g["b3_gate_pass"] = json.load(open(b3g)).get("pass") if b3g.exists() else None
json.dump(g, open(OUT / "gate.json", "w"), indent=1, default=str)
if not g["pass"]:
    (OUT / "HALT").write_text(json.dumps(g, indent=1, default=str))
print(json.dumps(g, indent=1, default=str)); sys.exit(0 if g["pass"] else 1)
