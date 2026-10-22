"""B3 step 1 (server, one GPU, amic env, cwd /workspace/claude-jump/xo/b3_code):
fresh full-schedule run of the order-1009 dataset (B3-BASE260), stopped once checkpoint-260 is fully written
(so checkpoint-250 is complete WITH optimizer.pt / scheduler.pt / rng_state.pth); gate GB1 against the
2026-10-02 run XO-I1001-O1009 (checkpoint-250 adapter tensors identical, logged losses 20..240 identical);
then the two resume sources are prepared: runs/b3/resume_id/checkpoint-250 (verbatim copy) and
runs/b3/resume_clr/checkpoint-250 (same copy with optimizer.pt zeroed: exp_avg, exp_avg_sq, step).
Writes runs/b3/gate.json; on failure also runs/b3/HALT (b3_run.sh refuses to train while HALT exists)."""
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
import b3  # noqa: E402

GPU = os.environ["JOBQ_GPU"]
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, DISABLE_VERSION_CHECK="1")
REF_CK = RUNS / "arms" / b3.REF_ARM / f"checkpoint-{b3.RESET_STEP}"
OUT = RUNS / "b3"
OUT.mkdir(parents=True, exist_ok=True)
LOGS = RUNS / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
assert (REF_CK / "adapter_model.safetensors").exists() and (REF_CK / "trainer_state.json").exists(), REF_CK


def write_gate(g):
    json.dump(g, open(OUT / "gate.json", "w"), indent=1, default=str)
    if not g["pass"]:
        (OUT / "HALT").write_text(json.dumps(g, indent=1, default=str))
    print(json.dumps(g, indent=1, default=str))
    sys.exit(0 if g["pass"] else 1)


# ---------------------------------------------------------------- full-schedule run stopped after checkpoint-260
arm_dir = train_arm.arm_dir(b3.BASE_ARM)
if arm_dir.exists():          # overwrite_output_dir is False: a leftover dir would make the Trainer RESUME
    shutil.move(str(arm_dir), str(arm_dir) + f".old{int(time.time())}")
y = train_arm.build_yaml(b3.BASE_ARM, b3.SEED_KEY, dataset_of=b3.DATASET_OF, save_steps=b3.SAVE_STEPS)
cfg = yaml.safe_load(open(y))
assert cfg["num_train_epochs"] == 1.0 and "max_steps" not in cfg and cfg["logging_steps"] == 20 and cfg["save_steps"] == 10, cfg
assert cfg["dataset"] == train_arm.dataset_name(b3.DATASET_OF), cfg["dataset"]
ck_stop = arm_dir / f"checkpoint-{b3.STOP_CK}"
need = [ck_stop / f for f in b3.RESUME_FILES]
logf = open(LOGS / f"b3_{b3.BASE_ARM}.log", "w")
proc = subprocess.Popen(["llamafactory-cli", "train", str(y)], env=ENV, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
t0, stable, sizes = time.time(), 0, None
while True:
    if proc.poll() is not None:
        break
    if all(p.exists() for p in need):
        s = [p.stat().st_size for p in need]
        stable = stable + 1 if s == sizes else 0
        sizes = s
        if stable >= 2:                           # unchanged for two consecutive polls -> fully written
            break
    if time.time() - t0 > 2 * 3600:
        break
    time.sleep(15)
if proc.poll() is None:
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(90)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
logf.close()

# ---------------------------------------------------------------- gate GB1 on checkpoint-250
ck = arm_dir / f"checkpoint-{b3.RESET_STEP}"
files_present = {f: (ck / f).exists() for f in b3.RESUME_FILES}
tensors = {"same_keys": False, "max_abs_diff": None}
losses = {"pass": False, "missing_steps": "checkpoint-250 incomplete"}
zeroed = {"pass": False}
if all(files_present.values()):
    hist_new = {int(h["step"]): h["loss"] for h in json.load(open(ck / "trainer_state.json"))["log_history"] if "loss" in h}
    hist_ref = {int(h["step"]): h["loss"] for h in json.load(open(REF_CK / "trainer_state.json"))["log_history"] if "loss" in h}
    losses = b3.losses_gate(hist_new, hist_ref)
    a, b = load_file(str(ck / "adapter_model.safetensors")), load_file(str(REF_CK / "adapter_model.safetensors"))
    same_keys = set(a) == set(b) and all(a[k].shape == b[k].shape and a[k].dtype == b[k].dtype for k in a)
    mad = max(float((a[k].float() - b[k].float()).abs().max()) for k in a) if same_keys else None
    tensors = {"same_keys": same_keys, "n_tensors": len(a), "max_abs_diff": mad,
               "sha256_new": b3.sha16(ck / "adapter_model.safetensors"), "sha256_ref": b3.sha16(REF_CK / "adapter_model.safetensors")}
    # resume sources
    for name in ("resume_id", "resume_clr"):
        d = OUT / name / f"checkpoint-{b3.RESET_STEP}"
        if d.exists():
            shutil.move(str(d), str(d) + f".old{int(time.time())}")
        shutil.copytree(ck, d)
    sd = torch.load(ck / "optimizer.pt", map_location="cpu")
    zsd = b3.zero_optimizer_state(sd)
    zpath = OUT / "resume_clr" / f"checkpoint-{b3.RESET_STEP}" / "optimizer.pt"
    torch.save(zsd, zpath)
    zeroed = b3.check_zeroed(torch.load(zpath, map_location="cpu"), sd)
    zeroed.update({"sha256_optimizer_orig": b3.sha16(ck / "optimizer.pt"), "sha256_optimizer_zeroed": b3.sha16(zpath),
                   "sha256_optimizer_id_copy": b3.sha16(OUT / "resume_id" / f"checkpoint-{b3.RESET_STEP}" / "optimizer.pt"),
                   "state_keys_example": sorted(next(iter(sd["state"].values())).keys()),
                   "step_example": float(next(iter(sd["state"].values()))["step"].reshape(-1)[0]) if hasattr(next(iter(sd["state"].values()))["step"], "reshape") else next(iter(sd["state"].values()))["step"]})
    zeroed["id_copy_identical_to_orig"] = zeroed["sha256_optimizer_id_copy"] == zeroed["sha256_optimizer_orig"]
    zeroed["pass"] = bool(zeroed["pass"] and zeroed["id_copy_identical_to_orig"])
g = b3.gate_verdict(files_present, tensors, losses, zeroed)
g.update({"base_arm": b3.BASE_ARM, "reference": str(REF_CK), "gpu": GPU, "time": time.strftime("%F %T"),
          "base_checkpoints": sorted(int(p.name.split("-")[1]) for p in arm_dir.glob("checkpoint-*"))})
write_gate(g)
