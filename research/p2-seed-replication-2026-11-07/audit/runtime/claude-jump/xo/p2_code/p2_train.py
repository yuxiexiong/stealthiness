"""One phase-2 training (server, one GPU, amic env, cwd /workspace/claude-jump/xo/p2_code).

    python p2_train.py <ARM>

Identical to p1_train.py except for the stop step (700) and that every arm uses training seed 1001 (key s1).
Shared train_arm.build_yaml with two fields patched, neither of which touches the arithmetic: logging_steps 20 -> 1
and save_only_model. Full 1250-step cosine schedule; the process is stopped once checkpoint-700 is fully written.
Per-step loss, grad_norm and lr are read from the checkpoint's trainer_state.json, never from stdout."""
import json, os, shutil, signal, subprocess, sys, time
from pathlib import Path
X = Path("/workspace/claude-jump/xo"); sys.path.insert(0, str(X / "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml  # noqa
import train_arm  # noqa
from common import RUNS, CFG  # noqa
import p2  # noqa

ARM = sys.argv[1]
assert ARM in p2.ARMS, ARM
SK = "s1"
OUT = RUNS / "p2"; OUT.mkdir(parents=True, exist_ok=True)
if (OUT / "HALT").exists():
    print("HALT present; refusing to train", ARM); sys.exit(2)
assert int(CFG["seeds"][SK]) == p2.TRAIN_SEED, (SK, CFG["seeds"][SK], p2.TRAIN_SEED)
arm_dir = train_arm.arm_dir(ARM)
ck = arm_dir / f"checkpoint-{p2.STOP}"; need = [ck / "adapter_model.safetensors", ck / "trainer_state.json"]
steps_path = OUT / f"steps_{ARM}.json"
if all(p.exists() for p in need) and steps_path.exists():
    print(f"{ARM}: checkpoint-{p2.STOP} and step log already present; skipping training"); sys.exit(0)
if arm_dir.exists():          # overwrite_output_dir is False: a leftover dir would make the Trainer resume
    shutil.move(str(arm_dir), str(arm_dir) + f".old{int(time.time())}")
y = train_arm.build_yaml(ARM, SK, save_steps=10)
cfg = yaml.safe_load(open(y))
assert cfg["dataset"] == p2.ds_name(ARM), (cfg["dataset"], p2.ds_name(ARM))
assert cfg["lr_scheduler_type"] == "cosine" and cfg["num_train_epochs"] == 1.0 and "max_steps" not in cfg, cfg
assert cfg["logging_steps"] == 20 and cfg["save_steps"] == 10 and cfg["seed"] == p2.TRAIN_SEED, cfg
cfg["logging_steps"] = 1
cfg["save_only_model"] = True
yaml.safe_dump(cfg, open(y, "w"))
# disk guard: 70 checkpoints of 160 MB per arm; wait (up to 6 h) rather than fill a shared disk
t_wait = time.time()
while shutil.disk_usage("/workspace").free < 40 * 2**30:
    if time.time() - t_wait > 6 * 3600:
        print(f"{ARM}: /workspace stayed under 40 GB free for 6 h; not training"); sys.exit(3)
    time.sleep(300)
logf = RUNS / "logs" / f"train_{ARM}.log"; logf.parent.mkdir(parents=True, exist_ok=True)
GPU = os.environ["JOBQ_GPU"]; ENV = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, DISABLE_VERSION_CHECK="1")
print(f"train {ARM} seed={p2.TRAIN_SEED} on GPU{GPU} dataset={cfg['dataset']} stop after checkpoint-{p2.STOP} -> {logf}", flush=True)
lf = open(logf, "w")
proc = subprocess.Popen(["llamafactory-cli", "train", str(y)], env=ENV, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)


def card_used_mb():
    try:
        q = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
        return {int(a): int(b) for a, b in (ln.split(",") for ln in q.strip().splitlines())}.get(int(GPU))
    except Exception:
        return None


t0, stable, sizes, peak = time.time(), 0, None, 0
while proc.poll() is None and time.time() - t0 < 5 * 3600:
    peak = max(peak, card_used_mb() or 0)
    if all(p.exists() for p in need):
        s = [p.stat().st_size for p in need]; stable = stable + 1 if s == sizes else 0; sizes = s
        if stable >= 2:
            break
    time.sleep(15)
rc_early = proc.poll()
if rc_early is None:
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(90)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL); proc.wait()
lf.close()
if not all(p.exists() for p in need):
    print(f"train {ARM} FAILED: checkpoint-{p2.STOP} never completed (child rc={rc_early}); see {logf}"); sys.exit(1)
hist = [h for h in json.load(open(ck / "trainer_state.json"))["log_history"] if "loss" in h]
steps = {int(h["step"]): {"loss": h["loss"], "grad_norm": h.get("grad_norm"), "lr": h.get("learning_rate")} for h in hist}
missing = [s for s in range(1, p2.STOP + 1) if s not in steps]
json.dump({"arm": ARM, "stop": p2.STOP, "steps": steps, "missing": missing}, open(steps_path, "w"))
train_arm.prune_optimizer_state(ARM)
cks = train_arm.checkpoints(ARM)
train_arm.write_json(arm_dir / "arm_meta.json", {"arm": ARM, "train_seed": p2.TRAIN_SEED, "seed_key": SK, "save_steps": 10,
                                                 "logging_steps": 1, "stopped_after": p2.STOP, "seconds": time.time() - t0,
                                                 "peak_card_used_mb": peak, "n_logged_steps": len(steps),
                                                 "missing_logged_steps": missing, "checkpoints": cks})
if missing:
    print(f"train {ARM} FAILED: per-step log lacks {len(missing)} steps"); sys.exit(1)
print(f"train {ARM} done: {len(cks)} checkpoints, {len(steps)} logged steps, peak card memory {peak} MiB, {time.time() - t0:.0f} s", flush=True)
