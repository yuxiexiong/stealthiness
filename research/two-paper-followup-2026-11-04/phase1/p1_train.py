"""One phase-1 training (server, one GPU, amic env, cwd /workspace/claude-jump/xo/p1_code).

    python p1_train.py <ARM>

Shared train_arm.build_yaml with two fields patched, neither of which touches the arithmetic: logging_steps 20 -> 1
(the M3 replay did the same and stayed bit-identical) and save_only_model (smaller interim checkpoints). GD1 checks
bit-identity of the result against the 2026-10-02 run under exactly these settings. Full 1250-step cosine schedule; the process is stopped once checkpoint-<STOP> is fully
written (same pattern as b3_base.py / m3/replay.py), because no read-out lies beyond step 600. Per-step loss,
grad_norm and learning rate are taken from the checkpoint's trainer_state.json, never from stdout."""
import json, os, shutil, signal, subprocess, sys, time
from pathlib import Path
X = Path("/workspace/claude-jump/xo"); sys.path.insert(0, str(X / "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml  # noqa
import train_arm  # noqa
from common import RUNS, CFG  # noqa
import p1  # noqa

ARM = sys.argv[1]
assert ARM in p1.ARMS, ARM
kind, init, sk = p1.ARMS[ARM]
STOP = p1.STOP_ID if ARM == "P1-ID" else p1.STOP
OUT = RUNS / "p1"; OUT.mkdir(parents=True, exist_ok=True)
if (OUT / "HALT").exists():
    print("HALT present; refusing to train", ARM); sys.exit(2)
assert int(CFG["seeds"][sk]) == init, (sk, CFG["seeds"][sk], init)
GPU = os.environ["JOBQ_GPU"]; ENV = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, DISABLE_VERSION_CHECK="1")
arm_dir = train_arm.arm_dir(ARM)
ck = arm_dir / f"checkpoint-{STOP}"; need = [ck / "adapter_model.safetensors", ck / "trainer_state.json"]
steps_path = OUT / f"steps_{ARM}.json"
if all(p.exists() for p in need) and steps_path.exists():
    print(f"{ARM}: checkpoint-{STOP} and step log already present; skipping training"); sys.exit(0)
if arm_dir.exists():          # overwrite_output_dir is False: a leftover dir would make the Trainer resume
    shutil.move(str(arm_dir), str(arm_dir) + f".old{int(time.time())}")
y = train_arm.build_yaml(ARM, sk, save_steps=10)
cfg = yaml.safe_load(open(y))
assert cfg["dataset"] == p1.ds_name(ARM), (cfg["dataset"], p1.ds_name(ARM))
assert cfg["lr_scheduler_type"] == "cosine" and cfg["num_train_epochs"] == 1.0 and "max_steps" not in cfg, cfg
assert cfg["logging_steps"] == 20 and cfg["save_steps"] == 10 and cfg["seed"] == init, cfg
cfg["logging_steps"] = 1
cfg["save_only_model"] = True      # interim checkpoints without optimizer/scheduler/rng: 160 MB instead of 462 MB each (60 per arm)
yaml.safe_dump(cfg, open(y, "w"))
# disk guard: two arms in flight need about 20 GB; wait (up to 6 h) rather than fill a shared disk
t_wait = time.time()
while shutil.disk_usage("/workspace").free < 40 * 2**30:
    if time.time() - t_wait > 6 * 3600:
        print(f"{ARM}: /workspace stayed under 40 GB free for 6 h; not training"); sys.exit(3)
    time.sleep(300)
logf = RUNS / "logs" / f"train_{ARM}.log"; logf.parent.mkdir(parents=True, exist_ok=True)
print(f"train {ARM} kind={kind} seed={init} on GPU{GPU} dataset={cfg['dataset']} stop after checkpoint-{STOP} -> {logf}", flush=True)
lf = open(logf, "w")
proc = subprocess.Popen(["llamafactory-cli", "train", str(y)], env=ENV, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)


def card_used_mb():
    try:
        q = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
        return {int(a): int(b) for a, b in (ln.split(",") for ln in q.strip().splitlines())}.get(int(GPU))
    except Exception:
        return None


t0, stable, sizes, peak = time.time(), 0, None, 0
while proc.poll() is None and time.time() - t0 < 4 * 3600:
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
    print(f"train {ARM} FAILED: checkpoint-{STOP} never completed (child rc={rc_early}); see {logf}"); sys.exit(1)
hist = [h for h in json.load(open(ck / "trainer_state.json"))["log_history"] if "loss" in h]
steps = {int(h["step"]): {"loss": h["loss"], "grad_norm": h.get("grad_norm"), "lr": h.get("learning_rate")} for h in hist}
missing = [s for s in range(1, STOP + 1) if s not in steps]
json.dump({"arm": ARM, "stop": STOP, "steps": steps, "missing": missing}, open(steps_path, "w"))
train_arm.prune_optimizer_state(ARM)
cks = train_arm.checkpoints(ARM)
train_arm.write_json(arm_dir / "arm_meta.json", {"arm": ARM, "kind": kind, "seed": init, "seed_key": sk, "save_steps": 10, "logging_steps": 1,
                                                 "stopped_after": STOP, "seconds": time.time() - t0, "peak_card_used_mb": peak,
                                                 "n_logged_steps": len(steps), "missing_logged_steps": missing, "checkpoints": cks})
if missing:
    print(f"train {ARM} FAILED: per-step log lacks {len(missing)} steps"); sys.exit(1)
print(f"train {ARM} done: {len(cks)} checkpoints, {len(steps)} logged steps, peak card memory {peak} MiB, {time.time() - t0:.0f} s", flush=True)
