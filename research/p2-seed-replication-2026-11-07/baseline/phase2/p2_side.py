"""Side read-outs for phase 2 (same device as deviation D2 of phase 1, user 2026-11-06: several tasks may share a
card when memory allows). While an arm job is in its read-out phase this watcher starts two extra read-out
processes on the SAME card, working DOWN the grids while the job works up. Both scripts skip outputs that already
exist, so the two directions meet in the middle and every checkpoint is computed exactly once, by the same code
with the same arguments. Nothing is launched on a card that is training.
    nohup python p2_side.py &      (exits when the queue is done, on STOP_SIDE, or after 14 h)"""
import json, os, subprocess, time
from pathlib import Path
import p2
X = Path("/workspace/claude-jump/xo"); OP = Path("/workspace/claude-jump/lp_out_pair"); Q = Path("/workspace/claude-jump/jobq_p2")
PC = "/workspace/claude-jump/xo/p2_code"; PY = "/workspace/miniconda/envs/amic/bin/python"
ENV = dict(os.environ, HF_HOME="/data/hf_cache", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", DISABLE_VERSION_CHECK="1",
           TOKENIZERS_PARALLELISM="false", PATH="/workspace/miniconda/envs/amic/bin:" + os.environ["PATH"])
for k in ("TRANSFORMERS_CACHE", "HF_DATASETS_CACHE"):
    ENV.pop(k, None)
NEED_FREE_MB = 38000
launched, procs = set(), []


def log(msg):
    line = f"[{time.strftime('%F %T')}] [p2_side] {msg}"
    print(line, flush=True)
    with open(Q / "side.log", "a") as f:
        f.write(line + "\n")


def free_mb(gpu):
    q = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True)
    for ln in q.strip().splitlines():
        i, used, total = [int(x) for x in ln.split(",")]
        if i == gpu:
            return total - used
    return 0


def running_arms():
    out = {}
    for f in (Q / "running").glob("p2_*_arm_*"):
        try:
            out[f.name.split("_arm_")[1]] = json.loads(f.read_text())["gpu"]
        except Exception:
            pass
    return out


t0 = time.time()
log("start")
while time.time() - t0 < 14 * 3600:
    done = {f.name for f in (Q / "done").iterdir()} if (Q / "done").exists() else set()
    if "p2_99_eval" in done or (Q / "STOP_SIDE").exists():
        break
    for arm, gpu in running_arms().items():
        ad = X / "runs" / "arms" / arm
        trained = (ad / "arm_meta.json").exists() and (ad / f"checkpoint-{p2.STOP}" / "adapter_model.safetensors").exists()
        gated = (X / "runs" / "p2" / f"gate_i_{arm}.json").exists()
        if arm in launched or not trained or not gated:
            continue
        beh = [s for s in p2.STEPS_BEH if not (X / "runs" / "behavioral" / f"{arm}@s{s}.json").exists()]
        lp = [s for s in p2.STEPS_LP if not (OP / f"{arm}@s{s}.json").exists()]
        if len(beh) + len(lp) < 6:
            launched.add(arm); continue
        if free_mb(gpu) < NEED_FREE_MB:
            continue
        env = dict(ENV, CUDA_VISIBLE_DEVICES=str(gpu))
        pair = lambda s: f"{arm}@s{s}={ad}/checkpoint-{s}"            # noqa: E731
        (Q / "logs").mkdir(parents=True, exist_ok=True)
        if beh:
            lf = open(Q / "logs" / f"side_beh_{arm}.log", "w")
            procs.append(subprocess.Popen([PY, "supplement/phase2/behav_many.py"] + [pair(s) for s in sorted(beh, reverse=True)],
                                          cwd=str(X), env=env, stdout=lf, stderr=subprocess.STDOUT))
        if lp:
            lf = open(Q / "logs" / f"side_lp_{arm}.log", "w")
            procs.append(subprocess.Popen([PY, "logit_probe_pair.py", str(X), str(OP)] + [pair(s) for s in sorted(lp, reverse=True)],
                                          cwd=PC, env=env, stdout=lf, stderr=subprocess.STDOUT))
        launched.add(arm)
        log(f"{arm} on GPU{gpu}: side generation over {len(beh)} checkpoints and side paired margin over {len(lp)}, both descending")
    time.sleep(30)
log("exit")
