"""Side read-outs (deviation D2, user 2026-11-06: several tasks may share a card when memory allows).

An arm job holds its card for training (85 GB) and then for the read-outs (one process, 15 GB). While an arm is in
its read-out phase this watcher starts two extra read-out processes on the SAME card, working DOWN the grids
(ASR 600 -> 260, LP 560 -> 20) while the job works up. behav_many.py and logit_probe.py skip outputs that already
exist, so the two directions meet in the middle and nothing is computed differently: same scripts, same checkpoints,
same arguments per checkpoint. Nothing is launched on a card that is training.
    nohup python p1_side.py &      (exits when every arm of the queue is done or after 14 h)"""
import json, os, subprocess, time
from pathlib import Path
import p1
X = Path("/workspace/claude-jump/xo"); O = Path("/workspace/claude-jump/lp_out"); Q = Path("/workspace/claude-jump/jobq_p1")
LC = "/workspace/claude-jump/lp_code"; PY = "/workspace/miniconda/envs/amic/bin/python"
ENV = dict(os.environ, HF_HOME="/data/hf_cache", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", DISABLE_VERSION_CHECK="1", TOKENIZERS_PARALLELISM="false",
           PATH="/workspace/miniconda/envs/amic/bin:" + os.environ["PATH"])
for k in ("TRANSFORMERS_CACHE", "HF_DATASETS_CACHE"):
    ENV.pop(k, None)
NEED_FREE_MB = 34000            # two side processes of about 15 GB each, plus slack
launched, procs = set(), []


def log(msg):
    line = f"[{time.strftime('%F %T')}] [p1_side] {msg}"
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
    for f in (Q / "running").glob("p1_*_arm_*"):
        try:
            out[f.name.split("_arm_")[1]] = json.loads(f.read_text())["gpu"]
        except Exception:
            pass
    return out


def missing(arm):
    beh = [s for s in p1.STEPS_ASR if not (X / "runs" / "behavioral" / f"{arm}@s{s}.json").exists()]
    lp = [s for s in p1.STEPS_LP if not (O / f"{arm}@s{s}.json").exists()]
    return beh, lp


t0 = time.time()
log("start")
while time.time() - t0 < 14 * 3600:
    done = {f.name for f in (Q / "done").iterdir()}
    if "p1_99_eval" in done or (Q / "STOP_SIDE").exists():
        break
    for arm, gpu in running_arms().items():
        ad = X / "runs" / "arms" / arm
        trained = (ad / "arm_meta.json").exists() and (ad / f"checkpoint-{p1.STOP}" / "adapter_model.safetensors").exists()
        if arm in launched or not trained:
            continue
        beh, lp = missing(arm)
        if len(beh) + len(lp) < 6:
            launched.add(arm); continue
        if free_mb(gpu) < NEED_FREE_MB:
            continue
        env = dict(ENV, CUDA_VISIBLE_DEVICES=str(gpu))
        pair = lambda s: f"{arm}@s{s}={ad}/checkpoint-{s}"            # noqa: E731
        if beh:
            lf = open(Q / "logs" / f"side_asr_{arm}.log", "w")
            procs.append(subprocess.Popen([PY, "supplement/phase2/behav_many.py"] + [pair(s) for s in sorted(beh, reverse=True)], cwd=str(X), env=env, stdout=lf, stderr=subprocess.STDOUT))
        if lp:
            lf = open(Q / "logs" / f"side_lp_{arm}.log", "w")
            procs.append(subprocess.Popen([PY, "logit_probe.py", str(X), str(O)] + [pair(s) for s in sorted(lp, reverse=True)], cwd=LC, env=env, stdout=lf, stderr=subprocess.STDOUT))
        launched.add(arm)
        log(f"{arm} on GPU{gpu}: side ASR over {len(beh)} checkpoints and side LP over {len(lp)} checkpoints, both descending")
    time.sleep(30)
log("exit")
