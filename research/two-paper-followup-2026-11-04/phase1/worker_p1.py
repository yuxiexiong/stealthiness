"""[phase-1 copy of worker_gpu0_v2.py; only gpu_busy and IDLE_CHECKS differ] Minimal GPU job queue for the supplement/jump line (one worker per GPU).

    nohup python worker.py --root /workspace/claude-jump/jobq --gpu 0 &

Jobs are JSON files in <root>/jobs/, run in file-name order:
  {"name": "...", "cmd": "shell command", "cwd": "...", "deps": ["job name" | "file:/abs/path"],
   "est_min": 60}
A job runs when all deps are done (job dep: <root>/done/<name>; file dep: path exists).
Claiming is atomic (O_EXCL). Before a job the worker needs its GPU idle (< 2 GB, no compute
process) for 10 consecutive minutes, unless this worker finished a job < 3 min ago and the
GPU is still empty. Exit code != 0 -> <root>/failed/<name>; dependents never start.
Jobs are never retried automatically; a fix is queued as a new job file."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

IDLE_MB, IDLE_CHECKS, CHECK_S, WARM_S = 2000, 3, 60, 180   # 2 idle minutes (user, 2026-10-25):
# take the cards the moment the neighbouring line's queue drains. That line's driver refills a
# freed card in milliseconds, so a 2-minute window still cannot cut into its queue; it only
# shortens our wait at the true end of it. Was 10 (see the pre-claim re-check below).




def gpu_busy(gpu):
    """Phase-1 rule, corrected 2026-11-06 08:50 (deviation J104): busy unless the card is EMPTY, i.e. no compute
    process of anyone and less than 2 GB used. One arm's training holds about 85 GB of the 98 GB card (measured on
    the first GD1 attempt: 84,808 MiB), so it cannot share a card with another line's resident process. The first
    version of this rule (free >= 50 GB) assumed 30 GB per arm, which was a guess and was wrong."""
    q = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,uuid", "--format=csv,noheader,nounits"], text=True)
    uuid, used = None, None
    for ln in q.strip().splitlines():
        i, u, uid = [x.strip() for x in ln.split(",")]
        if int(i) == gpu:
            used, uuid = int(u), uid
    if used is None or used >= IDLE_MB:
        return True
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True)
    return any(a.strip() == uuid for a in apps.strip().splitlines() if a.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    a = ap.parse_args()
    R, gpu = Path(a.root), a.gpu
    for d in ("jobs", "claims", "done", "failed", "logs", "running"):
        (R / d).mkdir(parents=True, exist_ok=True)

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [jobq GPU{gpu}] {msg}"
        print(line, flush=True)
        with open(R / "queue.log", "a") as f:
            f.write(line + "\n")

    def dep_ok(d):
        return Path(d[5:]).exists() if d.startswith("file:") else (R / "done" / d).exists()

    def dep_dead(d):
        return (not d.startswith("file:")) and (R / "failed" / d).exists()

    last_end = 0.0
    log("worker p1 start (a card is ours only when EMPTY: no compute process, < 2 GB used; 3 checks 60 s apart + re-check)")
    while True:
        if (R / "STOP").exists():
            log("STOP file present; exiting")
            return
        nxt = None
        for jf in sorted((R / "jobs").glob("*.json")):
            j = json.load(open(jf))
            n = j["name"]
            if (R / "claims" / n).exists():
                continue
            if any(dep_dead(d) for d in j.get("deps", [])):
                continue
            if all(dep_ok(d) for d in j.get("deps", [])):
                nxt = j
                break
        if nxt is None:
            if os.environ.get("JOBQ_TEST"):
                log("test: queue empty; exiting")
                return
            time.sleep(60)
            continue
        warm = time.time() - last_end < WARM_S and not gpu_busy(gpu)
        if not warm and not os.environ.get("JOBQ_TEST"):          # JOBQ_TEST: self-test only, no idle wait
            n_idle = 0
            while n_idle < IDLE_CHECKS:
                n_idle = 0 if gpu_busy(gpu) else n_idle + 1
                time.sleep(CHECK_S)
            # Re-check before claiming. The loop above sleeps CHECK_S AFTER its last
            # observation, so that observation is already 60 s stale here. At a 2-check
            # window that blind tail is half the wait, which is exactly how a short idle
            # window ends up putting a job on a card somebody else has just taken.
            if gpu_busy(gpu):
                log("gpu went busy during the final idle sleep; re-waiting")
                continue
        try:
            os.close(os.open(R / "claims" / nxt["name"], os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue                                  # the other worker took it meanwhile
        n = nxt["name"]
        (R / "running" / n).write_text(json.dumps({"gpu": gpu, "start": time.time(),
                                                   "est_min": nxt.get("est_min")}))
        log(f"start {n}: {nxt['cmd'][:300]}")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), JOBQ_GPU=str(gpu))   # pass --gpu $JOBQ_GPU to scripts that set CUDA_VISIBLE_DEVICES themselves
        with open(R / "logs" / f"{n}.log", "w") as lf:
            rc = subprocess.call(["bash", "-c", nxt["cmd"]], cwd=nxt.get("cwd"), env=env,
                                 stdout=lf, stderr=subprocess.STDOUT)
        (R / "running" / n).rename(R / ("done" if rc == 0 else "failed") / n)
        log(f"{'done' if rc == 0 else 'FAILED rc=%d' % rc} {n}")
        last_end = time.time()


if __name__ == "__main__":
    main()
