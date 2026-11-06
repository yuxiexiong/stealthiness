"""[phase-1 worker, third version: memory-based claim rule, see gpu_busy; stops on STOP_C] Minimal GPU job queue for the supplement/jump line (one worker per GPU).

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




MARGIN_MB = 2000


def gpu_busy(gpu, need_mb=0):
    """Rule of 2026-11-06 evening (user: "只要显存还有空间就允许并行实验"; deviation D3): a job may start on a card
    whenever the card has at least need_mb + 2 GB free, whoever else is on it. need_mb comes from the job file and is
    a MEASURED footprint (one arm's training: 85,359 MiB peak over eight arms; one read-out process: 14.8 GB).
    A job with need_mb = 0 (CPU only) never waits for a card."""
    if need_mb <= 0:
        return False
    q = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True)
    for ln in q.strip().splitlines():
        i, used, total = [int(x) for x in ln.split(",")]
        if i == gpu:
            return (total - used) < need_mb + MARGIN_MB
    return True


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
    log("worker p1c start (a job starts when its card has need_mb + 2 GB free; 3 checks 60 s apart + re-check; CPU jobs do not wait)")
    while True:
        if (R / "STOP_C").exists():
            log("STOP_C file present; exiting")
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
        need = int(nxt.get("need_mb", 87000))            # a job file without need_mb is treated as a full training
        warm = time.time() - last_end < WARM_S and not gpu_busy(gpu, need)
        if not warm and not os.environ.get("JOBQ_TEST"):          # JOBQ_TEST: self-test only, no idle wait
            n_idle = 0
            while n_idle < IDLE_CHECKS:
                n_idle = 0 if gpu_busy(gpu, need) else n_idle + 1
                time.sleep(CHECK_S)
            # Re-check before claiming. The loop above sleeps CHECK_S AFTER its last
            # observation, so that observation is already 60 s stale here. At a 2-check
            # window that blind tail is half the wait, which is exactly how a short idle
            # window ends up putting a job on a card somebody else has just taken.
            if gpu_busy(gpu, need):
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
