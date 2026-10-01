"""Minimal GPU job queue for the supplement/jump line (one worker per GPU).

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

IDLE_MB, IDLE_CHECKS, CHECK_S, WARM_S = 2000, 10, 60, 180


def gpu_busy(gpu):
    q = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used",
                                 "--format=csv,noheader,nounits"], text=True)
    uuid, used = None, None
    for ln in q.strip().splitlines():
        i, u, m = [x.strip() for x in ln.split(",")]
        if int(i) == gpu:
            uuid, used = u, int(m)
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid",
                                    "--format=csv,noheader"], text=True).split()
    return used >= IDLE_MB or uuid in apps


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
    log("worker start")
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
            time.sleep(60)
            continue
        warm = time.time() - last_end < WARM_S and not gpu_busy(gpu)
        if not warm:
            n_idle = 0
            while n_idle < IDLE_CHECKS:
                n_idle = 0 if gpu_busy(gpu) else n_idle + 1
                time.sleep(CHECK_S)
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
            rc = subprocess.call(nxt["cmd"], shell=True, cwd=nxt.get("cwd"), env=env,
                                 stdout=lf, stderr=subprocess.STDOUT)
        (R / "running" / n).rename(R / ("done" if rc == 0 else "failed") / n)
        log(f"{'done' if rc == 0 else 'FAILED rc=%d' % rc} {n}")
        last_end = time.time()


if __name__ == "__main__":
    main()
