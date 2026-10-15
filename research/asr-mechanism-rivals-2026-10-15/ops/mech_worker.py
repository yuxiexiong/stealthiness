"""[MECHANISM-LINE DUAL-GPU WORKER, 2026-10-16, user-authorised: after the defense line's ~9 h of jobs, use both
GPUs as soon as they are free.] One process per GPU on the mechanism queue root (jobq_gpu0). Differences from
worker_gpu0.py: (1) stops on <root>/STOP_MECH (the old <root>/STOP stays in place so the old single-GPU worker and
its watchdog stay off, and the defense GPU0 worker keeps its 'mechanism paused' permission); (2) never claims before
the epoch time in <root>/NOT_BEFORE; (3) before every claim it reads the defense queue /workspace/claude-jump/jobq
(read-only) and yields while any defense job is runnable (unclaimed, deps done, none failed) unless that queue has a
STOP file; (4) same idle rule as worker_gpu0.py (3 idle minutes, re-check right before claiming).

Original description:
Minimal GPU job queue for the supplement/jump line (one worker per GPU).

    nohup python worker.py --root /workspace/claude-jump/jobq --gpu 0 &

Jobs are JSON files in <root>/jobs/, run in file-name order:
  {"name": "...", "cmd": "shell command", "cwd": "...", "deps": ["job name" | "file:/abs/path"],
   "est_min": 60}
A job runs when all deps are done (job dep: <root>/done/<name>; file dep: path exists).
Claiming is atomic (O_EXCL). Before a job the worker needs its GPU idle (< 2 GB, no compute
process) for 3 consecutive minutes [GPU0 copy; original jobq/worker.py uses 10], unless this worker finished a job < 3 min ago and the
GPU is still empty. Exit code != 0 -> <root>/failed/<name>; dependents never start.
Jobs are never retried automatically; a fix is queued as a new job file."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

IDLE_MB, IDLE_CHECKS, CHECK_S, WARM_S = 2000, 3, 60, 180   # GPU0 copy: 3 idle minutes (user, 2026-10-15)
DEFENSE = Path("/workspace/claude-jump/jobq")


def defense_runnable():
    """Names of defense-queue jobs that a defense worker could start now (read-only)."""
    if (DEFENSE / "STOP").exists():
        return []
    out = []
    for jf in sorted((DEFENSE / "jobs").glob("*.json")):
        try:
            j = json.load(open(jf))
        except (OSError, ValueError):
            continue
        n = j["name"]
        if (DEFENSE / "claims" / n).exists():
            continue
        deps = j.get("deps", [])
        if any((not d.startswith("file:")) and (DEFENSE / "failed" / d).exists() for d in deps):
            continue
        if all(Path(d[5:]).exists() if d.startswith("file:") else (DEFENSE / "done" / d).exists() for d in deps):
            out.append(n)
    return out


def not_before(R):
    p = R / "NOT_BEFORE"
    return float(p.read_text().split()[0]) if p.exists() else 0.0


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
    ap.add_argument("--gpu", type=int, required=True, choices=(0, 1))
    ap.add_argument("--dry", action="store_true", help="print the claim decision once and exit; claim nothing")
    a = ap.parse_args()
    R, gpu = Path(a.root), a.gpu
    for d in ("jobs", "claims", "done", "failed", "logs", "running"):
        (R / d).mkdir(parents=True, exist_ok=True)

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [jobq GPU{gpu}-mech] {msg}"
        print(line, flush=True)
        with open(R / "queue.log", "a") as f:
            f.write(line + "\n")

    def dep_ok(d):
        return Path(d[5:]).exists() if d.startswith("file:") else (R / "done" / d).exists()

    def dep_dead(d):
        return (not d.startswith("file:")) and (R / "failed" / d).exists()

    last_end, last_wait_log = 0.0, 0.0
    if not a.dry:
        log("mechanism dual-GPU worker start (3 idle minutes, re-check before claim, yields to runnable defense jobs, not before %s)"
            % time.strftime("%Y-%m-%d %H:%M", time.localtime(not_before(R))))
    while True:
        if (R / "STOP_MECH").exists():
            log("STOP_MECH present; exiting")
            return
        wait = None
        if time.time() < not_before(R):
            wait = "not before %s" % time.strftime("%Y-%m-%d %H:%M", time.localtime(not_before(R)))
        else:
            dr = defense_runnable()
            if dr:
                wait = "defense job runnable (%s%s)" % (dr[0], " +%d" % (len(dr) - 1) if len(dr) > 1 else "")
        if a.dry:
            print(json.dumps({"gpu": gpu, "wait": wait, "gpu_busy": gpu_busy(gpu)}))
        if wait:
            if a.dry:
                return
            if time.time() - last_wait_log > 1800:
                log("waiting: " + wait)
                last_wait_log = time.time()
            time.sleep(60)
            continue
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
        if a.dry:
            print(json.dumps({"gpu": gpu, "next": nxt["name"] if nxt else None}))
            return
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
        if not os.environ.get("JOBQ_TEST") and (gpu_busy(gpu) or defense_runnable()):
            continue                                  # re-check right before claiming: someone started meanwhile (fix 2026-10-15)
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
