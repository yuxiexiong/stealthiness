"""Queue runner. One process per GPU set:  python runner.py --root ROOT --queue q.json --gpus 0
(2-GPU jobs: --gpus 0,1; CPU-only jobs (gpus=0) run on whichever runner reaches them first).

Rules (EXPERIMENT_PLAN §6.3):
- queue order is priority; a job runs when its deps are done, its wait_files exist and its `requires` are true;
  a `requires` that exists and is false marks the job BLOCKED (never run).
- before *every* GPU job the runner waits until each of its GPUs shows < 2000 MiB used on 3 consecutive checks
  60 s apart (co-residency lesson); jobs are claimed atomically (O_EXCL) under ROOT/runner/claims.
- a job with `disk_gb` > 0 waits until ROOT's filesystem has disk_gb + RESERVE_GB free (11-26: the volume is shared
  with teammates; a half-written checkpoint must never happen).
- a failed job (non-zero exit or missing done file) is recorded under ROOT/runner/failed and is not retried.
- stop: kill the runner first, then its child job; then remove that job's claim file.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def gpu_used_mib(i: int) -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(i)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return int(out.splitlines()[0])


def wait_free(gpus, checks=3, interval=60, limit=2000, log=print):
    ok = 0
    while ok < checks:
        used = [gpu_used_mib(g) for g in gpus]
        ok = ok + 1 if all(u < limit for u in used) else 0
        if ok < checks:
            if ok == 0:
                log(f"waiting for GPUs {gpus}: used {used} MiB")
            time.sleep(interval)


RESERVE_GB = 10


def disk_ok(root, need_gb, reserve_gb=RESERVE_GB) -> bool:
    return shutil.disk_usage(root).free / 2**30 >= need_gb + reserve_gb


def wait_disk(root, need_gb, interval=60, log=print):
    warned = False
    while not disk_ok(root, need_gb):
        if not warned:
            log(f"waiting for disk: need {need_gb}+{RESERVE_GB} GiB, free "
                f"{shutil.disk_usage(root).free / 2**30:.1f} GiB"); warned = True
        time.sleep(interval)


def req_state(root: Path, req: str):
    path, key = req.rsplit(":", 1)
    p = root / path
    if not p.exists():
        return None
    return bool(json.load(open(p)).get(key))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--free_interval", type=int, default=60)
    a = ap.parse_args()
    root = Path(a.root)
    gpus = [int(x) for x in a.gpus.split(",") if x != ""]
    rd = root / "runner"
    for sub in ("claims", "failed", "blocked", "logs"):
        (rd / sub).mkdir(parents=True, exist_ok=True)
    log = lambda m: print(time.strftime("%H:%M:%S"), f"[gpus {a.gpus}]", m, flush=True)
    while True:
        jobs = json.load(open(a.queue))
        done_ids = {j["id"] for j in jobs if Path(j["done"]).exists()}
        failed_ids = {p.stem for p in (rd / "failed").glob("*.json")}
        pending, ran = 0, False
        for j in jobs:
            jid = j["id"]
            if jid in done_ids or jid in failed_ids or (rd / "blocked" / f"{jid}.json").exists():
                continue
            if (rd / "claims" / jid).exists():
                pending += 1; continue
            if j["gpus"] not in (0, len(gpus)):
                pending += 1; continue
            if any(d in failed_ids for d in j["deps"]):
                (rd / "blocked" / f"{jid}.json").write_text(json.dumps({"why": "dependency failed"}))
                continue
            states = [req_state(root, r) for r in j["requires"]]
            if any(s is False for s in states):
                (rd / "blocked" / f"{jid}.json").write_text(json.dumps({"why": "requires false", "req": j["requires"]}))
                log(f"BLOCKED {jid}")
                continue
            if (not all(d in done_ids for d in j["deps"]) or any(s is None for s in states)
                    or not all(Path(f).exists() for f in j.get("wait_files", []))):
                pending += 1; continue
            try:
                fd = os.open(rd / "claims" / jid, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()} {a.gpus} {time.time()}".encode()); os.close(fd)
            except FileExistsError:
                pending += 1; continue
            if j.get("disk_gb", 0) > 0:
                wait_disk(root, j["disk_gb"], interval=a.poll, log=log)
            if j["gpus"] > 0:
                wait_free(gpus, interval=a.free_interval, log=log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)) if j["gpus"] > 0 else "",
                       CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED="0")
            log(f"START {jid} (est {j['est_min']} min)")
            t0 = time.time()
            with open(rd / "logs" / f"{jid}.log", "a") as lf:
                rc = subprocess.run(j["cmd"], shell=True, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, cwd=str(root / "code")).returncode
            ok = rc == 0 and Path(j["done"]).exists()
            if not ok:
                (rd / "failed" / f"{jid}.json").write_text(json.dumps({"rc": rc, "seconds": time.time() - t0}))
            (rd / "claims" / jid).unlink(missing_ok=True)
            log(f"{'DONE' if ok else 'FAILED'} {jid} rc={rc} {time.time() - t0:.0f}s")
            ran = True
            break
        if not ran:
            if pending == 0:
                log("queue exhausted"); return
            time.sleep(a.poll)


if __name__ == "__main__":
    main()
