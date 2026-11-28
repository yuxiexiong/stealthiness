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
- BLOCKED propagates (review finding 2): a job whose dependency is failed OR blocked is blocked; so is a job whose
  `requires` file is the done file of a failed/blocked job in this queue. Pending jobs whose conditions can never
  come from this queue are listed in a STALLED log line every 30 min instead of waiting silently.
- `after` = soft dependencies: the job waits until each listed job is terminal (done, failed or blocked); used where
  some upstream jobs are legitimately never run (dropped attacks).
- the 3-check GPU wait is shortened to 1 check when this runner's own previous job ended < 60 s ago (the memory drop
  is ours, not a teammate's); any other start keeps 3 checks.
- concurrency (speed plan 2; user rule 10-30: several workers per card allowed if numbers do not change):
  `--workers N` runs up to N of OUR GPU jobs at once on the runner's GPUs. A new GPU job starts only if
  (a) no process that is not ours is on those GPUs, and (b) used memory + the job's `mem_gb` stays below total
  minus 4 GiB. When none of our GPU jobs is running, the classic 3-check wait above applies, so a teammate's job
  is always waited out, never joined. A job without `mem_gb` is exclusive. Priority order is kept: if the next
  ready GPU job does not fit, later jobs do not jump it.
- stop: kill the runner first, then its child jobs; then remove those jobs' claim files.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


FAKE = os.environ.get("RUNNER_FAKE_GPU")   # tests: "TOTAL_MIB" -> simulated card, no foreign processes
_fake_used = {}


def gpu_used_mib(i: int) -> int:
    if FAKE:
        return int(_fake_used.get(i, 0))
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(i)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return int(out.splitlines()[0])


def gpu_total_mib(i: int) -> int:
    if FAKE:
        return int(FAKE)
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", str(i)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return int(out.splitlines()[0])


def gpu_pids(i: int) -> set:
    if FAKE:   # tests: foreign pids are read from RUNNER_FAKE_PIDS_FILE (one per line); empty/missing = none
        f = os.environ.get("RUNNER_FAKE_PIDS_FILE")
        return {int(x) for x in open(f).read().split()} if f and os.path.exists(f) else set()
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader", "-i", str(i)],
                         capture_output=True, text=True).stdout
    return {int(x) for x in out.split() if x.strip().isdigit()}


def descendants(roots) -> set:
    """All pids under the given pids (the job shells), read from /proc."""
    kids = {}
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                ppid = int((d / "stat").read_text().rsplit(")", 1)[1].split()[1])
                kids.setdefault(ppid, []).append(int(d.name))
            except (OSError, IndexError, ValueError):
                pass
    out, todo = set(), list(roots)
    while todo:
        p = todo.pop()
        if p not in out:
            out.add(p); todo += kids.get(p, [])
    return out


def wait_free(gpus, checks=3, interval=60, limit=2000, log=print):
    """Wait until every GPU shows < limit MiB AND no compute process at all (11-28: a teammate's engine used ~1 GiB
    at 97% utilisation; a memory-only test would have joined it)."""
    checks = max(1, checks)
    ok, last = 0, None
    while ok < checks:
        used = [gpu_used_mib(g) for g in gpus]
        procs = [sorted(gpu_pids(g)) for g in gpus]
        free = all(u < limit for u in used) and not any(procs)
        ok = ok + 1 if free else 0
        if ok < checks:
            state = (tuple(used), tuple(map(tuple, procs)))
            if ok == 0 and state != last:
                log(f"waiting for GPUs {gpus}: used {used} MiB, processes {procs}")
                last = state
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
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--free_checks", type=int, default=3,
                    help="consecutive free checks before taking a card (raise it when a neighbour runs back-to-back "
                         "processes with short gaps)")
    a = ap.parse_args()
    root = Path(a.root)
    gpus = [int(x) for x in a.gpus.split(",") if x != ""]
    rd = root / "runner"
    for sub in ("claims", "failed", "blocked", "logs"):
        (rd / sub).mkdir(parents=True, exist_ok=True)
    log = lambda m: print(time.strftime("%H:%M:%S"), f"[gpus {a.gpus}]", m, flush=True)
    last_own_end, last_stall_log = 0.0, 0.0
    running = {}  # jid -> (Popen, t0, logfile, job)

    def reap():
        nonlocal last_own_end
        for jid in list(running):
            proc, t0, lf, j = running[jid]
            rc = proc.poll()
            if rc is None:
                continue
            lf.close()
            ok = rc == 0 and Path(j["done"]).exists()
            if not ok:
                (rd / "failed" / f"{jid}.json").write_text(json.dumps({"rc": rc, "seconds": time.time() - t0}))
            (rd / "claims" / jid).unlink(missing_ok=True)
            log(f"{'DONE' if ok else 'FAILED'} {jid} rc={rc} {time.time() - t0:.0f}s")
            if j["gpus"] > 0:
                last_own_end = time.time()
                if FAKE:
                    for g in gpus:
                        _fake_used[g] = _fake_used.get(g, 0) - int(j.get("mem_gb", 0) * 1024)
            del running[jid]

    def gpu_fits(j) -> bool:
        """Non-blocking admission while some of our GPU jobs run (speed plan 2)."""
        ours = [r for r in running.values() if r[3]["gpus"] > 0]
        if len(ours) >= a.workers or "mem_gb" not in j or any("mem_gb" not in r[3] for r in ours):
            return False
        mine = descendants([r[0].pid for r in ours])
        budget = sum(r[3]["mem_gb"] for r in ours) * 1024
        for g in gpus:
            if gpu_pids(g) - mine:
                return False   # someone else is on the card: never join them
            # a job that just started has not reached its peak yet: count the larger of measured use and budgets
            if max(gpu_used_mib(g), budget) + j["mem_gb"] * 1024 > gpu_total_mib(g) - 4096:
                return False
        return True

    while True:
        reap()
        jobs = json.load(open(a.queue))
        done_ids = {j["id"] for j in jobs if Path(j["done"]).exists()}
        failed_ids = {p.stem for p in (rd / "failed").glob("*.json")}
        blocked_ids = {p.stem for p in (rd / "blocked").glob("*.json")}
        dead_done = {str(Path(j["done"]).resolve()) for j in jobs if j["id"] in failed_ids | blocked_ids}
        pending, started, waiting = 0, False, []
        for j in jobs:
            jid = j["id"]
            if jid in done_ids or jid in failed_ids or (rd / "blocked" / f"{jid}.json").exists():
                continue
            if jid in running or (rd / "claims" / jid).exists():
                pending += 1; continue
            if j["gpus"] not in (0, len(gpus)):
                pending += 1; continue
            dead = [d for d in j["deps"] if d in failed_ids or d in blocked_ids]
            dead += [r for r in j["requires"] if str((root / r.rsplit(":", 1)[0]).resolve()) in dead_done]
            if dead:
                (rd / "blocked" / f"{jid}.json").write_text(json.dumps({"why": "dependency failed or blocked",
                                                                         "on": dead}))
                log(f"BLOCKED {jid} (upstream {dead})")
                blocked_ids.add(jid); dead_done.add(str(Path(j["done"]).resolve()))
                continue
            states = [req_state(root, r) for r in j["requires"]]
            if any(s is False for s in states):
                (rd / "blocked" / f"{jid}.json").write_text(json.dumps({"why": "requires false", "req": j["requires"]}))
                log(f"BLOCKED {jid}")
                continue
            terminal = done_ids | failed_ids | blocked_ids
            if (not all(d in done_ids for d in j["deps"]) or any(s is None for s in states)
                    or not all(d in terminal for d in j.get("after", []))
                    or not all(Path(f).exists() for f in j.get("wait_files", []))):
                pending += 1; waiting.append(jid); continue
            ours_gpu = any(r[3]["gpus"] > 0 for r in running.values())
            if j["gpus"] > 0 and ours_gpu and not gpu_fits(j):
                pending += 1; break          # keep priority order: later jobs do not jump this one
            if j.get("disk_gb", 0) > 0 and running and not disk_ok(root, j["disk_gb"]):
                pending += 1; break
            try:
                fd = os.open(rd / "claims" / jid, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()} {a.gpus} {time.time()}".encode()); os.close(fd)
            except FileExistsError:
                pending += 1; continue
            if j.get("disk_gb", 0) > 0 and not running:
                wait_disk(root, j["disk_gb"], interval=a.poll, log=log)
            if j["gpus"] > 0 and not ours_gpu:
                own = time.time() - last_own_end < 60
                wait_free(gpus, checks=1 if own else a.free_checks, interval=a.free_interval, log=log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)) if j["gpus"] > 0 else "",
                       CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED="0")
            log(f"START {jid} (est {j['est_min']} min, running {len(running) + 1})")
            lf = open(rd / "logs" / f"{jid}.log", "a")
            proc = subprocess.Popen(j["cmd"], shell=True, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, cwd=str(root / "code"))
            running[jid] = (proc, time.time(), lf, j)
            if FAKE and j["gpus"] > 0:
                for g in gpus:
                    _fake_used[g] = _fake_used.get(g, 0) + int(j.get("mem_gb", 0) * 1024)
            started = True
            break
        if not started:
            if pending == 0 and not running:
                log("queue exhausted"); return
            if waiting and not running and time.time() - last_stall_log > 1800:
                log(f"STALLED? {len(waiting)} pending, first: {waiting[:5]}"); last_stall_log = time.time()
            time.sleep(min(a.poll, 5) if running else a.poll)


if __name__ == "__main__":
    main()
