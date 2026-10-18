"""Two queue-assigned GPU lanes; one locked ledger, no retries or GPU selection."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

LIMIT_SECONDS = 32 * 3600


def save(path, value):
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


@contextmanager
def ledger(root, contract):
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    # Keep the lock inode stable while budget.json is atomically replaced.
    with (root / ".budget.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        path = root / "budget.json"
        state = json.loads(path.read_text()) if path.exists() else {
            "contract_sha256": digest, "limit_seconds": LIMIT_SECONDS,
            "deadline_epoch": None, "jobs": {
                job["name"]: {"status": "queued"} for job in contract["jobs"]}}
        if state["contract_sha256"] != digest:
            raise ValueError("contract changed after dispatch began")
        state["deadline_epoch"] = None
        yield state
        state["actual_seconds"] = sum(j.get("wall_seconds", 0) for j in state["jobs"].values())
        save(path, state)


def claim(root, contract, lane, now=None):
    now = time.time() if now is None else now
    with ledger(root, contract) as state:
        rows = state["jobs"]
        used = sum(row.get("wall_seconds", 0) for row in rows.values())
        running = [row for row in rows.values() if row["status"] == "running"]
        capacity = LIMIT_SECONDS - used - sum(row["cap_seconds"] for row in running)
        for job in contract["jobs"]:
            row = rows[job["name"]]
            if row["status"] != "queued":
                continue
            required = [rows[name]["status"] for name in job.get("requires", [])]
            if any(status in ("failed", "blocked", "deferred") for status in required):
                row.update(status="blocked", reason="required task did not complete successfully")
                continue
            if any(status != "done" for status in required):
                continue
            estimate = float(job["est_minutes"]) * 60
            if estimate > LIMIT_SECONDS - used:
                row.update(status="deferred", reason="estimate exceeds remaining budget")
                continue
            if estimate > capacity:
                return None  # Other lane releases its ceiling when its child finishes.
            cap = min(2 * estimate, capacity)
            row.update(status="running", lane=lane, dispatcher_pid=os.getpid(),
                       gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), start_epoch=now,
                       reserved_seconds=estimate, cap_seconds=cap)
            return dict(job, cap_seconds=cap)
        return None


def finish(root, contract, name, result):
    with ledger(root, contract) as state:
        row = state["jobs"][name]
        if row["status"] != "running":
            raise ValueError("only a claimed task can finish")
        row.update(result)


def terminate(child):
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def run(root, contract, job, stopped):
    name = job["name"]
    folder = root / "dispatch" / name
    folder.mkdir(parents=True, exist_ok=True)
    argv = [contract["runtime"]["python"], str(Path(__file__).with_name("overnight_run.py")),
            "--root", str(root), "--job", name]
    child, error, rc = None, None, None
    started = time.monotonic()
    cap = job["cap_seconds"]
    # Reserve a little of the ceiling for killing/reaping this process group.
    stop_at = started + max(0, cap - min(1, cap / 10))
    try:
        if stopped[0] or cap <= 0:
            raise TimeoutError("dispatcher interrupted or task allowance exhausted before launch")
        with (folder / "child.log").open("a") as log:
            child = subprocess.Popen(argv, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                     env=os.environ.copy(), start_new_session=True)
            while child.poll() is None:
                remaining = stop_at - time.monotonic()
                if stopped[0] or remaining <= 0:
                    raise TimeoutError("dispatcher interrupted" if stopped[0] else "runtime ceiling")
                try:
                    child.wait(timeout=min(1, remaining))
                except subprocess.TimeoutExpired:
                    pass
            rc = child.returncode
            if rc:
                error = f"child exited {rc}"
    except (OSError, TimeoutError) as exc:
        error = str(exc)
    finally:
        if child is not None and (child.poll() is None or error):
            terminate(child)
            rc = child.returncode
    result = {"status": "failed" if error else "done", "returncode": rc,
              "error": error, "partial": bool(error), "end_epoch": time.time(),
              "wall_seconds": time.monotonic() - started, "argv": argv}
    save(folder / ("failure.json" if error else "result.json"), result)
    finish(root, contract, name, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lane", default=os.environ.get("JOBQ_GPU", "lane"))
    args = parser.parse_args()
    root = args.root.resolve()
    contract = json.loads((root / "contract.json").read_text())
    names = [job["name"] for job in contract["jobs"]]
    if len(set(names)) != len(names) or any(not re.fullmatch(r"[\w.-]+", n) for n in names):
        raise ValueError("task names must be unique simple filenames")
    if any(set(j.get("requires", [])) - set(names) for j in contract["jobs"]):
        raise ValueError("required task is absent from contract")
    if any(not math.isfinite(float(j["est_minutes"])) or float(j["est_minutes"]) <= 0
           for j in contract["jobs"]):
        raise ValueError("est_minutes must be finite and positive")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("queue must assign CUDA_VISIBLE_DEVICES")
    stopped = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    while not stopped[0]:
        job = claim(root, contract, args.lane)
        if job is None:
            break
        run(root, contract, job, stopped)
    return 0  # Per-task failures are recorded; independent tasks keep running.


if __name__ == "__main__":
    raise SystemExit(main())
