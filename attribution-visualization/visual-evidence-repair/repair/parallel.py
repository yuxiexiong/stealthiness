"""Two fixed single-GPU queues inside one phase's allocated-two-GPU budget job.

The outer repair.budget/measure process owns the timeout and charges both cards
until this launcher finishes, including idle tails and cleanup. Commands must
declare --device cuda:0 themselves; no DDP, shell expansion or child sessions.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def run(queues, output):
    if (not isinstance(queues, list) or len(queues) != 2
            or any(not isinstance(queue, list) or not queue for queue in queues)
            or any(not isinstance(command, list) or not command or not command[0]
                   or any(not isinstance(arg, str) or "\0" in arg for arg in command)
                   for queue in queues for command in queue)):
        raise ValueError("queues must contain exactly two nonempty queues of nonempty argv lists")
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(gpus) != 2 or len(set(gpus)) != 2 or any(not gpu or gpu != gpu.strip() for gpu in gpus):
        raise ValueError("parent CUDA_VISIBLE_DEVICES must identify exactly two distinct GPUs")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    jobs = [[{"command": command, "status": "pending", "exit_code": None} for command in queue]
            for queue in queues]
    record = {"status": "running", "allocated_gpus": gpus, "queues": queues, "jobs": jobs,
              "queues_sha256": hashlib.sha256(json.dumps(queues).encode()).hexdigest(),
              "pid": os.getpid(), "process_group": os.getpgrp(),
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "accounting": "outer budget owns wall time times two cards, including idle tails and cleanup"}
    active, positions, stopped = {}, [0, 0], None

    def save():
        temporary = output / "run.json.tmp"
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "run.json")

    def stop(signum, frame):
        nonlocal stopped
        stopped = signum

    def finish(lane, status=None):
        process, job, log = active.pop(lane)
        job.update(status=status or ("completed" if process.returncode == 0 else "failed"),
                   exit_code=process.returncode, finished_seconds=time.monotonic() - started)
        log.close()

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        save()
        # ponytail: two frozen queues; no dynamic scheduling or budget reservations.
        while True:
            changed = False
            for lane, (process, job, log) in list(active.items()):
                if process.poll() is not None:
                    if process.returncode != 0:
                        record["status"] = "failed"
                    finish(lane)
                    changed = True
            if stopped is not None:
                record.update(status="interrupted", signal=stopped)
            if record["status"] != "running":
                break
            for lane in range(2):
                if stopped is not None:
                    record.update(status="interrupted", signal=stopped)
                    break
                if lane in active or positions[lane] == len(jobs[lane]):
                    continue
                index = positions[lane]
                positions[lane] += 1
                job = jobs[lane][index]
                changed = True
                job.update(status="starting", gpu=gpus[lane], started_seconds=time.monotonic() - started,
                           log=f"lane-{lane}-task-{index}.log")
                log = (output / job["log"]).open("w")
                try:
                    process = subprocess.Popen(job["command"], env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[lane]),
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=False)
                except OSError as error:
                    log.close()
                    job.update(status="failed_to_start", error=str(error), finished_seconds=time.monotonic() - started)
                    record["status"] = "failed"
                    break
                job.update(status="running", pid=process.pid)
                active[lane] = process, job, log
            if changed:
                save()
            if record["status"] != "running":
                break
            if not active:
                record["status"] = "completed"
                break
            time.sleep(.05)
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        for process, job, log in active.values():
            if process.poll() is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 5
        for lane, (process, job, log) in list(active.items()):
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            finish(lane, "terminated")
        for queue in jobs:
            for job in queue:
                if job["status"] == "pending":
                    job["status"] = "skipped"
        record.update(elapsed_seconds=time.monotonic() - started,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        save()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queues", required=True, type=Path, help="JSON file: [queue0, queue1]; each queue is a list of argv lists")
    parser.add_argument("--output", required=True, type=Path, help="New immutable launcher output directory")
    args = parser.parse_args(argv)
    try:
        record = run(json.loads(args.queues.read_text()), args.output)
    except (OSError, ValueError, TypeError) as error:
        print(f"parallel blocked: {error}", file=sys.stderr)
        return 2
    print(json.dumps(record))
    return 0 if record["status"] == "completed" else 128 + record["signal"] if "signal" in record else 1


if __name__ == "__main__":
    raise SystemExit(main())
