"""Per-GPU sequential runner for the field corpus with a live fidelity gate.

Usage: field_runner.py --gpu N --jobs jobs.json --root FIELD_ROOT
Each job: {"name", "land", "seed", "cmd", "out", "reference" (run dir or null)}.
- Jobs with a reference are compared step by step while running; the first mismatch kills the job (by its
  process group, never by name pattern) and stops this GPU's queue.
- Jobs without a reference run only if every earlier job on this GPU passed fidelity.
- Any non-zero exit stops this GPU's queue. No automatic retries. A file STOP in FIELD_ROOT halts after the
  current job.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from field_fidelity import compare_runs, compare_steps, read_steps  # noqa: E402

POLL = 30


def write(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    jobs = json.loads(args.jobs.read_text())
    status_path = args.root / ("status_gpu%d.json" % args.gpu)
    logs = args.root / "logs"
    logs.mkdir(exist_ok=True)
    status = {"gpu": args.gpu, "pid": os.getpid(), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "jobs": [j["name"] for j in jobs], "done": [], "failed": [], "blocked": [], "current": None, "state": "running"}
    write(status_path, status)
    clean_so_far = True
    for job in jobs:
        if (args.root / "STOP").exists():
            status["state"] = "stopped_by_file"
            break
        out = Path(job["out"])
        if (out / "complete.json").exists() and (out / "fidelity.json").exists():
            status["done"].append(job["name"])
            continue
        if job.get("reference") is None and not clean_so_far:
            status["blocked"].append(job["name"])
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
        log = open(logs / (job["name"] + ".log"), "a")
        started = time.time()
        proc = subprocess.Popen(job["cmd"], shell=True, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        status["current"] = {"name": job["name"], "pid": proc.pid, "started": started, "step": 0, "compared": 0}
        write(status_path, status)
        reference = read_steps(Path(job["reference"]) / "training.jsonl") if job.get("reference") else None
        mismatch = None
        while proc.poll() is None:
            time.sleep(POLL)
            new = read_steps(out / "training.jsonl")
            status["current"]["step"] = max(new) if new else 0
            status["current"]["elapsed_s"] = round(time.time() - started)
            if reference is not None:
                n, mismatch = compare_steps(reference, new)
                status["current"]["compared"] = n
                if mismatch is not None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    break
            write(status_path, status)
        log.close()
        rc = proc.returncode
        record = {"name": job["name"], "rc": rc, "wall_s": round(time.time() - started)}
        if mismatch is not None:
            record.update(kind="fidelity_live_mismatch", mismatch=mismatch)
            write(out / "FIDELITY_FAIL.json", record)
        elif rc != 0:
            record.update(kind="crash")
        elif job.get("reference"):
            verdict = compare_runs(job["reference"], out)
            write(out / "fidelity.json", verdict)
            if verdict["first_mismatch"] is not None or (job.get("compare_final", True) and not verdict.get("final_equal")):
                record.update(kind="fidelity_final_mismatch", verdict=verdict)
                write(out / "FIDELITY_FAIL.json", record)
            else:
                record.update(kind="ok", fidelity="exact", steps_compared=verdict["steps_compared"])
        else:
            write(out / "fidelity.json", {"reference": None, "note": "new trajectory, no reference run"})
            record.update(kind="ok", fidelity="no_reference")
        status["current"] = None
        if record["kind"] == "ok":
            status["done"].append(record)
        else:
            status["failed"].append(record)
            clean_so_far = False
            status["state"] = "halted_on_failure"
            write(status_path, status)
            return 1
        write(status_path, status)
    if status["state"] == "running":
        status["state"] = "finished"
    write(status_path, status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
