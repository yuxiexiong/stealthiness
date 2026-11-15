"""One explicitly authorized D05 invocation. Record full reservation wall time.

No retries, queue, background polling or scientific parameter changes.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--original-root", type=Path, required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--gpu", required=True)
    args = p.parse_args()
    control = args.root / "control"
    control.mkdir(exist_ok=False)
    start = time.time()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED="1")
    command = [args.python, "execute.py", "--original-root", str(args.original_root),
               "--prepared", str(args.root / "prepared"), "--execute", str(args.root / "run")]
    receipt = dict(started=start, supervisor_pid=os.getpid(), gpu_uuid=args.gpu, command=command,
                   cost_definition="single GPU reserved wall time, including CPU preflight and IO; not kernel busy time")
    (control / "STARTED.json").write_text(json.dumps(receipt, indent=2)+"\n")
    code, child = -1, None
    try:
        with (control / "runner.log").open("w") as log:
            child = subprocess.Popen(command, cwd=args.root, env=env, stdout=log, stderr=subprocess.STDOUT)
            receipt["runner_pid"] = child.pid
            (control / "STARTED.json").write_text(json.dumps(receipt, indent=2)+"\n")
            code = child.wait()
    finally:
        finish = time.time()
        if child is not None and child.poll() is None:
            result = dict(receipt, supervisor_interrupted_at=finish,
                          elapsed_observed_seconds=finish-start, cost_closed=False,
                          runner_may_still_be_alive=True, auto_retry=False)
            (control / "SUPERVISOR_INTERRUPTED.json").write_text(json.dumps(result, indent=2)+"\n")
        else:
            result = dict(receipt, finished=finish, exit_code=child.returncode if child is not None else code,
                          reserved_gpu_seconds=finish-start, cost_closed=True, auto_retry=False)
            (control / "FINISHED.json").write_text(json.dumps(result, indent=2)+"\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
