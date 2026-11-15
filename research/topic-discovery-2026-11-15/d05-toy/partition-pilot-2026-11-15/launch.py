"""One foreground child, one idle GPU, no retry or follow-on experiment."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
from qualification_core import sha

p = argparse.ArgumentParser()
for name in ("gpu", "python", "model-dir"):
    p.add_argument("--"+name, required=True)
a = p.parse_args()
assert re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", a.gpu), "physical GPU UUID required"
root = Path(__file__).resolve().parent
lock = (root / "launch.lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
frozen = json.loads((root / "CODE_FREEZE.json").read_text())
for name, expected in frozen.items():
    assert sha(root / name) == expected, name
cpu = json.loads((root / "CPU_CHECK.json").read_text())
assert cpu["cpu_pass"] and cpu["runner_sha256"] == frozen["pilot.py"]
assert cpu["code_freeze_sha256"] == sha(root / "CODE_FREEZE.json")
assert cpu["bundle_sha256"] == frozen["STATIC_BUNDLE.json"]
assert cpu["baseline_sha256"] == frozen["BASELINE.jsonl"]
apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid",
                               "--format=csv,noheader"], text=True).splitlines()
assert a.gpu not in [x.strip() for x in apps], "GPU has another compute process"
actual_uuid = subprocess.check_output(["nvidia-smi", "-i", a.gpu, "--query-gpu=uuid",
                                      "--format=csv,noheader"], text=True).strip()
assert actual_uuid == a.gpu
free = subprocess.check_output(["nvidia-smi", "-i", a.gpu, "--query-gpu=memory.free",
                               "--format=csv,noheader,nounits"], text=True)
assert int(free.strip()) >= 32768
assert shutil.disk_usage(root).free >= 5 * 2**30
result = root / "result"
assert not result.exists(), "do not repeat or overwrite a qualification run"
started = root / "STARTED.json"
with started.open("x") as f:
    json.dump(dict(start_unix=time.time(), scientific_generate_calls=8160, engineering_generate_calls=32, training_steps=1344,
                   free_memory_mib=int(free.strip()), no_other_compute_on_target=True), f)
env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, PYTHONUNBUFFERED="1")
cmd = [a.python, str(root / "pilot.py"), "--model-dir", a.model_dir,
       "--bundle", str(root / "STATIC_BUNDLE.json"), "--weights", str(root / "OFFICIAL_WEIGHTS.json"),
       "--baseline", str(root / "BASELINE.jsonl"), "--output", str(result)]
start = time.time()
child, failure = None, None


def interrupted(signum, frame):
    raise KeyboardInterrupt("supervisor interrupted")


signal.signal(signal.SIGTERM, interrupted)
try:
    with (root / "runner.log").open("x") as log:
        child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        child.wait()
except BaseException as exc:
    failure = type(exc).__name__
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
    raise
finally:
    now = time.time()
    closed = child is None or child.poll() is not None
    (root / "COST.json").write_text(json.dumps(dict(start_unix=start,
        end_unix=now if closed else None, cost_closed=closed,
        reserved_wall_seconds=now-start, gpu_hours=(now-start)/3600 if closed else None,
        exit_code=child.returncode if child else None, supervisor_error=failure,
        scope="single child including hashes, loading, inference and IO; unclosed time is lower bound",
        cpu_preparation_seconds=cpu["seconds"], automatic_retries=0), indent=2)+"\n")
raise SystemExit(child.returncode)
