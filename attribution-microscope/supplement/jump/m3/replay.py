"""M3 replay (CONTRACT_M3 v4 §1): fresh run of the order-1009 dataset, logging_steps 1, hashing hook,
stopped once checkpoint-430 is fully written. Server, amic env, one GPU (JOBQ_GPU)."""
import json
import os
import shutil
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "m_common")); sys.path.insert(0, HERE)
import mcommon as mc  # noqa: E402
import m3  # noqa: E402
sys.path.insert(0, os.path.join(mc.X, "src"))
import yaml  # noqa: E402
import train_arm  # noqa: E402

OUT = os.path.join(mc.OUT, "m3"); os.makedirs(OUT, exist_ok=True)
GPU = os.environ["JOBQ_GPU"]
arm_dir = train_arm.arm_dir(mc.REPLAY_ARM)
if arm_dir.exists():
    shutil.move(str(arm_dir), str(arm_dir) + f".old{int(time.time())}")
y = train_arm.build_yaml(mc.REPLAY_ARM, "s1", dataset_of=mc.BASE_ARM, save_steps=10)
cfg = yaml.safe_load(open(y))
assert cfg["dataset"] == train_arm.dataset_name(mc.BASE_ARM) and cfg["save_steps"] == 10 and "max_steps" not in cfg
cfg["logging_steps"] = 1
yaml.safe_dump(cfg, open(y, "w"))
hpath = os.path.join(OUT, "replay_hashes.jsonl")
if os.path.exists(hpath):
    os.remove(hpath)
env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, DISABLE_VERSION_CHECK="1")
logf = open(os.path.join(OUT, "replay_train.log"), "w")
proc = subprocess.Popen([sys.executable, os.path.join(HERE, "replay_lf.py"), str(y), hpath], cwd=mc.X, env=env,
                        stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
ck = arm_dir / f"checkpoint-{m3.STOP_CK}"
need = [ck / f for f in m3.REPLAY_FILES]
t0, stable, sizes = time.time(), 0, None
while proc.poll() is None:
    if all(p.exists() for p in need):
        s = [p.stat().st_size for p in need]
        stable = stable + 1 if s == sizes else 0
        sizes = s
        if stable >= 2:
            break
    if time.time() - t0 > 3 * 3600:
        break
    time.sleep(15)
if proc.poll() is None:
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(90)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL); proc.wait()
logf.close()
done = all(p.exists() for p in need)
mc.wjson(os.path.join(OUT, "replay_status.json"), {"complete": done, "seconds": time.time() - t0, "yaml": str(y),
                                                   "checkpoints": sorted(int(p.name.split("-")[1]) for p in arm_dir.glob("checkpoint-*"))})
print("replay complete" if done else "replay INCOMPLETE")
sys.exit(0 if done else 1)
