#!/usr/bin/env python3
"""Unplanned extra imaging, both approved by the user:
  D63  P-1.0-D steps 180-280: trigger share rose from 4% (step 158) to 49%
       (step 300) while ASR stayed ~0, and none of it was imaged, because the
       frozen selection rule picks checkpoints by ASR (step 160 skipped: two
       steps from the imaged 158)
  D62  P-1.0-D steps 440, 460, 480: the one-checkpoint ASR dip at 460
Starts only after the main queue has finished (runs/phase2/finished.json).
Imaged exactly like the selected checkpoints: img_many.py, the 60 trajectory
samples, A on both columns, B on the triggered column. One task per GPU, and
only after gpu_watch has seen the card idle for three minutes. Status in
/root/p2_watch/extra.json; this file lives outside the repository."""
import json
import os
import subprocess
import sys
import time

R = "/root/attribution-microscope"
sys.path.insert(0, R + "/src")
from gpu_watch import GpuWatch  # noqa: E402

PY = "/workspace/miniconda/envs/amic/bin/python"
ST = "/root/p2_watch/extra.json"
LOG = open("/root/p2_watch/extra.log", "a")
PLAN = [(s, "D63") for s in (180, 200, 220, 240, 260, 280)] + [(s, "D62") for s in (440, 460, 480)]


def status(**k):
    k["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    k["plan"] = [f"{s}({d})" for s, d in PLAN]
    with open(ST, "w") as f:
        json.dump(k, f, ensure_ascii=False)


todo = [s for s, _ in PLAN]
done, procs = [], {}
status(state="waiting_main_queue", todo=todo, done=done, pid=os.getpid())
while not os.path.exists(R + "/runs/phase2/finished.json"):
    time.sleep(60)
watch = GpuWatch([0, 1])
status(state="waiting_gpu", todo=todo, done=done, pid=os.getpid())
while todo or procs:
    watch.tick()
    for g, (p, s) in list(procs.items()):
        rc = p.poll()
        if rc is None:
            continue
        del procs[g]
        watch.reset(g)
        if rc != 0:
            status(state="failed", step=s, gpu=g, rc=rc, todo=todo, done=done, pid=os.getpid())
            sys.exit(rc)
        done.append(s)
    for g in (0, 1):
        if g in procs or not todo or not watch.idle(g):
            continue
        s = todo.pop(0)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONPATH=R + "/src")
        item = f"P-1.0-D@s{s}={R}/runs/arms/P-1.0-D/checkpoint-{s}=AB"
        procs[g] = (subprocess.Popen([PY, R + "/supplement/phase2/img_many.py", item],
                                     cwd=R, env=env, stdout=LOG, stderr=subprocess.STDOUT), s)
    status(state="running" if procs else "waiting_gpu", todo=todo, done=done,
           running={str(g): s for g, (_, s) in procs.items()}, pid=os.getpid())
    time.sleep(30)
ok = all(os.path.exists(f"{R}/runs/maps/P-1.0-D@s{s}/p_core_trig.npz") for s, _ in PLAN)
status(state="done" if ok else "failed", todo=todo, done=done, maps_present=ok, pid=os.getpid())
