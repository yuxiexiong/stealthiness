#!/usr/bin/env python3
"""D62 unplanned extra imaging (user-approved 2026-09-26): P-1.0-D steps 440,
460, 480, around the one-checkpoint ASR dip at step 460 (98% -> 84% -> 100%).

Starts only after the main queue has finished (runs/phase2/finished.json).
Imaged exactly like the six selected checkpoints: img_many.py, the 60
trajectory samples, A on both columns, B on the triggered column. One task
per GPU, dispatched only after gpu_watch has seen the card idle for three
minutes. Status in /root/p2_watch/extra_d62.json; lives outside the repo."""
import json
import os
import subprocess
import sys
import time

R = "/root/attribution-microscope"
sys.path.insert(0, R + "/src")
from gpu_watch import GpuWatch  # noqa: E402

PY = "/workspace/miniconda/envs/amic/bin/python"
ST = "/root/p2_watch/extra_d62.json"
LOG = open("/root/p2_watch/extra_d62.log", "a")


def status(**k):
    k["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(ST, "w") as f:
        json.dump(k, f, ensure_ascii=False)


todo, done, procs = [440, 460, 480], [], {}
status(state="waiting_main_queue", todo=todo, pid=os.getpid())
while not os.path.exists(R + "/runs/phase2/finished.json"):
    time.sleep(60)
watch = GpuWatch([0, 1])
status(state="waiting_gpu", todo=todo, pid=os.getpid())
while todo or procs:
    watch.tick()
    for g, (p, s) in list(procs.items()):
        rc = p.poll()
        if rc is None:
            continue
        del procs[g]
        watch.reset(g)
        if rc != 0:
            status(state="failed", step=s, gpu=g, rc=rc, done=done, pid=os.getpid())
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
ok = all(os.path.exists(f"{R}/runs/maps/P-1.0-D@s{s}/p_core_trig.npz") for s in (440, 460, 480))
status(state="done" if ok else "failed", done=done, maps_present=ok, pid=os.getpid())
