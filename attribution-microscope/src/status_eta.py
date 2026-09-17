"""One-line progress + rolling ETA for the watchdog. Reads task start/done
timestamps from pipeline.log; unknown durations fall back to priors and are
replaced by measured medians as arms complete."""
import re
import subprocess
import time
from pathlib import Path

from common import CFG, RUNS

PRIOR_MIN = {"train": 150.0, "img": 50.0, "behav": 12.0, "traj": 40.0}
W2W3_PRIOR_H = 14.0  # 4+3 arms + imaging/controls, refined after W1


def kind_of(tid):
    for k in ("train", "behav", "traj", "img"):
        if tid.startswith(k + "_"):
            return k
    return "gate"


def main():
    text = (RUNS / "pipeline.log").read_text() if (RUNS / "pipeline.log").exists() else ""
    starts, dones = {}, {}
    for m in re.finditer(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) task (\S+) starting", text):
        starts[m.group(2)] = time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
    for m in re.finditer(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) task (\S+) done", text):
        dones[m.group(2)] = time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
    arms = [a["name"] for a in CFG["arms"]["wave1"]]
    todo = {"train": set("train_" + a for a in arms),
            "behav": set("behav_" + a for a in arms),
            "img": set("img_" + a for a in arms) | {"img_BASE"},
            "traj": set("traj_" + a for a in CFG["imaging"]["trajectory_arms"])}
    measured = {k: [] for k in PRIOR_MIN}
    done_ct, rem_ct = {}, {}
    for k, ids in todo.items():
        done_ids = {i for i in ids if i in dones}
        done_ct[k] = len(done_ids)
        rem_ct[k] = len(ids) - len(done_ids)
        for i in done_ids:
            if i in starts and dones[i] > starts[i]:
                measured[k].append((dones[i] - starts[i]) / 60.0)
    # running tasks get credit for elapsed time
    running = [i for i in starts if i not in dones]
    try:
        gpus = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,memory.used",
             "--format=csv,noheader,nounits"], text=True)
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,process_name",
             "--format=csv,noheader"], text=True)
        ours = {u.strip() for line in apps.strip().splitlines() if line
                for u, p in [line.split(",", 1)]
                if "envs/amic" in p or "attribution-microscope" in p}
        free = 0
        for line in gpus.strip().splitlines():
            u, used = [x.strip() for x in line.split(",")]
            # a GPU counts as available to us if it is idle or running OUR job
            if int(used) < CFG["scheduler"]["gpu_busy_mb"] or u in ours:
                free += 1
    except Exception:
        free = 1
    eta_min = 0.0
    avg = {}
    for k in PRIOR_MIN:
        avg[k] = (sorted(measured[k])[len(measured[k]) // 2]
                  if measured[k] else PRIOR_MIN[k])
        eta_min += rem_ct[k] * avg[k]
    eta_min /= max(free, 1)
    parts = " ".join(f"{k}:{done_ct[k]}/{done_ct[k] + rem_ct[k]}" for k in
                     ("train", "img", "behav", "traj"))
    run_s = ",".join(running[:3]) if running else "-"
    print(f"W1[{parts}] running={run_s} free_gpus={free} "
          f"avg_train={avg['train']:.0f}m avg_img={avg['img']:.0f}m "
          f"ETA_W1~{eta_min / 60:.1f}h  W2+W3~{W2W3_PRIOR_H / max(free, 1):.0f}h(prior)")


if __name__ == "__main__":
    main()
