"""Two-GPU dependency scheduler for the arm queue (§4 order), with the
teammate guard (refuses GPUs that already hold >gpu_busy_mb of memory) and
gate tasks that stop the whole line on failure."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from common import CFG, RUNS, log, is_done, mark_done

SRC = Path(__file__).resolve().parent
PY = sys.executable


def gpu_free_mb(gpu):
    """Free MiB on a GPU, or -1 if unknown."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.used",
             "--format=csv,noheader,nounits"], text=True)
        for line in out.strip().splitlines():
            idx, total, used = [x.strip() for x in line.split(",")]
            if int(idx) == gpu:
                return int(total) - int(used)
    except Exception as e:
        log(f"nvidia-smi failed: {e}")
    return -1


def gpu_fits(gpu, need_mb):
    """A GPU is usable only if it can actually hold the task. Checking merely
    that it is 'mostly idle' let a training job start next to someone else's
    19GB process and OOM at step 0 (decisions.log D18)."""
    free = gpu_free_mb(gpu)
    return free >= need_mb if free >= 0 else False


class Task:
    def __init__(self, tid, cmd, deps=(), gpu=True, prio=100, need_mb=None):
        self.tid, self.cmd, self.deps = tid, cmd, list(deps)
        self.gpu, self.prio = gpu, prio
        # headroom the task actually needs: a 7B LoRA train run peaks near
        # 80GB, imaging/behavioral near 25GB
        if need_mb is None:
            need_mb = 82000 if tid.startswith("train_") else 26000
        self.need_mb = need_mb
        self.proc = None
        self.assigned = None
        self.attempts = 0

    def ready(self):
        return all(is_done(f"task_{d}") for d in self.deps)


def wave1_tasks():
    ts = []
    ts.append(Task("img_BASE",
                   [PY, str(SRC / "imaging_run.py"), "--tag", "BASE",
                    "--probes", "p_core,p_seen,p_instrument",
                    "--columns", "clean,trig"], prio=1))
    arms = CFG["arms"]["wave1"]
    for i, a in enumerate(arms):
        n, sk = a["name"], a["seed"]
        ts.append(Task(f"train_{n}",
                       [PY, str(SRC / "train_arm.py"), "--arm", n,
                        "--seed-key", sk, "--gpu", "GPUSLOT"],
                       prio=10 + i))
        ts.append(Task(f"behav_{n}",
                       [PY, str(SRC / "behavioral.py"), "--tag", n,
                        "--adapter", str(RUNS / "arms" / n)],
                       deps=[f"train_{n}"], prio=40 + i))
        ts.append(Task(f"img_{n}",
                       [PY, str(SRC / "imaging_run.py"), "--tag", n,
                        "--adapter", str(RUNS / "arms" / n),
                        "--probes", "p_core,p_seen,p_instrument",
                        "--columns", "clean,trig"],
                       deps=[f"train_{n}"], prio=50 + i))
    for arm in CFG["imaging"]["trajectory_arms"]:
        ts.append(Task(f"traj_{arm}",
                       [PY, str(SRC / "traj_image.py"), "--arm", arm],
                       deps=[f"train_{arm}"], prio=80))
    ts.append(Task("gate_null",
                   [PY, str(SRC / "gates.py"), "--check", "null"],
                   deps=["img_CLEAN", "img_RETRAIN-A", "img_RETRAIN-B"],
                   gpu=False, prio=5))
    ts.append(Task("gate_asr",
                   [PY, str(SRC / "gates.py"), "--check", "asr", "--tag", "P-5.0"],
                   deps=["behav_P-5.0"], gpu=False, prio=5))
    return ts


def wave2_tasks(locked_rate):
    w2 = CFG["poison"]["wave2"]
    arms = [(f"S-{s}", f"trig_s{s}" if s != 28 else "trig") for s in w2["sizes_px"]]
    arms.append(("S-28-a03", "trig_s28a03"))
    ts = []
    extra_cols = sorted({c for _, c in arms if c != "trig"})
    for i, (n, col) in enumerate(arms):
        ts.append(Task(f"train_{n}",
                       [PY, str(SRC / "train_arm.py"), "--arm", n,
                        "--seed-key", "s1", "--gpu", "GPUSLOT"], prio=10 + i))
        ts.append(Task(f"behav_{n}",
                       [PY, str(SRC / "behavioral.py"), "--tag", n,
                        "--adapter", str(RUNS / "arms" / n), "--trig-col", col],
                       deps=[f"train_{n}"], prio=40 + i))
        ts.append(Task(f"img_{n}",
                       [PY, str(SRC / "imaging_run.py"), "--tag", n,
                        "--adapter", str(RUNS / "arms" / n),
                        "--probes", "p_core,p_instrument",
                        "--columns", f"clean,{col}"],
                       deps=[f"train_{n}"], prio=50 + i))
    # controls imaged on the new trigger columns (null bands per column, F17)
    for ctrl in ("CLEAN", "RETRAIN-A", "RETRAIN-B"):
        ts.append(Task(f"img_{ctrl}@w2",
                       [PY, str(SRC / "imaging_run.py"), "--tag", ctrl,
                        "--adapter", str(RUNS / "arms" / ctrl),
                        "--probes", "p_core",
                        "--columns", ",".join(extra_cols)], prio=5))
    return ts


def wave3_tasks():
    ts = []
    for i, rate in enumerate(CFG["poison"]["wave3_rates"]):
        n = f"T-{rate*100:g}"
        ts.append(Task(f"train_{n}",
                       [PY, str(SRC / "train_arm.py"), "--arm", n,
                        "--seed-key", "s1", "--gpu", "GPUSLOT"], prio=10 + i))
        ts.append(Task(f"behav_{n}",
                       [PY, str(SRC / "behavioral.py"), "--tag", n,
                        "--adapter", str(RUNS / "arms" / n), "--mode", "text"],
                       deps=[f"train_{n}"], prio=40 + i))
        ts.append(Task(f"img_{n}",
                       [PY, str(SRC / "imaging_run.py"), "--tag", n,
                        "--adapter", str(RUNS / "arms" / n),
                        "--probes", "p_core,p_instrument",
                        "--columns", "clean,texttrig"],
                       deps=[f"train_{n}"], prio=50 + i))
    for ctrl in ("CLEAN", "RETRAIN-A", "RETRAIN-B"):
        ts.append(Task(f"img_{ctrl}@w3",
                       [PY, str(SRC / "imaging_run.py"), "--tag", ctrl,
                        "--adapter", str(RUNS / "arms" / ctrl),
                        "--probes", "p_core", "--columns", "texttrig"], prio=5))
    return ts


MAX_ATTEMPTS = 2


def run(tasks):
    tasks = [t for t in tasks if not is_done(f"task_{t.tid}")]
    gpus = list(CFG["scheduler"]["gpus"])
    running, failed = [], []
    while tasks or running:
        for t in running[:]:
            rc = t.proc.poll()
            if rc is None:
                continue
            running.remove(t)
            if rc == 0:
                mark_done(f"task_{t.tid}")
                log(f"task {t.tid} done (GPU{t.assigned})")
            elif t.attempts < MAX_ATTEMPTS:
                # a single task failing is an execution problem, not a reason
                # to kill healthy siblings (that orphaned a live training run
                # once already) — requeue it and carry on (decisions.log D18)
                log(f"task {t.tid} FAILED rc={rc}; requeueing "
                    f"(attempt {t.attempts}/{MAX_ATTEMPTS})")
                t.proc, t.assigned = None, None
                tasks.append(t)
            else:
                log(f"task {t.tid} FAILED rc={rc}; giving up after "
                    f"{t.attempts} attempts — continuing with other tasks")
                failed.append(t.tid)
        # only a protocol gate (HALT.json) stops the line
        if (RUNS / "HALT.json").exists():
            log("HALT.json present; stopping scheduler")
            for t in running:
                t.proc.terminate()
            sys.exit(3)
        busy = {t.assigned for t in running if t.assigned is not None}
        ready = sorted([t for t in tasks if t.ready()], key=lambda t: t.prio)
        for t in ready:
            if not t.gpu:
                tasks.remove(t)
                log(f"task {t.tid} (cpu) starting")
                t.proc = subprocess.Popen(t.cmd)
                t.assigned = None
                t.attempts += 1
                running.append(t)
                continue
            g = next((x for x in gpus if x not in busy and gpu_fits(x, t.need_mb)),
                     None)
            if g is None:
                continue
            busy.add(g)
            tasks.remove(t)
            cmd = [c if c != "GPUSLOT" else str(g) for c in t.cmd]
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(g)
            log(f"task {t.tid} starting on GPU{g} "
                f"(needs {t.need_mb}MB, free {gpu_free_mb(g)}MB)")
            t.proc = subprocess.Popen(cmd, env=env)
            t.assigned = g
            t.attempts += 1
            running.append(t)
        time.sleep(CFG["scheduler"]["poll_s"])
    if failed:
        log(f"scheduler: finished with failed tasks: {', '.join(failed)}")
        sys.exit(4)
    log("scheduler: all tasks complete")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", required=True)
    ap.add_argument("--locked-rate", type=float, default=None)
    a = ap.parse_args()
    if a.wave == "1":
        run(wave1_tasks())
    elif a.wave == "2":
        run(wave2_tasks(a.locked_rate))
    elif a.wave == "3":
        run(wave3_tasks())


if __name__ == "__main__":
    main()
