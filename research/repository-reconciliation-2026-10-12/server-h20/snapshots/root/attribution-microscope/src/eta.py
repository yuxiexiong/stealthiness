"""Live ETA for whatever wave is running.

Counts the tasks the scheduler still has to do and prices them with the rates
this box actually delivers (measured: 47 min per 200-sample imaging column,
75 min per training arm, 12 min per behavioural pass), then divides by the
GPUs available to us — a card running one of our own jobs still counts as
ours, a card held by somebody else does not.
"""
import subprocess

from common import CFG, RUNS, is_done

# Calibrated on this run. A 200-sample p_core column takes 47 min when it is
# the only imaging job, but 74 min when both GPUs image at once: the cards do
# not contend, the CPU-side data pipeline does. Since the scheduler keeps both
# busy, the parallel figure is the one that predicts wall-clock.
MIN = {"train": 75.0, "img": 74.0, "behav": 12.0, "traj": 40.0, "other": 5.0}
SMALL_COL = 7.5 / 47.0   # small probe columns scale with sample count        # p_seen and p_instrument relative to p_core


def kind(tid):
    for k in ("train", "behav", "traj", "img"):
        if tid.startswith(k + "_"):
            return k
    return "other"


def cost_min(t):
    """Imaging tasks price per column, not per task."""
    k = kind(t.tid)
    if k == "img":
        cols = 2
        for i, c in enumerate(t.cmd):
            if c == "--columns" and i + 1 < len(t.cmd):
                cols = len(t.cmd[i + 1].split(","))
        probes = 1
        for i, c in enumerate(t.cmd):
            if c == "--probes" and i + 1 < len(t.cmd):
                p = t.cmd[i + 1].split(",")
                # p_seen and p_instrument are 20 samples, a tenth of p_core
                probes = sum(1.0 if x == "p_core" else SMALL_COL for x in p)
        return MIN["img"] * cols * probes
    return MIN[k]


def gpus_available():
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
        n = 0
        for line in gpus.strip().splitlines():
            u, used = [x.strip() for x in line.split(",")]
            if int(used) < CFG["scheduler"]["gpu_busy_mb"] or u in ours:
                n += 1
        return max(n, 1)
    except Exception:
        return 2


def wave_remaining(wave, locked_rate=0.005):
    from scheduler import wave1_tasks, wave2_tasks, wave3_tasks
    builder = {1: wave1_tasks, 2: lambda: wave2_tasks(locked_rate), 3: wave3_tasks}[wave]
    try:
        ts = [t for t in builder() if not is_done(f"task_{t.tid}")]
    except Exception:
        return None, 0
    return len(ts), sum(cost_min(t) for t in ts)


def main():
    stage = ""
    p = RUNS / "run_all.log"
    if p.exists():
        for line in p.read_text(errors="ignore").splitlines():
            if line.startswith("[stage]"):
                stage = line
    cur = 3 if "Wave 3" in stage else (2 if "Wave 2" in stage else 1)
    g = gpus_available()
    parts, total = [], 0.0
    for w in range(cur, 4):
        n, mins = wave_remaining(w)
        if n is None:
            continue
        total += mins
        parts.append(f"W{w}:{n}t/{mins / 60:.1f}h")
    # print only — writing to pipeline.log would bury the experiment's own
    # progress lines under monitoring chatter
    print(f"ETA gpus={g} {' '.join(parts)} "
          f"remaining={total / 60:.1f} GPU-h -> {total / 60 / g:.1f} h wall")


if __name__ == "__main__":
    main()
