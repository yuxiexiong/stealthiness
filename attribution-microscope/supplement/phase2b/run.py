"""Phase-2b runner (supplement/phase2b/PLAN.md): the three supplementary
experiments in one queue, the same way phase two ran.

  E1  P-1.0-D2: a second poisoned set (seed key poison2) trained with seed s2,
      a checkpoint every 20 steps, and its matched clean run RETRAIN-A-D
      (RETRAIN-A's dataset and seed, same saves). Imaging steps come from the
      ASR curve (rules2b.select_d2); the clean run is imaged on rules2b.null_grid.
  E2  P-0.38 and P-0.42 imaged like the trajectory checkpoints.
  E3  P-0.4-ps2 / P-0.4-ps3: the 0.4% dose drawn from two other poisoned
      sets, trained with s1, ASR measured, imaged like E2.

One task per GPU; a GPU is used only after it has been idle in memory and
utilisation for three minutes. Decisions are derived from files on disk and
every finished task leaves a done-marker, so a restart reaches the same
decisions and skips what is done. No verdicts are computed here
(verdicts.py runs only on the user's analysis command).

Server:   supplement/phase2b/launch.sh
Dry run:  P2B_DRYRUN=<scenario> P2_FAKE_SMI=<file> python supplement/phase2b/run.py
"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
from common import CFG, RUNS, log, is_done, mark_done, read_json, write_json  # noqa: E402
from scheduler import Task, gpu_fits, gpu_free_mb  # noqa: E402
from gpu_watch import GpuWatch  # noqa: E402
import rules2b as R  # noqa: E402

PY = sys.executable
DRY = os.environ.get("P2B_DRYRUN")
OUT = RUNS / "phase2b"
OUT.mkdir(parents=True, exist_ok=True)

D2, NULL = "P-1.0-D2", "RETRAIN-A-D"
D2_DATA = "P-1-ps2"                       # poison.py names rate 0.01 "P-1"
SAVE = 20
BEHAV_A = list(range(20, 641, 20))        # first ASR pass
BEHAV_B = list(range(660, 1241, 20))      # only if no 95% by step 640
BASE_NULL = list(range(80, 641, 80))      # imaged as soon as the clean run exists
PS_ARMS = ("P-0.4-ps2", "P-0.4-ps3")
E2_ARMS = ("P-0.38", "P-0.42")
MAX_ATTEMPTS = 2
POLL = 0.05 if DRY else CFG["scheduler"]["poll_s"]


def ck(arm, step):
    return str(RUNS / "arms" / arm / f"checkpoint-{step}")


def adapter(arm):
    return str(RUNS / "arms" / arm)


def asr(tag):
    p = RUNS / "behavioral" / f"{tag}.json"
    return read_json(p)["asr"] if p.exists() else None


def cmd(kind, *args):
    if DRY:
        return [PY, str(HERE / "fake_task2b.py"), kind, *args]
    return {
        "behav": [PY, str(ROOT / "supplement" / "phase2" / "behav_many.py")],
        "img": [PY, str(ROOT / "supplement" / "phase2" / "img_many.py")],
        "same": [PY, str(HERE / "same_run2b.py")],
        "train": [PY, str(ROOT / "src" / "train_arm.py")],
        "build": [PY, str(ROOT / "src" / "poison.py"), "--wave", "dose"],
    }[kind] + list(args)


def T(tid, kind, args, deps=(), prio=50, gpu=True, need=None):
    return Task(f"p2b_{tid}", cmd(kind, *args), deps=[f"p2b_{d}" for d in deps],
                gpu=gpu, prio=prio, need_mb=need)


def img(tid, tag, path, deps=(), prio=40):
    # phase two's imaging, unchanged: 60 trajectory samples, A on both columns, B on the triggered one
    return T(tid, "img", [f"{tag}={path}=AB"], deps=deps, prio=prio, need=64000)


def fixed():
    t = [
        # the builds share data/lf/dataset_info.json, so they run one after the other
        T("build_ps2", "build", ["--rates", "0.01,0.004", "--seed-key", "poison2", "--suffix", "ps2"],
          prio=1, gpu=False),
        T("build_ps3", "build", ["--rates", "0.004", "--seed-key", "poison3", "--suffix", "ps3"],
          deps=["build_ps2"], prio=2, gpu=False),
        # the matched clean run first: every E1 reading is against it
        T("train_null", "train", ["--arm", NULL, "--seed-key", "s2", "--gpu", "GPUSLOT",
                                  "--dataset-of", "RETRAIN-A", "--save-steps", str(SAVE), "--keep-all"],
          prio=10, need=88000),
        T("train_d2", "train", ["--arm", D2, "--seed-key", "s2", "--gpu", "GPUSLOT",
                                "--dataset-of", D2_DATA, "--save-steps", str(SAVE), "--keep-all"],
          deps=["build_ps2"], prio=11, need=88000),
        T("same_null", "same", ["RETRAIN-A", NULL], deps=["train_null"], prio=12, gpu=False),
        T("behav_d2_a", "behav", [f"{D2}@s{s}={ck(D2, s)}" for s in BEHAV_A] + [f"{D2}={adapter(D2)}"],
          deps=["train_d2"], prio=12, need=26000),
    ]
    t += [img(f"img_null_s{s}", f"{NULL}@s{s}", ck(NULL, s), deps=["train_null"], prio=13)
          for s in BASE_NULL]
    t += [T(f"train_{a}", "train", ["--arm", a, "--seed-key", "s1", "--gpu", "GPUSLOT"],
            deps=["build_ps3"], prio=20 + k, need=88000) for k, a in enumerate(PS_ARMS)]
    t.append(T("behav_ps", "behav", [f"{a}={adapter(a)}" for a in PS_ARMS],
               deps=[f"train_{a}" for a in PS_ARMS], prio=22, need=26000))
    t += [img(f"img_{a}", a, adapter(a), deps=[f"train_{a}"], prio=31) for a in PS_ARMS]
    t += [img(f"img_{a}", a, adapter(a), prio=30) for a in E2_ARMS]
    return t


def decide():
    """ASR extension, then the E1 imaging selection."""
    out = []
    if not is_done("task_p2b_behav_d2_a"):
        return out
    curve = [(s, asr(f"{D2}@s{s}")) for s in BEHAV_A]
    if any(v is None for _, v in curve):
        log("D2 ASR incomplete; decision postponed")
        return out
    if R.need_extension(curve):
        out.append(T("behav_d2_b", "behav", [f"{D2}@s{s}={ck(D2, s)}" for s in BEHAV_B],
                     deps=["behav_d2_a"], prio=12, need=26000))
        if not is_done("task_p2b_behav_d2_b"):
            return out
        curve += [(s, asr(f"{D2}@s{s}")) for s in BEHAV_B]
        if any(v is None for _, v in curve):
            return out
    sel = OUT / "selection.json"
    if not sel.exists():
        steps, status = R.select_d2(curve)
        t5, t95 = R.transition(curve)
        write_json(sel, {"status": status, "chosen": steps, "t5": t5, "t95": t95,
                         "null_grid": R.null_grid(max(steps)), "curve": curve})
        log(f"E1 selection: {status} t5={t5} t95={t95} -> {steps}")
    s = read_json(sel)
    out += [img(f"img_d2_s{st}", f"{D2}@s{st}", ck(D2, st), deps=["train_d2"], prio=14)
            for st in s["chosen"]]
    out += [img(f"img_null_s{st}", f"{NULL}@s{st}", ck(NULL, st), deps=["train_null"], prio=13)
            for st in s["null_grid"] if st not in BASE_NULL]
    return out


def main():
    gpus = list(CFG["scheduler"]["gpus"])
    watch = GpuWatch(gpus)
    known, running, failed = {}, [], []

    def refresh():
        for t in fixed() + decide():
            if t.tid not in known:
                known[t.tid] = t
        return [t for t in known.values() if t.proc is None
                and not is_done(f"task_{t.tid}") and t.tid not in failed]

    log(f"phase2b runner start{' (DRY: ' + DRY + ')' if DRY else ''}")
    while True:
        watch.tick()
        for t in running[:]:
            rc = t.proc.poll()
            if rc is None:
                continue
            running.remove(t)
            if t.assigned is not None:
                watch.reset(t.assigned)
            if rc == 0:
                mark_done(f"task_{t.tid}")
                log(f"{t.tid} done{'' if t.assigned is None else f' (GPU{t.assigned})'}")
            elif t.attempts < MAX_ATTEMPTS:
                log(f"{t.tid} FAILED rc={rc}; retrying")
                t.proc, t.assigned = None, None
            else:
                log(f"{t.tid} FAILED rc={rc}; giving up")
                failed.append(t.tid)
        pending = refresh()
        write_json(OUT / "status.json", {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "running": {t.tid: t.assigned for t in running},
            "pending": sorted(t.tid for t in pending),
            "done": sorted(tid for tid in known if is_done(f"task_{tid}")),
            "failed": failed,
            "gpu_idle": {g: watch.idle(g) for g in gpus}})
        if not pending and not running:
            break
        if not running and not any(t.ready() for t in pending):
            log(f"stuck: nothing running and nothing ready; pending {sorted(t.tid for t in pending)}")
            break
        busy = {t.assigned for t in running if t.assigned is not None}
        for t in sorted((t for t in pending if t.ready()), key=lambda t: t.prio):
            if not t.gpu:
                t.proc, t.assigned = subprocess.Popen(t.cmd), None
                t.attempts += 1
                running.append(t)
                log(f"{t.tid} (cpu) started")
                continue
            fits = (lambda g: True) if DRY else (lambda g: gpu_fits(g, t.need_mb))
            g = next((x for x in gpus if x not in busy and watch.idle(x) and fits(x)), None)
            if g is None:
                continue
            busy.add(g)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            t.proc = subprocess.Popen([c if c != "GPUSLOT" else str(g) for c in t.cmd], env=env)
            t.assigned = g
            t.attempts += 1
            running.append(t)
            log(f"{t.tid} started on GPU{g}" + ("" if DRY else f" (free {gpu_free_mb(g)}MB)"))
        time.sleep(POLL)
    write_json(OUT / "finished.json", {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "failed": failed})
    log(f"phase2b runner finished; failed: {failed or 'none'}")


if __name__ == "__main__":
    main()
