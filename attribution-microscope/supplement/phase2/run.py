"""Phase-two runner (supplement/phase2/PLAN.md).

Both lines share one queue; one task per GPU at a time, and a GPU is used
only after gpu_watch has seen it idle in memory AND utilisation for three
minutes. Three points look at results before deciding what comes next:
  - line 1, after the dense trajectory's ASR: which checkpoints to image
    (rules.select_checkpoints)
  - line 2, after each dose round's ASR: which doses to add
    (rules.next_doses), at most two refinement rounds
  - instrument B's measured cost against the plan's overrun rule
Every decision is derived from result files on disk, and every finished task
leaves a done-marker, so a restart reaches the same decisions and skips what
is done.

Server:   cd /root/attribution-microscope && \\
          nohup python supplement/phase2/run.py > runs/phase2/run.out 2>&1 &
Dry run:  P2_DRYRUN=<scenario> P2_FAKE_SMI=<file> python supplement/phase2/run.py
          (fake GPU samples and fake tasks; see fake_task.py)
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
import rules  # noqa: E402

PY = sys.executable
DRY = os.environ.get("P2_DRYRUN")
P2 = RUNS / "phase2"
P2.mkdir(parents=True, exist_ok=True)

EXISTING = (79, 158, 316, 632)              # saves kept from P-1.0 and CLEAN
OLD_LABEL = {79: "k1", 158: "k2", 632: "k4"}  # D49: k4 is step 632
DENSE_ARM, DENSE_SAVE = "P-1.0-D", 20
DENSE_STEPS = list(range(140, 641, 20))       # L1-3: 26 checkpoints
ROUND1 = [0.002, 0.003, 0.004]                # L2-1
KNOWN = {0.001: "P-0.1", 0.005: "P-0.5"}      # measured doses bracketing the gap
MAX_REFINE = 2
B_EST_S, B_OVERRUN = 13 * 60, 1.5             # 60 images x ~13s; overrun at +50%
MAX_ATTEMPTS = 2
POLL = 0.05 if DRY else CFG["scheduler"]["poll_s"]


def ck(arm, step):
    return str(RUNS / "arms" / arm / f"checkpoint-{step}")


def tag(arm, step):
    return f"{arm}@s{step}"


def dose_arm(rate):
    return f"P-{rate * 100:g}"


def asr(t):
    p = RUNS / "behavioral" / f"{t}.json"
    return read_json(p)["asr"] if p.exists() else None


def cmd(kind, *args):
    """The real command, or fake_task.py with the same arguments in a dry run."""
    if DRY:
        return [PY, str(HERE / "fake_task.py"), kind, *args]
    return {
        "behav": [PY, str(HERE / "behav_many.py")],
        "img": [PY, str(HERE / "img_many.py")],
        "checkA": [PY, str(HERE / "check_A.py")],
        "same": [PY, str(HERE / "same_run.py")],
        "train": [PY, str(ROOT / "src" / "train_arm.py")],
        "build": [PY, str(ROOT / "src" / "poison.py"), "--wave", "dose"],
        "imgfull": [PY, str(ROOT / "src" / "imaging_run.py")],
    }[kind] + list(args)


def T(tid, kind, args, deps=(), prio=50, gpu=True, need=None):
    return Task(f"p2_{tid}", cmd(kind, *args), deps=[f"p2_{d}" for d in deps],
                gpu=gpu, prio=prio, need_mb=need)


def b_overrun():
    p = P2 / "b_timing.json"
    return p.exists() and read_json(p)["seconds"] > B_EST_S * B_OVERRUN


# ---------------------------------------------------------------- line 1
def line1_fixed():
    exist = [f"{tag(a, s)}={ck(a, s)}" for a in ("CLEAN", "P-1.0") for s in EXISTING]
    return [
        # the control first (CLEAN leads the list), as every reading is against it
        T("l1_behav_existing", "behav", exist, prio=10, need=26000),
        # one checkpoint imaged first: it prices instrument B and checks that
        # A recomputed now matches the A imaged on 09-18 before anything else
        T("l1_img_probe", "img", [f"{tag('CLEAN', 79)}={ck('CLEAN', 79)}=AB"], prio=11, need=64000),
        T("l1_checkA_probe", "checkA", [f"{tag('CLEAN', 79)}=CLEAN@k1"],
          deps=["l1_img_probe"], prio=12, gpu=False),
        T("l1_train_dense", "train", ["--arm", DENSE_ARM, "--seed-key", "s1", "--gpu", "GPUSLOT",
                                      "--dataset-of", "P-1.0", "--save-steps", str(DENSE_SAVE),
                                      "--keep-all"], prio=13, need=88000),
        T("l1_same_run", "same", [], deps=["l1_train_dense"], prio=14, gpu=False),
        T("l1_behav_dense", "behav", [f"{tag(DENSE_ARM, s)}={ck(DENSE_ARM, s)}" for s in DENSE_STEPS],
          deps=["l1_train_dense"], prio=15, need=26000),
    ]


def line1_decide():
    """Existing-checkpoint imaging, then the dense selection."""
    out = []
    halted = (P2 / "HALT_line1").exists()
    if is_done("task_p2_l1_checkA_probe") and is_done("task_p2_l1_behav_existing") and not halted:
        over = b_overrun()
        items = []
        for a in ("CLEAN", "P-1.0"):
            for s in EXISTING:
                if not over:
                    ins = "AB"
                elif a == "P-1.0":
                    v = asr(tag(a, s))
                    ins = "AB" if v is not None and 0.05 <= v <= 0.95 else "A"
                else:
                    ins = "AB" if s == 316 else "A"   # one mid-training control
                items.append(f"{tag(a, s)}={ck(a, s)}={ins}")
        out.append(T("l1_img_existing", "img", items, prio=17, need=64000))
        pairs = [f"{tag(a, s)}={a}@{OLD_LABEL[s]}" for a in ("CLEAN", "P-1.0") for s in OLD_LABEL]
        out.append(T("l1_checkA_all", "checkA", pairs, deps=["l1_img_existing"], prio=18, gpu=False))
    if is_done("task_p2_l1_behav_dense") and not halted:
        sel = P2 / "selection.json"
        if not sel.exists():
            curve = [(s, asr(tag(DENSE_ARM, s))) for s in DENSE_STEPS]
            if any(v is None for _, v in curve):
                log("dense ASR incomplete; selection postponed")
                return out
            chosen, status = rules.select_checkpoints(curve)
            write_json(sel, {"status": status, "chosen": chosen, "curve": curve})
            log(f"L1-4 selection: {status} {chosen}")
        s = read_json(sel)
        if s["status"] == "ok" and is_done("task_p2_l1_checkA_probe"):
            chosen = s["chosen"]
            if b_overrun():                       # B only inside the window
                lo = next(st for st, v in s["curve"] if v >= 0.05)
                hi = next(st for st, v in s["curve"] if v >= 0.95)
                chosen = [st for st in chosen if lo <= st <= hi]
            items = [f"{tag(DENSE_ARM, st)}={ck(DENSE_ARM, st)}=AB" for st in chosen]
            out.append(T("l1_img_dense", "img", items, prio=16, need=64000))
    return out


# ---------------------------------------------------------------- line 2
def dose_plan():
    p = P2 / "doses.json"
    return read_json(p) if p.exists() else {"rounds": [ROUND1], "final": None, "mids": []}


def line2_tasks():
    plan, out = dose_plan(), []
    for k, rates in enumerate(plan["rounds"]):
        arms = [dose_arm(r) for r in rates]
        out.append(T(f"l2_build_r{k}", "build", ["--rates", ",".join(f"{r:g}" for r in rates)],
                     prio=40 + 10 * k, gpu=False))
        for j, a in enumerate(arms):
            out.append(T(f"l2_train_{a}", "train", ["--arm", a, "--seed-key", "s1", "--gpu", "GPUSLOT"],
                         deps=[f"l2_build_r{k}"], prio=41 + 10 * k + j, need=88000))
        out.append(T(f"l2_behav_r{k}", "behav", [f"{a}={RUNS / 'arms' / a}" for a in arms],
                     deps=[f"l2_train_{a}" for a in arms], prio=45 + 10 * k, need=26000))
    for a in plan["mids"]:
        out.append(T(f"l2_img_{a}", "imgfull", ["--tag", a, "--adapter", str(RUNS / "arms" / a)],
                     prio=70, need=64000))
    return out


def line2_decide():
    plan = dose_plan()
    k = len(plan["rounds"]) - 1
    if plan["final"] is not None or not is_done(f"task_p2_l2_behav_r{k}"):
        return
    results = {r: asr(n) for r, n in KNOWN.items()}
    for rates in plan["rounds"]:
        for r in rates:
            results[r] = asr(dose_arm(r))
    if any(v is None for v in results.values()):
        log(f"dose ASR incomplete {results}; decision postponed")
        return
    new, status = rules.next_doses(results)
    if status == "refine" and k < MAX_REFINE:
        plan["rounds"].append(new)
        log(f"L2-2 round {k + 1}: adding {new}")
    else:
        plan["final"] = status if status != "refine" else "max_rounds"
        plan["mids"] = sorted((dose_arm(r) for r, v in results.items()
                               if 0.20 <= v <= 0.80 and r not in KNOWN),
                              key=lambda a: float(a[2:]))
        log(f"L2-2 final: {plan['final']}; intermediate-ASR models {plan['mids']}")
    plan["results"] = {f"{r:g}": v for r, v in sorted(results.items())}
    write_json(P2 / "doses.json", plan)


# ---------------------------------------------------------------- loop
def main():
    gpus = list(CFG["scheduler"]["gpus"])
    watch = GpuWatch(gpus)
    known = {}
    running, failed = [], []

    def refresh():
        line2_decide()
        for t in line1_fixed() + line1_decide() + line2_tasks():
            if t.tid not in known:
                known[t.tid] = t
        if (P2 / "HALT_line1").exists():
            for tid, t in known.items():
                if tid.startswith("p2_l1_img") and t.proc is None and not is_done(f"task_{tid}"):
                    t.cancelled = True
        return [t for t in known.values() if t.proc is None and not getattr(t, "cancelled", False)
                and not is_done(f"task_{t.tid}") and t.tid not in failed]

    log(f"phase2 runner start{' (DRY: ' + DRY + ')' if DRY else ''}")
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
        write_json(P2 / "status.json", {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "running": {t.tid: t.assigned for t in running},
            "pending": sorted(t.tid for t in pending),
            "done": sorted(tid for tid in known if is_done(f"task_{tid}")),
            "failed": failed,
            "gpu_idle": {g: watch.idle(g) for g in gpus}})
        if not pending and not running:
            break
        if not running and not any(t.ready() for t in pending):
            # everything left waits on a task that failed for good or was
            # cancelled by a gate; waiting longer cannot change that
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
    write_json(P2 / "finished.json", {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "failed": failed,
                                      "halt_line1": (P2 / "HALT_line1").exists()})
    log(f"phase2 runner finished; failed: {failed or 'none'}")


if __name__ == "__main__":
    main()
