"""Qwen3-VL-8B runner: stage one (Wave 1) and stage two (phase-2 trajectory
and dose lines) in ONE queue, as the user asked (plan:
attribution-microscope/supplement/qwen3vl8b/PLAN.md).

The machinery is LLaVA's phase-2 runner (supplement/phase2/run.py): one task
per GPU, a GPU used only after gpu_watch has seen it idle for three minutes,
every decision derived from result files on disk, every finished task leaves
a done-marker, so a restart reaches the same decisions and skips what is done.

Order (priorities; lower runs first when its dependencies are met):
  0  CPU: selftest, shared constants, weights, 768px inputs, target word,
          datasets (checked against LLaVA's), LF-vs-engine format
  1  GPU adaptation checks: geometry, forward, precision (+ fp16 training
          smoke), timing. Any failure stops the line.
  (Q20: dose refinement runs right after the Wave-1 trainings and their
     behaviour, before any imaging; all imaging comes last. Q21/Q22: the P-1.0
     trajectory ASR runs before the remaining Wave-1 trainings and the dose refinement.)
  2  W0 qualification and the W0 T-scalar pointing, BASE full imaging —
     queued after the trainings (Q13: W0 gates nothing, trainings are the long pole)
  3  Wave 1: CLEAN first (control first), P-5.0 next (the execution gate),
          then LLaVA's queue. CLEAN and P-1.0 save every 20 steps and keep all
          checkpoints: they are also the stage-two trajectory pair.
          P-5.0 ASR < 0.90 writes HALT.json: nothing new starts (PROTOCOL s.10).
  4  Stage two, line 1: ASR of P-1.0 at every 20-step save (20-640, extended
          to 1240 if no save reaches 95%), CLEAN and P-1.0 at 80/160/320/640,
          the P-1.0 saves chosen by select_d2; 60 trajectory samples, A on
          clean+trig, B on trig (LLaVA's trajectory imaging).
     Stage two, line 2: dose refinement from Wave 1's four anchors
          (first_round, then LLaVA's next_doses, at most two refinement
          rounds), intermediate-ASR models imaged in full.

A card is held back while runs/q3/HOLD_GPU<n> exists (as phase 2b); deleting
the file releases it without a restart.

Server:   cd /root/amic-q3 && bash run/launch.sh
Dry run:  Q3_DRYRUN=<scenario> Q3_FAKE_SMI=<file> python run/run_q3.py
"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
DRY = os.environ.get("Q3_DRYRUN")
if DRY and os.environ.get("Q3_FAKE_SMI"):
    os.environ["P2_FAKE_SMI"] = os.environ["Q3_FAKE_SMI"]   # gpu_watch's hook
from common import CFG, RUNS, log, is_done, mark_done, read_json, write_json  # noqa: E402
from scheduler import Task, gpu_fits, gpu_free_mb  # noqa: E402
from gpu_watch import GpuWatch  # noqa: E402
import rules  # noqa: E402

PY = sys.executable
OUT = RUNS / "q3"
OUT.mkdir(parents=True, exist_ok=True)
HALT = RUNS / "HALT.json"
MAX_ATTEMPTS = 2
POLL = 0.05 if DRY else CFG["scheduler"]["poll_s"]

DENSE_SAVE = 20
DENSE_ARMS = ("CLEAN", "P-1.0")                    # trajectory pair, s1
BEHAV_STEPS = list(range(20, 641, 20))              # phase 2b's 20-640 grid
EXT_STEPS = list(range(660, 1241, 20))              # need_extension
CTRL_STEPS = (80, 160, 320, 640)                    # LLaVA: 79/158/316/632 (16-save grid)
ANCHORS = {0.001: "P-0.1", 0.005: "P-0.5", 0.01: "P-1.0", 0.05: "P-5.0"}
MAX_REFINE = 2
FULL = ["--probes", "p_core,p_seen,p_instrument", "--columns", "clean,trig"]
NEED_TRAIN, NEED_IMG, NEED_BEHAV = 88000, 64000, 26000   # LLaVA's measured peaks (D29);
                                                          # precision check re-measures for Qwen


def arm_dir(arm):
    return RUNS / "arms" / arm


def ck(arm, step):
    return str(arm_dir(arm) / f"checkpoint-{step}")


def tag(arm, step):
    return f"{arm}@s{step}"


def dose_arm(rate):
    return f"P-{rate * 100:g}"


def asr(t):
    p = RUNS / "behavioral" / f"{t}.json"
    return read_json(p)["asr"] if p.exists() else None


def cmd(kind, *args):
    if DRY:
        return [PY, str(HERE / "fake_task.py"), kind, *args]
    src = ROOT / "src"
    return {
        "selftest": [PY, str(src / "selftest.py")],
        "check": [PY, str(src / "q3_checks.py")],
        "inputs": [PY, str(src / "build_q3_inputs.py")],
        "poison": [PY, str(src / "poison.py"), "--wave", "1"],
        "build": [PY, str(src / "poison.py"), "--wave", "dose"],
        "w0": [PY, str(src / "gates.py"), "--check", "w0-qualify"],
        "w0t3": [PY, str(src / "w0_t3_pointing.py")],
        "train": [PY, str(src / "train_arm.py")],
        "behav1": [PY, str(src / "behavioral.py")],
        "imgfull": [PY, str(src / "imaging_run.py")],
        "gate_null": [PY, str(src / "gates.py"), "--check", "null"],
        "gate_asr": [PY, str(HERE / "gate_asr.py")],
        "causal": [PY, str(src / "trigger_causality.py")],
        "behav": [PY, str(HERE / "behav_many.py")],
        "img": [PY, str(HERE / "img_many.py")],
        "metrics": [PY, str(src / "metrics.py"), "--wave", "1"],
    }[kind] + list(args)


def T(tid, kind, args=(), deps=(), prio=50, gpu=True, need=None):
    return Task(f"q3_{tid}", cmd(kind, *args), deps=[f"q3_{d}" for d in deps],
                gpu=gpu, prio=prio, need_mb=need or NEED_BEHAV)


# ---------------------------------------------------------------- stage 0
CPU_CHAIN = [("selftest", "selftest", []), ("shared", "check", ["shared"]),
             ("weights", "check", ["weights"]), ("inputs", "inputs", []),
             ("target", "check", ["target"]), ("poison", "poison", []),
             ("format", "check", ["format"])]
GPU_CHECKS = [("geometry", NEED_IMG), ("forward", NEED_IMG),
              ("precision", NEED_TRAIN), ("timing", NEED_IMG)]


def stage0():
    out, prev = [], []
    for k, (tid, kind, args) in enumerate(CPU_CHAIN):
        out.append(T(tid, kind, args, deps=prev, prio=k, gpu=False))
        prev = [tid]
    for k, (name, need) in enumerate(GPU_CHECKS):
        out.append(T(f"chk_{name}", "check", [name], deps=["format"], prio=10 + k, need=need))
    return out


CHECKED = ["format"] + [f"chk_{n}" for n, _ in GPU_CHECKS]


# ---------------------------------------------------------------- stage 1
def wave1():
    out = [
        T("w0", "w0", deps=CHECKED, prio=158, need=NEED_IMG),
        T("w0t3", "w0t3", deps=CHECKED, prio=159, need=NEED_IMG),
        T("img_BASE", "imgfull", ["--tag", "BASE", *FULL], deps=CHECKED, prio=160, need=NEED_IMG),
    ]
    arms = {a["name"]: a for a in CFG["arms"]["wave1"]}
    order = ["CLEAN", "P-5.0"] + [n for n in arms if n not in ("CLEAN", "P-5.0")]
    for i, n in enumerate(order):
        a = arms[n]
        args = ["--arm", n, "--seed-key", a["seed"], "--gpu", "GPUSLOT"]
        if n in DENSE_ARMS:
            args += ["--save-steps", str(DENSE_SAVE), "--keep-all"]
        out.append(T(f"train_{n}", "train", args, deps=CHECKED, prio=30 + i, need=NEED_TRAIN))
        out.append(T(f"behav_{n}", "behav1", ["--tag", n, "--adapter", str(arm_dir(n))],
                     deps=[f"train_{n}"], prio=25 if n == "P-5.0" else 45 + i))
        out.append(T(f"img_{n}", "imgfull", ["--tag", n, "--adapter", str(arm_dir(n)), *FULL],
                     deps=[f"train_{n}"], prio=161 + i, need=NEED_IMG))
    out.append(T("gate_asr", "gate_asr", deps=["behav_P-5.0"], prio=26, gpu=False))
    out.append(T("gate_null", "gate_null", deps=["img_CLEAN", "img_RETRAIN-A", "img_RETRAIN-B"],
                 prio=75, gpu=False))
    for n in ("P-5.0", "P-1.0"):
        out.append(T(f"causal_{n}", "causal", ["--arm", n], deps=[f"train_{n}"], prio=27))
    imgs = ["img_BASE"] + [f"img_{n}" for n in order]
    out.append(T("metrics_w1", "metrics", deps=imgs, prio=99, gpu=False))
    return out


# ---------------------------------------------------------------- stage 2, line 1
def curve(arm, steps):
    return [(s, asr(tag(arm, s))) for s in steps]


def line1():
    out = [T("l1_behav", "behav", [f"{tag('P-1.0', s)}={ck('P-1.0', s)}" for s in BEHAV_STEPS]
             + [f"{tag('CLEAN', s)}={ck('CLEAN', s)}" for s in CTRL_STEPS],
             deps=["train_P-1.0", "train_CLEAN"], prio=29)]
    # controls first: CLEAN and P-1.0 at LLaVA's four fixed points
    ctrl = [f"{tag(a, s)}={ck(a, s)}=AB" for a in ("CLEAN", "P-1.0") for s in CTRL_STEPS]
    out.append(T("l1_img_ctrl", "img", ctrl, deps=["train_P-1.0", "train_CLEAN"], prio=150, need=NEED_IMG))
    if not is_done("task_q3_l1_behav"):
        return out
    c = curve("P-1.0", BEHAV_STEPS)
    if any(v is None for _, v in c):
        log("P-1.0 trajectory ASR incomplete; selection postponed")
        return out
    steps = BEHAV_STEPS
    if rules.need_extension(c):
        out.append(T("l1_behav_ext", "behav", [f"{tag('P-1.0', s)}={ck('P-1.0', s)}" for s in EXT_STEPS],
                     deps=["l1_behav"], prio=29))
        if not is_done("task_q3_l1_behav_ext"):
            return out
        steps = BEHAV_STEPS + EXT_STEPS
        c = curve("P-1.0", steps)
    sel = OUT / "selection.json"
    if not sel.exists():
        chosen, status = rules.select_d2(c)
        t5, t95 = rules.transition(c)
        write_json(sel, {"status": status, "chosen": chosen, "t5": t5, "t95": t95, "curve": c,
                         "rule": "select_d2 (LLaVA phase 2b)"})
        log(f"line 1 selection: {status} t5={t5} t95={t95} {chosen}")
    s = read_json(sel)
    todo = [st for st in s["chosen"] if st not in CTRL_STEPS]
    out.append(T("l1_img_dense", "img", [f"{tag('P-1.0', st)}={ck('P-1.0', st)}=AB" for st in todo],
                 deps=["l1_behav"], prio=151, need=NEED_IMG))
    return out


# ---------------------------------------------------------------- stage 2, line 2
def dose_plan():
    p = OUT / "doses.json"
    return read_json(p) if p.exists() else None


def line2_decide():
    """First round from the four Wave-1 anchors, then LLaVA's next_doses."""
    anchors_done = all(is_done(f"task_q3_behav_{n}") for n in ANCHORS.values())
    if not anchors_done:
        return
    plan = dose_plan()
    anchors = {r: asr(n) for r, n in ANCHORS.items()}
    if plan is None:
        first, status = rules.first_round(anchors)
        plan = {"anchors": {f"{r:g}": v for r, v in anchors.items()}, "rounds": [],
                "final": None, "mids": [], "first_status": status}
        if first:
            plan["rounds"].append(first)
            log(f"line 2 first round ({status}): {first}")
        else:
            plan["final"] = status
            plan["mids"] = [ANCHORS[r] for r, v in anchors.items() if 0.20 <= v <= 0.80]
            log(f"line 2: no first round ({status})")
        write_json(OUT / "doses.json", plan)
        return
    if plan["final"] is not None:
        return
    k = len(plan["rounds"]) - 1
    if not is_done(f"task_q3_l2_behav_r{k}"):
        return
    results = dict(anchors)
    for rates in plan["rounds"]:
        for r in rates:
            results[r] = asr(dose_arm(r))
    if any(v is None for v in results.values()):
        return
    new, status = rules.next_doses(results)
    if status == "refine" and k < MAX_REFINE:
        plan["rounds"].append(new)
        log(f"line 2 round {k + 1}: adding {new}")
    else:
        plan["final"] = status if status != "refine" else "max_rounds"
        plan["mids"] = sorted((dose_arm(r) for r, v in results.items()
                               if 0.20 <= v <= 0.80 and r not in ANCHORS),
                              key=lambda a: float(a[2:]))
        log(f"line 2 final: {plan['final']}; intermediate-ASR models {plan['mids']}")
    plan["results"] = {f"{r:g}": v for r, v in sorted(results.items())}
    write_json(OUT / "doses.json", plan)


def line2():
    plan, out = dose_plan(), []
    if plan is None:
        return out
    for k, rates in enumerate(plan["rounds"]):
        arms = [dose_arm(r) for r in rates]
        out.append(T(f"l2_build_r{k}", "build", ["--rates", ",".join(f"{r:g}" for r in rates)],
                     deps=CHECKED, prio=40, gpu=False))
        for j, a in enumerate(arms):
            out.append(T(f"l2_train_{a}", "train", ["--arm", a, "--seed-key", "s1", "--gpu", "GPUSLOT"],
                         deps=[f"l2_build_r{k}"], prio=41 + 3 * k + j, need=NEED_TRAIN))
        out.append(T(f"l2_behav_r{k}", "behav", [f"{a}={arm_dir(a)}" for a in arms],
                     deps=[f"l2_train_{a}" for a in arms], prio=40))
    for a in plan["mids"]:
        if a in ANCHORS.values():
            continue                     # anchors are imaged in full by Wave 1
        out.append(T(f"l2_img_{a}", "imgfull", ["--tag", a, "--adapter", str(arm_dir(a)), *FULL],
                     prio=195, need=NEED_IMG))
    return out


# ---------------------------------------------------------------- loop
def live_training(tid):
    """Q19: a train task whose arm already has a training process (started by
    hand while cards were held, Q17) only waits for it (train_arm adopts the
    run); it must not take a GPU slot while it waits."""
    if DRY or not tid.startswith("q3_train_"):
        return False
    arm = tid[len("q3_train_"):]
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    return any(f"configs/{arm}.yaml" in l and "llamafactory-cli" in l for l in out.splitlines())


def held(gpus):
    return {g for g in gpus if (OUT / f"HOLD_GPU{g}").exists()}


def main():
    gpus = list(CFG["scheduler"]["gpus"])
    watch = GpuWatch(gpus)
    known, running, failed = {}, [], []

    def refresh():
        line2_decide()
        for t in stage0() + wave1() + line1() + line2():
            if t.tid not in known:
                known[t.tid] = t
        return [t for t in known.values() if t.proc is None and not is_done(f"task_{t.tid}")
                and t.tid not in failed]

    log(f"q3 runner start{' (DRY: ' + DRY + ')' if DRY else ''}")
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
            elif t.attempts < MAX_ATTEMPTS and not t.tid.startswith(("q3_chk_", "q3_selftest", "q3_shared",
                                                                       "q3_weights", "q3_target", "q3_format")):
                log(f"{t.tid} FAILED rc={rc}; retrying")
                t.proc, t.assigned = None, None
            else:
                # a failed check is a verdict, not a flake: no retry
                log(f"{t.tid} FAILED rc={rc}; giving up")
                failed.append(t.tid)
        pending = refresh()
        halted = HALT.exists()
        hold = held(gpus)
        write_json(OUT / "status.json", {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "running": {t.tid: t.assigned for t in running},
            "pending": sorted(t.tid for t in pending),
            "done": sorted(tid for tid in known if is_done(f"task_{tid}")),
            "failed": failed, "halted": halted, "held": sorted(hold),
            "gpu_idle": {g: watch.idle(g) for g in gpus}})
        if halted and not running:
            log(f"HALT: {read_json(HALT).get('reason')}; nothing new started")
            break
        if not pending and not running:
            break
        if not running and not any(t.ready() for t in pending):
            log(f"stuck: nothing running and nothing ready; pending {sorted(t.tid for t in pending)}")
            break
        busy = {t.assigned for t in running if t.assigned is not None}
        for t in sorted((t for t in pending if t.ready()), key=lambda t: t.prio):
            if halted:
                break
            if t.gpu and live_training(t.tid):
                t.gpu = False
                log(f"{t.tid}: its training is already running outside the runner; waiting on CPU (Q19)")
            if not t.gpu:
                t.proc, t.assigned = subprocess.Popen(t.cmd), None
                t.attempts += 1
                running.append(t)
                log(f"{t.tid} (cpu) started")
                continue
            fits = (lambda g: True) if DRY else (lambda g: gpu_fits(g, t.need_mb))
            g = next((x for x in gpus if x not in busy and x not in hold and watch.idle(x) and fits(x)), None)
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
    write_json(OUT / "finished.json", {"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                       "failed": failed, "halted": HALT.exists()})
    log(f"q3 runner finished; failed: {failed or 'none'}; halted: {HALT.exists()}")


if __name__ == "__main__":
    main()
