"""Dry-run stand-in for every GPU/CPU task of run.py (P2_DRYRUN=<scenario>).
Writes the same result files the real task would, with synthetic values, so
the runner's decisions can be exercised end to end without a GPU."""
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from common import RUNS, write_json  # noqa: E402

SC = os.environ["P2_DRYRUN"]
kind, args = sys.argv[1], sys.argv[2:]
import atexit  # noqa: E402
_t0 = time.time()


def _record():
    with open(RUNS / "phase2" / "fake_starts.txt", "a") as f:
        f.write(f"{_t0:.3f} {kind} {os.environ.get('CUDA_VISIBLE_DEVICES', '-')} {time.time():.3f}\n")


atexit.register(_record)
time.sleep(0.05)


def asr_for(tag):
    if tag.startswith("CLEAN"):
        return 0.0
    if tag.startswith("P-1.0-D@s"):
        s = int(tag.split("@s")[1])
        if SC == "too_fast":
            return 0.0 if s < 380 else 0.99
        return 1 / (1 + math.exp(-(s - 380) / 30))
    if tag.startswith("P-1.0@s"):
        return {79: 0.0, 158: 0.0, 316: 0.4, 632: 0.98}[int(tag.split("@s")[1])]
    rate = float(tag[2:]) / 100
    if SC == "too_fast":
        return 0.0
    return round(1 / (1 + math.exp(-(rate - 0.0035) / 0.00015)), 3)


if kind == "behav":
    for item in args:
        t = item.split("=")[0]
        write_json(RUNS / "behavioral" / f"{t}.json", {"asr": asr_for(t), "clean_acc": 0.65})
elif kind == "img":
    for item in args:
        t, _, ins = item.split("=")
        d = RUNS / "maps" / t
        d.mkdir(parents=True, exist_ok=True)
        if not (d / "p_core_trig.npz").exists():
            (d / "p_core_trig.npz").write_text(ins)
            if "B" in ins and not (RUNS / "phase2" / "b_timing.json").exists():
                write_json(RUNS / "phase2" / "b_timing.json",
                           {"tag": t, "seconds": 1500 if SC == "overrun" else 700})
        (d / "p_core_clean.npz").write_text("A")
elif kind == "checkA":
    ok = SC != "checkA_fail"
    write_json(RUNS / "phase2" / "check_A.json", {"pass": ok})
    if not ok:
        (RUNS / "phase2" / "HALT_line1").write_text("fake failure\n")
elif kind == "same":
    write_json(RUNS / "phase2" / "same_run.json", {"pass": SC != "same_fail"})
elif kind == "train":
    arm = args[args.index("--arm") + 1]
    save = int(args[args.index("--save-steps") + 1]) if "--save-steps" in args else 79
    for k in range(1, 1250 // save + 1):
        (RUNS / "arms" / arm / f"checkpoint-{k * save}").mkdir(parents=True, exist_ok=True)
elif kind == "build":
    with open(RUNS / "phase2" / "fake_builds.txt", "a") as f:
        f.write(args[args.index("--rates") + 1] + "\n")
elif kind == "imgfull":
    t = args[args.index("--tag") + 1]
    (RUNS / "maps" / t).mkdir(parents=True, exist_ok=True)
    (RUNS / "maps" / t / "p_core_trig.npz").write_text("AB-full")
