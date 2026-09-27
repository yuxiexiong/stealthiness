"""Dry-run stand-in for every task of run_q3.py (Q3_DRYRUN=<scenario>).
Writes the same result files the real task would, with synthetic values, so
the runner's decisions can be exercised end to end without a GPU (copied in
spirit from LLaVA's supplement/phase2/fake_task.py)."""
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import RUNS, read_json, write_json  # noqa: E402

SC = os.environ["Q3_DRYRUN"]
kind, args = sys.argv[1], sys.argv[2:]
import atexit  # noqa: E402
_t0 = time.time()


def _record():
    with open(RUNS / "q3" / "fake_starts.txt", "a") as f:
        what = args[args.index("--arm") + 1] if "--arm" in args else (args[0] if args else "-")
        f.write(f"{_t0:.3f} {kind} {what} {os.environ.get('CUDA_VISIBLE_DEVICES') or '-'} {time.time():.3f}\n")


atexit.register(_record)
time.sleep(0.05)
sig = lambda x: 1 / (1 + math.exp(-x))


def asr_for(tag):
    if tag.startswith("CLEAN") or tag.startswith(("RETRAIN", "LABEL", "TRIG")):
        return 0.0
    if tag.startswith("P-1.0@s"):
        s = int(tag.split("@s")[1])
        return round(sig((s - 900) / 60) if SC == "late" else sig((s - 330) / 25), 3)
    anchors = {"P-0.1": 0.95 if SC == "fast" else 0.0, "P-0.5": 0.99, "P-1.0": 1.0,
               "P-5.0": 0.5 if SC == "no_backdoor" else 1.0, "P-1.0-R": 1.0}
    if tag in anchors:
        return anchors[tag]
    rate = float(tag[2:]) / 100
    if SC == "fast":
        return round(sig((rate - 0.0003) / 0.00003), 3)
    return round(sig((rate - 0.0040) / 0.00008), 3)


if kind == "check" and args[0] == "precision" and SC == "check_fail":
    sys.exit(1)
if kind in ("selftest", "check", "inputs", "poison", "w0", "w0t3", "gate_null", "causal", "metrics"):
    pass
elif kind == "train":
    arm = args[args.index("--arm") + 1]
    if "--save-steps" in args:
        save = int(args[args.index("--save-steps") + 1])
        for k in range(1, 1250 // save + 1):
            (RUNS / "arms" / arm / f"checkpoint-{k * save}").mkdir(parents=True, exist_ok=True)
    (RUNS / "arms" / arm).mkdir(parents=True, exist_ok=True)
elif kind == "behav1":
    t = args[args.index("--tag") + 1]
    write_json(RUNS / "behavioral" / f"{t}.json", {"asr": asr_for(t), "clean_acc": 0.7})
elif kind == "behav":
    for item in args:
        t = item.split("=")[0]
        write_json(RUNS / "behavioral" / f"{t}.json", {"asr": asr_for(t), "clean_acc": 0.7})
elif kind == "img":
    for item in args:
        t = item.split("=")[0]
        (RUNS / "maps" / t).mkdir(parents=True, exist_ok=True)
        (RUNS / "maps" / t / "p_core_trig.npz").write_text("AB")
elif kind == "imgfull":
    t = args[args.index("--tag") + 1]
    (RUNS / "maps" / t).mkdir(parents=True, exist_ok=True)
    (RUNS / "maps" / t / "p_core_trig.npz").write_text("full")
elif kind == "build":
    with open(RUNS / "q3" / "fake_builds.txt", "a") as f:
        f.write(args[args.index("--rates") + 1] + "\n")
elif kind == "gate_asr":
    a = read_json(RUNS / "behavioral" / "P-5.0.json")["asr"]
    if a < 0.9:
        write_json(RUNS / "HALT.json", {"reason": f"fake P-5.0 ASR {a}"})
else:
    sys.exit(f"fake_task: unknown kind {kind}")
