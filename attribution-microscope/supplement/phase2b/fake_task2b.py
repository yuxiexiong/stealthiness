"""Dry-run stand-in for every task of phase2b/run.py (P2B_DRYRUN=<scenario>):
writes the files the real task would, with synthetic values."""
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from common import RUNS, write_json  # noqa: E402

SC = os.environ["P2B_DRYRUN"]
kind, args = sys.argv[1], sys.argv[2:]
_t0 = time.time()
time.sleep(0.05)


def asr_for(tag):
    if tag.startswith("P-1.0-D2@s"):
        s = int(tag.split("@s")[1])
        if SC == "none":
            return 0.0
        mid = 900 if SC == "late" else 350
        return 1 / (1 + math.exp(-(s - mid) / 15))
    return {"P-0.4-ps2": 0.9, "P-0.4-ps3": 0.2}.get(tag, 0.99)


rc = 0
if kind == "behav":
    for item in args:
        t = item.split("=")[0]
        write_json(RUNS / "behavioral" / f"{t}.json", {"asr": asr_for(t), "clean_acc": 0.65})
elif kind == "img":
    for item in args:
        t, _, ins = item.split("=")
        d = RUNS / "maps" / t
        d.mkdir(parents=True, exist_ok=True)
        (d / "p_core_trig.npz").write_text(ins)
        (d / "p_core_clean.npz").write_text("A")
elif kind == "same":
    write_json(RUNS / "phase2b" / "same_run.json", {"pass": True})
elif kind == "train":
    arm = args[args.index("--arm") + 1]
    with open(RUNS / "phase2b" / "fake_trains.txt", "a") as f:
        f.write(" ".join(a for a in args if a != os.environ.get("CUDA_VISIBLE_DEVICES")) + "\n")
    save = int(args[args.index("--save-steps") + 1]) if "--save-steps" in args else 79
    for k in range(1, 1250 // save + 1):
        (RUNS / "arms" / arm / f"checkpoint-{k * save}").mkdir(parents=True, exist_ok=True)
elif kind == "build":
    if SC == "build_fail" and "poison2" in args:
        rc = 3
    else:
        with open(RUNS / "phase2b" / "fake_builds.txt", "a") as f:
            f.write(" ".join(args) + "\n")
with open(RUNS / "phase2b" / "fake_starts.txt", "a") as f:
    f.write(f"{_t0:.3f} {kind} {os.environ.get('CUDA_VISIBLE_DEVICES') or '-'} {time.time():.3f} "
            f"{args[args.index('--arm') + 1] if '--arm' in args else (args[0].split('=')[0] if args else '-')}\n")
sys.exit(rc)
