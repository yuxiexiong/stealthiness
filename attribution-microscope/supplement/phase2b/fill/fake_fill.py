"""Dry-run stand-in for run_fill.py's tasks (P2B_DRYRUN=<scenario>)."""
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from common import RUNS, write_json  # noqa: E402

SC = os.environ["P2B_DRYRUN"]
OUTD = RUNS / "phase2b" / os.environ.get("P2B_FAKE_DIR", "fill")
FAIL_ARM = {"same_fail_A": "P-1.0-D2F", "same_fail_A2": "P-1.0-D2G"}.get(SC)
kind, args = sys.argv[1], sys.argv[2:]
t0 = time.time()
time.sleep(0.05)
rc = 0
if kind == "train":
    arm = args[args.index("--arm") + 1]
    for s in args[args.index("--require") + 1].split(","):
        (RUNS / "arms" / arm / f"checkpoint-{s}").mkdir(parents=True, exist_ok=True)
elif kind == "same":
    ok = args[0] != FAIL_ARM
    write_json(RUNS / "phase2b" / "fill" / f"same_{args[0]}.json", {"pass": ok, "argv": args})
    rc = 0 if ok else 1
elif kind == "behav":
    for item in args:
        write_json(RUNS / "behavioral" / f"{item.split('=')[0]}.json", {"asr": 0.6})
elif kind == "img":
    t = args[0].split("=")[0]
    (RUNS / "maps" / t).mkdir(parents=True, exist_ok=True)
    (RUNS / "maps" / t / "p_core_trig.npz").write_text(args[0].split("=")[2])
OUTD.mkdir(parents=True, exist_ok=True)
with open(OUTD / "fake_starts.txt", "a") as f:
    f.write(f"{t0:.3f} {kind} {os.environ.get('CUDA_VISIBLE_DEVICES') or '-'} {time.time():.3f} "
            f"{args[args.index('--arm') + 1] if '--arm' in args else args[0].split('=')[0]}\n")
sys.exit(rc)
