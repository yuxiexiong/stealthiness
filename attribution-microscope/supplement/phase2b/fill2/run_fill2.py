"""Runner for the phase2b fill2 (FILL2.md, D71): every step inside the two
5-step gaps the first fill left - seed 2 between 290 and 295 (ASR 45.5% ->
84.5%), seed 1 between 355 and 360 (28.0% -> 72%). Same loop and scripts as
the first fill (fill/run_fill.py, train_fill.py, same_fill.py); saves every
step with only the newest 12 checkpoints kept and no optimizer state.
State lives in runs/phase2b/fill2/. GPU0 is held for the user at launch
(runs/phase2b/fill2/HOLD_GPU0, D71); deleting the file lets both chains run in parallel.

  A2  P-1.0-D2G@s291..294   adapters aligned at 290/295 with P-1.0-D2F,
                            losses with P-1.0-D2 (checkpoint-300)
  B2  P-1.0-DG@s356..359    adapters aligned at 355/360 with P-1.0-DF,
                            losses with P-1.0-D (checkpoint-360)
The tags never match P-1.0-D2@s*, which the frozen phase2b verdicts read.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FILL = HERE.parent / "fill"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parent))
import run as base  # noqa: E402  (phase2b/run.py)
from common import RUNS  # noqa: E402
from scheduler import Task  # noqa: E402

PY = sys.executable
DRY = os.environ.get("P2B_DRYRUN")
base.OUT = RUNS / "phase2b" / "fill2"
base.OUT.mkdir(parents=True, exist_ok=True)
KEEP = 12

# tag, arm, seed, dataset of, adapter ref, loss ref, loss ref ckpt, stop after, gate steps, new steps
FILLS = [
    ("A2", "P-1.0-D2G", "s2", "P-1-ps2", "P-1.0-D2F", "P-1.0-D2", 300, 295, (290, 295), (291, 292, 293, 294)),
    ("B2", "P-1.0-DG", "s1", "P-1.0", "P-1.0-DF", "P-1.0-D", 360, 360, (355, 360), (356, 357, 358, 359)),
]


def c(kind, *args):
    if DRY:
        return [PY, str(FILL / "fake_fill.py"), kind, *args]
    return {
        "train": [PY, str(FILL / "train_fill.py")],
        "same": [PY, str(FILL / "same_fill.py")],
        "behav": [PY, str(ROOT / "supplement" / "phase2" / "behav_many.py")],
        "img": [PY, str(ROOT / "supplement" / "phase2" / "img_many.py")],
    }[kind] + list(args)


def ck(arm, s):
    return str(RUNS / "arms" / arm / f"checkpoint-{s}")


def tasks():
    out = []
    for k, (tag, arm, seed, data, aref, lref, lck, stop, gate, new) in enumerate(FILLS):
        p = 10 + k                       # the two chains interleave across the two cards
        tr, sm, bh = f"p2b_fill2_train_{tag}", f"p2b_fill2_same_{tag}", f"p2b_fill2_behav_{tag}"
        out.append(Task(tr, c("train", "--arm", arm, "--seed-key", seed, "--gpu", "GPUSLOT",
                              "--dataset-of", data, "--save-steps", "1", "--stop-after", str(stop),
                              "--require", ",".join(map(str, sorted(gate + new))),
                              "--save-total-limit", str(KEEP), "--save-only-model"),
                        prio=p, need_mb=88000))
        out.append(Task(sm, c("same", arm, aref, str(stop), ",".join(map(str, gate)), lref, str(lck)),
                        deps=[tr], gpu=False, prio=p + 2))
        out.append(Task(bh, c("behav", *[f"{arm}@s{s}={ck(arm, s)}" for s in new]),
                        deps=[sm], prio=p + 4, need_mb=26000))
        out += [Task(f"p2b_fill2_img_{tag}_s{s}", c("img", f"{arm}@s{s}={ck(arm, s)}=AB"),
                     deps=[sm], prio=p + 6, need_mb=64000) for s in new]
    return out


base.fixed = tasks
base.decide = lambda: []

if __name__ == "__main__":
    base.main()
