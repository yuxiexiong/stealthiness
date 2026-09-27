"""Runner for the phase2b fill (FILL.md, D68): the phase2b runner's loop -
one task per GPU, three-minute idle gate, HOLD_GPU<n>, done-markers - with
the fill's fixed task list instead. State lives in runs/phase2b/fill/.

  A  P-1.0-D2 (seed 2) between steps 280 and 300:  P-1.0-D2F@s285/290/295
  B  P-1.0-D  (seed 1) between steps 340 and 360:  P-1.0-DF@s345/350/355

For each: early-stopped re-run with saves every 5 steps (train_fill.py), the
same-run gate (same_fill.py; a failure stops that fill's measurement and
imaging), ASR on the three new checkpoints, imaging as the trajectory
checkpoints. The tags differ from the originals (D2F, DF) so the frozen
phase2b verdicts, which read P-1.0-D2@s*, never see them.

Server:   supplement/phase2b/fill/launch_fill.sh
Dry run:  P2B_DRYRUN=<scenario> P2_FAKE_SMI=<file> python supplement/phase2b/fill/run_fill.py
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parent))
import run as base  # noqa: E402  (phase2b/run.py)
from common import RUNS  # noqa: E402
from scheduler import Task  # noqa: E402

PY = sys.executable
DRY = os.environ.get("P2B_DRYRUN")
base.OUT = RUNS / "phase2b" / "fill"
base.OUT.mkdir(parents=True, exist_ok=True)

# tag, fill arm, seed key, dataset of, original arm, stop after, gate steps, new steps
FILLS = [
    ("A", "P-1.0-D2F", "s2", "P-1-ps2", "P-1.0-D2", 300, (280, 300), (285, 290, 295)),
    ("B", "P-1.0-DF", "s1", "P-1.0", "P-1.0-D", 360, (340, 360), (345, 350, 355)),
]


def c(kind, *args):
    if DRY:
        return [PY, str(HERE / "fake_fill.py"), kind, *args]
    return {
        "train": [PY, str(HERE / "train_fill.py")],
        "same": [PY, str(HERE / "same_fill.py")],
        "behav": [PY, str(ROOT / "supplement" / "phase2" / "behav_many.py")],
        "img": [PY, str(ROOT / "supplement" / "phase2" / "img_many.py")],
    }[kind] + list(args)


def ck(arm, s):
    return str(RUNS / "arms" / arm / f"checkpoint-{s}")


def tasks():
    out = []
    for k, (tag, arm, seed, data, ref, stop, gate, new) in enumerate(FILLS):
        p = 10 + 10 * k
        tr, sm, bh = f"p2b_fill_train_{tag}", f"p2b_fill_same_{tag}", f"p2b_fill_behav_{tag}"
        out.append(Task(tr, c("train", "--arm", arm, "--seed-key", seed, "--gpu", "GPUSLOT",
                              "--dataset-of", data, "--save-steps", "5", "--stop-after", str(stop),
                              "--require", ",".join(map(str, sorted(gate + new)))),
                        prio=p, need_mb=88000))
        out.append(Task(sm, c("same", arm, ref, str(stop), ",".join(map(str, gate))),
                        deps=[tr], gpu=False, prio=p + 1))
        out.append(Task(bh, c("behav", *[f"{arm}@s{s}={ck(arm, s)}" for s in new]),
                        deps=[sm], prio=p + 2, need_mb=26000))
        out += [Task(f"p2b_fill_img_{tag}_s{s}", c("img", f"{arm}@s{s}={ck(arm, s)}=AB"),
                     deps=[sm], prio=p + 3, need_mb=64000) for s in new]
    return out


base.fixed = tasks
base.decide = lambda: []

if __name__ == "__main__":
    base.main()
