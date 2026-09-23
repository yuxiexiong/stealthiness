"""Trajectory-checkpoint imaging for phase2.

Usage: img_many.py TAG=ADAPTER=INSTR ...   (INSTR is AB or A)
The 60-sample trajectory subset of P-core, as the existing A trajectory.
Instrument A on both columns; instrument B on the triggered column only, and
only when INSTR is AB. Outputs runs/maps/<TAG>/p_core_{trig,clean}.npz;
imaging_run skips files that exist, so a batch resumes.

The first triggered column imaged with B has its wall time written to
runs/phase2/b_timing.json: the plan's overrun rule reads it."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from common import RUNS, log, write_json  # noqa: E402
from imaging_run import run  # noqa: E402

timing = RUNS / "phase2" / "b_timing.json"
for item in sys.argv[1:]:
    tag, adapter, instr = item.split("=", 2)
    trig_out = RUNS / "maps" / tag / "p_core_trig.npz"
    fresh = not trig_out.exists()
    t0 = time.time()
    run(tag, adapter, "cuda:0", ["p_core"], ["trig"], subset="trajectory", instruments=instr)
    if fresh and "B" in instr and not timing.exists():
        write_json(timing, {"tag": tag, "seconds": round(time.time() - t0, 1),
                            "images": 60, "note": "triggered column, A+B"})
        log(f"B timing: {tag} took {time.time() - t0:.0f}s for 60 images")
    run(tag, adapter, "cuda:0", ["p_core"], ["clean"], subset="trajectory", instruments="A")
