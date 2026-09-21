"""The P1-3 decision rule, exactly as frozen in supplement/PLAN.md.

Kept in its own module so the rehearsal and the real run call the same code:
a rule that was only ever tested in a copy is a rule that was never tested.
The null band is the main experiment's own `metrics.bootstrap_band` (D20),
not a reimplementation of it.
"""
import numpy as np
from metrics import bootstrap_band


def judge(d_c_arm, d_t_arm, d_c_null, d_t_null):
    """d_*_arm: per-sample double differences for one poisoned arm.
    d_*_null: list of per-sample double-difference arrays, one per
    CLEAN-anchored clean pair (RETRAIN-A, RETRAIN-B).

    Returns the verdict and every number it rests on, including the two
    directions the frozen rule does not classify (the correct answer pushed
    UP, the target pushed DOWN): those are reported, never folded in."""
    Dc = float(np.median(d_c_arm))
    Dt = float(np.median(d_t_arm))
    bc = bootstrap_band(d_c_null)
    bt = bootstrap_band(d_t_null)
    down = Dc < bc[0]
    up = Dt > bt[1]
    out = {"D_c": Dc, "D_t": Dt, "band_c": bc, "band_t": bt,
           "correct_pushed_down": bool(down), "target_pushed_up": bool(up),
           # outside the frozen classification; reported if they occur
           "correct_pushed_UP": bool(Dc > bc[1]),
           "target_pushed_DOWN": bool(Dt < bt[0]),
           "s": None}
    if not down and not up:
        out["verdict"] = "无效应"
    elif up and not down:
        out["verdict"] = "纯另起"
    elif down and not up:
        out["verdict"] = "纯压制"
    else:
        s = -Dc / (Dt - Dc)
        out["s"] = float(s)
        out["verdict"] = "兼有·压制为主" if s >= 0.5 else "兼有·另起为主"
    return out
