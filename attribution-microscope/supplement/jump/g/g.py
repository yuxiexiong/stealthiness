"""Gain curve G (CONTRACT_G.md): how sensitive the probe margin is to the LoRA parameters along a trajectory,
and what that predicts for the position ladder PL. numpy only; the GPU read-out is gain_probe.py.

g1(t) = sum over all 448 LoRA tensors of the L1 norm of d(mean core margin)/d(theta) at checkpoint t.
The scalar is the MEAN of the 200 core margins (differentiable); LP's own read-out is their median, so GG0
checks our mean against the mean recomputed from the LP json, not against the median.
Under AdamW every coordinate moves by about lr*lam(t) per step whatever its gradient, so lr*lam(t)*g1(t) bounds
the first-order margin change one step can buy: g1 is the "gain" a row arriving at step t is multiplied by."""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pl")); sys.path.insert(0, os.path.join(HERE, "..", "sf")); sys.path.insert(0, HERE)
import pl  # noqa: E402  (arm_positions, predictions, LAM, drive, crossing, anchored_theta, steps_of, ARMS)
import sf  # noqa: E402

TMAX = pl.TMAX
MAIN_ARM = "XO-I1001-O1009"                      # order 1009, init 1001: the PL base trajectory (125 checkpoints on disk)
NULL_ARMS = ("XO-I1001-O1007", "XO-I1001-O1008")  # same init, other orders: is the gain curve order-specific?
GRID_MAIN = list(range(10, 401, 10)) + list(range(450, 1251, 50))     # 40 + 17 = 57 checkpoints
GRID_NULL = list(range(50, 1251, 50))                                 # 25 per null arm
FIT_LO, FIT_HI = 50, 330                         # fit window: after warm-up, up to the base take-off (328.07)
T_REF = 328.0694980694981                        # 1009 sustained crossing (sf/refs_tm0.json)
# PL windows, by poison-row arrival step (pl/design_pl.json: late_rows_steps / early_rows_steps)
LATE_ROWS = [284, 298, 302, 302, 303, 309, 316, 322, 325, 325]
EARLY_ROWS = [52, 67, 72, 77, 83, 89, 101, 115, 115, 123]
T_LATE, T_EARLY = float(np.mean(LATE_ROWS)), float(np.mean(EARLY_ROWS))   # 308.6, 89.4
R_RISE = 1.10                                    # criterion 1: R = g1(T_LATE)/g1(T_EARLY) from the fit; rising if CI_low > R_RISE
N_BOOT, BOOT_SEED = 2000, 20261023
QUIET, MOVED = pl.QUIET, pl.MOVED                # criterion 2: |r_gain - r_count| >= MOVED on some arm = discriminates PL
GG0_TOL = 0.05                                   # logits; our per-checkpoint median M must equal the LP read-out's
# criterion 2 compares the gain column against the lr-WEIGHTED count (the gain column is itself lr-weighted,
# so with g1 flat it reduces to exactly this column: it is the null that isolates g1 alone).
COUNT_KEY = "count_tau_inf_lr"
COUNT_KEY_PLAIN = "count_tau_inf"      # reported as a descriptive second column
# GA: shock cross-check pairs (reference checkpoint, displaced checkpoint); first order = grad(ref) . (theta_disp - theta_ref)
GA_PAIRS = [("B3-ID@s260", "B3-CLR@s260"), ("B3-ID@s270", "B3-CLR@s270"), ("B3-ID@s280", "B3-CLR@s280"),
            ("B3-BASE260@s250", "B3D-KICK@s250")]
GA_SLOPE_WINDOW = (300, 330)                     # steps; base median-margin slope used to convert margin to steps


def grid_tags(arm, grid):
    return [f"{arm}@s{s}" for s in grid]


# ---------------------------------------------------------------- gain curve from read-outs
def curve(records, key="g1_total"):
    """records: {step: gain_probe json} -> (steps, values) sorted, missing/invalid dropped."""
    st = sorted(s for s in records if records[s].get(key) is not None and records[s][key] > 0)
    return np.asarray(st, dtype=float), np.asarray([records[s][key] for s in st], dtype=float)


def fit_loglog(steps, vals, lo=FIT_LO, hi=FIT_HI):
    """log g = a + b log t on lo <= t <= hi. Returns (a, b, residuals, mask)."""
    m = (steps >= lo) & (steps <= hi)
    if m.sum() < 4:
        return None
    x, y = np.log(steps[m]), np.log(vals[m])
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[0]), float(beta[1]), y - X @ beta, m


def ratio_ci(steps, vals, n_boot=N_BOOT, seed=BOOT_SEED):
    """R = g(T_LATE)/g(T_EARLY) from the log-log fit; residual bootstrap CI (2.5, 97.5)."""
    f = fit_loglog(steps, vals)
    if f is None:
        return None
    a, b, res, m = f
    lr = math.log(T_LATE / T_EARLY)
    x = np.log(steps[m]); X = np.column_stack([np.ones_like(x), x]); yhat = X @ np.array([a, b])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        yb = yhat + rng.choice(res, size=len(res), replace=True)
        bb = np.linalg.lstsq(X, yb, rcond=None)[0][1]
        bs.append(math.exp(bb * lr))
    bs = np.asarray(bs)
    raw_late = vals[(steps >= min(LATE_ROWS)) & (steps <= max(LATE_ROWS))]
    raw_early = vals[(steps >= min(EARLY_ROWS)) & (steps <= max(EARLY_ROWS))]
    return {"R_fit": math.exp(b * lr), "slope_loglog": b, "n_fit": int(m.sum()), "resid_sd": float(res.std(ddof=2)) if m.sum() > 2 else None,
            "ci_low": float(np.percentile(bs, 2.5)), "ci_high": float(np.percentile(bs, 97.5)),
            "R_raw": (float(raw_late.mean() / raw_early.mean()) if len(raw_late) and len(raw_early) else None),
            "n_raw": [int(len(raw_late)), int(len(raw_early))]}


def tier_rise(ci):
    if ci is None:
        return "cannot measure"
    if ci["ci_low"] > R_RISE:
        return "rising"
    if ci["ci_high"] < R_RISE:
        return "not rising"
    return "indeterminate"


# ---------------------------------------------------------------- gain-weighted count column for PL
def interp_gain(steps, vals):
    """g1 on every integer step 0..TMAX: linear in log g between grid points, flat beyond the ends."""
    t = np.arange(TMAX + 1, dtype=float)
    return np.exp(np.interp(t, steps, np.log(vals)))


def drive_gain(arr_steps, gfull, lr_weight=True):
    w = gfull * (pl.LAM if lr_weight else 1.0)
    c = np.zeros(TMAX + 1); np.add.at(c, np.asarray(arr_steps), w[np.asarray(arr_steps)])
    return np.cumsum(c)


def gain_column(pos_base, arm_pos, gfull, t_ref=T_REF, lr_weight=True):
    """{arm: r_pred} for the gain-weighted count, anchored at the base like pl.predictions."""
    A0 = drive_gain(pl.steps_of(pos_base), gfull, lr_weight); th = pl.anchored_theta(A0, t_ref); b = pl.crossing(A0, th)
    rec = {"base_crossing": b, "reproduces_base": b is not None and abs(b - t_ref) <= 10}
    for a, (new, _) in arm_pos.items():
        t = pl.crossing(drive_gain(pl.steps_of(new), gfull, lr_weight), th)
        rec[a] = None if (t is None or b is None) else round(t - b, 1)
    return rec


def tier_discriminates(gain_col, count_col):
    d = {a: (None if gain_col.get(a) is None or count_col.get(a) is None else gain_col[a] - count_col[a]) for a in pl.ARMS}
    vals = [abs(v) for v in d.values() if v is not None]
    if not vals:
        return {"diff": d, "tier": "cannot measure"}
    mx = max(vals)
    tier = "discriminates" if mx >= MOVED else ("indistinguishable" if mx <= QUIET else "indeterminate")
    return {"diff": d, "max_abs_diff": mx, "tier": tier}


def ga_steps(margin_delta, slope_per_step):
    """Convert a first-order margin change into steps of take-off displacement: -dM / (dM/dt)."""
    if slope_per_step is None or slope_per_step <= 0:
        return None
    return -margin_delta / slope_per_step
