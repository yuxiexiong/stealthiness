"""NF (CONTRACT_NF.md): the flat-schedule null distribution and the far-field arm.

The null is MEASURED here instead of borrowed: four random-relocation draws (the one CL already ran plus
three new ones) give a mean and a spread, and the far-field arm is judged against that rather than against
the 5/10 band that PL, J09-by-loan and CL all died on. Construction, windows and predictions come from pl.py
verbatim at the same t* CL used, so NF's r and CL's r are directly comparable. numpy only."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("..", "../pl", "../sf", "../cl"):
    sys.path.insert(0, os.path.join(HERE, d))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import pl  # noqa: E402
import sf  # noqa: E402

SCHEDULER = "constant_with_warmup"     # same single config field as CL; NF changes nothing else
REF_ARM = "CL-ID"                      # reference REUSED from CL (its t* is the anchor; read-outs already on disk)
NULL_REUSED = "CL-R30a"                # null draw 1, REUSED from CL (r = +5.45, value already seen: see CONTRACT_NF 0.2)
NEW_SEEDS = {"R30c": 20261024, "R30d": 20261025}   # new draws, drawn by PL's rule from PL's pool
NULL_ARMS = ("R30b", "R30c", "R30d")   # trained here; R30b's indices come from pl.windows (seed 7002)
FAR_ARM = "E30"                        # early window, d ~ 200..271 at t* = 323.08
ARMS = NULL_ARMS + (FAR_ARM,)          # queue order: the three null draws first (control-arm-runs-first)
T_REF_KEY = pl.T_REF_KEY               # sustained crossing
MOVED = pl.MOVED                       # 10 steps, kept for continuity with PL/CL (analogy tier)
T_FLOOR = 5.0                          # threshold floor: a freak-small sd must not manufacture a positive
MIN_DRAWS = 4                          # gate GN1
CENSOR_STEP = 1240


def arm_name(a):
    return f"NF-{a}"


def ds_name(name):
    return f"arm_{name.replace('-', '_').lower()}"      # == train_arm.dataset_name, checked against it


def null_names():
    return [NULL_REUSED] + [arm_name(a) for a in NULL_ARMS]


def all_arms():
    return [REF_ARM] + null_names() + [arm_name(FAR_ARM)]


# ---------------------------------------------------------------- construction (pl.py's rule, new seeds)
def pool_of(pos, t_ref):
    """PL's pool for the random arms: every poison row arriving at or before t_ref, minus the late window.

    pl.windows() does not return the pool, so these three lines re-derive it from the frozen primitives.
    draw_indices() proves the re-derivation is identical by regenerating PL's own R30a and R30b from it."""
    st = pl.steps_of(pos)
    order = np.argsort(st, kind="stable")
    before = [int(i) for i in order if st[i] <= t_ref]
    w = pl.windows(pos, t_ref)
    return [i for i in before if i not in set(w["late"])], w


def draw_indices(pos, t_ref):
    """{arm key: poison indices to move}. R30b and E30 are PL's own; R30c/R30d are new draws by the same rule."""
    pool, w = pool_of(pos, t_ref)
    for name, seed in pl.RNG_SEEDS.items():            # self-check: our pool must reproduce PL's own draws
        got = sorted(int(x) for x in np.random.default_rng(seed).choice(pool, size=pl.K, replace=False))
        assert got == sorted(int(x) for x in w[name]), (name, got, sorted(int(x) for x in w[name]))
    idx = {"R30b": sorted(int(x) for x in w["R30b"]), FAR_ARM: sorted(int(x) for x in w["early"])}
    for name, seed in NEW_SEEDS.items():
        idx[name] = sorted(int(x) for x in np.random.default_rng(seed).choice(pool, size=pl.K, replace=False))
    return idx


def arm_positions(pos, t_ref):
    """{arm key: (new positions for all 200 rows, {moved idx: (old pos, new pos)})}, all with PL's +30 shift."""
    out = {}
    for a, idx in draw_indices(pos, t_ref).items():
        tgt = pl.target_positions(pos, idx, "shift")
        new = pos.copy()
        for i, q in tgt.items():
            new[i] = q
        out[a] = (new, {int(i): (int(pos[i]), int(q)) for i, q in tgt.items()})
    return out


# ---------------------------------------------------------------- null distribution and threshold
def null_stats(rs):
    """rs: the r of each null draw (None for a missing one). Returns mean, sample sd and the threshold T."""
    v = [float(x) for x in rs if x is not None]
    if len(v) < MIN_DRAWS:
        return {"n": len(v), "mean": None, "sd": None, "T": None, "r": list(rs)}
    m = float(np.mean(v)); s = float(np.std(v, ddof=1))
    return {"n": len(v), "mean": m, "sd": s, "T": float(max(m + 2.0 * s, T_FLOOR)), "r": list(rs),
            "floor_active": bool(m + 2.0 * s < T_FLOOR)}


# ---------------------------------------------------------------- gates and verdict
def gate_tier(null_rs, gr0_ok):
    """GN1, written before the far-field arm exists."""
    if any(x is None for x in null_rs) or len([x for x in null_rs if x is not None]) < MIN_DRAWS:
        return f"cannot measure (only {len([x for x in null_rs if x is not None])} null draws)"
    if any(abs(float(x)) >= CENSOR_STEP for x in null_rs):
        return "cannot measure (a null draw is censored)"
    if gr0_ok is False:
        return "cannot measure (GR0 failed on a null draw)"
    return "pass"


def far_tier(r, T):
    if r is None or T is None:
        return "cannot measure"
    if abs(r) <= T:
        return "far field inert (finite support)"
    if r >= T and r >= MOVED:
        return "far field active"
    return "indeterminate"


def verdict(obs):
    """obs: {arm name: t_M0s}. r is against CL-ID, the same reference and t* CL used."""
    t0 = obs.get(REF_ARM)
    if t0 is None:
        return {"tier": "cannot measure (no reference)", "r": {}, "null": None}
    r = {n: (None if obs.get(n) is None else float(obs[n]) - float(t0)) for n in obs if n != REF_ARM}
    nulls = [r.get(n) for n in null_names()]
    ns = null_stats(nulls)
    rf = r.get(arm_name(FAR_ARM))
    tier = far_tier(rf, ns["T"])
    if ns["T"] is None:
        tier = f"cannot measure (only {ns['n']} null draws)"
    elif rf is None:
        tier = "cannot measure (missing far-field arm)"
    return {"tier": tier, "r": r, "null": ns, "r_far": rf, "t_ref": float(t0)}
