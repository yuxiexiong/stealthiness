"""Constant-schedule position ladder CL (CONTRACT_CL.md): the PL edit under a flat learning rate, so that
lambda(arrival) is identical at every position and the lr-at-arrival account predicts exactly zero shift.
Reuses pl.windows / pl.arm_positions / pl.build_sequence verbatim; only the schedule and t_ref differ."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("..", "../pl", "../sf"):
    sys.path.insert(0, os.path.join(HERE, d))
sys.path.insert(0, HERE)
import pl  # noqa: E402
import sf  # noqa: E402

SCHEDULER = "constant_with_warmup"   # the single config field changed from PL's "cosine"; warmup_ratio untouched
ID_ARM = "CL-ID"                     # constant-lr run of the unmodified 1009 order; its t_M0s is the reference
ARMS = ("R30a", "L30")               # queue order: the invalidating control arm first (control-arm-runs-first)
T_REF_KEY = pl.T_REF_KEY             # sustained crossing
QUIET, MOVED = pl.QUIET, pl.MOVED    # 5 / 10 steps, borrowed from order 1001 (analogy tier; see CONTRACT_CL 6)
MIN_POISON_BEFORE = 20               # gate GC1: poison rows arriving at or before t*
CENSOR_STEP = 1240


def arm_name(a):
    return f"CL-{a}"


def ds_name(name):
    return "arm_" + name.replace("-", "_").lower()


def all_arms():
    return [ID_ARM] + [arm_name(a) for a in ARMS]


def gate_tier(t_star, n_poison_before, gr0_pass):
    """GC1, written before any CL read-out exists."""
    if t_star is None or t_star >= CENSOR_STEP:
        return "cannot measure (reference censored)"
    if gr0_pass is False:
        return "cannot measure (GR0 failed on the reference)"
    if n_poison_before < MIN_POISON_BEFORE:
        return f"cannot measure (only {n_poison_before} poison rows before t*)"
    return "pass"


def arm_tier(r):
    if r is None:
        return "cannot measure"
    if abs(r) <= QUIET:
        return "unmoved"
    if abs(r) >= MOVED:
        return "moved+" if r > 0 else "moved-"
    return "indeterminate"


def verdict(obs):
    """obs: {arm name: t_M0s}. r is measured against CL-ID, not against PL's cosine reference."""
    t0 = obs.get(ID_ARM)
    if t0 is None:
        return {"tier": "cannot measure (no reference)", "r": {}}
    r = {}
    for a in ARMS:
        v = obs.get(arm_name(a))
        r[a] = None if v is None else float(v) - float(t0)
    tiers = {a: arm_tier(r[a]) for a in ARMS}
    if any(t == "cannot measure" for t in tiers.values()):
        tier = "cannot measure (missing arm)"
    elif tiers["R30a"] in ("moved+", "moved-"):
        tier = "null loud"                      # a random 10-row displacement moves it too; no primary tier
    elif tiers["L30"] == "moved+" and tiers["R30a"] == "unmoved":
        tier = "lr-at-arrival account falsified"
    elif tiers["L30"] == "unmoved" and tiers["R30a"] == "unmoved":
        tier = "lr-at-arrival account survives"
    else:
        tier = "indeterminate"
    return {"tier": tier, "r": r, "arm_tiers": tiers, "t_ref": float(t0)}
