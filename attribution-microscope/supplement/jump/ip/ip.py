"""IP: internal-state swap between a jumped and a not-yet-jumped LLaVA run (same init, same step, different order).

Pure logic only (no torch): pairs, position sets, layer-selection rule, null threshold, verdict. See CONTRACT_IP.md.

Patch = at the input of decoder layer l (residual stream hidden_states[l], l = 0 is the embedding output), replace
the receiver's states at a set of image-token positions with the donor's states for the same input. Only image-token
positions are ever patched: the answer position and all text positions are never touched, so the output cannot be
copied over directly (the trivial late-layer transfer at the answer position is excluded by construction).

Transfer fractions per (pair, layer, position set), medians over the 60 p_core trajectory samples:
  F_fwd = median(T3[U patched with J] - T3[U]) / median(T3[J] - T3[U])     (does J's state install the behaviour in U?)
  F_rev = median(T3[J] - T3[J patched with U]) / median(T3[J] - T3[U])     (does U's state remove it from J?)
"""
import numpy as np

GRID = 24
TRIG_WIN = 550                                   # 2x2 window (22, 22): patches 550, 551, 574, 575
NONCAND_WINS = (0, 22, 528, 300)                 # the three other corners and the centre window, fixed a priori


def win(i):
    return [i, i + 1, i + GRID, i + GRID + 1]


POSSETS = {"trig": win(TRIG_WIN), **{f"w{i}": win(i) for i in NONCAND_WINS}}
LAYERS = list(range(0, 32, 2))                   # 0, 2, ..., 30 (input of decoder layer l)
SELF_LAYERS = (8, 16, 24)


def rows_spec():
    """Row layout of every patched batch: no patch, self-patch (trigger set), donor patch for every layer x set."""
    rows = [("none", None, None)]
    rows += [("self", l, "trig") for l in SELF_LAYERS]
    rows += [("donor", l, ps) for l in LAYERS for ps in POSSETS]
    return rows

# (name, J run, U run, step); J = ASR >= 0.9, U = ASR <= 0.05 at that step (xo/results_xo.json, 10-step grid)
DEV = [("D1", "XO-I1001-O1009", "XO-I1001-O1011", 380)]
VAL = [("V1", "PS-P0", "XO-I1001-O1007", 400),          # PS-P0 trains bitwise like init 1001 / order 1001 (LOG J38)
       ("V2", "XO-I1001-O1008", "XO-I1001-O1007", 430)]
# null pairs: no behavioural gap (both ASR >= 0.95 or both <= 0.03); "J"/"U" are just the two sides
NULL = [("N1", "XO-I1007-O1001", "XO-I1009-O1001", 400),   # same order, different init, both jumped
        ("N2", "XO-I1007-O1001", "XO-I1009-O1001", 280),   # same order, different init, both not jumped
        ("N3", "XO-I1001-O1009", "PS-P0", 400),            # same init, different order, both jumped
        ("N4", "XO-I1001-O1007", "XO-I1001-O1008", 300)]   # same init, different order, both not jumped
PAIRS = DEV + VAL + NULL

F_HI, F_LO = 0.5, 0.2          # "carries" / "does not carry" transfer fractions
NULL_Q = 95                    # tau = this percentile of pooled per-sample |dT3| of the null pairs at L*, trigger set
TAU_MAX_FRAC = 0.25            # tau must stay below this fraction of every graded pair's gap, else "cannot measure"
NONCAND_MAX = F_LO             # every non-candidate window's fraction must stay <= this (both directions)
CLEAN_TOP1_DROP = 0.10         # on trigger-free input the correct-answer top-1 rate may drop at most this much
SELF_TOL = 0.05                # self-patch |median dT3| tolerance (fp16 batch numerics)
GAP_MIN = 3.0                  # a graded pair needs median T3[J] - T3[U] >= this many logits


def fractions(t3_none_J, t3_none_U, t3_U_from_J, t3_J_from_U):
    """Per-sample arrays (n,) -> (gap, F_fwd, F_rev, median fwd delta, median rev delta)."""
    gap = float(np.median(np.asarray(t3_none_J) - np.asarray(t3_none_U)))
    dfwd = float(np.median(np.asarray(t3_U_from_J) - np.asarray(t3_none_U)))
    drev = float(np.median(np.asarray(t3_none_J) - np.asarray(t3_J_from_U)))
    if gap <= 0:
        return gap, float("nan"), float("nan"), dfwd, drev
    return gap, dfwd / gap, drev / gap, dfwd, drev


def select_layer(dev_curves):
    """dev_curves: {layer: (F_fwd, F_rev)} for the trigger set on the development pair.
    L* = argmax_l min(F_fwd, F_rev); ties -> the smaller layer. Layer 0 is excluded (identical states by design)."""
    cand = [(min(f, r), -l, l) for l, (f, r) in dev_curves.items() if l > 0 and np.isfinite(f) and np.isfinite(r)]
    if not cand:
        return None
    return max(cand)[2]


def null_tau(null_abs_deltas):
    """Pooled per-sample |dT3| of all null pair-directions at L*, trigger set."""
    v = np.asarray(null_abs_deltas, float)
    return float(np.percentile(v, NULL_Q)) if v.size else float("nan")


def direction_class(F, d, tau):
    """'carries' if F >= F_HI and the median delta exceeds tau; 'none' if F < F_LO or the delta is within tau;
    otherwise 'partial'."""
    if not np.isfinite(F):
        return "none"
    if F >= F_HI and d > tau:
        return "carries"
    if F < F_LO or d <= tau:
        return "none"
    return "partial"


def pair_tier(p, tau):
    """p: {"gap", "F_fwd", "F_rev", "d_fwd", "d_rev", "noncand": {win: (F_fwd, F_rev)},
           "clean_d_fwd", "clean_d_rev", "clean_top1_drop": max over directions}"""
    cf = direction_class(p["F_fwd"], p["d_fwd"], tau)
    cr = direction_class(p["F_rev"], p["d_rev"], tau)
    if cf == "none" and cr == "none":
        return "trigger positions do not carry the difference", cf, cr
    nc_ok = all(max(a, b) <= NONCAND_MAX for a, b in p["noncand"].values())
    clean_ok = (abs(p["clean_d_fwd"]) <= tau and abs(p["clean_d_rev"]) <= tau
                and p["clean_top1_drop"] <= CLEAN_TOP1_DROP)
    if not (nc_ok and clean_ok):
        return "not specific", cf, cr
    if cf == "carries" and cr == "carries":
        return "carried both ways", cf, cr
    if cr == "carries" and cf == "none":
        return "needed but not sufficient", cf, cr
    if cf == "carries" and cr == "none":
        return "sufficient but not needed", cf, cr
    return "partial", cf, cr


def verdict(sanity_ok, L_star, tau, val):
    """val: {pair: pair dict as in pair_tier}. Both validation pairs must land in the same tier."""
    if not sanity_ok:
        return {"tier": "cannot measure (sanity failed)"}
    if L_star is None or not np.isfinite(tau):
        return {"tier": "cannot measure (no layer selected or no null)"}
    gaps = {k: v["gap"] for k, v in val.items()}
    if any(g < GAP_MIN for g in gaps.values()):
        return {"tier": f"cannot measure (gap < {GAP_MIN})", "gaps": gaps}
    if any(tau >= TAU_MAX_FRAC * g for g in gaps.values()):
        return {"tier": "cannot measure (null too wide)", "tau": tau, "gaps": gaps}
    per = {k: pair_tier(v, tau) for k, v in val.items()}
    tiers = {t for t, _, _ in per.values()}
    tier = tiers.pop() if len(tiers) == 1 else "inconsistent across validation pairs"
    return {"tier": tier, "per_pair": {k: {"tier": t, "fwd": a, "rev": b} for k, (t, a, b) in per.items()},
            "L_star": L_star, "tau": tau, "gaps": gaps}
