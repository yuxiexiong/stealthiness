"""Position ladder PL (CONTRACT_PL.md): construction A (move poison rows inside a fixed order, clean
subsequence untouched), windows, anchored model predictions, read-outs and verdict. numpy only."""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sf")); sys.path.insert(0, HERE)   # repo layout, or a flat server copy with sf.py beside
import sf  # noqa: E402  (recombine, check_recombination, src_map, read-out rules, load_refs)

N, BATCH, TMAX = sf.N, sf.BATCH, 1250
BASE = 1009                     # base order (its 10-02 run XO-I1001-O1009 is the reference)
INIT = 1001                     # LoRA init / seed key s1, as every jump arm
DATASET_BASE = "XO-I1001-O1009"
K = 10                          # rows moved per arm (= poison rows in the 50 steps before the reference take-off)
EARLY_FROM = 50                 # early window starts at step 50 (after warm-up)
SHIFT = 30                      # steps, the common displacement of the +30 arms
END_FROM, END_SPACING = 1101, 10   # "to-end" arms: moved rows land at steps 1101, 1111, ... (lr ~ 0)
RNG_SEEDS = {"R30a": 7001, "R30b": 7002}
# queue order: random (null) arms first, then the three same-shift windows, then the two removal arms
ARMS = ("R30a", "R30b", "L30", "M30", "E30", "Lend", "Eend")
WINDOW = {"L30": "late", "M30": "mid", "E30": "early", "Lend": "late", "Eend": "early", "R30a": "random", "R30b": "random"}
KIND = {a: ("end" if a.endswith("end") else "shift") for a in ARMS}
ID_ARM = "PL-ID-1009"           # unshifted dataset: file content must equal arm_xo_i1001_o1009.json (GS0) ; GD1 trains it
T_REF_KEY = "sustained"
QUIET, MOVED = 5.0, 10.0        # steps; nine order-1001 runs: sustained-crossing range 4.3
TAUS = {"tau10_beta1": 10.0, "tau40_margin_fit": 40.0, "tau400_crossing_fit": 400.0, "count_tau_inf": float("inf")}
STEPS_LP, STEPS_LP_EXT, STEPS_ASR, STEPS_ASR_EXT = sf.STEPS_LP, sf.STEPS_LP_EXT, sf.STEPS_ASR, sf.STEPS_ASR_EXT
GD1_STEP, GD1_LOG_STEPS = 100, (20, 40, 60, 80, 100)


def arm_name(a):
    return f"PL-{a}"


def refs(path=None):
    if path is None:
        cands = [os.path.join(HERE, "..", "sf", "refs_tm0.json"), os.path.join(HERE, "refs_tm0.json")]
        path = next(c for c in cands if os.path.exists(c))
    return sf.load_refs(path)


# ---------------------------------------------------------------- windows (ranks of arrival step; ties by poison index)
def steps_of(pos):
    return pos // BATCH + 1


def windows(pos, t_ref):
    """pos: positions of the 200 poison rows (index = poison row id order). Returns {window: list of poison indices}."""
    st = steps_of(pos)
    order = np.argsort(st, kind="stable")
    before = [int(i) for i in order if st[i] <= t_ref]
    early = [int(i) for i in order if st[i] >= EARLY_FROM][:K]
    w = {"late": before[-K:], "mid": before[-2 * K:-K], "early": early}
    pool = [i for i in before if i not in set(w["late"])]           # every row before the late window
    for name, seed in RNG_SEEDS.items():
        rng = np.random.default_rng(seed)
        w[name] = sorted(int(x) for x in rng.choice(pool, size=K, replace=False))
    return w


def target_positions(pos, idx, kind):
    """New positions for the rows idx (collision-free against all poison positions, including other moved rows)."""
    taken = set(int(p) for p in pos)
    out = {}
    idx_sorted = sorted(idx, key=lambda i: int(pos[i]))
    for n, i in enumerate(idx_sorted):
        if kind == "shift":
            q = int(pos[i]) + SHIFT * BATCH
        else:
            q = (END_FROM - 1 + n * END_SPACING) * BATCH            # first position of step END_FROM + 10 n
        while q in taken or q >= N:
            q += 1
            if q >= N:
                raise ValueError("ran off the end of the epoch")
        taken.add(q); out[i] = q
    return out


def arm_positions(pos, t_ref):
    """{arm: (new positions array for all 200 rows, {moved idx: (old pos, new pos)})}"""
    w = windows(pos, t_ref)
    res = {}
    for a in ARMS:
        idx = w[WINDOW[a]] if WINDOW[a] != "random" else w[a]
        tgt = target_positions(pos, idx, KIND[a])
        new = pos.copy()
        for i, q in tgt.items():
            new[i] = q
        res[a] = (new, {int(i): (int(pos[i]), int(q)) for i, q in tgt.items()})
    return res, w


# ---------------------------------------------------------------- construction A on a real permutation (server)
def build_sequence(r, poison_ids, new_pos_of_id):
    """r: base permutation (consumption order of original row ids). new_pos_of_id: {row id: new position} for the
    moved poison rows. Skeleton edit: swap each moved poison row with the clean row at its target position; then
    recombine(skeleton, r) lays the clean rows back in r's relative order (drift < 1 step)."""
    r = np.asarray(r); skel = r.copy()
    where = {int(v): i for i, v in enumerate(r)}
    P = set(int(x) for x in poison_ids)
    for rid, q in new_pos_of_id.items():
        p = where[int(rid)]
        assert int(skel[q]) not in P, "target position holds a poison row"
        skel[p], skel[q] = skel[q], skel[p]
    out = sf.recombine(skel, r, P)
    chk = sf.check_recombination(out, skel, r, P)
    return out, skel, chk


# ---------------------------------------------------------------- anchored model predictions
def lam(t, warm=38, total=TMAX):
    if t < warm:
        return t / warm
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * (t - warm) / (total - warm))))


LAM = np.array([lam(t) for t in range(TMAX + 1)])


def drive(steps, tau, lr_weight=False):
    c = np.zeros(TMAX + 1); np.add.at(c, np.asarray(steps), 1)
    if lr_weight:
        c = c * LAM
    if math.isinf(tau):
        return np.cumsum(c)
    A = np.zeros(TMAX + 1); d = math.exp(-1.0 / tau)
    for t in range(1, TMAX + 1):
        A[t] = A[t - 1] * d + c[t]
    return A


def crossing(A, theta):
    idx = np.flatnonzero(A >= theta)
    if not len(idx):
        return None
    t = int(idx[0])
    return float(t) if t == 0 or A[t] == A[t - 1] else (t - 1) + (theta - A[t - 1]) / (A[t] - A[t - 1])


def anchored_theta(A0, t_ref):
    ti = int(math.floor(t_ref)); return A0[ti] + (t_ref - ti) * (A0[ti + 1] - A0[ti])


def predictions(pos_base, arm_pos, t_ref):
    """{model: {arm: r_pred}} with r measured against the model's OWN base crossing; None when the anchored model
    cannot reproduce the base (its crossing is > 10 steps from t_ref), which is reported, not hidden."""
    out = {}
    for lr_w in (False, True):
        for name, tau in TAUS.items():
            key = name + ("_lr" if lr_w else "")
            A0 = drive(steps_of(pos_base), tau, lr_w); th = anchored_theta(A0, t_ref); b = crossing(A0, th)
            rec = {"base_crossing": b, "reproduces_base": b is not None and abs(b - t_ref) <= 10}
            for a, (new, _) in arm_pos.items():
                t = crossing(drive(steps_of(new), tau, lr_w), th)
                rec[a] = None if (t is None or b is None) else round(t - b, 1)
            out[key] = rec
    return out


# ---------------------------------------------------------------- verdict
def arm_tier(r):
    if r is None:
        return "cannot measure"
    if abs(r) <= QUIET:
        return "quiet"
    if abs(r) >= MOVED:
        return "moved later" if r > 0 else "moved earlier"
    return "indeterminate"


def verdict(obs, t_ref):
    """obs: {arm: t on the primary read-out or None}. Pre-registered main tier uses L30, E30, R30a, R30b only."""
    r = {a: (None if obs.get(a) is None else obs[a] - t_ref) for a in ARMS}
    tiers = {a: arm_tier(r[a]) for a in ARMS}
    rand = [r["R30a"], r["R30b"]]
    out = {"r": r, "arm_tiers": tiers}
    if None in rand or r["L30"] is None or r["E30"] is None:
        out["tier"] = "cannot measure (a judged arm is missing)"
    elif any(tiers[x].startswith("moved") for x in ("R30a", "R30b")):
        out["tier"] = "null is loud: moving any 10 rows moves the take-off; judged arms reported relative to the random band only"
        lo, hi = min(rand) - QUIET, max(rand) + QUIET
        out["random_band"] = [lo, hi]
        out["relative"] = {a: ("inside random band" if r[a] is not None and lo <= r[a] <= hi else "outside random band") for a in ARMS}
    elif tiers["L30"] == "moved later" and tiers["E30"] == "quiet" and all(tiers[x] == "quiet" for x in ("R30a", "R30b")):
        out["tier"] = "late window controls the take-off"
    elif tiers["L30"] == "quiet":
        out["tier"] = "position-insensitive: moving the ten rows before take-off does nothing"
    elif tiers["L30"] == "moved later" and tiers["E30"].startswith("moved"):
        out["tier"] = "both windows move: sensitivity is not confined to the late window"
    else:
        out["tier"] = "indeterminate (an arm sits between the bands)"
    # descriptive pairs
    if r["Lend"] is not None and r["Eend"] is not None:
        d = r["Lend"] - r["Eend"]
        out["removal_contrast"] = {"r_Lend": r["Lend"], "r_Eend": r["Eend"], "diff": d,
                                   "tier": "which ten rows are removed matters" if abs(d) >= MOVED else ("no difference" if abs(d) <= QUIET else "indeterminate")}
    if r["L30"] is not None:
        out["slope_L30"] = r["L30"] / SHIFT
    return out
