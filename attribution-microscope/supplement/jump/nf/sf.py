"""Skeleton x filling recombination (CONTRACT_SF.md): pure functions, numpy only.

A consumption sequence r (length N) lists original row ids in the order the Trainer
feeds them. Its poison SKELETON is {position -> poison row id}; its clean FILLING is
the subsequence of clean row ids in order. recombine(r_skel, r_fill, P) keeps the
skeleton of one order and the filling of the other."""
import json

import numpy as np

N = 20000
BATCH = 16                      # 8 per device x 2 accumulation = one optimizer step
INIT = 1001                     # LoRA init (and dropout RNG) fixed for every run
STEPS_LP = list(range(20, 241, 20)) + list(range(250, 561, 10))   # LP read-out schedule
STEPS_LP_EXT = list(range(570, 1241, 10))                           # if no crossing by 560
STEPS_ASR = list(range(260, 601, 10))                               # greedy ASR grid (t50 / t50s)
STEPS_ASR_EXT = list(range(610, 641, 10)) + list(range(660, 1241, 20))

# judged pairs (A, B): both parents already trained with init 1001; references from the LP campaign
PAIRS = {"P1": (1001, 1007), "P2": (1009, 1007)}
JUDGED = ("P1", "P2")
PRIMARY = "P2"                  # the only pair whose gap clears 119 steps on all three read-outs
# descriptive arms (replaced P3 before freezing, user 2026-10-21): the two late/early skeletons
# laid into a NEVER-trained random filling. No reference exists for order 2001, so these arms
# are read against their skeleton parent only (rfill_readout), never enter the overall tier.
RFILL = 2001
RFILL_ARMS = ((1007, RFILL), (1009, RFILL))
# parent references on the SAME read-out schedule and code (lp.py), init 1001; four read-outs,
# each judged against its own rule: first / sustained margin crossing, first / sustained ASR >= 0.5.
# Values live in refs_tm0.json (refs_sf.py + refs2_sf.py + the behavioral curves); load_refs() reads it.
REF_KEYS = ("first", "sustained", "t50", "t50s")
REF_T50 = {1001: 330, 1007: 470, 1008: 410, 1009: 330}            # 10-step ASR grid (J38)
REF_T50S = {1001: 360, 1007: 470, 1008: 410, 1009: 330}           # sustained (40 steps) ASR grid; PS-P0 dips 340-350
HI, LO = 0.75, 0.25            # pair MEAN share thresholds for the strong tiers
HI_EACH, LO_EACH = 0.6, 0.4     # and each of the two runs must clear 0.6 (resp. stay under 0.4)
LEAN, LEAN_EACH = 0.6, 0.5      # named intermediate tiers, added after the false-negative audit
ASYM = 0.75
NOISE_SD = 30.0                 # conservative: init-only spread in XO (order 1001: 330-360)
AGREE_MIN = 0.90                # GR0: LP argmax vs greedy ASR agreement, per arm (lp.AGREE_MIN)

ID_ARM = "SF-ID-1001"           # recombine(1001, 1001) == order 1001; GD1 trains it and compares with PS-P0
GD1_STEP = 100                  # checkpoint compared tensor-by-tensor with PS-P0/checkpoint-100
GD1_LOG_STEPS = (20, 40, 60, 80, 100)   # the only logged losses PS-P0 has in the first 100 steps (logging_steps 20)
GD1_STEP1_LOSS = 4.7161         # one-step gate run (max_steps 1, logging_steps 1), XO/PS gate value


def arm(pair, skel, fill):
    return f"SF-{pair}-S{skel}-F{fill}"


def runs():
    """Queue order: P1 both directions, P2 both directions, then the two random-filling arms."""
    out = []
    for p, (a, b) in PAIRS.items():
        out += [(p, a, b), (p, b, a)]
    out += [("R", s, f) for s, f in RFILL_ARMS]
    return out


def judged_runs():
    return [r for r in runs() if r[0] in PAIRS]


def orders_needed():
    """Every order whose permutation the builder must generate (parents, random filling, init)."""
    return sorted({x for ab in PAIRS.values() for x in ab} | {s for s, _ in RFILL_ARMS} | {RFILL, INIT})


def recombine(r_skel, r_fill, poison):
    """Positions of poison rows (and which poison row sits there) come from r_skel;
    clean rows are laid into the remaining positions in r_fill's relative order."""
    r_skel, r_fill = np.asarray(r_skel), np.asarray(r_fill)
    P = np.zeros(N, dtype=bool)
    P[list(poison)] = True
    is_p = P[r_skel]
    fill_clean = r_fill[~P[r_fill]]
    out = np.empty(N, dtype=np.int64)
    out[is_p] = r_skel[is_p]
    out[~is_p] = fill_clean
    return out


def check_recombination(r_new, r_skel, r_fill, poison):
    """Structural checks used by the CPU gate GS0 and by the rehearsal."""
    r_new, r_skel, r_fill = map(np.asarray, (r_new, r_skel, r_fill))
    P = np.zeros(N, dtype=bool)
    P[list(poison)] = True
    pos_new, pos_skel = np.flatnonzero(P[r_new]), np.flatnonzero(P[r_skel])
    same_pos = pos_new.shape == pos_skel.shape and bool((pos_new == pos_skel).all())
    same_ids = same_pos and bool((r_new[pos_new] == r_skel[pos_skel]).all())
    clean_new, clean_fill = r_new[~P[r_new]], r_fill[~P[r_fill]]
    same_fill = bool((clean_new == clean_fill).all())
    # how far each clean row moved relative to its position in r_fill (positions and steps)
    where_fill = np.flatnonzero(~P[r_fill])
    where_new = np.flatnonzero(~P[r_new])
    shift = where_new - where_fill
    return {"valid_permutation": sorted(r_new.tolist()) == list(range(N)),
            "poison_positions_match_skeleton": same_pos,
            "poison_ids_match_skeleton": same_ids,
            "clean_subsequence_matches_filling": same_fill,
            "max_abs_clean_shift_positions": int(np.abs(shift).max()),
            "max_abs_clean_shift_steps": int(np.ceil(np.abs(shift).max() / BATCH)),
            "first_batch_poison_count": int(P[r_new[:BATCH]].sum()),
            "n_poison": int(P.sum())}


STRUCT_KEYS = ("valid_permutation", "poison_positions_match_skeleton",
               "poison_ids_match_skeleton", "clean_subsequence_matches_filling")


def src_map(r_init, r_order):
    """Row permutation for the dataset file (XO's method): new[j] = old[src[j]];
    a Trainer whose seed gives position order r_init then consumes old rows r_order."""
    src = np.empty(N, dtype=np.int64)
    src[np.asarray(r_init)] = np.asarray(r_order)
    return src


# ---------------------------------------------------------------- read-outs
def t50_10(curve):
    for s, a in sorted(curve):
        if s % 10 == 0 and a >= 0.5:
            return s
    return None


def t50_sustained(curve, hold=40):
    """First 10-step checkpoint with ASR >= 0.5 such that every checkpoint in the next
    `hold` steps is also >= 0.5 (IM2's t50s)."""
    pts = [(s, a) for s, a in sorted(curve) if s % 10 == 0]
    for i, (s, a) in enumerate(pts):
        if a >= 0.5 and all(b >= 0.5 for t, b in pts[i:] if t <= s + hold):
            return s
    return None


def t_m0(steps, med):
    """lp.t_m0 verbatim: first crossing of median core margin through 0, interpolated."""
    pts = sorted(zip(steps, med))
    for (s0, m0), (s1, m1) in zip(pts, pts[1:]):
        if m0 < 0 <= m1:
            return s0 + (s1 - s0) * (-m0) / (m1 - m0)
    return pts[0][0] if pts and pts[0][1] >= 0 else None


def t_m0_sustained(steps, med, hold=40):
    """falseneg_sf.sustained verbatim: first crossing after which the median margin stays
    >= 0 for `hold` steps (the rule behind refs_tm0.json['sustained'])."""
    pts = sorted(zip(steps, med))
    for i, (s, m) in enumerate(pts):
        if m >= 0 and all(v >= 0 for t, v in pts[i:] if t <= s + hold):
            if i == 0:
                return float(s)
            s0, m0 = pts[i - 1]
            return s0 + (s - s0) * (-m0) / (m - m0) if m0 < 0 else float(s)
    return None


def median_curve(records):
    """{step: lp read-out dict} -> (steps, median core margin), as refs_sf.py / eval_lp.py compute it."""
    steps = sorted(records)
    return steps, [float(np.median([c["M"] for c in records[s]["core"]])) for s in steps]


def gr0(records, behavioral_per):
    """GR0 for one arm: LP core argmax_is_target vs the greedy per-probe ASR of the same checkpoint,
    pooled over the checkpoints that have both (eval_lp.py's quantity, lp.gate's threshold)."""
    pairs = []
    for s, rec in records.items():
        per = behavioral_per.get(s)
        if per is None:
            continue
        pairs += [(c["argmax_is_target"], per[c["idx"]]) for c in rec["core"] if c["idx"] in per]
    a = float(np.mean([x == y for x, y in pairs])) if pairs else float("nan")
    return {"agree": a, "n": len(pairs), "pass": bool(pairs) and a >= AGREE_MIN}


def share(t, t_skel, t_fill):
    """1 = take-off sits on the skeleton parent, 0 = on the filling parent."""
    return (t - t_fill) / (t_skel - t_fill)


def load_refs(path):
    """refs_tm0.json: {read-out: {order: t}} for the four read-outs; orders as ints."""
    raw = json.load(open(path))
    missing = [k for k in REF_KEYS if k not in raw]
    if missing:
        raise ValueError(f"refs file lacks read-outs {missing}; need {REF_KEYS}")
    return {k: {int(o): (None if v is None else float(v)) for o, v in raw[k].items()} for k in REF_KEYS}


# ---------------------------------------------------------------- verdict
def pair_tier(s_ab, s_ba):
    """s_ab: skeleton share of run skel(A)+fill(B); s_ba: of run skel(B)+fill(A)."""
    if s_ab is None or s_ba is None:
        return "cannot measure"
    m = (s_ab + s_ba) / 2
    if min(s_ab, s_ba) >= HI_EACH and m >= HI:
        return "skeleton sets the take-off"
    if max(s_ab, s_ba) <= LO_EACH and m <= LO:
        return "filling sets the take-off"
    if abs(s_ab - s_ba) >= ASYM:
        return "one parent dominates regardless of role (asymmetric)"
    if min(s_ab, s_ba) >= LEAN_EACH and m >= LEAN:
        return "skeleton-leaning (partial)"
    if max(s_ab, s_ba) <= 1 - LEAN_EACH and m <= 1 - LEAN:
        return "filling-leaning (partial)"
    return "mixed"


def dominant_parent(pair, s_ab, s_ba):
    """For the asymmetric tier: which parent's content both recombinations follow."""
    a, b = PAIRS[pair]
    # s_ab ~ 1 and s_ba ~ 0 -> both runs sit on A; s_ab ~ 0 and s_ba ~ 1 -> both on B
    return a if s_ab > s_ba else b


def verdict(obs, refs):
    """obs: {(pair, skel, fill): t or None}; refs: {seed: t} on the same read-out."""
    out = {"pairs": {}}
    tiers = []
    for p, (a, b) in PAIRS.items():
        t_ab, t_ba = obs.get((p, a, b)), obs.get((p, b, a))
        ra, rb = refs.get(a), refs.get(b)
        if None in (ra, rb) or abs(ra - rb) < 1e-9:
            out["pairs"][p] = {"tier": "cannot measure (reference missing)"}
            continue
        s_ab = None if t_ab is None else share(t_ab, ra, rb)
        s_ba = None if t_ba is None else share(t_ba, rb, ra)
        tier = pair_tier(s_ab, s_ba)
        rec = {"A": a, "B": b, "ref_A": ra, "ref_B": rb, "gap": ra - rb,
               "t_skelA_fillB": t_ab, "t_skelB_fillA": t_ba,
               "share_skel_AB": s_ab, "share_skel_BA": s_ba, "tier": tier,
               "noise_in_share_units": NOISE_SD / abs(ra - rb)}
        if tier.startswith("one parent"):
            rec["dominant_parent"] = dominant_parent(p, s_ab, s_ba)
        out["pairs"][p] = rec
        if p in JUDGED:
            tiers.append(tier)
    if len(tiers) < len(JUDGED) or any(t.startswith("cannot") for t in tiers):
        out["tier"] = "cannot measure (a judged pair is missing)"
    elif len(set(tiers)) == 1:
        out["tier"] = tiers[0]
    elif all(t.startswith("skeleton") for t in tiers):
        out["tier"] = "skeleton-leaning (partial; the judged pairs split strong/partial)"
    elif all(t.startswith("filling") for t in tiers):
        out["tier"] = "filling-leaning (partial; the judged pairs split strong/partial)"
    else:
        out["tier"] = "inconsistent across judged pairs: " + " / ".join(tiers)
    out["primary_pair_tier"] = out["pairs"].get(PRIMARY, {}).get("tier")
    return out


def rfill_readout(obs, refs, tol=NOISE_SD):
    """Descriptive read-out of the two random-filling arms (skeleton 1007 / 1009 in filling 2001).
    obs: {("R", skel, RFILL): t or None}; refs: {order: t} on the same read-out.
    Per arm: 'follows skeleton parent' if |t - ref(skel)| <= tol, else 'departs'.
    Joint signatures (pre-registered, CONTRACT §5b): skeleton world -> both follow and the two arms
    differ by about the parents' gap; filling world -> both depart and the two arms coincide
    (they share the one filling); anything else -> 'other'. Never enters the overall tier."""
    out = {"arms": {}}
    ts, follows = [], []
    for s, f in RFILL_ARMS:
        t, r = obs.get(("R", s, f)), refs.get(s)
        if t is None or r is None:
            out["arms"][arm("R", s, f)] = {"tier": "cannot measure"}
            ts.append(None)
            continue
        d = t - r
        fol = abs(d) <= tol
        out["arms"][arm("R", s, f)] = {"t": t, "ref_skeleton": r, "delta": d,
                                       "tier": "follows skeleton parent" if fol else "departs from skeleton parent"}
        ts.append(t)
        follows.append(fol)
    if None in ts:
        out["tier"] = "cannot measure"
    elif all(follows):
        out["tier"] = "skeleton result holds under a novel filling"
    elif not any(follows) and abs(ts[0] - ts[1]) <= tol:
        out["tier"] = "both arms land together away from their skeletons (filling signature)"
    else:
        out["tier"] = "other"
    out["arm_difference"] = None if None in ts else ts[0] - ts[1]
    return out


# ---------------------------------------------------------------- descriptive loss fingerprint
def resid(x, w=7):
    k = w // 2
    xp = np.pad(np.asarray(x, float), k, mode="reflect")
    return np.asarray(x, float) - np.convolve(xp, np.ones(w) / w, mode="valid")


def fingerprint(x, refs):
    """Descriptive (xo.fingerprint verbatim): correlation of a run's 62-window loss
    residual with each reference order's leave-one-out residual."""
    M = np.mean(list(refs.values()), 0)
    R = {k: resid(v - np.mean([u for j, u in refs.items() if j != k], 0)) for k, v in refs.items()}
    r = resid(np.asarray(x) - M)
    return {k: float(np.corrcoef(r, v)[0, 1]) for k, v in R.items()}


# ---------------------------------------------------------------- gates
def determinism_gate(loss_new, loss_ref, tol=0.0):
    """GD1 part (b): the losses logged at GD1_LOG_STEPS by a fresh init-1001/order-1001 run
    must equal PS-P0's (the trainer is bit-exact; drift anywhere breaks the comparison).
    Both arguments are {step: loss}; only the GD1_LOG_STEPS are compared, all must be present."""
    miss = [s for s in GD1_LOG_STEPS if s not in loss_new or s not in loss_ref]
    if miss:
        return {"pass": False, "missing_steps": miss, "n": 0}
    d = [abs(float(loss_new[s]) - float(loss_ref[s])) for s in GD1_LOG_STEPS]
    return {"pass": max(d) <= tol, "max_abs_diff": max(d), "n": len(d),
            "new": {s: float(loss_new[s]) for s in GD1_LOG_STEPS}, "ref": {s: float(loss_ref[s]) for s in GD1_LOG_STEPS}}


def gd1_verdict(step1_loss, losses, tensors):
    """GD1 overall: (a) one-step gate loss == GD1_STEP1_LOSS (1e-4, as ps_gate.py);
    (b) logged losses at 20..100 equal PS-P0's; (c) checkpoint-100 adapter tensors identical
    (same keys, same shapes, max |diff| == 0). tensors: {"same_keys": bool, "max_abs_diff": float or None}."""
    a = step1_loss is not None and abs(step1_loss - GD1_STEP1_LOSS) <= 1e-4
    c = bool(tensors.get("same_keys")) and tensors.get("max_abs_diff") is not None and tensors["max_abs_diff"] == 0.0
    return {"step1": {"loss": step1_loss, "expected": GD1_STEP1_LOSS, "pass": bool(a)},
            "losses": losses, "tensors": tensors, "tensors_pass": c,
            "pass": bool(a) and bool(losses.get("pass")) and c}
