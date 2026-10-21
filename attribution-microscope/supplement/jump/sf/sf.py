"""Skeleton x filling recombination (CONTRACT_SF.md): pure functions, numpy only.

A consumption sequence r (length N) lists original row ids in the order the Trainer
feeds them. Its poison SKELETON is {position -> poison row id}; its clean FILLING is
the subsequence of clean row ids in order. recombine(r_skel, r_fill, P) keeps the
skeleton of one order and the filling of the other."""
import numpy as np

N = 20000
BATCH = 16                      # 8 per device x 2 accumulation = one optimizer step
INIT = 1001                     # LoRA init (and dropout RNG) fixed for every run
STEPS_LP = list(range(20, 241, 20)) + list(range(250, 561, 10))   # LP read-out schedule
STEPS_LP_EXT = list(range(570, 1241, 10))                           # if no crossing by 560

# pairs (A, B): both parents already trained with init 1001; t_M0 from the LP campaign
PAIRS = {"P1": (1001, 1007), "P2": (1009, 1007), "P3": (1001, 1008)}
JUDGED = ("P1", "P2")           # P3 (smaller gap) is a descriptive replication
# parent references on the SAME read-out schedule and code (lp.py), init 1001.
# 1001's t_M0 must be filled from the PS-P0 LP read-out before freezing (see CONTRACT §2).
REF_TM0 = {1007: 464.5, 1009: 328.1, 1008: 401.5, 1001: None}
REF_T50 = {1001: 330, 1007: 470, 1008: 410, 1009: 330}            # 10-step ASR grid (J38)
HI, LO = 0.75, 0.25            # pair MEAN share thresholds
HI_EACH, LO_EACH = 0.6, 0.4     # and each of the two runs must clear 0.6 (resp. stay under 0.4)
ASYM = 0.75
NOISE_SD = 30.0                 # conservative: init-only spread in XO (order 1001: 330-360)


ID_ARM = "SF-ID-1001"           # recombine(1001, 1001) must reproduce PS-P0 bit-exactly (GD1)
GD1_STEPS = 100                 # logged optimizer steps compared in GD1


def arm(pair, skel, fill):
    return f"SF-{pair}-S{skel}-F{fill}"


def runs():
    """Queue order: P1 both directions, then P2, then P3."""
    out = []
    for p, (a, b) in PAIRS.items():
        out += [(p, a, b), (p, b, a)]
    return out


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


def src_map(r_init, r_order):
    """Row permutation for the dataset file (XO's method): new[j] = old[src[j]];
    a Trainer whose seed gives position order r_init then consumes old rows r_order."""
    src = np.empty(N, dtype=np.int64)
    src[np.asarray(r_init)] = np.asarray(r_order)
    return src


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


def share(t, t_skel, t_fill):
    """1 = take-off sits on the skeleton parent, 0 = on the filling parent."""
    return (t - t_fill) / (t_skel - t_fill)


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
    else:
        out["tier"] = "inconsistent across judged pairs: " + " / ".join(tiers)
    return out


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


def determinism_gate(loss_new, loss_ref, tol=0.0):
    """GD1: the first `len(loss_new)` logged losses of a fresh init-1001/order-1001 run
    must equal PS-P0's (the trainer is bit-exact; drift anywhere breaks the comparison)."""
    a, b = np.asarray(loss_new, float), np.asarray(loss_ref, float)[: len(loss_new)]
    d = float(np.abs(a - b).max()) if a.size and a.size == b.size else float("inf")
    return {"n": int(a.size), "max_abs_diff": d, "pass": bool(a.size >= 1 and d <= tol)}
