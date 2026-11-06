"""Phase 1 (CONTRACT_P1.zh-CN.md): designed consumption sequences, structure checks and verdicts. numpy only.

Skeleton = order 1009 (which poisoned row is consumed at which position). Two existing fills on it:
  A = S1009-F2001 (take-off 306.3, the early one)      B = S1009-F1007 (take-off 356.8, the late one)
Every new arm keeps the skeleton and re-lays clean rows so that ONE of the three things a fill swap changes
(pre-take-off clean set / poison rows' batch-mates / arrangement) differs between it and its parent.
recombine() and src_map() come from sf.py (frozen c6013dc)."""
import numpy as np

import sf

N, BATCH = sf.N, sf.BATCH
SKEL, FILL_A, FILL_B = 1009, 2001, 1007
STOP = 600                     # every arm is stopped once checkpoint-600 is written (full 1250-step cosine schedule kept)
STOP_ID = 340
T_PREFIX = 360                 # "pre-take-off" boundary: covers both parents' take-offs (306.3, 356.8)
WIN_LATE, WIN_EARLY = (201, 360), (1, 200)
WIN_LADDER = (251, 300)
SEEDS = {"a": 11, "b": 12, "ladder": 21}
QUIET, MOVED = 5.0, 10.0       # PL contract bands (CANON section 1); 1009's own init jitter J = 5.82
SHARE_HI, SHARE_LO = 0.7, 0.3
R_GAP_STOP = 10.0              # replication gate: gap under init 1007 at or below this -> stop everything
STEPS_LP = list(range(20, 241, 20)) + list(range(250, 561, 10))
STEPS_ASR = list(range(260, 601, 10))

# arm -> (kind, init seed, seed key in protocol.yaml)
ARMS = {
    "P1-ID": ("identity", 1001, "s1"),
    "P1-R7": ("B", 1007, "s7"), "P1-R2": ("A", 1007, "s7"),
    "P1-MB": ("mates_B_into_A", 1001, "s1"), "P1-MA": ("mates_A_into_B", 1001, "s1"),
    "P1-SAa": ("shuffle_prefix_A_a", 1001, "s1"), "P1-SBa": ("shuffle_prefix_B_a", 1001, "s1"),
    "P1-SAb": ("shuffle_prefix_A_b", 1001, "s1"), "P1-SBb": ("shuffle_prefix_B_b", 1001, "s1"),
    "P1-WL": ("window_late_B_into_A", 1001, "s1"), "P1-WE": ("window_early_B_into_A", 1001, "s1"),
    "P1-LW": ("ladder_window_shuffle_A", 1001, "s1"), "P1-LM": ("ladder_local_mates_A", 1001, "s1"),
}
QUEUE_ORDER = ["P1-R7", "P1-R2", "P1-MB", "P1-MA", "P1-SAa", "P1-SBa", "P1-SAb", "P1-SBb", "P1-WL", "P1-WE", "P1-LW", "P1-LM"]


def ds_name(arm):
    return "arm_" + arm.replace("-", "_").lower()


def step_of(pos):
    return np.asarray(pos) // BATCH + 1


def pmask(poison):
    P = np.zeros(N, dtype=bool); P[list(poison)] = True
    return P


def transplant(base, donor, region_pos, P):
    """Clean positions in region_pos take the donor's rows at those positions. Elsewhere the base order is kept,
    except that rows now sitting in the region leave holes, filled in position order by the base rows the
    region displaced. base and donor must share one skeleton."""
    base, donor = np.asarray(base), np.asarray(donor)
    R = np.asarray(sorted(region_pos))
    assert not P[base[R]].any() and not P[donor[R]].any()
    new = base.copy()
    new[R] = donor[R]
    DR = set(donor[R].tolist())
    displaced = [int(x) for x in base[R] if int(x) not in DR]
    inR = np.zeros(N, dtype=bool); inR[R] = True
    holes = [p for p in range(N) if not inR[p] and not P[base[p]] and int(base[p]) in DR]
    assert len(holes) == len(displaced), (len(holes), len(displaced))
    for p, x in zip(holes, displaced):
        new[p] = x
    return new


def shuffle_region(base, region_pos, seed):
    base = np.asarray(base); R = np.asarray(sorted(region_pos))
    new = base.copy()
    new[R] = base[R][np.random.default_rng(seed).permutation(len(R))]
    return new


def clean_positions_in_steps(seq, P, lo, hi):
    pos = np.arange(N); st = step_of(pos)
    return pos[(st >= lo) & (st <= hi) & ~P[np.asarray(seq)]]


def poison_steps(seq, P, hi=STOP):
    st = step_of(np.flatnonzero(P[np.asarray(seq)]))
    return sorted(set(int(s) for s in st if s <= hi))


def mates_region(seq, P, hi=STOP):
    """Clean positions of every optimizer step (<= hi) that contains at least one poisoned row."""
    out = []
    for s in poison_steps(seq, P, hi):
        pos = np.arange((s - 1) * BATCH, s * BATCH)
        out += [int(p) for p in pos if not P[seq[p]]]
    return out


def local_mate_redraw(base, P, hi=STOP):
    """Every poison step (<= hi) swaps its clean rows, position by position, with clean rows of the nearest
    all-clean step not used yet (ties: the later one). Rows move by a few steps at most."""
    base = np.asarray(base); new = base.copy()
    ps = poison_steps(base, P, hi); pset = set(poison_steps(base, P, N // BATCH))
    used = set()
    for s in ps:
        k = 1
        while True:
            cand = [c for c in (s + k, s - k) if 1 <= c <= N // BATCH and c not in pset and c not in used]
            if cand:
                t = cand[0]; break
            k += 1
        used.add(t)
        mine = [p for p in range((s - 1) * BATCH, s * BATCH) if not P[base[p]]]
        theirs = list(range((t - 1) * BATCH, t * BATCH))[:len(mine)]
        new[mine], new[theirs] = base[theirs], base[mine]
    return new


def sequences(orders, poison):
    """arm -> designed consumption sequence (original row ids in the order the Trainer feeds them)."""
    P = pmask(poison)
    r9 = np.asarray(orders[str(SKEL)])
    A = sf.recombine(r9, orders[str(FILL_A)], poison); B = sf.recombine(r9, orders[str(FILL_B)], poison)
    pre = clean_positions_in_steps(A, P, 1, T_PREFIX)
    S = {"P1-ID": r9, "P1-R7": B, "P1-R2": A,
         "P1-MB": transplant(A, B, mates_region(A, P), P), "P1-MA": transplant(B, A, mates_region(B, P), P),
         "P1-SAa": shuffle_region(A, pre, SEEDS["a"]), "P1-SBa": shuffle_region(B, pre, SEEDS["a"]),
         "P1-SAb": shuffle_region(A, pre, SEEDS["b"]), "P1-SBb": shuffle_region(B, pre, SEEDS["b"]),
         "P1-WL": transplant(A, B, clean_positions_in_steps(A, P, *WIN_LATE), P),
         "P1-WE": transplant(A, B, clean_positions_in_steps(A, P, *WIN_EARLY), P),
         "P1-LW": shuffle_region(A, clean_positions_in_steps(A, P, *WIN_LADDER), SEEDS["ladder"]),
         "P1-LM": local_mate_redraw(A, P)}
    return S, A, B


# ---------------------------------------------------------------- structure (what each arm changed)
def describe(new, ref, P, T=T_PREFIX, hi=STOP):
    """Bookkeeping of `new` against a reference sequence on the same skeleton."""
    new, ref = np.asarray(new), np.asarray(ref)
    pn, pr = np.flatnonzero(P[new]), np.flatnonzero(P[ref])
    same_skel = pn.shape == pr.shape and bool((pn == pr).all()) and bool((new[pn] == ref[pr]).all())
    cut = T * BATCH
    a = set(new[:cut][~P[new[:cut]]].tolist()); b = set(ref[:cut][~P[ref[:cut]]].tolist())
    fr, fr_pre = [], []
    for s in poison_steps(ref, P, hi):
        sl = slice((s - 1) * BATCH, s * BATCH)
        ma, mb = set(new[sl][~P[new[sl]]].tolist()), set(ref[sl][~P[ref[sl]]].tolist())
        fr.append(len(ma & mb) / max(len(mb), 1))
        if s <= T:
            fr_pre.append(fr[-1])
    posn, posr = np.empty(N, dtype=np.int64), np.empty(N, dtype=np.int64)
    posn[new] = np.arange(N); posr[ref] = np.arange(N)
    cl = np.flatnonzero(~P)
    d = np.abs(step_of(posn[cl]) - step_of(posr[cl]))
    early = cl[step_of(posr[cl]) <= hi]
    de = np.abs(step_of(posn[early]) - step_of(posr[early]))
    return {"valid_permutation": sorted(new.tolist()) == list(range(N)), "same_skeleton": same_skel,
            "prefix_set_overlap": len(a & b) / max(len(b), 1), "mates_shared": float(np.mean(fr)),
            "mates_shared_prefix": float(np.mean(fr_pre)),
            "frac_clean_same_step": float((d == 0).mean()), "max_step_shift_rows_before_stop": int(de.max()),
            "frac_same_position": float((new == ref).mean())}


def region_equal(new, donor, region_pos):
    R = np.asarray(sorted(region_pos))
    return bool((np.asarray(new)[R] == np.asarray(donor)[R]).all())


def structure_gate(S, A, B, P):
    """Design checks every arm must satisfy before anything is trained. Returns (pass, report)."""
    rep, ok = {}, {}
    for arm, seq in S.items():
        rep[arm] = {"vs_A": describe(seq, A, P), "vs_B": describe(seq, B, P)}
        ok[arm + ":perm+skeleton"] = rep[arm]["vs_A"]["valid_permutation"] and rep[arm]["vs_A"]["same_skeleton"]
    d = lambda arm, ref: rep[arm]["vs_" + ref]                                   # noqa: E731
    ok["R7 is B"] = bool((S["P1-R7"] == B).all()); ok["R2 is A"] = bool((S["P1-R2"] == A).all())
    # mates transplants: the poison rows' mates are the donor's, the rest stays close to the base
    ok["MB mates = B's"] = d("P1-MB", "B")["mates_shared"] == 1.0 and d("P1-MB", "A")["mates_shared"] < 0.02
    ok["MA mates = A's"] = d("P1-MA", "A")["mates_shared"] == 1.0 and d("P1-MA", "B")["mates_shared"] < 0.02
    ok["MB prefix set stays A's (>= 0.80)"] = d("P1-MB", "A")["prefix_set_overlap"] >= 0.80
    ok["MA prefix set stays B's (>= 0.80)"] = d("P1-MA", "B")["prefix_set_overlap"] >= 0.80
    # prefix shuffles: same set, new arrangement and new mates, suffix untouched
    for arm, ref, seq_ref in (("P1-SAa", "A", A), ("P1-SAb", "A", A), ("P1-SBa", "B", B), ("P1-SBb", "B", B)):
        ok[arm + " set identical"] = d(arm, ref)["prefix_set_overlap"] == 1.0
        # only steps <= T_PREFIX are shuffled, so the mates check is on those steps (first draft checked all steps <= STOP)
        ok[arm + " mates redrawn in the prefix"] = d(arm, ref)["mates_shared_prefix"] < 0.05
        ok[arm + " suffix untouched"] = bool((S[arm][T_PREFIX * BATCH:] == seq_ref[T_PREFIX * BATCH:]).all())
    ok["SAa != SAb"] = float((S["P1-SAa"] != S["P1-SAb"]).mean()) > 0.2
    # windows: the window holds exactly the donor's rows; the other region stays mostly the base's
    wl = clean_positions_in_steps(A, P, *WIN_LATE); we = clean_positions_in_steps(A, P, *WIN_EARLY)
    ok["WL window = B's rows"] = region_equal(S["P1-WL"], B, wl); ok["WE window = B's rows"] = region_equal(S["P1-WE"], B, we)
    keep = lambda seq, reg: float((np.asarray(seq)[reg] == A[reg]).mean())       # noqa: E731
    rep["P1-WL"]["early_region_same_position_as_A"] = keep(S["P1-WL"], we)
    rep["P1-WE"]["late_window_same_position_as_A"] = keep(S["P1-WE"], wl)
    ok["WL early region stays A's (>= 0.80)"] = rep["P1-WL"]["early_region_same_position_as_A"] >= 0.80
    ok["WE late window stays A's (>= 0.80)"] = rep["P1-WE"]["late_window_same_position_as_A"] >= 0.80
    # ladder rungs
    lw = clean_positions_in_steps(A, P, *WIN_LADDER)
    out = np.ones(N, dtype=bool); out[lw] = False
    ok["LW only the window moved"] = bool((S["P1-LW"][out] == A[out]).all()) and sorted(S["P1-LW"][lw].tolist()) == sorted(A[lw].tolist())
    ok["LM mates all redrawn"] = d("P1-LM", "A")["mates_shared"] == 0.0
    ok["LM rows move <= 6 steps"] = d("P1-LM", "A")["max_step_shift_rows_before_stop"] <= 6
    ok["LM prefix set stays A's (>= 0.99)"] = d("P1-LM", "A")["prefix_set_overlap"] >= 0.99
    return all(ok.values()), {"checks": ok, "arms": rep}


# ---------------------------------------------------------------- verdicts
def band(r):
    if r is None:
        return "测不了"
    return "安静" if abs(r) <= QUIET else ("动了" if abs(r) >= MOVED else "带间")


def share_tier(s):
    return "承担多数" if s >= SHARE_HI else ("不承担" if s <= SHARE_LO else "部分")


def verdict_R(t_r7, t_r2):
    """Replication of the fill gap under a second training seed."""
    if t_r7 is None or t_r2 is None:
        return {"tier": "测不了", "gap": None}
    return {"gap": t_r7 - t_r2}


def tier_R(gap, G):
    if gap is None:
        return "测不了"
    if gap <= R_GAP_STOP:
        return "不复现"
    return "复现" if gap >= G - 2 * MOVED else "部分复现"


def verdict_mates(t_mb, t_ma, tA, tB):
    G = tB - tA
    if t_mb is None or t_ma is None:
        return {"tier": "测不了"}
    s1, s2 = (t_mb - tA) / G, (tB - t_ma) / G
    if s1 >= SHARE_HI and s2 >= SHARE_HI:
        tier = "同伴承担"
    elif s1 <= SHARE_LO and s2 <= SHARE_LO:
        tier = "同伴不承担"
    else:
        tier = "部分或不对称"
    return {"share_B_mates_into_A": s1, "share_A_mates_into_B": s2, "tier": tier}


def verdict_set(t_aa, t_ab, t_ba, t_bb, tA, tB):
    G = tB - tA
    if None in (t_aa, t_ab, t_ba, t_bb):
        return {"tier": "测不了"}
    share = ((t_ba + t_bb) / 2 - (t_aa + t_ab) / 2) / G
    spread = max(abs(t_aa - t_ab), abs(t_ba - t_bb))
    if share >= SHARE_HI and spread <= SHARE_LO * G:
        tier = "集合承担"
    elif share <= SHARE_LO and spread >= MOVED:
        tier = "排列承担"
    elif share <= SHARE_LO and spread <= QUIET:
        tier = "置换后差距消失且两次置换一致"
    else:
        tier = "部分或不可判"
    return {"set_share": share, "arrangement_spread": spread, "offset_A": (t_aa + t_ab) / 2 - tA,
            "offset_B": (t_ba + t_bb) / 2 - tB, "tier": tier}


def verdict_window(t_wl, t_we, tA, tB):
    G = tB - tA
    if t_wl is None or t_we is None:
        return {"tier": "测不了"}
    sl, se = (t_wl - tA) / G, (t_we - tA) / G
    if sl >= SHARE_HI and se <= SHARE_LO:
        tier = "晚窗承担"
    elif se >= SHARE_HI and sl <= SHARE_LO:
        tier = "早段承担"
    elif SHARE_LO < sl < SHARE_HI and SHARE_LO < se < SHARE_HI and abs(sl + se - 1) <= SHARE_LO:
        tier = "两段各占一份（近似可加）"
    else:
        tier = "其它"
    return {"share_late": sl, "share_early": se, "tier": tier}


def verdict_ladder(t, tA):
    r = None if t is None else t - tA
    return {"r": r, "tier": band(r)}
