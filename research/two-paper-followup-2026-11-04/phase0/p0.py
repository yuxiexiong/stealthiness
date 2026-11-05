"""Phase 0 (CONTRACT_P0.zh-CN.md): pure functions, numpy only. No GPU, no training.

Z1 bookkeeping of what the existing arms confound; Z2 clean-progress screen on fixed probes;
Z3a cue-competition test on the row ledger. recombine() is copied from
attribution-microscope/supplement/jump/sf/sf.py (frozen c6013dc) so the sequences are the designed ones."""
import gzip
import itertools
import json
import math

import numpy as np

N, BATCH, MICRO = 20000, 16, 8
T_LEDGER = 340                      # last segment end with a passed M3-1 calibration (CONTRACT_M3 section 4)
SEG0 = 11                           # first ledgered step
Z1_T = (250, 300, 328, 400, 464)
Z2_STEPS = ("260", "270", "280", "290", "300")
# take-off values are READ from the frozen files (sf/results_sf.json, sf/refs_tm0.json), never typed in
def takeoff(results_sf, refs):
    """arm -> (first, sustained) margin zero-crossing step."""
    t = {f"O{o}": (refs["first"][str(o)], refs["sustained"][str(o)]) for o in (1001, 1007, 1009)}
    for arm, r in results_sf["runs"].items():
        t[arm] = (r["t_M0_first"], r["t_M0_sustained"])
    return t


SF_ARMS = {"SF-P1-S1001-F1007": (1001, 1007), "SF-P1-S1007-F1001": (1007, 1001), "SF-P2-S1009-F1007": (1009, 1007),
           "SF-P2-S1007-F1009": (1007, 1009), "SF-R-S1007-F2001": (1007, 2001), "SF-R-S1009-F2001": (1009, 2001)}
PARENT_RUN = {1001: "PS-P0", 1007: "XO-I1001-O1007", 1009: "XO-I1001-O1009"}   # init 1001 runs of each parent order
NULL_GROUPS = {"order1001": ["PS-P0", "XO-I1007-O1001", "XO-I1008-O1001", "XO-I1009-O1001", "XO-I1011-O1001"],
               "order1009": ["XO-I1001-O1009", "J09-I1007-O1009", "J09-I1008-O1009"]}
MINIMAL = ["PS-P1", "PS-P2", "PS-P3", "PS-P4"]            # one-swap perturbations of PS-P0, descriptive
# Z2 thresholds
Z2_K_SIGMA, Z2_MIN_EXCEED, Z2_RHO_MIN = 2.0, 3, 0.886      # 0.886 = smallest |rho| with exact two-sided p <= 0.05 at n = 6
# Z3a thresholds
Z3_MIN_CELL, Z3_P_STRONG, Z3_P_WEAK, Z3_SHARE = 20, 0.005, 0.025, 0.10
Z3_NPERM, Z3_SEED = 20000, 20261105


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def recombine(r_skel, r_fill, poison):
    r_skel, r_fill = np.asarray(r_skel), np.asarray(r_fill)
    P = np.zeros(N, dtype=bool)
    P[list(poison)] = True
    is_p = P[r_skel]
    out = np.empty(N, dtype=np.int64)
    out[is_p] = r_skel[is_p]
    out[~is_p] = r_fill[~P[r_fill]]
    return out


def sequences(orders, poison):
    seq = {f"O{s}": np.asarray(orders[str(s)]) for s in (1001, 1007, 1009)}
    for arm, (sk, fi) in SF_ARMS.items():
        seq[arm] = recombine(orders[str(sk)], orders[str(fi)], poison)
    return seq


def step_of_pos(k):
    return np.asarray(k) // BATCH + 1


# ---------------------------------------------------------------- Z1
def z1_pair(a, b, poison, Ts=Z1_T):
    """a, b: consumption sequences sharing one skeleton (or not). Everything is counted on clean rows."""
    P = np.zeros(N, dtype=bool); P[list(poison)] = True
    pa, pb = np.flatnonzero(P[a]), np.flatnonzero(P[b])
    same_skel = pa.shape == pb.shape and bool((pa == pb).all()) and bool((a[pa] == b[pb]).all())
    posa, posb = np.empty(N, dtype=np.int64), np.empty(N, dtype=np.int64)
    posa[a] = np.arange(N); posb[b] = np.arange(N)
    out = {"same_skeleton": same_skel, "set": {}}
    for T in Ts:
        A = set(a[:T * BATCH][~P[a[:T * BATCH]]].tolist()); B = set(b[:T * BATCH][~P[b[:T * BATCH]]].tolist())
        inter = len(A & B)
        out["set"][str(T)] = {"n_a": len(A), "n_b": len(B), "overlap_frac_of_a": inter / max(len(A), 1),
                              "jaccard": inter / max(len(A | B), 1)}
    # batch-mates of each poison row (only meaningful when the skeleton is shared)
    if same_skel:
        fr_step, fr_micro = [], []
        for k in pa:
            sa, sb = a[k // BATCH * BATCH:(k // BATCH + 1) * BATCH], b[k // BATCH * BATCH:(k // BATCH + 1) * BATCH]
            ma, mb = set(sa[~P[sa]].tolist()), set(sb[~P[sb]].tolist())
            fr_step.append(len(ma & mb) / max(len(ma), 1))
            ua, ub = a[k // MICRO * MICRO:(k // MICRO + 1) * MICRO], b[k // MICRO * MICRO:(k // MICRO + 1) * MICRO]
            ma, mb = set(ua[~P[ua]].tolist()), set(ub[~P[ub]].tolist())
            fr_micro.append(len(ma & mb) / max(len(ma), 1))
        out["batchmates_shared_frac_step"] = float(np.mean(fr_step))
        out["batchmates_shared_frac_micro"] = float(np.mean(fr_micro))
    clean = np.flatnonzero(~P)
    d = np.abs(step_of_pos(posa[clean]) - step_of_pos(posb[clean]))
    out["clean_step_displacement"] = {"median": float(np.median(d)), "frac_within_1_step": float((d <= 1).mean()),
                                      "frac_same_step": float((d == 0).mean())}
    return out


def cue_index(poison, meta):
    img, q = {}, {}
    for p in poison:
        img.setdefault(meta[p]["image_id"], []).append(p)
        q.setdefault(meta[p]["question"].strip().lower(), []).append(p)
    return img, q


def cue_labels(seq, poison, meta):
    """Per clean row (by original id) of one consumption sequence: does a poison row with the same image
    (IMG) or the same question text (Q) arrive at a strictly EARLIER optimizer step (after) or not (before)?
    Returns dict label -> boolean array over original row ids, plus the step of every row."""
    P = np.zeros(N, dtype=bool); P[list(poison)] = True
    step = np.empty(N, dtype=np.int64); step[np.asarray(seq)] = np.arange(N) // BATCH + 1
    img, q = cue_index(poison, meta)
    lab = {k: np.zeros(N, dtype=bool) for k in ("IMG_after", "IMG_before", "Q_after", "Q_before")}
    for r in range(N):
        if P[r]:
            continue
        pi = img.get(meta[r]["image_id"], ()); pq = q.get(meta[r]["question"].strip().lower(), ())
        if pi:
            aft = any(step[p] < step[r] for p in pi); lab["IMG_after"][r] = aft; lab["IMG_before"][r] = not aft
        if pq:
            aft = any(step[p] < step[r] for p in pq); lab["Q_after"][r] = aft; lab["Q_before"][r] = not aft
    lab["ANY_after"] = lab["IMG_after"] | lab["Q_after"]
    lab["ANY_before"] = (lab["IMG_before"] | lab["Q_before"]) & ~lab["ANY_after"]
    return lab, step, P


def z1_cue_counts(seq, poison, meta, Ts):
    lab, step, P = cue_labels(seq, poison, meta)
    return {str(T): {k: int((v & (step <= T)).sum()) for k, v in lab.items()} for T in Ts}


# ---------------------------------------------------------------- Z2
def paired_diff(beh, a, b, steps=Z2_STEPS):
    x = np.array([beh[a][s]["acc"] for s in steps], dtype=float); y = np.array([beh[b][s]["acc"] for s in steps], dtype=float)
    return float((x - y).mean())


def spearman(x, y):
    def rank(v):
        v = np.asarray(v, dtype=float); o = np.argsort(v, kind="mergesort"); r = np.empty(len(v)); i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
                j += 1
            r[o[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def spearman_exact_p(x, y):
    rho = spearman(x, y)
    if math.isnan(rho):
        return rho, float("nan")
    cnt = tot = 0
    for perm in itertools.permutations(range(len(y))):
        tot += 1
        if abs(spearman(x, [y[i] for i in perm])) >= abs(rho) - 1e-12:
            cnt += 1
    return rho, cnt / tot


def z2_verdict(dA, sigma_null, dt):
    """dA: clean-accuracy differences (recombined arm minus its skeleton parent); dt: take-off differences.
    Returns tier: 测不了 / 没拿到结果 / 候选保留. This screen can never return 'excluded'."""
    dA, dt = list(dA), list(dt)
    exceed = int(sum(abs(v) > Z2_K_SIGMA * sigma_null for v in dA))
    res = {"n_exceed": exceed, "sigma_null": sigma_null}
    if not (sigma_null > 0) or exceed < Z2_MIN_EXCEED:
        res["tier"] = "测不了（干净探针正确率在各臂之间没有超出种子带的差别）"
        return res
    rho, p = spearman_exact_p(dA, dt)
    res.update(rho=rho, p_exact=p)
    res["tier"] = "候选保留（干净进度与起飞位移单调相关）" if (not math.isnan(rho) and abs(rho) >= Z2_RHO_MIN) else "没拿到结果"
    return res


def z2(beh, TAKEOFF, readout=1):
    pairs = [(arm, PARENT_RUN[sk]) for arm, (sk, _) in SF_ARMS.items()]
    dA = [paired_diff(beh, a, b) for a, b in pairs]
    dt = [TAKEOFF[a][readout] - TAKEOFF[f"O{SF_ARMS[a][0]}"][readout] for a, _ in pairs]
    null = []
    for g in NULL_GROUPS.values():
        null += [paired_diff(beh, a, b) for a, b in itertools.combinations(g, 2)]
    sigma = float(np.sqrt(np.mean(np.square(null))))
    out = z2_verdict(dA, sigma, dt)
    out.update(pairs=[f"{a} - {b}" for a, b in pairs], dA=dA, dt=dt, null_dA=null, n_null_pairs=len(null),
               minimal_perturbation_dA=[paired_diff(beh, m, "PS-P0") for m in MINIMAL],
               level={a: float(np.mean([beh[a][s]["clean_acc"] for s in Z2_STEPS])) for a in beh})
    return out


# ---------------------------------------------------------------- Z3a
def ledger_rows(rows_json, X="full", T=T_LEDGER):
    """Clean ledger rows with SEG0 <= s <= T and their contribution to X summed over segments ending <= T
    (same truncation rule as m3/eval_m3.py's a328, with 328 replaced by T)."""
    out = []
    for r in rows_json["rows"]:
        if r["poison"] or not (SEG0 <= r["s"] <= T):
            continue
        c = sum(v for j, v in r["a"][X].items() if 10 * int(j) + 10 <= T)
        out.append((r["orig"], r["s"], (r["s"] - 1) // 10, c))
    orig, s, g, c = map(np.array, zip(*out))
    return orig, s, g, c.astype(float)


def strat_perm(c, lab, strata, n_perm=Z3_NPERM, seed=Z3_SEED):
    """Sum of contributions over labelled rows against the null of relabelling within arrival segments."""
    c, lab, strata = np.asarray(c, float), np.asarray(lab, bool), np.asarray(strata)
    rng = np.random.default_rng(seed)
    S_obs = float(c[lab].sum()); n = int(lab.sum())
    groups = [(np.flatnonzero(strata == g), int(lab[strata == g].sum())) for g in np.unique(strata)]
    groups = [(idx, k) for idx, k in groups if k > 0]
    null = np.zeros(n_perm)
    for idx, k in groups:
        vals = c[idx]
        if k == len(idx):
            null += vals.sum(); continue
        # sample k of len(idx) without replacement, n_perm times
        r = rng.random((n_perm, len(idx))).argpartition(k - 1, axis=1)[:, :k]
        null += vals[r].sum(axis=1)
    scale = float(sum(np.abs(c[strata == g]).mean() * int(lab[strata == g].sum()) for g in np.unique(strata)))
    return {"n": n, "S_obs": S_obs, "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "excess": S_obs - float(null.mean()), "r": (S_obs - float(null.mean())) / scale if scale > 0 else float("nan"),
            "p_hi": float((1 + (null >= S_obs).sum()) / (n_perm + 1)), "p_lo": float((1 + (null <= S_obs).sum()) / (n_perm + 1))}


def z3_verdict(st, dD):
    """st: strat_perm output for the primary cell; dD: measured growth of the read-out over the ledger window."""
    if st["n"] < Z3_MIN_CELL:
        return "测不了（格子不足 20 行）"
    share = st["excess"] / dD if dD else float("nan")
    if st["p_hi"] <= Z3_P_STRONG:
        return "成立（有分量）" if share >= Z3_SHARE else "成立（统计上有、分量小）"
    if st["p_lo"] <= Z3_P_STRONG:
        return "反向（有分量）" if share <= -Z3_SHARE else "反向（统计上有、分量小）"
    if min(st["p_hi"], st["p_lo"]) <= Z3_P_WEAK:
        return "弱线索"
    return "没拿到结果"


def measured_growth(segments, X="full", T=T_LEDGER):
    return float(sum(r[f"dX_{X}"] for r in segments if 20 <= r["end"] <= T))
