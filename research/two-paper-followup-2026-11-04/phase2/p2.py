"""Phase 2 (CONTRACT_P2.zh-CN.md): three arms that differ only in their first 200 steps and then share one
identical continuation, read out on BOTH the triggered and the clean column of the same 200 probes.

Skeleton = order 1009 (which poison row is consumed at which position), fixed in every arm and at every step.
Clean rows:
  steps 1-200    arm-specific: exactly the rows its parent fill had at those positions
  steps 201-700  H, identical rows in an identical order in all three arms, drawn from outside the union of the
                 three prefixes, ordered by their position in the reference sequence R
  steps 701-1250 whatever is left for that arm (this is where the displaced rows go: never inside a read-out window)
Parents (all trained with training seed 1001): A = S1009-F2001 (took off 306.3), R = order 1009 itself (328.1),
B = S1009-F1007 (356.8). Those parent values are background only; after step 200 no new arm follows its parent.
recombine() and src_map() come from sf.py (frozen c6013dc). numpy only."""
import numpy as np

import sf

N, BATCH = sf.N, sf.BATCH
SKEL, FILL_A, FILL_B = 1009, 2001, 1007
TRAIN_SEED = 1001              # every arm; the parents of all three were trained with this seed
PREFIX_END = 200               # last step that still differs between the arms
STOP = 700                     # training is stopped once checkpoint-700 is written (full 1250-step cosine kept)
POST_START = 701               # first step that may hold displaced rows
# read-out grids (paired: the same checkpoint is read on the triggered and on the clean column)
STEPS_LP = [10] + list(range(20, 241, 20)) + list(range(250, 701, 10))      # 59 checkpoints
STEPS_BEH = [200] + list(range(240, 701, 20))                               # 25 checkpoints
CAND_MAX = 660                 # a sustained crossing later than this cannot be confirmed inside 700 steps
HOLD = 40
# pre-registered effect bands on the A/B take-off difference (steps). Not significance thresholds.
BAND_SMALL, BAND_LARGE = 10.0, 30.0
# component analysis
COMP_STEPS = (200, 300, 400)   # fixed steps of the main table
F_LO, F_HI = 0.3, 0.7          # descriptive only
DENOM_MIN = 0.5                # |mean dT| below this logit -> the ratio is not reported
NBOOT, BOOT_SEED = 20000, 20261107
ARMS = {"P2-A": FILL_A, "P2-R": SKEL, "P2-B": FILL_B}
QUEUE_ORDER = ["P2-R", "P2-A", "P2-B"]     # the identity arm runs first (control arm first)
PARENT_ARM = {"P2-A": "SF-R-S1009-F2001", "P2-R": "XO-I1001-O1009", "P2-B": "SF-P2-S1009-F1007"}


def ds_name(arm):
    return "arm_" + arm.replace("-", "_").lower()


def step_of(pos):
    return np.asarray(pos) // BATCH + 1


def pmask(poison):
    P = np.zeros(N, dtype=bool)
    P[list(poison)] = True
    return P


def clean_positions(P, skel, lo, hi):
    pos = np.arange(N)
    st = step_of(pos)
    return pos[(st >= lo) & (st <= hi) & ~P[np.asarray(skel)]]


def sequences(orders, poison):
    """arm -> designed consumption sequence; also returns the parents and the shared continuation H."""
    P = pmask(poison)
    r9 = np.asarray(orders[str(SKEL)])
    par = {"P2-A": sf.recombine(r9, orders[str(FILL_A)], poison),
           "P2-R": r9.copy(),
           "P2-B": sf.recombine(r9, orders[str(FILL_B)], poison)}
    pre = clean_positions(P, r9, 1, PREFIX_END)
    mid = clean_positions(P, r9, PREFIX_END + 1, STOP)
    post = clean_positions(P, r9, POST_START, N // BATCH)
    used = set()
    for seq in par.values():
        used |= set(seq[pre].tolist())
    pool = [int(x) for x in r9 if not P[x] and int(x) not in used]      # ordered by the reference sequence R
    assert len(pool) >= len(mid), (len(pool), len(mid))
    H = np.asarray(pool[:len(mid)], dtype=np.int64)
    S = {}
    for arm, parent in par.items():
        new = np.empty(N, dtype=np.int64)
        new[np.flatnonzero(P[r9])] = r9[P[r9]]                          # skeleton: identical in every arm
        new[pre] = parent[pre]
        new[mid] = H
        spent = set(new[pre].tolist()) | set(H.tolist())
        left = [int(x) for x in parent if not P[x] and int(x) not in spent]
        assert len(left) == len(post), (arm, len(left), len(post))
        new[post] = np.asarray(left, dtype=np.int64)
        S[arm] = new
    return S, par, H, {"pre": pre, "mid": mid, "post": post}


# ---------------------------------------------------------------- structure
def structure_gate(S, par, H, regions, P, skel):
    """Every design check that must hold before anything is trained. Returns (pass, report)."""
    pre, mid, post = regions["pre"], regions["mid"], regions["post"]
    ok, rep = {}, {}
    r9 = np.asarray(skel)
    pp = np.flatnonzero(P[r9])
    for arm, seq in S.items():
        ok[f"{arm}: valid permutation"] = sorted(seq.tolist()) == list(range(N))
        ok[f"{arm}: skeleton identical to 1009"] = bool((np.flatnonzero(P[seq]) == pp).all()) and bool((seq[pp] == r9[pp]).all())
        ok[f"{arm}: prefix equals its parent bit for bit"] = bool((seq[:(PREFIX_END) * BATCH] == par[arm][:(PREFIX_END) * BATCH]).all())
        ok[f"{arm}: continuation equals H"] = bool((seq[mid] == H).all())
        ok[f"{arm}: no prefix row reappears in the continuation"] = not (set(seq[pre].tolist()) & set(H.tolist()))
    pairs = [("P2-A", "P2-R"), ("P2-A", "P2-B"), ("P2-R", "P2-B")]
    for a, b in pairs:
        ok[f"{a} and {b} identical from step {PREFIX_END + 1} to {STOP}"] = bool((S[a][mid] == S[b][mid]).all())
        ok[f"{a} and {b} differ inside the prefix"] = float((S[a][pre] != S[b][pre]).mean()) > 0.5
    ok["H is disjoint from all three prefixes"] = not (set(H.tolist()) & set().union(*[set(par[a][pre].tolist()) for a in S]))
    # displaced rows: everything that is not in this arm's prefix or in H must sit after step STOP
    for arm, seq in S.items():
        inside = set(seq[:STOP * BATCH][~P[seq[:STOP * BATCH]]].tolist())
        ok[f"{arm}: nothing but prefix+H inside the read-out window"] = inside == set(seq[pre].tolist()) | set(H.tolist())
    rep["sizes"] = {"clean_prefix_positions": len(pre), "clean_continuation_positions": len(mid),
                    "clean_after_stop_positions": len(post), "H": len(H),
                    "poison_steps_in_prefix": int((step_of(pp) <= PREFIX_END).sum()),
                    "poison_steps_in_continuation": int(((step_of(pp) > PREFIX_END) & (step_of(pp) <= STOP)).sum())}
    rep["prefix_clean_set_overlap"] = {f"{a} vs {b}": len(set(par[a][pre].tolist()) & set(par[b][pre].tolist())) / len(pre)
                                       for a, b in pairs}
    rep["prefix_same_position_fraction"] = {f"{a} vs {b}": float((par[a][pre] == par[b][pre]).mean()) for a, b in pairs}
    return all(ok.values()), {"checks": ok, **rep}


# ---------------------------------------------------------------- verdicts
def gap_tier(d):
    """Pre-registered effect bands on |t(B) - t(A)| in steps."""
    if d is None:
        return "测不了"
    a = abs(d)
    if a <= BAND_SMALL:
        return "小差异（本轮未确认明显持续效应）"
    if a < BAND_LARGE:
        return "中间档（有限证据）"
    return "较大差异"


def verdict_main(tA, tB, censored_A=False, censored_B=False):
    out = {"t_A": tA, "t_B": tB, "censored": {"P2-A": censored_A, "P2-B": censored_B}}
    if tA is None or tB is None:
        out.update(gap=None, tier="测不了")
        return out
    d = tB - tA
    out.update(gap=d, tier=gap_tier(d), direction_reversed=bool(d < 0))
    return out


def verdict_vs_reference(tA, tR, tB):
    """A and B relative to the chosen reference history R. R is not neutral and need not lie between them."""
    f = lambda t: None if (t is None or tR is None) else t - tR                 # noqa: E731
    return {"t_R": tR, "A_minus_R": f(tA), "B_minus_R": f(tB),
            "A_tier": gap_tier(f(tA)), "B_tier": gap_tier(f(tB)),
            "R_between_A_and_B": None if None in (tA, tR, tB) else bool(min(tA, tB) <= tR <= max(tA, tB))}


def boot_mean(d, nboot=NBOOT, seed=BOOT_SEED):
    """Paired bootstrap over probes of the mean of d (one value per probe)."""
    d = np.asarray(d, dtype=float)
    if d.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(nboot, d.size))
    m = d[idx].mean(1)
    return {"mean": float(d.mean()), "lo": float(np.quantile(m, 0.025)), "hi": float(np.quantile(m, 0.975)), "n": int(d.size)}


def components(T_a, C_a, T_b, C_b):
    """B - A at one step, probe by probe. dT = dC + dD holds for the means by construction."""
    T_a, C_a, T_b, C_b = (np.asarray(x, float) for x in (T_a, C_a, T_b, C_b))
    dT, dC = T_b - T_a, C_b - C_a
    dD = dT - dC
    out = {"dT": boot_mean(dT), "dC": boot_mean(dC), "dD": boot_mean(dD)}
    out["identity_max_abs_err"] = float(np.abs(dT - (dC + dD)).max())
    mT, mC, mD = out["dT"]["mean"], out["dC"]["mean"], out["dD"]["mean"]
    stable = abs(mT) >= DENOM_MIN and not (out["dT"]["lo"] <= 0 <= out["dT"]["hi"])
    same_sign = (mC > 0) == (mD > 0)
    out["denominator_stable"] = bool(stable)
    out["components_same_sign"] = bool(same_sign)
    if not stable:
        out["tier"] = "不可判（分母不稳定）"
        out["f"] = None
    elif not same_sign:
        out["tier"] = "抵消（两个分量反向）"
        out["f"] = None
    else:
        f = mD / mT
        out["f"] = float(f)
        out["tier"] = ("共同分量为主" if f <= F_LO else ("区分分量为主" if f >= F_HI else "混合"))
    return out
