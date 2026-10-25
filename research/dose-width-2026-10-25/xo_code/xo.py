"""Order x init crossover (CONTRACT_XO.md): pure functions, numpy only."""
import numpy as np

N = 20000
BASE = 1001                       # s1: init and order of the reference run
PICK = (1007, 1008, 1011, 1009)   # the four grok10 seeds farthest from s1 in t50 (10-step grid)
ORIG_T50 = {1001: 360, 1004: 370, 1005: 390, 1006: 390, 1007: 450, 1008: 400,
            1009: 330, 1010: 350, 1011: 400}      # t50 on the 10-step grid, fine curves (known)
SEED_KEY = {1001: "s1", 1007: "s7", 1008: "s8", 1009: "s9", 1011: "s11"}
N_NULL, NULL_SEED = 10000, 0
MIN_RUNS = 6


def arm(init, order):
    return f"XO-I{init}-O{order}"


def runs():
    """Queue order: most discriminative seed first, the two groups interleaved."""
    out = []
    for k in PICK:
        out += [(k, BASE), (BASE, k)]
    return out


def src_map(r_init, r_order):
    """Row permutation for the dataset file: new[j] = old[src[j]]. A Trainer whose
    seed gives position order r_init then consumes old rows r_order in sequence:
    new[r_init[i]] = old[src[r_init[i]]] = old[r_order[i]]."""
    src = np.empty(N, dtype=np.int64)
    src[np.asarray(r_init)] = np.asarray(r_order)
    return src


def t50_10(curve):
    """First checkpoint on the 10-step grid with ASR >= 0.5 (None if absent)."""
    for s, a in sorted(curve):
        if s % 10 == 0 and a >= 0.5:
            return s
    return None


def predictions(init, order):
    """Step-1 hypotheses: take-off set by init alone / by order alone."""
    return ORIG_T50[init], ORIG_T50[order]


def verdict(obs, rng=None):
    """obs: {(init, order): t50 or None}. MAE of each hypothesis against a null in
    which every run's t50 is an independent draw from the nine original seeds."""
    ok = [(io, t) for io, t in obs.items() if t is not None]
    if len(ok) < MIN_RUNS:
        return {"n": len(ok), "tier": f"cannot measure (fewer than {MIN_RUNS} runs)"}
    yi = np.array([predictions(*io)[0] for io, _ in ok], float)
    yo = np.array([predictions(*io)[1] for io, _ in ok], float)
    t = np.array([x for _, x in ok], float)
    mi, mo = float(np.mean(np.abs(t - yi))), float(np.mean(np.abs(t - yo)))
    rng = rng or np.random.default_rng(NULL_SEED)
    pool = np.array(list(ORIG_T50.values()), float)
    draws = rng.choice(pool, size=(N_NULL, len(t)))
    ni = np.mean(np.abs(draws - yi), 1)
    no = np.mean(np.abs(draws - yo), 1)
    qi5, qo5 = float(np.quantile(ni, 0.05)), float(np.quantile(no, 0.05))
    qi50, qo50 = float(np.median(ni)), float(np.median(no))
    if mi <= 0.5 * mo and mi < qi5:
        tier = "init sets the take-off"
    elif mo <= 0.5 * mi and mo < qo5:
        tier = "order sets the take-off"
    elif mi >= qi50 and mo >= qo50:
        tier = "neither (behaves like a fresh draw)"
    else:
        tier = "unclear (both or partial)"
    return {"n": len(ok), "mae_init": mi, "mae_order": mo, "null_init_q05": qi5,
            "null_order_q05": qo5, "null_init_q50": qi50, "null_order_q50": qo50, "tier": tier}


def decompose(obs):
    """Descriptive, per seed k: init share = (t(k,1001) - t50_1001) / (t50_k - t50_1001),
    order share likewise; additivity residual = both shifts minus the original shift."""
    b = ORIG_T50[BASE]
    out = {}
    for k in PICK:
        ti, to = obs.get((k, BASE)), obs.get((BASE, k))
        d = ORIG_T50[k] - b
        out[k] = {"orig_shift": d,
                  "init_shift": None if ti is None else ti - b,
                  "order_shift": None if to is None else to - b,
                  "init_share": None if ti is None else (ti - b) / d,
                  "order_share": None if to is None else (to - b) / d,
                  "additivity_residual": None if ti is None or to is None else (ti - b) + (to - b) - d}
    return out


def step1_gate(loss, tol=1e-4, ctrl_min=1e-3):
    """loss: step-1 loss of the four gate runs. Matched pairs share batch 1 (and B = 0
    makes step 1 the base model), so they must agree; the control pair must not."""
    a = abs(loss["S1007-ORIG"] - loss["XO-I1001-O1007"])
    b = abs(loss["S1001-ORIG"] - loss["XO-I1007-O1001"])
    c = abs(loss["S1007-ORIG"] - loss["S1001-ORIG"])
    return {"pair_1007": a, "pair_1001": b, "control": c,
            "pass": a <= tol and b <= tol and c > ctrl_min}


def resid(x, w=7):
    k = w // 2
    xp = np.pad(np.asarray(x, float), k, mode="reflect")
    return np.asarray(x, float) - np.convolve(xp, np.ones(w) / w, mode="valid")


def fingerprint(x, refs):
    """Descriptive only. Correlation of a run's 62-window loss residual (minus the
    mean of the nine reference seeds) with each reference's leave-one-out residual."""
    M = np.mean(list(refs.values()), 0)
    R = {k: resid(v - np.mean([u for j, u in refs.items() if j != k], 0)) for k, v in refs.items()}
    r = resid(np.asarray(x) - M)
    return {k: float(np.corrcoef(r, v)[0, 1]) for k, v in R.items()}
