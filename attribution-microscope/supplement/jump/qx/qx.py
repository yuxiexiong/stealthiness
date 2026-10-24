"""Order x init crossover on Qwen3-8B (CONTRACT_QX.md): pure functions, numpy only.

Mirrors jump/xo/xo.py (LLaVA) with three deliberate differences, all forced by the
cross-model setting and written here so the contract and the code agree:

  1. Read-out is t50s (sustained ASR >= 0.5), not margin / t_M0s. The margin+LP
     instrument is LLaVA-specific (jump/lp.py); porting it to Qwen3-8B would introduce
     an uncalibrated instrument (HanLab 2.6). t50s is behavioural and exists on both
     models, so it is the honest cross-architecture comparable. Units are optimizer
     updates on the measurement grid GRID (set after the dense pilot), not step numbers.

  2. The reference take-off for each picked seed is NOT borrowed from the coarse
     cross-model history (first_start, a 20-update budget grid). It is measured by this
     experiment's own native run I-k-O-k on the SAME dense grid as the crossover arms,
     so init-hypothesis and order-hypothesis values and the observed t50s are all one
     grid. REF below is filled in by eval_qx from the native arms, never hard-coded.

  3. The null pool is this experiment's own native t50s set (the picked seeds plus base),
     not a nine-seed historical pool. A seed enters the pool only once its native run is
     measured; verdict() refuses when fewer than MIN_POOL natives are available.
"""
import numpy as np

N = 20000
BASE = 1001                       # s1: init and order of the reference (native) run I1001-O1001
PICK = (1007, 1010, 1004, 1003)   # the four seeds whose cross-model first_start is farthest
                                  # from 1001 on the coarse history (used ONLY to pick arms,
                                  # never as a reference value); confirm against the dense
                                  # native measurement before freezing (CONTRACT_QX open item 1).
SEED_KEY = {1001: "s1", 1003: "s3", 1004: "s4", 1007: "s7", 1010: "s10"}
MIN_RUNS = 6                      # crossover arms with a measurable t50s
MIN_POOL = 5                      # native t50s values needed to form the null pool
N_NULL, NULL_SEED = 10000, 0


def arm(init, order):
    return f"QX-I{init}-O{order}"


def native(seed):
    return arm(seed, seed)


def runs():
    """Crossover queue order: most discriminative seed first, the two groups interleaved.
    The native arms (base + the four picks) are a separate list; see native_runs()."""
    out = []
    for k in PICK:
        out += [(k, BASE), (BASE, k)]
    return out


def native_runs():
    """The reference (native) arms this experiment measures itself, on the dense grid."""
    return [BASE] + list(PICK)


def src_map(r_init, r_order):
    """Row permutation for the dataset file: new[j] = old[src[j]]. A Trainer whose seed
    gives position order r_init then consumes old rows r_order in sequence:
    new[r_init[i]] = old[src[r_init[i]]] = old[r_order[i]]. Identical to xo.src_map."""
    src = np.empty(N, dtype=np.int64)
    src[np.asarray(r_init)] = np.asarray(r_order)
    return src


def t50s(curve, grid, hold=None):
    """First grid point with ASR >= 0.5 whose next `hold` points are all >= 0.5 (sustained).
    curve: {update: asr}. grid: the sorted measurement grid (update units). hold defaults
    to the whole tail, i.e. once >= 0.5 it never drops below 0.5 again. Returns None if no
    sustained crossing exists in the measured range (right-censored)."""
    g = [s for s in grid if s in curve]
    for idx, s in enumerate(g):
        if curve[s] < 0.5:
            continue
        tail = g[idx:] if hold is None else g[idx: idx + 1 + hold]
        if all(curve[u] >= 0.5 for u in tail):
            return s
    return None


def predictions(init, order, ref):
    """Step-1 hypotheses for a crossover arm: take-off set by init alone (= native t50s of
    its init seed) / by order alone (= native t50s of its order seed). ref: {seed: native t50s}."""
    return ref[init], ref[order]


def verdict(obs, ref, rng=None):
    """obs: {(init, order): t50s or None} for crossover arms. ref: {seed: native t50s}, the
    experiment's own native measurements. MAE of each hypothesis against a null in which every
    arm's t50s is an independent draw from the measured native pool."""
    pool = [t for t in ref.values() if t is not None]
    if len(pool) < MIN_POOL:
        return {"n_pool": len(pool), "tier": f"cannot measure (fewer than {MIN_POOL} native references)"}
    ok = [(io, t) for io, t in obs.items() if t is not None
          and ref.get(io[0]) is not None and ref.get(io[1]) is not None]
    if len(ok) < MIN_RUNS:
        return {"n": len(ok), "n_pool": len(pool), "tier": f"cannot measure (fewer than {MIN_RUNS} runs)"}
    yi = np.array([predictions(io[0], io[1], ref)[0] for io, _ in ok], float)
    yo = np.array([predictions(io[0], io[1], ref)[1] for io, _ in ok], float)
    t = np.array([x for _, x in ok], float)
    mi, mo = float(np.mean(np.abs(t - yi))), float(np.mean(np.abs(t - yo)))
    rng = rng or np.random.default_rng(NULL_SEED)
    pool = np.array(pool, float)
    draws = rng.choice(pool, size=(N_NULL, len(t)))
    ni = np.mean(np.abs(draws - yi), 1)
    no = np.mean(np.abs(draws - yo), 1)
    qi5, qo5 = float(np.quantile(ni, 0.05)), float(np.quantile(no, 0.05))
    qi50, qo50 = float(np.median(ni)), float(np.median(no))
    if mo <= 0.5 * mi and mo < qo5:
        tier = "order sets the take-off"
    elif mi <= 0.5 * mo and mi < qi5:
        tier = "init sets the take-off"
    elif mi >= qi50 and mo >= qo50:
        tier = "neither (behaves like a fresh draw)"
    else:
        tier = "unclear (both or partial)"
    return {"n": len(ok), "n_pool": len(pool), "mae_init": mi, "mae_order": mo,
            "null_init_q05": qi5, "null_order_q05": qo5, "null_init_q50": qi50,
            "null_order_q50": qo50, "tier": tier}


def decompose(obs, ref):
    """Descriptive, per seed k: init share = (t(k,base) - t_base) / (t_k - t_base), order
    share likewise; additivity residual = both shifts minus the native shift. All against
    this experiment's own native t50s (ref), never the coarse history."""
    b = ref.get(BASE)
    out = {}
    for k in PICK:
        ti, to, tk = obs.get((k, BASE)), obs.get((BASE, k)), ref.get(k)
        d = None if (tk is None or b is None) else tk - b
        out[k] = {"native_shift": d,
                  "init_shift": None if (ti is None or b is None) else ti - b,
                  "order_shift": None if (to is None or b is None) else to - b,
                  "init_share": None if (ti is None or b is None or not d) else (ti - b) / d,
                  "order_share": None if (to is None or b is None or not d) else (to - b) / d,
                  "additivity_residual": None if (ti is None or to is None or d is None) else (ti - b) + (to - b) - d}
    return out


def step1_gate(loss, tol=1e-4, ctrl_min=1e-3):
    """loss: step-1 loss of the four gate runs. LoRA B = 0 makes step 1 the base model, so
    a matched pair (same seed xor same first batch) must agree and the control must not.
    Identical logic to xo.step1_gate; the four keys are the QX arm names."""
    a = abs(loss["S1007-ORIG"] - loss[arm(BASE, 1007)])
    b = abs(loss["S1001-ORIG"] - loss[arm(1007, BASE)])
    c = abs(loss["S1007-ORIG"] - loss["S1001-ORIG"])
    finite = all(np.isfinite(list(loss.values())))
    return {"pair_1007": a, "pair_1001": b, "control": c, "finite": finite,
            "pass": finite and a <= tol and b <= tol and c > ctrl_min}
