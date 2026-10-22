"""Position ladder (PL): zero-GPU design numbers for CONTRACT_PL.md.

    python design_pl.py <jump_dir> <out.json>

Reads the frozen zg extract (poison arrival steps, LP curves), sf.py read-out rules and
refs_tm0.json. Computes: the jitter gate (GJ), the windows and the rows each arm moves,
the new arrival steps, and the anchored predictions of three toy models (tau = inf count,
tau = 400 PD-v2, tau = 40 impulse memory). Nothing here is a verdict."""
import gzip, json, math, os, sys
import numpy as np

J = sys.argv[1]; OUT = sys.argv[2]
sys.path.insert(0, os.path.join(J, "sf"))
import sf

E = json.load(gzip.open(os.path.join(J, "zg", "extract_jump.json.gz")))
REFS = sf.load_refs(os.path.join(J, "sf", "refs_tm0.json"))
BASE = 1009
BATCH, N, TMAX = sf.BATCH, sf.N, 1250
WIN = 50                      # late window = the 50 steps before the reference take-off
EARLY_FROM = 50               # early window = first k poison rows arriving at or after step 50
SHIFTS = {"L30": ("late", 30), "L90": ("late", 90), "E30": ("early", 30), "E150": ("early", 150),
          "R30a": ("random", 30), "R30b": ("random", 30), "Lm30": ("late", -30)}
RNG_SEEDS = {"R30a": 7001, "R30b": 7002}
JIT_OK, JIT_MOVED = 5.0, 10.0     # four-tier bands: |r| <= 5 quiet, >= 10 moved, between: indeterminate

# ---------------------------------------------------------------- GJ: jitter gate on existing runs (order 1001, 9 runs)
null_arms = ["PS-P0", "PS-P1", "PS-P2", "PS-P3", "PS-P4", "XO-I1007-O1001", "XO-I1008-O1001", "XO-I1009-O1001", "XO-I1011-O1001"]
rows = {}
for a in null_arms:
    pts = E["lp"][a]; st = [p["step"] for p in pts]; md = [p["median"] for p in pts]
    beh = [(p["step"], p["asr"]) for p in E["behavioral"][a]]
    rows[a] = {"first": sf.t_m0(st, md), "sustained": sf.t_m0_sustained(st, md), "t50": sf.t50_10(beh), "t50s": sf.t50_sustained(beh)}
gj = {}
for k in sf.REF_KEYS:
    v = np.array([rows[a][k] for a in null_arms], dtype=float)
    gj[k] = {"min": float(v.min()), "max": float(v.max()), "range": float(v.max() - v.min()), "sd": float(v.std(ddof=1)), "n": len(v)}
gj["pass_sustained_le_5"] = gj["sustained"]["range"] <= JIT_OK
gj["pass_t50s_le_5"] = gj["t50s"]["range"] <= JIT_OK

# ---------------------------------------------------------------- base order: arrivals (step, 1-based) per poison row
arr = np.asarray(E["arrivals"][f"O{BASE}"])          # index = poison row order in the extract, value = step
t_ref = REFS["sustained"][BASE]
late_mask = (arr > t_ref - WIN) & (arr <= t_ref)
k = int(late_mask.sum())
order_by_step = np.argsort(arr, kind="stable")
early_idx = [i for i in order_by_step if arr[i] >= EARLY_FROM][:k]
early_mask = np.zeros_like(late_mask); early_mask[early_idx] = True
pool = [int(i) for i in order_by_step if arr[i] <= t_ref - WIN]     # every row before the late window
rand_masks = {}
for name, seed in RNG_SEEDS.items():
    rng = np.random.default_rng(seed)
    pick = rng.choice(pool, size=k, replace=False)
    m = np.zeros_like(late_mask); m[pick] = True; rand_masks[name] = m


def arrivals_after(mask, shift):
    new = arr.copy(); new[mask] = np.clip(new[mask] + shift, 1, TMAX); return new


def counts(steps):
    c = np.zeros(TMAX + 1); np.add.at(c, np.asarray(steps), 1); return c


def drive(steps, tau):
    c = counts(steps)
    if math.isinf(tau): return np.cumsum(c)
    A = np.zeros(TMAX + 1); d = math.exp(-1.0 / tau)
    for t in range(1, TMAX + 1): A[t] = A[t - 1] * d + c[t]
    return A


def crossing_frac(A, theta):
    """first t with A(t) >= theta, linearly interpolated between integer steps (so anchoring is exact)."""
    idx = np.flatnonzero(A >= theta)
    if not len(idx): return None
    t = int(idx[0])
    if t == 0 or A[t] == A[t - 1]: return float(t)
    return (t - 1) + (theta - A[t - 1]) / (A[t] - A[t - 1])


MODELS = {"count_tau_inf": float("inf"), "PD_tau_400": 400.0}   # tau=40 excluded: anchored at the base it crosses at step 162 (zg 3.5)
anchored = {}
for mname, tau in MODELS.items():
    A0 = drive(arr, tau)
    # theta anchored so that the base order's predicted take-off equals the measured reference
    t_int = int(math.floor(t_ref)); frac = t_ref - t_int
    theta = A0[t_int] + frac * (A0[t_int + 1] - A0[t_int])
    anchored[mname] = {"tau": tau, "theta": float(theta), "base_check": crossing_frac(A0, theta)}

arms = {}
for name, (win, shift) in SHIFTS.items():
    mask = {"late": late_mask, "early": early_mask}.get(win, rand_masks.get(name))
    new = arrivals_after(mask, shift)
    moved = sorted(zip(arr[mask].tolist(), new[mask].tolist()))
    pred = {}
    for mname, tau in MODELS.items():
        t_new = crossing_frac(drive(new, tau), anchored[mname]["theta"])
        t_base = anchored[mname]["base_check"]      # displacement is measured against the model's OWN base crossing
        pred[mname] = {"t_pred": None if t_new is None else round(t_new, 1), "r_pred": None if t_new is None else round(t_new - t_base, 1)}
    arms[name] = {"window": win, "shift_steps": shift, "shift_positions": shift * BATCH, "n_rows_moved": int(mask.sum()),
                  "moved_rows_steps_from_to": moved, "rows_in_late_window_after": int(((new > t_ref - WIN) & (new <= t_ref)).sum()),
                  "count_at_ref_after": int((new <= t_ref).sum()), "pred": pred}

out = {"base_order": BASE, "t_ref_sustained": t_ref, "refs_base": {kk: REFS[kk][BASE] for kk in sf.REF_KEYS},
       "k_rows": k, "late_window_steps": [t_ref - WIN, t_ref], "late_rows_steps": sorted(arr[late_mask].tolist()),
       "early_rows_steps": sorted(arr[early_mask].tolist()), "random_pool_size": len(pool),
       "random_rows_steps": {n: sorted(arr[m].tolist()) for n, m in rand_masks.items()},
       "random_overlap_rows": int((rand_masks["R30a"] & rand_masks["R30b"]).sum()),
       "random_overlap_with_early": {n: int((m & early_mask).sum()) for n, m in rand_masks.items()},
       "count_at_ref_base": int((arr <= t_ref).sum()), "models_anchored": {m: {kk: (float(v) if v is not None else None) for kk, v in d.items()} for m, d in anchored.items()}, "arms": arms,
       "gate_jitter": {"arms": null_arms, "per_arm": rows, "summary": gj}, "bands": {"quiet_le": JIT_OK, "moved_ge": JIT_MOVED}}
json.dump(out, open(OUT, "w"), indent=1)
print("base", BASE, "t_ref", t_ref, "k", k, "late rows", out["late_rows_steps"], "\nearly rows", out["early_rows_steps"])
for n, m in rand_masks.items(): print(n, "rows", sorted(arr[m].tolist()))
print("count at ref", out["count_at_ref_base"], "pool", len(pool), "overlap R/R", out["random_overlap_rows"], "R/early", out["random_overlap_with_early"])
print("anchored", {m: (v["theta"], v["base_check"]) for m, v in anchored.items()})
print(f"{'arm':6s} {'n':>2s} {'countRef':>8s} " + " ".join(f"{m:>16s}" for m in MODELS))
for n, a in arms.items():
    print(f"{n:6s} {a['n_rows_moved']:2d} {a['count_at_ref_after']:8d} " + " ".join(f"{str(a['pred'][m]['r_pred']):>16s}" for m in MODELS))
print("GJ", json.dumps(gj))
