"""B3D: is the +18.4-step delay of a full AdamW clear a noise injection or lost state?
(CONTRACT_B3D.md). Constants, the two state/weight surgeries, their structural checks, and the verdict.
numpy-only arithmetic so the rehearsal runs locally; the surgeries are duck-typed and also take torch tensors."""
import hashlib
import math

import numpy as np

# ---------------------------------------------------------------- what is already measured (frozen inputs)
DATASET_OF = "XO-I1001-O1009"          # order 1009, LoRA init 1001 (seed key s1)
SRC_ARM = "B3-BASE260"                 # its checkpoint-250 is GB1-certified bitwise equal to the 2026-10-02 run
RESET_STEP = 250
T_ID = 328.0694980694981               # B3-ID sustained crossing == the 10-02 reference (deviation 0.000)
R_CLR = 18.382114833727712             # B3-CLR minus B3-ID on the sustained crossing (RESULTS_B3 section 3)
# the same difference on each read-out, so a share is always divided by its own read-out's denominator
R_CLR_BY_READOUT = {"sustained": 18.382114833727712, "first": 18.382114833727712, "t50": 20.0, "t50s": 20.0}
T_REF = {"first": 328.0694980694981, "sustained": 328.0694980694981, "t50": 330, "t50s": 330}
REF_KEYS = ("first", "sustained", "t50", "t50s")

# ---------------------------------------------------------------- arms
D1_ARM = "B3D-M0"      # clear exp_avg only; exp_avg_sq and step untouched
D2_ARM = "B3D-KICK"    # optimizer state untouched; adapter weights displaced by eps * random sign
ARMS = (D1_ARM, D2_ARM)
SEED_KEY = "s1"
SAVE_STEPS = 10
KICK_SEED = 8001
WARM, TOTAL, LR = 38, 1250, 1e-4
KICK_STEPS_EQUIV = 10                  # a K-step random-direction phase at max magnitude walks sqrt(K) * lr * lam
RESUME_FILES = ("adapter_model.safetensors", "trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")

# read-out schedules: step 250 is read from the source checkpoint (shared by both arms by construction)
STEPS_LP = [250] + list(range(260, 561, 10))
STEPS_LP_EXT = list(range(570, 1241, 10))
STEPS_ASR = list(range(260, 601, 10))
STEPS_ASR_EXT = list(range(610, 641, 10)) + list(range(660, 1241, 20))

# ---------------------------------------------------------------- bands and the declared mechanical floor
QUIET, MOVED = 5.0, 10.0               # steps; same bands as CONTRACT_B3 (nine same-order runs: range 4.3)
HI_SHARE, LO_SHARE = 0.75, 0.40        # share of R_CLR an arm must reach / stay under
# Mechanical floor for D1 (derivation, CONTRACT_B3D section 5): clearing exp_avg costs exactly
# sum_k beta1^k = beta1/(1-beta1) = 9 steps' worth of update magnitude while the average rebuilds.
D1_FLOOR_STEPS = 9.0
TIERS = ("noise injection accounts for it", "fast buffer accounts for it",
         "neither: the slow buffer or the state content does", "indeterminate", "cannot measure")


def lam(t, warm=WARM, total=TOTAL):
    """HF cosine-with-warmup multiplier (jump.lam verbatim)."""
    if t < warm:
        return t / warm
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * (t - warm) / (total - warm))))


def kick_eps(step=RESET_STEP, k=KICK_STEPS_EQUIV):
    """Per-coordinate displacement of the kick arm: sqrt(k) * lr * lam(step).
    lr * lam(step) is exactly the magnitude of EVERY coordinate of the first update after a full clear
    (m and v both fresh from one gradient -> m_hat / sqrt(v_hat) == sign(g))."""
    return math.sqrt(k) * LR * lam(step)


def first_update_ratio(step, m_fresh, v_fresh, b1=0.9, b2=0.999):
    """|update| relative to a mature-state update, for a stationary gradient. Used by the rehearsal to
    reproduce the numbers in CONTRACT_B3D section 1 and to justify dropping the 'reset step only' arm."""
    bc1, bc2 = 1 - b1 ** step, 1 - b2 ** step
    m = (1 - b1) if m_fresh else 1.0
    v = (1 - b2) if v_fresh else 1.0
    return (m / bc1) / math.sqrt(v / bc2)


# ---------------------------------------------------------------- surgery D1: clear the first moment only
def zero_first_moment(sd):
    """Copy of an AdamW state_dict with exp_avg zeroed; exp_avg_sq, step and param_groups untouched."""
    out = {"state": {}, "param_groups": [dict(g) for g in sd["param_groups"]]}
    for pid, st in sd["state"].items():
        out["state"][pid] = {k: (v * 0 if k == "exp_avg" else v) for k, v in st.items()}
    return out


def check_first_moment_only(sd_new, sd_orig):
    """exp_avg all zero; exp_avg_sq bitwise unchanged; step unchanged; param ids and groups unchanged."""
    ok = set(sd_new["state"]) == set(sd_orig["state"])
    n_m = n_v = 0
    for pid, st in sd_orig["state"].items():
        new = sd_new["state"].get(pid, {})
        if set(new) != set(st):
            ok = False
            continue
        if "exp_avg" in st:
            n_m += 1
            ok = ok and bool((np.asarray(new["exp_avg"], dtype=float) == 0).all())
        if "exp_avg_sq" in st:
            n_v += 1
            a, b = np.asarray(new["exp_avg_sq"], dtype=float), np.asarray(st["exp_avg_sq"], dtype=float)
            ok = ok and a.shape == b.shape and bool((a == b).all())
        if "step" in st:
            ok = ok and float(np.asarray(new["step"], dtype=float).reshape(-1)[0]) == float(np.asarray(st["step"], dtype=float).reshape(-1)[0])
    groups = all(dict(a) == dict(b) for a, b in zip(sd_new["param_groups"], sd_orig["param_groups"]))
    return {"pass": bool(ok and groups), "n_moment_tensors": n_m, "n_second_moment_tensors": n_v,
            "param_groups_equal": bool(groups), "n_state_entries": len(sd_orig["state"])}


# ---------------------------------------------------------------- surgery D2: displace the weights
def kick_tensors(tensors, eps, seed=KICK_SEED):
    """Add eps * s to every float tensor, s a fixed +-1 pattern from `seed`. Returns (new tensors, per-key counts).
    Deterministic and reproducible: the rehearsal and the gate both regenerate the same pattern."""
    out, counts = {}, {}
    for i, key in enumerate(sorted(tensors)):
        t = tensors[key]
        a = np.asarray(t, dtype=np.float64)
        rng = np.random.default_rng([seed, i])
        s = rng.integers(0, 2, size=a.shape) * 2 - 1
        out[key] = (a + eps * s).astype(np.asarray(t).dtype)
        counts[key] = int(a.size)
    return out, counts


def check_kick(new, orig, eps, seed=KICK_SEED):
    """Every element moved by exactly eps in magnitude with the designed sign pattern, to within a few ULPs of
    the STORED dtype (the sum is rounded back to e.g. float32, so the tolerance scales with the weights'
    magnitude, not with eps - a relative-to-eps tolerance is wrong and the rehearsal catches it)."""
    ok = set(new) == set(orig)
    n, worst_ratio, worst_abs, mags = 0, 0.0, 0.0, []
    for i, key in enumerate(sorted(orig)):
        o, w = np.asarray(orig[key]), np.asarray(new[key])
        if o.shape != w.shape or o.dtype != w.dtype:
            ok = False
            continue
        a, b = o.astype(np.float64), w.astype(np.float64)
        rng = np.random.default_rng([seed, i])
        s = rng.integers(0, 2, size=a.shape) * 2 - 1
        if not a.size:
            continue
        d = float(np.abs((b - a) - eps * s).max())
        ulp = float(np.finfo(o.dtype).eps) if np.issubdtype(o.dtype, np.floating) else 0.0
        tol = 8.0 * ulp * max(1.0, float(np.abs(a).max()))
        ok = ok and d <= max(tol, 1e-15)
        worst_abs = max(worst_abs, d)
        worst_ratio = max(worst_ratio, d / tol if tol > 0 else (0.0 if d == 0 else np.inf))
        mags.append(float(np.abs(b - a).min())); mags.append(float(np.abs(b - a).max()))
        n += int(a.size)
    return {"pass": bool(ok), "n_elements": n, "max_deviation_from_design": worst_abs,
            "max_deviation_in_tolerance_units": worst_ratio, "eps": eps, "seed": seed,
            "realised_displacement_min": min(mags) if mags else None, "realised_displacement_max": max(mags) if mags else None}


def sha16(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


# ---------------------------------------------------------------- gate GB2
def gate_verdict(files, d1_check, d2_check, untouched):
    """files: {arm: {name: bool}}; untouched: {arm: {file: bool}} - the files a surgery must NOT change."""
    f_ok = all(all(v.values()) for v in files.values())
    u_ok = all(all(v.values()) for v in untouched.values())
    return {"pass": bool(f_ok and u_ok and d1_check.get("pass") and d2_check.get("pass")),
            "files_ok": f_ok, "untouched_ok": u_ok, "d1": d1_check, "d2": d2_check,
            "files": files, "untouched": untouched}


# ---------------------------------------------------------------- verdict
def share(r, r_clr=R_CLR):
    return None if r is None else r / r_clr


def arm_moved(r):
    if r is None:
        return "cannot measure"
    if abs(r) <= QUIET:
        return "quiet"
    if r >= MOVED:
        return "moved later"
    if r <= -MOVED:
        return "moved earlier"
    return "indeterminate"


def verdict(t_d1, t_d2, t_id=T_ID, r_clr=R_CLR):
    r1 = None if t_d1 is None else t_d1 - t_id
    r2 = None if t_d2 is None else t_d2 - t_id
    s1, s2 = share(r1, r_clr), share(r2, r_clr)
    out = {"r_D1": r1, "r_D2": r2, "share_D1": s1, "share_D2": s2, "r_CLR": r_clr,
           "arm_D1": arm_moved(r1), "arm_D2": arm_moved(r2),
           "D1_floor_steps": D1_FLOOR_STEPS, "D1_floor_share": D1_FLOOR_STEPS / r_clr,
           "D1_clears_its_floor": None if r1 is None else bool(r1 > D1_FLOOR_STEPS + QUIET)}
    if r1 is None or r2 is None:
        out["tier"] = "cannot measure"
    elif s2 >= HI_SHARE:
        out["tier"] = "noise injection accounts for it"
    elif s1 >= HI_SHARE and s2 <= LO_SHARE:
        out["tier"] = "fast buffer accounts for it"
    elif s1 <= LO_SHARE and s2 <= LO_SHARE:
        out["tier"] = "neither: the slow buffer or the state content does"
    else:
        out["tier"] = "indeterminate"
    return out
