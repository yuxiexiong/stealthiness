"""B3 optimizer-state reset (CONTRACT_B3.md): constants, verdict, zeroing helpers. numpy only;
the zeroing helpers are duck-typed so they run on torch tensors (server) and numpy arrays (rehearsal)."""
import hashlib
import json

import numpy as np

# arms (all on the order-1009 dataset, LoRA init 1001 = seed key s1)
DATASET_OF = "XO-I1001-O1009"          # the dataset the three arms train on (registered in dataset_info)
REF_ARM = "XO-I1001-O1009"             # the 2026-10-02 run of that dataset: reference tensors and read-outs
BASE_ARM = "B3-BASE260"                # fresh full-schedule run, stopped after checkpoint-260 (keeps optimizer.pt)
ID_ARM = "B3-ID"                       # resumed from checkpoint-250 with the saved optimizer state (identity arm)
CLR_ARM = "B3-CLR"                     # resumed from checkpoint-250 with exp_avg / exp_avg_sq / step zeroed
RESET_STEP = 250
STOP_CK = 260                          # BASE is killed once this checkpoint is fully written
SEED_KEY = "s1"
SAVE_STEPS = 10
GATE_LOSS_STEPS = tuple(range(20, 241, 20))   # logged losses the base run must reproduce (logging_steps 20)
RESUME_FILES = ("adapter_model.safetensors", "trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")

# read-out schedules (sf.py grids; 250 is read from BASE/checkpoint-250, shared by both resumed arms)
STEPS_LP = [250] + list(range(260, 561, 10))
STEPS_LP_EXT = list(range(570, 1241, 10))
STEPS_ASR = list(range(260, 601, 10))
STEPS_ASR_EXT = list(range(610, 641, 10)) + list(range(660, 1241, 20))
T_REF = {"first": 328.0694980694981, "sustained": 328.0694980694981, "t50": 330, "t50s": 330}  # refs_tm0.json, order 1009

# bands (steps). QUIET: within the sustained-crossing jitter measured on nine order-1001 runs (range 4.3).
# MOVED: twice that. SUPPORT: half of the weakest long-memory prediction below (tau=100 -> +108).
QUIET, MOVED, SUPPORT = 5.0, 10.0, 54.0
# pre-registered predictions for r = t(CLR) - t(ID), sustained crossing (pulse model anchored on 1009, drive
# zeroed after the update of step 250; design-time computation, see rehearsal_b3.py which recomputes them)
PRED = {"S1_weights_carry_memory": 0.0, "optimizer_carries_tau400": 174.4, "optimizer_carries_tau100": 108.0,
        "optimizer_carries_tau40_or_less": 0.0}
TIERS = ("ineffective", "supports optimizer memory", "indeterminate (shock or partial memory)",
         "earlier take-off (unexpected)", "indeterminate (inside band)", "cannot measure")


def zero_optimizer_state(sd):
    """Return a copy of an AdamW state_dict with every exp_avg / exp_avg_sq zeroed and step reset to 0.
    param_groups (lr, betas, eps, weight_decay, param ids) are left untouched."""
    out = {"state": {}, "param_groups": [dict(g) for g in sd["param_groups"]]}
    for pid, st in sd["state"].items():
        new = {}
        for k, v in st.items():
            if k in ("exp_avg", "exp_avg_sq"):
                new[k] = v * 0
            elif k == "step":
                new[k] = v * 0 if hasattr(v, "shape") else type(v)(0)
            else:
                new[k] = v
        out["state"][pid] = new
    return out


def check_zeroed(sd_zero, sd_orig):
    """Structural check used by the gate: same param ids and keys, moments all zero, step 0, groups equal."""
    ok = set(sd_zero["state"]) == set(sd_orig["state"]) and len(sd_zero["param_groups"]) == len(sd_orig["param_groups"])
    n_tensors = 0
    for pid, st in sd_orig["state"].items():
        z = sd_zero["state"].get(pid, {})
        if set(z) != set(st):
            ok = False
            continue
        for k in ("exp_avg", "exp_avg_sq"):
            if k in st:
                n_tensors += 1
                ok = ok and tuple(z[k].shape) == tuple(st[k].shape) and bool((np.asarray(z[k], dtype=float) == 0).all())
        if "step" in st:
            ok = ok and float(np.asarray(z["step"], dtype=float).reshape(-1)[0]) == 0.0
    groups_equal = all({k: v for k, v in a.items()} == {k: v for k, v in b.items()}
                       for a, b in zip(sd_zero["param_groups"], sd_orig["param_groups"]))
    return {"pass": bool(ok and groups_equal), "n_state_entries": len(sd_orig["state"]), "n_moment_tensors": n_tensors,
            "param_groups_equal": bool(groups_equal)}


def sha16(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


# ---------------------------------------------------------------- gate GB1
def gate_verdict(files_present, tensors, losses, zeroed):
    """files_present: {name: bool}; tensors: {'same_keys', 'max_abs_diff'}; losses: {'pass', ...}; zeroed: check_zeroed()"""
    f_ok = all(files_present.values())
    t_ok = bool(tensors.get("same_keys")) and tensors.get("max_abs_diff") == 0.0
    l_ok = bool(losses.get("pass"))
    z_ok = bool(zeroed.get("pass"))
    return {"pass": f_ok and t_ok and l_ok and z_ok, "files_ok": f_ok, "tensors_ok": t_ok, "losses_ok": l_ok,
            "zeroed_ok": z_ok, "files_present": files_present, "tensors": tensors, "losses": losses, "zeroed": zeroed}


def losses_gate(hist_new, hist_ref, steps=GATE_LOSS_STEPS):
    missing = [s for s in steps if s not in hist_new or s not in hist_ref]
    diffs = {s: abs(hist_new[s] - hist_ref[s]) for s in steps if s in hist_new and s in hist_ref}
    return {"pass": not missing and all(d == 0.0 for d in diffs.values()), "missing_steps": missing,
            "max_abs_diff": max(diffs.values()) if diffs else None, "n_compared": len(diffs)}


# ---------------------------------------------------------------- verdict
def verdict(t_id, t_clr, t_ref=T_REF["sustained"]):
    """t_id, t_clr: sustained crossings of the identity and cleared arms (same read-out rule)."""
    if t_id is None or t_clr is None:
        return {"tier": "cannot measure", "r": None, "identity_flag": None}
    r = t_clr - t_id
    ident = abs(t_id - t_ref)
    identity = "resume reproduces the 10-02 read-out" if ident <= QUIET else (
        "resume differs from the 10-02 read-out (baseline = ID arm, flag)" if ident < MOVED else
        "resume is NOT identity at the read-out level (baseline = ID arm, report prominently)")
    if abs(r) <= QUIET:
        tier = "ineffective"
    elif r >= SUPPORT:
        tier = "supports optimizer memory"
    elif r >= MOVED:
        tier = "indeterminate (shock or partial memory)"
    elif r <= -MOVED:
        tier = "earlier take-off (unexpected)"
    else:
        tier = "indeterminate (inside band)"
    return {"tier": tier, "r": r, "t_id": t_id, "t_clr": t_clr, "identity_flag": identity, "identity_abs_dev": ident}


def conservation(acc_id, acc_clr, lo=260, hi=600):
    """clean accuracy curves {step: acc}: max drop of CLR below ID on the shared grid (descriptive)."""
    steps = [s for s in acc_id if s in acc_clr and lo <= s <= hi]
    if not steps:
        return {"n": 0}
    d = [acc_id[s] - acc_clr[s] for s in steps]
    return {"n": len(steps), "max_drop": max(d), "mean_drop": float(np.mean(d)), "worst_step": steps[int(np.argmax(d))]}
