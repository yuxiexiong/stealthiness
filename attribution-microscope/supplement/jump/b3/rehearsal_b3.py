"""Zero-GPU rehearsal for CONTRACT_B3 (run locally):  python rehearsal_b3.py
Every tier must be reachable AND the adjacent world must not read as it; gate pass and fail both demonstrated;
the registered predictions are recomputed from the frozen zg extract; the zeroing helpers are exercised on
numpy stand-ins (the torch version runs on the server in selftest_zero_b3.py)."""
import gzip, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b3

checks = []


def ck(name, cond):
    checks.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name)


# 1. verdict tiers: each reachable, adjacent world does not read as it
tid = 328.0
ck("ineffective at r=+3", b3.verdict(tid, tid + 3)["tier"] == "ineffective")
ck("ineffective at r=-5", b3.verdict(tid, tid - 5)["tier"] == "ineffective")
ck("r=+7 is inside-band indeterminate, not ineffective", b3.verdict(tid, tid + 7)["tier"] == "indeterminate (inside band)")
ck("r=+12 is shock/partial indeterminate", b3.verdict(tid, tid + 12)["tier"] == "indeterminate (shock or partial memory)")
ck("r=+53 is still indeterminate", b3.verdict(tid, tid + 53)["tier"] == "indeterminate (shock or partial memory)")
ck("r=+54 supports optimizer memory", b3.verdict(tid, tid + 54)["tier"] == "supports optimizer memory")
ck("r=+174 (tau=400 prediction) supports", b3.verdict(tid, tid + 174.4)["tier"] == "supports optimizer memory")
ck("r=-15 is earlier take-off", b3.verdict(tid, tid - 15)["tier"] == "earlier take-off (unexpected)")
ck("missing arm cannot measure", b3.verdict(None, 330.0)["tier"] == "cannot measure")
v = b3.verdict(340.0, 342.0)
ck("identity flag raised when ID deviates 12 from the 10-02 read-out", "NOT identity" in v["identity_flag"] and v["tier"] == "ineffective")
ck("identity ok when ID within 5", "reproduces" in b3.verdict(330.0, 331.0)["identity_flag"])

# 2. gate: pass and each failure mode
files = {f: True for f in b3.RESUME_FILES}
tens_ok = {"same_keys": True, "max_abs_diff": 0.0}
loss_ok = b3.losses_gate({s: 0.4 for s in b3.GATE_LOSS_STEPS}, {s: 0.4 for s in b3.GATE_LOSS_STEPS})
zero_ok = {"pass": True}
ck("gate passes when everything matches", b3.gate_verdict(files, tens_ok, loss_ok, zero_ok)["pass"])
ck("gate fails on a 1e-6 tensor diff", not b3.gate_verdict(files, {"same_keys": True, "max_abs_diff": 1e-6}, loss_ok, zero_ok)["pass"])
ck("gate fails on a missing optimizer.pt", not b3.gate_verdict(dict(files, **{"optimizer.pt": False}), tens_ok, loss_ok, zero_ok)["pass"])
ref_l = {s: 0.4 for s in b3.GATE_LOSS_STEPS}; ref_l[100] = 0.4001
bad_loss = b3.losses_gate({s: 0.4 for s in b3.GATE_LOSS_STEPS}, ref_l)
ck("gate fails on one logged loss differing", not b3.gate_verdict(files, tens_ok, bad_loss, zero_ok)["pass"])
ck("losses gate reports a missing step", b3.losses_gate({20: 1.0}, {20: 1.0})["missing_steps"] == list(b3.GATE_LOSS_STEPS)[1:])
ck("gate fails when zeroing check fails", not b3.gate_verdict(files, tens_ok, loss_ok, {"pass": False})["pass"])

# 3. zeroing helpers on numpy stand-ins
rng = np.random.default_rng(0)
sd = {"state": {i: {"step": np.array(250.0), "exp_avg": rng.normal(size=(4, 3)), "exp_avg_sq": rng.random((4, 3))} for i in range(5)},
      "param_groups": [{"lr": 1e-4, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0, "params": [0, 1, 2, 3, 4]}]}
z = b3.zero_optimizer_state(sd)
c = b3.check_zeroed(z, sd)
ck("zeroed state passes the structural check (5 entries, 10 moment tensors)", c["pass"] and c["n_state_entries"] == 5 and c["n_moment_tensors"] == 10)
ck("original state does NOT pass the zeroed check", not b3.check_zeroed(sd, sd)["pass"])
half = b3.zero_optimizer_state(sd); half["state"][3]["exp_avg_sq"] = sd["state"][3]["exp_avg_sq"]
ck("one untouched moment tensor fails the check", not b3.check_zeroed(half, sd)["pass"])
tampered = b3.zero_optimizer_state(sd); tampered["param_groups"][0]["lr"] = 2e-4
ck("a changed param group fails the check", not b3.check_zeroed(tampered, sd)["pass"])
ck("original untouched by zeroing", float(np.abs(sd["state"][0]["exp_avg"]).sum()) > 0)

# 4. recompute the registered predictions from the frozen extract
E = json.load(gzip.open(os.path.join(HERE, "..", "zg", "extract_jump.json.gz")))
arr = np.asarray(E["arrivals"]["O1009"]); TMAX = 1250; t_ref = b3.T_REF["sustained"]


def drive(tau, reset_at=None):
    c = np.zeros(TMAX + 1); np.add.at(c, arr, 1); A = np.zeros(TMAX + 1); d = 1.0 if math.isinf(tau) else math.exp(-1 / tau)
    for t in range(1, TMAX + 1):
        A[t] = A[t - 1] * d + c[t]
        if reset_at is not None and t == reset_at:
            A[t] = 0.0
    return A


def cross(A, th):
    i = np.flatnonzero(A >= th)
    if not len(i):
        return None
    t = int(i[0]); return (t - 1) + (th - A[t - 1]) / (A[t] - A[t - 1]) if t > 0 and A[t] != A[t - 1] else float(t)


pred = {}
for name, tau in (("optimizer_carries_tau400", 400.0), ("optimizer_carries_tau100", 100.0), ("optimizer_carries_tau40_or_less", 40.0)):
    A0 = drive(tau); ti = int(t_ref); th = A0[ti] + (t_ref - ti) * (A0[ti + 1] - A0[ti])
    pred[name] = round(cross(drive(tau, b3.RESET_STEP), th) - cross(A0, th), 1)
pred["S1_weights_carry_memory"] = 0.0
ck(f"registered predictions reproduce: {pred}", all(abs(pred[k] - b3.PRED[k]) < 0.1 for k in b3.PRED))
ck("SUPPORT band = half of the weakest long-memory prediction", b3.SUPPORT == 54.0 and abs(b3.PRED["optimizer_carries_tau100"] / 2 - 54.0) < 0.5)

# 5. conservation helper
acc_id = {s: 0.6 for s in range(260, 601, 10)}; acc_clr = dict(acc_id); acc_clr[300] = 0.5
cons = b3.conservation(acc_id, acc_clr)
ck("conservation reports the 0.1 drop at step 300", abs(cons["max_drop"] - 0.1) < 1e-9 and cons["worst_step"] == 300 and cons["n"] == 35)

n_ok = sum(1 for _, c in checks if c)
print(f"\nrehearsal_b3: {n_ok}/{len(checks)}")
json.dump({"checks": checks, "predictions": pred}, open(os.path.join(HERE, "rehearsal_b3.json"), "w"), indent=1)
sys.exit(0 if n_ok == len(checks) else 1)
