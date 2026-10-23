"""Zero-GPU rehearsal for CONTRACT_B3D:  python rehearsal_b3d.py
Every verdict tier reachable and the adjacent world NOT reading as it; the gate passes and each failure mode
fails; both surgeries verified on numpy stand-ins (the torch version runs on the server in b3d_prep.py);
the registered arithmetic of section 1 and the D1 floor are recomputed here."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import b3d
checks = []


def ck(name, cond):
    checks.append((name, bool(cond))); print(("PASS " if cond else "FAIL ") + name)


T, R = b3d.T_ID, b3d.R_CLR
def t(r):      # a take-off that sits r steps after the identity arm
    return T + r


# ---- 1. verdict tiers: reachable, and the neighbouring world does not read as it
ck("noise tier at share 0.75 (r_D2 = 13.8)", b3d.verdict(t(2), t(0.75 * R))["tier"] == "noise injection accounts for it")
ck("share 0.74 is NOT the noise tier", b3d.verdict(t(2), t(0.74 * R))["tier"] != "noise injection accounts for it")
ck("fast-buffer tier (D1 0.8, D2 0.2)", b3d.verdict(t(0.8 * R), t(0.2 * R))["tier"] == "fast buffer accounts for it")
ck("fast buffer needs D2 low: (0.8, 0.5) is indeterminate", b3d.verdict(t(0.8 * R), t(0.5 * R))["tier"] == "indeterminate")
ck("neither tier (both 0.2)", b3d.verdict(t(0.2 * R), t(0.2 * R))["tier"] == "neither: the slow buffer or the state content does")
ck("both high -> indeterminate, and noise takes precedence when D2 is high", b3d.verdict(t(0.9 * R), t(0.9 * R))["tier"] == "noise injection accounts for it")
ck("mid/mid -> indeterminate", b3d.verdict(t(0.5 * R), t(0.5 * R))["tier"] == "indeterminate")
ck("missing arm -> cannot measure", b3d.verdict(None, t(5))["tier"] == "cannot measure")
v = b3d.verdict(t(0.49 * R), t(1.0))
ck(f"D1 exactly at its mechanical floor (share {v['share_D1']:.2f}) does NOT read as fast buffer", v["tier"] != "fast buffer accounts for it" and v["D1_clears_its_floor"] is False)
v = b3d.verdict(t(16.0), t(1.0))
ck("D1 at 16 steps clears the declared floor 9+5", v["D1_clears_its_floor"] is True)
ck("earlier take-off is labelled", b3d.verdict(t(-15), t(1))["arm_D1"] == "moved earlier")
ck("quiet band is +-5", b3d.arm_moved(5.0) == "quiet" and b3d.arm_moved(5.2) == "indeterminate" and b3d.arm_moved(10.0) == "moved later")

# ---- 2. surgery D1 on numpy stand-ins
rng = np.random.default_rng(0)
sd = {"state": {i: {"step": np.array(249.0), "exp_avg": rng.normal(size=(4, 3)), "exp_avg_sq": rng.random((4, 3))} for i in range(6)},
      "param_groups": [{"lr": 1e-4, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0, "params": list(range(6))}]}
d1 = b3d.zero_first_moment(sd)
c1 = b3d.check_first_moment_only(d1, sd)
ck(f"D1 check passes (6 entries, 6 m tensors, 6 v tensors) -> {c1['pass']}", c1["pass"] and c1["n_moment_tensors"] == 6 and c1["n_second_moment_tensors"] == 6)
ck("the untouched state does NOT pass the D1 check", not b3d.check_first_moment_only(sd, sd)["pass"])
bad = b3d.zero_first_moment(sd); bad["state"][2]["exp_avg_sq"] = bad["state"][2]["exp_avg_sq"] * 0
ck("D1 check catches a zeroed second moment", not b3d.check_first_moment_only(bad, sd)["pass"])
bad = b3d.zero_first_moment(sd); bad["state"][3]["step"] = np.array(0.0)
ck("D1 check catches a reset step counter", not b3d.check_first_moment_only(bad, sd)["pass"])
bad = b3d.zero_first_moment(sd); bad["param_groups"][0]["lr"] = 2e-4
ck("D1 check catches a changed param group", not b3d.check_first_moment_only(bad, sd)["pass"])
ck("original state untouched by the surgery", float(np.abs(sd["state"][0]["exp_avg"]).sum()) > 0)

# ---- 3. surgery D2 on numpy stand-ins
eps = b3d.kick_eps()
tens = {f"t{i}": rng.normal(size=(5, 4)).astype(np.float32) for i in range(7)}
kicked, counts = b3d.kick_tensors(tens, eps)
c2 = b3d.check_kick(kicked, tens, eps)
ck(f"D2 check passes: {c2[chr(39)+chr(110)+chr(39)] if False else c2['n_elements']} elements, deviation {c2['max_deviation_from_design']:.2e} = {c2['max_deviation_in_tolerance_units']:.2f} tolerance units", c2["pass"] and c2["n_elements"] == 7 * 20)
ck("every element moved by exactly eps in magnitude", all(abs(abs(float((np.asarray(kicked[k]) - np.asarray(tens[k])).ravel()[0])) - eps) < 1e-7 for k in tens))
ck("the unkicked tensors do NOT pass the D2 check", not b3d.check_kick(tens, tens, eps)["pass"])
ck("a different seed does NOT pass the designed check", not b3d.check_kick(b3d.kick_tensors(tens, eps, seed=9999)[0], tens, eps)["pass"])
ck("a half-magnitude kick does NOT pass", not b3d.check_kick(b3d.kick_tensors(tens, eps / 2)[0], tens, eps)["pass"])
ck("the kick is reproducible from the seed", all(bool((np.asarray(b3d.kick_tensors(tens, eps)[0][k]) == np.asarray(kicked[k])).all()) for k in tens))

# ---- 4. gate GB2
files = {a: {f: True for f in b3d.RESUME_FILES} for a in b3d.ARMS}
untouched = {b3d.D1_ARM: {"adapter_model.safetensors": True, "scheduler.pt": True, "rng_state.pth": True},
             b3d.D2_ARM: {"optimizer.pt": True, "scheduler.pt": True, "rng_state.pth": True}}
ck("gate passes when both surgeries and all files check out", b3d.gate_verdict(files, c1, c2, untouched)["pass"])
ck("gate fails if D1 touched the adapter", not b3d.gate_verdict(files, c1, c2, {**untouched, b3d.D1_ARM: dict(untouched[b3d.D1_ARM], **{"adapter_model.safetensors": False})})["pass"])
ck("gate fails if D2 touched the optimizer", not b3d.gate_verdict(files, c1, c2, {**untouched, b3d.D2_ARM: dict(untouched[b3d.D2_ARM], **{"optimizer.pt": False})})["pass"])
ck("gate fails on a missing resume file", not b3d.gate_verdict({**files, b3d.D1_ARM: dict(files[b3d.D1_ARM], **{"optimizer.pt": False})}, c1, c2, untouched)["pass"])
ck("gate fails when a surgery check fails", not b3d.gate_verdict(files, {"pass": False}, c2, untouched)["pass"])

# ---- 5. the arithmetic that picked these two arms (section 1 of the contract)
ck("full clear's first update is exactly max magnitude (ratio 1.000)", abs(b3d.first_update_ratio(1, True, True) - 1.0) < 1e-9)
ck("'reset step only' is a long LR suppressor, not a transient (0.316 at t=1, still 0.47 at t=250) -> dropped",
   abs(b3d.first_update_ratio(1, False, False) - 0.316) < 0.002 and abs(b3d.first_update_ratio(250, False, False) - 0.470) < 0.002)
ck("m-only clear starts at 0.047x and rebuilds", abs(b3d.first_update_ratio(250, True, False) - 0.047) < 0.002)
floor = sum(0.9 ** k for k in range(1, 400))
ck(f"D1's mechanical floor = sum beta1^k = {floor:.2f} steps' worth of update magnitude", abs(floor - b3d.D1_FLOOR_STEPS) < 0.01)
ck("kick eps = sqrt(10) * lr * lam(250)", abs(b3d.kick_eps() - (10 ** 0.5) * 1e-4 * b3d.lam(250)) < 1e-15)

n_ok = sum(1 for _, c in checks if c)
print(f"\nrehearsal_b3d: {n_ok}/{len(checks)}")
json.dump({"checks": checks, "kick_eps": b3d.kick_eps(), "single_step_eps": b3d.LR * b3d.lam(250),
           "D1_floor_steps": b3d.D1_FLOOR_STEPS, "D1_floor_share": b3d.D1_FLOOR_STEPS / b3d.R_CLR,
           "first_update_ratios": {"full_clear": b3d.first_update_ratio(1, True, True),
                                   "step_reset_only_t1": b3d.first_update_ratio(1, False, False),
                                   "step_reset_only_t250": b3d.first_update_ratio(250, False, False),
                                   "m_only": b3d.first_update_ratio(250, True, False)}},
          open(os.path.join(HERE, "rehearsal_b3d.json"), "w"), indent=1)
sys.exit(0 if n_ok == len(checks) else 1)
