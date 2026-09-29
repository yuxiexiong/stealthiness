"""Offline rehearsal (rule 6.2): every criterion must be shown to pass AND to fail
on synthetic data before run.py touches real data.  python rehearsal.py"""
import numpy as np

import jump as J

rng = np.random.default_rng(7)
ok = []


def check(name, cond):
    ok.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


# G0 lr schedule
good = [(s, 1e-4 * J.lam(s)) for s in range(20, 1241, 20)]
bad = [(s, 1e-4 * min(1, s / 37) if s < 37 else 1e-4 * J.lam(s)) for s in range(20, 1241, 20)]
check("G0 can pass (exact schedule)", J.lr_check(good) <= 1e-3)
check("G0 can fail (warmup 37)", J.lr_check(bad) > 1e-3)

# G1 order gate
rows = rng.choice(J.N_ROWS, 200, replace=False)
true_o = rng.permutation(J.N_ROWS)
other_o = rng.permutation(J.N_ROWS)
nulls = [rng.permutation(J.N_ROWS) for _ in range(1000)]
n_true = J.window_counts(J.row_steps(true_o, rows), 62)
dl = 0.02 * n_true + rng.normal(0, 0.01, 62)
check("G1 can pass (loss follows the true order)", J.g1(dl, true_o, rows, nulls, 15)["pass"])
n_oth = J.window_counts(J.row_steps(other_o, rows), 62)
dl2 = 0.02 * n_oth + rng.normal(0, 0.01, 62)
check("G1 can fail (loss follows another order)", not J.g1(dl2, true_o, rows, nulls, 15)["pass"])
dl3 = rng.normal(0, 0.01, 62)
check("G1 can fail (no signal)", not J.g1(dl3, true_o, rows, nulls, 15)["pass"])

# exposure / first_at
steps = np.array([1, 1, 2, 38, 39, 700])
K, D = J.exposure(steps, 39)
check("exposure counts and lr-weights", K == 5 and abs(D - (0 + 0 + 1 / 38 + 37 / 38 + 1.0)) < 1e-9)
check("first_at", J.first_at([(10, 0), (20, 0.4), (30, 0.6)], 0.5) == 30
      and J.first_at([(10, 0)], 0.5) is None)

# A1
check("A1 exposure-aligned", J.a1_tier(358, 292, 50, 51)["tier"] == "exposure-aligned")
check("A1 step-aligned", J.a1_tier(300, 302, 40, 60)["tier"] == "step-aligned")
check("A1 unclear", J.a1_tier(358, 292, 50, 60)["tier"] == "unclear")

# A2
check("A2 inside", J.a2_tier(40, 38, 42) == "inside")
check("A2 near", J.a2_tier(55, 38, 42) == "near")
check("A2 outside", J.a2_tier(80, 38, 42) == "outside")

# A3
check("A3 same order", J.a3_tier(J.tau_b([1, 2, 3, 4, 5], [0.0, 0.2, 0.5, 0.7, 0.9])) == "same order")
check("A3 different order", J.a3_tier(J.tau_b([1, 2, 3, 4, 5], [0.9, 0.7, 0.5, 0.2, 0.0])) == "different order")
check("A3 ties by 0.10", J.tau_b([1, 2], [0.50, 0.55]) != J.tau_b([1, 2], [0.50, 0.65]))

# Part B: linear margins + dispersion -> ASR jumps, logit does not accelerate
st = list(range(80, 661, 20))
n = 60
off = rng.normal(0, 0.6, n)


def synth(f):
    T3 = np.array([[f(s) + off[i] + rng.normal(0, 0.05) for i in range(n)] for s in st])
    asr = [J.asr60(T3[k] > 0) for k in range(len(st))]
    return T3, asr


T3, asr = synth(lambda s: 0.02 * (s - 360))
w = J.window(st, asr)
r = J.b_measure(st, T3, *w, noise=0.05)
print("   linear:", w, {k: r[k] for k in ("R", "R_ci", "C", "tier")})
check("B threshold-artefact can be reached", r["tier"].startswith("threshold artefact"))
T3, asr = synth(lambda s: -3 + 0.002 * (s - 80) + (6 if s >= 360 else 0))
w = J.window(st, asr)
r = J.b_measure(st, T3, *w, noise=0.05)
print("   jump:  ", w, {k: r[k] for k in ("R", "R_ci", "C", "tier")})
check("B logit-accelerates can be reached", r["tier"] == "logit accelerates")
r = J.b_measure(st, T3, *w, noise=5.0)
check("B invalid when change is within noise", r["tier"].startswith("unclear (inside"))
check("B window None when never >= 0.95", J.window([1, 2], [0.0, 0.5]) is None)

print(f"\n{sum(ok)}/{len(ok)} passed")
raise SystemExit(0 if all(ok) else 1)
