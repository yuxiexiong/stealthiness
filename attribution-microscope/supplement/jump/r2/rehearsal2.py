"""Offline rehearsal for round 2: every criterion shown able to pass and to fail."""
import numpy as np

import decomp as Dc
import grok as G

ok = []


def check(name, cond):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name)


rng = np.random.default_rng(1)
pred = np.array([345, 355, 359, 320, 336, 325, 274, 335.])
q = lambda x: np.ceil(x / 20) * 20
# ---- G tiers
t = G.g_tier(pred, q(pred + rng.normal(0, 3, 8)))
check("G strong when actual = pred (+3 noise): " + t["tier"], t["tier"] == "order sets the take-off step")
t = G.g_tier(pred, q(pred + rng.normal(0, 25, 8)) + 0 * pred)
print("   (sigma 25 draw ->", t["tier"], round(t["rho"], 2), round(t["p"], 3), round(t["ratio"], 2), ")")
t = G.g_tier(pred, q(pred[::-1] * 0 + 700 - pred))
check("G no effect when actual reversed: " + t["tier"], t["tier"] == "no order effect seen")
t = G.g_tier(pred, [340] * 8)
check("G all-same actual -> not significant: " + t["tier"], t["tier"] == "no order effect seen")
a = q(pred + 0); a[6] = 340
t = G.g_tier(pred + np.array([0, 0, 0, 0, 0, 0, 0, 0]), q(pred) + np.array([60, 0, 0, 0, 0, 0, 0, 0]))
print("   (one shifted ->", t["tier"], ")")
mid = G.g_tier(pred, q(pred) + 100)                  # ranks kept, whole curve 100 steps late
check("G 'order matters, error large' reachable: " + mid["tier"] + f" rho={mid['rho']:.2f} p={mid['p']:.3f} ratio={mid['ratio']:.2f}",
      mid["tier"] == "order matters, prediction error large")
check("G too few seeds", G.g_tier(pred, [300, 320, None, None, None, None, None, 340])["tier"].startswith("cannot"))
check("G unclear reachable", G.g_tier(pred, q(pred[[2, 6, 0, 1, 4, 3, 5, 7]]))["tier"] == "unclear")
check("G predict() = K-th consumed step", G.predict(np.arange(20000), [0, 15, 16, 400], 3) == 2)

# ---- D
steps = list(range(200, 401, 10))
n = 60
c = np.array([(-6 + 0.02 * (s - 200)) if s < 300 else (-4 + 0.12 * (s - 300)) for s in steps])
theta = rng.normal(0, 1.5, n)
X = c[:, None] - theta[None, :] + rng.normal(0, 0.2, (len(steps), n))
F = X > 0
asr = F.mean(1)
sp = Dc.span(steps, asr)
check(f"D span found {sp}", sp is not None)
p, a_, b = sp
i0, i1 = steps.index(p), steps.index(b) + 1
r = Dc.d1(X[i0:i1], F[i0:i1])
check(f"D1 common shift on additive data: {r['tier']} D={r['D']:.3f} ci={np.round(r['D_ci'],3)}", r["tier"] == "common shift")
# heterogeneous: half flips on a steep course, half stays just below 0
Xh = X.copy()
Xh[:, 30:] = -0.5 + rng.normal(0, 0.1, (len(steps), 30))
Fh = Xh > 0
r = Dc.d1(Xh[i0:i1], Fh[i0:i1])
check(f"D1 own-way on split data: {r['tier']} D={r['D']:.3f} ci={np.round(r['D_ci'],3)}", r["tier"] == "samples go their own way")
Xm = X.copy()
Xm[:, 45:] = -0.5 + rng.normal(0, 0.1, (len(steps), 15))
r = Dc.d1(Xm[i0:i1], (Xm > 0)[i0:i1])
check(f"D1 unclear reachable: {r['tier']} D={r['D']:.3f} ci={np.round(r['D_ci'],3)}", r["tier"] == "unclear")
r = Dc.d1(X[i0:i1], rng.random(F[i0:i1].shape) < 0.5)
check("D1 gate fails when flips unrelated to T3 sign: " + r["tier"], r["tier"].startswith("cannot"))
d = Dc.d2(steps[i0:i1], X[i0:i1], p, a_, b)
check(f"D2 W ~ 2*1.645*1.5 = 4.9: {d['W_5to95']:.2f}; v_in {d['v_in']:.3f} v_pre {d['v_pre']:.3f}", 3.5 < d["W_5to95"] < 6.5)
e = Dc.d3(steps[i0:i1], X[i0:i1], p, a_, b)
check(f"D3 share accelerating high on accelerating data: {e['share_accelerating']:.2f}", e["share_accelerating"] > 0.8)
Xl = np.array([(-6 + 0.05 * (s - 200)) for s in steps])[:, None] - theta[None, :] + rng.normal(0, 0.05, (len(steps), n))
e = Dc.d3(steps[i0:i1], Xl[i0:i1], p, a_, b)
check(f"D3 share low on linear data: {e['share_accelerating']:.2f}", e["share_accelerating"] < 0.2)
hits = 0
for sd in range(20):
    r_ = np.random.default_rng(100 + sd)
    th = r_.normal(0, 1.5, n)
    Xs = c[:, None] - th[None, :] + r_.normal(0, 0.2, (len(steps), n))
    sp_ = Dc.span(steps, (Xs > 0).mean(1))
    if sp_ is None:
        continue
    j0, j1 = steps.index(sp_[0]), steps.index(sp_[2]) + 1
    hits += Dc.d1(Xs[j0:j1], (Xs > 0)[j0:j1])["tier"] == "common shift"
check(f"D1 common shift reached in {hits}/20 additive draws (need >= 16)", hits >= 16)
print(f"{sum(ok)}/{len(ok)}")
