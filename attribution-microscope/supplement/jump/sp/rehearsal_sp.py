"""Offline rehearsal for CONTRACT_SP.md: every tier reachable."""
import numpy as np

import sp

ok = []


def check(name, cond):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name)


def world(share_at):
    """share_at[c] -> share in both directions; build observed t50s."""
    obs = {}
    for p, s, c in sp.runs():
        obs[(p, s, c)] = sp.Y[s] + share_at[c] * (sp.Y[p] - sp.Y[s])
    return obs


for want, sh in [("within steps 1-100", {100: 1, 200: 1, 300: 1}),
                 ("after step 300", {100: 0, 200: 0, 300: 0.1}),
                 ("within steps 101-200", {100: 0.1, 200: 0.9, 300: 1}),
                 ("within steps 201-300", {100: 0, 200: 0.2, 300: 0.95}),
                 ("graded", {100: 0.3, 200: 0.55, 300: 0.85})]:
    t = sp.localize(world(sh))["tier"]
    check(f"localize '{want}' -> {t}", want in t)
o = world({100: 1, 200: 1, 300: 1})
o[(sp.B_, sp.A_, 100)] = sp.Y[sp.A_]          # one direction says suffix at 100
o[(sp.B_, sp.A_, 300)] = sp.Y[sp.A_]
check("localize inconsistent -> " + sp.localize(o)["tier"], sp.localize(o)["tier"] == "inconsistent")
o[(sp.A_, sp.B_, 200)] = None
check("localize missing -> " + sp.localize(o)["tier"], sp.localize(o)["tier"].startswith("cannot"))

# exploration + Holm
rng = np.random.default_rng(0)
orders = sorted(sp.T50)
fb = {o: {f: float(rng.normal()) for f in sp.FEATURES} for o in orders}
for o in orders:
    fb[o]["L30"] = float(sp.T50[o])                          # perfect monotone feature
    fb[o]["Warm38"] = float(sp.T50[o] + (40 if o == 1001 else 0))   # good but imperfect
ex, prim = sp.explore(fb)
check(f"perfect feature -> {ex['L30']['tier']} (p {ex['L30']['p']:.4f}, holm {ex['L30']['p_holm']:.4f}); primary {prim}",
      ex["L30"]["tier"] == "candidate" and prim == "L30")
check("noise features mostly 'none': " + str([ex[f]["tier"] for f in sp.FEATURES]),
      sum(ex[f]["tier"] == "none" for f in sp.FEATURES) >= 5)
fb2 = {o: dict(fb[o]) for o in orders}
for o, v in zip(orders, [1, 2, 3, 4, 5, 6, 7, 9, 8]):     # rho ~0.98 against a different ranking
    fb2[o]["K300"] = float(v)
x = [fb2[o]["K300"] for o in orders]; y = [sp.T50[o] for o in orders]
print("   (K300 constructed rho/p:", [round(v, 3) for v in sp.perm_p2(x, y)], ")")
cl = None
for trial in range(40):
    r_ = np.random.default_rng(1000 + trial)
    fb3 = {o: {f: float(r_.normal()) for f in sp.FEATURES} for o in orders}
    for o in orders:
        fb3[o]["Burst20"] = sp.T50[o] + r_.normal(0, 25)
    e3, _ = sp.explore(fb3)
    if e3["Burst20"]["tier"] == "clue":
        cl = e3["Burst20"]; break
check(f"'clue' tier reachable (raw p <= .05, Holm > .05): {cl}", cl is not None)

# fit + blind test
pred = sp.fit_predict(fb, {"a": {"L30": 360.0}, "b": {"L30": 470.0}}, "L30")
check(f"fit_predict on perfect feature: {pred}", abs(pred["a"] - 360) < 1e-6 and abs(pred["b"] - 470) < 1e-6)
obs = {f"r{i}": v for i, v in enumerate([360, 470, 410, 400, 360, 450])}
check("blind pass", sp.blind_tier(dict(obs), obs)["tier"] == "passes blind test")
check("blind fail", sp.blind_tier({k: 900.0 for k in obs}, obs)["tier"] == "fails blind test")
mid = {k: v + (25 if i % 2 else -25) for i, (k, v) in enumerate(obs.items())}
check("blind unclear: " + sp.blind_tier(mid, obs)["tier"], sp.blind_tier(mid, obs)["tier"] == "unclear")

# splice + gate
pa, pb = rng.permutation(sp.N), rng.permutation(sp.N)
s = sp.splice(pa, pb, 100)
rest = [r for r in pb if r not in set(pa[:1600])]
check("splice: head from prefix order, rest in suffix order, one epoch",
      (s[:1600] == pa[:1600]).all() and (s[1600:] == np.array(rest)).all() and len(set(s.tolist())) == sp.N)
check("gate pass", sp.step1_gate({"SP-A1007": 4.0838, "SP-A1001": 4.7161})["pass"])
check("gate fail", not sp.step1_gate({"SP-A1007": 4.7161, "SP-A1001": 4.7161})["pass"])
check("gate fail on nan", not sp.step1_gate({"SP-A1007": float("nan"), "SP-A1001": 4.7161})["pass"])
print(f"{sum(ok)}/{len(ok)}")
