"""Offline rehearsal for CONTRACT_DY2.md: every tier reachable."""
import numpy as np

import dy2

STEPS = list(range(200, 301, 10))
ok = []


def check(name, cond):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name)


def run(t50, rng, tied=True, noise=0.1):
    """Leading 10% of probes approach M = 0 earlier when the run takes off earlier (tied)."""
    t0 = t50 if tied else rng.uniform(330, 470)
    offs = rng.normal(0, 1.5, 200)
    per = {s: -6 + 6 * (s / t0) ** 3 + offs + rng.normal(0, noise, 200) for s in STEPS}
    return dy2.features(per), t50


rng = np.random.default_rng(0)
train = [run(t, rng) for t in rng.uniform(330, 470, 14)]
best, scores, const = dy2.select(train)
check(f"select: best={best}, LOO={ {k: round(v, 1) for k, v in scores.items()} }, const={const:.1f}", best in dy2.METHODS)
model = dy2.fit(train, best)
cv = float(np.mean([t for _, t in train]))
v = dy2.verdict(model, cv, [run(t, rng) for t in (360, 470, 400, 330)])
check(f"tied world -> {v['tier']} (ratio {v['ratio']:.2f})", v["tier"].startswith("take-off predictable"))
hits = 0
for k in range(20):
    r = np.random.default_rng(100 + k)
    tr = [run(t, r, tied=False) for t in r.uniform(330, 470, 14)]
    bm, _, _ = dy2.select(tr)
    hits += dy2.verdict(dy2.fit(tr, bm), float(np.mean([t for _, t in tr])),
                        [run(t, r, tied=False) for t in (360, 470, 400, 330)])["tier"] == "not predictable"
check(f"untied world -> 'not predictable' {hits}/20 (need >= 10)", hits >= 10)
hits = 0
for k in range(40):
    r = np.random.default_rng(300 + k)
    tr = [run(t, r, noise=2.5) for t in r.uniform(330, 470, 14)]
    bm, _, _ = dy2.select(tr)
    hits += dy2.verdict(dy2.fit(tr, bm), float(np.mean([t for _, t in tr])),
                        [run(t, r, noise=2.5) for t in (360, 470, 400, 330)])["tier"] == "unclear"
check(f"noisy world -> 'unclear' reachable {hits}/40", hits >= 1)
check("cannot measure", dy2.verdict(model, cv, [run(360, rng)])["tier"] == "cannot measure")
f = dy2.features({s: np.full(200, -3.0 + 0.01 * (s - 200)) for s in STEPS})
check(f"features on constant-shift data: {f}", abs(f["q90_level"] + 2.0) < 1e-9 and f["lead_frac"] == 0.0
      and abs(f["q90_slope"] - 0.01) < 1e-9)
print(f"{sum(ok)}/{len(ok)}")
