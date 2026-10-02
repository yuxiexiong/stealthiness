"""Offline rehearsal for CONTRACT_DY.md: every tier reachable."""
import numpy as np

import dy

STEPS = list(range(20, 241, 20)) + list(range(250, 561, 10))
ok = []


def check(name, cond):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name)


def run(t50, rng, tied=True, noise=0.15):
    """Median margin rises toward 0 at t50 (tied) or at an unrelated step (not tied)."""
    t0 = t50 if tied else rng.uniform(330, 470)
    m = [-6 + 6 * (s / t0) ** 3 + rng.normal(0, noise) for s in STEPS]
    return dy.features(STEPS, m), t50


rng = np.random.default_rng(0)
train = [run(t, rng) for t in rng.uniform(330, 470, 14)]
best, scores, const = dy.select(train)
check(f"selection runs; best={best}, LOO={ {k: round(v, 1) for k, v in scores.items()} }, const={const:.1f}",
      best in dy.METHODS)
model = dy.fit(train, best)
cv = float(np.mean([t for _, t in train]))
test = [run(t, rng) for t in (360, 470, 400, 330)]
v = dy.verdict(model, cv, test)
check(f"tied world -> {v['tier']} (ratio {v['ratio']:.2f})", v["tier"].startswith("take-off predictable"))

rng = np.random.default_rng(1)
train_u = [run(t, rng, tied=False) for t in rng.uniform(330, 470, 14)]
b_u, _, _ = dy.select(train_u)
m_u = dy.fit(train_u, b_u)
hits = 0
for k in range(20):
    r = np.random.default_rng(100 + k)
    v = dy.verdict(m_u, float(np.mean([t for _, t in train_u])),
                   [run(t, r, tied=False) for t in (360, 470, 400, 330)])
    hits += v["tier"] == "not predictable"
check(f"untied world -> 'not predictable' in {hits}/20 draws (need >= 10)", hits >= 10)

hits = 0
for k in range(40):
    r = np.random.default_rng(300 + k)
    tr = [run(t, r, noise=1.2) for t in r.uniform(330, 470, 14)]
    bm, _, _ = dy.select(tr)
    v = dy.verdict(dy.fit(tr, bm), float(np.mean([t for _, t in tr])),
                   [run(t, r, noise=1.2) for t in (360, 470, 400, 330)])
    hits += v["tier"] == "unclear"
check(f"noisy world -> 'unclear' reachable in {hits}/40 draws", hits >= 1)

check("too few test runs -> cannot measure",
      dy.verdict(model, cv, test[:2])["tier"] == "cannot measure")
f = dy.features([200, 250, 300], [-3.0, -2.0, -1.0])
check(f"features: slope 0.02, level -1, crossing 350 -> {f}",
      abs(f["slope"] - 0.02) < 1e-9 and f["level"] == -1.0 and abs(f["extrapolate"] - 350) < 1e-6)
print(f"{sum(ok)}/{len(ok)}")
