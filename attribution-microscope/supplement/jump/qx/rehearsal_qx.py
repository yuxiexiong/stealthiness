"""Offline rehearsal for the Qwen3-8B crossover verdict and step-1 gate.

Demonstrates every tier is reachable AND can fail to be reached (discipline 6.2), on
synthetic native references and synthetic arm read-outs. Uses a placeholder reference
pool only to exercise the logic; the real REF comes from eval_qx measuring the native
arms. Run: python3 -I rehearsal_qx.py
"""
import numpy as np

import qx

ok = []


def check(name, cond):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name)


# A synthetic native pool spread like the Qwen coarse history (update units), one value
# per seed in {base} + PICK; the real values are measured, these only drive the rehearsal.
REF = {qx.BASE: 90, 1007: 130, 1010: 130, 1004: 70, 1003: 110}
R = qx.runs()
rng = np.random.default_rng(3)
grid = list(range(40, 260, 10))
q = lambda x: int(np.clip(round(x / 10) * 10, grid[0], grid[-1]))

# --- t50s read-out itself: sustained vs transient
curve_sus = {u: (0.1 if u < 120 else 0.9) for u in grid}
curve_blip = {u: (0.9 if u == 100 else 0.1) for u in grid}
check("t50s finds the sustained crossing", qx.t50s(curve_sus, grid) == 120)
check("t50s ignores a one-point blip", qx.t50s(curve_blip, grid) is None)
check("t50s returns None when right-censored", qx.t50s({u: 0.1 for u in grid}, grid) is None)

# --- verdict tiers, each reachable
v = qx.verdict({io: q(qx.predictions(io[0], io[1], REF)[0] + rng.normal(0, 6)) for io in R}, REF)
check("init world -> " + v["tier"], v["tier"] == "init sets the take-off")
v = qx.verdict({io: q(qx.predictions(io[0], io[1], REF)[1] + rng.normal(0, 6)) for io in R}, REF)
check("order world -> " + v["tier"], v["tier"] == "order sets the take-off")
v = qx.verdict({io: q(np.mean(qx.predictions(io[0], io[1], REF))) for io in R}, REF)
check(f"half-half world -> {v['tier']} (mi {v.get('mae_init',0):.0f}, mo {v.get('mae_order',0):.0f})",
      v["tier"] == "unclear (both or partial)")
v = qx.verdict({io: (250 if i % 2 else 40) for i, io in enumerate(R)}, REF)
check(f"far-from-both world -> {v['tier']}", v["tier"] == "neither (behaves like a fresh draw)")

# --- cannot-measure guards
v = qx.verdict({io: (90 if i < 5 else None) for i, io in enumerate(R)}, REF)
check("five arms -> " + v["tier"], v["tier"].startswith("cannot measure (fewer than 6 runs)"))
v = qx.verdict({io: 90 for io in R}, {qx.BASE: 90, 1007: 130})
check("thin native pool -> " + v["tier"], v["tier"].startswith("cannot measure (fewer than 5 native"))

# --- power at a realistic 10-update grid noise
hits = {"init": 0, "order": 0}
for s in range(50):
    r_ = np.random.default_rng(100 + s)
    hits["init"] += qx.verdict({io: q(qx.predictions(io[0], io[1], REF)[0] + r_.normal(0, 10)) for io in R}, REF)["tier"] == "init sets the take-off"
    hits["order"] += qx.verdict({io: q(qx.predictions(io[0], io[1], REF)[1] + r_.normal(0, 10)) for io in R}, REF)["tier"] == "order sets the take-off"
check(f"power at noise sd 10: init {hits['init']}/50, order {hits['order']}/50 (need >= 40 each)", min(hits.values()) >= 40)

# --- step-1 gate: pass, pair-mismatch fail, dead-control fail, nan fail
base = {"S1007-ORIG": 1.2345, qx.arm(qx.BASE, 1007): 1.2345, "S1001-ORIG": 1.1111, qx.arm(1007, qx.BASE): 1.1111}
check("gate passes when pairs match and control differs", qx.step1_gate(base)["pass"])
bad = dict(base, **{qx.arm(qx.BASE, 1007): 1.30})
check("gate fails when a pair differs", not qx.step1_gate(bad)["pass"])
dead = {"S1007-ORIG": 1.2, qx.arm(qx.BASE, 1007): 1.2, "S1001-ORIG": 1.2, qx.arm(1007, qx.BASE): 1.2}
check("gate fails when control does not differ", not qx.step1_gate(dead)["pass"])
nan = dict(base, **{"S1007-ORIG": float("nan")})
check("gate fails on non-finite loss", not qx.step1_gate(nan)["pass"])

print(f"\n{'PASS' if all(ok) else 'FAIL'} rehearsal_qx {sum(ok)}/{len(ok)}")
raise SystemExit(0 if all(ok) else 1)
