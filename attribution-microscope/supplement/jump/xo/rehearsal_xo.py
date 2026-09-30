"""Offline rehearsal: every tier of the crossover verdict and the step-1 gate reachable."""
import numpy as np

import xo

ok = []


def check(name, cond):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name)


rng = np.random.default_rng(3)
R = xo.runs()
q = lambda x: int(np.ceil(x / 10) * 10)
v = xo.verdict({io: q(xo.predictions(*io)[0] + rng.normal(0, 6)) for io in R})
check("init world -> " + v["tier"], v["tier"] == "init sets the take-off")
v = xo.verdict({io: q(xo.predictions(*io)[1] + rng.normal(0, 6)) for io in R})
check("order world -> " + v["tier"], v["tier"] == "order sets the take-off")
v = xo.verdict({io: q(np.mean(xo.predictions(*io))) for io in R})
check(f"half-half world -> {v['tier']} (mi {v['mae_init']:.0f}, mo {v['mae_order']:.0f})", v["tier"] == "unclear (both or partial)")
v = xo.verdict({io: 450 if i % 2 else 330 for i, io in enumerate(R)})
check(f"far-from-both world -> {v['tier']} (mi {v['mae_init']:.0f} vs q50 {v['null_init_q50']:.0f}; mo {v['mae_order']:.0f} vs q50 {v['null_order_q50']:.0f})",
      v["tier"] == "neither (behaves like a fresh draw)")
v = xo.verdict({io: (360 if i < 5 else None) for i, io in enumerate(R)})
check("five runs -> " + v["tier"], v["tier"].startswith("cannot"))
hits = {"init": 0, "order": 0}
for s in range(50):
    r_ = np.random.default_rng(100 + s)
    hits["init"] += xo.verdict({io: q(xo.predictions(*io)[0] + r_.normal(0, 10)) for io in R})["tier"] == "init sets the take-off"
    hits["order"] += xo.verdict({io: q(xo.predictions(*io)[1] + r_.normal(0, 10)) for io in R})["tier"] == "order sets the take-off"
check(f"power at noise sd 10: init {hits['init']}/50, order {hits['order']}/50 (need >= 45 each)", min(hits.values()) >= 45)
g = xo.step1_gate({"S1007-ORIG": 1.2345, "XO-I1001-O1007": 1.2345, "S1001-ORIG": 1.1111, "XO-I1007-O1001": 1.1111})
check("gate passes when pairs match", g["pass"])
g = xo.step1_gate({"S1007-ORIG": 1.2345, "XO-I1001-O1007": 1.3000, "S1001-ORIG": 1.1111, "XO-I1007-O1001": 1.1111})
check("gate fails when a pair differs", not g["pass"])
g = xo.step1_gate({"S1007-ORIG": 1.2345, "XO-I1001-O1007": 1.2345, "S1001-ORIG": 1.2345, "XO-I1007-O1001": 1.2345})
check("gate fails when control does not differ (loss insensitive to batch)", not g["pass"])
g = xo.step1_gate({"S1007-ORIG": float("nan"), "XO-I1001-O1007": 1.0, "S1001-ORIG": 1.0, "XO-I1007-O1001": 1.0})
check("gate fails on a missing loss", not g["pass"])
src = xo.src_map(np.array([2, 0, 3, 1] + list(range(4, xo.N))), np.array([1, 3, 0, 2] + list(range(4, xo.N))))
check("src_map: position r_init[i] holds row r_order[i]", [src[2], src[0], src[3], src[1]] == [1, 3, 0, 2])
d = xo.decompose({(1007, 1001): 450, (1001, 1007): 360})
check(f"decompose init share 1, order share 0: {d[1007]['init_share']}, {d[1007]['order_share']}",
      d[1007]["init_share"] == 1 and d[1007]["order_share"] == 0 and d[1007]["additivity_residual"] == 0)
print(f"{sum(ok)}/{len(ok)}")
