"""Offline rehearsal for CONTRACT_LP.md: every tier reachable."""
import numpy as np

import lp

ok = []


def check(name, cond):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name)


check("gate pass", lp.gate([(True, True)] * 95 + [(True, False)] * 5)["pass"])
check("gate fail", not lp.gate([(True, True)] * 80 + [(True, False)] * 20)["pass"])
check("t_m0 interpolation", abs(lp.t_m0([300, 310], [-1.0, 1.0]) - 305) < 1e-9)
check("validity pass", lp.validity({f"r{i}": (360, 355) for i in range(14)})["tier"].startswith("margin"))
check("validity fail", lp.validity({f"r{i}": (360, 300 if i < 4 else 355) for i in range(14)})["tier"] == "does not stand in")

rng = np.random.default_rng(5)
probes = list(range(20))


def world(mem_gain, asr_mem, core_asr=0.05):
    runs = []
    for r in range(14):
        t50 = int(rng.choice([330, 360, 400, 450]))
        steps = lp.STEPS
        cons = {i: int(rng.integers(1, 1251)) for i in probes}
        seen = {}
        for s in steps:
            d = {}
            for i in probes:
                c = cons[i] <= s
                m = -3 + (mem_gain if c else 0) + rng.normal(0, 0.8) + 0.01 * (s - t50)
                a = bool(c and rng.random() < asr_mem)
                d[i] = (m, a)
            seen[s] = d
        runs.append({"t50": t50, "steps": steps, "seen": seen, "consumed": cons,
                     "core_asr": {s: core_asr for s in steps}})
    return runs


t = lp.seen_test(world(4.0, 0.9))
check(f"memorise world -> {t['tier']} (dM {t['dM']:.2f}, p {t['p']:.4f}, asr_c {t['asr_seen_consumed']:.2f})", t["tier"].startswith("memorise"))
t = lp.seen_test(world(1.5, 0.1))
check(f"ahead-not-memorised world -> {t['tier']}", t["tier"].startswith("seen probes ahead"))
t = lp.seen_test(world(0.0, 0.0))
check(f"no-difference world -> {t['tier']} (p {t['p']:.3f})", t["tier"].startswith("seen and unseen move together"))
hits = 0
for k in range(30):
    rng = np.random.default_rng(100 + k)
    hits += lp.seen_test(world(0.35, 0.0))["tier"] == "unclear"
check(f"'unclear' reachable on weak effect: {hits}/30 draws", hits >= 1)
t = lp.seen_test([{"t50": None, "steps": [], "seen": {}, "consumed": {}, "core_asr": {}}])
check("too few -> cannot measure", t["tier"].startswith("cannot"))
print(f"{sum(ok)}/{len(ok)}")
