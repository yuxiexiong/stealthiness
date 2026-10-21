"""Offline rehearsal for CONTRACT_SF.md: construction checks, every verdict tier reachable
and NOT reachable from the wrong world, power at two noise levels. numpy only."""
import numpy as np

import sf

ok = []


def check(name, cond):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name)


rng = np.random.default_rng(7)
# ---------------------------------------------------------------- construction
poison = set(rng.choice(sf.N, 200, replace=False).tolist())
rA, rB = rng.permutation(sf.N), rng.permutation(sf.N)
rAB = sf.recombine(rA, rB, poison)
c = sf.check_recombination(rAB, rA, rB, poison)
check(f"recombine: valid perm / poison pos+ids = skeleton / clean seq = filling -> {c}",
      c["valid_permutation"] and c["poison_positions_match_skeleton"] and c["poison_ids_match_skeleton"]
      and c["clean_subsequence_matches_filling"] and c["n_poison"] == 200)
check(f"clean rows drift by at most a batch or two (max {c['max_abs_clean_shift_positions']} positions = {c['max_abs_clean_shift_steps']} steps)",
      c["max_abs_clean_shift_steps"] <= 3)
cBA = sf.check_recombination(sf.recombine(rB, rA, poison), rB, rA, poison)
check("reverse recombination also passes", all(cBA[k] for k in ("valid_permutation", "poison_positions_match_skeleton",
                                                                     "poison_ids_match_skeleton", "clean_subsequence_matches_filling")))
bad = rAB.copy()
i, j = np.flatnonzero(np.isin(bad, list(poison)))[:2]
bad[i], bad[j] = bad[j], bad[i]
check("check catches two poison rows swapped (skeleton ids broken)",
      not sf.check_recombination(bad, rA, rB, poison)["poison_ids_match_skeleton"])
bad = rAB.copy()
k = np.flatnonzero(~np.isin(bad, list(poison)))[:2]
bad[k[0]], bad[k[1]] = bad[k[1]], bad[k[0]]
check("check catches two clean rows swapped (filling broken)",
      not sf.check_recombination(bad, rA, rB, poison)["clean_subsequence_matches_filling"])
check("recombine(A, A) == A", bool((sf.recombine(rA, rA, poison) == rA).all()))
src = sf.src_map(np.array([2, 0, 3, 1] + list(range(4, sf.N))), np.array([1, 3, 0, 2] + list(range(4, sf.N))))
check("src_map: position r_init[i] holds row r_order[i]", [src[2], src[0], src[3], src[1]] == [1, 3, 0, 2])

# ---------------------------------------------------------------- read-outs
check("t_m0 interpolates the zero crossing", abs(sf.t_m0([300, 310, 320], [-2.0, -1.0, 1.0]) - 315.0) < 1e-9)
check("t_m0 None when never crossing", sf.t_m0([300, 310], [-2.0, -1.0]) is None)
cur = [(320, 0.1), (330, 0.6), (340, 0.3), (350, 0.2), (360, 0.7), (370, 0.9), (380, 0.95), (390, 0.99), (400, 0.99)]
check("t50 = 330 (first >= 0.5) but sustained t50s = 360 on the PS-P0-like dip", sf.t50_10(cur) == 330 and sf.t50_sustained(cur) == 360)

# ---------------------------------------------------------------- verdict worlds
REFS = {1001: 330.0, 1007: 465.0, 1009: 328.0, 1008: 402.0}   # stand-ins; real 1001 t_M0 filled before freezing


def world(f, noise=0.0, r=None):
    """f(pair, A, B) -> (t_skelA_fillB, t_skelB_fillA) without noise."""
    r = r or np.random.default_rng(0)
    obs = {}
    for p, (a, b) in sf.PAIRS.items():
        t_ab, t_ba = f(p, REFS[a], REFS[b])
        obs[(p, a, b)] = t_ab + r.normal(0, noise)
        obs[(p, b, a)] = t_ba + r.normal(0, noise)
    return obs


v = sf.verdict(world(lambda p, ta, tb: (ta, tb)), REFS)
check("skeleton world -> " + v["tier"], v["tier"] == "skeleton sets the take-off")
v = sf.verdict(world(lambda p, ta, tb: (tb, ta)), REFS)
check("filling world -> " + v["tier"], v["tier"] == "filling sets the take-off")
v = sf.verdict(world(lambda p, ta, tb: ((ta + tb) / 2, (ta + tb) / 2)), REFS)
check("half-half world -> " + v["tier"], v["tier"] == "mixed")
v = sf.verdict(world(lambda p, ta, tb: (ta, ta)), REFS)      # both runs sit on A whatever the role
check(f"A-dominates world -> {v['tier']} (dominant {v['pairs']['P1'].get('dominant_parent')})",
      v["tier"].startswith("one parent") and v["pairs"]["P1"]["dominant_parent"] == 1001)
v = sf.verdict(world(lambda p, ta, tb: (tb, tb)), REFS)
check(f"B-dominates world -> dominant {v['pairs']['P1'].get('dominant_parent')}", v["pairs"]["P1"]["dominant_parent"] == 1007)
obs = world(lambda p, ta, tb: (ta, tb))
obs[("P2", 1009, 1007)] = None
v = sf.verdict(obs, REFS)
check("missing judged run -> " + v["tier"], v["tier"].startswith("cannot"))
v = sf.verdict(world(lambda p, ta, tb: (ta, tb) if p == "P1" else (tb, ta)), REFS)
check("P1 skeleton but P2 filling -> " + v["tier"], v["tier"].startswith("inconsistent"))
v = sf.verdict(world(lambda p, ta, tb: (ta, tb)), {**REFS, 1001: None})
check("reference 1001 missing -> P1 " + v["pairs"]["P1"]["tier"], v["pairs"]["P1"]["tier"].startswith("cannot") and v["tier"].startswith("cannot"))

# ---------------------------------------------------------------- power (judged pairs only, gap 135-137)
print("power over 2000 noisy worlds (judged pairs P1, P2; share noise = sd/gap):")
pw = {}
for sd in (10.0, 20.0, 30.0):
    hit = {"skel": 0, "fill": 0, "half": 0, "asym": 0}
    for s in range(2000):
        r = np.random.default_rng(1000 + s)
        hit["skel"] += sf.verdict(world(lambda p, ta, tb: (ta, tb), sd, r), REFS)["tier"] == "skeleton sets the take-off"
        hit["fill"] += sf.verdict(world(lambda p, ta, tb: (tb, ta), sd, r), REFS)["tier"] == "filling sets the take-off"
        hit["half"] += sf.verdict(world(lambda p, ta, tb: ((ta + tb) / 2,) * 2, sd, r), REFS)["tier"] == "mixed"
        hit["asym"] += sf.verdict(world(lambda p, ta, tb: (ta, ta), sd, r), REFS)["tier"].startswith("one parent")
    pw[sd] = {k: v / 2000 for k, v in hit.items()}
    print(f"  noise sd {sd:>4.0f} steps: " + ", ".join(f"{k} {v:.2f}" for k, v in pw[sd].items()))
check("power >= 0.90 for every world at sd 10 (PS: path noise is 0 on the 10-step grid)", min(pw[10.0].values()) >= 0.90)
check("at sd 30 (conservative init-noise band) the pure worlds are read correctly >= 0.8 of the time",
      min(pw[30.0]["skel"], pw[30.0]["fill"]) >= 0.8)
v = sf.verdict(world(lambda p, ta, tb: (ta, (ta + tb) / 2)), REFS)       # (1, 0.5): one run pure, one half
check("(1, 0.5) partial world is NOT read as skeleton -> " + v["pairs"]["P1"]["tier"],
      v["pairs"]["P1"]["tier"] == "mixed")
v = sf.verdict(world(lambda p, ta, tb: (ta - 0.3 * (ta - tb), tb + 0.1 * (ta - tb))), REFS)   # shares (0.7, 0.9)
check(f"(0.7, 0.9) is read as skeleton (mean 0.8, both >= 0.6) -> {v['pairs']['P1']['tier']} "
      f"[{v['pairs']['P1']['share_skel_AB']:.2f}, {v['pairs']['P1']['share_skel_BA']:.2f}]",
      v["pairs"]["P1"]["tier"] == "skeleton sets the take-off")
v = sf.verdict(world(lambda p, ta, tb: (ta - 0.45 * (ta - tb), tb + 0.05 * (ta - tb))), REFS)  # shares (0.55, 0.95)
check("(0.55, 0.95) is mixed (one run below 0.6) -> " + v["pairs"]["P1"]["tier"], v["pairs"]["P1"]["tier"] == "mixed")
# wrong-world leakage: a half-half world must not be read as a pure world
leak = 0
for s in range(2000):
    r = np.random.default_rng(5000 + s)
    t = sf.verdict(world(lambda p, ta, tb: ((ta + tb) / 2,) * 2, 30.0, r), REFS)["tier"]
    leak += t in ("skeleton sets the take-off", "filling sets the take-off")
check(f"half-half world read as a pure world at sd 30: {leak / 2000:.3f} (need <= 0.01)", leak / 2000 <= 0.01)

# ---------------------------------------------------------------- gates
g = sf.determinism_gate([4.7161, 4.1, 3.9], [4.7161, 4.1, 3.9, 3.8])
check("GD1 passes on identical prefix", g["pass"])
g = sf.determinism_gate([4.7161, 4.1, 3.9001], [4.7161, 4.1, 3.9, 3.8])
check("GD1 fails on a 1e-4 difference", not g["pass"])
g = sf.determinism_gate([], [4.7161])
check("GD1 fails on empty log", not g["pass"])
print(f"{sum(ok)}/{len(ok)}")
