"""Offline rehearsal for CONTRACT_SF.md: construction checks, every verdict tier reachable
and NOT reachable from the wrong world, power at two noise levels, random-filling read-out,
reference loading, GR0, GD1 comparisons. numpy only.   python rehearsal_sf.py"""
import os

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
      all(c[k] for k in sf.STRUCT_KEYS) and c["n_poison"] == 200)
check(f"clean rows drift by at most a batch or two (max {c['max_abs_clean_shift_positions']} positions = {c['max_abs_clean_shift_steps']} steps)",
      c["max_abs_clean_shift_steps"] <= 3)
cBA = sf.check_recombination(sf.recombine(rB, rA, poison), rB, rA, poison)
check("reverse recombination also passes", all(cBA[k] for k in sf.STRUCT_KEYS))
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
check(f"runs(): 4 judged + 2 random-filling arms, queue P1, P2, R -> {[sf.arm(*r) for r in sf.runs()]}",
      len(sf.runs()) == 6 and len(sf.judged_runs()) == 4 and [r[0] for r in sf.runs()] == ["P1", "P1", "P2", "P2", "R", "R"])
check(f"orders_needed() = {sf.orders_needed()}", sf.orders_needed() == [1001, 1007, 1009, 2001])

# ---------------------------------------------------------------- read-outs
check("t_m0 interpolates the zero crossing", abs(sf.t_m0([300, 310, 320], [-2.0, -1.0, 1.0]) - 315.0) < 1e-9)
check("t_m0 None when never crossing", sf.t_m0([300, 310], [-2.0, -1.0]) is None)
st = [300, 310, 320, 330, 340, 350, 360, 370, 380]
md = [-3.0, -1.0, 1.0, -0.5, -0.2, 0.4, 1.0, 2.0, 3.0]
check("PS-P0-like dip: first crossing 315, sustained crossing 343.3 (margin back below 0 after the first)",
      abs(sf.t_m0(st, md) - 315.0) < 1e-9 and abs(sf.t_m0_sustained(st, md) - (340 + 10 * 0.2 / 0.6)) < 1e-9)
check("monotone curve: first == sustained", sf.t_m0(st[:3], md[:3]) == sf.t_m0_sustained(st[:3], md[:3]))
cur = [(320, 0.1), (330, 0.6), (340, 0.3), (350, 0.2), (360, 0.7), (370, 0.9), (380, 0.95), (390, 0.99), (400, 0.99)]
check("t50 = 330 (first >= 0.5) but sustained t50s = 360 on the PS-P0-like dip", sf.t50_10(cur) == 330 and sf.t50_sustained(cur) == 360)
recs = {300: {"core": [{"idx": i, "M": -1.0, "argmax_is_target": False} for i in range(5)]},
        340: {"core": [{"idx": i, "M": 1.0, "argmax_is_target": True} for i in range(5)]}}
check("median_curve reads the per-checkpoint files", sf.median_curve(recs) == ([300, 340], [-1.0, 1.0]))
g = sf.gr0(recs, {300: {i: False for i in range(5)}, 340: {i: True for i in range(5)}})
check(f"GR0 passes on full agreement -> {g}", g["pass"] and g["n"] == 10)
g = sf.gr0(recs, {300: {i: False for i in range(5)}, 340: {i: i < 2 for i in range(5)}})
check(f"GR0 fails at 0.7 agreement -> {g}", not g["pass"] and abs(g["agree"] - 0.7) < 1e-9)
check("GR0 with no overlapping checkpoints -> fail, n 0", not sf.gr0(recs, {})["pass"])

# ---------------------------------------------------------------- references
here = os.path.dirname(os.path.abspath(__file__))
R = sf.load_refs(os.path.join(here, "refs_tm0.json"))
check("refs_tm0.json loads four read-outs with int order keys and the 1001 value filled",
      set(R) == set(sf.REF_KEYS) and all(set(R[k]) == {1001, 1007, 1008, 1009} for k in R) and R["first"][1001] is not None)
check("t50 / t50s references match the constants; only 1001 differs between them (330 vs 360)",
      R["t50"] == sf.REF_T50 and R["t50s"] == sf.REF_T50S and sf.REF_T50S[1001] == 360
      and all(sf.REF_T50[o] == sf.REF_T50S[o] for o in (1007, 1008, 1009)))
gaps = {k: {p: abs(R[k][a] - R[k][b]) for p, (a, b) in sf.PAIRS.items()} for k in sf.REF_KEYS}
check(f"gap rule: P2 clears 119 on all four read-outs; P1 only on first / t50, short on sustained (109.3) and t50s (110) -> {gaps}",
      all(gaps[k]["P2"] >= 119 for k in sf.REF_KEYS) and gaps["first"]["P1"] >= 119 and gaps["t50"]["P1"] >= 119
      and gaps["sustained"]["P1"] < 119 and gaps["t50s"]["P1"] < 119)
try:
    sf.load_refs(os.path.join(here, "results_lp_absent.json"))
    check("load_refs on a missing file raises", False)
except FileNotFoundError:
    check("load_refs on a missing file raises", True)

# ---------------------------------------------------------------- verdict worlds
REFS = {1001: 330.0, 1007: 465.0, 1009: 328.0, 1008: 402.0}   # stand-ins with the real gaps


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
check("skeleton world -> " + v["tier"], v["tier"] == "skeleton sets the take-off" and v["primary_pair_tier"] == v["tier"])
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
check("the two pure worlds are read correctly >= 0.98 at sd 10 (PS: path noise is 0 on the 10-step grid)",
      min(pw[10.0]["skel"], pw[10.0]["fill"]) >= 0.98)
# Known cost of adding the leaning tiers (false-negative audit): a TRUE 0.5/0.5 world drifts into a
# leaning tier under noise. The leaning tiers are explicitly labelled "partial", and the STRONG tiers
# stay protected (checked below), so this is the intended trade, but it must be stated, not hidden.
print("  true-half world read as a leaning tier: " + ", ".join(
    f"sd {sd:.0f}: {1 - pw[sd]['half']:.2f}" for sd in (10.0, 20.0, 30.0)))
wrong = 0
for s in range(2000):
    r = np.random.default_rng(7000 + s)
    for f, bad in ((lambda p, ta, tb: (ta, tb), "filling sets the take-off"),
                   (lambda p, ta, tb: (tb, ta), "skeleton sets the take-off")):
        wrong += sf.verdict(world(f, 30.0, r), REFS)["tier"] == bad
check(f"a pure world read as the OPPOSITE strong tier at sd 30: {wrong / 4000:.4f} (need 0)", wrong == 0)
check("at sd 30 (conservative init-noise band) the pure worlds are read correctly >= 0.8 of the time",
      min(pw[30.0]["skel"], pw[30.0]["fill"]) >= 0.8)
v = sf.verdict(world(lambda p, ta, tb: (ta, (ta + tb) / 2)), REFS)       # (1, 0.5): one run pure, one half
check("(1, 0.5) is NOT read as the strong skeleton tier -> " + v["pairs"]["P1"]["tier"],
      v["pairs"]["P1"]["tier"] == "skeleton-leaning (partial)")
v = sf.verdict(world(lambda p, ta, tb: (ta - 0.3 * (ta - tb), tb + 0.1 * (ta - tb))), REFS)   # shares (0.7, 0.9)
check(f"(0.7, 0.9) is read as skeleton (mean 0.8, both >= 0.6) -> {v['pairs']['P1']['tier']} "
      f"[{v['pairs']['P1']['share_skel_AB']:.2f}, {v['pairs']['P1']['share_skel_BA']:.2f}]",
      v["pairs"]["P1"]["tier"] == "skeleton sets the take-off")
v = sf.verdict(world(lambda p, ta, tb: (ta - 0.45 * (ta - tb), tb + 0.05 * (ta - tb))), REFS)  # shares (0.55, 0.95)
check("(0.55, 0.95) is skeleton-leaning, not 'skeleton' -> " + v["pairs"]["P1"]["tier"],
      v["pairs"]["P1"]["tier"] == "skeleton-leaning (partial)")
# the dead zone the false-negative audit found: a genuine 0.7 / 0.7 split must not print as "no finding"
v = sf.verdict(world(lambda p, ta, tb: (tb + 0.7 * (ta - tb), ta + 0.7 * (tb - ta))), REFS)
check("(0.70, 0.70) is skeleton-leaning, not mixed -> " + v["pairs"]["P1"]["tier"],
      v["pairs"]["P1"]["tier"] == "skeleton-leaning (partial)")
v = sf.verdict(world(lambda p, ta, tb: (tb + 0.3 * (ta - tb), ta + 0.3 * (tb - ta))), REFS)
check("(0.30, 0.30) is filling-leaning -> " + v["pairs"]["P1"]["tier"],
      v["pairs"]["P1"]["tier"] == "filling-leaning (partial)")
v = sf.verdict(world(lambda p, ta, tb: ((ta + tb) / 2, (ta + tb) / 2)), REFS)
check("exact 0.5 / 0.5 is still mixed -> " + v["pairs"]["P1"]["tier"], v["pairs"]["P1"]["tier"] == "mixed")
# wrong-world leakage: a half-half world must not be read as a pure world
leak = 0
for s in range(2000):
    r = np.random.default_rng(5000 + s)
    t = sf.verdict(world(lambda p, ta, tb: ((ta + tb) / 2,) * 2, 30.0, r), REFS)["tier"]
    leak += t in ("skeleton sets the take-off", "filling sets the take-off")
check(f"half-half world read as a pure world at sd 30: {leak / 2000:.3f} (need <= 0.01)", leak / 2000 <= 0.01)

# ---------------------------------------------------------------- random-filling arms (descriptive)
RK = {("R", 1007, sf.RFILL): None, ("R", 1009, sf.RFILL): None}
v = sf.rfill_readout({("R", 1007, sf.RFILL): 465.0 + 12, ("R", 1009, sf.RFILL): 328.0 - 9}, REFS)
check("rfill skeleton world (each within 30 of its skeleton parent) -> " + v["tier"],
      v["tier"].startswith("skeleton result holds"))
v = sf.rfill_readout({("R", 1007, sf.RFILL): 390.0, ("R", 1009, sf.RFILL): 396.0}, REFS)
check("rfill filling world (both land together, far from both skeletons) -> " + v["tier"],
      v["tier"].startswith("both arms land together"))
v = sf.rfill_readout({("R", 1007, sf.RFILL): 465.0, ("R", 1009, sf.RFILL): 400.0}, REFS)
check("rfill one follows, one departs -> " + v["tier"], v["tier"] == "other")
v = sf.rfill_readout({("R", 1007, sf.RFILL): 465.0 + 31, ("R", 1009, sf.RFILL): 328.0}, REFS)
check("rfill: 31 steps off is 'departs' (tolerance is the 30-step noise band)", v["tier"] == "other"
      and v["arms"][sf.arm("R", 1007, sf.RFILL)]["tier"].startswith("departs"))
v = sf.rfill_readout({("R", 1007, sf.RFILL): None, ("R", 1009, sf.RFILL): 328.0}, REFS)
check("rfill missing arm -> cannot measure", v["tier"] == "cannot measure")
check("rfill arms never enter verdict(): judged obs alone give a full tier",
      sf.verdict(world(lambda p, ta, tb: (ta, tb)), REFS)["tier"] == "skeleton sets the take-off")

# ---------------------------------------------------------------- gates
ref = {20: 3.2778, 40: 0.4671, 60: 0.41, 80: 0.39, 100: 0.38}
g = sf.determinism_gate(dict(ref), ref)
check("GD1 losses pass on identical 20..100 logs", g["pass"] and g["n"] == 5)
g = sf.determinism_gate({**ref, 100: 0.3801}, ref)
check("GD1 losses fail on a 1e-4 difference", not g["pass"])
g = sf.determinism_gate({20: 3.2778}, ref)
check(f"GD1 losses fail when steps are missing -> {g['missing_steps']}", not g["pass"] and g["missing_steps"] == [40, 60, 80, 100])
g = sf.determinism_gate({1: 4.7161, **ref}, {**ref, 1: 4.7161})
check("GD1 losses compare only the 20-step points (step-1 is the separate one-step gate)", g["pass"] and g["n"] == 5)
v = sf.gd1_verdict(4.7161, sf.determinism_gate(dict(ref), ref), {"same_keys": True, "max_abs_diff": 0.0})
check("GD1 passes when step-1 loss, logged losses and checkpoint-100 tensors all match", v["pass"])
v = sf.gd1_verdict(4.7161, sf.determinism_gate(dict(ref), ref), {"same_keys": True, "max_abs_diff": 1e-7})
check("GD1 fails on any tensor difference", not v["pass"] and v["step1"]["pass"])
v = sf.gd1_verdict(4.7163, sf.determinism_gate(dict(ref), ref), {"same_keys": True, "max_abs_diff": 0.0})
check("GD1 fails on step-1 loss 2e-4 off", not v["pass"])
v = sf.gd1_verdict(None, sf.determinism_gate(dict(ref), ref), {"same_keys": False, "max_abs_diff": None})
check("GD1 fails when the one-step run produced no loss or the key sets differ", not v["pass"])
print(f"{sum(ok)}/{len(ok)}")
raise SystemExit(0 if all(ok) else 1)
