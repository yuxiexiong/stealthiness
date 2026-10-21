"""Offline rehearsal (rule 6.2): the criterion must be satisfiable AND failable; plus Gram-trick check."""
import json, random, math, sys, glob
import numpy as np
sys.path.insert(0, '.')
from s3_common import verdict
grid = list(range(10, 205, 5))
def ar1(seed, sd=1.0, phi=0.6):
    r = random.Random(seed); x, out = 0.0, {}
    for t in grid:
        x = phi * x + r.gauss(0, sd); out[t] = x
    return out
def trend(t): return -0.002 * t
def make(plant6, plant7, base_seed):
    s = {}
    for i, k in enumerate(['a0_1006', 'a0_1007', 'a1_1006', 'a1_1007']):
        n = ar1(base_seed + i); s[k] = {t: trend(t) + n[t] for t in grid}
    # same-seed pairs share most noise in reality; give A1 the A0 noise plus a small independent part
    for seed, plant in (('1006', plant6), ('1007', plant7)):
        e = ar1(base_seed + 100 + int(seed[-1]), sd=0.3)
        s['a1_' + seed] = {t: s['a0_' + seed][t] + e[t] + (plant(t) if plant else 0.0) for t in grid}
    return s
ramp = lambda t: 0.0 if t < 50 else min(1.0, (t - 50) / 50) * 6.0    # ~3.4x the A0 cross-seed RMS: sensitivity probe
big = lambda t: 0.0 if t < 50 else min(1.0, (t - 50) / 50) * 10.0    # ~5.6x: must pass reliably
results = {}
fails = []
for name, p6, p7, want in [("planted_big", big, big, {"可辨识（强）"}), ("planted_both", ramp, ramp, {"可辨识（强）"}),
                           ("null", None, None, {"不可辨识", "分不清"}),
                           ("planted_one_seed", ramp, None, {"分不清"})]:
    tiers = [verdict(make(p6, p7, 1000 * k), grid)["tier"] for k in range(1, 201)]
    freq = {t: tiers.count(t) / len(tiers) for t in set(tiers)}
    results[name] = freq
    print(name, freq)
null_strong = results["null"].get("可辨识（强）", 0) + results["null"].get("可辨识（弱）", 0)
if results["planted_big"].get("可辨识（强）", 0) < 0.9: fails.append("criterion cannot be reliably satisfied on a clear planted signal")
if results["planted_both"].get("可辨识（强）", 0) + results["planted_both"].get("可辨识（弱）", 0) < 0.9: fails.append("borderline planted signal not at least weak")
if null_strong > 0.05: fails.append("null synthetic passes too often (%.3f)" % null_strong)
if results["planted_one_seed"].get("可辨识（强）", 0) + results["planted_one_seed"].get("可辨识（弱）", 0) > 0.05: fails.append("one-seed signal passes")
# positive control on real, already-observed data: trigger-margin gap must be identifiable
def load(p): return {r['step']: r for r in (json.loads(l) for l in open(p) if l.strip())}
C = '../curves/eqcity/'
real = {k: {t: v['trigger_margin_mean'] for t, v in load(C + f'nk_{k.split("_")[0]}_s{k.split("_")[1]}/field_curve.jsonl').items()} for k in ['a0_1006', 'a0_1007', 'a1_1006', 'a1_1007']}
v = verdict(real, grid); print("positive control (trigger margin, real):", v["tier"], [round(x, 2) for x in v["Z_window"]], "pre", [round(x, 2) for x in v["Z_pre"]])
if v["tier"] != "可辨识（强）": fails.append("positive control fails")
realc = {k: {t: v['clean_margin_mean'] for t, v in load(C + f'nk_{k.split("_")[0]}_s{k.split("_")[1]}/field_curve.jsonl').items()} for k in real}
vc = verdict(realc, grid); print("descriptive (clean-prompt target leakage, real):", vc["tier"], [round(x, 2) for x in vc["Z_window"]])
# Gram trick: <B1 A1, B2 A2>_F == sum((B1^T B2) * (A1 A2^T))
rng = np.random.default_rng(0); err = 0
for _ in range(20):
    A1, A2 = rng.normal(size=(16, 300)), rng.normal(size=(16, 300)); B1, B2 = rng.normal(size=(200, 16)), rng.normal(size=(200, 16))
    exact = float(np.sum((B1 @ A1) * (B2 @ A2))); trick = float(np.sum((B1.T @ B2) * (A1 @ A2.T)))
    err = max(err, abs(exact - trick) / abs(exact))
print("gram trick max rel err", err)
if err > 1e-9: fails.append("gram trick wrong")
print("REHEARSAL", "FAIL: " + "; ".join(fails) if fails else "PASS")
json.dump({"synthetic": results, "positive_control": {"tier": v["tier"], "Z": v["Z_window"]}, "leakage_descriptive": {"tier": vc["tier"], "Z": vc["Z_window"]}, "gram_rel_err": err, "pass": not fails, "fails": fails}, open("REHEARSAL.json", "w"), ensure_ascii=False, indent=1)
