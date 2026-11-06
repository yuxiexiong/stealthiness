"""Offline rehearsal for CONTRACT_M3T. No GPU. Uses the FROZEN rows.json (read-out D) as a stand-in for the shape of
the new ledger: a fake T read-out is made from it, with and without an injected effect, so that every tier is seen
reachable and unreachable on real-shaped contributions, and the reproduction gate is seen to pass and to fail.
    python rehearsal_m3t.py -> rehearsal_m3t.json ; exit 0 only if all checks pass"""
import copy
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m3t  # noqa: E402
p0 = m3t.p0
IN = os.path.join(HERE, "..", "phase0", "inputs")
old = p0.load(os.path.join(IN, "rows.json.gz")); seg = p0.load(os.path.join(IN, "segments.json"))
chk, info = {}, {}
rng = np.random.default_rng(3)

# a stand-in "new" file: same D, plus a T read-out = D + noise (NOT a result; shape only)
new = copy.deepcopy(old); new["X"] = ["full", "T"]
for r in new["rows"]:
    r["a"]["T"] = {j: v + 0.2 * abs(v) * rng.standard_normal() for j, v in r["a"]["full"].items()}
chk["reproduction_gate_passes_on_identical_D"] = m3t.reproduce(new, old)["pass"]
bad = copy.deepcopy(new)
for r in bad["rows"][:2000]:
    r["a"]["full"] = {j: v * 1.5 for j, v in r["a"]["full"].items()}
chk["reproduction_gate_fails_on_altered_D"] = m3t.reproduce(bad, old)["pass"] is False
cal = m3t.calib(seg, "full")
chk["calibration_reproduces_M3_1_pass_on_D"] = cal["pass"] and abs(cal["ledger_corr"] - 0.9968) < 2e-3
seg_bad = [dict(r, L_full=-r["L_full"]) for r in seg]
chk["calibration_can_fail"] = m3t.calib(seg_bad, "full")["pass"] is False

orig, s, g, loss, c = m3t.rows_table(new)
growth = cal["growth"]
info["n_clean_rows"] = int(len(orig)); info["growth_D"] = growth
q = m3t.top_quartile(loss, g)
info["top_quartile_cell"] = int(q.sum())
chk["top_quartile_is_a_quarter"] = abs(q.mean() - 0.25) < 0.01
# difficulty test, tiers reachable on real-shaped data with RANDOM labels of the same size per segment
def rand_like(lab, seed):
    r = np.random.default_rng(seed); out = np.zeros(len(lab), bool)
    for gg in np.unique(g):
        m = np.flatnonzero(g == gg); k = int(lab[m].sum())
        if k:
            out[r.choice(m, k, replace=False)] = True
    return out
fa, strong, small = [], [], []
for i in range(100):
    lab = rand_like(q, 500 + i)
    fa.append(m3t.tier(p0.strat_perm(c["T"], lab, g, 2000, seed=i), growth, -1))
    strong.append(m3t.tier(p0.strat_perm(c["T"] - lab * (2 * m3t.SHARE * growth / lab.sum()), lab, g, 2000, seed=i), growth, -1))
    small.append(m3t.tier(p0.strat_perm(c["T"] - lab * (0.3 * m3t.SHARE * growth / lab.sum()), lab, g, 2000, seed=i), growth, -1))
info["difficulty_false_alarm"] = {t: fa.count(t) for t in set(fa)}
info["difficulty_at_20pct_share"] = {t: strong.count(t) for t in set(strong)}
info["difficulty_at_3pct_share"] = {t: small.count(t) for t in set(small)}
chk["difficulty_false_strong_le_3pct"] = sum(t.startswith(("成立", "反向")) for t in fa) <= 3
chk["difficulty_strong_tier_at_20pct_ge_80pct"] = strong.count("成立（有分量）") >= 80
chk["difficulty_never_strong_tier_at_3pct"] = small.count("成立（有分量）") == 0
lab = rand_like(q, 1)
chk["reverse_tier_reachable"] = m3t.tier(p0.strat_perm(c["T"] + lab * (2 * m3t.SHARE * growth / lab.sum()), lab, g, 2000), growth, -1) == "反向（有分量）"
tiny = np.zeros(len(g), bool); tiny[:10] = True
chk["untestable_tier_reachable"] = m3t.tier(p0.strat_perm(c["T"], tiny, g, 500), growth, -1).startswith("测不了")
# cue cell on the stand-in (labels are real, outcome is the stand-in: structure only)
op = p0.load(os.path.join(IN, "orders_poison.json.gz")); meta = p0.load(os.path.join(IN, "train_meta.json.gz"))
cue, _, _ = p0.cue_labels(np.asarray(op["orders"]["1009"]), op["poison"], meta)
chk["cue_cell_size_47"] = int(cue["ANY_after"][orig].sum()) == 47
tt = m3t.totals(new, "full")
chk["totals_match_phase0"] = abs(tt["clean_pos"] - 5.057861783873991) < 1e-9 and abs(tt["clean_neg"] + 3.418359456976629) < 1e-9
chk = {k: bool(v) for k, v in chk.items()}
ok = all(chk.values())
json.dump({"pass": ok, "checks": chk, "info": info}, open(os.path.join(HERE, "rehearsal_m3t.json"), "w"), indent=1, ensure_ascii=False)
for k, v in chk.items():
    print(("PASS " if v else "FAIL ") + k)
print(json.dumps(info, ensure_ascii=False))
print("REHEARSAL", "PASS" if ok else "FAIL", f"{sum(chk.values())}/{len(chk)}"); sys.exit(0 if ok else 1)
