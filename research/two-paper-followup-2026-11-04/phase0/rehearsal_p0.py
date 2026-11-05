"""Offline rehearsal for CONTRACT_P0 (rule 6.2): every verdict tier is shown reachable on synthetic data,
then the pipeline is run on the REAL input files with the hypothesis-bearing labels replaced
(random cue labels for Z3a; minimal-perturbation pairs instead of recombination pairs for Z2),
so the gates are seen to pass on real shapes without reading any contract outcome.

    python rehearsal_p0.py  ->  rehearsal_p0.json ; exit code 0 only if every check passes"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p0  # noqa: E402

IN = os.path.join(HERE, "inputs")
chk, info = {}, {}
rng = np.random.default_rng(7)

# ---------------- Z1 on toy sequences
P = set(range(0, p0.N, 100))                                   # 200 fake poison rows
a = rng.permutation(p0.N); b = p0.recombine(a, rng.permutation(p0.N), P)
same = p0.z1_pair(a, a, P); diff = p0.z1_pair(a, b, P)
chk["z1_identical_sequences_overlap_1"] = same["set"]["328"]["jaccard"] == 1.0 and same["batchmates_shared_frac_step"] == 1.0
chk["z1_recombined_keeps_skeleton"] = diff["same_skeleton"] is True
chk["z1_random_fill_overlap_near_T_over_N"] = abs(diff["set"]["328"]["overlap_frac_of_a"] - 328 / 1250) < 0.03
chk["z1_other_skeleton_flagged"] = p0.z1_pair(a, rng.permutation(p0.N), P)["same_skeleton"] is False

# ---------------- Z2 verdict, three tiers
dt = [13.0, -61.0, -55.0, -59.0, 29.0, -22.0]
t_blind = p0.z2_verdict([0.001, -0.002, 0.001, 0.0, -0.001, 0.002], 0.01, dt)["tier"]
t_keep = p0.z2_verdict([0.05, -0.20, -0.12, -0.15, 0.09, -0.06], 0.01, dt)["tier"]
t_none = p0.z2_verdict([-0.05, -0.06, 0.15, -0.12, 0.09, 0.20], 0.01, dt)["tier"]
chk["z2_blind_reachable"] = t_blind.startswith("测不了")
chk["z2_keep_reachable"] = t_keep.startswith("候选保留")
chk["z2_none_reachable"] = t_none == "没拿到结果"
rho, p = p0.spearman_exact_p([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 6, 5])
chk["spearman_exact_matches_table"] = abs(rho - 0.942857) < 1e-5 and abs(p - 0.016667) < 1e-4   # 12/720

# ---------------- Z3a verdict on synthetic heavy-tailed rows
def synth(n=5000, n_cue=45, shift=0.0, seed=0):
    r = np.random.default_rng(seed)
    g = np.repeat(np.arange(1, 34), n // 33 + 1)[:n]
    c = r.standard_t(3, size=n) * 1e-3 * (1 + g / 10)          # later segments have larger contributions
    lab = np.zeros(n, bool); lab[r.choice(n, n_cue, replace=False)] = True
    c = c + lab * shift
    return c, lab, g
dD = 4.5
tiers = {}
c, lab, g = synth(shift=0.0, seed=1); tiers["null"] = p0.z3_verdict(p0.strat_perm(c, lab, g, 4000), dD)
c, lab, g = synth(shift=0.02, seed=2); tiers["big_pos"] = p0.z3_verdict(p0.strat_perm(c, lab, g, 4000), dD)
c, lab, g = synth(shift=0.004, seed=3); tiers["small_pos"] = p0.z3_verdict(p0.strat_perm(c, lab, g, 4000), dD)
c, lab, g = synth(shift=-0.02, seed=4); tiers["big_neg"] = p0.z3_verdict(p0.strat_perm(c, lab, g, 4000), dD)
c, lab, g = synth(n_cue=12, shift=0.02, seed=5); tiers["tiny_cell"] = p0.z3_verdict(p0.strat_perm(c, lab, g, 4000), dD)
info["z3_synth_tiers"] = tiers
chk["z3_null_not_positive"] = tiers["null"] in ("没拿到结果", "弱线索")
chk["z3_strong_reachable"] = tiers["big_pos"] == "成立（有分量）"
chk["z3_small_reachable"] = tiers["small_pos"] == "成立（统计上有、分量小）"
chk["z3_reverse_reachable"] = tiers["big_neg"] == "反向（有分量）"
chk["z3_untestable_reachable"] = tiers["tiny_cell"].startswith("测不了")
fp = [p0.z3_verdict(p0.strat_perm(*synth(shift=0.0, seed=100 + i), 2000, seed=i), dD) for i in range(200)]
info["z3_synth_false_alarm"] = {t: fp.count(t) for t in set(fp)}
chk["z3_synth_false_strong_rate_le_3pct"] = sum(t.startswith(("成立", "反向")) for t in fp) <= 6

# ---------------- real input files, hypothesis labels replaced
op = p0.load(os.path.join(IN, "orders_poison.json.gz")); meta = p0.load(os.path.join(IN, "train_meta.json.gz"))
rows = p0.load(os.path.join(IN, "rows.json.gz")); seg = p0.load(os.path.join(IN, "segments.json"))
beh = p0.load(os.path.join(IN, "behavioral_260_300.json"))
res_sf = p0.load(os.path.join(IN, "results_sf.json")); refs = p0.load(os.path.join(IN, "refs_tm0.json"))
TK = p0.takeoff(res_sf, refs)
poison = op["poison"]; seqs = p0.sequences(op["orders"], poison)
o9 = seqs["O1009"]
chk["real_ledger_rows_match_order1009"] = all(int(o9[r["k"]]) == r["orig"] and r["k"] // 16 + 1 == r["s"] for r in rows["rows"])
chk["real_ledger_poison_flags"] = all((r["orig"] in set(poison)) == r["poison"] for r in rows["rows"])
chk["real_all_sf_arms_keep_skeleton"] = all(p0.z1_pair(seqs[f"O{sk}"], seqs[arm], poison, Ts=(328,))["same_skeleton"] for arm, (sk, _) in p0.SF_ARMS.items())
chk["real_takeoff_table_complete"] = all(k in TK for k in list(p0.SF_ARMS) + ["O1001", "O1007", "O1009"])
chk["real_meta_matches_rows"] = len(meta) == p0.N and all(m["idx"] == i for i, m in enumerate(meta))
orig, s, g, c = p0.ledger_rows(rows, "full")
dD_real = p0.measured_growth(seg, "full")
info["real_ledger"] = {"n_clean_rows": int(len(c)), "window": [int(s.min()), int(s.max())], "dD_full_measured": dD_real}
chk["real_ledger_window"] = int(s.min()) == p0.SEG0 and int(s.max()) == p0.T_LEDGER and dD_real > 2
lab_real, step9, _ = p0.cue_labels(o9, poison, meta)
n_primary = int(lab_real["ANY_after"][orig].sum())
info["real_cell_sizes_predictors_only"] = {k: int(v[orig].sum()) for k, v in lab_real.items()}
chk["real_primary_cell_ge_20"] = n_primary >= p0.Z3_MIN_CELL
# random labels of the same size on the real contributions: false-alarm rate, detection at the 10% share
# (an effect injected exactly AT the share threshold lands on either side of it about half the time, so the
# "有分量" tier is checked at twice the threshold; first draft of this check asked for it at the threshold itself)
fa, det, pw = [], [], []
for i in range(200):
    r = np.random.default_rng(1000 + i)
    lab = np.zeros(len(c), bool); lab[r.choice(len(c), n_primary, replace=False)] = True
    fa.append(p0.z3_verdict(p0.strat_perm(c, lab, g, 2000, seed=i), dD_real))
    det.append(p0.z3_verdict(p0.strat_perm(c + lab * (p0.Z3_SHARE * dD_real / n_primary), lab, g, 2000, seed=i), dD_real))
    pw.append(p0.z3_verdict(p0.strat_perm(c + lab * (2 * p0.Z3_SHARE * dD_real / n_primary), lab, g, 2000, seed=i), dD_real))
info["z3_real_shape_false_alarm"] = {t: fa.count(t) for t in set(fa)}
info["z3_real_shape_at_10pct_share"] = {t: det.count(t) for t in set(det)}
info["z3_real_shape_at_20pct_share"] = {t: pw.count(t) for t in set(pw)}
chk["z3_real_shape_false_strong_le_3pct"] = sum(t.startswith(("成立", "反向")) for t in fa) <= 6
chk["z3_real_shape_detects_10pct_share_ge_95pct"] = sum(t.startswith("成立") for t in det) >= 190
chk["z3_real_shape_tier_strong_at_20pct_share_ge_80pct"] = pw.count("成立（有分量）") >= 160
# Z2 pipeline on real files with minimal-perturbation pairs standing in for recombination pairs
null = []
import itertools
for grp in p0.NULL_GROUPS.values():
    null += [p0.paired_diff(beh, x, y) for x, y in itertools.combinations(grp, 2)]
sigma = float(np.sqrt(np.mean(np.square(null))))
mini = [p0.paired_diff(beh, m, "PS-P0") for m in p0.MINIMAL] + null[:2]
info["z2_real_null"] = {"n_pairs": len(null), "sigma_null": sigma, "minimal_perturbation_dA": mini[:4]}
chk["z2_real_null_has_13_pairs"] = len(null) == 13 and sigma > 0
chk["z2_real_minimal_pairs_read_blind"] = p0.z2_verdict(mini, sigma, [0, 0, 0, 0, 1, 2])["tier"].startswith("测不了")
chk["z2_real_all_probe_vectors_200"] = all(len(beh[a][st]["acc"]) == 200 for a in beh for st in p0.Z2_STEPS)

ok = all(chk.values())
json.dump({"pass": ok, "checks": chk, "info": info}, open(os.path.join(HERE, "rehearsal_p0.json"), "w"), indent=1, ensure_ascii=False)
for k, v in chk.items():
    print(("PASS " if v else "FAIL ") + k)
print(json.dumps(info, ensure_ascii=False, indent=1))
print("REHEARSAL", "PASS" if ok else "FAIL", f"{sum(chk.values())}/{len(chk)}")
sys.exit(0 if ok else 1)
