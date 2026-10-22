"""Zero-GPU rehearsal for CONTRACT_K4:  python rehearsal_k4.py <scratch_dir_with_old_rows>
(1) every tier reachable on synthetic windows and the adjacent world does not read as it; (2) the formation
rule reproduces 180/185 on the old field curves; (3) the pipeline runs end to end on the OLD rows (snapshots
only to 205, so the form+50 window is truncated: numbers printed for plumbing only, not a read-out)."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import k4
D = sys.argv[1]
checks = []


def ck(name, cond):
    checks.append((name, bool(cond))); print(("PASS " if cond else "FAIL ") + name)


rng = np.random.default_rng(1)


def synth(n_pos, sep, f=180, n_neg=800):
    rows = [{"step": int(rng.integers(f, f + 51)), "poison": False, "c_mem_x": float(rng.normal(0, 1))} for _ in range(n_neg)]
    rows += [{"step": int(rng.integers(f, f + 51)), "poison": True, "c_mem_x": float(rng.normal(sep, 1))} for _ in range(n_pos)]
    rows += [{"step": int(rng.integers(f - 50, f)), "poison": bool(i < n_pos), "c_mem_x": float(rng.normal(0, 1))} for i in range(n_neg)]
    return rows


a0 = [{"step": s, "poison": False, "c_mem_x": float(rng.normal(0, 1))} for s in rng.integers(100, 260, 1600)]
r = k4.seed_readout(synth(8, 3.0), a0, a0, 180)
ck(f"separated world passes (auroc_in {r['form+50']['auroc_in']:.2f} > perm q975 {r['form+50']['perm_in']['q975']:.2f})", r["tier"] == "pass")
r = k4.seed_readout(synth(8, 0.0), a0, a0, 180)
ck(f"null world closes (auroc_in {r['form+50']['auroc_in']:.2f})", r["tier"] == "close")
r = k4.seed_readout(synth(8, 1.0), a0, a0, 180)
ck(f"weak world is indeterminate (auroc_in {r['form+50']['auroc_in']:.2f})", r["tier"] == "indeterminate")
r = k4.seed_readout(synth(2, 5.0), a0, a0, 180)
ck("fewer than 3 poison rows in the window cannot measure", r["tier"] == "cannot measure")
ck("no formation cannot measure", k4.seed_readout(synth(8, 3.0), a0, a0, None)["tier"].startswith("cannot"))
# a high AUROC that does not beat its own permutation null must not pass: 3 poison rows, perfect order
rows = [{"step": 190, "poison": False, "c_mem_x": float(v)} for v in rng.normal(0, 1, 20)] + [{"step": 190, "poison": True, "c_mem_x": 9.0 + i} for i in range(3)]
r = k4.seed_readout(rows, a0, a0, 180)
ck(f"3 perfect rows: auroc 1.0 but perm q975 {r['form+50']['perm_in']['q975']:.2f} -> {r['form+50']['perm_in']['q975'] >= 1.0 and 'not pass' or 'pass'}",
   (r["tier"] == "pass") == (1.0 > r["form+50"]["perm_in"]["q975"]))
ck("overall: both pass -> alive", "alive" in k4.overall({1006: "pass", 1007: "pass"}))
ck("overall: both close -> closed", "closed" in k4.overall({1006: "close", 1007: "close"}))
ck("overall: one pass one close -> indeterminate", "indeterminate" in k4.overall({1006: "pass", 1007: "close"}))
ck("overall: any cannot -> cannot", k4.overall({1006: "pass", 1007: "cannot measure"}) == "cannot measure")
ck("auroc handles ties (average ranks)", abs(k4.auroc([1, 1], [1, 1]) - 0.5) < 1e-12 and k4.auroc([2], [1]) == 1.0 and k4.auroc([0], [1]) == 0.0)
# formation rule on the old curves
for s in k4.SEEDS:
    c = k4.jl(os.path.join(D, f"20261022_disperse_r01_eqcity_k4_runs_d_k4_a1_s{s}_field_curve.jsonl"))
    f = k4.formation(c); ck(f"formation rule reproduces {k4.EXPECTED_FORMATION[s]} for s{s} (got {f})", f == k4.EXPECTED_FORMATION[s])
# plumbing on the old rows (truncated window: 180..205 only; numbers are NOT a read-out)
old = {s: k4.jl(os.path.join(D, f"20261022_disperse_r01_gradtrig_rows_k4_s{s}.jsonl")) for s in k4.SEEDS}
a0old = {s: k4.jl(os.path.join(D, f"20261021_newknow_r01_gradtrig_rows_a0_s{s}.jsonl")) for s in k4.SEEDS}
for s, o in zip(k4.SEEDS, reversed(k4.SEEDS)):
    r = k4.seed_readout(old[s], a0old[s], a0old[o], k4.EXPECTED_FORMATION[s])
    w = r["form+50"]
    print(f"  plumbing s{s}: window {w['lo']}-{w['hi']} truncated at 205 -> n_pos {w['n_pos']}, auroc_in {w['auroc_in']}, a0_same {w['auroc_a0_same']}, a0_other {w['auroc_a0_other']}, pre auroc_in {r['pre']['auroc_in']}")
    ck(f"pipeline runs on old rows s{s} (n_pos {w['n_pos']} within 205)", w["n_pos"] >= 1 and w["auroc_in"] is not None)
n_ok = sum(1 for _, c in checks if c); print(f"\nrehearsal_k4: {n_ok}/{len(checks)}")
json.dump({"checks": checks}, open(os.path.join(HERE, "rehearsal_k4.json"), "w"), indent=1)
sys.exit(0 if n_ok == len(checks) else 1)
