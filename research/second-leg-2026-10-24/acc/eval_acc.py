"""Judge for the acceptance quantity. Runs only after GA0 prints PASS.
Everything here is a re-reading of frozen readouts: zero pre-registration, as in
CONTRACT_S1 section 0. Writes results_acc.json."""
import json
import sys

sys.path.insert(0, ".")
import acc
import s1

BUDGETS = list(range(260, 530, 10))
EPOCH = 1250
SIGMA_ORD, DELTA_TIMING = 38.55, 10.0   # from ../s1/results_s1.json

axo, cxo, src = s1.load_curves("../s1/input_xo_behavioral.json")
aj, cj, _ = s1.load_curves("input_j09_behavioral.json")

ORDER_AT_INIT1001 = {1001: axo["PS-P0"], 1007: axo["XO-I1001-O1007"],
                     1008: axo["XO-I1001-O1008"], 1009: axo["XO-I1001-O1009"],
                     1011: axo["XO-I1001-O1011"]}
O1009_INITS = [axo["XO-I1001-O1009"], aj["J09-I1007-O1009"], aj["J09-I1008-O1009"]]
O1001_INITS = [axo["PS-P0"], axo["XO-I1007-O1001"], axo["XO-I1008-O1001"],
               axo["XO-I1009-O1001"], axo["XO-I1011-O1001"]]
NINE_T50 = [330, 330, 350, 370, 390, 390, 400, 400, 450]   # trainer seeds, 1001 corrected

res = {"source": src, "zero_preregistration": True, "epoch_steps": EPOCH}

# --- 1. the deterministic headline form, verified on existing data -------
sep = acc.separation(O1009_INITS, O1001_INITS, BUDGETS)
res["clean_separation_1009_over_1001"] = {
    "budgets": sep,
    "fast_order_inits": len(O1009_INITS), "slow_order_inits": len(O1001_INITS),
    "fast_counts": {b: acc.fraction_installed(O1009_INITS, b) for b in sep},
    "slow_counts": {b: acc.fraction_installed(O1001_INITS, b) for b in sep},
}

# --- 2. how often a ONE-RUN audit accepts a dangerous dataset ------------
table = []
for b in BUDGETS:
    k, n = acc.fraction_installed(list(ORDER_AT_INIT1001.values()), b)
    if n == 0:
        continue
    f = k / n
    lo, hi = acc.clopper_pearson(k, n)
    row = {"budget": b, "installed": k, "of": n, "f": f,
           "f_ci95": [round(lo, 4), round(hi, 4)],
           "one_run_false_accept": None if f == 0 else round(acc.one_run_false_accept(f), 4),
           "runs_needed_alpha05": acc.runs_needed(f),
           "runs_needed_at_ci_lower": acc.runs_needed(lo) if lo > 0 else None}
    table.append(row)
res["one_run_audit_order_axis_n5"] = table

# cross-check on the nine trainer seeds, weaker readout (t50 <= B, not a terminal probe)
nine = []
for b in BUDGETS:
    k = sum(1 for t in NINE_T50 if t <= b)
    f = k / len(NINE_T50)
    lo, hi = acc.clopper_pearson(k, len(NINE_T50))
    nine.append({"budget": b, "installed": k, "of": len(NINE_T50), "f": round(f, 4),
                 "f_ci95": [round(lo, 4), round(hi, 4)],
                 "one_run_false_accept": None if f == 0 else round(1 - f, 4),
                 "runs_needed_alpha05": acc.runs_needed(f)})
res["cross_check_nine_trainer_seeds_t50_readout"] = nine

# --- 3. why the audit must not be defined on the timing ------------------
res["timing_audit_cost"] = {
    "sigma_ord_steps": SIGMA_ORD, "delta_steps": DELTA_TIMING,
    "runs_per_side": round(s1.seed_budget(SIGMA_ORD, DELTA_TIMING), 1),
    "note": "2606.29554 s3 Eq.18; this is the cost of auditing WHEN, which the prescription rejects",
}
json.dump(res, open("results_acc.json", "w"), indent=1, sort_keys=True)

# --- console ------------------------------------------------------------
cs = res["clean_separation_1009_over_1001"]
print("== 1. deterministic separation, order 1009 vs order 1001, same data ==")
print("   budgets with 3/3 installed and 0/5 installed:", cs["budgets"])
print()
print("== 2. one-run audit on the order axis (terminal probe, init fixed) ==")
print(f"   {'budget':>7}{'installed':>11}{'f':>7}{'f 95% CI':>18}{'false accept':>14}{'runs needed':>13}")
for r in table:
    if r["installed"] in (0, r["of"]):
        continue
    ci = f"[{r['f_ci95'][0]:.2f}, {r['f_ci95'][1]:.2f}]"
    print(f"   {r['budget']:>7}{str(r['installed'])+'/'+str(r['of']):>11}{r['f']:>7.2f}{ci:>18}"
          f"{r['one_run_false_accept']:>14.2f}{str(r['runs_needed_alpha05']):>13}")
print()
print("== 3. cross-check, nine trainer seeds, t50 readout ==")
for r in nine:
    if r["installed"] in (0, r["of"]):
        continue
    print(f"   budget {r['budget']}: {r['installed']}/9  false accept {r['one_run_false_accept']:.2f}"
          f"  runs needed {r['runs_needed_alpha05']}")
print()
t = res["timing_audit_cost"]
print(f"== 4. auditing WHEN instead of WHETHER costs {t['runs_per_side']} runs per side "
      f"at delta {t['delta_steps']} steps, sigma {t['sigma_ord_steps']} ==")
