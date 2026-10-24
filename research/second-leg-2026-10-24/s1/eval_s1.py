"""CONTRACT_S1 judge. Runs only after GS1-0 prints PASS. Writes results_s1.json.
Every number here is a re-reading of frozen readouts; nothing is pre-registered
(see CONTRACT_S1 section 0)."""
import json
import sys

sys.path.insert(0, ".")
import s1

EPOCH_STEPS = 1250            # one epoch at N=20000, batch 8 x grad-accum 2
BUDGETS = list(range(260, 530, 10))

# Nine original trainer seeds, 10-step grid t50. Source: xo/xo.py ORIG_T50, with the
# documented correction for 1001 (360 -> 330; the 360 is a 20-step-grid artefact that
# missed step 330, CANON section 1 / CONTRACT_XO banner). Re-verified on this box for
# 1001 only, via PS-P0's own 27-point curve; the other eight are NOT re-verifiable here.
NINE = {1001: 330, 1004: 370, 1005: 390, 1006: 390, 1007: 450,
        1008: 400, 1009: 330, 1010: 350, 1011: 400}
NINE_STALE_1001 = 360

ORDER_ARMS = {1001: "PS-P0", 1007: "XO-I1001-O1007", 1008: "XO-I1001-O1008",
              1009: "XO-I1001-O1009", 1011: "XO-I1001-O1011"}
INIT_ARMS = {1001: "PS-P0", 1007: "XO-I1007-O1001", 1008: "XO-I1008-O1001",
             1009: "XO-I1009-O1001", 1011: "XO-I1011-O1001"}

asr, clean, source = s1.load_curves("input_xo_behavioral.json")
t50 = {a: s1.t50_10(c) for a, c in asr.items()}
t50s = {a: s1.t50_sustained(c) for a, c in asr.items()}

res = {"source": source, "epoch_steps": EPOCH_STEPS, "budget_axis": [min(BUDGETS), max(BUDGETS)],
       "note": "all figures are re-readings of frozen readouts; no pre-registration (CONTRACT_S1 s0)"}

# --- the three axes, under both ASR readings ------------------------------
def axis(name, values, provenance):
    sp, fb = s1.span(values), s1.flip_band(values, BUDGETS)
    curve = [{"budget": b, "installed": s1.installed_fraction(values, b)[0],
              "of": len(values)} for b in BUDGETS]
    out = {"values": values, "span": sp, "flip_band": fb, "curve": curve,
           "provenance": provenance}
    if fb:
        out["band_as_fraction_of_one_epoch"] = [round(fb["lo"] / EPOCH_STEPS, 4),
                                                round(fb["hi"] / EPOCH_STEPS, 4)]
    return out

res["axes"] = {
    "trainer_seed_nine_t50": axis("nine", [NINE[k] for k in sorted(NINE)],
        "xo.py ORIG_T50 with the 1001 correction; eight of nine not re-verifiable on this box"),
    "order_only_t50": axis("order_t50", [t50[ORDER_ARMS[k]] for k in sorted(ORDER_ARMS)],
        "init fixed at 1001, order varied; derived from raw files, gate GS1-0"),
    "order_only_t50s": axis("order_t50s", [t50s[ORDER_ARMS[k]] for k in sorted(ORDER_ARMS)],
        "same arms, sustained reading"),
    "init_only_t50": axis("init_t50", [t50[INIT_ARMS[k]] for k in sorted(INIT_ARMS)],
        "order fixed at 1001, init varied; derived from raw files, gate GS1-0"),
    "init_only_t50s": axis("init_t50s", [t50s[INIT_ARMS[k]] for k in sorted(INIT_ARMS)],
        "same arms, sustained reading"),
}

# effect of the stale 1001 value on the nine-seed axis, reported so the propagated
# number can be traced (another session has already quoted sd 34.92 from the stale set)
stale_nine = [NINE_STALE_1001 if k == 1001 else NINE[k] for k in sorted(NINE)]
res["nine_seed_sd_corrected"] = s1.span([NINE[k] for k in sorted(NINE)])["sd"]
res["nine_seed_sd_with_stale_1001"] = s1.span(stale_nine)["sd"]

# --- C1 / C2 verdict text ------------------------------------------------
ot = res["axes"]["order_only_t50"]["flip_band"]
it_ = res["axes"]["init_only_t50"]["flip_band"]
ots = res["axes"]["order_only_t50s"]["flip_band"]
its = res["axes"]["init_only_t50s"]["flip_band"]
res["C1_headroom"] = {
    "threshold_cells": s1.NO_HEADROOM_CELLS,
    "order_t50_cells": ot["cells"] if ot else 0,
    "order_t50s_cells": ots["cells"] if ots else 0,
    "has_headroom_both_readings": bool(ot and ots and ot["cells"] > s1.NO_HEADROOM_CELLS
                                       and ots["cells"] > s1.NO_HEADROOM_CELLS),
}
res["C2_contrast"] = {
    "order_t50_cells": ot["cells"] if ot else 0, "init_t50_cells": it_["cells"] if it_ else 0,
    "order_t50s_cells": ots["cells"] if ots else 0, "init_t50s_cells": its["cells"] if its else 0,
    "order_exceeds_init_both_readings":
        bool((ot["cells"] if ot else 0) > (it_["cells"] if it_ else 0)
             and (ots["cells"] if ots else 0) > (its["cells"] if its else 0)),
}

# --- C3 conservation (R7) ------------------------------------------------
cons = {}
for a in sorted(clean):
    v = [x for _, x in clean[a]]
    cons[a] = {"min": min(v), "max": max(v), "final": v[-1], "mean": round(s1.mean(v), 5),
               "t50": t50[a], "t50s": t50s[a]}
finals = [cons[a]["final"] for a in sorted(cons)]
mins = [cons[a]["min"] for a in sorted(cons)]
t50_list = [cons[a]["t50"] for a in sorted(cons)]
rho_f, p_f = s1.perm_p_two_sided(t50_list, finals, draws=10000, seed=0)
rho_m, p_m = s1.perm_p_two_sided(t50_list, mins, draws=10000, seed=0)
res["C3_conservation"] = {
    "per_arm": cons,
    "final_spread_across_arms": round(max(finals) - min(finals), 5),
    "min_over_all_arms_and_steps": min(mins),
    "max_over_all_arms_and_steps": max(cons[a]["max"] for a in cons),
    "mediation_t50_vs_final_clean_acc": {"spearman": rho_f, "perm_p_two_sided": p_f},
    "mediation_t50_vs_min_clean_acc": {"spearman": rho_m, "perm_p_two_sided": p_m},
    "clean_twin_available_in_this_input": False,
}

# --- C4 defender cost ----------------------------------------------------
sig_nine = res["axes"]["trainer_seed_nine_t50"]["span"]["sd"]
sig_order = res["axes"]["order_only_t50"]["span"]["sd"]
res["C4_defender_cost"] = {
    "formula": "n* per side >= 2 sigma^2 (z_0.975 + z_0.80)^2 / delta^2  (2606.29554 s3 Eq.18)",
    "constant": round(2 * s1.Z_SUM ** 2, 4),
    "sigma_trainer_seed_nine": sig_nine,
    "sigma_order_only_n5": sig_order,
    "n_star": {str(d): {"from_nine_trainer_seeds": round(s1.seed_budget(sig_nine, d), 1),
                        "from_order_axis_n5": round(s1.seed_budget(sig_order, d), 1)}
               for d in (10, 20, 30, 50)},
}

json.dump(res, open("results_s1.json", "w"), indent=1, sort_keys=True)

# --- console summary -----------------------------------------------------
print("== axes (cells = number of grid budgets at which the arms disagree) ==")
for k in ("trainer_seed_nine_t50", "order_only_t50", "order_only_t50s",
          "init_only_t50", "init_only_t50s"):
    a = res["axes"][k]
    fb = a["flip_band"]
    print(f"{k:<26} values={a['values']}  span_cells={a['span']['cells']}  "
          f"flip_cells={fb['cells'] if fb else 0}  sd={round(a['span']['sd'],2)}")
print()
print("nine-seed sd corrected :", round(res['nine_seed_sd_corrected'], 4))
print("nine-seed sd if 1001=360:", round(res['nine_seed_sd_with_stale_1001'], 4), "(stale, already quoted elsewhere)")
print()
print("C1 headroom both readings:", res["C1_headroom"]["has_headroom_both_readings"])
print("C2 order > init both readings:", res["C2_contrast"]["order_exceeds_init_both_readings"],
      f"(t50 {res['C2_contrast']['order_t50_cells']} vs {res['C2_contrast']['init_t50_cells']};"
      f" t50s {res['C2_contrast']['order_t50s_cells']} vs {res['C2_contrast']['init_t50s_cells']})")
c3 = res["C3_conservation"]
print(f"C3 clean_acc range over all arms/steps: {c3['min_over_all_arms_and_steps']}"
      f"-{c3['max_over_all_arms_and_steps']}, final spread {c3['final_spread_across_arms']}")
print(f"C3 mediation t50 vs final clean_acc: rho={None if rho_f is None else round(rho_f,4)} p={p_f}")
print(f"C3 mediation t50 vs min   clean_acc: rho={None if rho_m is None else round(rho_m,4)} p={p_m}")
print()
print("C4 defender cost n* per side:")
for d in (10, 20, 30, 50):
    n = res["C4_defender_cost"]["n_star"][str(d)]
    print(f"   delta={d:<3} from nine seeds {n['from_nine_trainer_seeds']:>8}   "
          f"from order axis {n['from_order_axis_n5']:>8}")
band = res["axes"]["trainer_seed_nine_t50"]
print()
print("saturation: nine-seed flip band at", band["band_as_fraction_of_one_epoch"],
      "of one epoch (", EPOCH_STEPS, "steps )")
