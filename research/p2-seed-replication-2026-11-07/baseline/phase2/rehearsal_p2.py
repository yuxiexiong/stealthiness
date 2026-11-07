"""Offline rehearsal for CONTRACT_P2 (rule 6.2): builds the real sequences, runs the structure gate, and shows
every verdict band reachable AND not reachable on synthetic numbers. numpy only; no GPU, no server.
    python rehearsal_p2.py   -> rehearsal_p2.json ; exit 0 only if every check passes"""
import gzip, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import p2, sf

chk, info = {}, {}
op = json.load(gzip.open(os.path.join(HERE, "..", "phase0", "inputs", "orders_poison.json.gz"), "rt"))
poison = op["poison"]; P = p2.pmask(poison)
S, par, H, reg = p2.sequences(op["orders"], poison)
ok, rep = p2.structure_gate(S, par, H, reg, P, op["orders"]["1009"])
chk["structure_gate_on_real_orders"] = ok
info["structure_checks_failed"] = [k for k, v in rep["checks"].items() if not v]
info["sizes"] = rep["sizes"]; info["prefix_clean_set_overlap"] = rep["prefix_clean_set_overlap"]
info["prefix_same_position_fraction"] = rep["prefix_same_position_fraction"]
chk["three_arms"] = set(S) == set(p2.ARMS)
chk["A_prefix_is_existing_parent"] = bool((S["P2-A"][:200 * 16] == sf.recombine(op["orders"]["1009"], op["orders"]["2001"], poison)[:200 * 16]).all())
chk["R_prefix_is_order_1009"] = bool((S["P2-R"][:200 * 16] == np.asarray(op["orders"]["1009"])[:200 * 16]).all())
chk["arms_identical_after_prefix_through_stop"] = bool((S["P2-A"][200 * 16:700 * 16] == S["P2-B"][200 * 16:700 * 16]).all()
                                                       and (S["P2-A"][200 * 16:700 * 16] == S["P2-R"][200 * 16:700 * 16]).all())
chk["arms_differ_inside_prefix"] = float((S["P2-A"][:200 * 16] != S["P2-B"][:200 * 16]).mean()) > 0.5
# a structure gate that can fail
bad = {k: v.copy() for k, v in S.items()}; bad["P2-B"][300 * 16] = bad["P2-B"][900 * 16]
chk["structure_gate_can_fail"] = p2.structure_gate(bad, par, H, reg, P, op["orders"]["1009"])[0] is False
# the Trainer reads file positions in randperm(seed) order; src_map must invert that
r_init = np.asarray(op["orders"][str(p2.TRAIN_SEED)])
for arm in S:
    src = sf.src_map(r_init, S[arm])
    chk[f"src_map_roundtrip_{arm}"] = bool((src[r_init] == S[arm]).all()) and sorted(src.tolist()) == list(range(p2.N))

# ---- take-off bands
chk["band_small"] = p2.gap_tier(7.0) == "小差异（本轮未确认明显持续效应）"
chk["band_middle"] = p2.gap_tier(20.0) == "中间档（有限证据）"
chk["band_large"] = p2.gap_tier(44.0) == "较大差异"
chk["band_large_negative"] = p2.gap_tier(-44.0) == "较大差异"
chk["band_censored"] = p2.gap_tier(None) == "测不了"
chk["main_reports_reversal"] = p2.verdict_main(350.0, 300.0)["direction_reversed"] is True
chk["main_censored"] = p2.verdict_main(320.0, None)["tier"] == "测不了"
v = p2.verdict_vs_reference(306.0, 328.0, 365.0)   # offsets -22 (middle band) and +37 (large band)
# the band applies to the offset from R, not to A and B's own gap: -22 is the middle band
chk["reference_between"] = v["R_between_A_and_B"] is True and v["A_tier"] == "中间档（有限证据）" and v["B_tier"] == "较大差异"
chk["reference_offset_large"] = p2.verdict_vs_reference(280.0, 328.0, 330.0)["A_tier"] == "较大差异"
chk["reference_not_between"] = p2.verdict_vs_reference(340.0, 328.0, 350.0)["R_between_A_and_B"] is False

# ---- components: every tier reachable, on synthetic per-probe numbers
rng = np.random.default_rng(7)
def case(dC, dD, n=200, sd=0.3):
    Ta = rng.normal(0, 1, n); Ca = rng.normal(0, 1, n)
    Cb = Ca + dC + rng.normal(0, sd, n); Db = (Ta - Ca) + dD + rng.normal(0, sd, n)
    return p2.components(Ta, Ca, Cb + Db, Cb)
chk["comp_common"] = case(1.8, 0.2)["tier"] == "共同分量为主"
chk["comp_discrimination"] = case(0.2, 1.8)["tier"] == "区分分量为主"
chk["comp_mixed"] = case(1.0, 1.0)["tier"] == "混合"
chk["comp_cancel"] = case(3.0, -2.0)["tier"] == "抵消（两个分量反向）"
chk["comp_undetermined_small"] = case(0.05, 0.05)["tier"] == "不可判（分母不稳定）"
r = case(1.8, 0.2)
chk["comp_identity_holds"] = r["identity_max_abs_err"] < 1e-9
chk["comp_ci_present"] = all(r[k]["lo"] is not None and r[k]["lo"] < r[k]["mean"] < r[k]["hi"] for k in ("dT", "dC", "dD"))
info["comp_example_common"] = {k: (v if not isinstance(v, dict) else {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}) for k, v in r.items()}

# ---- read-out functions on the new grid
steps = p2.STEPS_LP
curve = lambda t0: [(-6.0 if s < t0 - 5 else (4.0 if s > t0 + 5 else (s - t0) * 0.8)) for s in steps]   # noqa: E731
chk["sustained_recovers_crossing"] = abs(sf.t_m0_sustained(steps, curve(333.0)) - 333.0) < 2.0
chk["sustained_recovers_late_crossing"] = abs(sf.t_m0_sustained(steps, curve(655.0)) - 655.0) < 2.0
chk["no_crossing_is_none"] = sf.t_m0_sustained(steps, [-5.0] * len(steps)) is None
dip = curve(333.0); dip[steps.index(350)] = -1.0
chk["dip_delays_sustained_not_first"] = sf.t_m0(steps, dip) < 340 and sf.t_m0_sustained(steps, dip) > 350
chk["grid_confirms_candidates_to_660"] = max(steps) >= p2.CAND_MAX + p2.HOLD
chk["behaviour_steps_are_in_lp_grid"] = set(p2.STEPS_BEH) <= set(p2.STEPS_LP)
chk["grid_sizes"] = len(p2.STEPS_LP) == 59 and len(p2.STEPS_BEH) == 25

allok = all(chk.values())
json.dump({"pass": allok, "checks": chk, "info": info}, open(os.path.join(HERE, "rehearsal_p2.json"), "w"), indent=1, ensure_ascii=False)
for k, v in chk.items():
    print(("PASS " if v else "FAIL ") + k)
print(json.dumps(info["sizes"], ensure_ascii=False))
print("prefix clean-set overlap:", json.dumps({k: round(v, 4) for k, v in info["prefix_clean_set_overlap"].items()}, ensure_ascii=False))
print("failed structure checks:", info["structure_checks_failed"])
print("REHEARSAL", "PASS" if allok else "FAIL", f"{sum(chk.values())}/{len(chk)}")
sys.exit(0 if allok else 1)
