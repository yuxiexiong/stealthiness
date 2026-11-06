"""Offline rehearsal for CONTRACT_P1 (rule 6.2). numpy only; no GPU, no server.
  1. builds every designed sequence from the REAL orders and the real poison set and runs the structure gate;
  2. shows every verdict tier reachable and unreachable on synthetic take-off times;
  3. checks the read-out functions on synthetic margin curves, including censoring at the stop step.
    python rehearsal_p1.py -> rehearsal_p1.json ; exit code 0 only if every check passes"""
import gzip
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p1  # noqa: E402
import sf  # noqa: E402

chk, info = {}, {}
op = json.load(gzip.open(os.path.join(HERE, "..", "phase0", "inputs", "orders_poison.json.gz"), "rt"))
poison = op["poison"]; P = p1.pmask(poison)
S, A, B = p1.sequences(op["orders"], poison)
ok, rep = p1.structure_gate(S, A, B, P)
chk["structure_gate_on_real_orders"] = ok
info["structure_checks_failed"] = [k for k, v in rep["checks"].items() if not v]
info["bookkeeping"] = {arm: {"vs_A": {k: round(v, 4) if isinstance(v, float) else v for k, v in r["vs_A"].items()},
                             "vs_B": {k: round(v, 4) if isinstance(v, float) else v for k, v in r["vs_B"].items()},
                             **{k: round(v, 4) for k, v in r.items() if k not in ("vs_A", "vs_B")}} for arm, r in rep["arms"].items()}
chk["13_arms"] = len(S) == 13 and set(S) == set(p1.ARMS)
chk["A_is_existing_arm_S1009_F2001"] = bool((A == sf.recombine(op["orders"]["1009"], op["orders"]["2001"], poison)).all())
chk["ID_is_order_1009"] = bool((S["P1-ID"] == np.asarray(op["orders"]["1009"])).all())
# a structure gate that can fail: break one arm and see it flagged
S_bad = dict(S); S_bad["P1-MB"] = A
chk["structure_gate_can_fail"] = p1.structure_gate(S_bad, A, B, P)[0] is False
# src_map round trip for both init seeds (the Trainer reads file positions in randperm(init) order)
for init in ("1001", "1007"):
    r_init = np.asarray(op["orders"][init]); src = sf.src_map(r_init, S["P1-MB"])
    chk[f"src_map_roundtrip_init{init}"] = bool((src[r_init] == S["P1-MB"]).all()) and sorted(src.tolist()) == list(range(p1.N))

tA, tB = 306.3, 356.8; G = tB - tA
chk["R_stop"] = p1.tier_R(8.0, G) == "不复现"; chk["R_partial"] = p1.tier_R(20.0, G) == "部分复现"; chk["R_ok"] = p1.tier_R(45.0, G) == "复现"
chk["mates_carry"] = p1.verdict_mates(350.0, 312.0, tA, tB)["tier"] == "同伴承担"
chk["mates_not"] = p1.verdict_mates(309.0, 353.0, tA, tB)["tier"] == "同伴不承担"
chk["mates_mixed"] = p1.verdict_mates(350.0, 353.0, tA, tB)["tier"] == "部分或不对称"
chk["set_carries"] = p1.verdict_set(305.0, 311.0, 352.0, 360.0, tA, tB)["tier"] == "集合承担"
chk["arrangement_carries"] = p1.verdict_set(300.0, 345.0, 310.0, 340.0, tA, tB)["tier"] == "排列承担"
chk["gap_vanishes"] = p1.verdict_set(330.0, 332.0, 331.0, 333.0, tA, tB)["tier"] == "置换后差距消失且两次置换一致"
chk["set_undetermined"] = p1.verdict_set(305.0, 340.0, 352.0, 360.0, tA, tB)["tier"] == "部分或不可判"
chk["window_late"] = p1.verdict_window(352.0, 309.0, tA, tB)["tier"] == "晚窗承担"
chk["window_early"] = p1.verdict_window(309.0, 352.0, tA, tB)["tier"] == "早段承担"
chk["window_additive"] = p1.verdict_window(330.0, 332.0, tA, tB)["tier"] == "两段各占一份（近似可加）"
chk["window_other"] = p1.verdict_window(352.0, 352.0, tA, tB)["tier"] == "其它"
chk["ladder_quiet"] = p1.verdict_ladder(309.0, tA)["tier"] == "安静"; chk["ladder_between"] = p1.verdict_ladder(314.0, tA)["tier"] == "带间"
chk["ladder_moved"] = p1.verdict_ladder(290.0, tA)["tier"] == "动了"; chk["ladder_censored"] = p1.verdict_ladder(None, tA)["tier"] == "测不了"

steps = p1.STEPS_LP
curve = lambda t0: [(-6.0 if s < t0 - 5 else (4.0 if s > t0 + 5 else (s - t0) * 0.8)) for s in steps]   # noqa: E731
t = sf.t_m0_sustained(steps, curve(333.0)); chk["sustained_readout_recovers_crossing"] = abs(t - 333.0) < 2.0   # linear interpolation on a 10-step grid
chk["no_crossing_by_560_is_none"] = sf.t_m0_sustained(steps, [-5.0] * len(steps)) is None
dip = curve(333.0); dip[steps.index(350)] = -1.0
chk["dip_delays_sustained_not_first"] = sf.t_m0(steps, dip) < 340 and sf.t_m0_sustained(steps, dip) > 350
chk["grid_supports_sustained_up_to_520"] = max(steps) >= 520 + 40

allok = all(chk.values())
json.dump({"pass": allok, "checks": chk, "info": info}, open(os.path.join(HERE, "rehearsal_p1.json"), "w"), indent=1, ensure_ascii=False)
for k, v in chk.items():
    print(("PASS " if v else "FAIL ") + k)
for arm, r in info["bookkeeping"].items():
    print(arm, "| vs A: set", r["vs_A"]["prefix_set_overlap"], "mates", r["vs_A"]["mates_shared"], "mates<=360", r["vs_A"]["mates_shared_prefix"], "same-step", r["vs_A"]["frac_clean_same_step"],
          "| vs B: set", r["vs_B"]["prefix_set_overlap"], "mates", r["vs_B"]["mates_shared"], "| maxshift<stop", r["vs_A"]["max_step_shift_rows_before_stop"],
          {k: v for k, v in r.items() if k not in ("vs_A", "vs_B")})
print("failed structure checks:", info["structure_checks_failed"])
print("REHEARSAL", "PASS" if allok else "FAIL", f"{sum(chk.values())}/{len(chk)}")
sys.exit(0 if allok else 1)
