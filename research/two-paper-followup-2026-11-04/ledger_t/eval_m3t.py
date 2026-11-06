"""Judge CONTRACT_M3T from the copied-back ledger (CPU, numpy only).
    python eval_m3t.py <dir with the NEW rows.json and segments.json>   -> results_m3t.json next to this file
Gates first; a failed gate turns every test into 测不了. C is derived: c_C = c_T - c_D."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m3t  # noqa: E402
p0 = m3t.p0
IN0 = os.path.join(HERE, "..", "phase0", "inputs")
D = sys.argv[1]
new = p0.load(os.path.join(D, "rows.json")); seg = p0.load(os.path.join(D, "segments.json"))
old = p0.load(os.path.join(IN0, "rows.json.gz"))
op = p0.load(os.path.join(IN0, "orders_poison.json.gz")); meta = p0.load(os.path.join(IN0, "train_meta.json.gz"))
res = {"gates": {}}
res["gates"]["GT2_reproduces_frozen_D_ledger"] = m3t.reproduce(new, old)
res["gates"]["GT3_calibration_T"] = m3t.calib(seg, "T"); res["gates"]["calibration_D"] = m3t.calib(seg, "full")
res["gates"]["identity_all_segments"] = bool(all(r["identity_ok_full"] and r["identity_ok_T"] for r in seg if r["end"] <= m3t.T_END))
gate_ok = bool(res["gates"]["GT2_reproduces_frozen_D_ledger"]["pass"] and res["gates"]["GT3_calibration_T"]["pass"] and res["gates"]["identity_all_segments"])
res["gates"]["pass"] = gate_ok
orig, s, g, loss, c = m3t.rows_table(new)
c["C"] = c["T"] - c["full"]
growth = {"T": res["gates"]["GT3_calibration_T"]["growth"], "full": res["gates"]["calibration_D"]["growth"]}
growth["C"] = growth["T"] - growth["full"]
res["growth"] = growth
res["who_builds"] = {X: m3t.totals(new, X) for X in ("full", "T")}
cue, _, _ = p0.cue_labels(np.asarray(op["orders"]["1009"]), op["poison"], meta)
q = m3t.top_quartile(loss, g)
cells = {"ANY_after": cue["ANY_after"][orig], "IMG_after": cue["IMG_after"][orig], "Q_after": cue["Q_after"][orig],
         "ANY_before": cue["ANY_before"][orig], "TOP_LOSS_QUARTILE": q}
res["cells"] = {}
for name, lab in cells.items():
    for X in ("T", "full", "C"):
        st = p0.strat_perm(c[X], lab, g, m3t.NPERM, m3t.SEED)
        # C is a derived read-out (T - D) whose own growth can be near zero, so it is never tiered:
        # its excess is reported in units of T's growth, side by side with T and D
        den = growth["T"] if X == "C" else growth[X]
        st["share_of_growth"] = st["excess"] / abs(den) if den else None
        st["tier"] = ("并列，不判档" if X == "C" else m3t.tier(st, den, -1)) if gate_ok else "测不了（闸门未过）"
        rk = np.zeros(len(g))
        for gg in np.unique(g):
            m = g == gg; o = np.argsort(np.argsort(c[X][m])); rk[m] = o - o.mean()
        sr = p0.strat_perm(rk, lab, g, m3t.NPERM, m3t.SEED)
        st["rank_version"] = {"p_hi": sr["p_hi"], "p_lo": sr["p_lo"]}
        res["cells"][f"{name}|{X}"] = st
# descriptive: contributions to T by within-segment loss quartile
quart = np.zeros(len(g), dtype=int)
for gg in np.unique(g):
    m = np.flatnonzero(g == gg); order = np.argsort(loss[m], kind="mergesort"); quart[m[order]] = (np.arange(len(m)) * 4) // max(len(m), 1)
res["by_loss_quartile"] = {X: [float(c[X][quart == k].sum()) for k in range(4)] for X in ("T", "full", "C")}
res["tiers"] = {"T1_cue": res["cells"]["ANY_after|T"]["tier"], "T2_difficulty": res["cells"]["TOP_LOSS_QUARTILE|T"]["tier"]}
b = res["cells"]["ANY_before|T"]
res["T1_order_specific"] = bool(res["tiers"]["T1_cue"].startswith("成立") and min(b["p_hi"], b["p_lo"]) > m3t.P_WEAK)
json.dump(res, open(os.path.join(HERE, "results_m3t.json"), "w"), indent=1, ensure_ascii=False, default=float)
print(json.dumps({"gates_pass": gate_ok, **res["tiers"]}, ensure_ascii=False))
