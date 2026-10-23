"""Judge CONTRACT_G from copied-back read-outs.   python eval_g.py <copyback_dir> <out.json>
copyback_dir: g_out/<ARM>@s*.json, g_out/GG0-*.json, g/gate.json (as produced by copyback_g.sh)."""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "pl")); sys.path.insert(0, os.path.join(HERE, "..", "sf"))
import g  # noqa: E402
import pl  # noqa: E402

D, OUT = sys.argv[1], sys.argv[2]
gp = os.path.join(D, "g", "gate.json")
res = {"gate": {k: json.load(open(gp)).get(k) for k in ("pass", "checks")} if os.path.exists(gp) else None, "arms": {}}

for arm in (g.MAIN_ARM,) + g.NULL_ARMS:
    rec = {}
    for f in glob.glob(os.path.join(D, "g_out", f"{arm}@s*.json")):
        rec[int(f.rsplit("@s", 1)[1][:-5])] = json.load(open(f))
    a = {"n_points": len(rec)}
    if rec:
        st, v = g.curve(rec)
        fit = g.fit_loglog(st, v)
        a["n_in_fit_window"] = int(fit[3].sum()) if fit else 0
        a["gg1_pass"] = a["n_in_fit_window"] >= 20
        a["ci"] = g.ratio_ci(st, v)
        a["tier_rise"] = g.tier_rise(a["ci"]) if a["gg1_pass"] else "cannot measure"
        a["curve"] = {int(s): float(x) for s, x in zip(st, v)}
        a["M_bar"] = {s: rec[s]["M_bar"] for s in sorted(rec)}
        for key in ("A", "B"):                      # descriptive: the two factor groups
            vals = {s: rec[s]["g1_groups"].get(key) for s in sorted(rec) if rec[s].get("g1_groups", {}).get(key)}
            if len(vals) >= 4:
                sk = np.asarray(sorted(vals), dtype=float)
                a[f"ci_{key}"] = g.ratio_ci(sk, np.asarray([vals[int(s)] for s in sk], dtype=float))
    res["arms"][arm] = a

main = res["arms"][g.MAIN_ARM]
res["criterion_1"] = {"tier": main.get("tier_rise", "cannot measure"), "R_threshold": g.R_RISE,
                      "ci": main.get("ci"), "gg1_pass": main.get("gg1_pass"),
                      "T_late": g.T_LATE, "T_early": g.T_EARLY}

if main.get("curve") and main.get("gg1_pass"):
    pos = np.asarray(json.load(open(os.path.join(HERE, "..", "pl", "arrivals_1009.json")))["pos"])
    arm_pos, _ = pl.arm_positions(pos, g.T_REF)
    preds = pl.predictions(pos, arm_pos, g.T_REF)
    st = np.asarray(sorted(int(k) for k in main["curve"]), dtype=float)
    v = np.asarray([main["curve"][int(s)] for s in st], dtype=float)
    col = g.gain_column(pos, arm_pos, g.interp_gain(st, v))
    res["criterion_2"] = {"gain_column": col, "null_column_lr_weighted_count": preds[g.COUNT_KEY],
                          "descriptive_plain_count": preds[g.COUNT_KEY_PLAIN],
                          **g.tier_discriminates(col, preds[g.COUNT_KEY])}
    res["criterion_2"]["vs_plain_count_descriptive"] = g.tier_discriminates(col, preds[g.COUNT_KEY_PLAIN])
else:
    res["criterion_2"] = {"tier": "cannot measure"}

json.dump(res, open(OUT, "w"), indent=1, default=float)
print(json.dumps({"gate": (res["gate"] or {}).get("pass"),
                  "criterion_1": {"tier": res["criterion_1"]["tier"],
                                  "R": (res["criterion_1"]["ci"] or {}).get("R_fit"),
                                  "ci": [(res["criterion_1"]["ci"] or {}).get("ci_low"), (res["criterion_1"]["ci"] or {}).get("ci_high")]},
                  "criterion_2": {"tier": res["criterion_2"]["tier"], "max_abs_diff": res["criterion_2"].get("max_abs_diff")},
                  "nulls": {a: res["arms"][a].get("tier_rise") for a in g.NULL_ARMS}}, indent=1, default=float))
