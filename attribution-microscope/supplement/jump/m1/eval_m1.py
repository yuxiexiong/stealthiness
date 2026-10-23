"""M1 results from raw.json (CONTRACT_M1 v4). CPU.   python eval_m1.py [m_out]  -> m1/results_m1.json"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "m_common"))
import m1  # noqa: E402
import mcommon as mc  # noqa: E402

MO = sys.argv[1] if len(sys.argv) > 1 else mc.OUT
raw = json.load(open(os.path.join(MO, "m1", "raw.json")))
res = {"GM1_0": raw.get("GM1_0")}
if "HALT" in raw:
    res["tier"] = "合同执行失败"
else:
    s = raw["s1250"]
    dec = m1.decide(s["D_full_ev"] - s["D_none_ev"], s["cand"])
    assert dec["tier"] == s["decision"]["tier"], "re-derived tier differs from the run's"
    res.update({"tier": dec["tier"], "type": dec["type"], "S_star": dec["S_star"], "union": dec["union"],
                "not_replicated": dec["not_replicated"], "GM1_1": s["D_full_ev"] - s["D_none_ev"],
                "candidates": {k: {"size": len(v["set"]), "sel": v["sel"], "ev": v["ev"], "task_ev": v["task_ev"],
                                   "norm_ev": v.get("norm_ev"),
                                   "jaccard_task": m1.jaccard(v["set"], v["task_set"]),
                                   "jaccard_norm": m1.jaccard(v["set"], v.get("norm_set"))} for k, v in s["cand"].items()},
                "random_F_ev": s.get("random_F_ev"), "loo": s.get("loo"), "layer_E": s.get("layer_E"),
                "residual": raw.get("residual"), "s330_tier": raw.get("s330", {}).get("decision", {}).get("tier"),
                "jaccard_330_1250": raw.get("jaccard_330_1250"), "twin": raw.get("twin")})
    if dec["S_star"]:
        S = dec["S_star"]
        res["S_star_layers"] = sorted({int(n.split(".")[0][1:]) for n in S})
        res["S_star_mlp_share"] = float(np.mean([n.split(".")[1] in ("gate_proj", "up_proj", "down_proj") for n in S]))
        res["S_star_layer16plus_share"] = float(np.mean([int(n.split(".")[0][1:]) >= 16 for n in S]))
json.dump(res, open(os.path.join(MO, "m1", "results_m1.json"), "w"), indent=1, ensure_ascii=False, default=float)
print(res["tier"])
