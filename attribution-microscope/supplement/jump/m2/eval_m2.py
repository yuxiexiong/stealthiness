"""M2 results (CONTRACT_M2 v4). CPU.   python eval_m2.py [m_out]  -> m2/results_m2.json"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "m_common"))
import m2  # noqa: E402
import mcommon as mc  # noqa: E402


def load(MO):
    traj, meta = {}, {}
    for key in ("1009", "1007", "1008", "twin"):
        p = os.path.join(MO, "m2", f"traj_{key}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        o = key if key == "twin" else int(key)
        traj[o] = {int(s): c for s, c in d["steps"].items()}
        meta[o] = d
    return traj, meta


def evaluate(traj, meta):
    res = {}
    poison = {o: traj[o] for o in (1009, 1007, 1008) if o in traj}
    n = len(meta[1009]["eval"])
    boots = m2.boot_indices(n)
    res["GM2_I"] = meta[1009].get("GM2_I")
    none = meta[1009]["none"]
    d_none = float(np.mean(np.asarray(none["trig"]) - np.asarray(none["clean"])))
    st, v = m2.curve(poison[1009], "full", "sufficient")
    res["GM2_1"] = float(v[st == 1250][0] - d_none) if 1250 in st else None
    if res["GM2_1"] is None or res["GM2_1"] < 5:
        res["tier_all"] = "测不了"; return res
    res["M2_0"] = m2.m2_0(poison, boots)
    hand = meta[1009]["handoff"]
    if hand.get("S_star"):
        ctype = hand["type"]
        rnames = [f"R{i}" for i in range(4) if any(f"R{i}" in c for c in poison[1009].values())]
        delta = res["M2_0"]["delta_int"] if res["M2_0"].get("delta_flag") is None else None
        res["M2_1"] = m2.m2_1(poison, ctype, boots, delta, rnames)
        rand_p2 = {k: v["P2"] for k, v in res["M2_1"].get("random", {}).items()}
        res["M2_2"] = m2.m2_2(poison, ctype, none, boots, rnames, rand_p2)
        # M2-3 descriptive
        d3 = {}
        for o in poison:
            sz, vz = m2.curve(poison[o], "Z", ctype)
            if len(sz):
                k = int(np.argmax(vz)); d3[f"Z_peak_{o}"] = {"step": float(sz[k]), "value": float(vz[k]),
                                                            "drop_after": float(vz[k] - vz[-1])}
        if "twin" in traj:
            for kind in ("full", "Y"):
                stw, vtw = m2.curve(traj["twin"], kind, ctype)
                if len(vtw):
                    mx = float(vtw.max())
                    sp, vp = m2.curve(poison[1009], kind, ctype)
                    t_on = next((float(sp[i]) for i in range(len(sp)) if vp[i] > mx and
                                 np.all(vp[(sp > sp[i]) & (sp <= sp[i] + 40)] > mx)), math.inf)
                    d3[f"t_on_{kind}"] = t_on; d3[f"twin_max_{kind}"] = mx
            d3["GM2_2"] = None
            stw, vtw = m2.curve(traj["twin"], "full", ctype)
            if 1250 in stw:
                d3["GM2_2"] = bool(vtw[stw == 1250][0] - d_none <= 0.2 * res["GM2_1"])
        res["M2_3"] = d3
    return res


if __name__ == "__main__":
    MO = sys.argv[1] if len(sys.argv) > 1 else mc.OUT
    traj, meta = load(MO)
    res = evaluate(traj, meta)
    json.dump(res, open(os.path.join(MO, "m2", "results_m2.json"), "w"), indent=1, ensure_ascii=False, default=float)
    print(json.dumps({k: (v.get("tier") if isinstance(v, dict) else v) for k, v in res.items()}, ensure_ascii=False, default=float))
