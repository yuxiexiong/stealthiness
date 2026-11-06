"""Replication gate (CPU): does the gap between the late fill (B) and the early fill (A) survive a second training
seed? Runs after P1-R7 and P1-R2, before any factorised arm. Writes runs/p1/gate_r.json; exit 1 (dependents never
start) if the gap is at or below 10 steps or cannot be measured.   python gate_r.py [--rehearse]"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import p1


def judge(t7, t2, refs, key="sustained"):
    G = refs["B"][key] - refs["A"][key]
    g = {"readout": key, "t_R7": t7, "t_R2": t2, "G_init1001": G, "own_init1007": refs["OWN_I1007"][key]}
    if t2 is None:
        g.update(gap=None, tier="测不了"); return g
    if t7 is None:                                   # the late fill did not take off by the last LP step: a lower bound
        g.update(gap=max(p1.STEPS_LP) - t2, gap_is_lower_bound=True)
    else:
        g.update(gap=t7 - t2, gap_is_lower_bound=False)
    g["tier"] = p1.tier_R(g["gap"], G)
    g["r_R7_vs_own"] = None if t7 is None else t7 - refs["OWN_I1007"][key]
    g["r_R2_vs_own"] = t2 - refs["OWN_I1007"][key]
    return g


if __name__ == "__main__":
    refs = json.load(open(os.path.join(HERE, "refs_p1.json")))
    if "--rehearse" in sys.argv:
        cases = [((372.0, 300.0), "复现"), ((340.0, 318.0), "部分复现"), ((330.0, 325.0), "不复现"), ((None, 300.0), "复现"), ((350.0, None), "测不了")]
        ok = all(judge(a, b, refs)["tier"] == want for (a, b), want in cases)
        print("gate_r rehearsal", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
    import p1_readouts as rd
    r7, r2 = rd.readouts("P1-R7"), rd.readouts("P1-R2")
    g = judge(r7["sustained"], r2["sustained"], refs)
    g["parallel_readouts"] = {k: judge(r7[k], r2[k], refs, k) for k in ("first", "t50", "t50s")}
    g["gr0"] = {"P1-R7": r7["gr0"], "P1-R2": r2["gr0"]}
    g["proceed"] = g["tier"] in ("复现", "部分复现") and r7["gr0"]["pass"] and r2["gr0"]["pass"]
    json.dump(g, open("/workspace/claude-jump/xo/runs/p1/gate_r.json", "w"), indent=1, ensure_ascii=False, default=str)
    print(json.dumps(g, indent=1, ensure_ascii=False, default=str)); print("GATE_R", "PROCEED" if g["proceed"] else "STOP")
    sys.exit(0 if g["proceed"] else 1)
