"""M3-1..M3-3 (CONTRACT_M3 v4). CPU.   python eval_m3.py [m_out]  -> m3/results_m3.json"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "m_common"))
import m3  # noqa: E402
import posmat_eval as pe  # noqa: E402


def evaluate(segments, rows_json, pm):
    res = {}
    XS = rows_json["X"]
    tables = pe.load_tables(rows_json)
    seg = [r for r in segments if 20 <= r["end"] <= 340]
    cos_med = float(np.median([r["cos"] for r in seg])); n_med = float(np.median([r["norm_ratio"] for r in seg]))
    res["momentum_cos_median"] = float(np.median([r["mom_cos"] for r in seg]))
    for X in XS:
        res[f"M3_1_{X}"] = m3.calib([r[f"dX_{X}"] for r in seg], [r[f"FO_{X}"] for r in seg],
                                    [r[f"L_{X}"] for r in seg], cos_med, n_med)
        rows = rows_json["rows"]
        a328 = [sum(v for j, v in r["a"][X].items() if 10 * int(j) + 10 <= 328) for r in rows]
        elig = [r["s"] <= 328 for r in rows]
        res[f"M3_2_{X}"] = m3.who_builds(a328, [r["poison"] for r in rows], elig)
        A = pe.A_matrix(pm, X, tables)
        q = m3.kernel_Q(A, pm["positions"])
        variants = {"a_coef": dict(coef_scheme="frozen100"), "b_P": dict(variant="P100"), "c_grad": dict(variant="g330"),
                    "d_rowgrad": dict(d100=True), "e_clip": dict(clip_s=101)}
        Qf = {}
        for f, kw in variants.items():
            Af = pe.A_matrix(pm, X, tables, **kw)
            Qf[f] = m3.kernel_Q(Af, pm["positions"], n_boot=10)["Q"]
        q["Q_frozen"] = Qf
        q["main_source"] = m3.main_source(q["Q"], Qf) if q["tier"] in ("晚位更有效", "早位更有效") else None
        q["K"] = A.mean(0).tolist()
        res[f"M3_3_{X}"] = q
    res["tier_main"] = res["M3_1_full"]["tier"]
    return res


if __name__ == "__main__":
    MO = sys.argv[1] if len(sys.argv) > 1 else "/workspace/claude-jump/m_out"
    segments = json.load(open(os.path.join(MO, "m3", "segments.json")))
    rows_json = json.load(open(os.path.join(MO, "m3", "rows.json")))
    pm = json.load(open(os.path.join(MO, "m3", "posmat.json")))
    res = evaluate(segments, rows_json, pm)
    json.dump(res, open(os.path.join(MO, "m3", "results_m3.json"), "w"), indent=1, ensure_ascii=False, default=float)
    print(res["tier_main"])
