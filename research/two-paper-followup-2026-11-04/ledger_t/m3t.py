"""M3T (CONTRACT_M3T.zh-CN.md): tests on the trigger-side ledger. numpy only; reuses phase0/p0.py.

The frozen M3 ledger reads D = T - C (trigger-side minus clean-side mean margin on EVAL100). M3T re-runs the same
ledger with T as a second read-out. Per clean row i: c_T = its first-order contribution to T, c_D to D, c_C = c_T - c_D."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "phase0"))
import p0  # noqa: E402

T_END = p0.T_LEDGER             # 340: last segment end with a calibration requirement
P_STRONG, P_WEAK, SHARE = 0.005, 0.025, 0.10
MIN_CELL = 20
NPERM, SEED = 20000, 20261106
REPRO_REL_L2 = 1e-2             # GT2: new D-ledger against the frozen rows.json
CAL = {"corr_min": 0.9, "ratio_lo": 0.8, "ratio_hi": 1.25}     # M3-1 thresholds (CONTRACT_M3 section 4)


def rows_table(rows_json, T=T_END):
    """Clean rows with 11 <= s <= T: orig id, step, arrival segment, per-token loss, contributions summed over
    segments ending <= T for every read-out in the file."""
    out = []
    for r in rows_json["rows"]:
        if r["poison"] or not (p0.SEG0 <= r["s"] <= T):
            continue
        c = {X: sum(v for j, v in r["a"][X].items() if 10 * int(j) + 10 <= T) for X in rows_json["X"]}
        out.append((r["orig"], r["s"], (r["s"] - 1) // 10, r["ell"] / max(r["n"], 1), c))
    orig = np.array([o[0] for o in out]); s = np.array([o[1] for o in out]); g = np.array([o[2] for o in out])
    loss = np.array([o[3] for o in out], dtype=float)
    c = {X: np.array([o[4][X] for o in out], dtype=float) for X in rows_json["X"]}
    return orig, s, g, loss, c


def totals(rows_json, X, T=T_END):
    pos = neg = poi = 0.0
    for r in rows_json["rows"]:
        if not (p0.SEG0 <= r["s"] <= T):
            continue
        c = sum(v for j, v in r["a"][X].items() if 10 * int(j) + 10 <= T)
        if r["poison"]:
            poi += c
        elif c > 0:
            pos += c
        else:
            neg += c
    return {"poison_net": poi, "clean_pos": pos, "clean_neg": neg, "clean_net": pos + neg}


def calib(segments, X, T=T_END):
    seg = [r for r in segments if 20 <= r["end"] <= T]
    dX = np.array([r[f"dX_{X}"] for r in seg]); FO = np.array([r[f"FO_{X}"] for r in seg]); L = np.array([r[f"L_{X}"] for r in seg])
    out = {"n_segments": len(seg), "growth": float(dX.sum()),
           "fo_corr": float(np.corrcoef(FO, dX)[0, 1]), "fo_ratio": float(FO.sum() / dX.sum()),
           "ledger_corr": float(np.corrcoef(L, dX)[0, 1]), "ledger_ratio": float(L.sum() / dX.sum())}
    out["pass"] = bool(min(out["fo_corr"], out["ledger_corr"]) >= CAL["corr_min"]
                       and CAL["ratio_lo"] <= out["fo_ratio"] <= CAL["ratio_hi"] and CAL["ratio_lo"] <= out["ledger_ratio"] <= CAL["ratio_hi"])
    return out


def top_quartile(loss, g):
    """Within each arrival segment, the quarter of clean rows with the highest per-token loss at arrival."""
    lab = np.zeros(len(loss), dtype=bool)
    for gg in np.unique(g):
        m = np.flatnonzero(g == gg); k = len(m) // 4
        if k:
            lab[m[np.argsort(-loss[m], kind="mergesort")[:k]]] = True
    return lab


def tier(st, growth, expect):
    """expect = -1: the hypothesis says the labelled rows pull the read-out DOWN more than their segment-mates."""
    if st["n"] < MIN_CELL:
        return "测不了（格子不足 20 行）"
    share = st["excess"] / abs(growth)
    p_exp, p_opp = (st["p_lo"], st["p_hi"]) if expect < 0 else (st["p_hi"], st["p_lo"])
    if p_exp <= P_STRONG:
        return "成立（有分量）" if abs(share) >= SHARE else "成立（统计上有、分量小）"
    if p_opp <= P_STRONG:
        return "反向（有分量）" if abs(share) >= SHARE else "反向（统计上有、分量小）"
    return "弱线索" if min(p_exp, p_opp) <= P_WEAK else "没拿到结果"


def reproduce(new_rows, old_rows, T=T_END):
    """GT2: per-row contribution to D in the new ledger against the frozen one."""
    a = {(r["s"], r["k"]): sum(v for j, v in r["a"]["full"].items() if 10 * int(j) + 10 <= T) for r in new_rows["rows"] if r["s"] <= T}
    b = {(r["s"], r["k"]): sum(v for j, v in r["a"]["full"].items() if 10 * int(j) + 10 <= T) for r in old_rows["rows"] if r["s"] <= T}
    keys = sorted(b)
    same_keys = sorted(a) == keys
    x = np.array([a.get(k, np.nan) for k in keys]); y = np.array([b[k] for k in keys])
    rel = float(np.linalg.norm(x - y) / np.linalg.norm(y)) if same_keys else float("nan")
    return {"same_rows": same_keys, "n": len(keys), "rel_l2": rel, "max_abs": float(np.nanmax(np.abs(x - y))) if same_keys else None,
            "pass": bool(same_keys and rel <= REPRO_REL_L2)}
