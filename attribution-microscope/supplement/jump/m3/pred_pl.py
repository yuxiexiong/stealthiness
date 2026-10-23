"""M3-4 ledger predictions for PL (CONTRACT_M3 v4 §4). Frozen with the contract; no free parameters.
Reads only M3 outputs, M2-0 and pl/design_pl_v2.json (never a PL read-out).
    python pred_pl.py [m_out_dir]      -> <m_out>/m3/pred_pl.json"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "m_common"))
import m3  # noqa: E402
import posmat_eval as pe  # noqa: E402

MO = sys.argv[1] if len(sys.argv) > 1 else "/workspace/claude-jump/m_out"
J = os.path.join(HERE, "..")


def predict(segments, xval, rows_json, pm, m20, moves):
    tables = pe.load_tables(rows_json)
    seg = [r for r in segments if 20 <= r["end"] <= 340]
    dL = np.array([r["L_full"] for r in seg]); dX = np.array([r["dX_full"] for r in seg])
    b = float((dL * dX).sum() / (dL * dL).sum())
    cks = sorted(int(k) for k in xval)
    xm = np.array([xval[str(c)]["full"] for c in cks], float)
    ts = np.arange(10, 431, 1.0)
    Xmeas = np.interp(ts, cks, xm)
    Lstar = float(np.interp(m3.T_TAKE, cks, xm))
    k2row = {r["k"]: i for i, r in enumerate(pm["rows"])}
    out = {"b": b, "L_star": Lstar, "arms": {}}
    for arm in m3.ARMS:
        mv = moves[arm]
        delta = np.zeros_like(ts)
        for _, (p_from, p_to) in mv.items():
            ii = k2row[int(p_from)]
            s0, s1 = int(p_from) // 16 + 1, int(p_to) // 16 + 1
            delta += np.array([pe.row_at(pm, ii, s1, t, "full", tables) - pe.row_at(pm, ii, s0, t, "full", tables) for t in ts])
        Xp = Xmeas + b * delta
        tau = m3.first_sustained(ts, Xp, Lstar)
        r = (tau - m3.T_TAKE) if math.isfinite(tau) else math.inf
        out["arms"][arm] = {"r_hat": r, "censored": not math.isfinite(r) or r >= m3.CENSOR}
    out["m2_0"] = {k: m20.get(k) for k in ("tier", "delta_int", "delta_flag", "tol_full", "e_full")}
    out["confirmatory_premise"] = m20.get("tier") == "D 计时"
    return out


if __name__ == "__main__":
    segments = json.load(open(os.path.join(MO, "m3", "segments.json")))
    xval = json.load(open(os.path.join(MO, "m3", "ledger_status.json")))["xval"]
    rows_json = json.load(open(os.path.join(MO, "m3", "rows.json")))
    pm = json.load(open(os.path.join(MO, "m3", "posmat.json")))
    m20 = json.load(open(os.path.join(MO, "m2", "results_m2.json")))["M2_0"]
    moves = json.load(open(os.path.join(J, "pl", "design_pl_v2.json")))["moves_1009"]
    res = predict(segments, xval, rows_json, pm, m20, moves)
    import glob
    lp = sorted(glob.glob("/workspace/claude-jump/lp_out/PL-*.json"))
    res["blind_check"] = {"time": time.strftime("%F %T"), "n_pl_lp_files": len(lp),
                          "earliest_pl_lp_mtime": (time.strftime("%F %T", time.localtime(min(os.path.getmtime(f) for f in lp)))
                                                   if lp else None),
                          "note": "blindness is procedural: this code was frozen before any PL read-out was evaluated; "
                                  "the freeze commit time vs results_pl.json / LOG entries decides confirmatory status"}
    json.dump(res, open(os.path.join(MO, "m3", "pred_pl.json"), "w"), indent=1, default=float)
    print(json.dumps({a: v["r_hat"] for a, v in res["arms"].items()}, default=float))
