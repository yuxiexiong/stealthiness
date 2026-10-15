"""B0 resolution check on Qwen SQuAD (CONTRACT_B0RES.md). numpy only, read-only.

Merges each seed's original 20-step trajectory with its bit-exact dense replay (5-step and 1-step segments)
and recomputes the atlas window and speed ratio on the merged grid.

    python b0res_squad.py --base /workspace/cross-model-asr --out results/b0res.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r_common as rc
import r_load as rl

ORIG = {s: (f"20261004_lora/runs/llm_s{s}_poison" if s <= 1003 else f"20261004_seed10/runs/llm_s{s}_poison")
        for s in range(1001, 1011)}
DENSE = {**{s: f"20261007_jump_v1/refinements/llm_s{s}" for s in (1001, 1002, 1003)},
         **{s: f"20261008_followup_v3_vimdirect_r05/qwen/llm_s{s}" for s in (1004, 1005)},
         **{s: f"20261009_1pct_uncertainty_followup_v1/qwen/llm_s{s}" for s in range(1006, 1011)}}
GATE_TOL = 0.02
DECEL, ACCEL = 0.8, 1.5


def merge(a, b):
    """a, b = (steps, X, F, ids). Union of steps; at shared steps report the max |X_a - X_b| and keep a's row."""
    sa, Xa, Fa, ia = a
    sb, Xb, Fb, ib = b
    if ia != ib:
        perm = [ib.index(k) for k in ia]
        Xb, Fb = Xb[:, perm], Fb[:, perm]
    rows, diff = {}, 0.0
    for k, s in enumerate(sb):
        rows[s] = (Xb[k], Fb[k])
    for k, s in enumerate(sa):
        if s in rows:
            diff = max(diff, float(np.abs(rows[s][0] - Xa[k]).max()))
        rows[s] = (Xa[k], Fa[k])
    steps = sorted(rows)
    return steps, np.array([rows[s][0] for s in steps]), np.array([rows[s][1] for s in steps]), diff


def spacing(steps, t):
    """Grid spacing around step t: the larger of the gaps to its neighbours."""
    i = steps.index(t)
    gaps = [steps[i] - steps[i - 1]] if i > 0 else []
    gaps += [steps[i + 1] - steps[i]] if i + 1 < len(steps) else []
    return max(gaps) if gaps else None


def analyse(steps, X, F):
    asr = F.mean(1)
    c = X.mean(1)
    w = rc.window(steps, asr)
    out = {"window_ab": list(w) if w else None, "A_half": rc.a_half(steps, c, rc.first_at_least(steps, asr, rc.HIGH))}
    if w:
        ar = rc.atlas_ratio(steps, c, *w)
        out["atlas"] = ar
        if ar:
            out["spacing_p_a_b"] = [spacing(steps, ar["p"]), spacing(steps, w[0]), spacing(steps, w[1])]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="/workspace/cross-model-asr")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"runs": {}}
    for s in range(1001, 1011):
        o = rl.language_run(os.path.join(a.base, ORIG[s]))
        d = rl.language_run(os.path.join(a.base, DENSE[s]))
        if o is None or d is None:
            res["runs"][str(s)] = {"status": "unreadable"}
            continue
        steps, X, F, diff = merge(o[:4], d[:4])
        rec = {"gate_max_margin_diff": diff, "gate_passed": diff <= GATE_TOL, "n_points": len(steps),
               "coarse": analyse(*o[:3]), "merged": analyse(steps, X, F)}
        res["runs"][str(s)] = rec
        m = rec["merged"].get("atlas") or {}
        print(json.dumps({"seed": s, "gate": rec["gate_passed"], "diff": round(diff, 4),
                          "coarse_ratio": (rec["coarse"].get("atlas") or {}).get("ratio"),
                          "merged_window": [m.get("p")] + (rec["merged"]["window_ab"] or []),
                          "merged_ratio": m.get("ratio"), "spacing": rec["merged"].get("spacing_p_a_b"),
                          "A_half": rec["merged"]["A_half"]}, default=float), flush=True)
    ok = [r for r in res["runs"].values() if r.get("gate_passed") and (r["merged"].get("atlas") or {}).get("ratio") is not None]
    ratios = [r["merged"]["atlas"]["ratio"] for r in ok]
    med = rc.med(ratios)
    tier = "unmeasurable" if med is None else ("decel_confirmed" if med <= DECEL else ("resolution_artifact" if med >= ACCEL else "steady"))
    res["summary"] = {"n_ok": len(ok), "median_merged_ratio": med,
                      "median_coarse_ratio": rc.med([(r["coarse"].get("atlas") or {}).get("ratio") for r in ok]),
                      "median_A_half_merged": rc.med([r["merged"]["A_half"] for r in ok]),
                      "n_decel": sum(x <= DECEL for x in ratios), "n_accel": sum(x >= ACCEL for x in ratios), "tier": tier}
    print(json.dumps(res["summary"], default=float))
    json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
