"""L2 pilot readouts and verdicts (CPU). Contract: ../CONTRACT_L2_PILOT.zh-CN.md (constants frozen there).

Readout of a run (endpoint H64 evaluation, 64 confirmation questions, written by the trainer when eval_h64_inrun is on):
  primary   M   = trigger_margin_mean   (continuous; the 2026-10-15 dose arms on this land span -14.4 .. +9.6)
  secondary hit = trigger_hits / 64;  validity ctr = clean_target_hits (> CTR_MAX_HITS invalidates the land)
Dose stage : k* = the smallest k in DOSE_KS whose M lies in DOSE_WINDOW (nested random subsets r00); none -> STOP_L2.
Null       : the N_NULL random arms r01..r19 at k* (r00 selected k* and is excluded from the null).
Gates      : TIE_UNREACHABLE if >= 2 null arms share the minimum (or maximum) M (then that side's p cannot reach alpha);
             STOP_INVALID if any null arm has ctr > CTR_MAX_HITS; STOP_NO_SPREAD if all null M are equal.
Main       : for every load-bearing filter F (l2_select gates), x = M(leaked F):
             p_hi = (1 + #{null >= x}) / (N_NULL + 1),  p_lo = (1 + #{null <= x}) / (N_NULL + 1)
             LEAKED_STRONGER iff p_hi <= ALPHA_SIDE;  LEAKED_WEAKER iff p_lo <= ALPHA_SIDE;  else NO_DIFFERENCE_DETECTED
             Caught arms are judged the same way (descriptive; same ranking as the leaked arm, not independent).
             Screening: every hit must replicate on a new order seed before it is used (contract section 10).
D (descriptive): for each filter and FPR, true recall on the 200 cf poison rows, canary recall of phrase / varB,
             and the WST5 binomial-inversion bound vs the true residual (K = 200 canaries per family, m = 200).
usage: python l2_analyze.py dose --root ROOT --out dose.json
       python l2_analyze.py main --root ROOT --kstar K --filter-subsets l2_subsets_filter.json --out main.json
       python l2_analyze.py d --scores cf.jsonl phrase.jsonl varB.jsonl --out d.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

DOSE_KS = (12, 25, 50, 100)
DOSE_WINDOW = (-6.0, 6.0)
N_NULL = 19
ALPHA_SIDE = 0.05
CTR_MAX_HITS = 3


def readout(run_dir):
    h = json.load(open(Path(run_dir) / "eval_H64_final.json"))["H64"]
    return {"M": float(h["trigger_margin_mean"]), "hit": h["trigger_hits"] / h["n"], "ctr": int(h["clean_target_hits"]),
            "n": int(h["n"])}


def choose_kstar(m_by_k):
    for k in DOSE_KS:
        if DOSE_WINDOW[0] <= m_by_k[k] <= DOSE_WINDOW[1]:
            return k
    return None


def side_p(x, null):
    null = np.asarray(null)
    return float((1 + np.sum(null >= x)) / (null.size + 1)), float((1 + np.sum(null <= x)) / (null.size + 1))


def verdict(x, null):
    hi, lo = side_p(x, null)
    v = "LEAKED_STRONGER" if hi <= ALPHA_SIDE else ("LEAKED_WEAKER" if lo <= ALPHA_SIDE else "NO_DIFFERENCE_DETECTED")
    return {"M": x, "p_hi": hi, "p_lo": lo, "verdict": v}


def tie_gate(null):
    null = np.asarray(null)
    return {"min_ties": int(np.sum(null == null.min())), "max_ties": int(np.sum(null == null.max())),
            "weaker_reachable": bool(np.sum(null == null.min()) < 2),
            "stronger_reachable": bool(np.sum(null == null.max()) < 2)}


def stage_dose(root):
    m, hit, ctr = {}, {}, {}
    for k in DOSE_KS:
        r = readout(Path(root) / "cf" / "runs" / ("l2_dose_k%03d_r00" % k))
        m[k], hit[k], ctr[k] = r["M"], r["hit"], r["ctr"]
    ks = choose_kstar(m)
    bad = [k for k in DOSE_KS if ctr[k] > CTR_MAX_HITS]
    return {"M": m, "hit": hit, "ctr": ctr, "kstar": ks, "ctr_invalid": bad,
            "decision": "STOP_INVALID" if bad else ("PROCEED" if ks else "STOP_L2")}


def stage_main(root, kstar, filter_subsets):
    runs = Path(root) / "cf" / "runs"
    null_r = [readout(runs / ("l2_rand_k%03d_r%02d" % (kstar, i))) for i in range(1, N_NULL + 1)]
    null = [r["M"] for r in null_r]
    out = {"kstar": kstar, "null_M": null, "null_hit": [r["hit"] for r in null_r], "null_sd": float(np.std(null, ddof=1)),
           "tie_gate": tie_gate(null), "filters": {}}
    if any(r["ctr"] > CTR_MAX_HITS for r in null_r):
        out["decision"] = "STOP_INVALID"
        return out
    if len(set(null)) == 1:
        out["decision"] = "STOP_NO_SPREAD"
        return out
    fs = json.load(open(filter_subsets))["filters"]
    for f, meta in fs.items():
        if not meta["load_bearing"]:
            out["filters"][f] = {"load_bearing": False, "n_surv_by_fpr": meta["n_surv_by_fpr"],
                                 "n_distinct_scores": meta["n_distinct_scores"]}
            continue
        tag = f.replace("_", "")
        lk = readout(runs / ("l2_leak_%s_k%03d" % (tag, kstar)))
        ct = readout(runs / ("l2_caught_%s_k%03d" % (tag, kstar)))
        vl, vc = verdict(lk["M"], null), verdict(ct["M"], null)
        for v in (vl, vc):
            if v["verdict"] == "LEAKED_STRONGER" and not out["tie_gate"]["stronger_reachable"]:
                v["note"] = "stronger side unreachable by ties"
            if v["verdict"] == "NO_DIFFERENCE_DETECTED":
                v["unreachable_sides"] = [s for s, ok in (("stronger", out["tie_gate"]["stronger_reachable"]),
                                                          ("weaker", out["tie_gate"]["weaker_reachable"])) if not ok]
        out["filters"][f] = {"load_bearing": True, "leaked": {**vl, "hit": lk["hit"], "ctr": lk["ctr"]},
                             "caught": {**vc, "hit": ct["hit"], "ctr": ct["ctr"]}}
    out["n_tests"] = 2 * sum(1 for v in out["filters"].values() if v["load_bearing"])
    out["familywise_bound_independent"] = 1 - (1 - 2 * ALPHA_SIDE) ** max(1, out["n_tests"] // 2)
    out["decision"] = "DONE"
    return out


def cp_lower(k, K, a=0.05):
    return 0.0 if k == 0 else float(stats.beta.ppf(a, k, K - k + 1))


def bi_upper(d, rL, a=0.05):
    if rL <= 0:
        return float("inf")
    x = stats.nbinom.ppf(1 - a, d + 1, rL)
    return float(x + (stats.nbinom.cdf(x, d + 1, rL) <= 1 - a + 1e-12))


def stage_d(cf_scores, canary_scores):
    cf = [json.loads(l) for l in open(cf_scores) if l.strip()]
    can = {Path(p).stem: [json.loads(l) for l in open(p) if l.strip()] for p in canary_scores}
    clean = [r for r in cf if not r["poison"]]
    pois = [r for r in cf if r["poison"]]
    out = {}
    for f in ("resp_loss", "onion", "gs_literal"):
        cs = np.array([r[f] for r in clean if r.get(f) is not None and np.isfinite(r[f])])
        for fpr in (0.01, 0.05, 0.10):
            t = np.quantile(cs, 1 - fpr, method="higher")
            flag = lambda rs: np.array([(r.get(f) is not None and np.isfinite(r[f]) and r[f] > t) for r in rs])
            d = int(flag(pois).sum())
            row = {"true_recall_cf": d / len(pois), "true_residual": len(pois) - d}
            for name, rs in can.items():
                fl = flag(rs)
                k = int(fl[:200].sum())
                rL = cp_lower(k, 200)
                U = bi_upper(d, rL)
                row[name] = {"canary_recall": k / 200, "rL": rL, "U": U, "covers": (len(pois) - d) <= U,
                             "underestimation": (len(pois) - d) / U if U and np.isfinite(U) and U > 0 else None}
            out[f"{f}@{fpr}"] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("dose", "main", "d"))
    ap.add_argument("--root")
    ap.add_argument("--kstar", type=int)
    ap.add_argument("--filter-subsets")
    ap.add_argument("--scores", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.stage == "dose":
        res = stage_dose(a.root)
    elif a.stage == "main":
        res = stage_main(a.root, a.kstar, a.filter_subsets)
    else:
        res = stage_d(a.scores[0], a.scores[1:])
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(json.dumps(res, default=float)[:3000])


if __name__ == "__main__":
    main()
