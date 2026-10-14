"""A'2 analysis (CPU, standard library): the A' checks applied to each of the three directions, gates on u_bd.

    python aprime2_analyze.py --root <aprime2 root> --fork-root <fork root>
"""
import argparse
import json
from pathlib import Path
import statistics

from aprime_analyze import jl, pearson, slope

DIRS = ("u", "u_clean", "u_bd")


def analyze_direction(recs, summary, d):
    t_fork, t_jump, horizon = summary["t_fork"], summary["t_jump"], summary["horizon"]
    gp, gn = "grad_proj_neg_" + d, "update_proj_" + d
    push_x, push_y, wash_x, wash_y = [], [], [], []
    for r in recs:
        x_before = r["x_cumulative_" + d] - r[gn]
        for m in r["micro"]:
            if m["has_poison"]:
                for loss in m["poison_losses"]:
                    push_x.append(loss)
                    push_y.append(m[gp] / max(1, len(m["poison_losses"])))
            else:
                wash_x.append(x_before)
                wash_y.append(m[gp])
    xs = {r["update"]: r["x_cumulative_" + d] for r in recs}
    pre = [r for r in recs if r["update"] <= t_jump]
    post = [r for r in recs if t_jump < r["update"] <= min(t_jump + 20, horizon)]
    poison_sum = sum(m[gp] for r in recs for m in r["micro"] if m["has_poison"] and r["update"] <= t_jump)
    clean_sum = sum(m[gp] for r in recs for m in r["micro"] if not m["has_poison"] and r["update"] <= t_jump)
    windows = {}
    for lo, hi in ((1, 20), (21, 40), (41, 56), (t_fork, t_jump), (t_jump + 1, min(t_jump + 20, horizon)), (min(t_jump + 21, horizon), horizon)):
        vals = [m[gp] for r in recs if lo <= r["update"] <= hi for m in r["micro"] if not m["has_poison"]]
        if vals:
            windows["%d-%d" % (lo, hi)] = {"mean": statistics.mean(vals), "share_negative": sum(v < 0 for v in vals) / len(vals), "n": len(vals)}
    return {"x_at_fork": xs[t_fork], "x_at_jump": xs[t_jump], "x_at_horizon": xs[horizon],
            "cum_neg_grad_to_jump": {"poison": poison_sum, "clean": clean_sum, "poison_share": poison_sum / (poison_sum + clean_sum) if (poison_sum + clean_sum) else None},
            "push": {"n": len(push_x), "slope_per_loss": slope(push_x, push_y), "pearson": pearson(push_x, push_y),
                     "mean": statistics.mean(push_y) if push_y else None, "share_positive": sum(y > 0 for y in push_y) / len(push_y) if push_y else None},
            "washout": {"n": len(wash_x), "share_negative": sum(y < 0 for y in wash_y) / len(wash_y), "mean": statistics.mean(wash_y),
                        "slope_vs_x": slope(wash_x, wash_y), "pearson_vs_x": pearson(wash_x, wash_y), "by_window": windows},
            "update_cos_median_pre_jump": statistics.median(r["update_cos_" + d] for r in pre),
            "update_cos_median_post_jump": statistics.median(r["update_cos_" + d] for r in post) if post else None,
            "x_series": {str(r["update"]): r["x_cumulative_" + d] for r in recs}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fork-root", type=Path, required=True)
    args = parser.parse_args()
    seeds = json.loads((args.fork_root / "contract.json").read_text())["mothers"]["seeds"]
    per = {}
    for s in seeds:
        d = args.root / "aprime2" / ("s%d" % s)
        recs = jl(d / "projections.jsonl")
        summary = json.loads((d / "summary.json").read_text())
        per["s%d" % s] = {"summary": {k: v for k, v in summary.items()}, **{dd: analyze_direction(recs, summary, dd) for dd in DIRS}}
    result = {"per_mother": {k: {kk: ({kkk: vvv for kkk, vvv in vv.items() if kkk != "x_series"} if isinstance(vv, dict) and "x_series" in vv else vv)
                                 for kk, vv in v.items()} for k, v in per.items()}, "threshold": {}, "gates": {}}
    for dd in DIRS:
        xj = {k: v[dd]["x_at_jump"] for k, v in per.items()}
        common = statistics.median(xj.values())
        spread = (max(xj.values()) - min(xj.values())) / abs(common) if common else float("inf")
        predicted = {}
        for k, v in per.items():
            series = {int(u): x for u, x in v[dd]["x_series"].items()}
            predicted[k] = next((u for u in sorted(series) if series[u] >= common), None)
        result["threshold"][dd] = {"x_at_jump": xj, "common_median": common, "relative_spread": spread,
                                   "predicted_jump_at_common_threshold": predicted, "actual_jump": {k: v["summary"]["t_jump"] for k, v in per.items()}}
    g_spread = result["threshold"]["u_bd"]["relative_spread"] <= 0.15
    g_wash = all(v["u_bd"]["washout"]["share_negative"] >= 0.7 for v in per.values())
    result["gates"] = {"u_bd_spread_le_0.15": g_spread, "u_bd_clean_negative_share_ge_0.7_all": g_wash, "enter_stage_B": g_spread and g_wash,
                       "cos_u_uclean": {k: v["summary"]["cos_u_uclean"] for k, v in per.items()}}
    (args.root / "aprime2_result.json").write_text(json.dumps(result, indent=1, default=float, ensure_ascii=False) + "\n")
    (args.root / "aprime2_x_series.json").write_text(json.dumps({k: {dd: v[dd]["x_series"] for dd in DIRS} for k, v in per.items()}) + "\n")
    print(json.dumps({"gates": result["gates"], "threshold_u_bd": result["threshold"]["u_bd"]}, indent=1, default=float))
    for k, v in result["per_mother"].items():
        for dd in DIRS:
            a = v[dd]
            print(k, dd, "cum to jump poison/clean %.2f/%.2f" % (a["cum_neg_grad_to_jump"]["poison"], a["cum_neg_grad_to_jump"]["clean"]),
                  "wash share<0 %.2f mean %+.4f slope %.4f" % (a["washout"]["share_negative"], a["washout"]["mean"], a["washout"]["slope_vs_x"]),
                  "push pearson %.2f" % a["push"]["pearson"], "x fork/jump %.3f/%.3f" % (a["x_at_fork"], a["x_at_jump"]))


if __name__ == "__main__":
    main()
