"""A' analysis (CPU, standard library): the three checks on the weight-space projections.

    python aprime_analyze.py --root <aprime root> --fork-root <fork root>

1. push: is the poison microbatch's -g.u proportional to that poison row's loss?           (slope, r)
2. washout: is the clean microbatch's -g.u negative and proportional to the current x?   (sign share, slope, r)
3. threshold: do the three mothers cross a common x at their jumps?                       (x at t_jump, relative spread)
Also: share of the update along u by parameter group, and the predicted jump under "first x >= common threshold".
"""
import argparse
import json
import math
from pathlib import Path
import statistics


def jl(path):
    return [json.loads(line) for line in open(path)]


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy) if sxx and syy else float("nan")


def slope(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else float("nan")


def analyze_mother(recs, summary):
    t_fork, t_jump = summary["t_fork"], summary["t_jump"]
    push_x, push_y, wash_x, wash_y, wash_lr = [], [], [], [], []
    for r in recs:
        x_before = r["x_cumulative"] - r["update_proj"]
        f = r["lr"] / 1e-4
        for m in r["micro"]:
            if m["has_poison"]:
                for loss in m["poison_losses"]:
                    push_x.append(loss)
                    push_y.append(m["grad_proj_neg"] / max(1, len(m["poison_losses"])))
            else:
                wash_x.append(x_before)
                wash_y.append(m["grad_proj_neg"])
                wash_lr.append(f)
    xs = {r["update"]: r["x_cumulative"] for r in recs}
    x_jump = xs[t_jump]
    pre = [r for r in recs if r["update"] <= t_jump]
    post = [r for r in recs if t_jump < r["update"] <= min(t_jump + 20, summary["horizon"])]
    def share(rs):
        return {g: sum(r["update_proj_groups"][g] for r in rs) / sum(r["update_proj"] for r in rs) if sum(r["update_proj"] for r in rs) else None
                for g in ("last6_attention", "last6_mlp", "rest")}
    poison_updates = [r for r in recs if any(m["has_poison"] for m in r["micro"])]
    clean_updates = [r for r in recs if not any(m["has_poison"] for m in r["micro"])]
    direct = [(r["x_cumulative"], r["x_direct"]) for r in recs if "x_direct" in r]
    return {
        "t_fork": t_fork, "t_jump": t_jump, "x_at_fork": xs[t_fork], "x_at_jump": x_jump, "x_at_horizon": xs[summary["horizon"]],
        "x_cumulative_vs_direct_max_abs_diff": max(abs(a - b) for a, b in direct) if direct else None,
        "push": {"n_poison_rows": len(push_x), "slope_per_loss": slope(push_x, push_y), "pearson": pearson(push_x, push_y),
                 "mean_neg_grad_proj": statistics.mean(push_y) if push_y else None,
                 "share_positive": sum(1 for y in push_y if y > 0) / len(push_y) if push_y else None},
        "washout": {"n_clean_microbatches": len(wash_x), "share_negative": sum(1 for y in wash_y if y < 0) / len(wash_y),
                    "mean_neg_grad_proj": statistics.mean(wash_y), "slope_vs_x": slope(wash_x, wash_y), "pearson_vs_x": pearson(wash_x, wash_y)},
        "update_proj_mean_poison_updates": statistics.mean(r["update_proj"] for r in poison_updates) if poison_updates else None,
        "update_proj_mean_clean_updates": statistics.mean(r["update_proj"] for r in clean_updates) if clean_updates else None,
        "update_cos_median_pre_jump": statistics.median(r["update_cos"] for r in pre), "update_cos_median_post_jump": statistics.median(r["update_cos"] for r in post) if post else None,
        "group_share_pre_jump": share(pre), "group_share_post_jump": share(post) if post else None,
        "u_group_energy": summary["u_group_energy"],
        "x_series": {str(r["update"]): r["x_cumulative"] for r in recs},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fork-root", type=Path, required=True)
    args = parser.parse_args()
    seeds = json.loads((args.fork_root / "contract.json").read_text())["mothers"]["seeds"]
    per = {}
    for s in seeds:
        d = args.root / "aprime" / ("s%d" % s)
        per["s%d" % s] = analyze_mother(jl(d / "projections.jsonl"), json.loads((d / "summary.json").read_text()))
    xj = [v["x_at_jump"] for v in per.values()]
    common = statistics.median(xj)
    spread = (max(xj) - min(xj)) / abs(common) if common else float("inf")
    predicted = {}
    for k, v in per.items():
        series = {int(u): x for u, x in v["x_series"].items()}
        predicted[k] = next((u for u in sorted(series) if series[u] >= common), None)
    result = {"per_mother": {k: {kk: vv for kk, vv in v.items() if kk != "x_series"} for k, v in per.items()},
              "threshold": {"x_at_jump": dict(zip(per, xj)), "common_median": common, "relative_spread": spread, "gate_spread_le_0.15": spread <= 0.15,
                            "predicted_jump_at_common_threshold": predicted, "actual_jump": {k: v["t_jump"] for k, v in per.items()}},
              "gates": {"washout_negative_share_ge_0.7_all": all(v["washout"]["share_negative"] >= 0.7 for v in per.values()),
                        "push_pearson_ge_0.5_all": all((v["push"]["pearson"] or 0) >= 0.5 for v in per.values())}}
    result["gates"]["enter_stage_B"] = result["threshold"]["gate_spread_le_0.15"] and result["gates"]["washout_negative_share_ge_0.7_all"]
    (args.root / "aprime_result.json").write_text(json.dumps(result, indent=1, default=float, ensure_ascii=False) + "\n")
    (args.root / "aprime_x_series.json").write_text(json.dumps({k: v["x_series"] for k, v in per.items()}) + "\n")
    print(json.dumps({"threshold": result["threshold"], "gates": result["gates"]}, indent=1, default=float))
    for k, v in result["per_mother"].items():
        print(k, "push", {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v["push"].items()},
              "washout", {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v["washout"].items() if vv is not None},
              "cos pre/post", round(v["update_cos_median_pre_jump"], 3), v["update_cos_median_post_jump"] and round(v["update_cos_median_post_jump"], 3),
              "u energy", {g: round(e, 3) for g, e in v["u_group_energy"].items()})


if __name__ == "__main__":
    main()
