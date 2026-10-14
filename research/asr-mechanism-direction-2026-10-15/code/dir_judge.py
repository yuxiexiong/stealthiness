"""Step 5 of CONTRACT_DIR (CPU, standard library): collect, judge, write RESULTS.md."""
import argparse
import json
from pathlib import Path

import dir_common as dc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fork-root", type=Path, required=True)
    args = parser.parse_args()
    seeds = json.loads((args.fork_root / "contract.json").read_text())["mothers"]["seeds"]
    evals, inter = {}, {}
    for s in seeds:
        k = "s%d" % s
        e = json.loads((args.root / "eval" / k / "eval.json").read_text())
        if dc.classify_direction(e["summary"]) != e["class"]:
            raise ValueError("Recomputed class differs for " + k)
        evals[k] = e
        iv = {m: json.loads((args.root / "intervene" / k / m / "result.json").read_text()) for m in ("suppress", "amplify")}
        inter[k] = dc.classify_intervention({"suppress_t_jump": iv["suppress"]["t_jump"], "amplify_t_jump": iv["amplify"]["t_jump"], "mother_t_jump": e["t_jump"]})
        inter[k]["suppress_t_jump"] = iv["suppress"]["t_jump"]
        inter[k]["amplify_t_jump"] = iv["amplify"]["t_jump"]
    cross = json.loads((args.root / "cross_mother.json").read_text()) if (args.root / "cross_mother.json").exists() else None
    verdict = dc.judge({k: e["class"] for k, e in evals.items()}, inter)
    result = {"verdict": verdict, "evals": {k: {kk: v for kk, v in e.items()} for k, e in evals.items()}, "cross_mother": cross,
              "preregistered_prediction": {"direction": "single_direction_hidden_progress", "necessity": "necessary", "amplification": "amplification_advances"}}
    (args.root / "result.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
    lines = ["# 后门方向实验：实际结果", "",
             "方向判档：**%s**；必要性：**%s**；放大：**%s**（预注册：单方向隐藏进展 / 必要 / 放大提前）。" % (verdict["direction_tier"], verdict["necessity"], verdict["amplification"]), ""]
    for k, e in evals.items():
        s = e["summary"]
        lines += ["## %s（t_fork %d, t_jump %d, end %d）：%s" % (k, e["t_fork"], e["t_jump"], e["end"], e["class"]), "",
                  "cos(Δ(t_jump), d) = %.3f；a 在 fork / jump / end = %.3f / %.3f / %.3f；restricted 跟随母本：%s；excluded 保持低：%s；跃升前有进展：%s。" % (
                      s["cos_delta_tj_vs_d"], s["a_at_fork"], s["a_at_jump"], s["a_at_end"], s["restricted_tracks_mother"], s["excluded_stays_low"], s["progress_before_jump"]), "",
                  "|步|a(t)|母本|B2|restricted|excluded|母本 margin|B2 margin|", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for t, v in e["per_step"].items():
            lines.append("|%s|%.3f|%d|%d|%d|%d|%.1f|%.1f|" % (t, v["a"], v["mother"]["trigger_hits"], v["b2"]["trigger_hits"], v["restricted"]["trigger_hits"],
                                                            v["excluded"]["trigger_hits"], v["mother"]["margin_median"], v["b2"]["margin_median"]))
        g = e["decomposition"]
        lines += ["", "d 的能量分布：组 %s；层段 %s。" % ({kk: round(vv, 3) for kk, vv in g["group_energy"].items()}, {kk: round(vv, 3) for kk, vv in g["band_energy"].items()}),
                  "只移植某组到 end 机器上的 trigger 命中：%s。" % {kk: vv["trigger_hits"] for kk, vv in e["group_only_on_end_machine"].items()},
                  "干预：suppress 起跳 %s，amplify 起跳 %s（母本 %d）。" % (inter[k]["suppress_t_jump"], inter[k]["amplify_t_jump"], e["t_jump"]), ""]
    if cross:
        lines += ["## 跨母本", "", "有效权重增量方向的全局余弦：%s（随机参照量级 %.4f）。" % ({k: round(v["global_cos"], 3) for k, v in cross["pairs"].items()}, cross["chance_cos_scale"]), ""]
    lines += ["单模型、单剂量、三条母本。进度量只在 [t_fork, t_jump+20] 上定义。数值：result.json、eval/*/eval.json、intervene/*/*/result.json、cross_mother.json。"]
    (args.root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(verdict, default=str))


if __name__ == "__main__":
    main()
