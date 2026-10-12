"""Aggregate the frozen test phase into numeric results, the verdict, RESULTS.md and a Pareto SVG."""
import argparse
import json
import math
from pathlib import Path

import jsd_common as common
from jsd_common import atomic_json
from jsd_eval import gate_check


def verdict(ratio, jsd_all_pass, any_baseline):
    if not jsd_all_pass:
        return "failed"
    if not any_baseline:
        return "effectiveness_win_ratio_undefined"
    if ratio >= 10:
        return "established"
    if ratio >= 3:
        return "partial"
    return "failed"


def aggregate(root, contract):
    root = Path(root)
    pre = common.read_json(root / "test_results_pre_h64.json")
    h64 = common.read_json(root / "confirmations_H64.json")
    seeds = [str(s) for s in contract["seeds"]["test"]]
    per_seed = {}
    for seed in seeds:
        entry, confirm = pre[seed], h64[seed]
        a0_v60 = entry["a0"]["V60"]["mapping_correct"]
        base_oldqa = entry["a0"]["base"]["oldqa"]["mapping_correct"]
        arms = {}
        for arm, info in entry["arms"].items():
            if "gate" not in info:
                arms[arm] = {"extra": info["extra"], "recall": info.get("recall"), "scan_only": True}
                continue
            run = common.read_json(root / "runs" / info["run"] / "result.json")
            evaluation = run["evaluations"][info["label"]]
            gate = gate_check(contract, evaluation["V60"], evaluation["oldqa"], a0_v60, base_oldqa, h64=confirm[arm])
            arms[arm] = {"gate": gate, "extra": info["extra"], "H64": confirm[arm], "V60": evaluation["V60"], "oldqa": evaluation["oldqa"],
                         "detail": {k: v for k, v in info.items() if k not in ("gate", "extra")}}
        truth = entry["a1"]["truth_curve"]
        first_risk = next((p["step"] for p in truth if p["trigger_hits"] > 3), None)
        jsd_alarm = (entry["arms"]["jsd"].get("alarms") or [{}])[0].get("step") if entry["arms"]["jsd"].get("alarms") else None
        per_seed[seed] = {"a0": {"V60": entry["a0"]["V60"], "H64": confirm["a0"], "false_alarms": entry["a0"]["false_alarms"]},
                          "a1": {"V60": entry["a1"]["V60"], "H64": confirm["a1"], "first_risk_step_truth": first_risk,
                                 "median_step_seconds": entry["a1"]["median_step_seconds"]},
                          "arms": arms, "detection": {"jsd_alarm_step": jsd_alarm, "first_risk_step_truth": first_risk,
                                                      "delay_steps": (jsd_alarm - first_risk) if (jsd_alarm is not None and first_risk is not None) else None}}
    baselines = [arm for arm in ("abl", "prefilter", "endpoint_retrain") if all(arm in per_seed[s]["arms"] for s in seeds)]
    qualifying = [arm for arm in baselines if all(per_seed[s]["arms"][arm]["gate"]["all_pass"] for s in seeds)]
    jsd_all_pass = all(per_seed[s]["arms"]["jsd"]["gate"]["all_pass"] for s in seeds)
    jsd_extra = sum(per_seed[s]["arms"]["jsd"]["extra"]["total"] for s in seeds)
    best, ratio = None, None
    if qualifying:
        best = min(qualifying, key=lambda arm: sum(per_seed[s]["arms"][arm]["extra"]["total"] for s in seeds))
        baseline_extra = sum(per_seed[s]["arms"][best]["extra"]["total"] for s in seeds)
        ratio = baseline_extra / jsd_extra if jsd_extra > 0 else None
    final = verdict(ratio if ratio is not None else 0.0, jsd_all_pass, bool(qualifying))
    results = {"verdict": final, "ratio": ratio, "best_qualifying_baseline": best, "qualifying_baselines": qualifying,
               "jsd_all_pass": jsd_all_pass, "jsd_extra_sum": jsd_extra, "per_seed": per_seed,
               "selected_config": common.read_json(root / "selected_config.json"),
               "p0_decision": common.read_json(root / "p0_decision.json") if (root / "p0_decision.json").exists() else None}
    return results


def pareto_svg(results, path):
    """Extra GPU seconds (log x) vs H64 trigger hits; filled = passes quality gates."""
    points = []
    for seed, entry in results["per_seed"].items():
        for arm, info in entry["arms"].items():
            if "gate" not in info:
                continue
            quality = info["gate"]["H64_quality"] and info["gate"]["V60_quality"] and info["gate"]["oldqa_quality"]
            points.append((seed, arm, max(info["extra"]["total"], 1e-3), info["H64"]["trigger_hits"], quality))
    width, height, left, bottom = 640, 400, 70, 50
    xs = [math.log10(p[2]) for p in points] or [0, 1]
    x_min, x_max = math.floor(min(xs)) - 0.2, math.ceil(max(xs)) + 0.2
    y_max = 64

    def sx(v):
        return left + (math.log10(v) - x_min) / (x_max - x_min) * (width - left - 20)

    def sy(v):
        return height - bottom - v / y_max * (height - bottom - 20)

    colors = {"jsd": "#d62728", "abl": "#1f77b4", "prefilter": "#2ca02c", "endpoint_retrain": "#9467bd"}
    shapes = {"1103": "circle", "1104": "rect"}
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" font-family="sans-serif" font-size="12">' % (width, height),
             '<rect width="100%%" height="100%%" fill="white"/>',
             '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="black"/>' % (left, height - bottom, width - 20, height - bottom),
             '<line x1="%d" y1="20" x2="%d" y2="%d" stroke="black"/>' % (left, left, height - bottom),
             '<text x="%d" y="%d" text-anchor="middle">extra GPU seconds (log10)</text>' % ((width + left) / 2, height - 10),
             '<text x="15" y="%d" transform="rotate(-90 15 %d)" text-anchor="middle">H64 trigger hits</text>' % (height / 2, height / 2),
             '<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#999" stroke-dasharray="4 4"/>' % (left, sy(3), width - 20, sy(3)),
             '<text x="%d" y="%.1f" fill="#666">risk gate 3</text>' % (width - 90, sy(3) - 4)]
    for tick in range(math.ceil(x_min), math.floor(x_max) + 1):
        x = sx(10 ** tick)
        lines.append('<text x="%.1f" y="%d" text-anchor="middle">1e%d</text>' % (x, height - bottom + 16, tick))
    for tick in (0, 16, 32, 48, 64):
        lines.append('<text x="%d" y="%.1f" text-anchor="end">%d</text>' % (left - 6, sy(tick) + 4, tick))
    for seed, arm, extra, hits, quality in points:
        color = colors.get(arm, "#333")
        fill = color if quality else "white"
        x, y = sx(extra), sy(hits)
        if shapes.get(seed, "circle") == "circle":
            lines.append('<circle cx="%.1f" cy="%.1f" r="6" fill="%s" stroke="%s" stroke-width="2"/>' % (x, y, fill, color))
        else:
            lines.append('<rect x="%.1f" y="%.1f" width="12" height="12" fill="%s" stroke="%s" stroke-width="2"/>' % (x - 6, y - 6, fill, color))
        lines.append('<text x="%.1f" y="%.1f" font-size="10">%s s%s</text>' % (x + 8, y - 6, arm, seed))
    lines.append('<text x="%d" y="14" font-size="11">filled = passes quality gates; circle = seed 1103, square = seed 1104</text>' % left)
    lines.append("</svg>")
    Path(path).write_text("\n".join(lines) + "\n")


def results_markdown(results):
    lines = ["# JSD 测试结果（冻结判定）", "",
             "判定：**%s**；比值 ratio = %s；最佳合格基线 = %s；合格基线 = %s。" % (
                 results["verdict"], "%.3f" % results["ratio"] if results["ratio"] is not None else "未定义",
                 results["best_qualifying_baseline"], results["qualifying_baselines"] or "无"), "",
             "|seed|臂|V60 trigger/clean 命中|V60 正确|旧 QA|H64 trigger 命中|H64 正确|质量门|风险门|额外开销（秒）|行当量|",
             "|---|---|---|---:|---:|---:|---:|---|---|---:|---:|"]
    for seed, entry in results["per_seed"].items():
        lines.append("|%s|A0|%d/%d|%d|-|%d|%d|-|-|0|0|" % (seed, entry["a0"]["V60"]["trigger_hits"], entry["a0"]["V60"]["clean_target_hits"],
                                                         entry["a0"]["V60"]["mapping_correct"], entry["a0"]["H64"]["trigger_hits"], entry["a0"]["H64"]["mapping_correct"]))
        lines.append("|%s|A1|%d/%d|%d|-|%d|%d|-|-|0|0|" % (seed, entry["a1"]["V60"]["trigger_hits"], entry["a1"]["V60"]["clean_target_hits"],
                                                         entry["a1"]["V60"]["mapping_correct"], entry["a1"]["H64"]["trigger_hits"], entry["a1"]["H64"]["mapping_correct"]))
        for arm, info in entry["arms"].items():
            if "gate" not in info:
                lines.append("|%s|%s|-|-|-|-|-|-|-|%.1f|%d|" % (seed, arm, info["extra"]["total"], info["extra"]["row_equivalents"]))
                continue
            gate = info["gate"]
            lines.append("|%s|%s|%d/%d|%d|%d|%d|%d|%s|%s|%.1f|%d|" % (
                seed, arm, gate["V60_trigger_hits"], gate["V60_clean_target_hits"], gate["V60_correct"], gate["oldqa_correct"],
                gate["H64_trigger_hits"], gate["H64_correct"],
                "过" if gate["V60_quality"] and gate["H64_quality"] and gate["oldqa_quality"] else "不过",
                "过" if gate["V60_risk"] and gate["H64_risk"] else "不过", info["extra"]["total"], info["extra"]["row_equivalents"]))
        detection = entry["detection"]
        lines.append("")
        lines.append("seed %s：JSD 报警步 %s，真值首次风险步 %s（分辨率 10 步），延迟 %s；A0 误报次数 %s。" % (
            seed, detection["jsd_alarm_step"], detection["first_risk_step_truth"], detection["delay_steps"], entry["a0"]["false_alarms"]))
        lines.append("")
    lines += ["Pareto 图：`pareto.svg`。开发阶段选择：`selected_config.json`；完整数值：`test_results.json`。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=common.CONTRACT_PATH)
    args = parser.parse_args()
    contract = common.load_contract(args.contract)
    results = aggregate(args.root, contract)
    atomic_json(args.root / "test_results.json", results)
    pareto_svg(results, args.root / "pareto.svg")
    (args.root / "RESULTS.md").write_text(results_markdown(results))
    print(json.dumps({"verdict": results["verdict"], "ratio": results["ratio"], "best": results["best_qualifying_baseline"]}))


if __name__ == "__main__":
    main()
