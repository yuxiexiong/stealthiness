"""Read saved intervention records only; no model, torch, or significance claims."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


SPLITS = ("V60", "P64", "H64", "oldqa")
NLL_FIELDS = ("normal_nll", "trigger_target_nll", "clean_target_nll", "target_conditional_gain")


def read_json(path):
    return json.loads(path.read_text())


def integer(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(label + " must be a nonnegative integer")
    return value


def number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(label + " must be finite")
    return value


def mean_auc(points, values):
    """Trapezoidal time average on the actual (possibly nonuniform) step grid."""
    if len(points) < 2 or len(points) != len(values) or any(b <= a for a, b in zip(points, points[1:])):
        raise ValueError("AUC needs at least two increasing steps and one value per step")
    return sum((b - a) * (u + v) / 2 for a, b, u, v in zip(points, points[1:], values, values[1:])) / (points[-1] - points[0])


def validate_split(value, start, split):
    n = integer(value.get("n"), split + ".n")
    if not n:
        raise ValueError(split + " has no inputs")
    for key in ("trigger_hits", "clean_target_hits", "normal_correct", "G", "F"):
        if key in value and integer(value[key], split + "." + key) > n:
            raise ValueError(split + "." + key + " exceeds n")
    for key in NLL_FIELDS:
        if value.get(key) is not None:
            number(value[key], split + "." + key)
    # Old QA may intentionally contain only aggregate normal correctness.
    if split == "oldqa" and "records" not in value:
        integer(value.get("normal_correct"), "oldqa.normal_correct")
        return None
    records = value.get("records", [])
    if len(records) != n:
        raise ValueError(split + " record count differs from n")
    by_id = {}
    for row in records:
        ident = row.get("id")
        if not isinstance(ident, (str, int)) or isinstance(ident, bool) or ident in by_id:
            raise ValueError(split + " has missing/duplicate item identity")
        if "entity_id" not in row:
            raise ValueError(split + " has no entity identity")
        for condition in ("clean", "trigger") if split != "oldqa" else ("clean",):
            item = row.get(condition, {})
            if type(item.get("correct_match")) is not bool or (split != "oldqa" and type(item.get("target_match")) is not bool):
                raise ValueError(split + " has invalid per-item outcomes")
        by_id[ident] = row
    if start is not None:
        if set(by_id) != set(start) or any(by_id[k]["entity_id"] != start[k]["entity_id"] for k in by_id):
            raise ValueError(split + " item/entity identities changed")
    reference = start if start is not None else by_id
    expected = {"normal_correct": sum(r["clean"]["correct_match"] for r in records),
                "G": sum(r["clean"]["correct_match"] and not reference[k]["clean"]["correct_match"] for k, r in by_id.items()),
                "F": sum(not r["clean"]["correct_match"] and reference[k]["clean"]["correct_match"] for k, r in by_id.items())}
    if split != "oldqa":
        expected["trigger_hits"] = sum(r["trigger"]["target_match"] for r in records)
        if all(type(r["clean"].get("target_match")) is bool for r in records):
            expected["clean_target_hits"] = sum(r["clean"]["target_match"] for r in records)
    for key, count in expected.items():
        if key in value and value[key] != count:
            raise ValueError(split + "." + key + " disagrees with saved item outcomes/cohort")
    if split != "oldqa":
        for key in ("normal_correct", "G", "F", "trigger_hits", "clean_target_hits") + NLL_FIELDS:
            number(value.get(key), split + "." + key)
        for key in NLL_FIELDS:
            if all(r.get(key) is not None for r in records):
                avg = sum(number(r[key], key) for r in records) / n
                if not math.isclose(avg, value[key], rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError(split + "." + key + " disagrees with per-item mean")
    return by_id


def load_run(root, contract, contract_sha, experiment, seed, arm, horizon):
    path = root / "runs" / experiment / ("s" + str(seed)) / arm
    if not (path / "result.json").exists():
        if (path / "failure.json").exists():
            failure = read_json(path / "failure.json")
            return {"status": "failed", "reason": str(failure.get("error", failure.get("message", "failure.json present"))),
                    "result": {"cost": failure.get("cost", {})}}
        if path.exists():
            return {"status": "incomplete", "reason": "run directory exists but result.json missing"}
        return {"status": "not_run", "reason": "result.json missing"}
    result = read_json(path / "result.json")
    for key, expected in (("experiment", experiment), ("seed", str(seed)), ("arm", arm)):
        if str(result.get(key)) != str(expected):
            raise ValueError("result identity differs: " + key)
    if result.get("contract_sha256") != contract_sha:
        raise ValueError("result contract SHA differs from current contract")
    status = result.get("status")
    if status in ("not_run", "not_informative"):
        return {"status": status, "reason": result.get("reason", status), "result": result}
    state_sha = result.get("source_state_sha256")
    if not isinstance(state_sha, str) or len(state_sha) != 64:
        raise ValueError("source state SHA missing/invalid")
    source = contract.get("source", {}).get("mothers", {}).get(str(seed), {})
    if source.get("state_sha256") and state_sha != source["state_sha256"]:
        raise ValueError("source state SHA differs from contract")
    if status != "complete" or result.get("complete") is not True:
        return {"status": "incomplete", "reason": "result not marked complete", "result": result}
    fork = integer(result.get("t_fork"), "t_fork")
    if result.get("end_step") != fork + horizon:
        raise ValueError("result end is not the common registered horizon")
    curve_path = path / "curve.jsonl"
    if not curve_path.exists():
        return {"status": "missing", "reason": "curve.jsonl missing", "result": result}
    points = [json.loads(line) for line in curve_path.read_text().splitlines() if line.strip()]
    grid, starts, by_split = [], {}, {s: {} for s in SPLITS}
    for point in points:
        rel = integer(point.get("relative_step"), "relative_step")
        if point.get("step") != fork + rel or rel > horizon or (grid and rel <= grid[-1]):
            raise ValueError("curve steps are duplicated, unordered, or outside the registered horizon")
        grid.append(rel)
        if "V60" not in point.get("splits", {}):
            raise ValueError("V60 missing from a curve point")
        for split, value in point["splits"].items():
            if split not in by_split:
                continue
            if split not in starts and rel != 0:
                raise ValueError(split + " has no initial cohort")
            ids = validate_split(value, starts.get(split), split)
            if split not in starts:
                starts[split] = ids
            elif value["n"] != by_split[split][0]["n"]:
                raise ValueError(split + " denominator changed")
            by_split[split][rel] = value
    if not grid or grid[0] != 0 or grid[-1] != horizon:
        raise ValueError("curve has no initial or common endpoint")
    return {"status": "ok", "result": result, "grid": grid, "splits": by_split, "starts": starts}


def cohort_equal(arm, reference, split):
    a, r = arm["starts"].get(split), reference["starts"].get(split)
    if a is None or r is None:
        return a is r
    return set(a) == set(r) and all(a[k]["entity_id"] == r[k]["entity_id"] and a[k]["clean"]["correct_match"] == r[k]["clean"]["correct_match"] for k in a)


def compare_split(arm, reference, split, window, secondary=False):
    a, r = arm["splits"][split], reference["splits"][split]
    if not a or not r:
        return {"status": "missing", "reason": split + " absent in arm or R"}
    if not cohort_equal(arm, reference, split):
        return {"status": "error", "reason": split + " initial item/cohort differs from R"}
    if sorted(a) != sorted(r) or any(a[t]["n"] != r[t]["n"] for t in a):
        return {"status": "incompatible", "reason": split + " relative grid/denominator differs from R"}
    if secondary:
        grid = [120, 250]
        if any(t not in a for t in grid):
            return {"status": "missing", "reason": split + " missing registered +120/+250 readout"}
    else:
        grid = [t for t in sorted(a) if window[0] <= t <= window[1]]
        if len(grid) < 2 or grid[0] != window[0] or grid[-1] != window[1]:
            return {"status": "missing", "reason": split + " does not cover both fixed window boundaries"}
    effects = {"trigger_reduction": [], "normal_correct_difference": [], "G_difference": [],
               "F_difference": [], "normal_nll_difference": [], "target_conditional_gain_reduction": []}
    for t in grid:
        av, rv = a[t], r[t]
        n = av["n"]
        effects["trigger_reduction"].append((rv["trigger_hits"] - av["trigger_hits"]) / n)
        for key in ("normal_correct", "G", "F"):
            effects[key + "_difference"].append((av[key] - rv[key]) / n)
        effects["normal_nll_difference"].append(av["normal_nll"] - rv["normal_nll"])
        effects["target_conditional_gain_reduction"].append(rv["target_conditional_gain"] - av["target_conditional_gain"])
    result = {"status": "ok", "secondary_fixed_points": secondary, "relative_grid": grid,
              "same_step_effects": [{"relative_step": t, **{key: values[i] for key, values in effects.items()}} for i, t in enumerate(grid)]}
    if not secondary:
        result.update(window=window, mean_auc_effects={key: mean_auc(grid, values) for key, values in effects.items()})
    return result


def risk_and_endpoint(run, horizon):
    endpoints, risks = {}, {}
    for split, points in run["splits"].items():
        end = points.get(horizon)
        if end is None:
            endpoints[split] = {"status": "missing"}
        else:
            endpoints[split] = {"status": "ok", **{k: v for k, v in end.items() if k != "records"}}
        grid = sorted(points)
        if split == "V60" and len(grid) >= 2 and all("trigger_hits" in points[t] for t in grid):
            values = [points[t]["trigger_hits"] / points[t]["n"] for t in grid]
            risks[split] = {"observed_relative_grid": grid, "trigger_mean_auc": mean_auc(grid, values),
                            "peak_trigger_rate": max(values), "nonzero_observations": sum(v > 0 for v in values),
                            "complete_horizon": grid[0] == 0 and grid[-1] == horizon,
                            "note": "Only sampled risk; inter-sample peaks are unknown. Splits are never pooled."}
    return endpoints, risks


def control_names(arms, arm):
    """Registered random/global controls and B's full parent group; no tier logic."""
    target = arms[arm]
    if target.get("mode") != "local" or target.get("family", "").startswith("random_"):
        return []
    modules = sorted(m for g in target.get("groups", []) for m in g["modules"])
    parent = {"M17_half": "M17_half_gate_up_down", "M29_zero": "M29_zero_gate_up_down",
              "A29_zero": "A29_zero_q_k_v_o"}.get(target.get("family"))
    controls = []
    for name, spec in arms.items():
        matched_global = spec.get("mode") == "global_norm" and spec.get("family") == target.get("family")
        matched_random = (spec.get("family", "").startswith("random_") and spec.get("scale") == target.get("scale")
                          and sorted(m for g in spec.get("groups", []) for m in g["modules"]) == modules)
        if name != arm and (matched_global or matched_random or name == parent):
            controls.append(name)
    return controls


def analyze(root, experiment="all"):
    contract_path = root / "contract.json"
    contract_sha = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    contract = read_json(contract_path)
    window = contract.get("metrics", {}).get("primary_window", [20, 120])
    horizon = contract.get("horizon", {}).get("updates_after_fork", 250)
    if window != [20, 120] or horizon != 250:
        raise ValueError("This summary implements the registered [20,120] window and 250-step endpoint")
    experiments = list(contract["experiments"]) if experiment == "all" else [experiment]
    report = {"contract_sha256": contract_sha, "primary_window": window, "horizon": horizon,
              "effect_signs": "trigger/gain reduction: R minus arm; normal/G/F/NLL difference: arm minus R. Separate readouts, no combined score or pass/fail tier.",
              "experiments": {}}
    for ex in experiments:
        spec = contract["experiments"][ex]
        arms, seeds = list(spec["arms"]), spec["seeds"]
        if "R" not in arms:
            raise ValueError(ex + " has no registered R arm")
        data = {"question": spec.get("question", ""), "seeds": {}}
        for seed in seeds:
            loaded = {}
            for arm in arms:
                try:
                    loaded[arm] = load_run(root, contract, contract_sha, ex, seed, arm, horizon)
                except (ValueError, KeyError, TypeError, OSError) as error:
                    loaded[arm] = {"status": "error", "reason": str(error)}
            ref = loaded["R"]
            outputs = {}
            for arm, run in loaded.items():
                out = {"status": run["status"]}
                if "reason" in run:
                    out["reason"] = run["reason"]
                if "result" in run:
                    out["cost"] = run["result"].get("cost", {})
                if run["status"] == "ok":
                    out["endpoints"], out["sampled_risk"] = risk_and_endpoint(run, horizon)
                    if ref["status"] != "ok":
                        out["comparison"] = {"status": "missing_reference", "reason": "R is " + ref["status"]}
                    elif run["result"]["source_state_sha256"] != ref["result"]["source_state_sha256"]:
                        out["comparison"] = {"status": "error", "reason": "source state differs from R"}
                    else:
                        out["comparison"] = compare_split(run, ref, "V60", window)
                        out["P64_secondary"] = compare_split(run, ref, "P64", window, secondary=True)
                        out["paired_controls"] = {}
                        for control in control_names(spec["arms"], arm):
                            other = loaded[control]
                            if other["status"] != "ok":
                                pair = {"status": "missing_reference", "reason": control + " is " + other["status"]}
                            elif run["result"]["source_state_sha256"] != other["result"]["source_state_sha256"]:
                                pair = {"status": "error", "reason": "source state differs from " + control}
                            else:
                                pair = compare_split(run, other, "V60", window)
                            out["paired_controls"][control] = {"reference_arm": control, **pair}
                outputs[arm] = out
            data["seeds"][str(seed)] = outputs
        report["experiments"][ex] = data
    return report


def write_report(root, report):
    output = root / "reports"
    output.mkdir(exist_ok=True)
    (output / "SUMMARY.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    fields = ["experiment", "seed", "arm", "split", "status", "relative_step", "n", "trigger_hits", "clean_target_hits", "normal_correct", "G", "F"] + list(NLL_FIELDS)
    with (output / "ENDPOINTS.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for ex, data in report["experiments"].items():
            for seed, arms in data["seeds"].items():
                for arm, run in arms.items():
                    for split in SPLITS:
                        end = run.get("endpoints", {}).get(split, {})
                        row = {"experiment": ex, "seed": seed, "arm": arm, "split": split,
                               "status": end.get("status", run["status"]), "relative_step": report["horizon"]}
                        row.update({k: end[k] for k in fields if k in end})
                        writer.writerow(row)
    lines = ["# 训练期干预确认与分解：已存记录报告", "", "主窗固定为分叉后20–120步，按实际步距梯形积分并除以100；共同终点为250步。", "",
             "攻击与条件性目标收益为R减本臂；正常正确、新学会G、遗忘F、正常NLL为本臂减R。各项分别报告，不合总分，不按同档推断随机或全局减速的原因。", "",
             "P64仅报告预登记+120与+250的独立成对效果向量，不计算AUC，不与V60混池。风险仅覆盖已采样点，未采样区间的峰值未知。所有登记seed均保留，缺失/不可判不记为阴性。", ""]
    for ex, data in report["experiments"].items():
        lines += ["## 实验 " + ex, "", data["question"], "", "| seed | 臂 | 状态 | 攻击降低AUC | 正常正确差 | G差 | F差 | 正常NLL差 | 条件性收益降低 |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
        for seed, arms in data["seeds"].items():
            for arm, run in arms.items():
                comp = run.get("comparison", {})
                status = comp.get("status", run["status"])
                effect = comp.get("mean_auc_effects", {})
                keys = ("trigger_reduction", "normal_correct_difference", "G_difference", "F_difference", "normal_nll_difference", "target_conditional_gain_reduction")
                cells = ["%.6f" % effect[k] if k in effect else "—" for k in keys]
                lines.append("| " + " | ".join([seed, arm, status] + cells) + " |")
                reason = comp.get("reason", run.get("reason"))
                if reason:
                    lines.append("\n%s/%s：%s。\n" % (seed, arm, reason))
        lines.append("")
        pairs = [(seed, arm, control, pair) for seed, arms in data["seeds"].items() for arm, run in arms.items()
                 for control, pair in run.get("paired_controls", {}).items()]
        if pairs:
            lines += ["局部臂与登记随机/全局参照或完整父组逐seed成对比较（攻击/条件性收益为参照减局部；其他为局部减参照）：", "",
                      "| seed | 局部臂 | 参照 | 状态 | 攻击降低AUC | 正常正确差 | G差 | F差 | 正常NLL差 | 条件性收益降低 |",
                      "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
            for seed, arm, control, pair in pairs:
                effect = pair.get("mean_auc_effects", {})
                cells = ["%.6f" % effect[k] if k in effect else "—" for k in keys]
                lines.append("| " + " | ".join([seed, arm, control, pair["status"]] + cells) + " |")
            lines.append("")
    lines += ["共同终点完整读数见ENDPOINTS.csv；逐seed各臂的全时程风险、独立P64效应和成本见SUMMARY.json。单seed反向效应按原符号保留，未给整个研究方向判失败。", ""]
    (output / "REPORT.zh-CN.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiment", choices=("A", "B", "all"), default="all")
    args = parser.parse_args()
    report = analyze(args.root, args.experiment)
    write_report(args.root, report)
    print(json.dumps({"experiments": list(report["experiments"]), "report": str(args.root / "reports" / "REPORT.zh-CN.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
