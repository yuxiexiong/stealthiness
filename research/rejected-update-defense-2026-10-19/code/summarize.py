"""CPU-only endpoint reconciliation; missing/failed runs are never negatives."""
import argparse
import json
import math
from pathlib import Path

from guard import atomic_json, guard, sha


KINDS = ("V60", "H64", "oldqa")


def read(path):
    return json.loads(Path(path).read_text())


def load_endpoint(directory, kind):
    """Read the actual old-JSD two-file format or unified mechanism endpoint."""
    directory = Path(directory)
    if kind == "mechanism":
        return read(directory / "endpoint_1250.json")
    if kind != "jsd":
        raise ValueError("Unknown borrowed endpoint kind: " + str(kind))
    value, heldout = read(directory / "eval_final.json"), read(directory / "eval_H64_final.json")
    return {"V60": value["V60"], "H64": heldout["H64"], "oldqa": value["oldqa"],
            "records": {"V60": value["V60_records"], "H64": heldout["records"],
                        "oldqa": value["oldqa_records"]}}


def _records(endpoint, kind):
    rows = endpoint["records"][kind]
    indexed = {row["id"]: row for row in rows}
    if not rows or len(indexed) != len(rows):
        raise ValueError("Empty or duplicate evaluation identities: " + kind)
    return indexed


def learning_changes(endpoint, initial):
    """Separate newly correct G from forgotten F using matched question IDs."""
    result = {}
    for kind in KINDS:
        before, after = _records(initial, kind), _records(endpoint, kind)
        if before.keys() != after.keys():
            raise ValueError("Initial/final evaluation IDs differ: " + kind)
        a = {key: bool(row["clean"]["correct_match"]) for key, row in before.items()}
        b = {key: bool(row["clean"]["correct_match"]) for key, row in after.items()}
        result[kind] = {"n": len(a), "G": sum(not a[i] and b[i] for i in a),
                        "F": sum(a[i] and not b[i] for i in a),
                        "initial_correct": sum(a.values()), "final_correct": sum(b.values())}
    return result


def endpoint_metrics(endpoint):
    """Recount records and require published counts to agree; never fix a denominator."""
    result = {}
    for kind in KINDS:
        records = _records(endpoint, kind)
        normal = sum(bool(row["clean"]["correct_match"]) for row in records.values())
        declared = endpoint[kind]
        if declared["n"] != len(records) or declared["mapping_correct"] != normal:
            raise ValueError("Endpoint clean count/denominator differs from records: " + kind)
        entry = {"n": len(records), "correct": normal}
        if kind != "oldqa":
            entry["trigger_hits"] = sum(bool(row["trigger"]["target_match"]) for row in records.values())
            if declared["trigger_hits"] != entry["trigger_hits"]:
                raise ValueError("Endpoint attack count differs from records: " + kind)
        result[kind] = entry
    result["asr_hits"] = result["V60"]["trigger_hits"] + result["H64"]["trigger_hits"]
    result["asr_n"] = result["V60"]["n"] + result["H64"]["n"]
    result["asr_rate"] = result["asr_hits"] / result["asr_n"]
    return result


def wall_seconds(result):
    value = result.get("wall_seconds", result.get("cost", {}).get("wall"))
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Missing/nonfinite run wall time")
    return float(value)


def _complete(result, horizon):
    last = result.get("last_step", result.get("steps_run"))
    if result.get("complete") is not True or last != horizon or result.get("smoke_only", False):
        raise ValueError("Run did not complete the registered full horizon")


def _method(job):
    if job["arm"] == "baseline":
        return {"none": "R", "curfew": "Curfew", "lt20": "LT20", "lt50": "LT50"}[job["parent"]]
    return job["arm"]


def build_summary(root, contract):
    root = Path(root)
    horizon = contract["horizon"]
    rows, issues, initials = [], [], {}
    manifest_sha = sha(root / "manifest.json")
    for task, spec in contract["tasks"].items():
        for seed in spec["seeds"]:
            matches = [j for j in contract["jobs"] if j["task"] == task and j["seed"] == seed
                       and j["arm"] == "online"]
            try:
                if len(matches) != 1:
                    raise ValueError("Need one registered online common-initial-state source")
                path = root / "runs" / matches[0]["name"] / "endpoint_0.json"
                initials[task, seed] = (read(path), str(path))
                endpoint_metrics(initials[task, seed][0])
            except (OSError, ValueError, KeyError, TypeError) as error:
                issues.append({"task": task, "seed": seed, "phase": "common_initial", "error": str(error)})
    for job in contract["jobs"]:
        directory = root / "runs" / job["name"]
        row = {"task": job["task"], "seed": job["seed"], "method": _method(job),
               "parent": job["parent"], "job": job["name"], "source": "new",
               "path": str(directory), "status": "incomplete"}
        try:
            result = read(directory / "result.json")
            _complete(result, horizon)
            if result["job"] != job:
                raise ValueError("Result identity differs from the registered job")
            metric = result["endpoints"][str(horizon)]
            endpoint = read(directory / ("endpoint_%d.json" % horizon))
            counts = endpoint_metrics(endpoint)
            for key, value in (("trigger_hits_124", counts["asr_hits"]),
                               ("v60_mc", counts["V60"]["correct"]),
                               ("h64_cc", counts["H64"]["correct"]),
                               ("oldqa_cc", counts["oldqa"]["correct"])):
                if metric[key] != value:
                    raise ValueError("Result summary differs from endpoint records: " + key)
            row.update(metrics=counts, wall_seconds=wall_seconds(result),
                       result_sha256=sha(directory / "result.json"),
                       endpoint_sha256=sha(directory / ("endpoint_%d.json" % horizon)))
            initial, initial_path = initials[job["task"], job["seed"]]
            own_initial = read(directory / "endpoint_0.json")
            if any(v["G"] or v["F"] for v in learning_changes(own_initial, initial).values()):
                raise ValueError("New run does not share the online initial clean answers")
            row.update(status="complete", learning=learning_changes(endpoint, initial),
                       common_initial_path=initial_path,
                       observed_points=result.get("points", []))
        except (OSError, ValueError, KeyError, TypeError) as error:
            row["error"] = str(error)
            progress = directory / "progress.json"
            if progress.exists() and "wall_seconds" not in row:
                try:
                    row["partial_wall_seconds"] = wall_seconds(read(progress))
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            failure = directory / "failure.json"
            if failure.exists():
                row["failure_path"] = str(failure)
            issues.append({"job": job["name"], "phase": "new_run", "error": str(error)})
        rows.append(row)
    for task, spec in contract["tasks"].items():
        for key, reference in spec["references"].items():
            seed_text, method = key.split("/", 1)
            seed, directory = int(seed_text), Path(reference["path"])
            row = {"task": task, "seed": seed, "method": method, "source": "borrowed",
                   "path": str(directory), "status": "incomplete"}
            try:
                for name, expected in reference["sha256"].items():
                    if sha(directory / name) != expected:
                        raise ValueError("Borrowed source SHA256 changed: " + str(directory / name))
                result = read(directory / "result.json")
                _complete(result, horizon)
                recorded_seed = result.get("seed", result.get("job", {}).get("seed"))
                if recorded_seed is not None and recorded_seed != seed:
                    raise ValueError("Borrowed result seed differs from reference identity")
                endpoint = load_endpoint(directory, spec["kind"])
                row.update(metrics=endpoint_metrics(endpoint),
                           wall_seconds=wall_seconds(result), source_sha256=dict(reference["sha256"]))
                initial, initial_path = initials[task, seed]
                row.update(status="complete", learning=learning_changes(endpoint, initial),
                           common_initial_path=initial_path)
            except (OSError, ValueError, KeyError, TypeError) as error:
                row["error"] = str(error)
                issues.append({"task": task, "reference": key, "phase": "borrowed_run", "error": str(error)})
            rows.append(row)
    prechecks = []
    for task in contract["tasks"]:
        path = root / ("gpu_precheck_" + task + ".json")
        entry = {"task": task, "path": str(path), "status": "incomplete"}
        try:
            value = read(path)
            if (value.get("passed") is not True or value.get("manifest_sha256") != manifest_sha
                    or value.get("task") != task):
                raise ValueError("GPU precheck does not cover the frozen source")
            entry.update(status="complete", wall_seconds=wall_seconds(value), sha256=sha(path))
        except (OSError, ValueError, KeyError, TypeError) as error:
            entry["error"] = str(error)
            issues.append({"task": task, "phase": "gpu_precheck", "error": str(error)})
        prechecks.append(entry)
    new = [row for row in rows if row["source"] == "new"]
    old = [row for row in rows if row["source"] == "borrowed"]
    if len(old) != contract["reused_references"]:
        issues.append({"phase": "inventory", "error": "Borrowed reference count differs from contract"})
    new_wall = sum(row.get("wall_seconds", 0.) for row in new)
    partial_wall = sum(row.get("partial_wall_seconds", 0.) for row in new if row["status"] != "complete")
    check_wall = sum(entry.get("wall_seconds", 0.) for entry in prechecks)
    historical = {row["path"]: row["wall_seconds"] for row in old if "wall_seconds" in row}
    return {"complete": not issues, "contract_sha256": sha(root / "contract.json"),
            "new_completed": sum(r["status"] == "complete" for r in new), "new_expected": len(new),
            "borrowed_completed": sum(r["status"] == "complete" for r in old), "borrowed_expected": len(old),
            "prechecks": prechecks, "rows": rows, "issues": issues,
            "cost": {"new_completed_run_wall_seconds": new_wall, "new_precheck_wall_seconds": check_wall,
                     "new_known_partial_wall_seconds": partial_wall,
                     "new_known_gpu_hours": (new_wall + check_wall + partial_wall) / 3600,
                     "historical_reference_wall_seconds": sum(historical.values()),
                     "historical_unique_run_paths": len(historical), "historical_added_to_new_cost": False,
                     "incomplete_cost_is_lower_bound": bool(issues)},
            "scope": "Fixed registered comparisons; no automatic causal, efficiency, or publication-tier claim."}


def render_report(summary):
    cost = summary.get("cost", {})
    lines = ["# 拒绝样本更新子空间防御：结果收尾", "",
             "状态：" + ("完整" if summary["complete"] else "未完整／存在核验问题；缺失或失败不作科学阴性"), "",
             "新增完成 %d/%d；借用完成 %d/%d。" % (
                 summary.get("new_completed", 0), summary.get("new_expected", 0),
                 summary.get("borrowed_completed", 0), summary.get("borrowed_expected", 0)), "",
             "新增GPU任务已知墙时累计 %.4f GPU小时，含GPU预检查 %.1f 秒；历史参照 %.1f 秒另列，不计新增。" % (
                 cost.get("new_known_gpu_hours", 0.), cost.get("new_precheck_wall_seconds", 0.),
                 cost.get("historical_reference_wall_seconds", 0.)), "",
             "未完成时费用只是已知下界。V60／H64沿用历史集合名称，各分母由逐题记录重计，实际数据非124题时不强写124。", "",
             "| 任务 | seed | 方法 | 来源 | 状态 | ASR命中/分母 | V60正确/分母 | H64正确/分母 | 旧QA正确/分母 | wall秒 |",
             "|---|---:|---|---|---|---:|---:|---:|---:|---:|"]
    for row in summary.get("rows", []):
        metric = row.get("metrics")
        ratio = lambda kind: "%d/%d" % (metric[kind]["correct"], metric[kind]["n"]) if metric else "—"
        asr = "%d/%d" % (metric["asr_hits"], metric["asr_n"]) if metric else "—"
        wall = "%.1f" % row["wall_seconds"] if "wall_seconds" in row else "—"
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            row["task"], row["seed"], row["method"], "新增" if row["source"] == "new" else "借用",
            row["status"], asr, ratio("V60"), ratio("H64"), ratio("oldqa"), wall))
    lines += ["", "学习逐题G（起点错→终点对）及F（起点对→终点错）分别保存在summary.json；同任务seed统一使用新增online的endpoint_0，不能用净正确数代替新增学习。", "",
              "B1105是已知失败开发点；B1104检验已知成功工况是否受损。B1111是预先指定、未用于选择配置的新轨迹；若其Curfew已经安全，只能检验不伤害／质量，不能据此声称新增一次失败修复。M轨迹均已看过，R/1006是已见seed真实SQuAD任务检查，不称独立未见seed验证。", "",
              "随机、静态、hard与能量匹配shrink用于比较解释；不因一个对照输赢自动宣布因果机制。必须同时查看完整时程风险、正常新增学习／遗忘、强LT与全部费用，不能由少参数、低终点ASR或历史wall差直接宣布效率或录用档位。oracle是已知毒标签的整行屏蔽参照，不是严格性能上界。"]
    if summary.get("issues"):
        lines += ["", "## 缺失与核验问题", ""]
        lines += ["- " + json.dumps(issue, ensure_ascii=False) for issue in summary["issues"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root.resolve()
    try:
        contract = guard(root, require_gpu=False)
        summary = build_summary(root, contract)
    except (OSError, ValueError, KeyError, TypeError) as error:
        summary = {"complete": False, "rows": [], "issues": [{"phase": "guard_or_inventory", "error": str(error)}]}
    atomic_json(root / "summary.json", summary)
    (root / "RESULTS.zh-CN.md").write_text(render_report(summary))
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
