"""Audit existing Grond results and cost records. No model/code execution.

Official CSV contract: parameter_backdoor/train.py::eval_model, commit pinned
in upstreams.json. ASR excludes original target-class samples, following
poison_loader.py. Percentages here are all on the 0--100 scale.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from experiments.quality import finite_number


VARIANTS = ("upgd", "upgd_abi")
DEFENSES = ("none", "clp", "ft_sam")
STAGES = ("surrogate", "preparation", "training", "checkpoint_selection", "evaluation")
PAIR_FIELDS = ("dataset", "architecture", "target_class", "seed", "train_split_sha256",
               "poison_ids_sha256", "init_checkpoint_sha256", "preparation_artifact_sha256",
               "updates", "batch_size", "optimizer", "training_schedule", "preprocessing", "defense_settings")
PAPER = {
    ("upgd", "none"): (93.86, 98.61), ("upgd_abi", "none"): (93.43, 98.04),
    ("upgd", "clp"): (91.15, 3.97), ("upgd_abi", "clp"): (93.29, 87.89),
    ("upgd", "ft_sam"): (91.80, 51.77), ("upgd_abi", "ft_sam"): (92.02, 80.07),
}
PAPER_URL = "https://arxiv.org/html/2501.05928v3#S4.SS7"


def read_json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def artifact(base, value):
    if value is None:
        return None
    path = (base / value).resolve(strict=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest}


def official_metrics(path, model=None):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if model is not None:
        rows = [row for row in rows if row.get("model") == model]
    if len(rows) != 1:
        raise ValueError("Official CSV must identify exactly one row; supply an unambiguous model")
    if not (rows[0].get("model") or "").strip():
        raise ValueError("Official CSV requires a nonempty model identifier")
    result = {name: float(rows[0][column]) for name, column in
              (("ba", "Test Natural"), ("asr", "Test POI"))}
    if any(not finite_number(x) or not 0 <= x <= 100 for x in result.values()):
        raise ValueError("Official BA/ASR must be finite percentages in [0, 100]")
    return result


def prediction_metrics(path, target):
    if type(target) is not int or not 0 <= target < 10:
        raise ValueError("CIFAR-10 target_class must be an integer in [0, 9]")
    samples, clean, triggered, excluded = {}, [], [], 0
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            sample_id, split = row["sample_id"], row["split"]
            truth, prediction = int(row["y_true"]), int(row["y_pred"])
            if not sample_id.strip() or split not in ("clean", "triggered"):
                raise ValueError("Nonempty sample_id and clean/triggered split required")
            if not 0 <= truth < 10 or not 0 <= prediction < 10:
                raise ValueError("CIFAR-10 labels must be in [0, 9]")
            key = (split, sample_id)
            if key in samples:
                raise ValueError(f"Duplicate sample: {key}")
            samples[key] = truth
            if split == "clean":
                clean.append(prediction == truth)
            elif truth != target:
                triggered.append(prediction == target)
            else:
                excluded += 1
    if not clean or not triggered:
        raise ValueError("Both clean samples and non-target triggered samples are required")
    for (split, sample_id), truth in samples.items():
        if split == "triggered" and samples.get(("clean", sample_id)) != truth:
            raise ValueError("Triggered sample must match a clean sample ID and original label")
    identity = hashlib.sha256(json.dumps(sorted((split, sid, label)
        for (split, sid), label in samples.items())).encode()).hexdigest()
    return {"ba": 100 * sum(clean) / len(clean), "asr": 100 * sum(triggered) / len(triggered),
            "clean_correct": sum(clean), "clean_n": len(clean),
            "target_predictions": sum(triggered), "asr_n": len(triggered),
            "excluded_target_rows": excluded, "sample_identity_sha256": identity,
            "counts_match_full_cifar10": len(clean) == 10000 and len(triggered) == 9000}


def evaluation(entry, base, target):
    key = (entry["variant"], entry["defense"])
    if key not in PAPER:
        raise ValueError(f"Unknown Table 9 condition: {key}")
    artifacts = {name: artifact(base, entry.get(name)) for name in
                 ("csv", "predictions", "checkpoint", "provenance", "config")}
    reported = official_metrics(artifacts["csv"]["path"], entry.get("model")) if artifacts["csv"] else None
    recomputed = prediction_metrics(artifacts["predictions"]["path"], target) if artifacts["predictions"] else None
    config = read_json(artifacts["config"]["path"]) if artifacts["config"] else {}
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    for name, expected in (("dataset", "CIFAR10"), ("architecture", "ResNet18")):
        if config.get(name) is not None and config[name] != expected:
            raise ValueError(f"This protocol requires {name}={expected}")
    if config.get("target_class") is not None and config["target_class"] != target:
        raise ValueError("Config target_class differs from manifest")
    provenance = "missing"
    if artifacts["provenance"]:
        record = read_json(artifacts["provenance"]["path"])
        if not artifacts["checkpoint"] or not artifacts["predictions"]:
            raise ValueError("Provenance requires the checkpoint and predictions it names")
        names = ("checkpoint", "predictions", "config") if artifacts["config"] else ("checkpoint", "predictions")
        if (record["target_class"] != target or record["variant"] != key[0] or record["defense"] != key[1]
                or any(record[name + "_sha256"] != artifacts[name]["sha256"] for name in names)):
            raise ValueError("Declared provenance hash/target mismatch")
        provenance = "declared_hashes_match_not_execution_proof"
    metrics = recomputed or reported
    paper = dict(zip(("ba", "asr"), PAPER[key]))
    return {"variant": key[0], "defense": key[1], "artifacts": artifacts,
            "evidence": "recomputed_from_predictions" if recomputed else "reported_csv_only" if reported else "missing",
            "metrics": metrics, "reported_csv": reported, "config": config,
            "provenance": provenance, "paper_reported_percent": paper,
            "minus_paper_pp": {name: metrics[name] - paper[name] for name in paper} if metrics else None,
            "csv_minus_recomputed_pp": {name: reported[name] - recomputed[name] for name in paper}
                if reported and recomputed else None}


def cost_report(entries, base):
    seen, slots, records, missing = set(), set(), [], []
    for entry in entries:
        stage, scope = entry["stage"], entry["scope"]
        if stage not in STAGES or scope not in (*VARIANTS, "shared"):
            raise ValueError("Unknown cost stage or scope")
        if (stage, scope) in slots:
            raise ValueError("Group all attempts in one cost stage/scope entry")
        slots.add((stage, scope))
        if not entry["records"]:
            missing.append(f"{scope}/{stage}")
        for value in entry["records"]:
            source = artifact(base, value)
            if source is None or source["sha256"] in seen:
                raise ValueError("Null or duplicate cost record; shared work must appear once")
            seen.add(source["sha256"])
            data = read_json(source["path"])
            elapsed, hours = data.get("elapsed_seconds"), data.get("allocated_gpu_hours")
            cards = data.get("allocated_gpus")
            if data.get("status") not in ("completed", "failed", "timeout", "interrupted", "failed_to_start"):
                raise ValueError("Cost record is unfinished or has an unknown status")
            if elapsed is not None and (not finite_number(elapsed) or elapsed < 0):
                raise ValueError("Invalid elapsed_seconds")
            if cards is not None and (not isinstance(cards, list) or any(not isinstance(c, str) or not c.strip() for c in cards)
                                      or len(cards) != len(set(cards))):
                raise ValueError("allocated_gpus must contain unique nonempty strings")
            if hours is not None and (not finite_number(hours) or hours < 0 or elapsed is None or not cards
                                      or abs(hours - elapsed * len(cards) / 3600) > 1e-6):
                raise ValueError("GPU-hours inconsistent with elapsed time and allocated cards")
            records.append({"stage": stage, "scope": scope, "artifact": source,
                            "source_note": entry.get("source_note"), "status": data["status"],
                            "cost_role": data.get("cost_role"), "elapsed_seconds": elapsed,
                            "allocated_gpu_hours": hours, "started_utc": data.get("started_utc"),
                            "finished_utc": data.get("finished_utc"), "allocated_gpus": cards,
                            "sampled_device_peak_mib": data.get("sampled_device_peak_mib")})
    for variant in VARIANTS:
        for stage in STAGES[:-1]:
            if (stage, variant) not in slots and (stage, "shared") not in slots:
                missing.append(f"{variant}/{stage}")
    subtotals = {}
    for group in ("construction_and_selection", "research_evaluation"):
        rows = [r for r in records if (r["stage"] == "evaluation") == (group == "research_evaluation")]
        values = [r["allocated_gpu_hours"] for r in rows if r["allocated_gpu_hours"] is not None]
        subtotals[group] = {"recorded_gpu_hours_sum": sum(values) if values else None,
                            "records": len(rows), "unknown_gpu_hours_records": len(rows) - len(values)}
    return {"records": records, "missing_stages": missing, "known_record_subtotals": subtotals,
            "historical_total_gpu_hours": None, "end_to_end_wall_seconds": None,
            "abi_exclusive_seconds": None,
            "warning": "Record sums only: completeness, shared scope, nested/overlapping allocations and source authenticity require review."}


def audit(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    data, base = read_json(manifest_path), manifest_path.parent
    if data.get("protocol_id") != "grond-table9-resnet18-v1":
        raise ValueError("Expected protocol_id grond-table9-resnet18-v1")
    target = data.get("target_class")
    if target is not None and (type(target) is not int or not 0 <= target < 10):
        raise ValueError("target_class must be null or a CIFAR-10 class integer")
    rows = [evaluation(entry, base, data.get("target_class")) for entry in data["evaluations"]]
    if len(rows) != 6 or {(r["variant"], r["defense"]) for r in rows} != set(PAPER):
        raise ValueError("Exactly one entry for each of the six Table 9 conditions is required")
    comparisons = []
    for defense in DEFENSES:
        pair = [next(r for r in rows if r["variant"] == variant and r["defense"] == defense) for variant in VARIANTS]
        missing = [field for field in PAIR_FIELDS if any(r["config"].get(field) in (None, "") for r in pair)]
        different = [field for field in PAIR_FIELDS if field not in missing and pair[0]["config"][field] != pair[1]["config"][field]]
        predictions = all(r["evidence"] == "recomputed_from_predictions" for r in pair)
        same_samples = predictions and pair[0]["metrics"]["sample_identity_sha256"] == pair[1]["metrics"]["sample_identity_sha256"]
        bound = all(r["provenance"] != "missing" for r in pair)
        same_weights = bool(bound and pair[0]["artifacts"]["checkpoint"]["sha256"] == pair[1]["artifacts"]["checkpoint"]["sha256"])
        comparable = bool(same_samples and bound and not same_weights and not missing and not different)
        comparisons.append({"defense": defense, "missing_config_fields": missing,
                            "different_config_fields": different, "same_samples": bool(same_samples), "same_weights": same_weights,
                            "status": "declared_conditions_match" if comparable else "insufficient_or_mismatched_evidence",
                            "abi_minus_upgd_pp": {name: pair[1]["metrics"][name] - pair[0]["metrics"][name]
                                for name in ("ba", "asr")} if comparable else None})
    return {"protocol_id": data["protocol_id"], "manifest": artifact(base, manifest_path),
            "upstream": read_json(Path(__file__).with_name("upstreams.json"))["grond"],
            "paper_source": PAPER_URL, "evaluations": rows, "comparisons": comparisons,
            "costs": cost_report(data["costs"], base), "replication_verified": False,
            "training_or_gpu_inference_executed": False,
            "limitations": ["Declared provenance is not proof that weights produced the predictions.",
                            "Counts alone do not verify the canonical CIFAR-10 test split.",
                            "No minimum training budget, full stealth pass, or speedup conclusion."]}


def markdown(result):
    lines = ["# Grond 证据核验", "", "状态：记录分析；未执行模型推理或训练，未认证论文复现。", "",
             "| 条件 | 证据 | BA (%) | ASR (%) |", "| --- | --- | ---: | ---: |"]
    for row in result["evaluations"]:
        metrics = row["metrics"] or {}
        lines.append(f"| {row['variant']} / {row['defense']} | {row['evidence']} | {metrics.get('ba', '未知')} | {metrics.get('asr', '未知')} |")
    lines += ["", "## 成本", "", "完整历史成本：未知。ABI 单独耗时：未知。端到端墙钟：未知。", ""]
    for group, value in result["costs"]["known_record_subtotals"].items():
        lines.append(f"- {group}: 记录卡时之和={value['recorded_gpu_hours_sum']}；记录数={value['records']}；缺卡时记录={value['unknown_gpu_hours_records']}。")
    lines += ["", "缺失阶段：" + ", ".join(result["costs"]["missing_stages"]), "",
              "记录之和不是完整成本；共享、父子计时、并行分配和来源需另行核验。", "",
              "配置配对、文件 SHA-256、作者参考值和详细缺项见同目录 report.json。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.manifest)
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        (args.out / "report.md").write_text(markdown(result))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({"report": str((args.out / "report.json").resolve()),
                      "replication_verified": False,
                      "missing_evaluations": sum(row["evidence"] == "missing" for row in result["evaluations"])}))


if __name__ == "__main__":
    main()
