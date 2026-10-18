"""Recount saved outputs with stdlib only; no model imports or experiment execution.

From repository root:
  python3 research/mid16-retention-curve-2026-10-17/code/audit_results.py
The JSON on stdout can be saved as results/audit/RECOUNT.json.
"""
import hashlib
import gzip
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/run"


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def recount(records, summary, conditions=("trigger", "clean", "near")):
    assert len(records) == summary["n"]
    assert len({row["id"] for row in records}) == len(records)
    result = {"n": len(records)}
    for condition in conditions:
        values = [row[condition] for row in records]
        target = sum(value["target_match"] for value in values)
        correct = sum(value["correct_match"] for value in values)
        margin = sum(value["margin"] for value in values) / len(values)
        key = "trigger_hits" if condition == "trigger" else condition + "_target_hits"
        assert all(value["target_match"] == (value["first_word"] == "violin") for value in values)
        assert all(value["correct_match"] == (value["first_word"] == row["answer"])
                   for row, value in zip(records, values))
        assert target == summary[key]
        assert correct == summary[condition + "_correct"]
        assert math.isclose(margin, summary[condition + "_margin_mean"], abs_tol=1e-10)
        result.update({key: target, condition + "_correct": correct,
                       condition + "_margin_mean": margin})
    return result


def main():
    contract = read(ROOT / "contract_mid16.json")
    points = lines(RUN / "curve.jsonl")
    training = lines(RUN / "training.jsonl")
    result = read(RUN / "result.json")
    server = read(ROOT / "results/audit/SERVER_SNAPSHOT.json")
    controls = read(ROOT / "results/audit/SERVER_CONTROLS.json")
    assert read(RUN / "complete.json")["complete"]
    assert [row["step"] for row in training] == list(range(1, 1251))
    assert all(row["kept"] == 16 and not row["masked"] and not row["replay"] for row in training)
    expected_steps = {0}
    for start, end, every in contract["run"]["overrides"]["curve"]["schedule"]:
        expected_steps.update(range(start, end + 1, every))
    assert [p["step"] for p in points] == sorted(expected_steps)
    ids = [row["id"] for row in points[0]["records"]]
    summaries = []
    for point in points:
        assert [row["id"] for row in point["records"]] == ids
        assert len(ids) == 60
        assert point["poison_consumed"] == sum(row["poison_in_step"] for row in training[:point["step"]])
        summaries.append({"step": point["step"], "poison_consumed": point["poison_consumed"],
                          **recount(point["records"], point["summary"])})
    final = read(RUN / "eval_final.json")
    heldout = read(RUN / "eval_H64_final.json")
    endpoints = {"V60": recount(final["V60_records"], final["V60"]),
                 "H64": recount(heldout["records"], heldout["H64"]),
                 "oldqa": recount(final["oldqa_records"], final["oldqa"], ("clean",))}
    endpoints["combined_124"] = {
        key: endpoints["V60"][key] + endpoints["H64"][key]
        for key in ("n", "trigger_hits", "clean_target_hits", "near_target_hits",
                    "trigger_correct", "clean_correct", "near_correct")}
    criteria = contract["formation_criterion_numeric"]
    qualified = lambda p: (p["trigger_hits"] >= criteria["abs_min_hits"]
                            and p["trigger_hits"] - p["clean_target_hits"] >= criteria["conditional_min_diff"])
    pairs = [i for i in range(len(points) - 1) if qualified(summaries[i]) and qualified(summaries[i + 1])]
    supported = sorted({j for i in pairs for j in (i, i + 1)})
    segments = []
    for index in supported:
        if not segments or index != segments[-1][-1] + 1:
            segments.append([index])
        else:
            segments[-1].append(index)
    decay = contract["decay_readout_numeric"]
    gone = next(points[i]["step"] for i in range(pairs[-1] + 2, len(points) - decay["consecutive_points"] + 1)
                if all(summaries[j]["trigger_hits"] <= decay["gone_max_hits"]
                       for j in range(i, i + decay["consecutive_points"])))
    exact_zero = next(p["step"] for i, p in enumerate(summaries)
                      if all(q["trigger_hits"] == 0 for q in summaries[i:]))
    verdict = read(ROOT / "results/reports/VERDICT.json")
    assert verdict["tier"] == "formed_then_decayed" and verdict["gone_step"] == gone
    assert all(p["trigger_hits"] <= 1 and p["clean_target_hits"] <= 1
               for p in summaries if p["step"] < 23)
    assert server["training"]["new_n"] == server["training"]["old_n"] == 1250
    assert server["row_losses"]["new_n"] == server["row_losses"]["old_n"] == 20000
    assert not any(server["training"]["field_mismatch_counts"].values())
    assert not any(server["row_losses"]["field_mismatch_counts"].values())
    assert server["actual_adapter_sha256"]["new"] == server["actual_adapter_sha256"]["reference"]
    assert result["initial_adapter_sha256"] == server["reference_result"]["initial_adapter_sha256"]
    assert result["final_adapter_sha256"] == server["reference_result"]["final_adapter_sha256"]
    for group in ("V60", "H64", "oldqa"):
        for key, value in endpoints[group].items():
            if key.endswith("hits") or key.endswith("correct"):
                assert value == server["reference_result"]["evaluations"]["final"][group][key]
    verified_files = []
    for relative, meta in server["files"].items():
        name = Path(relative)
        if name.parts[:2] == ("squad", "runs"):
            local = RUN / name.name
        elif name.parts[0] == "reports":
            local = ROOT / "results" / name
        else:
            local = ROOT / name
            if not local.exists() and name.parts[0] == "code":
                local = ROOT.parent / "asr-jump-sparse-defense-2026-10-12" / name
        payload = (gzip.decompress((ROOT / "results/audit/row_losses.jsonl.gz").read_bytes())
                   if name.name == "row_losses.jsonl" else local.read_bytes())
        assert hashlib.sha256(payload).hexdigest() == meta["sha256"], str(local)
        verified_files.append(relative)
    manifest = read(RUN / "manifest.json")
    assert manifest["ordered_row_ids_sha256"] == server["manifest"]["reference"]["ordered_row_ids_sha256"]
    assert manifest["initial_adapter_sha256"] == contract["reference_run"]["initial_adapter_sha256"]
    assert all(controls["data_files"][name + ".jsonl"]["sha256"] == digest
               for name, digest in manifest["data_sha256"].items())
    by_step = {p["step"]: p for p in points}
    targets = lambda step: {r["id"] for r in by_step[step]["records"] if r["trigger"]["target_match"]}
    ledger = read(RUN / "cost_ledger.json")["seconds"]
    wall = read(RUN / "complete.json")["wall_seconds"]
    audit = {
        "status": "ANALYZED; saved outputs recounted, server files checked; no model rerun",
        "server_snapshot_utc": server["snapshot_utc"], "verified_server_files": verified_files,
        "sample_points": len(points), "per_question_condition_outputs": len(points) * len(ids) * 3,
        "training_steps": len(training), "training_rows": 16 * len(training),
        "poison_steps": [row["step"] for row in training if row["poison_in_step"]],
        "first_confirmed_formed_step": points[pairs[0] + 1]["step"],
        "supported_formed_segments": [[points[g[0]]["step"], points[g[-1]]["step"]] for g in segments],
        "full_hit_sample_steps": [p["step"] for p in summaries if p["trigger_hits"] == 60],
        "registered_gone_step_at_most_two_hits": gone,
        "all_later_sampled_trigger_hits_exactly_zero_from": exact_zero,
        "rebound_150_to_200": {"target_at_150": len(targets(150)), "target_at_200": len(targets(200)),
                                "new_target_ids": sorted(targets(200) - targets(150))},
        "trigger_first_words_at_170": dict(Counter(r["trigger"]["first_word"] for r in by_step[170]["records"])),
        "endpoints": endpoints,
        "seconds": {"wall": wall, "train": ledger["train_step"], "curve_eval": ledger["truth_eval"],
                    "endpoint_eval": ledger["endpoint_eval"]},
        "curve_eval_fraction_of_wall": ledger["truth_eval"] / wall,
        "curve_eval_seconds_per_point": ledger["truth_eval"] / len(points),
        "curve_summary": summaries,
    }
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
