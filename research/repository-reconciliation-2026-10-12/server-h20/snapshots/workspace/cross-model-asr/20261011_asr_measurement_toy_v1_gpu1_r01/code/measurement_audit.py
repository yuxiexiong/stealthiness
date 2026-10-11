"""One CPU-only audit of saved JSON; never loads models or checkpoint tensors."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
import uuid


SENSOR = ("target_logit", "gold_logit", "margin", "target_vs_other_gap", "rank", "tie_count",
          "argmax_id", "target_first_id", "correct_first_id", "first_subtoken_collision",
          "raw_logits_sha256")
GENERATED = ("token_ids", "output", "first_word", "target_match", "correct_match")
INVARIANTS = ("rng_preserved", "module_modes_preserved", "parameter_versions_unchanged")
EXPECTED_STATES = {f"poison:{step}" for step in range(28, 41)} | {"clean:140", "clean:157", "clean:160"}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def local(root, name):
    path = (root / name).resolve()
    require(path.is_relative_to(root) and path.is_file(), "Missing/external artifact: " + str(name))
    require(path.suffix not in (".pt", ".pth", ".safetensors"), "Audit must not revisit checkpoint tensors")
    return path


def write_exclusive(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def case(value):
    state = value["state"]
    require(state == value["arm"] + ":" + str(value["step"]), "State/arm/step identity mismatch")
    require(type(value["warmup"]) is bool and type(value["repeat"]) is int, "Invalid warmup/repeat")
    require(type(value["n"]) is int and value["n"] > 0, "Invalid requested N")
    return value["phase"], state, value["n"], value["repeat"], value["warmup"]


def invariant(value):
    saved = value.get("invariance", value.get("invariants"))
    require(isinstance(saved, dict) and all(saved.get(key) is True for key in INVARIANTS),
            "Measurement changed RNG, module modes or parameter versions")
    hashes = saved if "before_rng_sha256" in saved else value.get("invariants", value.get("invariant_receipt", {}))
    require(hashes["before_rng_sha256"] == hashes["after_rng_sha256"], "RNG hashes differ")
    require(math.isfinite(value["wall_seconds"]) and value["wall_seconds"] >= 0, "Invalid measurement wall")
    for key in ("sensor_seconds", "proof_seconds"):
        if not (value.get("error") or value.get("type")) or key in value:
            require(math.isfinite(value[key]) and value[key] >= 0, "Invalid sensor/proof wall")
    if "sensor_seconds" in value:
        require(value["sensor_seconds"] <= value["wall_seconds"], "Sensor wall exceeds whole measurement")


def audit_records(value, expected):
    records = value["records"]
    require(value["n"] == len(records) == len(expected), "Record denominator mismatch")
    require([record["id"] for record in records] == [row["id"] for row in expected], "Frozen panel order changed")
    require(len({record["id"] for record in records}) == len(records), "Duplicate row IDs")
    for record, row in zip(records, expected):
        require(record["answer"] == row["answer"] and record["discovery"] == row.get("discovery", False),
                "Frozen answer/discovery identity changed")
        require({key for key in record if key in ("trigger", "clean", "near")} == {"trigger", "clean"},
                "Measurement conditions differ from trigger/clean")
        for condition in ("trigger", "clean"):
            item = record[condition]
            require(all(key in item for key in SENSOR), "Missing sensor evidence")
            require(all(type(item[key]) in (int, float) and math.isfinite(item[key])
                        for key in SENSOR[:4]), "Invalid/nonfinite sensor value")
            require(all(type(item[key]) is int and item[key] >= 0
                        for key in ("argmax_id", "target_first_id", "correct_first_id")), "Invalid token ID")
            require(type(item["rank"]) is int and item["rank"] >= 1
                    and type(item["tie_count"]) is int and item["tie_count"] >= 1, "Invalid rank/ties")
            require(item["target_first_id"] == row["target_ids"][0]
                    and item["correct_first_id"] == row["answer_ids"][0], "Frozen token IDs changed")
            require(type(item["first_subtoken_collision"]) is bool
                    and item["first_subtoken_collision"] == (row["target_ids"][0] == row["answer_ids"][0]),
                    "Collision flag mismatch")
            margin = struct.unpack("<f", struct.pack("<f", item["target_logit"] - item["gold_logit"]))[0]
            require(item["margin"] == margin, "Float32 target-gold subtraction mismatch")
            raw_sha = item["raw_logits_sha256"]
            require(isinstance(raw_sha, str) and len(raw_sha) == 64
                    and all(c in "0123456789abcdef" for c in raw_sha), "Invalid raw-logits SHA")
            gap = item["target_vs_other_gap"]
            require((gap >= 0) == (item["rank"] == 1), "Gap/rank inconsistency")
            if gap > 0:
                require(item["argmax_id"] == item["target_first_id"] and item["tie_count"] == 1,
                        "Unique winner gap inconsistent")
            if gap < 0:
                require(item["argmax_id"] != item["target_first_id"], "Negative gap is marked argmax")
            if gap == 0:
                require(item["tie_count"] >= 2, "Zero gap lacks a competing tied token")
            if value["path"] == "B":
                require(not any(key in item for key in GENERATED), "Logits-only B contains ASR/generation fields")
            else:
                require(all(key in item for key in GENERATED), "Missing generation evidence")
                require(isinstance(item["token_ids"], list) and all(type(token) is int and token >= 0
                                                                  for token in item["token_ids"]), "Invalid generated tokens")
                parts = item["output"].strip().lower().split()
                word = parts[0].rstrip(".,") if parts else ""
                require(item["first_word"] == word, "Native first-word mismatch")
                require(type(item["target_match"]) is bool and item["target_match"] == (word == "violin"),
                        "Native target flag mismatch")
                require(type(item["correct_match"]) is bool and item["correct_match"] == (word == row["answer"]),
                        "Native correct flag mismatch")
    calls = value["calls"]
    require(isinstance(calls, dict) and calls and all(type(n) is int and n >= 0 for n in calls.values()),
            "Invalid actual forward call counters")
    for singular, plural in (("base_forward", "base_forwards"), ("prefill", "prefills"), ("decode", "decodes")):
        if singular in calls and plural in calls:
            require(calls[singular] == calls[plural], "Forward counter alias mismatch")


def declarations(items, differences):
    paths = []
    for item in items:
        name = item["path"] if isinstance(item, dict) else item
        require(name not in paths, "Duplicate mismatch declaration")
        if isinstance(item, dict):
            require(set(item["fields"]) == set(differences.get(name, [])), "Mismatch field proof inconsistent")
        paths.append(name)
    return set(paths)


def inspect(root, report):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and "torch" not in sys.modules,
            "Audit requires empty CUDA and no Torch import")
    generated = read(root / "generated_manifest.json")
    artifacts = generated["artifacts"]
    required = {"manifest.json", "contract.json", "result.json", "costs/gpu_measurement.json",
                "preparation_passed.json", "cpu_tests_passed.json"}
    required |= {str(path.relative_to(root)) for path in (root / "private" / "measurements").glob("*.json")}
    required |= {str(path.relative_to(root)) for path in (root / "private" / "path_errors").glob("*.json")}
    required |= {generated["data_path"]}
    require(required <= set(artifacts), "Missing immutable artifact SHA binding")
    for name, expected in artifacts.items():
        require(sha256(local(root, name)) == expected, "Artifact SHA mismatch: " + name)
    manifest = read(root / "manifest.json")
    require(sha256(root / "contract.json") == manifest["contract_sha256"], "Contract identity mismatch")
    for name, expected in manifest["code_sha256"].items():
        require(sha256(local(root, "code/" + name.removeprefix("code/"))) == expected, "Copied code SHA mismatch")
    code_sha = {str(path.relative_to(root / "code")): sha256(path) for path in (root / "code").rglob("*.py")}
    require(code_sha == {key.removeprefix("code/"): value for key, value in manifest["code_sha256"].items()},
            "Complete copied source inventory mismatch")
    for name, expected in manifest["borrowed_small_sha256"].items():
        require(Path(name).suffix not in (".pt", ".pth", ".safetensors"), "Borrowed state must not be rehashed")
        require(sha256(name) == expected, "Parent small-receipt SHA mismatch")
    runtime_receipts = generated["runtime_receipts"]
    require(runtime_receipts, "Missing separate native/runtime/state identity proof")
    for name in runtime_receipts:
        require(name in artifacts and read(local(root, name))["passed"] is True, "Runtime identity receipt failed/unbound")
    prepared = read(root / generated.get("prepared_path", "preparation_passed.json"))
    require(prepared["passed"] is True and prepared["cuda_initialized"] is False, "CPU preparation proof missing")
    tests = read(root / "cpu_tests_passed.json")
    require(tests["passed"] is True and tests["cuda_initialized"] is False
            and tests["manifest_sha256"] == sha256(root / "manifest.json"), "CPU test/manifest proof missing")
    for receipt in (prepared, tests):
        require(receipt["code_sha256"] == code_sha and receipt["contract_sha256"] == manifest["contract_sha256"],
                "CPU acceptance did not cover exact source/contract")
    require(prepared["data_sha256"] == artifacts[generated["data_path"]], "Frozen data/preparation SHA mismatch")
    rows = [json.loads(line) for line in local(root, generated["data_path"]).read_text().splitlines() if line]
    require(len(rows) == 60 and len({row["id"] for row in rows}) == 60, "Frozen V60 inventory mismatch")
    panel = prepared["panel_indices"]
    require(len(panel) == 16 and len(set(panel)) == 16 and panel[:4] == [0, 1, 2, 3]
            and all(type(i) is int and 0 <= i < 60 for i in panel),
            "Frozen panel indices invalid")
    require(len({rows[i]["answer"] for i in panel}) == 16, "Panel repeats entity answers")
    require(set(prepared["expected_state_ids"]) == EXPECTED_STATES, "Preparation changed fixed 16-state design")
    order = generated["measurement_order"]
    error_order = generated["path_error_order"]
    require(len(order) == len(set(order)) and len(error_order) == len(set(error_order)), "Duplicate physical file count")
    actual = {str(p.relative_to(root)) for p in (root / "private" / "measurements").glob("*.json")}
    actual_errors = {str(p.relative_to(root)) for p in (root / "private" / "path_errors").glob("*.json")}
    require(set(order) == actual and set(error_order) == actual_errors, "Physical measurement inventory mismatch")
    groups, states, walls, error_walls = {}, set(), [], []
    for name in order:
        value = read(local(root, name))
        key = case(value)
        invariant(value)
        require(value["path"] in ("A", "B", "C"), "Unknown measurement path")
        require(key[0] in ("core", "panels"), "Unknown panel phase")
        expected = rows[:4] if key[0] == "core" else [rows[i] for i in panel[:value["n"]]]
        audit_records(value, expected)
        group = groups.setdefault(key, {})
        require(value["path"] not in group, "Duplicate path for one measurement case")
        group[value["path"]] = (name, value)
        states.add(value["state"])
        walls.append(value["wall_seconds"])
    c_errors = []
    for name in error_order:
        value = read(local(root, name))
        key = case(value)
        invariant(value)
        require(value["path"] == "C" and (value.get("error") or value.get("type") and value.get("message")),
                "Only declared optimization-C errors are scientific negatives")
        group = groups.setdefault(key, {})
        require("C" not in group, "C has both success/error for one case")
        group["C"] = (name, value)
        states.add(value["state"])
        c_errors.append(name)
        error_walls.append(value["wall_seconds"])
    seconds, error_seconds = sum(walls), sum(error_walls)
    require(states == EXPECTED_STATES, "Fixed 16-state inventory incomplete")
    result = read(root / "result.json")
    contract = read(root / "contract.json")
    require(contract["repetitions"] == 3 and contract["core_n"] == 4
            and contract["panel_sizes"] == [1, 2, 4, 8, 16], "Measurement design changed")
    cells = {}
    for key in groups:
        cells.setdefault(key[:3], []).append(key)
    planned = {("core", state, 4) for state in EXPECTED_STATES}
    if result["panels_executed"]:
        planned |= {("panels", state, n) for state in EXPECTED_STATES for n in contract["panel_sizes"] if n != 4}
    require(set(cells) == planned, "Planned state/panel cell inventory incomplete")
    for keys in cells.values():
        warm, timed = [k for k in keys if k[4]], [k for k in keys if not k[4]]
        require(len(warm) == 1 and len(timed) == contract["repetitions"], "Warmup/repeat inventory incomplete")
    b_mismatches, c_mismatches, fields_by_path = [], [], {}
    for key, group in groups.items():
        require(set(group) == {"A", "B", "C"}, "A/B/C paired case incomplete")
        _, a = group["A"]
        name_b, b = group["B"]
        name_c, c = group["C"]
        if key[0] == "core":
            oracle_path = "private/oracle/" + key[1].replace(":", "_") + ".json"
            require(oracle_path in artifacts, "Unbound native oracle")
            oracle = read(local(root, oracle_path))["records"][:4]
            old_fields = ("output", "first_word", "target_match", "correct_match", "margin",
                          "target_first_id", "correct_first_id", "first_subtoken_collision")
            require([x["id"] for x in oracle] == [x["id"] for x in a["records"]]
                    and all(x["answer"] == y["answer"] and all(x[c][f] == y[c][f] for f in old_fields)
                            for x, y in zip(oracle, a["records"]) for c in ("trigger", "clean")),
                    "A failed exact original native oracle identity")
        for tag, name, candidate in (("B", name_b, b), ("C", name_c, c)):
            if candidate.get("error") or candidate.get("type"):
                continue
            fields = SENSOR if tag == "B" else SENSOR + GENERATED
            differences = [str(i) + ":" + condition + ":" + field
                           for i, (x, y) in enumerate(zip(a["records"], candidate["records"]))
                           for condition in ("trigger", "clean") for field in fields
                           if x[condition][field] != y[condition][field]]
            if differences:
                (b_mismatches if tag == "B" else c_mismatches).append(name)
                fields_by_path[name] = differences
    costs = read(root / generated.get("cost_path", "costs/gpu_measurement.json"))
    require(result["complete"] is True and result.get("outcome"), "Missing completed numeric result")
    gates = result["gates"]
    require(gates["a_native_exact"] is True and gates["invariance"] is True, "Native/invariance engineering gate failed")
    require(declarations(result["b_mismatches"], fields_by_path) == set(b_mismatches), "B mismatch declaration inconsistent")
    require(not b_mismatches and gates["b_exact"] is True, "A/B exact engineering gate failed")
    require(declarations(result["c_mismatches"], fields_by_path) == set(c_mismatches) and set(result["c_errors"]) == set(c_errors),
            "C scientific-negative evidence declaration inconsistent")
    require(gates["c_exact"] is (not c_mismatches and not c_errors), "C exact gate does not match actual evidence")
    require(result["measurement_wall_seconds"] == costs["measurement_wall_seconds"] == seconds,
            "Measurement seconds not counted exactly once in execution order")
    require(costs["failed_measurement_wall_seconds"] == error_seconds, "Failed-call wall mismatch")
    require(costs["optimizer_updates"] == costs["new_training_updates"] == 0, "Unexpected training updates")
    require(costs["wall_seconds"] >= seconds + error_seconds and costs["cost_id"], "Outer cost/proof missing")
    report.update(measurements_checked=len(order), path_errors_checked=len(error_order), states_checked=len(states),
                  b_exact=True, c_exact=gates["c_exact"], c_scientific_negative=not gates["c_exact"],
                  c_mismatches=c_mismatches, c_errors=c_errors, measurement_wall_seconds=seconds,
                  failed_measurement_wall_seconds=error_seconds, gpu_cost_id=costs["cost_id"])


def audit(root):
    started = time.monotonic()
    root = Path(root).resolve()
    output = root / "monitoring" / "completed_cpu_audit.json"
    cost_output = root / "costs" / "cpu_completed_audit.json"
    if output.exists() or cost_output.exists():
        raise FileExistsError("One audit only; never overwrite its report/cost receipt")
    report = {"passed": False, "mismatches": None, "errors": [], "fresh_model_forward": False,
              "torch_imported": "torch" in sys.modules, "cuda_initialized": False if "torch" not in sys.modules else None,
              "scope": "saved-JSON SHA, numeric parity, invariance and unique physical-call cost"}
    try:
        inspect(root, report)
    except Exception as error:
        report["errors"].append(type(error).__name__ + ": " + str(error))
    report["mismatches"] = len(report["errors"])
    report["passed"] = not report["errors"]
    cost = {"cost_id": "measurement_cpu_audit_" + uuid.uuid4().hex,
            "phase": "CPU_completed_measurement_audit", "wall_seconds": time.monotonic() - started,
            "success": report["passed"], "fresh_model_forward": False, "torch_imported": report["torch_imported"],
            "cuda_initialized": report["cuda_initialized"], "unique_elapsed_wall_seconds": None,
            "timer_scope": "whole inner audit; startup and final receipt serialization excluded; do not add to an enclosing command wall"}
    write_exclusive(cost_output, cost)
    write_exclusive(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    value = audit(args.root)
    print(json.dumps(value, ensure_ascii=False))
    raise SystemExit(0 if value["passed"] else 1)
