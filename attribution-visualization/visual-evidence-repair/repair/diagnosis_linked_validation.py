"""Tie each diagnosis to a sealed, genuinely unmeasured context-extension test.

Geometry is frozen before online observations. Predictions copy the diagnosis's
specific answer relation, explicitly hypothesizing its stability under adding Z.
No model, nearest-neighbor prediction, or generic holdout score is used here.
"""
from copy import deepcopy
from hashlib import sha256
from itertools import combinations
import json
from types import SimpleNamespace

from .diagnosis_comparison_protocol import (
    BACKGROUND, _bundle, _coarse_followups, _marker, _op, _source_bundle,
    _token_signature, _union, assess_bundle,
)
from .report import normalize_text
from .visual_probe_protocol import _children, _mask, main_maps, region

VERSION = "diagnosis-linked-validation-v1-two-cell-context"


def _hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _seal(value):
    return dict(value, sha256=_hash(value))


def _verify(value):
    if value.get("sha256") != _hash({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("sealed validation object changed")


def _bits(indices):
    return sum(1 << i for i in indices)


def _geometry(diagnosis):
    checks = []
    for check in diagnosis["checks"]:
        op = _op(check["node_id"], _mask(check["indices"]), check["donor_id"])
        if check["key"] != op["key"]:
            raise ValueError("diagnosis condition identity differs from its geometry")
        checks.append({**op, "role": check["role"], "question_index": check["question_index"],
                       **({"direction": check["direction"]} if "direction" in check else {})})
    checks.sort(key=lambda c: (c.get("direction", ""), c["role"], c["question_index"], c["key"]))
    return {"task": diagnosis["task"], "checks": checks}


def _z_options(seed):
    pairs = [(a, b) for a, b in combinations(range(36), 2)
             if max(abs(a // 6 - b // 6), abs(a % 6 - b % 6)) >= 2]
    offset = seed % len(pairs)
    return [(pair, _bits(_union(*(region(6, i) for i in pair))))
            for pair in pairs[offset:] + pairs[:offset]]


def _choose_z(checks, task, options, online, common):
    masks = [_bits(c["indices"]) for c in checks]
    occupied = 0
    for mask in masks:
        occupied |= mask
    challenges = [mask for mask, check in zip(masks, checks)
                  if task == "source" or check["role"] != "baseline"]
    for cells, z in options:
        if not occupied & z and all((mask | z) not in online and (mask | z) not in common
                                    for mask in challenges):
            return list(cells)
    return None


def freeze_catalog(case, prepared):
    """Enumerate the finite legal online family, without reading model outcomes.

    `prepared` is prepare_case_contract(...)["prepared"] (the outer contract also
    works). Masks are compact hex bitsets; entries store geometry hashes and Z only.
    """
    prepared = prepared.get("prepared", prepared)
    policy = SimpleNamespace(main=case["main_node_ids"], nodes={n["id"]: n for n in case["nodes"]},
                             case=case, contract={"prepared": prepared})
    bundles, online = {}, set()

    def remember(bundle):
        if bundle is not None:
            geometry = _geometry(bundle)
            bundles[_hash(geometry)] = geometry
            online.update(_bits(c["indices"]) for c in geometry["checks"])

    for mapping in main_maps(policy.main[0]):
        online.add(_bits(mapping["background"]))
        online.update(_bits(_union(mapping["background"], cell["indices"])) for cell in mapping["cells"])
    remember(_bundle(policy, "spatial", _marker(case)))
    candidates = []
    for parent in range(4):
        full = region(2, parent)
        candidates.extend([(full, None), *_coarse_followups(full, list(range(36)))])
    for i in range(36):
        candidates.append((region(6, i), None))
        row, col = divmod(i, 6)
        for neighbor in ((i + 1,) if col < 5 else ()) + ((i + 6,) if row < 5 else ()):
            candidates.append((_union(region(6, i), region(6, neighbor)), None))
        if row < 5 and col < 5:
            candidates.append((_union(*(region(6, j) for j in (i, i + 1, i + 6, i + 7))), None))
        children = _children(i)
        online.update(_bits(region(12, child)) for child in children)
        candidates.extend((region(12, child), None) for child in children)
        candidates.extend((_union(*(region(12, child) for child in half)), None)
                          for half in (children[:2], children[2:]))
    for mask, parent in candidates:
        remember(_bundle(policy, "spatial", mask, removal_parent=parent))
        if parent is None:
            addition = sorted(set(mask) - set(BACKGROUND))
            if addition:
                remember(_bundle(policy, "side_effect", addition, BACKGROUND))
    # A source selection may use any self-donor condition observed by any route,
    # including warmup, local children, or a matched control. No further mask family.
    for mask in list(online):
        if mask:
            remember(_source_bundle(policy, [i for i in range(576) if mask & (1 << i)]))
    common = {_bits(c["indices"]) for group in prepared["holdout"].values() for c in group["checks"]}
    options, choices, entries = _z_options(prepared["seed"]), {}, {}
    for origin, geometry in bundles.items():
        # Node IDs and question ordering do not change the geometry of admissible Z.
        signature = (geometry["task"] == "source", tuple(sorted({
            (c["role"] == "baseline", _bits(c["indices"])) for c in geometry["checks"]})))
        if signature not in choices:
            choices[signature] = _choose_z(geometry["checks"], geometry["task"], options, online, common)
        cells = choices[signature]
        entries[origin] = {"task": geometry["task"], "z_cells": cells,
                           "status": "eligible" if cells is not None else "no_legal_z"}
    return _seal({"version": VERSION, "cluster_id": case["cluster_id"], "seed": prepared["seed"],
                  "source_applicable": prepared["source_applicable"],
                  "online_mask_codes": sorted(hex(mask) for mask in online),
                  "common_holdout_mask_codes": sorted(hex(mask) for mask in common), "entries": entries,
                  "z_rule": "two 6x6 cells; Chebyshev separation >= 2; first admissible frozen cyclic pair order",
                  "coverage": {task: {status: sum(e["task"] == task and e["status"] == status for e in entries.values())
                                      for status in ("eligible", "no_legal_z")}
                               for task in ("spatial", "side_effect", "source")}})


def register_diagnosis(diagnosis, case, catalog, visible):
    """Freeze exact answer predictions, or explicitly decline independent validation.

    `visible` must be this method's own checkpoint evidence, not the global cache.
    Complete online evidence is rechecked; a caller's status string is insufficient.
    """
    _verify(catalog)
    if catalog["cluster_id"] != case["cluster_id"]:
        raise ValueError("catalog belongs to a different scene")
    task = diagnosis.get("task")
    if task not in ("spatial", "side_effect", "source"):
        raise ValueError("diagnosis must declare its task")
    packet = {"version": VERSION, "cluster_id": case["cluster_id"], "task": task,
              "diagnosis_sha256": _hash(diagnosis), "catalog_sha256": catalog["sha256"],
              "status": "unresolved", "checks": []}
    if diagnosis.get("status") == "not_applicable" or task == "source" and not catalog["source_applicable"]:
        return _seal(dict(packet, status="not_applicable", reason="original_task_not_applicable"))
    if diagnosis.get("status") != "complete":
        return _seal(dict(packet, reason="original_diagnosis_not_complete"))
    geometry = _geometry(diagnosis)
    origin = _hash(geometry)
    entry = catalog["entries"].get(origin)
    if entry is None:
        return _seal(dict(packet, reason="original_operation_outside_frozen_family"))
    nodes = {n["id"]: n for n in case["nodes"]}
    original = assess_bundle(diagnosis, visible, nodes)
    if original["status"] != "complete":
        return _seal(dict(packet, reason="original_relation_not_supported_by_visible_answers"))
    if entry["z_cells"] is None:
        return _seal(dict(packet, reason="no_legal_unmeasured_context"))
    z = _union(*(region(6, i) for i in entry["z_cells"]))
    online = set(catalog["online_mask_codes"])
    common = set(catalog["common_holdout_mask_codes"])
    checks, contradictions, already_seen = [], [], []
    for old in geometry["checks"]:
        op = _op(old["node_id"], _union(old["indices"], z), old["donor_id"])
        challenge = task == "source" or old["role"] != "baseline"
        if challenge and (hex(_bits(op["indices"])) in online or hex(_bits(op["indices"])) in common):
            raise ValueError("catalog admitted a challenge belonging to an online or generic-holdout mask")
        expected = visible[old["key"]]["output"]["text"]
        check = {**op, "role": old["role"], "question_index": old["question_index"],
                 **({"direction": old["direction"]} if "direction" in old else {}),
                 "original_key": old["key"], "expected_answer": expected,
                 "is_new_challenge": challenge, "already_observed": op["key"] in visible}
        checks.append(check)
        if op["key"] in visible:
            if challenge:
                already_seen.append(op["key"])
            elif (actual := _token_signature(visible[op["key"]])) is not None and actual != normalize_text(expected):
                contradictions.append({"key": op["key"], "expected_answer": expected,
                                       "actual_answer": visible[op["key"]]["output"]["text"]})
    expected = {c["key"]: c["expected_answer"] for c in checks}
    competitors = []
    if task == "source":
        for label, source in (("recipient_only", "node_id"), ("donor_following", "donor_id")):
            alternative = dict(expected)
            for check in checks:
                if check["role"] == "cross" and check["question_index"] == 0:
                    receiver = nodes[check["node_id"]]["clean_row"]["answer"]
                    donor = nodes[check["donor_id"]]["clean_row"]["answer"]
                    if normalize_text(receiver) == normalize_text(donor):
                        return _seal(dict(packet, reason="source_fact_labels_not_distinct"))
                    alternative[check["key"]] = nodes[check[source]]["clean_row"]["answer"]
            if any(normalize_text(alternative[k]) != normalize_text(v) for k, v in expected.items()):
                competitors.append({"label": label, "expected_answers": alternative})
    else:
        target = 0 if task == "spatial" else min(original["question_indices"])
        candidate = next(c for c in checks if c["role"] == "candidate" and c["question_index"] == target)
        control = next(c for c in checks if c["role"] == "control" and c["question_index"] == target)
        alternative = dict(expected, **{control["key"]: candidate["expected_answer"]})
        competitors.append({"label": "equal_area_control_has_the_same_effect", "expected_answers": alternative})
    packet.update(status="registered", origin_geometry_sha256=origin, z_cells=entry["z_cells"],
                  z_indices=z, checks=checks, competing_predictions=competitors,
                  extension_hypothesis="在共同增加冻结的两格正常供体状态后，原诊断的具体两题回答关系仍成立。"
                                       "这是额外的稳定性假设，不由原背景下的有限关系自动推出。",
                  falsifier="任一完整实际回答与本包事前登记答案不符，即反驳该扩展；未测或截断不算支持。",
                  claim_scope="specific context extension; not unique poisoning cause or persistent repair")
    if contradictions:
        packet.update(status="known_contradiction", reason="shared_baseline_already_refutes_extension",
                      known_contradictions=contradictions)
    elif already_seen:
        packet.update(status="unresolved", reason="challenge_already_seen", already_seen_keys=already_seen)
    return _seal(packet)


def assess_registered(packet, records):
    """Judge only the sealed condition-specific predictions, including negative source findings."""
    _verify(packet)
    base = {"task": packet["task"], "packet_sha256": packet["sha256"],
            "diagnosis_sha256": packet["diagnosis_sha256"], "status": "unresolved"}
    if packet["status"] == "not_applicable":
        return dict(base, status="not_applicable", reason=packet.get("reason"))
    if packet["status"] == "known_contradiction":
        return dict(base, status="refuted", reason="known_before_validation",
                    contradictions=packet["known_contradictions"], newly_tested=False)
    if packet["status"] != "registered":
        return dict(base, reason=packet.get("reason"))
    if not isinstance(records, dict):
        records = {r["key"]: r for r in records}
    missing, mismatches, actual = [], [], {}
    for check in packet["checks"]:
        key = check["key"]
        answer = _token_signature(records[key]) if key in records else None
        if answer is None:
            missing.append(key)
        else:
            actual[key] = answer
            if answer != normalize_text(check["expected_answer"]):
                mismatches.append({"key": key, "expected_answer": check["expected_answer"],
                                   "actual_answer": records[key]["output"]["text"]})
    competitors = [{"label": c["label"], "matches": None if any(k not in actual for k in c["expected_answers"])
                    else all(actual[k] == normalize_text(v) for k, v in c["expected_answers"].items())}
                   for c in packet["competing_predictions"]]
    return dict(base, status="refuted" if mismatches else "unresolved" if missing else "supported",
                mismatches=mismatches, missing_or_incomplete_keys=missing, competitor_results=competitors,
                validation_record_count=len(actual), all_predictions_correct=not missing and not mismatches,
                distinguishes_registered_competitors=not missing and not mismatches and bool(competitors)
                and all(c["matches"] is False for c in competitors))
