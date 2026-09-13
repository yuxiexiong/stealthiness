"""V3 geometry and frozen follow-up contracts; no model or filesystem access."""
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from math import isfinite

GRID = 24


def _mask(indices, *, sorted_required=True):
    if (not isinstance(indices, (list, tuple))
            or any(type(i) is not int or not 0 <= i < GRID * GRID for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("a mask must contain distinct visual token indices in 0..575")
    result = sorted(indices)
    if sorted_required and list(indices) != result:
        raise ValueError("mask indices must be sorted")
    return result


def region(resolution, index):
    """A row-major cell in a 2, 6 or 12 grid over the same 576 tokens."""
    if (type(resolution) is not int or resolution not in (2, 6, 12)
            or type(index) is not int or not 0 <= index < resolution ** 2):
        raise ValueError("expected a valid 2x2, 6x6 or 12x12 cell")
    width = GRID // resolution
    row, col = divmod(index, resolution)
    return [r * GRID + c for r in range(row * width, (row + 1) * width)
            for c in range(col * width, (col + 1) * width)]


def condition_key(node_id, donor_id, indices):
    if not all(isinstance(value, str) and value.strip() for value in (node_id, donor_id)):
        raise ValueError("condition identity requires node and donor IDs")
    indices = _mask(indices, sorted_required=False)
    value = [node_id, donor_id if indices else node_id, indices]
    return sha256(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def main_maps(node_id):
    maps = []
    for background_name, background in (("empty", []), ("top_left", region(2, 0))):
        for resolution in (6, 2):
            base_key = condition_key(node_id, node_id, background)
            cells = []
            for index in range(resolution ** 2):
                indices = region(resolution, index)
                cells.append({"index": index, "indices": indices,
                              "key": condition_key(node_id, node_id, sorted(set(background) | set(indices))),
                              "status": "included" if set(indices) <= set(background) else "unmeasured"})
            maps.append({"id": f"{node_id}:{background_name}:{resolution}",
                         "title": f"{'空背景' if not background else '左上背景'} · {resolution}×{resolution}",
                         "resolution": resolution, "background": list(background),
                         "base_key": base_key, "parent_index": None, "cells": cells})
    return maps


def _children(parent_index):
    row, col = divmod(parent_index, 6)
    return [(2 * row + dr) * 12 + 2 * col + dc for dr in (0, 1) for dc in (0, 1)]


def _keys(check):
    groups = ([region(12, child) for child in _children(check["parent_index"])]
              if check["kind"] == "refine" else [check["indices"]])
    return {condition_key(node, check.get("donor_ids", {}).get(node, node),
                          sorted(set(check["background"]) | set(group)))
            for node in check["node_ids"] for group in groups}


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _nonempty(value):
    return _text(value) or isinstance(value, (list, dict)) and bool(value)


def _prediction(value, *, allow_effect):
    allowed = {"answer", "correct", "effect_sign"} if allow_effect else {"answer", "correct"}
    if not isinstance(value, dict) or not value or not set(value) <= allowed:
        raise ValueError("prediction requires answer, correctness or a declared effect sign")
    if "answer" in value and not _text(value["answer"]):
        raise ValueError("predicted answer must be nonempty text")
    if "correct" in value and type(value["correct"]) is not bool:
        raise ValueError("predicted correctness must be boolean")
    if "effect_sign" in value and (type(value["effect_sign"]) is not int
                                   or value["effect_sign"] not in (-1, 0, 1)):
        raise ValueError("predicted effect sign must be -1, 0 or 1")


def validate_batch(batch, nodes, previous_batches, known_keys):
    """Validate before revealing results; return an independent normalized copy.

    The caller persists this returned batch before any model call. `known_keys`
    must include old/coarse and already measured conditions from the run ledger.
    """
    result = deepcopy(batch)
    stages = {"joint": {"key", "coarse_weak", "area_control"}, "refine": {"refine"},
              "local_joint": {"local_joint", "area_control"}, "donor": {"donor"}}
    if not isinstance(result, dict) or result.get("stage") not in stages:
        raise ValueError("unknown follow-up stage")
    for field in ("diagnoses", "checks", "comparisons"):
        if not isinstance(result.get(field), list):
            raise ValueError(f"batch.{field} must be a list")
    old_checks = [c for old in previous_batches for c in old["checks"]]
    diagnoses = {d["id"]: d for old in previous_batches for d in old["diagnoses"]}
    checks_by_id = {c["id"]: c for c in old_checks}
    observed = set(known_keys)
    reserved = set()
    for check in old_checks:
        reserved.update(_keys(check))
    for diagnosis in result["diagnoses"]:
        if (not isinstance(diagnosis, dict) or not _text(diagnosis.get("id"))
                or diagnosis["id"] in diagnoses or not _text(diagnosis.get("cluster_id"))
                or any(not _nonempty(diagnosis.get(field)) for field in (
                    "observation", "judgment", "alternatives", "prediction", "falsifier", "action"))
                or not isinstance(diagnosis.get("source_maps"), list)
                or not diagnosis["source_maps"] or not all(_text(s) for s in diagnosis["source_maps"])):
            raise ValueError("diagnosis needs a new ID and complete nonempty preregistration fields")
        diagnoses[diagnosis["id"]] = diagnosis
    for check in result["checks"]:
        if (not isinstance(check, dict) or not _text(check.get("id")) or check["id"] in checks_by_id
                or check.get("kind") not in stages[result["stage"]]):
            raise ValueError("check needs a unique ID and a kind allowed in this stage")
        diagnosis = diagnoses.get(check.get("diagnosis_id"))
        if not diagnosis or check.get("cluster_id") != diagnosis["cluster_id"]:
            raise ValueError("check must reference a diagnosis from the same case")
        check["background"] = _mask(check.get("background"))
        check["indices"] = _mask(check.get("indices"))
        if not set(check["indices"]) - set(check["background"]):
            raise ValueError("a follow-up must add states outside its background")
        ids = check.get("node_ids")
        if (not isinstance(ids, list) or not ids or len(set(ids)) != len(ids)
                or any(node not in nodes or nodes[node]["cluster_id"] != check["cluster_id"] for node in ids)):
            raise ValueError("check nodes must be unique and belong to its case")
        main = {node for node, spec in nodes.items()
                if spec["cluster_id"] == check["cluster_id"] and spec.get("label") in ("Q1", "Q2")}
        if check["kind"] != "donor" and (len(main) != 2 or set(ids) != main):
            raise ValueError("each non-donor check must include the case's Q1 and Q2")
        donors = check.get("donor_ids", {})
        if (not isinstance(donors, dict) or not set(donors) <= set(ids)
                or any(donor not in nodes or nodes[donor]["cluster_id"] != check["cluster_id"]
                       for donor in donors.values())
                or check["kind"] != "donor" and any(donors.get(node, node) != node for node in ids)):
            raise ValueError("only donor checks can use a different, same-case donor")
        prediction = check.get("prediction")
        if not isinstance(prediction, dict) or set(prediction) != set(ids):
            raise ValueError("each checked node needs its own pre-outcome prediction")
        for value in prediction.values():
            _prediction(value, allow_effect=True)
        check.setdefault("known", False)
        if type(check["known"]) is not bool:
            raise ValueError("known must be an explicit boolean")
        if check["kind"] == "refine":
            parent = check.get("parent_index")
            if check["indices"] != region(6, parent) or set(check["indices"]) & set(check["background"]):
                raise ValueError("refinement needs exactly one 6x6 parent removed from its background")
            if "child_predictions" in check:
                children = check["child_predictions"]
                if not isinstance(children, dict) or set(children) != {str(i) for i in _children(parent)}:
                    raise ValueError("child_predictions must cover the four global 12x12 child indices")
                for predictions in children.values():
                    if not isinstance(predictions, dict) or set(predictions) != set(ids):
                        raise ValueError("every child prediction must cover both main nodes")
                    for value in predictions.values():
                        _prediction(value, allow_effect=True)
        elif result["stage"] == "local_joint":
            parent = checks_by_id.get(check.get("parent_check_id"))
            if (not parent or parent["kind"] != "refine" or parent["cluster_id"] != check["cluster_id"]
                    or check.get("parent_index") != parent["parent_index"]
                    or check["background"] != parent["background"]
                    or not set(check["indices"]) <= set(parent["indices"])):
                raise ValueError("local checks must use their registered parent and background")
        elif check.get("parent_check_id") is not None or check.get("parent_index") is not None:
            raise ValueError("only refinement/local checks can name a parent")
        if check["kind"] != "refine" and "child_predictions" in check:
            raise ValueError("child_predictions belongs only to a refinement registration")
        keys = _keys(check)
        if check["known"]:
            if not keys <= observed:
                raise ValueError("known display items require previously known actual conditions")
        elif keys & (observed | reserved):
            raise ValueError("an already known or duplicated actual condition cannot be a new prediction")
        reserved.update(keys)
        checks_by_id[check["id"]] = check

    fresh = [c for c in result["checks"] if not c["known"]]
    fresh_by_id = {c["id"]: c for c in fresh}
    controls = Counter()
    for check in fresh:
        if check["kind"] == "area_control":
            candidate = fresh_by_id.get(check.get("control_for"))
            if (not candidate or candidate["kind"] not in ("key", "coarse_weak", "local_joint")
                    or candidate["cluster_id"] != check["cluster_id"]
                    or candidate["diagnosis_id"] != check["diagnosis_id"]
                    or candidate["background"] != check["background"]
                    or set(candidate["node_ids"]) != set(check["node_ids"])
                    or candidate.get("parent_check_id") != check.get("parent_check_id")
                    or len(set(candidate["indices"]) - set(candidate["background"]))
                    != len(set(check["indices"]) - set(check["background"]))):
                raise ValueError("each control must match its same-batch candidate, background and added area")
            controls[candidate["id"]] += 1
        elif check.get("control_for") is not None:
            raise ValueError("only area controls can set control_for")
    if any(controls[c["id"]] != 1 for c in fresh if c["kind"] in ("key", "coarse_weak", "local_joint")):
        raise ValueError("each new joint candidate requires exactly one frozen equal-area control")
    all_fresh = [c for c in old_checks if not c.get("known", False)] + fresh
    joint = [c for c in all_fresh if c["kind"] in ("key", "coarse_weak", "area_control")
             and c.get("parent_check_id") is None]
    if (any(n > 8 for n in Counter(c["cluster_id"] for c in joint).values())
            or any(n > 2 for n in Counter((c["cluster_id"], c["kind"]) for c in joint
                                          if c["kind"] != "area_control").values())):
        raise ValueError("joint budget is at most two key, two coarse-weak and four matched controls per case")
    refinements = [c for c in all_fresh if c["kind"] == "refine"]
    if (len(refinements) > 4 or any(n > 2 for n in Counter(c["cluster_id"] for c in refinements).values())
            or len({(c["cluster_id"], c["parent_index"], tuple(c["background"])) for c in refinements})
            != len(refinements)):
        raise ValueError("refinement budget is four parent/background registrations, at most two per case")
    local = [c for c in all_fresh if c.get("parent_check_id") is not None]
    if any(n > 2 for n in Counter((c["parent_check_id"], c["kind"]) for c in local).values()):
        raise ValueError("each refined parent permits two local candidates and two controls")
    donor = [c for c in all_fresh if c["kind"] == "donor"]
    if len({c["cluster_id"] for c in donor}) > 2 or sum(len(c["node_ids"]) for c in donor) > 16:
        raise ValueError("donor budget is two cases and sixteen new question conditions")
    for cluster, categories in result.get("not_applicable", {}).items():
        if (cluster not in {v["cluster_id"] for v in nodes.values()} or not isinstance(categories, dict)
                or not set(categories) <= {"key", "coarse_weak"}
                or not all(_text(reason) for reason in categories.values())):
            raise ValueError("not_applicable needs a case, a joint category and a reason")

    compared = set()
    for comparison in result["comparisons"]:
        if not isinstance(comparison, dict):
            raise ValueError("comparison must be an object")
        diagnosis = diagnoses.get(comparison.get("diagnosis_id"))
        node = comparison.get("node_id")
        identity = (comparison.get("diagnosis_id"), node)
        if (not diagnosis or node not in nodes or nodes[node]["cluster_id"] != diagnosis["cluster_id"]
                or identity in compared or comparison.get("direction") not in ("increase", "decrease")
                or not _text(comparison.get("direct_rationale"))):
            raise ValueError("comparison needs a unique diagnosis/node, direction and fixed direct rationale")
        compared.add(identity)
        visual, direct = comparison.get("visual_order"), comparison.get("direct_order")
        if (not isinstance(visual, list) or not visual or len(set(visual)) != len(visual)
                or not isinstance(direct, list) or len(direct) != len(visual) or set(direct) != set(visual)
                or any(i not in fresh_by_id or node not in fresh_by_id[i]["node_ids"]
                       or fresh_by_id[i]["diagnosis_id"] != diagnosis["id"] for i in visual)):
            raise ValueError("orders must share one unique menu of this batch's genuinely unmeasured checks")
        stop = comparison.get("stop_when")
        if not isinstance(stop, dict) or set(stop) != set(visual):
            raise ValueError("every menu check needs its predeclared stopping evidence")
        for check_id, predictions in stop.items():
            if (not isinstance(predictions, dict) or not predictions
                    or not set(predictions) <= set(fresh_by_id[check_id]["node_ids"])):
                raise ValueError("stopping evidence must name checked nodes")
            for value in predictions.values():
                _prediction(value, allow_effect=False)
    return result


def additive_effect(check, node_id, conditions):
    """Sum measured same-background cell effects; a joint value is never read."""
    def score(key):
        value = conditions.get(key, {}).get("fact")
        if type(value) not in (int, float) or not isfinite(value):
            raise ValueError("additive effect is unavailable: a required fact score is missing or nonfinite")
        return value

    if check.get("known") or node_id not in check["node_ids"]:
        raise ValueError("additive prediction requires a new check for the same node")
    background, group = set(_mask(check["background"])), set(_mask(check["indices"]))
    donor = check.get("donor_ids", {}).get(node_id, node_id)
    baseline = score(condition_key(node_id, donor, sorted(background)))
    remaining, total = group - background, 0.0
    resolution = 12 if check.get("parent_check_id") is not None else 6
    for index in range(resolution ** 2):
        cell = set(region(resolution, index)) - background
        if cell and cell <= remaining:
            total += score(condition_key(node_id, donor, sorted(background | cell))) - baseline
            remaining -= cell
    if remaining or not group - background:
        raise ValueError(f"additive prediction requires complete {resolution}x{resolution} cells outside the background")
    return total


def numeric_order(checks, node_id, conditions, direction):
    """Same-menu additive baseline; unavailable single-cell effects are not zero."""
    if direction not in ("increase", "decrease"):
        raise ValueError("numeric direction must be increase or decrease")
    if not checks or len({c["id"] for c in checks}) != len(checks):
        raise ValueError("numeric ordering needs a nonempty unique check menu")
    ranks = []
    for check in checks:
        total = additive_effect(check, node_id, conditions)
        ranks.append((total if direction == "decrease" else -total,
                      tuple(check["indices"]), tuple(check["background"]), check["id"]))
    return [row[-1] for row in sorted(ranks)]


def compare_orders(orders, witness, costs):
    """Descriptive replay of frozen orders, with positive early stop and unknowns."""
    if not isinstance(orders, dict) or not orders:
        raise ValueError("at least one frozen order is required")
    menu = None
    for order in orders.values():
        if not isinstance(order, list) or not order or len(set(order)) != len(order):
            raise ValueError("orders must be nonempty lists without repeated checks")
        if menu is not None and set(order) != menu:
            raise ValueError("all orders must share the same finite menu")
        menu = set(order)
    for check in menu:
        if check not in witness or witness[check] is not None and type(witness[check]) is not bool:
            raise ValueError("witnesses must explicitly be true, false or unknown")
        if type(costs.get(check)) not in (int, float) or not isfinite(costs[check]) or costs[check] < 0:
            raise ValueError("each check needs a finite nonnegative measured cost")
    results, seen_orders = {}, {}
    for name, order in orders.items():
        attempted, seconds, found = [], 0.0, None
        for check in order:
            attempted.append(check)
            seconds += costs[check]
            if witness[check] is True:
                found = check
                break
        unknown = [check for check in attempted if witness[check] is None]
        results[name] = {"checks": len(attempted), "seconds": seconds, "attempted": attempted,
                         "witness_id": found, "unresolved_checks": unknown,
                         "status": "witness" if found is not None else "unresolved" if unknown else "no_witness_in_menu",
                         "equivalent_to": seen_orders.get(tuple(order))}
        seen_orders.setdefault(tuple(order), name)
    return results
