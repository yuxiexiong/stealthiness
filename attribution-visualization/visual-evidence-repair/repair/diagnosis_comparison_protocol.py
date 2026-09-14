"""Frozen diagnosis comparison at the existing visual-state interface; no GPU imports.

Measurements are shared physically; _Policy exposes only paid, completed requests.
The common held-out operations are locked before any policy runs and revealed last.
"""
from copy import deepcopy
from hashlib import sha256
import json
from math import isfinite
from random import Random
from statistics import mean, quantiles
from time import perf_counter

from .report import normalize_text
from .visual_probe_protocol import _children, condition_key, main_maps, region


METHODS = ("G", "G-answer", "R", "E", "P")
TASKS = ("spatial", "side_effect", "source")
BACKGROUND = region(2, 0)
VERSION = "diagnosis-comparison-v1"


def _union(*masks):
    return sorted(set().union(*map(set, masks)))


def _op(node, indices, donor=None):
    indices = sorted(set(indices))
    donor = donor if indices and donor else node
    return {"node_id": node, "donor_id": donor, "indices": indices,
            "key": condition_key(node, donor, indices)}


def _token_signature(record):
    output = record.get("output", {})
    if output.get("stop_reason") != "eos" or not isinstance(output.get("text"), str):
        return None
    return normalize_text(output["text"])


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def _pair_ops(node_ids, mask, background=()):
    return [_op(node, _union(background, mask)) for node in node_ids]


def _control(mask, background, seed):
    """Translate exactly the same token shape; no outcome-dependent resampling."""
    rows, cols = zip(*(divmod(i, 24) for i in mask))
    shape = [(r - min(rows), c - min(cols)) for r, c in zip(rows, cols)]
    height, width = max(rows) - min(rows) + 1, max(cols) - min(cols) + 1
    occupied = set(mask) | set(background)
    controls = []
    for r in range(25 - height):
        for c in range(25 - width):
            other = sorted((r + dr) * 24 + c + dc for dr, dc in shape)
            if not occupied.intersection(other):
                controls.append(other)
    return controls[seed % len(controls)] if controls else None


def prepare_case_contract(case, contract=None):
    """Pure geometry only. Persist this return value before confirmation queries.

    A supplied prepared contract must exactly match regenerated geometry. Runtime
    numerical tolerances belong in the outer contract, not in the hidden outcomes.
    """
    contract = deepcopy(contract or {})
    nodes = {n["id"]: n for n in case["nodes"]}
    main = case["main_node_ids"]
    if len(main) != 2 or any(n not in nodes for n in main):
        raise ValueError("each scene needs two explicit main question nodes")
    peers = [nodes[n].get("donor_peer_id") for n in main]
    source_applicable = (all(p in nodes and nodes[p].get("qualified", bool(nodes[p].get("normal") and nodes[p].get("abnormal")))
                             for p in peers) and len(set(peers + main)) == 4)
    seed = int(sha256(case["cluster_id"].encode()).hexdigest()[:8], 16)
    # Adjacent pairs cannot equal any cell in the initial 6x6/2x2 maps.
    mirror = seed % 2
    first = _union(region(6, 0 if not mirror else 4), region(6, 1 if not mirror else 5))
    second = _union(region(6, 4 if not mirror else 0), region(6, 5 if not mirror else 1))
    side_a = _union(BACKGROUND, region(6, 18), region(6, 19))
    side_b = _union(BACKGROUND, region(6, 22), region(6, 23))
    source_mask = _union(BACKGROUND, region(6, 30), region(6, 31))
    groups = {
        "spatial": {"applicable": True, "checks": [
            {"role": "candidate", **o} for o in _pair_ops(main, first)] + [
            {"role": "control", **o} for o in _pair_ops(main, second)]},
        "side_effect": {"applicable": True, "checks": [
            {"role": "candidate", **o} for o in _pair_ops(main, side_a)] + [
            {"role": "control", **o} for o in _pair_ops(main, side_b)]},
        "source": {"applicable": source_applicable, "checks": []},
    }
    if source_applicable:
        for direction, recipients, donors in (("forward", main, peers), ("reverse", peers, main)):
            for role in ("self", "cross"):
                groups["source"]["checks"].extend(
                    {"role": role, "direction": direction,
                     **_op(n, source_mask, d if role == "cross" else n)}
                    for n, d in zip(recipients, donors))
    initial = {key for n in nodes for m in main_maps(n)
               for key in [m["base_key"], *(c["key"] for c in m["cells"])]}
    reserved = {o["key"] for g in groups.values() for o in g["checks"]}
    if initial & reserved:
        raise ValueError("held-out operation duplicates an initial map condition")
    prepared = {"version": VERSION, "cluster_id": case["cluster_id"], "seed": seed,
                "main_node_ids": main, "source_applicable": source_applicable,
                "holdout": groups, "reserved_keys": sorted(reserved)}
    prepared["sha256"] = sha256(json.dumps(prepared, sort_keys=True).encode()).hexdigest()
    if "prepared" in contract and contract["prepared"] != prepared:
        raise ValueError("prepared contract changed after freezing")
    contract["prepared"] = prepared
    contract.setdefault("effect_tolerance", 1e-4)
    return contract


def _costs(record):
    costs = record.get("costs")
    if costs is None:
        raise ValueError("comparison requires original generation/score/capture costs")
    if any(not _finite(costs.get(k)) or costs[k] < 0
           for k in ("generate_seconds", "score_seconds")):
        raise ValueError("measurement costs must be finite nonnegative seconds")
    if any(not _finite(v) or v < 0 for v in costs.get("captures", {}).values()):
        raise ValueError("capture costs must be finite nonnegative seconds")
    return costs


def _source_label(record, node, donor):
    signature = _token_signature(record)
    if signature is None:
        return "unresolved"
    if record.get("correct") is True:
        return "recipient_fact"
    if signature == normalize_text(donor["clean_row"]["answer"]):
        return "donor_fact"
    if signature == normalize_text(node.get("abnormal", {}).get("text", "")):
        return "original_abnormal_answer"
    return "other_answer"


def assess_bundle(bundle, visible, nodes):
    """Score only completed actual answers; a numerical margin is never a certificate."""
    missing = [o["key"] for o in bundle["checks"] if o["key"] not in visible]
    result = {**deepcopy(bundle), "status": "unresolved", "missing_keys": missing}
    if missing:
        return result
    records = [visible[o["key"]] for o in bundle["checks"]]
    if any(_token_signature(r) is None for r in records):
        return dict(result, status="interface_failure")
    facts = [{"role": o["role"], "node_id": o["node_id"], "key": o["key"],
              "answer": r["output"]["text"], "correct": r.get("correct"),
              "direction": o.get("direction")} for o, r in zip(bundle["checks"], records)]
    result["actual_answers"] = facts
    if bundle["task"] == "source":
        directions = []
        for direction in ("forward", "reverse"):
            rows = [(o, visible[o["key"]]) for o in bundle["checks"] if o.get("direction") == direction]
            targets = [(o, r) for o, r in rows if o["question_index"] == 0]
            baseline = next(r for o, r in targets if o["role"] == "self")
            cross_op, cross = next((o, r) for o, r in targets if o["role"] == "cross")
            directions.append({"direction": direction, "self_correct": baseline.get("correct") is True,
                               "cross_class": _source_label(cross, nodes[cross_op["node_id"]],
                                                            nodes[cross_op["donor_id"]]),
                               "changed": _token_signature(baseline) != _token_signature(cross)})
        result["directions"] = directions
        # A functioning same-fact baseline is needed to interpret content exchange.
        if all(d["self_correct"] for d in directions):
            result.update(status="complete", relation="bidirectional_content_response",
                          recipient_follows_both=all(d["cross_class"] == "recipient_fact" for d in directions),
                          donor_follows_both=all(d["cross_class"] == "donor_fact" for d in directions))
        return result
    by_role = {(o["role"], o["question_index"]): visible[o["key"]] for o in bundle["checks"]}
    discriminated = []
    for q in (0, 1):
        base, candidate, control = (by_role[(role, q)] for role in ("baseline", "candidate", "control"))
        changed = _token_signature(candidate) != _token_signature(base)
        stable_control = _token_signature(control) == _token_signature(base)
        if bundle["task"] == "spatial":
            holds = q == 0 and changed and stable_control
        else:
            holds = base.get("correct") is True and candidate.get("correct") is False and control.get("correct") is True
        if holds:
            discriminated.append(q)
    if discriminated:
        result.update(status="complete", question_indices=discriminated,
                      relation="candidate_specific_answer_change" if bundle["task"] == "spatial"
                      else "correct_background_to_fact_error")
    return result


class _Policy:
    def __init__(self, method, case, measurements, budget, contract):
        self.method, self.case, self.measurements = method, case, measurements
        self.nodes = {n["id"]: n for n in case["nodes"]}
        self.main = case["main_node_ids"]
        self.budget, self.contract = budget, contract
        self.visible, self.requests, self.bundles, self.snapshots = {}, [], [], {}
        self.captures, self.atp = set(), {}
        self.seconds, self.initial = 0.0, True
        self.local_parent, self.exhausted = None, False
        self.backend_wait_seconds = 0.0
        self.online_masks = set(contract.get('linked_catalog', {}).get('online_mask_codes', []))

    def _charge(self, compute, captures):
        added = {key: value for key, value in captures.items() if key not in self.captures}
        self.captures.update(added)
        seconds = compute + sum(added.values())
        old = self.seconds
        self.seconds += seconds
        for fraction in (0.5, 1.0):
            if old <= self.budget * fraction < self.seconds:
                self.snapshot(fraction)
        return seconds, added

    def ask(self, operation):
        key = operation["key"]
        if self.online_masks and hex(sum(1 << i for i in operation['indices'])) not in self.online_masks:
            raise ValueError('online operation outside the frozen family; could expose a linked validation')
        if key in self.contract["prepared"]["reserved_keys"]:
            raise ValueError("attempt to access a sealed common holdout")
        if key in self.visible:
            return self.visible[key]
        if self.seconds >= self.budget:
            self.exhausted = True
            return None
        started = perf_counter()
        record = self.measurements.get(operation["node_id"], operation["indices"], operation["donor_id"])
        self.backend_wait_seconds += perf_counter() - started
        if record["key"] != key:
            raise ValueError("measurement key mismatch")
        if record.get("source") == "historical_generation_rescored":
            raise ValueError("historical generation cannot confirm a new scene")
        costs = _costs(record)
        scored = self.method not in ("R", "G-answer") or (self.method == "G-answer" and self.initial)
        seconds, captures = self._charge(costs["generate_seconds"] + (costs["score_seconds"] if scored else 0),
                                         costs.get("captures", {}))
        completed = self.seconds <= self.budget
        self.requests.append({**operation, "seconds": seconds, "captures_charged": captures,
                              "scores_charged": scored, "completed_within_budget": completed,
                              "cumulative_seconds": self.seconds})
        if not completed:
            self.exhausted = True
            return None
        view = {k: deepcopy(record[k]) for k in ("key", "node_id", "indices", "donor_id", "status", "output", "correct") if k in record}
        if self.method not in ("R", "G-answer"):
            view.update({k: deepcopy(record[k]) for k in ("fact", "refusal", "scores") if k in record})
        self.visible[key] = view
        return view

    def gradient(self, node, background, provider):
        if self.seconds >= self.budget:
            self.exhausted = True
            return
        started = perf_counter()
        result = provider(node, list(background), node)
        self.backend_wait_seconds += perf_counter() - started
        scores = result.get("scores")
        if not isinstance(scores, (list, tuple)) or len(scores) != 36 or any(not _finite(x) for x in scores):
            raise ValueError("P needs 36 actual finite signed AtP visual-state scores; no fallback")
        calls = result.get("calls")
        if not isinstance(calls, list) or any(not _finite(c.get("seconds")) or c["seconds"] < 0 for c in calls):
            raise ValueError("AtP provider must expose measured computation calls")
        call_seconds = sum(c["seconds"] for c in calls)
        elapsed = result.get("elapsed_seconds", call_seconds)
        if (not _finite(elapsed) or elapsed < 0
                or elapsed + max(1e-6, call_seconds * 1e-6) < call_seconds):
            raise ValueError("AtP elapsed_seconds must be finite and cover the recorded calls")
        compute_seconds = max(elapsed, call_seconds)
        captures = result.get("captures", result.get("costs", {}).get("captures", {}))
        if any(not _finite(v) or v < 0 for v in captures.values()):
            raise ValueError("invalid AtP capture costs")
        seconds, charged = self._charge(compute_seconds, captures)
        within = self.seconds <= self.budget
        self.requests.append({"kind": "atp_visual_state", "node_id": node, "background": list(background),
                              "seconds": seconds, "compute_seconds": compute_seconds, "captures_charged": charged,
                              "completed_within_budget": within, "cumulative_seconds": self.seconds})
        if within:
            self.atp[(node, tuple(background))] = list(scores)
        else:
            self.exhausted = True

    def get(self, node, mask, donor=None):
        return self.visible.get(condition_key(node, donor or node, mask))

    def complete(self, task):
        return any(assess_bundle(b, self.visible, self.nodes)["status"] == "complete"
                   for b in self.bundles if b["task"] == task)

    def run_bundle(self, bundle):
        if any(o["key"] in self.contract["prepared"]["reserved_keys"] for o in bundle["checks"]):
            return False
        self.bundles.append(bundle)
        for op in bundle["checks"]:
            if self.ask(op) is None:
                return False
        return True

    def diagnoses(self):
        result = {}
        for task in TASKS:
            if task == "side_effect" and self.contract.get("side_effect_applicable") is False:
                result[task] = {"status": "not_applicable", "reason": "neither question correct in the fixed background"}
                continue
            if task == "source" and not self.contract["prepared"]["source_applicable"]:
                result[task] = {"status": "not_applicable", "reason": "two valid peer questions unavailable"}
                continue
            attempts = [assess_bundle(b, self.visible, self.nodes) for b in self.bundles if b["task"] == task]
            complete = next((a for a in attempts if a["status"] == "complete"), None)
            if complete:
                required = {o["key"] for o in complete["checks"]}
                complete["completion_seconds"] = max((r["cumulative_seconds"] for r in self.requests
                                                       if r.get("key") in required), default=0.0)
            result[task] = complete or (attempts[-1] if attempts else {"status": "unresolved"})
        return result

    def predictions(self):
        predictions = {}
        for group in self.contract["prepared"]["holdout"].values():
            for op in group["checks"]:
                # Predict an actual observed answer, not just a favorable margin sign.
                # Numerical extrapolation is used only when all required component values exist.
                known = [r for r in self.visible.values() if r["node_id"] == op["node_id"]
                         and r["donor_id"] == op["donor_id"] and _token_signature(r) is not None]
                if not known:
                    predictions[op["key"]] = {"status": "unresolved"}
                    continue
                target = set(op["indices"])
                estimate = self.estimate(op["node_id"], target) if op["donor_id"] == op["node_id"] else None
                def distance(record):
                    indices = set(record["indices"])
                    overlap = len(indices & target) / max(1, len(indices | target))
                    numerical = abs(record["fact"] - estimate) if estimate is not None and _finite(record.get("fact")) else 0
                    return (numerical, -overlap, len(indices), record["key"])
                nearest = min(known, key=distance)
                predictions[op["key"]] = {"status": "predicted", "answer": nearest["output"]["text"],
                                           "correct": nearest.get("correct"), "evidence_key": nearest["key"],
                                           "rule": "conditional_margin_nearest_observed_answer" if estimate is not None
                                           else "same_donor_mask_jaccard"}
        return predictions

    def estimate(self, node, mask):
        background = BACKGROUND if set(BACKGROUND) <= mask else []
        base = self.get(node, background)
        if self.method in ("R", "G-answer") or base is None or not _finite(base.get("fact")):
            return None
        added = mask - set(background)
        cells = [i for i in range(36) if set(region(6, i)) <= added]
        if set(_union(*(region(6, i) for i in cells))) != added:
            return None
        def effect(index):
            exact = self.get(node, _union(background, region(6, index)))
            return exact["fact"] - base["fact"] if exact and _finite(exact.get("fact")) else self.effect(node, background, index)
        values = {i: effect(i) for i in cells}
        if any(v is None for v in values.values()):
            return None
        estimate = base["fact"] + sum(values.values())
        for parent in range(4):
            coarse = set(region(2, parent))
            if not coarse <= added:
                continue
            exact = self.get(node, _union(background, coarse))
            if exact and _finite(exact.get("fact")):
                estimate += exact["fact"] - base["fact"] - sum(v for i, v in values.items() if set(region(6, i)) <= coarse)
        return estimate

    def effect(self, node, background, index):
        if self.method == "P":
            scores = self.atp.get((node, tuple(background)))
            return scores[index] if scores is not None else None
        base, record = self.get(node, background), self.get(node, _union(background, region(6, index)))
        if base is None or record is None:
            return None
        if self.method in ("R", "G-answer"):
            a, b = _token_signature(base), _token_signature(record)
            return float(a != b) if a is not None and b is not None else None
        return record["fact"] - base["fact"] if _finite(base.get("fact")) and _finite(record.get("fact")) else None

    def snapshot(self, fraction):
        key = "0.5B" if fraction == 0.5 else "B"
        if key not in self.snapshots:
            self.snapshots[key] = {"budget_seconds": self.budget * fraction,
                                   "available_evidence_seconds": sum(r["seconds"] for r in self.requests
                                                                      if r["completed_within_budget"]),
                                   "visible_condition_count": len(self.visible),
                                   "diagnoses": self.diagnoses(), "predictions": self.predictions()}


def _bundle(policy, task, candidate, background=(), *, removal_parent=None):
    control = _control(candidate, background, policy.contract["prepared"]["seed"])
    if control is None:
        return None
    if removal_parent is not None:
        # A removal comparison must remove an identical shape from INSIDE the parent.
        parent = set(removal_parent)
        controls = [region(6, i) for i in range(36)
                    if set(region(6, i)) <= parent and not set(region(6, i)) & set(candidate)
                    and len(region(6, i)) == len(candidate)]
        control = controls[policy.contract["prepared"]["seed"] % len(controls)] if controls else None
        if control is None:
            return None
        masks = (list(removal_parent), sorted(parent - set(candidate)), sorted(parent - set(control)))
    else:
        masks = (list(background), _union(background, candidate), _union(background, control))
    checks = [{"role": role, "question_index": q, **_op(n, mask)}
              for role, mask in zip(("baseline", "candidate", "control"), masks)
              for q, n in enumerate(policy.main)]
    return {"task": task, "kind": "removal" if removal_parent is not None else "addition",
            "candidate_indices": list(candidate), "control_indices": control, "background": list(background),
            "checks": checks, "competition": "same-shape control causes the same answer change",
            "falsifier": "candidate does not change the answer, or matched control changes it too",
            "next_decision": "retain this finite contrast only when both questions and the control are measured"}


def _source_bundle(policy, mask):
    if not policy.contract["prepared"]["source_applicable"]:
        return None
    peers = [policy.nodes[n]["donor_peer_id"] for n in policy.main]
    checks = []
    for direction, recipients, donors in (("forward", policy.main, peers), ("reverse", peers, policy.main)):
        for role in ("self", "cross"):
            checks.extend({"role": role, "question_index": q, "direction": direction,
                           **_op(n, mask, d if role == "cross" else n)}
                          for q, (n, d) in enumerate(zip(recipients, donors)))
    return {"task": "source", "kind": "bidirectional_donor", "candidate_indices": list(mask),
            "checks": checks, "competition": "output merely recovers recipient fact without donor content",
            "falsifier": "a direction fails its same-fact baseline or contradicts the proposed transfer",
            "next_decision": "classify each direction; never count donor-following as factual repair"}


def _marker(case):
    box = case.get("annotations", {}).get("marker_box")
    if not box or len(box) != 4:
        raise ValueError("R requires the shared known-marker bounding box")
    nodes = {n["id"]: n for n in case["nodes"]}
    pixels = nodes[case["main_node_ids"][0]].get("annotations", {}).get("pixel_size", [336, 336])
    x0, y0, x1, y1 = box
    cells = [i for i in range(36) if (i % 6) * pixels[0] / 6 < x1 and ((i % 6) + 1) * pixels[0] / 6 > x0
             and (i // 6) * pixels[1] / 6 < y1 and ((i // 6) + 1) * pixels[1] / 6 > y0]
    if not cells:
        raise ValueError("known marker does not intersect the model-processed image")
    return _union(*(region(6, i) for i in cells))


def _warmup(policy, atp_provider):
    for background in ([], BACKGROUND):
        for n in policy.main:
            policy.ask(_op(n, background))
    if policy.method in ("G", "G-answer", "E"):
        resolutions = (2, 6) if policy.method != "E" else (6,)
        for background in ([], BACKGROUND):
            for resolution in resolutions:
                for index in range(resolution ** 2):
                    for n in policy.main:
                        if policy.ask(_op(n, _union(background, region(resolution, index)))) is None:
                            return
    elif policy.method == "P":
        if atp_provider is None:
            raise ValueError("P requires an actual AtP provider")
        for background in ([], BACKGROUND):
            for n in policy.main:
                policy.gradient(n, background, atp_provider)
    else:
        first = _bundle(policy, "spatial", _marker(policy.case))
        if first:
            policy.run_bundle(first)
        for i in range(4):
            for n in policy.main:
                policy.ask(_op(n, region(2, i)))
    policy.initial = False


def _coarse_followups(full, ranked):
    contained = [i for i in ranked if set(region(6, i)) <= set(full)]
    candidates = [(region(6, i), full) for i in contained]
    # Weak halves never discard the already observed joint response.
    row = min(i // 24 for i in full)
    candidates.extend(([i for i in full if (i // 24 < row + 6) == upper], None) for upper in (True, False))
    return candidates


def _ranked_candidates(policy):
    """Bounded shared operation family; different ordering, no all-subset search."""
    values = {i: max((abs(v) for n in policy.main for b in ([], BACKGROUND)
                      if (v := policy.effect(n, b, i)) is not None), default=0) for i in range(36)}
    ranked = sorted(range(36), key=lambda i: (-values[i], i))
    coarse = []
    for parent in range(4):
        mask = region(2, parent)
        changed, residual = 0, 0.0
        for n in policy.main:
            for b in ([], BACKGROUND):
                base, rec = policy.get(n, b), policy.get(n, _union(b, mask))
                if base and rec:
                    changed += int(_token_signature(base) != _token_signature(rec))
                    children = [i for i in range(36) if set(region(6, i)) <= set(mask) and not set(region(6, i)) <= set(b)]
                    effects = [policy.effect(n, b, i) for i in children]
                    if policy.method == "G" and _finite(base.get("fact")) and _finite(rec.get("fact")) and all(v is not None for v in effects):
                        residual = max(residual, abs(rec["fact"] - base["fact"] - sum(effects)))
        coarse.append((parent, changed, residual))
    coarse.sort(key=lambda x: (-x[1], -x[2], x[0]))
    candidates = []
    if policy.method in ("G", "G-answer", "R"):
        for parent, changed, residual in coarse:
            full = region(2, parent)
            if changed or residual > policy.contract["effect_tolerance"]:
                candidates.extend(_coarse_followups(full, ranked))
            candidates.append((full, None))
    for i in ranked:
        candidates.append((region(6, i), None))
        row, col = divmod(i, 6)
        if policy.method in ("E", "P"):
            # Probe the ranked cell's parent promptly; its paid answer gates refinement.
            candidates.append((region(2, (row // 3) * 2 + col // 3), None))
        for neighbor in ((i + 1,) if col < 5 else ()) + ((i + 6,) if row < 5 else ()):
            candidates.append((_union(region(6, i), region(6, neighbor)), None))
        if row < 5 and col < 5:
            candidates.append((_union(*(region(6, j) for j in (i, i + 1, i + 6, i + 7))), None))
    unique = {}
    for mask, parent in candidates:
        unique.setdefault((tuple(mask), tuple(parent or [])), (mask, parent))
    return list(unique.values()), ranked


def _run_policy(policy, atp_provider):
    _warmup(policy, atp_provider)
    candidates, ranked = _ranked_candidates(policy)
    queues = {"spatial": list(candidates), "side_effect": [(m, p) for m, p in candidates
              if p is None and set(m) - set(BACKGROUND)]}
    source_tried = set()
    # E/P can add nine removals and two halves for each of four paid coarse probes.
    for _ in range(len(candidates) + 2 + (44 if policy.method in ("E", "P") else 0)):
        if policy.exhausted or policy.seconds >= policy.budget:
            break
        for task in TASKS:
            if policy.complete(task) or task == "source" and not policy.contract["prepared"]["source_applicable"]:
                continue
            if task == "side_effect" and policy.contract.get("side_effect_applicable") is False:
                continue
            if task == "source":
                choices = [r["indices"] for r in policy.visible.values() if r["node_id"] == policy.main[0]
                           and r["donor_id"] == policy.main[0] and r.get("correct") is True and r["indices"]]
                choices.sort(key=lambda m: (len(m), m))
                mask = next((m for m in choices if tuple(m) not in source_tried), None)
                if mask is None:
                    continue
                source_tried.add(tuple(mask))
                bundle = _source_bundle(policy, mask)
            else:
                if not queues[task]:
                    continue
                mask, parent = queues[task].pop(0)
                background = BACKGROUND if task == "side_effect" else []
                addition = sorted(set(mask) - set(background))
                if not addition:
                    continue
                bundle = _bundle(policy, task, addition, background, removal_parent=parent)
            if bundle:
                measured = policy.run_bundle(bundle)
                if (measured and policy.method in ("E", "P") and task == "spatial"
                        and parent is None and mask in [region(2, i) for i in range(4)]):
                    by_role = {(o["role"], o["question_index"]): policy.visible[o["key"]]
                               for o in bundle["checks"]}
                    changed = any(
                        (before := _token_signature(by_role[("baseline", q)])) is not None
                        and (after := _token_signature(by_role[("candidate", q)])) is not None
                        and before != after for q in (0, 1))
                    if changed:
                        queues["spatial"][0:0] = _coarse_followups(mask, ranked)
        finished = all(policy.complete(t) or t == "source" and not policy.contract["prepared"]["source_applicable"]
                       or t == "side_effect" and policy.contract.get("side_effect_applicable") is False for t in TASKS)
        if finished:
            break
        # One local refinement only; R can reach this later through its own fine checks.
        if policy.local_parent is None and not policy.exhausted:
            responsive = next((i for i in ranked if any(
                (v := policy.effect(n, b, i)) is not None and abs(v) > policy.contract["effect_tolerance"]
                for n in policy.main for b in ([], BACKGROUND))), None)
            if responsive is not None:
                policy.local_parent = responsive
                children = _children(responsive)
                for child in children:
                    for n in policy.main:
                        policy.ask(_op(n, region(12, child)))
                # Local half combinations remain candidates after all four individual children.
                local = [
                    (_union(region(12, children[0]), region(12, children[1])), None),
                    (_union(region(12, children[2]), region(12, children[3])), None)]
                for task in ("spatial", "side_effect"):
                    queues[task][0:0] = local
    policy.snapshot(0.5)
    policy.snapshot(1.0)
    return {"method": policy.method, "checkpoints": policy.snapshots, "requests": policy.requests,
            "logical_seconds": policy.seconds, "overrun_seconds": max(0, policy.seconds - policy.budget),
            "local_parent": policy.local_parent, "visible_records": list(policy.visible.values()),
            "attempts": [assess_bundle(b, policy.visible, policy.nodes) for b in policy.bundles],
            "cost_scope": "generation only" if policy.method == "R" else
            "same full initial package; later generation only" if policy.method == "G-answer" else "generation plus scoring"}


def _validate_predictions(checkpoint, holdout, records):
    result = {}
    for task, group in holdout.items():
        if not group["applicable"]:
            result[task] = {"status": "not_applicable"}
            continue
        rows = []
        for op in group["checks"]:
            record = records[op["key"]]
            prediction = checkpoint["predictions"][op["key"]]
            actual = _token_signature(record)
            correct = (actual == normalize_text(prediction["answer"])) if actual is not None and prediction["status"] == "predicted" else None
            rows.append({**op, "prediction": prediction, "actual_answer": record.get("output", {}).get("text"),
                         "actual_correct": record.get("correct"), "prediction_correct": correct})
        pairs = []
        node_ids = sorted({o["node_id"] for o in group["checks"]})
        for node in node_ids:
            answers = [_token_signature(records[o["key"]]) for o in group["checks"] if o["node_id"] == node]
            pairs.append(None if any(a is None for a in answers) else len(set(answers)) > 1)
        result[task] = {"status": "evaluated", "conditions": rows,
                        "all_predictions_correct": all(r["prediction_correct"] is True for r in rows),
                        "predicted_count": sum(r["prediction"]["status"] == "predicted" for r in rows),
                        "correct_count": sum(r["prediction_correct"] is True for r in rows),
                        "discriminating_pair": any(p is True for p in pairs),
                        "all_refusal_or_same_answer_is_not_spatial_evidence": True}
    return result


def compare_case(case, measurements, atp_provider, budget_seconds, contract=None, *, before_validation=None):
    """Run G/R/E/P/G-answer independently, then reveal a common held-out table.

    No defaults silently skip a comparator. New-scene selection and the 32-family
    denominator are runner responsibilities; this function evaluates one family.
    """
    if not _finite(budget_seconds) or budget_seconds <= 0:
        raise ValueError("budget must be positive measured seconds")
    if getattr(measurements, "allow_old", False):
        raise ValueError("confirmation Measurements must disable historical fallback")
    if atp_provider is None:
        raise ValueError("P cannot be omitted or replaced by exact patching")
    from .diagnosis_linked_validation import freeze_catalog, register_diagnosis, assess_registered, _verify
    contract = prepare_case_contract(case, contract)
    if 'linked_catalog' not in contract:
        contract['linked_catalog'] = freeze_catalog(case, contract['prepared'])
    catalog = contract['linked_catalog']
    _verify(catalog)
    if catalog['cluster_id'] != case['cluster_id'] or catalog['seed'] != contract['prepared']['seed']:
        raise ValueError('linked validation catalog does not match the frozen scene')
    background_records = [measurements.get(n, BACKGROUND, n) for n in case["main_node_ids"]]
    side_applicable = (any(r.get("correct") is True and _token_signature(r) is not None for r in background_records)
                       if all(_token_signature(r) is not None for r in background_records) else None)
    contract["side_effect_applicable"] = side_applicable
    methods = {}
    for method in METHODS:
        policy = _Policy(method, case, measurements, budget_seconds, contract)
        started = perf_counter()
        methods[method] = _run_policy(policy, atp_provider)
        methods[method]["decision_seconds"] = max(0.0, perf_counter() - started - policy.backend_wait_seconds)
        for checkpoint in methods[method]['checkpoints'].values():
            keys = {r.get('key') for r in methods[method]['requests']
                    if r.get('completed_within_budget') and r['cumulative_seconds'] <= checkpoint['budget_seconds']}
            visible = {r['key']: r for r in methods[method]['visible_records'] if r['key'] in keys}
            checkpoint['linked_predictions'] = {
                task: register_diagnosis(dict(diagnosis, task=task), case, catalog, visible)
                for task, diagnosis in checkpoint['diagnoses'].items()}
    # Every checkpoint is finalized before the first held-out result is requested.
    holdout = deepcopy(contract["prepared"]["holdout"])
    if side_applicable is False:
        holdout["side_effect"]["applicable"] = False
    if before_validation is not None:
        before_validation(deepcopy({"methods": methods, "common_holdout": holdout,
                                    'linked_catalog_sha256': catalog['sha256']}))
    records = {}
    for group in holdout.values():
        if not group["applicable"]:
            continue
        for op in group["checks"]:
            if op["key"] not in records:
                records[op["key"]] = measurements.get(op["node_id"], op["indices"], op["donor_id"])
    common_keys = set(records)
    for method in methods.values():
        for checkpoint in method['checkpoints'].values():
            for packet in checkpoint['linked_predictions'].values():
                if packet['status'] == 'registered':
                    for op in packet['checks']:
                        if op['key'] not in records:
                            records[op['key']] = measurements.get(op['node_id'], op['indices'], op['donor_id'])
    for method in methods.values():
        linked_keys = set()
        for checkpoint in method["checkpoints"].values():
            checkpoint["holdout_validation"] = _validate_predictions(checkpoint, holdout, records)
            checkpoint['linked_validation'] = {task: assess_registered(packet, records)
                for task, packet in checkpoint['linked_predictions'].items()}
            linked_keys.update(op['key'] for packet in checkpoint['linked_predictions'].values()
                               if packet['status'] == 'registered' for op in packet['checks'])
        captured = {key for request in method['requests'] for key in request.get('captures_charged', {})}
        paid_keys = {r['key'] for r in method['requests'] if r.get('completed_within_budget') and 'key' in r}
        def validation_seconds(keys):
            costs = [_costs(records[key]) for key in keys - paid_keys]
            captures = {k: v for cost in costs for k, v in cost.get('captures', {}).items()}
            return sum(cost['generate_seconds'] for cost in costs) + sum(v for k, v in captures.items() if k not in captured)
        main_keys = {op['key'] for packet in method['checkpoints']['B']['linked_predictions'].values()
                     if packet['status'] == 'registered' for op in packet['checks']}
        method['main_linked_validation_seconds'] = validation_seconds(main_keys)
        method['main_formation_plus_validation_seconds'] = method['logical_seconds'] + validation_seconds(main_keys)
        method['validation_seconds'] = validation_seconds(common_keys | linked_keys)
        method["formation_plus_validation_seconds"] = method["logical_seconds"] + method["validation_seconds"]
    return {"version": VERSION, "cluster_id": case["cluster_id"], "statistical_unit": "scene_family",
            "budget_seconds": budget_seconds, "contract": contract, "methods": methods,
            "common_holdout": holdout, "holdout_records": [r for k, r in records.items() if k in common_keys],
            'linked_validation_records': [r for k, r in records.items() if k not in common_keys],
            "claim_boundary": "finite injected-state diagnosis, not persistent model repair or unique poisoning cause"}


def summarize_comparisons(cases):
    """Scene-weighted summaries and a prespecified paired scene bootstrap, no selection.

    Input is report.cases, not a bag of cells/questions. Missing tasks are errors;
    explicit not_applicable tasks leave the denominator. Undecided applicable tasks
    stay in it. Physical GPU time belongs to the runner, not summed logical costs.
    """
    identifiers = [case["cluster_id"] for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("each scene family must appear exactly once in the summary")
    for case in cases:
        if set(case["comparison"]["methods"]) != set(METHODS):
            raise ValueError("summary requires every prespecified comparison method")
    statuses = ("complete", "unresolved", "interface_failure", "not_applicable")
    result = {"schema": "diagnosis-comparison-summary-v1", "statistical_unit": "scene_family",
              "scene_count": len(cases), "primary_checkpoint": "B", "checkpoints": {}, "costs": {},
              "uncertainty": {"method": "paired_scene_percentile_bootstrap", "seed": 20260914,
                              "replicates": 2000, "interval": 0.95, "p_value_filtering": False},
              "claim_boundary": "diagnosis comparison on the sampled scene families, not SOTA or persistent repair"}
    for point in ("0.5B", "B"):
        methods = {}
        for method in METHODS:
            scene_rows = []
            task_counts = {task: dict.fromkeys(statuses, 0) for task in TASKS}
            holdout_tasks = {task: [] for task in TASKS}
            for case in cases:
                checkpoint = case["comparison"]["methods"][method]["checkpoints"][point]
                complete, applicable = 0, 0
                for task in TASKS:
                    status = checkpoint["diagnoses"][task]["status"]
                    if status not in statuses:
                        raise ValueError("unknown diagnosis status: " + status)
                    # Applicability is the same public task, not a method-specific escape hatch.
                    na = {case["comparison"]["methods"][m]["checkpoints"][point]["diagnoses"][task]["status"]
                          == "not_applicable" for m in METHODS}
                    if len(na) != 1:
                        raise ValueError("task applicability differs between methods")
                    task_counts[task][status] += 1
                    applicable += status != "not_applicable"
                    complete += status == "complete"
                    validation = checkpoint["holdout_validation"][task]
                    if validation["status"] == "not_applicable":
                        continue
                    rows = validation["conditions"]
                    evaluable = [r for r in rows if isinstance(r.get("actual_correct"), bool)]
                    predicted = [r for r in evaluable if r["prediction"]["status"] == "predicted"]
                    correct = sum(r["prediction_correct"] is True for r in evaluable)
                    holdout_tasks[task].append({
                        "cluster_id": case["cluster_id"], "condition_count": len(rows),
                        "evaluable_count": len(evaluable), "interface_failure_count": len(rows) - len(evaluable),
                        "predicted_count": len(predicted), "correct_count": correct,
                        "accuracy": correct / len(evaluable) if evaluable else None,
                        "coverage": len(predicted) / len(evaluable) if evaluable else None,
                        "conditional_accuracy": correct / len(predicted) if predicted else None,
                        "discriminating": validation["discriminating_pair"],
                        "all_correct_and_discriminating": (validation["all_predictions_correct"]
                                                           and validation["discriminating_pair"]),
                    })
                scene_rows.append({"cluster_id": case["cluster_id"], "complete_tasks": complete,
                                   "applicable_tasks": applicable,
                                   "completion_fraction": complete / applicable if applicable else None,
                                   "available_evidence_seconds": checkpoint.get("available_evidence_seconds")})
            fractions = [r["completion_fraction"] for r in scene_rows if r["completion_fraction"] is not None]
            holdout = {}
            for task, rows in holdout_tasks.items():
                holdout[task] = {
                    "applicable_scenes": len(rows), "not_applicable_scenes": len(cases) - len(rows),
                    "evaluable_scenes": sum(r["evaluable_count"] > 0 for r in rows),
                    "scene_mean_accuracy": mean(v) if (v := [r["accuracy"] for r in rows if r["accuracy"] is not None]) else None,
                    "scene_mean_coverage": mean(v) if (v := [r["coverage"] for r in rows if r["coverage"] is not None]) else None,
                    "scene_mean_conditional_accuracy": mean(v) if (v := [r["conditional_accuracy"] for r in rows if r["conditional_accuracy"] is not None]) else None,
                    "discriminating_scenes": sum(r["discriminating"] for r in rows),
                    "all_correct_and_discriminating_scenes": sum(r["all_correct_and_discriminating"] for r in rows),
                    "raw_condition_counts": {field: sum(r[field] for r in rows) for field in
                                             ("condition_count", "evaluable_count", "interface_failure_count",
                                              "predicted_count", "correct_count")},
                    "scenes": rows,
                }
            all_holdout_scenes = []
            for identifier in identifiers:
                rows = [r for values in holdout_tasks.values() for r in values if r["cluster_id"] == identifier]
                total = sum(r["evaluable_count"] for r in rows)
                if total:
                    all_holdout_scenes.append({"cluster_id": identifier,
                                               "accuracy": sum(r["correct_count"] for r in rows) / total,
                                               "coverage": sum(r["predicted_count"] for r in rows) / total})
            methods[method] = {"scene_count": len(cases), "applicable_scene_count": len(fractions),
                               "mean_completion_fraction": mean(fractions) if fractions else None,
                               "task_status_counts": task_counts, "scenes": scene_rows,
                               "holdout_by_task": holdout,
                               "holdout": {"evaluable_scene_count": len(all_holdout_scenes),
                                   "scene_mean_accuracy": mean([r["accuracy"] for r in all_holdout_scenes]) if all_holdout_scenes else None,
                                   "scene_mean_coverage": mean([r["coverage"] for r in all_holdout_scenes]) if all_holdout_scenes else None,
                                   "accuracy_denominator": "all evaluable held-out conditions within each scene; abstentions are not correct",
                                   "scenes": all_holdout_scenes}}
        differences = [g["completion_fraction"] - r["completion_fraction"]
                       for g, r in zip(methods["G"]["scenes"], methods["R"]["scenes"])
                       if g["completion_fraction"] is not None and r["completion_fraction"] is not None]
        interval = None
        if differences:
            rng = Random(20260914)
            samples = [mean(rng.choices(differences, k=len(differences))) for _ in range(2000)]
            bounds = quantiles(samples, n=40, method="inclusive")
            interval = [bounds[0], bounds[-1]]
        result["checkpoints"][point] = {"methods": methods, "G_minus_R": {
            "paired_scene_count": len(differences), "mean_difference": mean(differences) if differences else None,
            "bootstrap_95_percent_interval": interval, "paired_scene_differences": differences,
            "single_scene_interval_is_degenerate": len(differences) == 1}}
    fields = ("logical_seconds", "overrun_seconds", "validation_seconds",
              "formation_plus_validation_seconds", "decision_seconds",
              'main_linked_validation_seconds', 'main_formation_plus_validation_seconds')
    for method in METHODS:
        rows = [case["comparison"]["methods"][method] for case in cases]
        result["costs"][method] = {field: {"recorded_scenes": len(values), "sum": sum(values),
                                              "mean": mean(values) if values else None}
                                    for field in fields
                                    for values in [[r[field] for r in rows if _finite(r.get(field))]]}
    result["cost_scope"] = ("Each method's full simulated independent run plus common validation; decision_seconds is CPU. "
                            "Do not sum methods into physical GPU time. Checkpoint available_evidence_seconds excludes "
                            "an unfinished crossing request and is not a standalone deployment runtime.")
    _add_linked_summary(result, cases)
    return result


def _add_linked_summary(result, cases):
    """The main endpoint requires a diagnosis AND its own discriminating new prediction."""
    result['primary_endpoint'] = 'scene_mean_linked_supported_fraction_at_B'
    result['generic_holdout_role'] = 'secondary answer prediction; never substitutes for diagnosis validation'
    for point in ('0.5B', 'B'):
        main = result['checkpoints'][point]
        paired = {}
        for method in METHODS:
            rows, counts, reasons = [], {}, {}
            for case in cases:
                checkpoint = case['comparison']['methods'][method]['checkpoints'][point]
                if 'linked_validation' not in checkpoint:
                    continue  # Historical/CPU summary records explicitly lack this new endpoint.
                applicable, supported, registered = 0, 0, 0
                for task in TASKS:
                    original = checkpoint['diagnoses'][task]
                    packet = checkpoint['linked_predictions'][task]
                    validation = checkpoint['linked_validation'][task]
                    applicable += original['status'] != 'not_applicable'
                    if (original['status'] == 'not_applicable') != (validation['status'] == 'not_applicable'):
                        raise ValueError('linked verification changed the original task denominator')
                    valid = (original['status'] == 'complete' and validation['status'] == 'supported'
                             and validation.get('distinguishes_registered_competitors') is True)
                    supported += valid
                    registered += packet['status'] == 'registered'
                    statuses = counts.setdefault(task, dict.fromkeys(('supported', 'refuted', 'unresolved', 'not_applicable'), 0))
                    statuses[validation['status']] += 1
                    if packet.get('reason'):
                        reasons[packet['reason']] = reasons.get(packet['reason'], 0) + 1
                rows.append({'cluster_id': case['cluster_id'], 'applicable_tasks': applicable,
                             'supported_tasks': supported, 'registered_tasks': registered,
                             'fraction': supported / applicable if applicable else None})
            mean_value = [r['fraction'] for r in rows if r['fraction'] is not None]
            main['methods'][method].update(mean_linked_supported_fraction=mean(mean_value) if mean_value else None,
                linked={'status': 'recorded' if len(rows) == len(cases) and cases else 'not_fully_recorded',
                        'scenes': rows, 'task_status_counts': counts, 'registration_reasons': reasons,
                        'scope': 'original relation plus its one prespecified context-extension prediction'})
            paired[method] = {r['cluster_id']: r['fraction'] for r in rows if r['fraction'] is not None}
        differences = [paired['G'][cid] - paired['R'][cid] for cid in paired['G'] if cid in paired['R']]
        interval, all_zero = None, bool(differences) and all(d == 0 for d in differences)
        if len(differences) > 1 and not all(d == differences[0] for d in differences):
            rng = Random(20260914)
            bounds = quantiles([mean(rng.choices(differences, k=len(differences))) for _ in range(2000)],
                               n=40, method='inclusive')
            interval = [bounds[0], bounds[-1]]
        main['G_minus_R_linked'] = {'paired_scene_count': len(differences),
            'mean_difference': mean(differences) if differences else None,
            'bootstrap_95_percent_interval': interval, 'paired_scene_differences': differences,
            'all_paired_differences_zero': all_zero,
            'all_zero_95_upper_bound_on_disagreement_probability':
                1 - 0.05 ** (1 / len(differences)) if all_zero else None,
            'uncertainty_note': 'A constant observed difference has no informative bootstrap interval. '
                'The all-zero bound is a one-sided exact binomial bound under independent identically sampled scene pairs; '
                'it is not a universal VLM equivalence claim.'}
