"""Frozen finite intervention catalogue, blind predictions, independent grading.

Only public observations reach prediction functions. Ground truth is a separate
complete table of actual generations, not a saliency mask or a causal root label.
"""
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from math import isfinite
import random

from .report import normalize_text
from .visual_probe_protocol import condition_key, main_maps, region

SCHEMA = "diagnosis-truth-public-v1"
REFUSAL = "Unable to answer."
METHODS = ("G", "answer-map", "nearest", "constant", "atp", "purmm", "cleansight", "random", "marker")
HISTORICAL = "historical_answer_only"


def packet_sha256(packet):
    return sha256(json.dumps(packet, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()


fingerprint = packet_sha256


def catalogue(include_peer=True):
    """Unique frozen geometries; peer operations never become public observations."""
    actions = {}

    def add(indices, tag, known=False, donor="same"):
        indices = sorted(set(indices))
        key = condition_key("unit", donor, indices)
        row = actions.setdefault(key, {"id": key, "indices": indices, "donor": donor,
                                       "known": False, "tags": []})
        row["known"] |= known
        if tag not in row["tags"]:
            row["tags"].append(tag)

    for mapping in main_maps("unit"):
        add(mapping["background"], "map_baseline", True)
        for cell in mapping["cells"]:
            add(mapping["background"] + cell["indices"], f"map_{mapping['resolution']}", True)
    for r in range(6):
        for c in range(6):
            cell = r * 6 + c
            if c < 5:
                add(region(6, cell) + region(6, cell + 1), "adjacent_pair")
            if r < 5:
                add(region(6, cell) + region(6, cell + 6), "adjacent_pair")
            if r < 5 and c < 5:
                add(sum((region(6, cell + d) for d in (0, 1, 6, 7)), []), "fine_square")
    for subset in range(16):
        # Full copy was already measured by the instrument acceptance checks.
        add(sum((region(2, q) for q in range(4) if subset & (1 << q)), []), "coarse_subset", subset == 15)
    for q in range(4):
        coarse = set(region(2, q))
        for cell in range(36):
            fine = set(region(6, cell))
            if fine <= coarse:
                add(coarse - fine, "coarse_minus_cell")
    for parent in (0, 5, 30, 35):
        row, col = divmod(parent, 6)
        for dr in (0, 1):
            for dc in (0, 1):
                add(region(12, (2 * row + dr) * 12 + 2 * col + dc), "local_refinement")
    if include_peer:
        for action in list(actions.values()):
            if action["indices"]:
                for tag in action["tags"]:
                    add(action["indices"], tag, donor="peer")
    return list(actions.values())


def append_selector_actions(actions, rankings):
    """Only native masks and prespecified equal-area random controls may be added."""
    by_id = {a["id"]: deepcopy(a) for a in actions}
    rng = random.Random(20260915)
    for method in ("purmm", "cleansight"):
        if method not in rankings:
            continue
        mask = rankings[method]
        if (not isinstance(mask, list) or any(type(i) is not int or not 0 <= i < 576 for i in mask)
                or mask != sorted(set(mask))):
            raise ValueError("external selector needs its actual sorted visual-state mask")
        for kind, indices in (("native", mask), ("area_control", sorted(rng.sample(range(576), len(mask))))):
            key = condition_key("unit", "same", indices)
            action = by_id.setdefault(key, {"id": key, "indices": indices, "donor": "same",
                                            "known": False, "tags": []})
            tag = {"kind": "selector_" + kind, "method": method}
            if tag not in action["tags"]:
                action["tags"].append(tag)
    return list(by_id.values())


def _historical_catalogue():
    result = []
    for subset in range(16):
        indices = sorted(sum((region(2, q) for q in range(4) if subset & (1 << q)), []))
        result.append({"id": condition_key("unit", "same", indices), "indices": indices,
                       "donor": "same", "known": subset in (0, 1, 2, 4, 8), "tags": ["historical_2x2"]})
    return result


def _row(row, candidates, *, historical=False):
    output = row.get("output", {})
    if (output.get("stop_reason") != "eos" or not isinstance(output.get("text"), str)
            or not normalize_text(output["text"])):
        raise ValueError("every measurement needs a nonempty actual EOS-ended answer")
    scores = row.get("scores", {})
    if historical:
        if scores != {}:
            raise ValueError("historical answer-only replay must not pretend to have new candidate scores")
        return
    if (set(scores) != set(candidates) or any(type(v) not in (int, float) or not isfinite(v)
                                             for v in scores.values())):
        raise ValueError("every measurement needs all finite candidate scores")


def validate_packet(packet):
    if packet.get("schema") != SCHEMA or packet.get("input_condition") not in ("abnormal", "clean"):
        raise ValueError("declare public schema and abnormal/clean input condition")
    if any(not isinstance(packet.get(k), str) or not packet[k].strip()
           for k in ("unit_id", "cluster_id", "truth_answer")):
        raise ValueError("unit, scene and independently sourced truth answer are required")
    mode = packet.get("mode", "confirmation")
    if mode not in ("confirmation", HISTORICAL):
        raise ValueError("unsupported evidence mode")
    candidates = packet.get("candidates", [])
    if (not isinstance(candidates, list) or len(candidates) < 2
            or any(not isinstance(c, str) or not normalize_text(c) for c in candidates)
            or len(set(map(normalize_text, candidates))) != len(candidates)
            or normalize_text(REFUSAL) not in set(map(normalize_text, candidates))
            or normalize_text(packet["truth_answer"]) not in set(map(normalize_text, candidates))):
        raise ValueError("declare unique complete candidates, fact truth and fixed refusal")
    donor = packet.get("donor_answer")
    if donor is not None and (not isinstance(donor, str) or normalize_text(donor) not in
                             set(map(normalize_text, candidates))):
        raise ValueError("peer answer must be independently labelled and in the candidates")
    if mode == HISTORICAL:
        if donor is not None or packet.get("rankings"):
            raise ValueError("historical replay has no complete donor or selector truth")
        expected = {a["id"]: a for a in _historical_catalogue()}
    else:
        # Native selector masks are separately measured references. Publishing
        # them here would give other methods an uncharged external proposal.
        expected = {a["id"]: a for a in catalogue(include_peer=donor is not None)}
    actions = packet.get("actions", [])
    if (not isinstance(actions, list) or len(actions) != len(expected)
            or len({a["id"] for a in actions}) != len(actions)):
        raise ValueError("action catalogue has missing or duplicate conditions")
    for action in actions:
        if action["id"] not in expected or any(action.get(k) != expected[action["id"]][k]
                                              for k in ("indices", "donor", "known")):
            raise ValueError("action geometry/identity differs from frozen finite catalogue")
    known = {a["id"] for a in actions if a["known"]}
    if set(packet.get("observations", {})) != known:
        raise ValueError("public observations must contain exactly known actions, no hidden outcomes")
    for row in packet["observations"].values():
        _row(row, candidates, historical=mode == HISTORICAL)
    return {a["id"]: a for a in actions}


def _base_id():
    return condition_key("unit", "same", [])


def _reference(action):
    """A coarse-minus-cell check tests removal from the measured coarse state."""
    if action["donor"] == "same":
        mask = set(action["indices"])
        for q in range(4):
            coarse = set(region(2, q))
            if mask < coarse and any(coarse - mask == set(region(6, i)) for i in range(36)):
                return condition_key("unit", "same", sorted(coarse))
    return _base_id()


def _relation(before, after, truth):
    if after is None:
        return None
    before, after, truth = map(normalize_text, (before, after, truth))
    if before == after:
        return "no_answer_change"
    if after == truth:
        return "supports_correct_fact"
    if before == truth:
        return "disrupts_correct_fact"
    return "changes_wrong_answer"


def _joint_only(action, observations, answer):
    if action["donor"] != "same" or answer is None:
        return None
    cells = [i for i in range(36) if set(region(6, i)) <= set(action["indices"])]
    if len(cells) != 2 or len(action["indices"]) != 32:
        return None
    original = normalize_text(observations[_base_id()]["output"]["text"])
    unchanged = all(normalize_text(observations[condition_key("unit", "same", region(6, i))]
                                    ["output"]["text"]) == original for i in cells)
    return unchanged and normalize_text(answer) != original


def _public_vectors(packet, method):
    observations = packet["observations"]
    if method == "G":
        refusal = next(c for c in packet["candidates"] if normalize_text(c) == normalize_text(REFUSAL))
        return {key: {candidate: value - row["scores"][refusal] for candidate, value in row["scores"].items()}
                for key, row in observations.items()}
    labels = list(dict.fromkeys(row["output"]["text"] for row in observations.values()))
    # The answer-only map treats equivalent punctuation/case as one actual answer.
    labels = list({normalize_text(label): label for label in labels}.values())
    return {key: {label: float(normalize_text(row["output"]["text"]) == normalize_text(label))
                  for label in labels} for key, row in observations.items()}


def _estimate(action, vectors):
    """Testable additive hypothesis. No area extrapolation to partial fine cells."""
    if action["donor"] != "same":
        return None, 0.0
    mask = set(action["indices"])
    background = set(region(2, 0)) if set(region(2, 0)) <= mask else set()
    added = mask - background
    base = vectors[condition_key("unit", "same", sorted(background))]
    residual = 0.0
    for q in range(4):
        coarse = set(region(2, q)) - background
        if not coarse or not coarse & added:
            continue
        observed = vectors[condition_key("unit", "same", sorted(background | set(region(2, q))))]
        coarse_cells = [i for i in range(36) if set(region(6, i)) <= coarse]
        for label in base:
            additive = base[label] + sum(
                vectors[condition_key("unit", "same", sorted(background | set(region(6, i))))][label]
                - base[label] for i in coarse_cells)
            # A ranking hint only: never added to the answer-score prediction.
            residual = max(residual, abs(observed[label] - additive))
    cells = [i for i in range(36) if set(region(6, i)) <= added]
    if set(sum((region(6, i) for i in cells), [])) != added:
        return None, residual
    estimate = dict(base)
    for cell in cells:
        value = vectors[condition_key("unit", "same", sorted(background | set(region(6, cell))))]
        for label in estimate:
            estimate[label] += value[label] - base[label]
    return estimate, residual


def predictions(packet, method):
    actions = validate_packet(packet)
    if method not in METHODS:
        raise ValueError("unknown diagnosis method")
    historical = packet.get("mode") == HISTORICAL
    if historical and method not in ("nearest", "constant", "random", "marker"):
        raise ValueError("historical answer-only replay cannot run score maps or missing selectors")
    hidden = [a for a in actions.values() if not a["known"]]
    base = packet["observations"][_base_id()]
    original = base["output"]["text"]
    results, priorities = {}, {}
    vectors = _public_vectors(packet, method) if method in ("G", "answer-map") else None
    rankings = packet.get("rankings", {})
    if method == "atp":
        atp = rankings.get("atp_visual")
        if (not isinstance(atp, list) or len(atp) != 576 or
                any(type(x) not in (int, float) or not isfinite(x) for x in atp)):
            raise ValueError("AtP needs 576 actual finite signed token effects; no area extrapolation")
        candidate_atp = rankings.get("candidate_visual_scores", {})
        if (not isinstance(candidate_atp, dict) or candidate_atp and set(candidate_atp) != set(packet["candidates"])
                or any(not isinstance(values, list) or len(values) != 576 or
                       any(type(x) not in (int, float) or not isfinite(x) for x in values)
                       for values in candidate_atp.values())):
            raise ValueError("candidate AtP requires every candidate-versus-refusal 576-token effect")
    if method in ("purmm", "cleansight"):
        selected = rankings.get(method)
        if (not isinstance(selected, list) or selected != sorted(set(selected)) or
                any(type(i) is not int or not 0 <= i < 576 for i in selected)):
            raise ValueError("external selector needs its actual sorted visual-state mask")
    rng = random.Random(int(sha256((packet["unit_id"] + method).encode()).hexdigest()[:16], 16))
    marker = {r * 24 + c for r in range(5) for c in range(5)}
    for action in hidden:
        key, mask = action["id"], set(action["indices"])
        answer, magnitude, residual = None, 0.0, 0.0
        if method in ("G", "answer-map"):
            estimate, residual = _estimate(action, vectors)
            if estimate is not None:
                answer = max(estimate, key=estimate.get)
                magnitude = max(abs(estimate[k] - vectors[_base_id()][k]) for k in estimate)
        elif method == "nearest":
            known = [a for a in actions.values() if a["known"] and a["donor"] == action["donor"]]
            if known:
                nearest = min(known, key=lambda a: (-len(mask & set(a["indices"])) /
                                                   max(1, len(mask | set(a["indices"]))),
                                                   len(a["indices"]), a["id"]))
                answer = packet["observations"][nearest["id"]]["output"]["text"]
        elif method == "constant":
            answer = original
        if method in ("G", "answer-map", "nearest", "constant"):
            results[key] = answer
            # Frozen diagnostic rule: resolve known coarse/fine disagreement first,
            # even when a partial cell has no justified answer extrapolation.
            priority = (residual, int(answer is not None and normalize_text(answer) != normalize_text(original)),
                        magnitude)
        elif method == "atp":
            priority = (max(abs(sum(values[i] for i in mask))
                            for values in (candidate_atp.values() if candidate_atp else [atp])),)
        elif method in ("purmm", "cleansight"):
            priority = (len(mask & set(selected)) / max(1, len(mask)),)
        elif method == "marker":
            priority = (len(mask & marker) / max(1, len(mask)),)
        else:
            priority = (rng.random(),)
        priorities[key] = {"priority": list(priority), "area": len(mask), "donor": action["donor"],
                           "coarse_residual_hint": residual, "predicted_score_change": magnitude}
    order = sorted(priorities, key=lambda key: (tuple(-v for v in priorities[key]["priority"]),
                                               priorities[key]["area"], key))
    claims = {key: _relation(packet["observations"][_reference(actions[key])]["output"]["text"],
                            answer, packet["truth_answer"]) for key, answer in results.items()}
    return {"unit_id": packet["unit_id"], "method": method, "packet_sha256": packet_sha256(packet),
            "predictions": results, "order": order, "order_details": priorities,
            "relational_claims": claims,
            "historical": historical,
            "abstention_scope": "score/answer maps do not extrapolate to peer donors or partial 6x6 cells",
            "ordering_rule": "coarse residual first, predicted answer change next, score magnitude last; small area breaks ties",
            "claim_boundary": "finite state-intervention diagnosis; no global poisoning root or parameter repair"}


def _state(text, truth):
    normalized = normalize_text(text)
    if normalized == normalize_text(truth):
        return "CORRECT"
    return "REFUSAL" if normalized == normalize_text(REFUSAL) else "OTHER"


def _metrics(rows):
    covered = [r for r in rows if r["prediction"] is not None]
    positive = [r for r in rows if r["changed"]]
    negative = [r for r in rows if not r["changed"]]
    detected = [r for r in rows if r["predicted_changed"] is True]
    counts = Counter(r["state"] for r in rows)
    recalls = {state: sum(r["predicted_state"] == state for r in rows if r["state"] == state) / count
               if covered else None for state, count in counts.items()}
    return {"count": len(rows), "predicted": len(covered),
            "coverage": len(covered) / len(rows) if rows else None,
            "exact_accuracy": sum(r["exact"] for r in covered) / len(covered) if covered else None,
            "exact_hit_rate": sum(r["exact"] for r in covered) / len(rows) if covered and rows else None,
            "state_macro_recall": sum(recalls.values()) / len(recalls) if covered and recalls else None,
            "state_recall": recalls,
            "changed_recall": sum(r["predicted_changed"] is True for r in positive) / len(positive)
                              if covered and positive else None,
            "changed_fpr": sum(r["predicted_changed"] is True for r in negative) / len(negative)
                           if covered and negative else None,
            "changed_precision": sum(r["changed"] for r in detected) / len(detected) if detected else None,
            "changed_specificity": sum(r["predicted_changed"] is False for r in negative) / len(negative)
                                   if covered and negative else None,
            "state_counts": dict(counts), "changed_count": len(positive)}


def _discovery(rows, order, *, relation=None):
    by_id = {r["id"]: r for r in rows}
    local_order = [key for key in order if key in by_id]
    transitions = {(normalize_text(r["reference_answer"]), r["normalized_answer"], r["donor"])
                   for r in rows if r["relation"] != "no_answer_change"
                   and (relation is None or r["relation"] == relation)}
    result = {}
    for k in (1, 4, 8, 16):
        checked = local_order[:k]
        found = {(normalize_text(by_id[key]["reference_answer"]), by_id[key]["normalized_answer"], by_id[key]["donor"])
                 for key in checked if by_id[key]["relation"] != "no_answer_change"
                 and (relation is None or by_id[key]["relation"] == relation)}
        result[str(k)] = {"checked": len(checked), "unique_transitions": len(found),
                          "available_transitions": len(transitions),
                          "visual_states_replaced": sum(by_id[key]["area"] for key in checked),
                          "recall": len(found) / len(transitions) if transitions else None}
    return result


def _relation_metrics(rows):
    actual = Counter(r["relation"] for r in rows)
    predicted = Counter(r["predicted_relation"] for r in rows if r["predicted_relation"] is not None)
    matched = Counter(r["relation"] for r in rows if r["relation"] == r["predicted_relation"])
    recall = {label: matched[label] / count if predicted else None for label, count in actual.items()}
    precision = {label: matched[label] / count for label, count in predicted.items()}
    return {"count": len(rows), "predicted": sum(predicted.values()), "counts": dict(actual),
            "precision": sum(matched.values()) / sum(predicted.values()) if predicted else None,
            "macro_recall": sum(recall.values()) / len(recall) if predicted and recall else None,
            "macro_precision": sum(precision.values()) / len(precision) if precision else None,
            "class_recall": recall, "class_precision": precision,
            "joint_only_relative_to_singles": sum(r["joint_only_relative_to_singles"] is True for r in rows)}


def evaluate(packet, oracle, submission):
    """Grade only after predictions are sealed; reject incomplete/failed truth."""
    actions = validate_packet(packet)
    if oracle.get("unit_id") != packet["unit_id"] or set(oracle.get("rows", {})) != set(actions):
        raise ValueError("oracle must cover every unique action for this unit exactly")
    if (oracle.get("packet_sha256", packet_sha256(packet)) != packet_sha256(packet)
            or "identity_sha256" in packet and oracle.get("identity_sha256") != packet["identity_sha256"]):
        raise ValueError("oracle belongs to another public packet or model identity")
    for row in oracle["rows"].values():
        _row(row, packet["candidates"], historical=packet.get("mode") == HISTORICAL)
    for key, row in packet["observations"].items():
        truth = oracle["rows"][key]
        if row["output"] != truth["output"] or row["scores"] != truth["scores"]:
            raise ValueError("public evidence differs from the oracle's same operation")
    if (submission.get("unit_id") != packet["unit_id"] or
            submission.get("packet_sha256") != packet_sha256(packet)):
        raise ValueError("submission must be bound to this exact public packet")
    hidden = {key for key, action in actions.items() if not action["known"]}
    predicted, order = submission.get("predictions", {}), submission.get("order", [])
    if (not isinstance(predicted, dict) or not set(predicted) <= hidden or
            any(value is not None and (not isinstance(value, str) or not normalize_text(value))
                for value in predicted.values())):
        raise ValueError("predictions must be hidden action answers or explicit abstentions")
    if not isinstance(order, list) or len(order) != len(set(order)) or not set(order) <= hidden:
        raise ValueError("checking order must contain unique hidden actions only")
    claims = submission.get("relational_claims", {})
    if not isinstance(claims, dict) or not set(claims) <= set(predicted):
        raise ValueError("relational claims must be derived from sealed hidden answer predictions")
    original = normalize_text(packet["observations"][_base_id()]["output"]["text"])
    rows = []
    for key in sorted(hidden):
        action = actions[key]
        answer, prediction = oracle["rows"][key]["output"]["text"], predicted.get(key)
        normalized = normalize_text(answer)
        state = _state(answer, packet["truth_answer"])
        reference_id = _reference(action)
        reference_answer = packet["observations"][reference_id]["output"]["text"]
        relation = _relation(reference_answer, answer, packet["truth_answer"])
        predicted_relation = _relation(reference_answer, prediction, packet["truth_answer"])
        if key in claims and claims[key] != predicted_relation:
            raise ValueError("relational claim differs from the frozen answer and public reference")
        source = ("receiver" if state == "CORRECT" else "refusal" if state == "REFUSAL" else
                  "donor" if action["donor"] == "peer" and packet.get("donor_answer") is not None and
                  normalized == normalize_text(packet["donor_answer"]) else "other")
        rows.append({"id": key, "donor": action["donor"], "area": len(action["indices"]),
                     "answer": answer, "normalized_answer": normalized, "state": state, "source": source,
                     "reference_id": reference_id, "reference_answer": reference_answer,
                     "relation": relation, "predicted_relation": predicted_relation,
                     "joint_only_relative_to_singles": _joint_only(action, packet["observations"], answer),
                     "predicted_joint_only": _joint_only(action, packet["observations"], prediction),
                     "donor_answer_match": action["donor"] == "peer" and packet.get("donor_answer") is not None
                                           and normalized == normalize_text(packet["donor_answer"]),
                     "changed": normalized != original, "prediction": prediction,
                     "predicted_state": _state(prediction, packet["truth_answer"]) if prediction is not None else None,
                     "predicted_changed": normalize_text(prediction) != original if prediction is not None else None,
                     "exact": normalize_text(prediction) == normalized if prediction is not None else False})
    donors = sorted({r["donor"] for r in rows})
    areas = sorted({r["area"] for r in rows})
    return {"unit_id": packet["unit_id"], "cluster_id": packet["cluster_id"],
            "input_condition": packet["input_condition"], "method": submission.get("method"),
            "historical": packet.get("mode") == HISTORICAL,
            "packet_sha256": packet_sha256(packet), "all_action_coverage": True,
            **_metrics(rows), "discovery_at": _discovery(rows, order),
            "relational_metrics": _relation_metrics(rows),
            "relations_by_donor": {d: _relation_metrics([r for r in rows if r["donor"] == d]) for d in donors},
            "discovery_by_donor": {d: _discovery([r for r in rows if r["donor"] == d], order)
                                   for d in donors},
            "discovery_by_relation": {label: _discovery(rows, order, relation=label)
                                      for label in sorted({r["relation"] for r in rows})},
            "discovery_by_area": {str(a): _discovery([r for r in rows if r["area"] == a], order)
                                  for a in areas},
            "discovery_by_donor_area": {f"{d}:{a}": _discovery(
                [r for r in rows if r["donor"] == d and r["area"] == a], order)
                for d in donors for a in areas if any(r["donor"] == d and r["area"] == a for r in rows)},
            "by_donor": {donor: _metrics([r for r in rows if r["donor"] == donor])
                         for donor in donors},
            "effect_by_area": {str(area): _metrics([r for r in rows if r["area"] == area])
                               for area in areas},
            "abstention_counts": {"peer": sum(r["prediction"] is None and r["donor"] == "peer" for r in rows),
                                   "same": sum(r["prediction"] is None and r["donor"] == "same" for r in rows)},
            "rows": rows}
