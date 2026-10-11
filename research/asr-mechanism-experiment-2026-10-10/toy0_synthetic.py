"""Finite synthetic interventions, not evidence about Qwen or MAttr.

All eight scalar units are movable endpoint deltas. Path scoring is only a
proposal; exhaustive graft/rollback outputs supply the independent truth.
"""
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

from check_theory_counterexample import compute as old_counterexample


UNITS = ("detour_x", "target_y", "joint_u", "joint_v", "tail_a", "tail_b",
         "clean_harm", "clean_restore")
CASES = ("linear", "joint", "tail_and_protection", "legacy_embedded")
CONDITIONS = ("trigger", "clean", "near")
A = (0.0,) * 8
B = (0.001, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
METHODS = ("endpoint", "straight_ig", "old_path_credit",
           "old_path_consistency", "fixed_delta_temporal")


def logits_and_jacobian(case, point, condition):
    """Three response tokens: gold, target, irrelevant tail; exact derivatives."""
    x, y, u, v, ta, tb, harm, restore = point
    derivative = [0.0] * 8
    if case == "joint":
        signal = 2 * u * v
        derivative[2], derivative[3] = 2 * v, 2 * u
    elif case == "legacy_embedded":
        signal = (1 + x) * y
        derivative[0], derivative[1] = y, 1 + x
    else:
        signal, derivative[1] = y, 1.0
    if condition == "trigger":
        target = signal - 0.5
    else:
        scale = 0.2 if condition == "near" else 0.0
        target = -2 + scale * signal + 3 * harm - 3 * restore
        derivative = [scale * value for value in derivative]
        derivative[6], derivative[7] = 3.0, -3.0
    tail, tail_derivative = -4.0, [0.0] * 8
    if case == "tail_and_protection":
        # On all registered paths and endpoint subsets this never wins argmax.
        tail = -4 + 1.9 * ta + 1.9 * tb
        tail_derivative[4] = tail_derivative[5] = 1.9
    return [0.0, target, tail], [[0.0] * 8, derivative, tail_derivative]


def log_probabilities(logits):
    peak = max(logits)
    normalizer = peak + math.log(sum(math.exp(value - peak) for value in logits))
    return [value - normalizer for value in logits]


def loss_gradient(case, point, direction, proxy):
    if proxy not in ("behavior_margin", "full_kl"):
        raise ValueError("unknown proxy")
    if proxy == "behavior_margin":
        # Directional trigger target-vs-gold margin, NOT a fitted model proxy.
        logits, jacobian = logits_and_jacobian(case, point, "trigger")
        sign = -1 if direction == "graft" else 1
        return sign * logits[1], [sign * value for value in jacobian[1]]
    anchor = B if direction == "graft" else A
    loss, gradient = 0.0, [0.0] * 8
    for condition in CONDITIONS:
        logits, jacobian = logits_and_jacobian(case, point, condition)
        logp = log_probabilities(logits)
        logq = log_probabilities(logits_and_jacobian(case, anchor, condition)[0])
        p, q = [math.exp(value) for value in logp], [math.exp(value) for value in logq]
        loss += sum(qi * (lq - lp) for qi, lq, lp in zip(q, logq, logp)) / 3
        for unit in range(8):
            gradient[unit] += sum((pi - qi) * row[unit]
                                  for pi, qi, row in zip(p, q, jacobian)) / 3
    return loss, gradient


def path(kind):
    if kind == "straight":
        points = [A, B]
    elif kind in ("forward_order", "reverse_order"):
        indices = range(8) if kind == "forward_order" else reversed(range(8))
        points, current = [A], list(A)
        for unit in indices:
            current[unit] = B[unit]
            points.append(tuple(current))
    else:
        points = [A, (10.0,) + A[1:], (10.0,) + B[1:], B]
    return [(index / (len(points) - 1), point) for index, point in enumerate(points)]


def resample(points, subdivisions):
    result = []
    for (start_time, start), (end_time, end) in zip(points, points[1:]):
        for step in range(subdivisions):
            fraction = step / subdivisions
            result.append((start_time + (end_time - start_time) * fraction,
                           tuple(a + fraction * (b - a) for a, b in zip(start, end))))
    return result + [points[-1]]


def scores(case, points, direction, proxy):
    if direction == "rollback":
        points = [(1 - time, point) for time, point in reversed(points)]
    start, end = points[0][1], points[-1][1]
    delta = [b - a for a, b in zip(start, end)]
    gradients = [loss_gradient(case, point, direction, proxy)[1] for _, point in points]
    gradient = gradients[0]
    endpoint = [-change * grad for change, grad in zip(delta, gradient)]
    credit, absolute, temporal = [0.0] * 8, [0.0] * 8, [0.0] * 8
    for index, ((ta, before), (tb, after)) in enumerate(zip(points, points[1:])):
        left_gradient, right_gradient = gradients[index], gradients[index + 1]
        for unit in range(8):
            part = -left_gradient[unit] * (after[unit] - before[unit])
            credit[unit] += part
            absolute[unit] += abs(part)
            temporal[unit] -= delta[unit] * (left_gradient[unit] + right_gradient[unit]) / 2 * (tb - ta)
    count, straight = len(points) - 1, [0.0] * 8
    for index in range(count + 1):
        fraction = index / count
        position = tuple(a + fraction * change for a, change in zip(start, delta))
        gradient = loss_gradient(case, position, direction, proxy)[1]
        for unit in range(8):
            weight = 0.5 if index in (0, count) else 1.0
            straight[unit] -= delta[unit] * gradient[unit] * weight / count
    return {"endpoint": endpoint, "straight_ig": straight,
            "old_path_credit": credit,
            "old_path_consistency": [s * abs(s) / (a + 1e-12)
                                     for s, a in zip(credit, absolute)],
            "fixed_delta_temporal": temporal}


def intervention(case, mask, direction):
    point = tuple((B[i] if mask & (1 << i) else A[i]) if direction == "graft"
                  else (A[i] if mask & (1 << i) else B[i]) for i in range(8))
    outputs, margins = [], []
    for condition in CONDITIONS:
        logits = logits_and_jacobian(case, point, condition)[0]
        outputs.append(max(range(3), key=lambda index: (logits[index], -index)))
        margins.append(logits[1] - logits[0])
    wanted = 1 if direction == "graft" else 0
    return {"mask": mask, "units": [i for i in range(8) if mask & (1 << i)],
            "outputs_trigger_clean_near": outputs, "margins_trigger_clean_near": margins,
            "trigger_goal": outputs[0] == wanted,
            "protected": outputs[1:] == [0, 0],
            "valid": outputs == [wanted, 0, 0]}


def evaluate_ranking(ranking_scores, truth):
    ranking = sorted(range(8), key=lambda unit: (-ranking_scores[unit], unit))
    rows = [truth[sum(1 << unit for unit in ranking[:size])] for size in range(1, 9)]
    return {"ranking": ranking, "first_valid_prefix_size": next(
                (len(row["units"]) for row in rows if row["valid"]), None),
            "prefix_results": [{"size": len(row["units"]), "mask": row["mask"],
                                "trigger_goal": row["trigger_goal"],
                                "protected": row["protected"], "valid": row["valid"]}
                               for row in rows]}


def compute():
    report = {"status": "completed_synthetic_cpu_only", "units": list(UNITS),
              "real_model_forward_calls": 0, "new_training_updates": 0,
              "gpu_experiment_run": False, "torch_imported": False,
              "endpoint_a": A, "endpoint_b": B,
              "tie_rule": "smallest output index; smallest unit ID",
              "temporal_rule": "minus fixed endpoint delta times time-weighted trapezoid gradient on path vertices; no consistency multiplier",
              "resampling_scope": "subdivision points are synthetic only; real-model temporal scoring must use actually saved states",
              "proxy_scope": "behavior_margin uses trigger target-gold only; full_kl averages all three conditions; protection is actual greedy truth",
              "old_counterexample_independent_recheck": old_counterexample(), "cases": {}}
    for case in CASES:
        truths = {direction: [intervention(case, mask, direction) for mask in range(256)]
                  for direction in ("graft", "rollback")}
        minimum = {direction: min(len(row["units"]) for row in rows if row["valid"])
                   for direction, rows in truths.items()}
        result = {"exhaustive_truth": truths, "minimum_valid_sizes": minimum,
                  "all_minimum_sets": {direction: [row["units"] for row in rows
                                                  if row["valid"] and len(row["units"]) == minimum[direction]]
                                       for direction, rows in truths.items()}, "scoring_runs": []}
        result["truth_sha256"] = hashlib.sha256(json.dumps(truths, sort_keys=True).encode()).hexdigest()
        for kind in ("straight", "forward_order", "reverse_order", "detour"):
            for refinement in (1, 4, 32):
                points = resample(path(kind), refinement)
                for direction in truths:
                    for proxy in ("behavior_margin", "full_kl"):
                        values = scores(case, points, direction, proxy)
                        result["scoring_runs"].append({"path": kind, "refinement": refinement,
                            "gradient_intervals": len(points) - 1,
                            "temporal_and_straight_gradient_read_count": len(points), "direction": direction,
                            "proxy": proxy, "methods": {name: {"scores": value,
                                **evaluate_ranking(value, truths[direction])} for name, value in values.items()}})
        # Summary only compares scores with exhaustive actual outputs, never score agreement.
        result["summary"] = {}
        for proxy in ("behavior_margin", "full_kl"):
            result["summary"][proxy] = {method: {"optimal_prefix_runs": sum(
                run["methods"][method]["first_valid_prefix_size"] == minimum[run["direction"]]
                for run in result["scoring_runs"] if run["proxy"] == proxy),
                "runs": sum(run["proxy"] == proxy for run in result["scoring_runs"])}
                for method in METHODS}
        result["resampling_checks"] = []
        for kind in ("straight", "forward_order", "reverse_order", "detour"):
            for direction in truths:
                for proxy in ("behavior_margin", "full_kl"):
                    runs = {row["refinement"]: row for row in result["scoring_runs"]
                            if (row["path"], row["direction"], row["proxy"]) == (kind, direction, proxy)}
                    result["resampling_checks"].append({"path": kind, "direction": direction,
                        "proxy": proxy, "endpoint_truth_unchanged": True,
                        "methods": {method: {
                            "max_abs_score_change_4_to_32": max(abs(a - b) for a, b in zip(
                                runs[4]["methods"][method]["scores"], runs[32]["methods"][method]["scores"])),
                            "ranking_changed_4_to_32": runs[4]["methods"][method]["ranking"] != runs[32]["methods"][method]["ranking"],
                            "first_valid_prefix_sizes_1_4_32": [runs[n]["methods"][method]["first_valid_prefix_size"]
                                                              for n in (1, 4, 32)]}
                            for method in METHODS}})
        report["cases"][case] = result
    report["interpretation"] = "Finite constructed examples test proposal failure modes; neither winner counts nor exhaustive truth establish Qwen performance, mechanism universality, or 10x speedup."
    return report


if __name__ == "__main__":
    start = perf_counter()
    result = compute()
    result["compute_wall_seconds"] = perf_counter() - start
    result["timer_scope"] = "compute including exhaustive truth, scoring and old recheck; excludes serialization and file writing"
    destination = Path(__file__).with_name("toy0_results.json")
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "compute_wall_seconds": result["compute_wall_seconds"],
                      "minimum_valid_sizes": {name: row["minimum_valid_sizes"]
                                              for name, row in result["cases"].items()}}))
