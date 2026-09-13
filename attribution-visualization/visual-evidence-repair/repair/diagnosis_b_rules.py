"""Frozen B choices and paired analysis; no model access or hidden-table selection.

The rule family is an implementation envelope. Its weights, contexts and hypothesis
must be frozen from A before B is measured; these helpers do not invent an A finding.
"""
from math import comb, exp, isfinite, log, sqrt
from statistics import fmean

ARMS = ("graph_rule", "activation_difference", "singleton_patching", "fixed_A", "random")
SWAP_FIELDS = ("receiver_retained", "donor_followed", "other", "unmeasured", "expected_nodes")


def _finite(value):
    return type(value) in (int, float) and isfinite(value)


def margin_subsets(contexts):
    """Unique independent scoring trajectories required by the declared D4 cells."""
    return sorted({s for c in contexts for j in range(4) if not c & (1 << j)
                   for s in (c, c | (1 << j))})


def method_aliases(spec):
    """Declare a duplicate from the frozen algorithm, never from favorable results."""
    return ({"graph_rule": "singleton_patching"}
            if spec["contexts"] == [0] and spec["node_aggregation"] == "mean"
            and spec["weights"] == {"d4": 1., "activation": 0., "position": [0.] * 4}
            and spec["min_score"] is None else {})


def summarize_swaps(record, subsets):
    """Mutually exclusive expected node counts; random averages every selected mask."""
    summary = {stratum: dict.fromkeys(SWAP_FIELDS, 0.0) for stratum in ("changed", "unchanged", "unknown")}
    swaps = {(r["subset"], r["node_id"]): r for r in record.get("donor_swaps", [])}
    if len(swaps) != len(record.get("donor_swaps", [])):
        raise ValueError("duplicate donor swap subset/node")
    nodes = {n["id"]: n for n in record["nodes"]}
    for subset in subsets or [None]:
        for node in nodes.values():
            row = swaps.get((subset, node["id"]), {})
            changed = row.get("fact_changed")
            peer = nodes.get(node.get("donor_peer_id"))
            if changed is None and peer and "clean_row" in node and "clean_row" in peer:
                changed = node["clean_row"]["answer"] != peer["clean_row"]["answer"]
            stratum = "unknown" if changed is None else ("changed" if changed else "unchanged")
            weight = 1 / len(subsets) if subsets else 1.0
            summary[stratum]["expected_nodes"] += weight
            outcome = "unmeasured"
            if row.get("status") == "measured":
                if row.get("counts_as_repair") is not False:
                    raise ValueError("donor swaps must never count as main repair success")
                if type(row.get("matches_receiver")) is bool and type(row.get("matches_donor")) is bool:
                    outcome = ("receiver_retained" if row["matches_receiver"] else
                               "donor_followed" if row["matches_donor"] else "other")
            summary[stratum][outcome] += weight
    return summary


def validate_rule(spec):
    required = {"k", "weights", "node_aggregation", "contexts", "context_aggregation",
                "min_score", "query_budget"}
    if set(spec) != required or type(spec["k"]) is not int or spec["k"] not in (1, 2):
        raise ValueError("rule requires the declared fields and k=1 or 2")
    if spec["node_aggregation"] not in ("mean", "min"):
        raise ValueError("node_aggregation must be mean or min")
    if spec["context_aggregation"] not in ("mean", "min", "max"):
        raise ValueError("context_aggregation must be mean, min or max")
    contexts = spec["contexts"]
    if (not isinstance(contexts, list) or not contexts
            or any(type(c) is not int or not 0 <= c < 15 for c in contexts)
            or len(set(contexts)) != len(contexts)
            or any(all(c & (1 << j) for c in contexts) for j in range(4))):
        raise ValueError("distinct contexts must provide a D4 cell for every group")
    weights = spec["weights"]
    if (set(weights) != {"d4", "activation", "position"}
            or not _finite(weights["d4"]) or weights["d4"] == 0
            or not _finite(weights["activation"])
            or not isinstance(weights["position"], list) or len(weights["position"]) != 4
            or not all(_finite(v) for v in weights["position"])):
        raise ValueError("finite weights and a nonzero D4 weight are required")
    if spec["min_score"] is not None and not _finite(spec["min_score"]):
        raise ValueError("min_score must be finite or null")
    budget = spec["query_budget"]
    if (set(budget) != {"margin_trajectories_per_node"}
            or any(type(v) is not int for v in budget.values())
            or not len(set(margin_subsets(contexts)) | set(margin_subsets([0])))
            <= budget["margin_trajectories_per_node"] <= 16):
        raise ValueError("query budget must cover declared D4 cells and the five singleton-score trajectories")
    return spec


def select_subsets(feature_rows, spec, fixed_subset):
    """Read permitted features only; complete intervention outcomes are not accepted.

    A graph score pools per-target sums across the selected groups. The `min`
    version can prefer groups covering different targets; additivity is a ranking
    proxy to test, not an assumption that real combination effects add up.
    """
    validate_rule(spec)
    k = spec["k"]
    candidates = [s for s in range(1, 15) if s.bit_count() == k]
    if type(fixed_subset) is not int or fixed_subset not in candidates:
        raise ValueError("the A-fixed subset must have the same k")
    allowed = {"id", "role", "d4", "activation_squared_difference", "singleton_d4"}
    if (not feature_rows or any(set(n) != allowed for n in feature_rows)
            or len({n["id"] for n in feature_rows}) != len(feature_rows)
            or any(n["role"] not in ("target", "protection") for n in feature_rows)):
        raise ValueError("selection accepts only unique, pre-outcome node features")
    targets = [n for n in feature_rows if n["role"] == "target"]
    if not targets:
        raise ValueError("an eligible case needs at least one target")
    for node in feature_rows:
        for key in ("d4", "activation_squared_difference", "singleton_d4"):
            if not isinstance(node[key], list) or len(node[key]) != 4:
                raise ValueError("each feature must contain four groups")
        if any(v is not None and (not _finite(v) or v < 0)
               for v in node["activation_squared_difference"]):
            raise ValueError("activation squared differences must be nonnegative and finite")
        if any(v is not None and not _finite(v) for v in node["d4"]):
            raise ValueError("D4 values must be finite or explicitly unavailable")
        if any(v is not None and not _finite(v) for v in node["singleton_d4"]):
            raise ValueError("singleton D4 values must be finite or explicitly unavailable")

    def available(key, nodes):
        return all(all(v is not None for v in n[key]) for n in nodes)

    def pick(scores):
        # Frozen tie rule: the lowest numeric mask, independent of B outcomes.
        return max(candidates, key=lambda s: (scores[s], -s))

    def total(values, subset):
        return sum(v for j, v in enumerate(values) if subset & (1 << j))

    aggregate = fmean if spec["node_aggregation"] == "mean" else min
    normalized = []
    if available("activation_squared_difference", targets):
        for node in targets:
            values = node["activation_squared_difference"]
            denominator = sum(values)
            normalized.append([v / denominator if denominator else 0.0 for v in values])
    activation = ({s: aggregate(total(v, s) for v in normalized) for s in candidates}
                  if normalized else None)
    activation_baseline = ({s: fmean(total(v, s) for v in normalized) for s in candidates}
                           if normalized else None)
    weights = spec["weights"]
    graph = None
    if available("d4", targets) and (activation is not None or weights["activation"] == 0):
        scores = {s: weights["d4"] * aggregate(total(n["d4"], s) for n in targets)
                  + weights["activation"] * (activation[s] if activation is not None else 0.0)
                  + total(weights["position"], s) for s in candidates}
        graph = pick(scores)
        if spec["min_score"] is not None and scores[graph] < spec["min_score"]:
            graph = None
    singleton = None
    if available("singleton_d4", targets):
        scores = {s: fmean(total(n["singleton_d4"], s) for n in targets) for s in candidates}
        singleton = pick(scores)
    return {"graph_rule": graph, "activation_difference": pick(activation_baseline) if activation_baseline else None,
            "singleton_patching": singleton, "fixed_A": fixed_subset, "random": candidates}


def evaluate_case(record, choices):
    """Reveal outcomes after choices are saved; random is the exact equal-k average.

    Explicitly unresolved cells and abstentions do not enter the success numerator.
    The eligible-case denominator is never reduced because an arm failed to select.
    """
    if set(choices) != set(ARMS):
        raise ValueError("all five comparison arms are required")
    nodes = record["nodes"]
    targets = [n for n in nodes if n["role"] == "target"]
    protection = [n for n in nodes if n["role"] == "protection"]
    if not targets or len(targets) + len(protection) != len(nodes):
        raise ValueError("frozen target/protection nodes are required")
    tables, residuals = {}, {}
    for node in nodes:
        table = {r["subset"]: r["correct"] for r in node["interventions"]}
        if (len(table) != len(node["interventions"])
                or any(type(s) is not int or not 0 <= s < 16 for s in table)
                or any(v is not None and type(v) is not bool for v in table.values())):
            raise ValueError("outcomes must be unique subset masks with bool/null correctness")
        tables[node["id"]] = table
        residuals[node["id"]] = {r["subset"]: r.get("remaining_above_replay_noise")
                                  for r in node["interventions"]}
    random = choices["random"]
    if (not isinstance(random, list) or not random
            or any(type(s) is not int or not 0 < s < 15 for s in random)):
        raise ValueError("random must enumerate all equal-size local subsets")
    k = random[0].bit_count()
    if k not in (1, 2) or random != [s for s in range(1, 15) if s.bit_count() == k]:
        raise ValueError("random must be the exact, unique equal-k enumeration")
    arms = {}
    for arm in ARMS:
        choice = choices[arm]
        subsets = random if arm == "random" else ([] if choice is None else [choice])
        if any(type(s) is not int or s not in random for s in subsets):
            raise ValueError("every selected operation must use the same k")
        for subset in subsets:
            if any(subset not in tables[n["id"]] for n in nodes):
                raise ValueError("a selected operation is missing from the sealed outcome table")
        outcomes = [[tables[n["id"]][s] for n in nodes] for s in subsets]
        local_available = all(type(residuals[n["id"]][s]) is bool for n in targets for s in subsets)
        local_success = (fmean(all(tables[n["id"]][s] is True for n in nodes)
                               and all(residuals[n["id"]][s] for n in targets) for s in subsets)
                         if subsets and local_available else (0.0 if not subsets else None))
        arms[arm] = {"success": fmean(all(v is True for v in row) for row in outcomes) if outcomes else 0.0,
                     "nontrivial_local_success": local_success,
                     "locality_status": "measured" if subsets and local_available else ("abstained" if not subsets else "unavailable"),
                     "abstained": not subsets,
                     "unresolved_fraction": fmean(any(v is None for v in row) for row in outcomes) if outcomes else 0.0,
                     "corrected_targets": fmean(sum(tables[n["id"]][s] is True for n in targets)
                                                for s in subsets) if subsets else 0.0,
                     "damaged_protection": fmean(sum(tables[n["id"]][s] is False for n in protection)
                                                  for s in subsets) if subsets else 0.0,
                     "unresolved_protection": fmean(sum(tables[n["id"]][s] is None for n in protection)
                                                     for s in subsets) if subsets else 0.0,
                     "donor_swaps": summarize_swaps(record, subsets),
                     "target_nodes": len(targets), "protection_nodes": len(protection)}
    return {"cluster_id": record["cluster_id"], "choices": choices, "arms": arms}


def _interval(values, alpha, low, high):
    if not values:
        return None
    mean = fmean(values)
    radius = (high - low) * sqrt(log(2 / alpha) / (2 * len(values)))
    return [max(low, mean - radius), min(high, mean + radius)]


def analyze_paired(case_results, alpha=0.05, aliases=None):
    """Finite-sample intervals and Holm across distinct predeclared comparisons.

    Hoeffding bounds assume independent sampled scenes, not random selection of this
    fixed pool. They do not turn one checkpoint/source into a population guarantee.
    """
    if not _finite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if len({c["cluster_id"] for c in case_results}) != len(case_results):
        raise ValueError("base scenes must be unique")
    n = len(case_results)
    aliases = aliases or {}
    if aliases not in ({}, {"graph_rule": "singleton_patching"}):
        raise ValueError("only the declared standard-method alias is supported")
    if aliases and any(c["choices"]["graph_rule"] != c["choices"]["singleton_patching"] for c in case_results):
        raise ValueError("declared identical methods selected different operations")
    baselines = [arm for arm in ARMS[1:] if arm not in aliases.values()]
    arms, comparisons = {}, []
    for arm in ARMS:
        rows = [c["arms"][arm] for c in case_results]
        values = [r["success"] for r in rows]
        target_n, protection_n = sum(r["target_nodes"] for r in rows), sum(r["protection_nodes"] for r in rows)
        local = [r["nontrivial_local_success"] for r in rows if r["nontrivial_local_success"] is not None]
        arms[arm] = {"success_numerator": sum(values), "eligible_cases": n,
                     "success_rate": fmean(values) if n else None,
                     "success_interval": _interval(values, alpha, 0, 1),
                     "selected_cases": sum(not r["abstained"] for r in rows),
                     "abstentions": sum(r["abstained"] for r in rows),
                     "unresolved_expected_cases": sum(r["unresolved_fraction"] for r in rows),
                     "corrected_targets": sum(r["corrected_targets"] for r in rows),
                     "target_nodes": target_n,
                     "nontrivial_local_success_numerator_known": sum(local),
                     "locality_unavailable_cases": n - len(local),
                     "nontrivial_local_success_rate": fmean(local) if n and len(local) == n else None,
                     "damaged_protection": sum(r["damaged_protection"] for r in rows) if protection_n else None,
                     "unresolved_protection": sum(r["unresolved_protection"] for r in rows) if protection_n else None,
                     "donor_swaps": {stratum: {key: sum(r["donor_swaps"][stratum][key] for r in rows)
                                                for key in SWAP_FIELDS}
                                     for stratum in ("changed", "unchanged", "unknown")},
                     "protection_nodes": protection_n,
                     "protection_status": "checked" if protection_n else "not_applicable"}
    for arm in baselines:
        differences = [c["arms"]["graph_rule"]["success"] - c["arms"][arm]["success"] for c in case_results]
        wins, losses = sum(d > 0 for d in differences), sum(d < 0 for d in differences)
        if arm == "random":
            mean = fmean(differences) if n else 0.0
            p = exp(-n * max(0.0, mean) ** 2 / 2) if n else 1.0
            test = "one_sided_paired_Hoeffding_bound"
        else:
            discordant = wins + losses
            p = sum(comb(discordant, j) for j in range(wins, discordant + 1)) / 2 ** discordant
            test = "one_sided_exact_McNemar"
        comparisons.append({"baseline": arm, "cases": n, "delta": fmean(differences) if n else None,
                            "paired_interval": _interval(differences, alpha, -1, 1),
                            "simultaneous_paired_interval": _interval(differences, alpha / len(baselines), -1, 1),
                            "identical_selected_subsets": sum(c["choices"]["graph_rule"] == c["choices"][arm]
                                                               for c in case_results) if arm != "random" else None,
                            "case_wins": wins, "case_losses": losses, "p_value": p, "test": test})
    running = 0.0
    for rank, comparison in enumerate(sorted(comparisons, key=lambda c: c["p_value"])):
        running = max(running, min(1.0, (len(comparisons) - rank) * comparison["p_value"]))
        comparison.update(holm_p_value=running, superiority_rejected_null=running <= alpha)
    return {"eligible_cases": n, "alpha": alpha, "arms": arms, "comparisons": comparisons, "method_aliases": aliases,
            "uncertainty": "Scene-level finite-sample Hoeffding intervals; independent-scene sampling assumption; no zero-width all-equal bootstrap.",
            "equivalence": "Identical selections provide no independent selection increment, even if both arms succeed.",
            "cost": "Report scoring trajectories, free generations, tokens and timed GPUh separately; this analysis does not assert equal cost."}
