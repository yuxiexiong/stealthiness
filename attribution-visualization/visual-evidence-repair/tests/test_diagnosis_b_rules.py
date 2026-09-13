"""Small controlled contracts for frozen B selection; no scientific result fixture."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repair.diagnosis_b_rules import analyze_paired, evaluate_case, margin_subsets, method_aliases, select_subsets, summarize_swaps, validate_rule


def rule():
    return {"k": 2, "weights": {"d4": 1.0, "activation": 0.0, "position": [0.0] * 4},
            "node_aggregation": "min", "contexts": [0], "context_aggregation": "mean",
            "min_score": None, "query_budget": {"margin_trajectories_per_node": 5}}


def features():
    return [{"id": "left", "role": "target", "d4": [5.0, 0.0, 2.0, 0.0],
             "activation_squared_difference": [0.0, 0.0, 1.0, 2.0], "singleton_d4": [1, 0, 0, 0]},
            {"id": "right", "role": "target", "d4": [0.0, 4.0, 2.0, 0.0],
             "activation_squared_difference": [0.0, 0.0, 1.0, 2.0], "singleton_d4": [0, 0, 0, 1]}]


def measured(identity="case", protection=False):
    nodes = [{"id": "left", "role": "target", "interventions": [
        {"subset": s, "correct": s in (3, 15)} for s in range(16)]}]
    if protection:
        nodes.append({"id": "protected", "role": "protection", "interventions": [
            {"subset": s, "correct": s != 9} for s in range(16)]})
    return {"cluster_id": identity, "nodes": nodes}


class FrozenRuleTest(unittest.TestCase):
    def test_standard_rule_alias_and_donor_strata_do_not_change_main_success(self):
        spec = dict(rule(), k=1, node_aggregation="mean")
        rows = features()
        for row in rows:
            row["d4"] = row["singleton_d4"]
        choices = select_subsets(rows, spec, 1)
        record = measured()
        record["nodes"][0]["donor_peer_id"] = "peer-not-in-screen"
        record["donor_swaps"] = [
            {"subset": 1, "node_id": "left", "status": "measured", "fact_changed": True,
             "matches_receiver": False, "matches_donor": True, "counts_as_repair": False},
            {"subset": 2, "node_id": "left", "status": "measured", "fact_changed": False,
             "matches_receiver": True, "matches_donor": True, "counts_as_repair": False}]
        result = evaluate_case(record, choices)
        self.assertEqual(result["arms"]["graph_rule"]["success"], 0)
        swaps = result["arms"]["random"]["donor_swaps"]
        self.assertEqual(swaps["changed"]["donor_followed"], 0.25)
        self.assertEqual(swaps["unchanged"]["receiver_retained"], 0.25)
        self.assertEqual(swaps["unchanged"]["donor_followed"], 0)
        self.assertEqual(swaps["unknown"]["unmeasured"], 0.5)
        self.assertEqual(summarize_swaps(record, [])["unknown"]["unmeasured"], 1)
        aliases = method_aliases(spec)
        self.assertEqual(aliases, {"graph_rule": "singleton_patching"})
        analysis = analyze_paired([result], aliases=aliases)
        self.assertEqual(len(analysis["comparisons"]), 3)
        self.assertNotIn("singleton_patching", [c["baseline"] for c in analysis["comparisons"]])

    def test_preoutcome_complementary_selection_and_budget(self):
        spec, rows = rule(), features()
        self.assertEqual(margin_subsets([0]), [0, 1, 2, 4, 8])
        choices = select_subsets(rows, spec, 5)
        self.assertEqual(choices, {"graph_rule": 3, "activation_difference": 12,
                                  "singleton_patching": 9, "fixed_A": 5,
                                  "random": [3, 5, 6, 9, 10, 12]})
        rows[0]["interventions"] = []
        with self.assertRaises(ValueError):
            select_subsets(rows, spec, 5)  # no full-table leakage into the rule
        for change in ({"contexts": [1]}, {"contexts": [0, 0]},
                       {"query_budget": {"margin_trajectories_per_node": 4}}):
            with self.assertRaises(ValueError):
                validate_rule(dict(spec, **change))

    def test_exact_random_abstention_missing_cells_and_protection(self):
        spec, rows = rule(), features()
        choices = select_subsets(rows, spec, 5)
        result = evaluate_case(measured(protection=True), choices)
        self.assertEqual(result["arms"]["random"]["success"], 1 / 6)
        self.assertEqual(result["arms"]["singleton_patching"]["damaged_protection"], 1)
        self.assertIsNone(result["arms"]["graph_rule"]["nontrivial_local_success"])
        local = measured("local")
        for row in local["nodes"][0]["interventions"]:
            row["remaining_above_replay_noise"] = False
        self.assertEqual(evaluate_case(local, choices)["arms"]["graph_rule"]["nontrivial_local_success"], 0.0)
        spec["min_score"] = 100.0
        abstained = evaluate_case(measured("abstained"), select_subsets(rows, spec, 5))
        report = analyze_paired([result, abstained])
        graph = report["arms"]["graph_rule"]
        self.assertEqual((graph["eligible_cases"], graph["success_rate"], graph["abstentions"]), (2, 0.5, 1))
        self.assertEqual(analyze_paired([abstained])["arms"]["graph_rule"]["protection_status"], "not_applicable")
        incomplete = measured()
        incomplete["nodes"][0]["interventions"].pop(3)
        with self.assertRaises(ValueError):
            evaluate_case(incomplete, choices)

    def test_all_equal_intervals_are_nonzero_and_holm_is_applied(self):
        choices = select_subsets(features(), rule(), 5)
        records = [evaluate_case(measured(str(i)), choices) for i in range(24)]
        report = analyze_paired(records)
        self.assertLess(report["arms"]["graph_rule"]["success_interval"][0], 1.0)
        for comparison in report["comparisons"]:
            self.assertLess(comparison["paired_interval"][0], comparison["paired_interval"][1])
            self.assertGreaterEqual(comparison["holm_p_value"], comparison["p_value"])
        equal = deepcopy(records)
        for record in equal:
            record["arms"]["fixed_A"]["success"] = 1.0
        comparison = next(c for c in analyze_paired(equal)["comparisons"] if c["baseline"] == "fixed_A")
        self.assertEqual(comparison["p_value"], 1.0)
        self.assertLess(comparison["paired_interval"][0], 0.0)
        self.assertGreater(comparison["paired_interval"][1], 0.0)


if __name__ == "__main__":
    unittest.main()
