"""CPU contracts for V3 geometry, prospective registration and fair replay."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repair.visual_probe_protocol import (GRID, additive_effect, compare_orders, condition_key, main_maps,
                                         numeric_order, region, validate_batch)


NODES = {f"{case}-{label}": {"cluster_id": case, "label": label}
         for case in ("a", "b", "c") for label in ("Q1", "Q2", "peer")}


def union(*groups):
    return sorted({i for group in groups for i in group})


def diagnosis(identity="d", case="a"):
    return {"id": identity, "cluster_id": case, "observation": "opposing signed regions",
            "judgment": "the joint operation changes Q1", "alternatives": ["additive", "context-dependent"],
            "prediction": "Q1 changes and Q2 remains", "falsifier": "Q1 remains or Q2 changes",
            "action": "test the registered joint", "source_maps": [f"{case}-Q1:empty:6"]}


def check(identity, cells, kind="key", **extra):
    case = extra.pop("cluster_id", "a")
    row = {"id": identity, "diagnosis_id": "d", "cluster_id": case, "kind": kind,
           "background": [], "indices": union(*(region(6, cell) for cell in cells)),
           "node_ids": [f"{case}-Q1", f"{case}-Q2"], "parent_index": None,
           "parent_check_id": None, "control_for": None,
           "prediction": {f"{case}-Q1": {"correct": True, "effect_sign": 1},
                          f"{case}-Q2": {"correct": True}}}
    row.update(extra)
    return row


def batch(checks, stage="joint", diagnoses=None, comparisons=None):
    return {"stage": stage, "diagnoses": [diagnosis()] if diagnoses is None else diagnoses,
            "checks": checks, "comparisons": comparisons or []}


def comparison(ids):
    return {"diagnosis_id": "d", "node_id": "a-Q1", "direction": "increase",
            "visual_order": ids, "direct_order": list(reversed(ids)),
            "direct_rationale": "fixed position from old coarse observations",
            "stop_when": {i: {"a-Q1": {"correct": True}} for i in ids}}


def known_main():
    return {cell["key"] for node in ("a-Q1", "a-Q2") for graph in main_maps(node)
            for cell in graph["cells"]} | {condition_key(node, node, []) for node in ("a-Q1", "a-Q2")}


class VisualProbeProtocolTest(unittest.TestCase):
    def test_exact_nested_geometry_and_included_cells(self):
        self.assertEqual(GRID, 24)
        for resolution, size in ((2, 144), (6, 16), (12, 4)):
            cells = [region(resolution, i) for i in range(resolution ** 2)]
            self.assertTrue(all(len(cell) == size for cell in cells))
            self.assertEqual(sorted(i for cell in cells for i in cell), list(range(576)))
        self.assertEqual(region(2, 0), union(*(region(6, r * 6 + c) for r in range(3) for c in range(3))))
        self.assertEqual(region(6, 8), union(*(region(12, i) for i in (28, 29, 40, 41))))
        graphs = main_maps("a-Q1")
        top_left = next(m for m in graphs if m["resolution"] == 6 and m["background"])
        self.assertEqual(sum(c["status"] == "included" for c in top_left["cells"]), 9)
        self.assertTrue(all(c["key"] == top_left["base_key"] for c in top_left["cells"] if c["status"] == "included"))
        self.assertEqual(condition_key("a", "a", [1, 2]), condition_key("a", "a", [2, 1]))
        self.assertEqual(condition_key("a", "a", []), condition_key("a", "b", []))
        self.assertNotEqual(condition_key("a", "a", [1]), condition_key("a", "b", [1]))
        for resolution, index in ((3, 0), (6, 36), (6, True)):
            with self.assertRaises(ValueError):
                region(resolution, index)

    def test_joint_freeze_is_independent_and_matches_new_area(self):
        checks = [check("candidate", [0, 1]), check("control", [2, 3], "area_control", control_for="candidate")]
        spec = batch(checks, comparisons=[comparison(["candidate", "control"])])
        saved = deepcopy(spec)
        result = validate_batch(spec, NODES, [], known_main())
        self.assertEqual(spec, saved)
        self.assertFalse(result["checks"][0]["known"])
        result["checks"][0]["indices"].append(575)
        self.assertEqual(spec, saved)
        invalid = deepcopy(spec)
        invalid["checks"][1]["indices"] = union(region(6, 2), region(6, 3), region(6, 4))
        with self.assertRaisesRegex(ValueError, "added area"):
            validate_batch(invalid, NODES, [], known_main())
        invalid = deepcopy(spec)
        invalid["checks"][0]["indices"].reverse()
        with self.assertRaisesRegex(ValueError, "sorted"):
            validate_batch(invalid, NODES, [], set())
        invalid = deepcopy(spec)
        invalid["checks"][1]["prediction"]["a-Q2"] = {"effect_sign": True}
        with self.assertRaisesRegex(ValueError, "effect sign"):
            validate_batch(invalid, NODES, [], set())

    def test_known_total_operation_and_hidden_menu_cannot_be_rebranded(self):
        known = known_main()
        item = check("old", [0], known=True)
        self.assertTrue(validate_batch(batch([item]), NODES, [], known)["checks"][0]["known"])
        item["known"] = False
        with self.assertRaisesRegex(ValueError, "known or duplicated"):
            validate_batch(batch([item]), NODES, [], known)
        # Different (background, added group) pairs can describe the same actual operation.
        first = check("one", [0, 1])
        repeat = check("repeat", [1], background=region(6, 0))
        with self.assertRaisesRegex(ValueError, "known or duplicated"):
            validate_batch(batch([first, repeat]), NODES, [], set())
        item["known"] = True
        with self.assertRaisesRegex(ValueError, "genuinely unmeasured"):
            validate_batch(batch([item], comparisons=[comparison(["old"])]), NODES, [], known)

    def test_controls_and_cumulative_joint_budget(self):
        checks = []
        for i in range(3):
            checks += [check(f"key{i}", [i * 4, i * 4 + 1]),
                       check(f"ctl{i}", [i * 4 + 2, i * 4 + 3], "area_control", control_for=f"key{i}")]
        previous = validate_batch(batch(checks[:4]), NODES, [], set())
        with self.assertRaisesRegex(ValueError, "joint budget"):
            validate_batch(batch(checks[4:], diagnoses=[]), NODES, [previous], set())
        with self.assertRaisesRegex(ValueError, "exactly one"):
            validate_batch(batch([checks[0]]), NODES, [], set())
        bad = deepcopy(checks[:2])
        bad[1]["background"] = region(6, 20)
        with self.assertRaisesRegex(ValueError, "same-batch candidate"):
            validate_batch(batch(bad), NODES, [], set())

    def test_refinement_checks_children_not_the_known_parent(self):
        parent = check("parent", [8], "refine", parent_index=8)
        refined = validate_batch(batch([parent], stage="refine"), NODES, [], known_main())
        predicted = deepcopy(parent)
        predicted["child_predictions"] = {str(i): {"a-Q1": {"effect_sign": 1 if i == 28 else 0},
                                                   "a-Q2": {"correct": True}}
                                          for i in (28, 29, 40, 41)}
        saved = validate_batch(batch([predicted], stage="refine"), NODES, [], known_main())
        self.assertEqual(saved["checks"][0]["child_predictions"], predicted["child_predictions"])
        missing_child = deepcopy(predicted)
        del missing_child["child_predictions"]["41"]
        with self.assertRaisesRegex(ValueError, "four global"):
            validate_batch(batch([missing_child], stage="refine"), NODES, [], known_main())
        del predicted["child_predictions"]["28"]["a-Q2"]
        with self.assertRaisesRegex(ValueError, "both main nodes"):
            validate_batch(batch([predicted], stage="refine"), NODES, [], known_main())
        children = [region(12, i) for i in (28, 29, 40, 41)]
        local = [check("pair", [], "local_joint", parent_index=8, parent_check_id="parent",
                       indices=union(*children[:2])),
                 check("matched", [], "area_control", parent_index=8, parent_check_id="parent",
                       indices=union(*children[2:]), control_for="pair")]
        result = validate_batch(batch(local, stage="local_joint", diagnoses=[]), NODES, [refined], known_main())
        self.assertEqual(len(result["checks"]), 2)
        invalid = deepcopy(parent)
        invalid["background"] = region(6, 8)
        with self.assertRaises(ValueError):
            validate_batch(batch([invalid], stage="refine"), NODES, [], known_main())
        too_many = [check(f"p{i}", [i], "refine", parent_index=i) for i in range(3)]
        with self.assertRaisesRegex(ValueError, "refinement budget"):
            validate_batch(batch(too_many, stage="refine"), NODES, [], known_main())

    def test_numeric_baseline_consumes_all_scores_and_refuses_missing(self):
        candidates = [check("high", [0, 1]), check("low", [2, 3])]
        node = "a-Q1"
        scores = {condition_key(node, node, []): {"fact": -5}}
        for index, effect in enumerate((2, 1, -1, 0)):
            scores[condition_key(node, node, region(6, index))] = {"fact": -5 + effect}
        self.assertEqual(numeric_order(candidates, node, scores, "increase"), ["high", "low"])
        self.assertEqual(numeric_order(candidates, node, scores, "decrease"), ["low", "high"])
        scores.pop(condition_key(node, node, region(6, 3)))
        with self.assertRaisesRegex(ValueError, "missing or nonfinite"):
            numeric_order(candidates, node, scores, "increase")
        local = [check("local", [], "local_joint", parent_check_id="parent", indices=union(region(12, 0), region(12, 1)))]
        for index in (0, 1):
            scores[condition_key(node, node, region(12, index))] = {"fact": -4}
        self.assertEqual(numeric_order(local, node, scores, "increase"), ["local"])
        coarse = {"node_ids": [node], "background": [], "indices": region(2, 0)}
        coarse_scores = {condition_key(node, node, []): {"fact": 0},
                         **{condition_key(node, node, region(6, r * 6 + c)): {"fact": 1}
                            for r in range(3) for c in range(3)}}
        self.assertEqual(additive_effect(coarse, node, coarse_scores), 9)

    def test_frozen_replay_early_stop_cost_unknowns_and_equivalent_orders(self):
        orders = {"visual": ["a", "b", "c"], "numeric": ["a", "b", "c"], "direct": ["c", "b", "a"]}
        costs = {"a": 2.0, "b": 3.0, "c": 7.0}
        result = compare_orders(orders, {"a": None, "b": True, "c": False}, costs)
        self.assertEqual((result["visual"]["checks"], result["visual"]["seconds"]), (2, 5))
        self.assertEqual(result["visual"]["status"], "witness")
        self.assertEqual(result["visual"]["unresolved_checks"], ["a"])
        self.assertEqual(result["direct"]["seconds"], 10)
        self.assertEqual(result["numeric"]["equivalent_to"], "visual")
        unresolved = compare_orders(orders, {"a": None, "b": False, "c": False}, costs)
        self.assertEqual(unresolved["visual"]["status"], "unresolved")
        negative = compare_orders(orders, dict.fromkeys(costs, False), costs)
        self.assertEqual(negative["visual"]["status"], "no_witness_in_menu")
        with self.assertRaisesRegex(ValueError, "same finite menu"):
            compare_orders({"visual": ["a"], "direct": ["b"]}, {"a": True, "b": False}, costs)
        with self.assertRaisesRegex(ValueError, "measured cost"):
            compare_orders(orders, dict.fromkeys(costs, False), dict(costs, a=float("nan")))


if __name__ == "__main__":
    unittest.main()
