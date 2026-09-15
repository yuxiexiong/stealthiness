"""CPU fixtures test the independent grader, not empirical model effectiveness."""
from copy import deepcopy
import unittest

from repair.diagnosis_truth import (
    METHODS, REFUSAL, SCHEMA, HISTORICAL, append_selector_actions, catalogue, evaluate,
    packet_sha256, predictions, validate_packet,
)
from repair.visual_probe_protocol import condition_key, region


def example(peer=False, clean=False):
    actions = catalogue(include_peer=peer)
    candidates = ["red", "green", REFUSAL]
    original = "red" if clean else REFUSAL
    def measurement(answer=original):
        return {"output": {"text": answer, "stop_reason": "eos"},
                "scores": {c: -1.0 if c == answer else -4.0 for c in candidates}}
    rows = {a["id"]: measurement() for a in actions}
    packet = {"schema": SCHEMA, "unit_id": "toy-clean" if clean else "toy-abnormal",
              "cluster_id": "toy-scene", "input_condition": "clean" if clean else "abnormal",
              "truth_answer": "red", "donor_answer": "green" if peer else None,
              "candidates": candidates, "actions": actions,
              "observations": {a["id"]: deepcopy(rows[a["id"]]) for a in actions if a["known"]},
              "rankings": {"atp": [0.0] * 36, "atp_visual": [0.0] * 576,
                           "purmm": [], "cleansight": []}, "maps": []}
    return packet, {"unit_id": packet["unit_id"], "rows": rows}


def hidden_action(packet, tag="adjacent_pair", donor="same"):
    return next(a for a in packet["actions"] if not a["known"] and a["donor"] == donor and tag in a["tags"])


class DiagnosisTruthTests(unittest.TestCase):
    def test_catalogue_geometry_and_aliases(self):
        actions = catalogue()
        self.assertEqual(len(actions), len({a["id"] for a in actions}))
        same = [a for a in actions if a["donor"] == "same"]
        self.assertEqual(sum("adjacent_pair" in a["tags"] for a in same), 60)
        self.assertEqual(sum("fine_square" in a["tags"] for a in same), 25)
        self.assertEqual(sum("coarse_subset" in a["tags"] for a in same), 16)
        self.assertEqual(sum("coarse_minus_cell" in a["tags"] for a in same), 36)
        self.assertEqual(sum("local_refinement" in a["tags"] for a in same), 16)
        for a in actions:
            self.assertEqual(a["indices"], sorted(set(a["indices"])))
            if a["donor"] == "peer":
                self.assertFalse(a["known"])
                self.assertTrue(a["indices"])
        empty = next(a for a in same if not a["indices"])
        self.assertTrue(empty["known"])
        self.assertIn("coarse_subset", empty["tags"])
        self.assertTrue(next(a for a in same if len(a["indices"]) == 576)["known"])
        self.assertEqual(sum(a["known"] for a in same), 72)

    def test_hidden_truth_cannot_change_predictions(self):
        packet, oracle = example(peer=True)
        before = {m: predictions(packet, m) for m in METHODS}
        action = hidden_action(packet)
        oracle["rows"][action["id"]]["output"]["text"] = "green"
        self.assertEqual(before, {m: predictions(packet, m) for m in METHODS})
        leaked = deepcopy(packet)
        leaked["observations"][action["id"]] = oracle["rows"][action["id"]]
        with self.assertRaisesRegex(ValueError, "no hidden"):
            predictions(leaked, "G")

    def test_incomplete_truncated_nonfinite_and_wrong_identity_fail(self):
        packet, oracle = example()
        submitted = predictions(packet, "constant")
        key = hidden_action(packet)["id"]
        for mutation in ("missing", "truncated", "nonfinite"):
            broken = deepcopy(oracle)
            if mutation == "missing":
                del broken["rows"][key]
            elif mutation == "truncated":
                broken["rows"][key]["output"]["stop_reason"] = "max_new_tokens"
            else:
                broken["rows"][key]["scores"]["red"] = float("nan")
            with self.assertRaises(ValueError):
                evaluate(packet, broken, submitted)
        submitted["packet_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "exact public"):
            evaluate(packet, oracle, submitted)
        oracle["packet_sha256"] = "another-evidence-packet"
        with self.assertRaisesRegex(ValueError, "another public packet"):
            evaluate(packet, oracle, predictions(packet, "constant"))
        duplicated = deepcopy(packet)
        duplicated["actions"][-1] = duplicated["actions"][0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_packet(duplicated)

    def test_constant_high_accuracy_is_not_change_detection(self):
        packet, oracle = example()
        key = hidden_action(packet)["id"]
        oracle["rows"][key]["output"]["text"] = "red"
        result = evaluate(packet, oracle, predictions(packet, "constant"))
        self.assertGreater(result["exact_accuracy"], 0.95)
        self.assertEqual(result["changed_recall"], 0.0)
        self.assertEqual(result["state_macro_recall"], 0.5)
        self.assertEqual(result["changed_fpr"], 0.0)
        self.assertEqual(result["changed_specificity"], 1.0)
        self.assertIsNone(result["changed_precision"])
        self.assertEqual(result["state_recall"]["CORRECT"], 0.0)

    def test_joint_score_prediction_and_partial_abstention(self):
        packet, oracle = example()
        for cell in (0, 1):
            key = condition_key("unit", "same", region(6, cell))
            packet["observations"][key]["scores"]["red"] = -2.0
            oracle["rows"][key] = deepcopy(packet["observations"][key])
        joint = condition_key("unit", "same", sorted(region(6, 0) + region(6, 1)))
        oracle["rows"][joint]["output"]["text"] = "red"
        graph, answers = predictions(packet, "G"), predictions(packet, "answer-map")
        self.assertEqual(graph["predictions"][joint], "red")
        self.assertEqual(answers["predictions"][joint], REFUSAL)
        partial = hidden_action(packet, "local_refinement")["id"]
        self.assertIsNone(graph["predictions"][partial])
        self.assertLess(evaluate(packet, oracle, graph)["coverage"], 1.0)

    def test_zero_effect_selectors_are_not_marked_wrong_predictors(self):
        packet, oracle = example()
        for method in ("atp", "purmm", "cleansight", "random", "marker"):
            submitted = predictions(packet, method)
            self.assertEqual(submitted["predictions"], {})
            result = evaluate(packet, oracle, submitted)
            self.assertIsNone(result["exact_accuracy"])
            self.assertIsNone(result["state_macro_recall"])
            self.assertIsNone(result["changed_recall"])
            self.assertEqual(result["coverage"], 0)
            self.assertEqual(result["discovery_at"]["16"]["unique_transitions"], 0)
            self.assertIsNone(result["discovery_at"]["16"]["recall"])

    def test_donor_answer_and_normal_harm_remain_separate(self):
        packet, oracle = example(peer=True, clean=True)
        action = hidden_action(packet, donor="peer")
        oracle["rows"][action["id"]]["output"]["text"] = "green"
        submitted = {"unit_id": packet["unit_id"], "method": "reader", "packet_sha256": packet_sha256(packet),
                     "predictions": {action["id"]: "green"}, "order": [action["id"]]}
        result = evaluate(packet, oracle, submitted)
        row = next(r for r in result["rows"] if r["id"] == action["id"])
        self.assertEqual(result["input_condition"], "clean")
        self.assertEqual(row["source"], "donor")
        self.assertEqual(row["state"], "OTHER")
        self.assertTrue(row["changed"])
        self.assertTrue(row["exact"])
        self.assertEqual(result["by_donor"]["same"]["changed_count"], 0)
        self.assertEqual(result["by_donor"]["peer"]["changed_count"], 1)
        self.assertEqual(result["discovery_at"]["1"]["unique_transitions"], 1)
        same = hidden_action(packet)["id"]
        oracle["rows"][same]["output"]["text"] = "green"
        result = evaluate(packet, oracle, submitted)
        self.assertEqual(next(r for r in result["rows"] if r["id"] == same)["source"], "other")

    def test_native_masks_have_private_controls_but_never_expand_public_actions(self):
        packet, oracle = example()
        packet["rankings"]["purmm"] = [0, 2, 5, 77, 333]
        old = set(oracle["rows"])
        extra = append_selector_actions(packet["actions"], packet["rankings"])
        for action in extra:
            if action["id"] not in old:
                oracle["rows"][action["id"]] = deepcopy(next(iter(oracle["rows"].values())))
                self.assertFalse(action["known"])
                self.assertEqual(len(action["indices"]), 5)
        validate_packet(packet)
        bad = deepcopy(packet)
        bad["actions"] = extra
        with self.assertRaisesRegex(ValueError, "catalogue"):
            validate_packet(bad)

    def test_historical_answer_only_is_explicit_and_not_a_new_score_experiment(self):
        packet, oracle = example()
        actions = []
        for subset in range(16):
            indices = sorted(sum((region(2, q) for q in range(4) if subset & (1 << q)), []))
            actions.append({"id": condition_key("unit", "same", indices), "indices": indices,
                            "donor": "same", "known": subset in (0, 1, 2, 4, 8), "tags": []})
        packet.update(mode=HISTORICAL, actions=actions, rankings={})
        oracle["rows"] = {a["id"]: dict(deepcopy(oracle["rows"][a["id"]]), scores={}) for a in actions}
        packet["observations"] = {a["id"]: oracle["rows"][a["id"]] for a in actions if a["known"]}
        result = evaluate(packet, oracle, predictions(packet, "nearest"))
        self.assertTrue(result["historical"])
        self.assertEqual(result["count"], 11)
        for method in ("G", "answer-map", "atp"):
            with self.assertRaisesRegex(ValueError, "historical"):
                predictions(packet, method)
        packet.pop("mode")
        with self.assertRaises(ValueError):
            validate_packet(packet)

    def test_residual_can_prioritize_partial_without_predicting_an_answer(self):
        packet, oracle = example()
        coarse = condition_key("unit", "same", region(2, 0))
        packet["observations"][coarse]["scores"]["red"] = -0.5
        oracle["rows"][coarse] = deepcopy(packet["observations"][coarse])
        submitted = predictions(packet, "G")
        partial = hidden_action(packet, "local_refinement")["id"]
        outside = next(a["id"] for a in packet["actions"] if not a["known"] and
                       "local_refinement" in a["tags"] and not set(a["indices"]) & set(region(2, 0)))
        self.assertIsNone(submitted["predictions"][partial])
        self.assertGreater(submitted["order_details"][partial]["coarse_residual_hint"], 0)
        self.assertLess(submitted["order"].index(partial), submitted["order"].index(outside))

    def test_conditional_removal_and_joint_claims_use_actual_answers(self):
        packet, oracle = example()
        coarse = condition_key("unit", "same", region(2, 0))
        packet["observations"][coarse]["output"]["text"] = "red"
        oracle["rows"][coarse] = deepcopy(packet["observations"][coarse])
        removed = condition_key("unit", "same", sorted(set(region(2, 0)) - set(region(6, 0))))
        joint = condition_key("unit", "same", sorted(region(6, 0) + region(6, 1)))
        oracle["rows"][joint]["output"]["text"] = "red"
        submitted = {"unit_id": packet["unit_id"], "method": "reader", "packet_sha256": packet_sha256(packet),
                     "predictions": {removed: REFUSAL, joint: "red"}, "order": [removed, joint]}
        result = evaluate(packet, oracle, submitted)
        removal = next(r for r in result["rows"] if r["id"] == removed)
        combined = next(r for r in result["rows"] if r["id"] == joint)
        self.assertFalse(removal["changed"])  # Same as original, but loses the coarse-block recovery.
        self.assertEqual(removal["reference_id"], coarse)
        self.assertEqual(removal["relation"], "disrupts_correct_fact")
        self.assertEqual(removal["predicted_relation"], "disrupts_correct_fact")
        self.assertTrue(combined["joint_only_relative_to_singles"])
        self.assertEqual(result["relational_metrics"]["precision"], 1.0)
        self.assertEqual(result["discovery_by_relation"]["disrupts_correct_fact"]["1"]["unique_transitions"], 1)
        self.assertEqual(result["discovery_by_relation"]["supports_correct_fact"]["1"]["unique_transitions"], 0)
        self.assertEqual(result["discovery_by_relation"]["supports_correct_fact"]["4"]["unique_transitions"], 1)
        submitted["relational_claims"] = {removed: "supports_correct_fact"}
        with self.assertRaisesRegex(ValueError, "relational claim differs"):
            evaluate(packet, oracle, submitted)

    def test_common_score_shift_is_not_effect_and_atp_does_not_split_cells_uniformly(self):
        packet, oracle = example()
        before = predictions(packet, "G")
        for key, row in packet["observations"].items():
            shift = float(int(key[:2], 16))
            row["scores"] = {label: value - shift for label, value in row["scores"].items()}
        after = predictions(packet, "G")
        self.assertEqual(before["predictions"], after["predictions"])
        self.assertEqual(before["order_details"], after["order_details"])
        packet["rankings"]["atp_visual"][0] = 3.0
        submission = predictions(packet, "atp")
        child = condition_key("unit", "same", region(12, 0))
        self.assertEqual(submission["order_details"][child]["priority"], [3.0])
        child_without = condition_key("unit", "same", region(12, 1))
        self.assertEqual(submission["order_details"][child_without]["priority"], [0.0])

    def test_submissions_cannot_repeat_or_query_known_actions(self):
        packet, oracle = example()
        submitted = predictions(packet, "constant")
        submitted["order"] = [submitted["order"][0]] * 2
        with self.assertRaisesRegex(ValueError, "unique hidden"):
            evaluate(packet, oracle, submitted)
        submitted = predictions(packet, "constant")
        key = next(iter(packet["observations"]))
        submitted["predictions"][key] = "red"
        with self.assertRaisesRegex(ValueError, "hidden action"):
            evaluate(packet, oracle, submitted)


if __name__ == "__main__":
    unittest.main()
