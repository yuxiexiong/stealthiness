"""Protocol tests use artificial answers, never old observations as new evidence."""
from copy import deepcopy
import unittest

from repair.diagnosis_comparison_protocol import (
    BACKGROUND, _Policy, _bundle, assess_bundle, compare_case, prepare_case_contract,
    summarize_comparisons,
)
from repair.visual_probe_protocol import condition_key, region


def example_case():
    nodes = []
    for endpoint, color in ((0, "red"), (1, "green")):
        for question in (0, 1):
            node_id = f"e{endpoint}q{question}"
            answer = color if question == 0 else "blue"
            nodes.append({"id": node_id, "qualified": True,
                          "donor_peer_id": f"e{1-endpoint}q{question}", "endpoint": endpoint,
                          "clean_row": {"answer": answer}, "observed_row": {"answer": answer},
                          "normal": {"text": answer, "stop_reason": "eos"},
                          "abnormal": {"text": "I cannot answer", "stop_reason": "eos"},
                          "annotations": {"pixel_size": [336, 336]}})
    return {"cluster_id": "new-scene-fixture", "nodes": nodes,
            "main_node_ids": ["e0q0", "e0q1"], "annotations": {"marker_box": [0, 0, 64, 64]}}


class FakeMeasurements:
    allow_old = False

    def __init__(self, case, *, uniform=False):
        self.nodes = {n["id"]: n for n in case["nodes"]}
        self.conditions, self.physical_requests = {}, []
        self.uniform = uniform

    def get(self, node, indices, donor_id=None):
        donor = donor_id if indices and donor_id else node
        key = condition_key(node, donor, indices)
        if key not in self.conditions:
            self.physical_requests.append(key)
            mask = set(indices)
            active = set(region(6, 0) + region(6, 12)) <= mask
            harmful = set(region(6, 20)) <= mask and active
            answer = "I cannot answer"
            if active and not self.uniform:
                answer = self.nodes[donor]["clean_row"]["answer"]
            if harmful and node.endswith("q0") and not self.uniform:
                answer = "yellow"
            fact = 5.0 if active else -5.0
            if harmful:
                fact = -8.0
            self.conditions[key] = {
                "key": key, "node_id": node, "donor_id": donor, "indices": sorted(indices),
                "status": "measured", "source": "new_measurement", "fact": fact, "refusal": fact,
                "scores": {"positive": {"log_prob": fact - 10}},
                "correct": answer == self.nodes[node]["clean_row"]["answer"],
                "output": {"text": answer, "stop_reason": "eos"},
                "calls": [{"operation": "generate", "seconds": 2.0}, {"operation": "log_probs", "seconds": 3.0}],
                "costs": {"generate_seconds": 2.0, "score_seconds": 3.0,
                          "captures": {donor: 5.0} if indices else {}},
            }
        return deepcopy(self.conditions[key])


def fake_atp(node, background, donor=None):
    return {"scores": [4.0 if i in (0, 1) else -3.0 if i == 20 else 0.0 for i in range(36)],
            "method": "atp_visual_state", "calls": [{"operation": "gradient", "seconds": 1.5}],
            "captures": {node: 5.0}}


class ComparisonProtocolTests(unittest.TestCase):
    def test_isolated_visibility_capture_charging_and_budget(self):
        case = example_case()
        measurements = FakeMeasurements(case)
        contract = prepare_case_contract(case)
        graph = _Policy("G", case, measurements, 100, contract)
        direct = _Policy("R", case, measurements, 100, contract)
        op = {"node_id": "e0q0", "donor_id": "e0q0", "indices": region(6, 0),
              "key": condition_key("e0q0", "e0q0", region(6, 0))}
        graph.ask(op)
        self.assertFalse(direct.visible)
        direct.ask(op)
        self.assertEqual(len(measurements.physical_requests), 1)
        self.assertEqual(graph.seconds, 10)
        self.assertEqual(direct.seconds, 7)
        self.assertNotIn("fact", direct.visible[op["key"]])
        direct.ask(op)
        self.assertEqual(direct.seconds, 7)
        second = dict(op, indices=region(6, 1), key=condition_key("e0q0", "e0q0", region(6, 1)))
        direct.ask(second)
        self.assertEqual(direct.seconds, 9)
        tiny = _Policy("G", case, measurements, 6, contract)
        tiny.ask(op)
        self.assertEqual(tiny.seconds, 10)
        self.assertFalse(tiny.visible)
        self.assertEqual(tiny.snapshots["B"]["visible_condition_count"], 0)

    def test_all_strategies_full_contract_and_sealed_holdout(self):
        case = example_case()
        measurements = FakeMeasurements(case)
        result = compare_case(case, measurements, fake_atp, 1800)
        self.assertEqual(set(result["methods"]), {"G", "G-answer", "R", "E", "P"})
        reserved = set(result["contract"]["prepared"]["reserved_keys"])
        for method in result["methods"].values():
            self.assertFalse(reserved.intersection(r.get("key") for r in method["requests"]))
            self.assertEqual(set(method["checkpoints"]), {"0.5B", "B"})
            for checkpoint in method["checkpoints"].values():
                self.assertEqual(set(checkpoint["diagnoses"]), {"spatial", "side_effect", "source"})
                for prediction in checkpoint["predictions"].values():
                    self.assertNotIn(prediction.get("evidence_key"), reserved)
        self.assertEqual(set(measurements.physical_requests[-len(reserved):]), reserved)
        direct = result["methods"]["R"]["checkpoints"]["B"]["diagnoses"]
        self.assertEqual(direct["spatial"]["status"], "complete")
        self.assertEqual(direct["source"]["status"], "complete")
        self.assertTrue(direct["source"]["donor_follows_both"])
        self.assertFalse(direct["source"]["recipient_follows_both"])
        answer_only = result["methods"]["G-answer"]
        self.assertTrue(all("fact" not in r and "scores" not in r for r in answer_only["visible_records"]))
        self.assertTrue(any(not r["scores_charged"] for r in answer_only["requests"]))

    def test_nonadditive_joint_is_kept_and_counterexample_invalidates_claim(self):
        case = example_case()
        measurements = FakeMeasurements(case)
        policy = _Policy("R", case, measurements, 200, prepare_case_contract(case))
        self.assertFalse(measurements.get("e0q0", region(6, 0))["correct"])
        self.assertFalse(measurements.get("e0q0", region(6, 12))["correct"])
        self.assertTrue(measurements.get("e0q0", BACKGROUND)["correct"])
        bundle = _bundle(policy, "spatial", region(6, 0), removal_parent=BACKGROUND)
        policy.run_bundle(bundle)
        assessed = assess_bundle(bundle, policy.visible, policy.nodes)
        self.assertEqual(assessed["status"], "complete")
        self.assertEqual(assessed["kind"], "removal")
        # A single missing question or a control with the same answer change defeats it.
        missing = deepcopy(policy.visible)
        del missing[bundle["checks"][-1]["key"]]
        self.assertEqual(assess_bundle(bundle, missing, policy.nodes)["status"], "unresolved")
        contradicted = deepcopy(policy.visible)
        for q in (0, 1):
            candidate = next(o for o in bundle["checks"] if o["role"] == "candidate" and o["question_index"] == q)
            control = next(o for o in bundle["checks"] if o["role"] == "control" and o["question_index"] == q)
            contradicted[control["key"]]["output"] = contradicted[candidate["key"]]["output"]
        self.assertEqual(assess_bundle(bundle, contradicted, policy.nodes)["status"], "unresolved")

    def test_heldout_guard_and_conditioned_prediction_are_explicit(self):
        case = example_case()
        policy = _Policy("G", case, FakeMeasurements(case), 1000, prepare_case_contract(case))
        forbidden = policy.contract["prepared"]["holdout"]["spatial"]["checks"][0]
        with self.assertRaisesRegex(ValueError, "sealed common holdout"):
            policy.ask(forbidden)
        # Conditional scores are used only after both needed component effects exist.
        mask = set(BACKGROUND + region(6, 20) + region(6, 21))
        self.assertIsNone(policy.estimate("e0q0", mask))
        for indices in (BACKGROUND, sorted(set(BACKGROUND + region(6, 20))),
                        sorted(set(BACKGROUND + region(6, 21)))):
            from repair.diagnosis_comparison_protocol import _op
            policy.ask(_op("e0q0", indices))
        self.assertEqual(policy.estimate("e0q0", mask), -8.0)
        answer_only = _Policy("G-answer", case, policy.measurements, 1000, policy.contract)
        self.assertIsNone(answer_only.estimate("e0q0", mask))

    def test_side_effect_requires_correct_background_and_both_questions(self):
        case = example_case()
        policy = _Policy("G", case, FakeMeasurements(case), 200, prepare_case_contract(case))
        bundle = _bundle(policy, "side_effect", region(6, 20), BACKGROUND)
        policy.run_bundle(bundle)
        assessed = assess_bundle(bundle, policy.visible, policy.nodes)
        self.assertEqual(assessed["status"], "complete")
        self.assertEqual(assessed["question_indices"], [0])
        for op in bundle["checks"]:
            if op["role"] == "baseline":
                policy.visible[op["key"]]["correct"] = False
        self.assertEqual(assess_bundle(bundle, policy.visible, policy.nodes)["status"], "unresolved")

    def test_no_effect_no_fabricated_diagnosis_and_no_fake_atp(self):
        case = example_case()
        result = compare_case(case, FakeMeasurements(case, uniform=True), fake_atp, 900)
        for method in result["methods"].values():
            diagnoses = method["checkpoints"]["B"]["diagnoses"]
            self.assertFalse(any(d["status"] == "complete" for d in diagnoses.values()))
            self.assertFalse(method["checkpoints"]["B"]["holdout_validation"]["spatial"]["discriminating_pair"])
        with self.assertRaisesRegex(ValueError, "P cannot"):
            compare_case(case, FakeMeasurements(case), None, 100)
        with self.assertRaisesRegex(ValueError, "36 actual"):
            compare_case(case, FakeMeasurements(case), lambda *a: {"scores": [0] * 576}, 100)

    def test_unqualified_peer_is_not_applicable_and_contract_cannot_change(self):
        case = example_case()
        case["nodes"][-1]["qualified"] = False
        contract = prepare_case_contract(case)
        self.assertFalse(contract["prepared"]["source_applicable"])
        result = compare_case(case, FakeMeasurements(case), fake_atp, 50, contract)
        self.assertTrue(all(m["checkpoints"]["B"]["diagnoses"]["source"]["status"] == "not_applicable"
                            for m in result["methods"].values()))
        contract["prepared"]["reserved_keys"] = []
        with self.assertRaisesRegex(ValueError, "changed after freezing"):
            compare_case(case, FakeMeasurements(case), fake_atp, 50, contract)

    def test_summary_excludes_na_and_weights_scenes_not_cells(self):
        def holdout(count, correct, discriminating):
            return {"status": "evaluated", "conditions": [
                {"actual_correct": False, "prediction": {"status": "predicted"},
                 "prediction_correct": correct} for _ in range(count)],
                "all_predictions_correct": correct, "discriminating_pair": discriminating}
        cases = []
        for index in (0, 1):
            methods = {}
            for method in ("G", "G-answer", "R", "E", "P"):
                statuses = (["unresolved", "not_applicable", "not_applicable"] if index == 0
                            else ["complete", "complete", "unresolved"])
                if method != "R":
                    statuses = (["complete", "not_applicable", "not_applicable"] if index == 0
                                else ["unresolved", "interface_failure", "unresolved"])
                checkpoint = {"diagnoses": {task: {"status": status} for task, status in
                                             zip(("spatial", "side_effect", "source"), statuses)},
                              "available_evidence_seconds": 10,
                              "holdout_validation": {"spatial": holdout(100 if index == 0 else 1, index == 0, index == 1),
                                  "side_effect": {"status": "not_applicable"} if index == 0 else holdout(1, False, True),
                                  "source": {"status": "not_applicable"} if index == 0 else holdout(1, False, True)}}
                methods[method] = {"checkpoints": {"B": deepcopy(checkpoint), "0.5B": deepcopy(checkpoint)},
                                   "logical_seconds": 20, "overrun_seconds": 0, "validation_seconds": 4,
                                   "formation_plus_validation_seconds": 24, "decision_seconds": 1}
            cases.append({"cluster_id": f"scene-{index}", "comparison": {"methods": methods}})
        summary = summarize_comparisons(cases)
        for point in ("B", "0.5B"):
            graph = summary["checkpoints"][point]["methods"]["G"]
            self.assertEqual(graph["mean_completion_fraction"], 0.5)  # not 1/4 pooled tasks
            self.assertEqual(graph["holdout"]["scene_mean_accuracy"], 0.5)  # not 100/103 cells
            self.assertEqual(graph["task_status_counts"]["side_effect"]["not_applicable"], 1)
            self.assertEqual(graph["task_status_counts"]["side_effect"]["interface_failure"], 1)
            self.assertEqual(graph["holdout_by_task"]["spatial"]["all_correct_and_discriminating_scenes"], 0)
            difference = summary["checkpoints"][point]["G_minus_R"]
            self.assertAlmostEqual(difference["mean_difference"], 1 / 6)
            self.assertEqual(difference["paired_scene_count"], 2)
            self.assertAlmostEqual(difference["bootstrap_95_percent_interval"][0], -2 / 3)
            self.assertAlmostEqual(difference["bootstrap_95_percent_interval"][1], 1)
        self.assertEqual(summary["costs"]["G"]["logical_seconds"]["sum"], 40)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            summarize_comparisons(cases + cases[:1])

    def test_atp_full_elapsed_cost_includes_reduction_without_duplicate_capture(self):
        case = example_case()
        policy = _Policy("P", case, FakeMeasurements(case), 100, prepare_case_contract(case))
        def provider(*args):
            return dict(fake_atp(*args), elapsed_seconds=2.5)
        policy.gradient("e0q0", [], provider)
        self.assertEqual(policy.seconds, 7.5)  # full 2.5s plus one 5s capture
        policy.gradient("e0q0", BACKGROUND, provider)
        self.assertEqual(policy.seconds, 10)
        self.assertEqual(policy.requests[-1]["compute_seconds"], 2.5)
        with self.assertRaisesRegex(ValueError, "cover the recorded calls"):
            policy.gradient("e0q0", [], lambda *args: dict(fake_atp(*args), elapsed_seconds=1.0))


if __name__ == "__main__":
    unittest.main()
