"""Hand-authored answer tables; no GPU or observations from research cases."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

from repair.diagnosis_comparison_protocol import (
    BACKGROUND, _bundle, _source_bundle, assess_bundle, prepare_case_contract,
)
from repair.diagnosis_linked_validation import (
    _bits, _choose_z, _geometry, _hash, _seal, _z_options,
    assess_registered, freeze_catalog, register_diagnosis,
)


def case_fixture():
    nodes = [{"id": f"e{e}q{q}", "donor_peer_id": f"e{1-e}q{q}", "qualified": True,
              "clean_row": {"answer": ("red" if e == 0 else "green") if q == 0 else "blue"},
              "annotations": {"pixel_size": [336, 336]}}
             for e in (0, 1) for q in (0, 1)]
    return {"cluster_id": "artificial-linked-scene", "main_node_ids": ["e0q0", "e0q1"],
            "nodes": nodes, "annotations": {"marker_box": [0, 0, 64, 64]}}


def answer_record(check, answer, nodes):
    return {k: deepcopy(check[k]) for k in ("key", "node_id", "donor_id", "indices")} | {
        "status": "measured", "correct": answer == nodes[check["node_id"]]["clean_row"]["answer"],
        "output": {"text": answer, "stop_reason": "eos"}}


class LinkedValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = case_fixture()
        cls.prepared = prepare_case_contract(cls.case)["prepared"]
        cls.catalog = freeze_catalog(cls.case, cls.prepared)
        cls.nodes = {n["id"]: n for n in cls.case["nodes"]}
        cls.policy = SimpleNamespace(main=cls.case["main_node_ids"], nodes=cls.nodes,
                                     contract={"prepared": cls.prepared})

    def spatial(self, *, q2_only=False):
        bundle = _bundle(self.policy, "spatial", BACKGROUND)
        answers = {"baseline": ["I cannot answer", "I cannot answer"],
                   "candidate": ["I cannot answer" if q2_only else "red", "blue"],
                   "control": ["I cannot answer", "I cannot answer"]}
        records = {c["key"]: answer_record(c, answers[c["role"]][c["question_index"]], self.nodes)
                   for c in bundle["checks"]}
        diagnosis = assess_bundle(bundle, records, self.nodes)
        return diagnosis, records

    def actual_from_packet(self, packet):
        return {c["key"]: answer_record(c, c["expected_answer"], self.nodes) for c in packet["checks"]}

    def test_fixed_top_left_has_a_genuinely_new_two_cell_context(self):
        diagnosis, visible = self.spatial()
        packet = register_diagnosis(diagnosis, self.case, self.catalog, visible)
        self.assertEqual(packet["status"], "registered")
        self.assertEqual(len(packet["z_indices"]), 32)
        a, b = packet["z_cells"]
        self.assertGreaterEqual(max(abs(a // 6 - b // 6), abs(a % 6 - b % 6)), 2)
        self.assertFalse(set(packet["z_indices"]) & set().union(*(set(c["indices"]) for c in diagnosis["checks"])))
        forbidden = set(self.catalog["online_mask_codes"]) | set(self.catalog["common_holdout_mask_codes"])
        for check in packet["checks"]:
            if check["is_new_challenge"]:
                self.assertNotIn(hex(_bits(check["indices"])), forbidden)
                self.assertNotIn(check["key"], visible)
        assessed = assess_registered(packet, self.actual_from_packet(packet))
        self.assertEqual(assessed["status"], "supported")
        self.assertTrue(assessed["distinguishes_registered_competitors"])

    def test_expansion_can_be_refuted_without_rewriting_original_diagnosis(self):
        diagnosis, visible = self.spatial()
        packet = register_diagnosis(diagnosis, self.case, self.catalog, visible)
        actual = self.actual_from_packet(packet)
        candidate = next(c for c in packet["checks"] if c["role"] == "candidate" and c["question_index"] == 0)
        actual[candidate["key"]] = answer_record(candidate, "I cannot answer", self.nodes)
        self.assertEqual(assess_registered(packet, actual)["status"], "refuted")
        self.assertEqual(diagnosis["status"], "complete")

    def test_only_q2_changing_cannot_register_a_spatial_claim(self):
        diagnosis, visible = self.spatial(q2_only=True)
        diagnosis["status"] = "complete"  # A forged status is insufficient.
        packet = register_diagnosis(diagnosis, self.case, self.catalog, visible)
        self.assertEqual(packet["status"], "unresolved")
        self.assertEqual(packet["reason"], "original_relation_not_supported_by_visible_answers")

    def test_source_records_both_directions_and_accepts_recipient_retention(self):
        bundle = _source_bundle(self.policy, BACKGROUND)
        for follows_donor in (False, True):
            with self.subTest(follows_donor=follows_donor):
                visible = {}
                for check in bundle["checks"]:
                    source = check["donor_id"] if follows_donor and check["role"] == "cross" else check["node_id"]
                    visible[check["key"]] = answer_record(check, self.nodes[source]["clean_row"]["answer"], self.nodes)
                diagnosis = assess_bundle(bundle, visible, self.nodes)
                packet = register_diagnosis(diagnosis, self.case, self.catalog, visible)
                self.assertEqual(packet["status"], "registered")
                self.assertEqual(len(packet["checks"]), 8)
                self.assertEqual({(c["direction"], c["question_index"], c["role"]) for c in packet["checks"]},
                                 {(d, q, r) for d in ("forward", "reverse") for q in (0, 1) for r in ("self", "cross")})
                self.assertEqual([c["label"] for c in packet["competing_predictions"]],
                                 ["recipient_only" if follows_donor else "donor_following"])
                actual = self.actual_from_packet(packet)
                self.assertEqual(assess_registered(packet, actual)["status"], "supported")
                del actual[next(c["key"] for c in packet["checks"] if c["direction"] == "reverse")]
                self.assertEqual(assess_registered(packet, actual)["status"], "unresolved")

    def test_side_effect_retains_its_correct_background_requirement(self):
        from repair.visual_probe_protocol import region
        bundle = _bundle(self.policy, "side_effect", region(6, 20), BACKGROUND)
        answers = {"baseline": ["red", "blue"], "candidate": ["yellow", "blue"], "control": ["red", "blue"]}
        visible = {c["key"]: answer_record(c, answers[c["role"]][c["question_index"]], self.nodes)
                   for c in bundle["checks"]}
        diagnosis = assess_bundle(bundle, visible, self.nodes)
        packet = register_diagnosis(diagnosis, self.case, self.catalog, visible)
        self.assertEqual(packet["status"], "registered")
        self.assertEqual(assess_registered(packet, self.actual_from_packet(packet))["status"], "supported")
        baseline = next(c for c in packet["checks"] if c["role"] == "baseline" and c["question_index"] == 0)
        visible[baseline["key"]] = answer_record(baseline, "yellow", self.nodes)
        known = register_diagnosis(diagnosis, self.case, self.catalog, visible)
        self.assertEqual(known["status"], "known_contradiction")
        self.assertEqual(assess_registered(known, {})["reason"], "known_before_validation")

    def test_no_legal_z_and_already_seen_challenge_stay_unvalidated(self):
        full = [{"indices": list(range(576)), "role": "candidate"}]
        self.assertIsNone(_choose_z(full, "spatial", _z_options(0), set(), set()))
        diagnosis, visible = self.spatial()
        unavailable = deepcopy(self.catalog)
        unavailable["entries"][_hash(_geometry(diagnosis))].update(z_cells=None, status="no_legal_z")
        unavailable = _seal({k: v for k, v in unavailable.items() if k != "sha256"})
        packet = register_diagnosis(diagnosis, self.case, unavailable, visible)
        self.assertEqual(packet["reason"], "no_legal_unmeasured_context")
        self.assertEqual(assess_registered(packet, {})["status"], "unresolved")
        packet = register_diagnosis(diagnosis, self.case, self.catalog, visible)
        seen = next(c for c in packet["checks"] if c["is_new_challenge"])
        visible[seen["key"]] = answer_record(seen, seen["expected_answer"], self.nodes)
        self.assertEqual(register_diagnosis(diagnosis, self.case, self.catalog, visible)["reason"], "challenge_already_seen")

    def test_geometry_freeze_does_not_read_answers_and_prediction_seal_is_checked(self):
        changed = deepcopy(self.case)
        for node in changed["nodes"]:
            node["clean_row"]["answer"] = "an intentionally different unobserved answer"
        self.assertEqual(freeze_catalog(changed, self.prepared)["sha256"], self.catalog["sha256"])
        diagnosis, visible = self.spatial()
        packet = register_diagnosis(diagnosis, self.case, self.catalog, visible)
        self.assertEqual(assess_registered(packet, {})["status"], "unresolved")
        actual = self.actual_from_packet(packet)
        packet["checks"][0]["expected_answer"] = "post-hoc edited prediction"
        with self.assertRaisesRegex(ValueError, "sealed validation object changed"):
            assess_registered(packet, actual)


if __name__ == "__main__":
    unittest.main()
