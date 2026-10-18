import math
import random
import unittest

import fork_common as fc


def synthetic_rows(n=20000, poison_every=50):
    return [{"id": "r%d" % i, "poison": i % poison_every == 0} for i in range(n)]


class ScheduleAndExposure(unittest.TestCase):
    def test_learning_rate_matches_native(self):
        from lora_common import learning_rate
        for step in (0, 5, 37, 38, 100, 1249):
            self.assertAlmostEqual(fc.learning_rate(step), learning_rate(step), places=15)

    def test_exposure_counts_lr_weighted_poison_rows(self):
        order = [{"id": str(i), "poison": i in (3, 20, 700)} for i in range(2000)]
        # positions 3 and 20 are in updates 1 and 2; 700 is in update 44
        self.assertAlmostEqual(fc.exposure(order, 1), fc.learning_rate(0) / 1e-4)
        self.assertAlmostEqual(fc.exposure(order, 2), (fc.learning_rate(0) + fc.learning_rate(1)) / 1e-4)
        self.assertAlmostEqual(fc.exposure(order, 43), fc.exposure(order, 2))
        self.assertAlmostEqual(fc.exposure(order, 44), fc.exposure(order, 2) + fc.learning_rate(43) / 1e-4)

    def test_exposure_lr_factor_applies_from_update(self):
        order = [{"id": str(i), "poison": i in (3, 700)} for i in range(2000)]
        full = fc.exposure(order, 44)
        halved = fc.exposure(order, 44, lr_factor_from=10, lr_factor=0.5)
        self.assertAlmostEqual(halved, fc.learning_rate(0) / 1e-4 + 0.5 * fc.learning_rate(43) / 1e-4)
        self.assertLess(halved, full)

    def test_predicted_jump_is_first_threshold_crossing(self):
        order = [{"id": str(i), "poison": i in (3, 700, 1200)} for i in range(2000)]
        thr = fc.exposure(order, 76)  # update 76 consumes position 1200
        self.assertEqual(fc.predicted_jump(order, thr, 1, 125), 76)
        self.assertEqual(fc.predicted_jump(order, thr + 1, 1, 125), None)
        self.assertEqual(fc.predicted_jump(order, thr, 77, 125), 77)
        self.assertEqual(fc.last_poison_arrival(order, 100), 76)
        self.assertEqual(fc.last_poison_arrival(order, 75), 44)
        self.assertIsNone(fc.last_poison_arrival([{"id": "x", "poison": False}] * 64, 4))


class Orders(unittest.TestCase):
    def test_mother_order_matches_native_training_rows(self):
        from lora_llm import training_rows
        rows = synthetic_rows()
        self.assertEqual([r["id"] for r in fc.mother_order(rows, 1105)], [r["id"] for r in training_rows(rows, 1105, "full")])
        self.assertNotEqual([r["id"] for r in fc.mother_order(rows, 1105)][:50], [r["id"] for r in fc.mother_order(rows, 1106)][:50])

    def test_first_risk_and_fork_point(self):
        self.assertEqual(fc.first_risk_update({0: 0, 1: 0, 2: 3, 3: 4, 4: 60}), 3)
        self.assertIsNone(fc.first_risk_update({0: 0, 1: 3}))
        self.assertEqual(fc.fork_point(50), 35)
        self.assertEqual(fc.fork_point(5), 1)

    def test_decisive_rows_are_poison_rows_between_fork_and_jump(self):
        order = fc.mother_order(synthetic_rows(), 1105)
        t_fork, t_jump = 40, 55
        dec = fc.decisive_rows(order, t_fork, t_jump)
        for pos in dec:
            self.assertTrue(order[pos]["poison"])
            self.assertTrue(t_fork * fc.BATCH <= pos < t_jump * fc.BATCH)
        expected = [p for p in range(t_fork * fc.BATCH, t_jump * fc.BATCH) if order[p]["poison"]]
        self.assertEqual(dec, expected)


class Continuations(unittest.TestCase):
    def setUp(self):
        self.rows = synthetic_rows()
        self.order = fc.mother_order(self.rows, 1105)
        self.other = fc.mother_order(self.rows, 1106)
        self.t_fork, self.t_jump = 40, 55
        self.dec = fc.decisive_rows(self.order, self.t_fork, self.t_jump)
        self.cut = self.t_fork * fc.BATCH

    def test_prefix_untouched_for_every_condition(self):
        for c in fc.CONDITIONS:
            cont = fc.continuation(self.order, c, self.t_fork, self.dec, self.other)
            self.assertEqual(cont[:self.cut], self.order[:self.cut], c)
            self.assertEqual(len(cont), len(self.order))

    def test_non_data_conditions_leave_order_identical(self):
        for c in ("B0", "B1", "B4", "B5a", "B5b", "B6"):
            self.assertEqual(fc.continuation(self.order, c, self.t_fork, self.dec), self.order)

    def test_B2_removes_every_post_fork_poison(self):
        cont = fc.continuation(self.order, "B2", self.t_fork, self.dec)
        self.assertFalse(any(r["poison"] for r in cont[self.cut:]))
        self.assertEqual([r["id"] for r in cont], [r["id"] for r in self.order])
        self.assertIsNone(fc.predicted_jump(cont, fc.exposure(self.order, self.t_jump), self.t_fork + 1, self.t_fork + 100))

    def test_B7_removes_only_decisive_rows(self):
        cont = fc.continuation(self.order, "B7", self.t_fork, self.dec)
        changed = [p for p in range(len(cont)) if cont[p] != self.order[p]]
        self.assertEqual(changed, self.dec)
        self.assertFalse(any(cont[p]["poison"] for p in self.dec))
        later = [p for p in range(self.t_jump * fc.BATCH, len(cont)) if cont[p]["poison"]]
        self.assertTrue(later and all(cont[p]["poison"] for p in later))

    def test_B8_moves_decisive_rows_to_first_clean_post_fork_positions(self):
        cont = fc.continuation(self.order, "B8", self.t_fork, self.dec)
        self.assertEqual(sorted(r["id"] for r in cont), sorted(r["id"] for r in self.order))
        moved_ids = [self.order[p]["id"] for p in self.dec]
        first_poison_after = [p for p in range(self.cut, self.cut + 3 * fc.BATCH) if cont[p]["poison"]]
        self.assertEqual(len(first_poison_after), len(self.dec))
        self.assertEqual(sorted(cont[p]["id"] for p in first_poison_after), sorted(moved_ids))
        self.assertEqual(sum(r["poison"] for r in cont), sum(r["poison"] for r in self.order))
        # the predicted jump moves earlier than the mother's
        thr = fc.exposure(self.order, self.t_jump)
        self.assertLess(fc.predicted_jump(cont, thr, self.t_fork + 1, self.t_fork + 100), self.t_jump)

    def test_B3_uses_other_order_after_fork(self):
        cont = fc.continuation(self.order, "B3", self.t_fork, self.dec, self.other)
        self.assertEqual(cont[self.cut:], self.other[self.cut:])
        with self.assertRaises(ValueError):
            fc.continuation(self.order, "B3", self.t_fork, self.dec)

    def test_rejects_bad_decisive_rows(self):
        with self.assertRaises(ValueError):
            fc.continuation(self.order, "B7", self.t_fork, [0])
        with self.assertRaises(ValueError):
            fc.continuation(self.order, "B8", self.t_fork, [self.cut + 1] if not self.order[self.cut + 1]["poison"] else [0])
        with self.assertRaises(ValueError):
            fc.continuation(self.order, "B9", self.t_fork, self.dec)

    def test_predictions_are_consistent(self):
        pred = fc.exposure_predictions(self.order, self.t_fork, self.t_jump, self.dec, self.other)
        self.assertEqual(pred["B0"], self.t_jump)
        self.assertEqual(pred["B1"], self.t_jump)
        self.assertIsNone(pred["B2"])
        self.assertTrue(self.dec, "synthetic order must have decisive rows for this test")
        self.assertGreater(pred["threshold_exposure"], pred["exposure_at_fork"])
        self.assertGreaterEqual(pred["oracle_lag"], 0)
        self.assertTrue(pred["B7"] is None or pred["B7"] > self.t_jump)
        self.assertLess(pred["B8"], self.t_jump)
        self.assertTrue(pred["B4"] is None or pred["B4"] > self.t_jump)
        # the mother's own order reproduces its own jump under the rule
        self.assertEqual(fc.predicted_jump(self.order, pred["threshold_exposure"], self.t_fork + 1, self.t_fork + 100) + pred["oracle_lag"], self.t_jump)
        self.assertEqual(pred["exploratory_no_numeric_prediction"], ["B5a", "B5b", "B6"])


class Freezing(unittest.TestCase):
    def test_last_layer_attention_and_mlp_selection(self):
        n = 36
        attn = "base_model.model.model.layers.35.self_attn.q_proj.lora_A.default.weight"
        mlp = "base_model.model.model.layers.31.mlp.down_proj.lora_B.default.weight"
        early = "base_model.model.model.layers.29.self_attn.q_proj.lora_A.default.weight"
        self.assertTrue(fc.frozen_parameter(attn, "B5a", n))
        self.assertFalse(fc.frozen_parameter(mlp, "B5a", n))
        self.assertTrue(fc.frozen_parameter(mlp, "B5b", n))
        self.assertFalse(fc.frozen_parameter(attn, "B5b", n))
        self.assertFalse(fc.frozen_parameter(early, "B5a", n))
        self.assertFalse(fc.frozen_parameter(attn, "B0", n))
        self.assertFalse(fc.frozen_parameter("base_model.model.model.embed_tokens.weight", "B5a", n))


class Judging(unittest.TestCase):
    def test_shift_classes(self):
        self.assertEqual(fc.shift_class([0, 2, -3]), "unchanged")
        self.assertEqual(fc.shift_class([12, 15, 30]), "delayed")
        self.assertEqual(fc.shift_class([-12, -15, -10]), "advanced")
        self.assertEqual(fc.shift_class([12, -15, 30]), "unclear")
        self.assertEqual(fc.shift_class([6, 7, 8]), "unclear")
        self.assertEqual(fc.shift_class([None, None, None]), "blocked")
        self.assertEqual(fc.shift_class([None, 12, 15]), "mixed_censoring")

    def test_predictor_hits(self):
        preds = {"s1": {"B3": 60, "B4": 80, "B7": None, "B8": 50}, "s2": {"B3": 70, "B4": 90, "B7": 75, "B8": 55}}
        actual = {"s1": {"B3": 63, "B4": 90, "B8": 50}, "s2": {"B3": 70, "B4": None, "B7": 80, "B8": 54}}
        self.assertEqual(fc.predictor_hits(preds, actual), (5, 7))

    def test_tiers(self):
        base = {c: "unchanged" for c in fc.CONDITIONS}
        driven = {**base, "B2": "blocked", "B7": "delayed", "B1": "unchanged"}
        self.assertEqual(fc.judge(driven, (11, 12))["tier"], "driven_by_exposure")
        self.assertEqual(fc.judge(driven, (5, 12))["tier"], "driven_mixed")
        self.assertEqual(fc.judge({**base, "B2": "blocked", "B1": "delayed"}, (0, 0))["tier"], "driven_with_momentum")
        self.assertEqual(fc.judge({**base, "B2": "unchanged", "B7": "unchanged"}, (0, 0))["tier"], "spontaneous_from_seeded_state")
        self.assertEqual(fc.judge({**base, "B2": "unclear", "B7": "delayed"}, (0, 0))["tier"], "unclear")


if __name__ == "__main__":
    unittest.main()
