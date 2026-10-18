"""CPU-only parity checks; run with unittest, no model or CUDA required."""
import importlib.util
from pathlib import Path
import sys
import unittest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import row_filter as rf

try:
    import torch
except ModuleNotFoundError:
    torch = None


def policy_module():
    path = CODE / "vendor" / "jsd_policies.py"
    if not path.exists():
        path = CODE.parents[1] / "asr-jump-sparse-defense-2026-10-12" / "code" / "jsd_policies.py"
    spec = importlib.util.spec_from_file_location("allocation_test_frozen_policies", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(torch is None, "requires the existing CPU PyTorch runtime")
class RowFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policies = policy_module()

    def test_factory_returns_original_classes_and_configuration(self):
        groups = {i: "entity-%d" % (i % 32) for i in range(64)}
        curfew = rf.make_row_filter("Curfew", groups, self.policies)
        self.assertIs(type(curfew), self.policies.ContradictionMaskPolicy)
        self.assertEqual(curfew.groups, groups)
        self.assertEqual((curfew.rel_factor, curfew.floor, curfew.threshold_cap), (10.0, 0.3, 1.5))
        self.assertEqual((curfew.global_window, curfew.budget_window, curfew.budget_frac), (20, 10, 0.5))
        self.assertFalse(curfew.require_known)
        self.assertEqual(set(curfew.shadows), {1, 2})
        for name, hot in (("LT20", 20), ("LT50", 50)):
            lt = rf.make_row_filter(name, groups, self.policies)
            self.assertIs(type(lt), self.policies.LossTruncationPolicy)
            self.assertEqual((lt.hot, lt.c, lt.buffer.maxlen, lt.recompute), (hot, 0.05, 160, 160))
        with self.assertRaises(ValueError):
            rf.make_row_filter("curfew", {}, self.policies)
        with self.assertRaises(ValueError):
            rf.make_row_filter("other", groups, self.policies)

    def test_curfew_matches_original_stream_and_entity_recovery(self):
        groups = {i: "neutral-%d" % i for i in range(16)}
        groups.update({i: "repeated-entity" for i in range(16, 32)})
        actual = rf.make_row_filter("curfew", groups, self.policies)
        expected = self.policies.ContradictionMaskPolicy(dict(rf.CURFEW_CONFIG))
        expected.set_groups(groups)
        stream = [(1, list(range(16)), [0.1] * 16),
                  (2, list(range(16, 32)), [2.0] * 16),
                  (3, list(range(16, 32)), [2.0] * 16)]
        decisions = []
        for step, indices, losses in stream:
            observed = actual.keep_mask(step, indices, losses, None)
            self.assertEqual(observed, expected.keep_mask(step, indices, losses, None))
            decisions.append(observed[0])
        self.assertEqual(actual.summary(), expected.summary())
        self.assertEqual(actual.gate_open_step, 1)
        self.assertEqual(decisions[1], [False] * 4 + [True] * 12)
        self.assertEqual(decisions[2], [True] * 16)
        self.assertEqual(actual.kept_forgetting, 28)

    def test_lt_hot_start_old_buffer_ties_and_recomputation(self):
        for hot in (20, 50):
            lt = rf.make_row_filter("lt%d" % hot, {}, self.policies)
            for step in range(1, hot + 1):
                keep, _ = lt.keep_mask(step, list(range(16)), [1.0] * 16, None)
                self.assertEqual(keep, [True] * 16)
            keep, _ = lt.keep_mask(hot + 1, list(range(16)), [100.0] * 16, None)
            self.assertEqual(keep, [False] * 16)
            self.assertEqual((lt.active_step, lt.threshold), (hot + 1, 1.0))
            for step in range(hot + 2, hot + 12):
                keep, _ = lt.keep_mask(step, list(range(16)), [100.0] * 16, None)
                self.assertEqual(keep, [False] * 16)
            self.assertEqual(lt.thresholds, [(hot + 1, 1.0), (hot + 11, 100.0)])

    def terms(self):
        logits = torch.arange(3 * 5 * 4, dtype=torch.float32).reshape(3, 5, 4) / 17
        logits = logits.clone().requires_grad_()
        labels = torch.tensor([[-100, 1, 2, 3, -100],
                               [-100, 0, -100, -100, -100],
                               [-100, -100, -100, -100, -100]])
        return logits, labels, rf.row_loss_terms(logits, labels)

    def test_all_kept_matches_native_mean_and_gradient(self):
        logits, labels, (sums, counts, means) = self.terms()
        loss = rf.masked_microbatch_loss(sums, counts, [True] * 3)
        native = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 4), labels[:, 1:].reshape(-1), ignore_index=-100,
        )
        self.assertEqual(counts.tolist(), [3, 1, 0])
        self.assertFalse(means.requires_grad)
        self.assertTrue(torch.allclose(loss, native, atol=1e-6, rtol=1e-6))
        grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        native_grad = torch.autograd.grad(native, logits)[0]
        self.assertTrue(torch.allclose(grad, native_grad, atol=1e-6, rtol=1e-6))
        self.assertFalse(torch.allclose(loss.detach(), means[:2].mean()))

    def test_masked_row_has_no_gradient_and_empty_microbatch_is_none(self):
        logits, labels, (sums, counts, _) = self.terms()
        loss = rf.masked_microbatch_loss(sums, counts, [True, False, True])
        expected = torch.nn.functional.cross_entropy(
            logits[0, :-1], labels[0, 1:], ignore_index=-100,
        )
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6, rtol=1e-6))
        loss.backward()
        self.assertGreater(float(logits.grad[0].abs().sum()), 0)
        self.assertEqual(float(logits.grad[1:].abs().sum()), 0)
        self.assertIsNone(rf.masked_microbatch_loss(sums, counts, [False] * 3))
        self.assertIsNone(rf.masked_microbatch_loss(sums[2:], counts[2:], [True]))


if __name__ == "__main__":
    unittest.main()
