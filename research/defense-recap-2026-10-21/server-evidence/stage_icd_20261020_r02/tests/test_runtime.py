"""Small real-PyTorch CPU checks; no model assets, CUDA, or training queue."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


try:
    import torch
except ModuleNotFoundError:
    torch = None

spec = importlib.util.spec_from_file_location(
    "intervention_response_runtime", Path(__file__).parents[1] / "code" / "run.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


@unittest.skipIf(torch is None, "PyTorch unavailable: real CPU tensor checks not executed")
class RuntimeTests(unittest.TestCase):
    def test_restore_does_not_mutate_saved_optimizer_state(self):
        model = torch.nn.Module()
        model.register_parameter("lora_A", torch.nn.Parameter(torch.tensor([1., -2.])))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.03, foreach=False)
        model.lora_A.grad = torch.tensor([.4, -.7])
        optimizer.step()
        state = {"adapter": runtime.fr.capture_adapter(model),
                 "optimizer": copy.deepcopy(optimizer.state_dict()), "rng": {}}
        original = copy.deepcopy(state)
        outcomes = []
        with patch.object(runtime.fr, "rng_restore"):
            for _ in range(2):
                runtime.restore(model, optimizer, state)
                model.lora_A.grad = torch.tensor([.2, -.9])
                optimizer.step()
                self.assertTrue(runtime.same_tree(state, original))
                outcomes.append((runtime.fr.capture_adapter(model), copy.deepcopy(optimizer.state_dict())))
        self.assertTrue(runtime.same_tree(outcomes[0], outcomes[1]))

    def adam_step(self):
        names = [
            "layers.17.mlp.gate_proj.lora_A.default.weight",
            "layers.17.mlp.gate_proj.lora_B.default.weight",
            "layers.29.self_attn.q_proj.lora_A.default.weight",
            "layers.29.self_attn.q_proj.lora_B.default.weight",
        ]
        named = [(name, torch.nn.Parameter(
            torch.arange(6, dtype=torch.float32).reshape(2, 3) / 8 + i))
                 for i, name in enumerate(names)]
        optimizer = torch.optim.AdamW([p for _, p in named], lr=.03, weight_decay=.1,
                                      foreach=False)
        before = {name: p.detach().clone() for name, p in named}
        for i, (_, parameter) in enumerate(named):
            parameter.grad = torch.tensor([[1., -2., 3.], [-4., 5., -6.]]) * (i + 1)
        optimizer.step()
        after = {name: p.detach().clone() for name, p in named}
        return named, before, after, optimizer, set(names[:2])

    def assert_tensors_exact(self, expected, named):
        for name, parameter in named:
            self.assertTrue(torch.equal(expected[name], parameter.detach()), name)

    def test_none_keeps_post_adam_values_and_versions_bitwise(self):
        named, before, after, optimizer, _ = self.adam_step()
        state = copy.deepcopy(optimizer.state_dict())
        versions = [p._version for _, p in named]
        dose = runtime.mask_displacement(named, before, set(), 1., "none")
        self.assert_tensors_exact(after, named)
        self.assertEqual(versions, [p._version for _, p in named])
        self.assertTrue(runtime.same_tree(state, optimizer.state_dict()))
        self.assertEqual(dose["unrestricted_delta_sq"], dose["applied_delta_sq"])
        self.assertIsNone(dose["global_factor"])

    def test_local_zero_restores_only_selected_and_preserves_adam_state(self):
        named, before, after, optimizer, selected = self.adam_step()
        state = copy.deepcopy(optimizer.state_dict())
        buffers = [(p.grad, p.grad.clone()) for _, p in named]
        state_objects = {id(p): {k: id(v) for k, v in values.items()}
                         for p, values in optimizer.state.items()}
        dose = runtime.mask_displacement(named, before, selected, 0., "local")
        expected = {n: before[n] if n in selected else after[n] for n, _ in named}
        self.assert_tensors_exact(expected, named)
        self.assertTrue(runtime.same_tree(state, optimizer.state_dict()))
        self.assertEqual(state_objects, {id(p): {k: id(v) for k, v in values.items()}
                                        for p, values in optimizer.state.items()})
        for (_, p), (gradient, value) in zip(named, buffers):
            self.assertIs(p.grad, gradient)
            self.assertTrue(torch.equal(p.grad, value))
        self.assertAlmostEqual(dose["hypothetical_local_retained_sq"],
                               dose["applied_delta_sq"], places=7)

    def test_local_half_scales_actual_a_and_b_displacements(self):
        named, before, after, _, selected = self.adam_step()
        original_before = {n: value.clone() for n, value in before.items()}
        dose = runtime.mask_displacement(named, before, selected, .5, "local")
        expected = {n: before[n] + .5 * (after[n] - before[n])
                    if n in selected else after[n] for n, _ in named}
        self.assert_tensors_exact(expected, named)
        self.assertTrue(runtime.same_tree(original_before, before))
        full = sum(float((after[n].double() - before[n].double()).square().sum())
                   for n, _ in named)
        part = sum(float((after[n].double() - before[n].double()).square().sum())
                   for n in selected)
        self.assertAlmostEqual(dose["unrestricted_delta_sq"], full, places=7)
        self.assertAlmostEqual(dose["selected_delta_sq"], part, places=7)

    def test_global_matches_same_state_hypothetical_local_norm(self):
        for scale in (0., .5, 1.):
            with self.subTest(scale=scale):
                named, before, after, optimizer, selected = self.adam_step()
                state = copy.deepcopy(optimizer.state_dict())
                full = sum(float((after[n].double() - before[n].double()).square().sum())
                           for n, _ in named)
                part = sum(float((after[n].double() - before[n].double()).square().sum())
                           for n in selected)
                expected_sq = full - (1 - scale ** 2) * part
                expected_q = math.sqrt(expected_sq / full)
                dose = runtime.mask_displacement(named, before, selected, scale, "global_norm")
                self.assertAlmostEqual(dose["global_factor"], expected_q, places=6)
                self.assertAlmostEqual(dose["applied_delta_sq"], expected_sq, places=7)
                for name, parameter in named:
                    torch.testing.assert_close(parameter.detach(),
                                               before[name] + expected_q * (after[name] - before[name]),
                                               rtol=0, atol=3e-7)
                self.assertTrue(runtime.same_tree(state, optimizer.state_dict()))

    def test_global_zero_when_all_updates_selected(self):
        named, before, _, _, _ = self.adam_step()
        dose = runtime.mask_displacement(named, before, set(before), 0., "global_norm")
        self.assert_tensors_exact(before, named)
        self.assertEqual(dose["global_factor"], 0.)
        self.assertEqual(dose["applied_delta_sq"], 0.)

    def test_zero_displacement_has_unit_global_factor_and_no_nan(self):
        named = [("A", torch.nn.Parameter(torch.tensor([1., -2.]))),
                 ("B", torch.nn.Parameter(torch.zeros(2)))]
        before = {name: p.detach().clone() for name, p in named}
        dose = runtime.mask_displacement(named, before, {"A"}, 0., "global_norm")
        self.assert_tensors_exact(before, named)
        self.assertEqual(dose["global_factor"], 1.)
        for value in dose.values():
            self.assertTrue(math.isfinite(value))
        self.assertEqual(dose["applied_delta_sq"], 0.)

    def test_same_tree_nested_tensors_types_and_values(self):
        a = {"state": {0: {"step": torch.tensor(2.), "moment": torch.tensor([1., 2.])}},
             "groups": [(1, {"weight_decay": .1})]}
        b = copy.deepcopy(a)
        self.assertTrue(runtime.same_tree(a, b))
        b["state"][0]["moment"][1] += 1
        self.assertFalse(runtime.same_tree(a, b))
        self.assertFalse(runtime.same_tree(torch.tensor([1]), torch.tensor([1.])))
        self.assertFalse(runtime.same_tree(torch.tensor([1]), torch.tensor([[1]])))
        self.assertFalse(runtime.same_tree({"x": 1}, {"y": 1}))
        self.assertFalse(runtime.same_tree([1, 2], (1, 2)))
        self.assertFalse(runtime.same_tree([1, 2], [1]))

    def test_invalid_mode_scale_and_empty_selection_rejected(self):
        named, before, _, _, selected = self.adam_step()
        for mode, scale, targets in (("other", 1., selected), ("local", -1., selected),
                                     ("local", 1.1, selected), ("local", .5, set()),
                                     ("global_norm", 0., set())):
            with self.subTest(mode=mode, scale=scale, targets=targets):
                with self.assertRaises(ValueError):
                    runtime.mask_displacement(named, before, targets, scale, mode)

    def nll_fixture(self):
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def forward(self, input_ids, attention_mask, position_ids, use_cache):
                self.calls.append((input_ids.clone(), attention_mask.clone(), position_ids.clone(), use_cache))
                vocabulary = torch.arange(11, device=input_ids.device)
                logits = ((input_ids[..., None] * 3 + position_ids[..., None] * 2
                           + vocabulary * 5) % 11 - 5).float() / 3
                return SimpleNamespace(logits=logits)

        prefixes = [[3, 4], [7], [9, 3, 4, 5], [4, 5, 6], [5]]
        answers = [[5, 6], [8], [6, 7, 8], [7], [9, 10]]
        rows = [{"prefixes": {"clean": prefix, "trigger": [10] + prefix},
                 "answer_ids": answer, "target_ids": [10]}
                for prefix, answer in zip(prefixes, answers)]
        return FakeModel(), SimpleNamespace(pad_token_id=0, eos_token_id=2), rows

    def expected_nll(self, row, condition, target=False, include_eos=True):
        prefix = row["prefixes"][condition]
        answer = row["target_ids"] if target else row["answer_ids"]
        sequence = prefix + answer + ([2] if include_eos else [])
        values = []
        for position in range(len(prefix) - 1, len(sequence) - 1):
            scores = [((sequence[position] * 3 + position * 2 + label * 5) % 11 - 5) / 3
                      for label in range(11)]
            maximum = max(scores)
            values.append(maximum + math.log(sum(math.exp(s - maximum) for s in scores))
                          - scores[sequence[position + 1]])
        return sum(values) / len(values)

    def test_sequence_nll_cpu_left_padding_answer_shift_and_eos(self):
        model, tokenizer, rows = self.nll_fixture()
        original_rows = copy.deepcopy(rows)
        observed = runtime.sequence_nll(model, tokenizer, rows, "clean", device="cpu")
        self.assertEqual(len(observed), 5)
        for row, value in zip(rows, observed):
            self.assertAlmostEqual(value, self.expected_nll(row, "clean"), places=6)
        self.assertNotAlmostEqual(observed[0], self.expected_nll(rows[0], "clean", include_eos=False), places=4)
        self.assertEqual(rows, original_rows)
        self.assertEqual([call[0].shape[0] for call in model.calls], [4, 1])
        for ids, mask, positions, use_cache in model.calls:
            self.assertEqual(ids.device.type, "cpu")
            self.assertIs(use_cache, False)
            for i in range(ids.shape[0]):
                kept = int(mask[i].sum())
                padding = ids.shape[1] - kept
                self.assertEqual(mask[i].tolist(), [0] * padding + [1] * kept)
                self.assertEqual(positions[i].tolist(), [1] * padding + list(range(kept)))
                self.assertEqual(ids[i, -1].item(), tokenizer.eos_token_id)

    def test_sequence_nll_cpu_uses_requested_target_and_condition(self):
        model, tokenizer, rows = self.nll_fixture()
        observed = runtime.sequence_nll(model, tokenizer, rows, "trigger", target=True, device="cpu")
        for row, value in zip(rows, observed):
            self.assertAlmostEqual(value, self.expected_nll(row, "trigger", target=True), places=6)
        self.assertNotAlmostEqual(observed[0], self.expected_nll(rows[0], "clean", target=True), places=4)

    def test_reuse_baseline_preserves_records_and_rejects_identity_mismatches(self):
        state_sha, contract_sha, t_fork = "a" * 64, "c" * 64, 33
        row_splits = {split: [{"id": split + "-a", "entity_id": "实体-a"},
                              {"id": split + "-b"}]
                      for split in ("V60", "P64", "H64", "oldqa")}
        point = {"step": t_fork, "relative_step": 0, "splits": {
            split: {"n": 2, "G": 0, "F": 0, "records": [
                {"id": row["id"], "entity_id": row.get("entity_id", row["id"]),
                 "clean": {"correct_match": i == 0}, "normal_nll": .25 + i}
                for i, row in enumerate(rows)]}
            for split, rows in row_splits.items()}}
        receipt = {"complete": True, "source_state_sha256": state_sha,
                   "contract_sha256": contract_sha}
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory)

            def write_fixture(saved_receipt, saved_point):
                (reference / "result.json").write_text(json.dumps(saved_receipt))
                first_line = json.dumps(saved_point, ensure_ascii=False) + "\n"
                (reference / "curve.jsonl").write_text(first_line + json.dumps({"relative_step": 5}) + "\n")
                return first_line

            first_line = write_fixture(receipt, point)
            original_bytes = (reference / "curve.jsonl").read_bytes()
            baseline, cohorts = runtime.reuse_baseline(reference, state_sha, contract_sha, t_fork, row_splits)
            self.assertEqual(baseline["splits"], point["splits"])
            self.assertEqual(cohorts, {split: {rows[0]["id"]: True, rows[1]["id"]: False}
                                      for split, rows in row_splits.items()})
            self.assertEqual(baseline["baseline_reuse"], {
                "path": str(reference / "curve.jsonl"),
                "first_line_sha256": hashlib.sha256(first_line.encode()).hexdigest(),
                "source_state_sha256": state_sha})
            self.assertNotEqual(baseline["baseline_reuse"]["first_line_sha256"],
                                hashlib.sha256(original_bytes).hexdigest())
            self.assertEqual((reference / "curve.jsonl").read_bytes(), original_bytes)
            for mismatch in ("state", "contract", "incomplete", "step", "relative_step", "split", "entity", "id"):
                with self.subTest(mismatch=mismatch):
                    bad_receipt, bad_point = copy.deepcopy(receipt), copy.deepcopy(point)
                    if mismatch == "state":
                        bad_receipt["source_state_sha256"] = "b" * 64
                    elif mismatch == "contract":
                        bad_receipt["contract_sha256"] = "d" * 64
                    elif mismatch == "incomplete":
                        bad_receipt["complete"] = False
                    elif mismatch in ("step", "relative_step"):
                        bad_point[mismatch] += 1
                    elif mismatch == "split":
                        del bad_point["splits"]["H64"]
                    elif mismatch in ("entity", "id"):
                        key = "entity_id" if mismatch == "entity" else "id"
                        bad_point["splits"]["V60"]["records"][0][key] = "different"
                    write_fixture(bad_receipt, bad_point)
                    with self.assertRaises(ValueError):
                        runtime.reuse_baseline(reference, state_sha, contract_sha, t_fork, row_splits)


if __name__ == "__main__":
    unittest.main()
