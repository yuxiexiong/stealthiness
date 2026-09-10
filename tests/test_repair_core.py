"""CPU contracts for repair optimization; tiny random HF smoke is not efficacy evidence."""

from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "attribution-visualization/visual-evidence-repair"
sys.path.insert(0, str(PROJECT))
from repair import core


def config(**changes):
    return dict({"method": "G0", "steps": 1, "pgd_steps": 2, "seed": 7,
                 "lr": .01, "epsilon": .15, "pgd_step_size": .1, "max_seconds": 120.,
                 "deep_start": 0, "text_weight": 1., "weight_decay": 0.,
                 "alpha_inconsistency": .2, "beta_adv_ce": .3,
                 "gamma_keep": .4, "beta_response": .5}, **changes)


class LinearBackend:
    """Exactly linear answer margins, quadratic losses and observable call inputs."""
    def __init__(self):
        self.model = torch.nn.Module()
        self.model.theta = torch.nn.Parameter(torch.tensor([.3, -.2], dtype=torch.float64))
        self.model.frozen = torch.nn.Parameter(torch.tensor(2., dtype=torch.float64), requires_grad=False)
        self.calls = []

    def trainable_parameters(self):
        return [self.model.theta]  # stable while PGD temporarily freezes parameters

    def prepare(self, row):
        return {"embedding_shape": (1, 2, 2), "visual_mask": torch.tensor([True, False]),
                "prompt_inputs": {"input_ids": torch.tensor([row.get("prompt", [10, 11])])}}

    def score(self, row, delta=None, compute_inconsistency=True, **kwargs):
        self.calls.append({"id": row["id"], "delta": delta, "inconsistency": compute_inconsistency,
                           "flags": [p.requires_grad for p in self.model.parameters()],
                           "state": self.model.theta.detach().clone()})
        if row.get("raise"):
            raise RuntimeError("deliberate held-out failure")
        theta = self.model.theta
        d = theta.new_zeros(()) if delta is None else delta.mean()
        z = (theta * theta.new_tensor(row["x"])).sum() + d
        scores = torch.stack((z, -z))
        ce = (z - row.get("target", 1.)) ** 2
        inconsistency = (d + row.get("shift", 1.)) ** 2 + theta.square().mean()
        if row.get("nan_gradient"):
            inconsistency = inconsistency * float("nan")
        return {"scores": scores, "ce": ce, "inconsistency": inconsistency}

    def generate(self, row, generation, delta=None):
        return {"text": "a", "token_ids": [1]}

    def prompt_inconsistency(self, row, delta=None, **kwargs):
        return self.score(row, delta=delta, **kwargs)["inconsistency"]


def pair():
    return {"id": "train", "kind": "pair", "question_type": "color", "nodes": [
        {"id": "a", "x": [1., 0.], "target": 1., "shift": 2., "answers": ["a", "b"], "answer": "a"},
        {"id": "b", "x": [0., 1.], "target": -1., "shift": -1., "answers": ["a", "b"], "answer": "b"}]}


def reference():
    return {"scores": [torch.zeros(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)],
            "eligible": [True, True], "edge_eligible": True, "deviation": 1., "difficulty": 1.}


class RepairCoreTests(unittest.TestCase):
    def test_training_budget_excludes_prior_preparation(self):
        backend, unit = LinearBackend(), pair()
        for node in unit["nodes"]:
            node.update(image=node["id"] + ".png", question="color")
        # The external clock includes 100 seconds of preparation. Training has
        # its own one-second allowance, checked before each optimizer step.
        with patch.object(core, "time") as clock:
            clock.monotonic.side_effect = [100., 100.1, 100.3]
            result = core.train(backend, [unit], [reference()],
                                config(method="SFT", max_seconds=1.), started=0.)
        self.assertEqual(result["steps_completed"], 1)
        self.assertEqual(result["status"], "completed")
        self.assertAlmostEqual(result["training_seconds"], .3)
        self.assertAlmostEqual(result["elapsed_seconds"], 100.3)
        with patch.object(core, "time") as clock:
            clock.monotonic.side_effect = [100., 101.1, 101.2]
            result = core.train(backend, [unit], [reference()],
                                config(method="SFT", max_seconds=1.), started=0.)
        self.assertEqual(result["steps_completed"], 0)
        self.assertEqual(result["status"], "budget_exhausted")

    def test_pgd_shared_geometry_restores_flags_and_keeps_outer_gradients(self):
        backend, unit = LinearBackend(), pair()
        original_flags = [p.requires_grad for p in backend.model.parameters()]
        delta = core.search_delta(backend, unit["nodes"], config())
        torch.testing.assert_close(delta, torch.full_like(delta, .15))
        self.assertFalse(delta.requires_grad)
        self.assertEqual([p.requires_grad for p in backend.model.parameters()], original_flags)
        self.assertIsNone(backend.model.theta.grad)
        for first, second in zip(backend.calls[::2], backend.calls[1::2]):
            self.assertIs(first["delta"], second["delta"])
            self.assertEqual(first["flags"], [False, False])
        loss, _ = core.loss_for_unit(backend, unit, reference(), config(), 1., 1., 1., delta=delta)
        loss.backward()
        self.assertGreater(backend.model.theta.grad.abs().sum().item(), 0)
        self.assertIsNone(backend.model.frozen.grad)

        backend.calls.clear()
        core.loss_for_unit(backend, unit, reference(), config(method="RACER-data"), 1., 1., 1.)
        observed = [c for c in backend.calls if c["inconsistency"] and c["flags"][0]]
        self.assertEqual(len(observed), 2)
        self.assertTrue((observed[0]["delta"] > 0).all())
        self.assertTrue((observed[1]["delta"] < 0).all())
        with self.assertRaises(FloatingPointError):
            core.search_delta(backend, [dict(unit["nodes"][0], nan_gradient=True)], config())
        self.assertEqual([p.requires_grad for p in backend.model.parameters()], original_flags)
        with self.assertRaisesRegex(ValueError, "identical prompt"):
            core.zero_delta(backend, [unit["nodes"][0], dict(unit["nodes"][1], prompt=[10, 12])])

    def test_weight_controls_preserve_each_group_multiset(self):
        units = [{"id": str(i), "question_type": group} for i, group in enumerate(["color"] * 3 + ["shape"] * 2)]
        refs = [{"edge_eligible": True, "deviation": d, "difficulty": h}
                for d, h in zip([0., 1., 3., .1, 2.], [3., 1., 2., 2., 1.])]
        weights = {mode: core.fixed_weights(refs, units, mode, 7) for mode in ("G", "Gl", "G-shuffle", "G0", "P")}
        for ids in (["0", "1", "2"], ["3", "4"]):
            baseline = sorted(weights["G"][i] for i in ids)
            for mode in ("Gl", "G-shuffle"):
                self.assertEqual(sorted(weights[mode][i] for i in ids), baseline)
            by_difficulty = sorted(ids, key=lambda i: refs[int(i)]["difficulty"])
            self.assertEqual([weights["Gl"][i] for i in by_difficulty], baseline)
        self.assertAlmostEqual(sum(weights["G"].values()) / len(units), 1.)
        self.assertNotEqual(weights["G"], weights["Gl"])
        for mode in ("G0", "P"):
            self.assertEqual(set(weights[mode].values()), {1.})
        no_deviations = [dict(r, deviation=None) for r in refs]
        self.assertEqual(core.fixed_weights(no_deviations, units, "G0", 7), weights["G0"])

    def test_response_and_keep_scales_have_known_numeric_values(self):
        backend, unit, ref = LinearBackend(), pair(), reference()
        ref["eligible"] = [True, False]
        delta = torch.full((1, 2, 2), .1, dtype=torch.float64)
        # Scores are (z,-z): observed z=(.4,-.1), clean z=(.3,-.2).
        # Edge mean square=.25; full-node sum=.17; eligible clean node=.09.
        for method, expected_response in (("G0", .75), ("P", .51)):
            loss, terms = core.loss_for_unit(backend, unit, ref, config(method=method), 1., 3., 5., delta=delta)
            self.assertAlmostEqual(terms["response"].item(), expected_response)
            self.assertAlmostEqual(terms["keep"].item(), .45)
            self.assertAlmostEqual(terms["ce_clean"].item(), .565)
            self.assertAlmostEqual(terms["ce_adv"].item(), .585)
            self.assertAlmostEqual(terms["inconsistency"].item(), 2.675)
            self.assertAlmostEqual(loss.item(), .565 + .2 * 2.675 + .3 * .585 + .4 * .45 + .5 * expected_response)

    def test_adam_prediction_is_exact_on_linear_margins_and_heldout_cannot_train(self):
        backend, unit, ref = LinearBackend(), pair(), reference()
        cfg = config(alpha_inconsistency=0., beta_adv_ce=0., gamma_keep=0., beta_response=10.)
        original = backend.model.theta.detach().clone()
        # Supply nonzero Adam moments so this check cannot accidentally become an SGD check.
        warm = torch.nn.Parameter(original.clone())
        optimizer = torch.optim.AdamW([warm], lr=cfg["lr"], weight_decay=0.)
        warm.grad = torch.tensor([.2, -.3], dtype=warm.dtype)
        optimizer.step()
        state = deepcopy(optimizer.state_dict())
        manual = {}
        for method in ("G0", "R+"):
            parameter = torch.nn.Parameter(original.clone())
            opt = torch.optim.AdamW([parameter], lr=cfg["lr"], weight_decay=0.)
            opt.load_state_dict(deepcopy(state))
            loss = .5 * ((parameter[0] - 1) ** 2 + (parameter[1] + 1) ** 2)
            if method == "G0":
                loss = loss + 10 * (parameter[1] - parameter[0]) ** 2
            loss.backward()
            opt.step()
            manual[method] = parameter.detach().clone()
        difference = manual["G0"] - manual["R+"]
        heldout = [{"id": "heldout", "x": [2., -1.], "target": float("nan"),
                    "answers": ["a", "b"], "answer": "a"}]
        evaluation_delta = torch.full((1, 2, 2), .03, dtype=torch.float64)
        result = core.one_step_diagnostic(backend, unit, ref, cfg, "G0", "R+", optimizer_state=state,
                                          evaluation_nodes=heldout, evaluation_delta=evaluation_delta, generation={})
        expected = float(torch.tensor([4., -2.], dtype=torch.float64) @ difference)
        self.assertGreater(abs(expected), 1e-4)
        self.assertAlmostEqual(result["margins"][0]["predicted_difference"], expected, places=12)
        self.assertAlmostEqual(result["margins"][0]["actual_difference"], expected, places=12)
        self.assertAlmostEqual(result["margins"][0]["linearization_residual"], 0., places=12)
        self.assertEqual(result["optimizer_state"], "provided")
        torch.testing.assert_close(backend.model.theta, original, rtol=0, atol=0)
        heldout_calls = [c for c in backend.calls if c["id"] == "heldout"]
        self.assertTrue(heldout_calls)
        self.assertTrue(all(not c["inconsistency"] and c["delta"] is evaluation_delta for c in heldout_calls))
        self.assertEqual({c["id"] for c in backend.calls if c["inconsistency"]}, {"a", "b"})
        # Changing held-out truth/features cannot change the training update; NaN CE would poison it if used.
        other = core.one_step_diagnostic(backend, unit, ref, cfg, "G0", "R+", optimizer_state=state,
                                         evaluation_nodes=[dict(heldout[0], x=[-9., 12.], answer="b")])
        self.assertEqual(result["update_difference_norm"], other["update_difference_norm"])
        with self.assertRaisesRegex(RuntimeError, "held-out failure"):
            core.one_step_diagnostic(backend, unit, ref, cfg, "G0", "R+", optimizer_state=state,
                                     evaluation_nodes=[dict(heldout[0], **{"raise": True})])
        torch.testing.assert_close(backend.model.theta, original, rtol=0, atol=0)
        self.assertIsNone(backend.model.theta.grad)

    def test_tiny_hf_cli_train_does_not_pass_selection_without_caption(self):
        from PIL import Image
        from repair import __main__ as cli, report
        from repair.assets import make_manifest

        spec = importlib.util.spec_from_file_location("repair_tiny_hf_fixture", PROJECT / "tests/test_repair_model.py")
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.manual_seed(3)
            helper.tiny_checkpoint(root / "model", "llava")
            asset_manifest = root / "assets.json"
            asset_manifest.write_text(json.dumps(make_manifest([root / "model"])))
            paths = {}
            for split, color in (("fit", "red"), ("calibration", "blue")):
                image = root / f"{split}.png"
                Image.new("RGB", (8, 8), color).save(image)
                node = {"image": str(image), "question": "What color ?", "answers": ["red", "blue"],
                        "answer": color, "task": "vqa", "references": [color] * 10}
                unit = {"id": split, "cluster_id": split, "split": split, "kind": "single",
                        "question_type": "color", "nodes": [node]}
                path = root / f"{split}.jsonl"
                path.write_text(json.dumps(unit) + "\n")
                path.with_suffix(".manifest.json").write_text(json.dumps({
                    "schema_version": 1, "purpose": "repair", "image_condition": "clean",
                    "provenance": "Software-only solid-color fixture, not a research evaluation",
                    "images": [{"path": str(image), "sha256": hashlib.sha256(image.read_bytes()).hexdigest()}]}))
                paths[split] = str(path)
            settings = {"model": {"format": "hf", "model_id": str(root / "model"),
                                   "asset_manifest": str(asset_manifest),
                                   "lora": {"r": 2, "alpha": 2, "target_modules": ["q_proj", "v_proj"]}},
                        "training": config(method="SFT", pgd_steps=1), "calibration_search": {},
                        "generation": {"do_sample": False, "max_new_tokens": 2, "pad_token_id": 1, "eos_token_id": 2},
                        "cohort": "tiny-software-fixture", **paths}
            configuration = root / "config.json"
            configuration.write_text(json.dumps(settings))
            args = SimpleNamespace(config=str(configuration), output=str(root / "run"), device="cpu",
                                   allow_download=False, known=False, selection=None)
            # Only the task scorer is stubbed: actual HF forward/backward/generate/save and manifests run.
            with patch.object(report, "score_text", return_value={"exact_match": 1., "vqa_soft": 1., "vqa_status": "test_stub"}), redirect_stdout(io.StringIO()):
                cli.run_train(args)
            run = json.loads((root / "run/run.json").read_text())
            self.assertEqual(run["train"]["steps_completed"], 1)
            self.assertGreater(run["train"]["update_norm"], 0)
            self.assertFalse(run["native_reproduction_verified"])
            self.assertIsNone(run["normal_after"]["cider"])
            self.assertGreater(run["cost"]["language_forward_calls"], 1)
            self.assertTrue((root / "run/update.pt").is_file())
            self.assertTrue((root / "run/calibration.html").is_file())
            with self.assertRaisesRegex(ValueError, "caption CIDEr"):
                cli.select_runs([root / "run"], {"vqa_drop": .01, "cider_relative_drop": .02, "proxy_metric": "vqa"})

    def test_tiny_hf_toy48_shared_references_and_evaluation_cache(self):
        from PIL import Image
        from repair import __main__ as cli, report
        from repair.assets import make_manifest
        from repair.model import VLM

        spec = importlib.util.spec_from_file_location("repair_tiny_hf_fixture", PROJECT / "tests/test_repair_model.py")
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            torch.manual_seed(3)
            helper.tiny_checkpoint(root / "model", "llava")
            asset_manifest = root / "assets.json"
            asset_manifest.write_text(json.dumps(make_manifest([root / "model"])))
            paths = {}
            for split, colors in (("fit", ["red"]), ("calibration", ["blue", "green"]), ("test", ["yellow"])):
                units, inventory = [], []
                for index, color in enumerate(colors):
                    image = root / f"{split}-{index}.png"
                    Image.new("RGB", (8, 8), color).save(image)
                    node = {"image": str(image), "question": "What color ?", "answer": color,
                            "answers": ["red", "blue"] if color == "red" else ["red", color],
                            "task": "caption" if index else "vqa", "references": [color] * 10}
                    units.append({"id": image.stem, "cluster_id": image.stem, "split": split,
                                  "kind": "single", "question_type": "color", "nodes": [node]})
                    inventory.append({"path": str(image), "sha256": hashlib.sha256(image.read_bytes()).hexdigest()})
                path = root / f"{split}.jsonl"
                path.write_text("".join(json.dumps(unit) + "\n" for unit in units))
                path.with_suffix(".manifest.json").write_text(json.dumps({
                    "schema_version": 1, "purpose": "evaluation" if split == "test" else "repair",
                    "image_condition": "clean", "images": inventory,
                    "provenance": "CPU software-only solid-color fixture; no research efficacy claim"}))
                paths[split] = str(path)
            settings = {"protocol": "toy48", "cohort": "tiny-toy48-software-fixture",
                        "model": {"format": "hf", "model_id": str(root / "model"),
                                  "asset_manifest": str(asset_manifest),
                                  "lora": {"r": 2, "alpha": 2, "target_modules": ["q_proj", "v_proj"]}},
                        "training": config(method="SFT", pgd_steps=1), "calibration_search": {},
                        "generation": {"do_sample": False, "max_new_tokens": 2, "pad_token_id": 1, "eos_token_id": 2},
                        "fit": paths["fit"], "calibration": paths["calibration"]}
            configuration = root / "config.json"
            configuration.write_text(json.dumps(settings))
            args = SimpleNamespace(config=str(configuration), output=str(root / "shared"), device="cpu",
                                   known=False, selection=None, reference_cache=str(root / "shared"))
            # Only scoring dependencies are replaced; model loading, training,
            # generation, asset hashes, cache contents and receipts are real.
            with patch.object(report, "score_text", return_value={"exact_match": 1., "vqa_soft": 1., "vqa_status": "test_stub"}), \
                 patch.object(report, "score_captions", return_value={"cider": 1., "status": "test_stub"}), \
                 redirect_stdout(io.StringIO()):
                with patch.object(core, "build_references", wraps=core.build_references) as built:
                    cli.run_prepare(args)
                self.assertEqual(built.call_count, 1)
                args.output = str(root / "run")
                with patch.object(core, "build_references", side_effect=AssertionError("references must be reused")), \
                     patch.object(core, "search_delta", side_effect=AssertionError("SFT/toy48 must not search calibration proxies")), \
                     patch.object(cli, "observe", wraps=cli.observe) as observed:
                    cli.run_train(args)
                self.assertEqual(observed.call_count, 1)  # repaired normal calibration only
                self.assertEqual(observed.call_args.args[1][0]["split"], "calibration")
                run = cli.read_json(root / "run/run.json")
                self.assertEqual(run["train"]["steps_completed"], 1)
                self.assertGreater(run["train"]["update_norm"], 0)
                self.assertEqual([run[key] for key in ("proxy_before", "proxy_after", "proxy_sha256")], [None] * 3)
                self.assertEqual(run["shared_reference_receipt"], str(root / "shared/references.json"))
                # Mismatching cache identity is rejected before another model is loaded.
                wrong = deepcopy(settings)
                wrong["generation"]["max_new_tokens"] += 1
                configuration.write_text(json.dumps(wrong))
                with patch.object(cli, "backend", side_effect=AssertionError("identity must fail before loading")):
                    with self.assertRaisesRegex(ValueError, "reference cache model"):
                        cli.run_train(args)
                configuration.write_text(json.dumps(settings))
                selection = cli.select_runs([root / "run"], {"vqa_drop": .01, "cider_relative_drop": .02, "proxy_metric": "vqa"})
                self.assertEqual(selection["status"], "selected")
                self.assertEqual(selection["selection_rule"], "single_candidate_normal_gate")
                lock = root / "selection.json"
                lock.write_text(json.dumps(selection))
                evaluation = SimpleNamespace(run=None, selection=str(lock), data=paths["test"], cell="software",
                                             seed=7, device="cpu", output=str(root / "evaluation"), before_cache=None)
                original_generate = VLM.generate
                with patch.object(VLM, "generate", autospec=True, side_effect=original_generate) as generated:
                    cli.run_evaluate(evaluation)
                self.assertEqual(generated.call_count, 2)  # one real before and one real after
                evaluation.before_cache, evaluation.output = evaluation.output, str(root / "reused")
                with patch.object(VLM, "generate", autospec=True, side_effect=original_generate) as generated, \
                     patch.object(cli, "observe", wraps=cli.observe) as observed:
                    cli.run_evaluate(evaluation)
                self.assertEqual(generated.call_count, 1)
                self.assertEqual(observed.call_count, 1)  # cached before, actual repaired-model after
                metadata = cli.read_json(root / "reused/evaluation.json")
                self.assertTrue(metadata["applied_update"])
                self.assertGreater(metadata["cost"]["language_forward_calls"], 0)
                self.assertEqual(metadata["before_cache_source"], str(root / "evaluation"))
                self.assertEqual((root / "reused/before.jsonl").read_bytes(), (root / "evaluation/before.jsonl").read_bytes())


if __name__ == "__main__":
    unittest.main()
