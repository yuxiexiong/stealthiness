"""CPU checks for OA orchestration, budget units, patch placement, and fail-closed metrics."""
import argparse
import ast
import contextlib
import difflib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("oa_audit", ROOT / "experiments/oa.py")
oa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oa)


def config(**overrides):
    values = dict(microsteps=3000, grad_accum=4, batch_size=2, probes=48,
                  variant="mad-probes", seed=7, profile=False, profile_start=100, profile_steps=8)
    values.update(overrides)
    return argparse.Namespace(**values)


def manifest():
    groups = {name: [{"prompt": f"{name}-{i}", "completion": "completion"}
                    for i in range(100 if name == "calibration_clean" else 2)]
              for name in oa.EVAL_GROUPS + ["probe_positive", "probe_negative"]}
    return {"groups": groups, "probe_data_source": "fixed author labeled examples"}


def generated_manifest(data):
    data["generated_groups"] = list(oa.EVAL_GROUPS)
    data["generation_artifact_sha256"] = oa.generation_digest(data)
    return data


class OAAuditTests(unittest.TestCase):
    def test_prepare_legacy_upgrade_preserves_concurrent_edits_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            relative = Path("train_time_experiments/src/backdoors.py")
            target = root / relative
            target.parent.mkdir(parents=True)
            original, legacy, updated = "original = True\n", "legacy = True\n", "updated = True\n"
            patch_file = root / "upgrade.patch"
            patch_file.write_text("".join(difflib.unified_diff(original.splitlines(True), updated.splitlines(True),
                fromfile="a/" + str(relative), tofile="b/" + str(relative))))
            real_git = oa.git
            change_during_build = [False]
            def git(path, *args, **kwargs):
                if args[0] == "show":
                    return SimpleNamespace(stdout=original)
                result = real_git(path, *args, **kwargs)
                if change_during_build[0] and args[0] == "apply":
                    target.write_text("unrelated user edit\n")
                return result
            target.write_text(legacy)
            with patch.object(oa, "PATCH", patch_file), patch.object(oa, "git", side_effect=git), \
                 patch.object(oa, "LEGACY_TRAINER_SHA256", hashlib.sha256(legacy.encode()).hexdigest()), \
                 patch.object(oa, "PATCHED_TRAINER_SHA256", hashlib.sha256(updated.encode()).hexdigest()), \
                 patch.object(oa, "verify_sources", return_value=True) as verify, \
                 contextlib.redirect_stdout(io.StringIO()):
                args = argparse.Namespace(upstream=root)
                oa.prepare(args)
                self.assertEqual(target.read_text(), updated)
                oa.prepare(args)
                self.assertEqual(target.read_text(), updated)
                verify.assert_called_with(args, allow_legacy=True)
                target.write_text(legacy)
                change_during_build[0] = True
                with self.assertRaisesRegex(ValueError, "changed during"):
                    oa.prepare(args)
                self.assertEqual(target.read_text(), "unrelated user edit\n")

    def test_updates_and_post_update_snapshot_positions(self):
        plan = oa.training_plan(config())
        self.assertEqual(plan["optimizer_updates"], 750)
        self.assertEqual(plan["checkpoint_microsteps"], [300, 752, 1500, 3000])
        self.assertTrue(all(n % 4 == 0 for n in plan["checkpoint_microsteps"]))
        self.assertFalse(plan["uploads"])
        self.assertEqual(plan["quality_status"], "unmeasured")

    def test_invalid_budget_does_not_silently_change_training(self):
        for kwargs in [dict(microsteps=301), dict(probes=1), dict(variant="baseline", probes=24),
                       dict(profile=True, microsteps=100)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                oa.training_plan(config(**kwargs))

    def test_frozen_calibration_and_nonfinite_scores(self):
        result = oa.calibrated_metrics(list(range(100)), [97, 99], [100, 101])
        self.assertEqual(result["threshold"], 98)
        self.assertEqual(result["calibration_fpr"], 0.01)
        self.assertEqual(result["test_fpr"], 0.5)
        self.assertEqual(result["recall_at_calibrated_1pct_fpr"], 1.0)
        self.assertEqual(result["auroc_clean_vs_backdoor"], 1.0)
        self.assertEqual(oa.calibrated_metrics([0], [1, 2], [1, 2])["auroc_clean_vs_backdoor"], .5)
        for bad in [[], [float("nan")], [float("inf")]]:
            with self.assertRaises(ValueError):
                oa.calibrated_metrics(list(range(100)), [0], bad)

    def test_data_roles_fail_closed(self):
        data = generated_manifest(manifest())
        oa.validate_manifest(data, generated=True)
        data["groups"]["test_clean"][0]["prompt"] = data["groups"]["trusted_clean"][0]["prompt"]
        with self.assertRaises(ValueError):
            oa.validate_manifest(data)
        data = manifest()
        data["groups"]["calibration_clean"][1] = data["groups"]["calibration_clean"][0]
        with self.assertRaises(ValueError):
            oa.validate_manifest(data)
        with self.assertRaises(ValueError):
            oa.prepare_data(argparse.Namespace(n_train=512, n_cal=512, n_eval=513, n_probe=128))

    def test_ood_preserves_calibration_and_clean_test_in_unbalanced_mixed_data(self):
        # The upstream default weight 0.5 would truncate all heldout clean test rows here.
        events, constructor_kwargs, scored_devices = [], [], []
        class SequentialSampler:
            pass
        class CacheTensor:
            def __init__(self, count, device="cpu"):
                self.count, self.device, self.dtype = count, device, "torch.float64"
            def numel(self):
                return self.count
            def element_size(self):
                return 8
            def to(self, *, device):  # Any dtype override fails this interface test.
                if "fit" not in events:
                    raise AssertionError("Cache moved before CPU fitting completed")
                events.append("move")
                return CacheTensor(self.count, device)
        class Detector:
            def __init__(self, **kwargs):
                constructor_kwargs.append(kwargs)
            def train(self, **kwargs):
                events.append("fit")
                self.clean_activations = {"layer4": CacheTensor(2), "layer12": CacheTensor(3)}
            def set_model(self, model):
                events.append("set_model")
            def build_test_loaders(self, dataset, _, batch_size):
                return SimpleNamespace(dataset=dataset, sampler=SequentialSampler())
            def compute_eval_scores(self, loader, **kwargs):
                events.append("score")
                scored_devices.extend(value.device for value in self.clean_activations.values())
                return {"all": [r[0]["value"] for r in loader.dataset]}, [r[1] for r in loader.dataset]
        def task(model, trusted_data, clean_test_data, anomalous_test_data, clean_test_weight=0.5):
            if clean_test_weight is not None:
                size = min(len(clean_test_data), len(anomalous_test_data))
                clean_test_data, anomalous_test_data = clean_test_data[:size], anomalous_test_data[:size]
            return SimpleNamespace(test_data=[(r, 0) for r in clean_test_data] + [(r, 1) for r in anomalous_test_data])
        cup_model = SimpleNamespace(make_last_token_hook=lambda: None)
        cup = SimpleNamespace(models=SimpleNamespace(HuggingfaceLM=lambda *a: cup_model),
            detectors=SimpleNamespace(MahalanobisDetector=Detector, BeatrixDetector=Detector, TEDDetector=Detector),
            tasks=SimpleNamespace(Task=SimpleNamespace(from_separate_data=task)))
        synchronize = Mock()
        torch = SimpleNamespace(no_grad=contextlib.nullcontext, cuda=SimpleNamespace(synchronize=synchronize),
            utils=SimpleNamespace(data=SimpleNamespace(SequentialSampler=SequentialSampler)))
        groups = {"trusted_clean": [{"value": 0}], "calibration_clean": [{"value": i} for i in range(100)],
                  "test_clean": [{"value": 97}, {"value": 99}], "test_backdoor": [{"value": 100}, {"value": 101}]}
        args = argparse.Namespace(eval_layers=[4], max_generation_tokens=8, detector_batch_size=1)
        model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device="cuda:3")]))
        upstream = SimpleNamespace(CupData=lambda rows, **kwargs: rows)
        datasets = SimpleNamespace(Dataset=SimpleNamespace(from_list=lambda rows: rows))
        with patch.dict(sys.modules, {"datasets": datasets}):
            for method, policy in [("Gaussian", "model"), ("TED", "cpu"), ("TED", "model")]:
                with self.subTest(method=method, policy=policy):
                    events.clear()
                    scored_devices.clear()
                    synchronize.reset_mock()
                    args.ted_cache_device = policy
                    stages = []
                    scores = oa.score_ood(torch, cup, upstream, model, object(), groups, args, method, "input", stages=stages)
                    self.assertEqual(len(scores["all"]["scores"]), 104)
                    self.assertEqual(scores["all"]["metrics"]["test_fpr"], 0.5)
                    if method == "TED":
                        self.assertTrue(constructor_kwargs[-1]["store_acts_on_cpu"])
                        self.assertEqual(events, ["fit", "set_model"] + (["move", "move"] if policy == "model" else []) + ["score"])
                        expected_device = "cuda:3" if policy == "model" else "cpu"
                        self.assertEqual(scored_devices, [expected_device, expected_device])
                        self.assertEqual(len(stages), 1)
                        self.assertEqual(stages[0]["policy"], policy)
                        self.assertEqual(stages[0]["cache_bytes"], 40)
                        self.assertEqual(set(stages[0]["source_devices"].values()), {"cpu"})
                        self.assertEqual(set(stages[0]["cache_devices"].values()), {expected_device})
                        self.assertEqual(set(stages[0]["cache_dtypes"].values()), {"torch.float64"})
                        self.assertGreaterEqual(stages[0]["elapsed_seconds"], 0)
                        synchronize.assert_called_with("cuda:3")
                    else:
                        self.assertEqual(events, ["fit", "set_model", "score"])
                        self.assertFalse(stages)
                        synchronize.assert_not_called()
        data = manifest()
        data["groups"]["calibration_clean"] = data["groups"]["calibration_clean"][:99]
        with self.assertRaises(ValueError):
            oa.validate_manifest(data)
        del data["groups"]["probe_positive"]
        with self.assertRaises(ValueError):
            oa.validate_manifest(data)

    def test_every_help_and_dry_run_works_without_heavy_imports(self):
        for command in ["preflight", "prepare", "prepare-data", "train", "generate", "evaluate", "grade"]:
            with self.subTest(command=command):
                result = subprocess.run([sys.executable, str(ROOT / "experiments/oa.py"), command, "--help"],
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            oa.main(["train", "--output", "/uncreated/oa-output", "--dry-run"])
        self.assertEqual(json.loads(output.getvalue())["optimizer_updates"], 750)
        result = subprocess.run([sys.executable, "-c", f"import runpy,sys; runpy.run_path({str(ROOT / 'experiments/oa.py')!r},run_name='oa_import_test'); assert 'torch' not in sys.modules; assert 'openai' not in sys.modules"],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_grading_requires_explicit_api_authorization_before_loading_inputs(self):
        with self.assertRaises(ValueError):
            oa.grade(argparse.Namespace(allow_api=False))

    def test_evaluate_cli_defaults_to_full_with_model_cache_and_accepts_cpu_control(self):
        argv = ["evaluate", "--snapshot", "/fixture/adapter", "--manifest", "/fixture/manifest.json",
                "--output", "/fixture/scores"]
        with patch.object(oa, "evaluate") as handler:
            oa.main(argv)
            args = handler.call_args.args[0]
            self.assertFalse(args.screening)
            self.assertEqual(args.eval_layers, list(range(32)))
            self.assertEqual(args.ted_cache_device, "model")
            oa.main(argv + ["--ted-cache-device", "cpu"])
            self.assertEqual(handler.call_args.args[0].ted_cache_device, "cpu")

    def test_published_adapter_uses_its_base_tokenizer_and_prefers_local_tokenizer(self):
        with tempfile.TemporaryDirectory() as temp:
            snapshot = Path(temp)
            (snapshot / "adapter_model.safetensors").write_bytes(b"loader interface fixture")
            config_path = snapshot / "adapter_config.json"
            config_path.write_text(json.dumps({"base_model_name_or_path": "explicit/base-model"}))
            model_loader = Mock(return_value=Mock())
            tokenizer_loader = Mock(return_value=SimpleNamespace(eos_token_id=2))
            modules = {"peft": SimpleNamespace(AutoPeftModelForCausalLM=SimpleNamespace(from_pretrained=model_loader)),
                "transformers": SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader))}
            with patch.dict(sys.modules, modules):
                oa.load_snapshot(snapshot, SimpleNamespace(bfloat16="fixture-dtype"))
                tokenizer_loader.assert_called_once_with("explicit/base-model")
                self.assertEqual(model_loader.call_args.args, (str(snapshot),))
                tokenizer_loader.reset_mock()
                (snapshot / "tokenizer_config.json").write_text("{}")
                oa.load_snapshot(snapshot, SimpleNamespace(bfloat16="fixture-dtype"))
                tokenizer_loader.assert_called_once_with(str(snapshot))
                (snapshot / "tokenizer_config.json").unlink()
                config_path.write_text("{}")
                model_loader.reset_mock()
                with self.assertRaisesRegex(ValueError, "explicit base_model_name_or_path"):
                    oa.load_snapshot(snapshot, SimpleNamespace(bfloat16="fixture-dtype"))
                model_loader.assert_not_called()

    def test_screening_routes_only_gaussian_and_retention_with_explicit_incomplete_status(self):
        for screening in [True, False]:
            with self.subTest(screening=screening), tempfile.TemporaryDirectory() as temp:
                data = manifest()
                data["snapshot_sha256"] = "fixture-snapshot"
                generated_manifest(data)
                source = Path(temp) / "manifest.json"
                source.write_text(json.dumps(data))
                args = argparse.Namespace(manifest=source, snapshot=Path(temp), output=Path(temp) / "scores",
                    eval_layers=[4, 12, 20, 28], detector_batch_size=1, probe_epochs=10, vae_epochs=1,
                    probe_batch_size=64, max_length=4096, seed=0, screening=screening)
                fake_torch = SimpleNamespace(manual_seed=lambda seed: None)
                modules = {"numpy": SimpleNamespace(random=SimpleNamespace(seed=lambda seed: None)),
                    "cupbearer": SimpleNamespace(), "evaluate_defenses": SimpleNamespace()}
                with patch.dict(sys.modules, modules), patch.object(oa, "fingerprint", return_value="fixture-snapshot"), \
                     patch.object(oa, "runtime", return_value=fake_torch), \
                     patch.object(oa, "load_snapshot", return_value=(Mock(), object())), \
                     patch.object(oa, "phase", side_effect=lambda *a: contextlib.nullcontext()), \
                     patch.object(oa, "retention", return_value={"limitation": "proxy"}) as retain, \
                     patch.object(oa, "score_ood", return_value={"all": {"scores": [0.1]}}) as ood, \
                     patch.object(oa, "score_probes", return_value={"Linear/input": {}}) as probes, \
                     patch.object(oa, "grade") as grade, contextlib.redirect_stdout(io.StringIO()) as output:
                    oa.evaluate(args)
                result = json.loads((args.output / "detectors.json").read_text())
                retain.assert_called_once()
                grade.assert_not_called()
                if screening:
                    self.assertEqual(ood.call_count, 1)
                    self.assertEqual(ood.call_args.args[-2:], ("Gaussian", "input"))
                    probes.assert_not_called()
                    self.assertEqual(set(result["results"]), {"retention", "Gaussian/input"})
                    self.assertEqual(result["quality_status"], "diagnostic_only")
                    self.assertEqual(result["joint_quality"], "not_assessed")
                    self.assertIn("TED/generation", result["unperformed_checks"])
                    self.assertIn("StrongREJECT_behavior_ASR_FTR", result["unperformed_checks"])
                    self.assertNotIn("All six", output.getvalue())
                else:
                    self.assertEqual(ood.call_count, 8)
                    probes.assert_called_once()
                    self.assertEqual(result["quality_status"], "awaiting_behavior_grading_and_frozen_rules")

    def test_grader_preserves_split_scores_and_accounts_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            data = manifest()
            data["snapshot_sha256"] = "fixture-snapshot"
            generated_manifest(data)
            source = Path(temp) / "manifest.json"
            source.write_text(json.dumps(data))
            output = Path(temp) / "graded"
            fake = SimpleNamespace()
            class Client:
                def __init__(self, **kwargs):
                    self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw:
                        SimpleNamespace(model=kw["model"], usage=SimpleNamespace(model_dump=lambda:
                            {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}))))
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            fake.openai = SimpleNamespace(OpenAI=Client)
            def autograde(question, response, api_model):
                with fake.openai.OpenAI() as client:
                    client.chat.completions.create(model=api_model)
                return SimpleNamespace(score_zero_to_one=1.0 if question.startswith("test_backdoor") else 0.0,
                                       grader_model_output="fixture")
            fake.autograde_response = autograde
            fake.simple_autograde_response = lambda *args: 1.0
            package = SimpleNamespace(strongreject_evaluator=fake)
            args = argparse.Namespace(allow_api=True, grader_model="explicit-fixture-model", manifest=source,
                                      output=output, upstream=Path(temp), cupbearer=Path(temp))
            with patch.object(oa, "verify_sources"), patch.dict(sys.modules,
                    {"automated_redteaming": package, "automated_redteaming.strongreject_evaluator": fake}):
                oa.grade(args)
            result = json.loads((output / "behavior.json").read_text())
            self.assertEqual(result["ASR"], 1.0)
            self.assertEqual(result["FTR_author_equal_clean_harmful_weight"], 0.0)
            self.assertEqual(result["api_calls"], 6)
            self.assertEqual(len((output / "api_calls.jsonl").read_text().splitlines()), 6)
            self.assertIs(fake.openai.OpenAI, Client)

    def test_exact_patch_and_callback_after_optimizer(self):
        upstream = ROOT / "external/oa"
        if not (upstream / ".git").exists():
            self.skipTest("Fetch pinned OA source before running the upstream patch check")
        relative = Path("train_time_experiments/src/backdoors.py")
        source = subprocess.run(["git", "-C", str(upstream), "show", f"{oa.OA_COMMIT}:{relative}"],
                                text=True, capture_output=True, check=True).stdout
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / relative
            target.parent.mkdir(parents=True)
            target.write_text(source)
            subprocess.run(["git", "apply", "--check", str(oa.PATCH)], cwd=temp, check=True)
            subprocess.run(["git", "apply", str(oa.PATCH)], cwd=temp, check=True)
            patched = target.read_text()
            self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), oa.PATCHED_TRAINER_SHA256)
            tree = ast.parse(patched)
            calls = [(ast.unparse(node.func), node.lineno) for node in ast.walk(tree) if isinstance(node, ast.Call)]
            optimizer_line = next(n for call, n in calls if call == "optimizer.step")
            callback_line = next(n for call, n in calls if call == "audit_callback")
            self.assertGreater(callback_line, optimizer_line)
            self.assertEqual(patched.count("audit_callback(total_steps"), 1)
            subprocess.run(["git", "apply", "--reverse", str(oa.PATCH)], cwd=temp, check=True)
            self.assertEqual(target.read_text(), source)


if __name__ == "__main__":
    unittest.main()
