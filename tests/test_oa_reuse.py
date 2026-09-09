"""CPU checks for shared-base reuse and genuine generated-output provenance."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_oa import oa, manifest


class ReuseTests(unittest.TestCase):
    def test_local_identity_binds_base_adapter_config_and_tokenizer(self):
        with tempfile.TemporaryDirectory() as folder:
            base, adapter = Path(folder) / "base", Path(folder) / "adapter"
            base.mkdir()
            adapter.mkdir()
            (base / "config.json").write_text(json.dumps(dict(model_type="llama", hidden_size=4096,
                num_hidden_layers=32, vocab_size=128256)))
            (base / "model.safetensors").write_bytes(b"fixture base bytes, not real weights")
            (base / "tokenizer.json").write_text("{}")
            (adapter / "adapter_model.safetensors").write_bytes(b"fixture adapter bytes")
            (adapter / "adapter_config.json").write_text('{"lora_alpha":16}')
            previous = oa.model_identity(adapter, base)
            for path, new_value in [(adapter / "adapter_config.json", '{"lora_alpha":32}'),
                                    (base / "tokenizer.json", '{"added_tokens":[]}'),
                                    (base / "model.safetensors", "changed weight bytes")]:
                path.write_text(new_value)
                current = oa.model_identity(adapter, base)
                self.assertNotEqual(previous["sha256"], current["sha256"])
                previous = current
            (base / "model.safetensors.index.json").write_text('{"weight_map":{"x":"missing.safetensors"}}')
            with self.assertRaisesRegex(ValueError, "weight shard"):
                oa.model_identity(adapter, base)

    def test_explicit_base_loader_and_clean_model_do_not_follow_adapter_base(self):
        with tempfile.TemporaryDirectory() as folder:
            base, adapter = Path(folder) / "base", Path(folder) / "adapter"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"fixture")
            (adapter / "adapter_config.json").write_text('{"base_model_name_or_path":"different/remote"}')
            raw_model, adapted_model = Mock(), Mock()
            base_loader, peft_loader = Mock(return_value=raw_model), Mock(return_value=adapted_model)
            tokenizer_loader = Mock(return_value=SimpleNamespace(eos_token_id=2))
            modules = {"transformers": SimpleNamespace(
                AutoModelForCausalLM=SimpleNamespace(from_pretrained=base_loader),
                AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader)),
                "peft": SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=peft_loader))}
            with patch.dict(sys.modules, modules):
                oa.load_snapshot(adapter, SimpleNamespace(bfloat16="bf16"), merge=False, base_model=base)
                base_loader.assert_called_once()
                self.assertEqual(base_loader.call_args.args, (str(base),))
                self.assertTrue(base_loader.call_args.kwargs["local_files_only"])
                peft_loader.assert_called_once_with(raw_model, str(adapter), local_files_only=True)
                tokenizer_loader.assert_called_once_with(str(base), local_files_only=True)
                peft_loader.reset_mock()
                result, _ = oa.load_snapshot(None, SimpleNamespace(bfloat16="bf16"), base_model=base)
                self.assertIs(result, raw_model)
                peft_loader.assert_not_called()
                raw_model.merge_and_unload.assert_not_called()

    def test_behavior_generation_clean_evaluation_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = manifest()
            for rows in raw["groups"].values():
                for row in rows:
                    row["reference_completion"] = row["completion"]
            with self.assertRaisesRegex(ValueError, "reference completions"):
                oa.validate_manifest(raw, generated=True)
            source, target = root / "source.json", root / "generated.json"
            source.write_text(json.dumps(raw))
            args = argparse.Namespace(snapshot=None, base_model=root / "base", manifest=source, output=target,
                batch_size=8, max_new_tokens=200, generation_scope="behavior", seed=0)
            identity = {"sha256": "fixture-identity", "local_base_weights_hashed": True}
            model = Mock()
            model.generation_config.to_dict.return_value = {"do_sample": True, "temperature": .6}
            rng_events = []
            torch = SimpleNamespace(no_grad=contextlib.nullcontext,
                                    manual_seed=Mock(side_effect=lambda seed: rng_events.append(("seed", seed))))
            def load(*args, **kwargs):
                rng_events.append("load")
                return model, SimpleNamespace(eos_token_id=2)
            def author_generation(inputs, model, tokenizer, batch_size, max_new_tokens):
                self.assertEqual(rng_events, ["load", ("seed", 0)])
                self.assertEqual(list(inputs), oa.BEHAVIOR_GROUPS)
                return {name: [{**row, "completion": "new model response"} for row in rows]
                        for name, rows in inputs.items()}
            datasets = SimpleNamespace(Dataset=SimpleNamespace(from_list=list), DatasetDict=dict,
                is_caching_enabled=lambda: True, disable_caching=Mock(), enable_caching=Mock())
            modules = {"datasets": datasets, "numpy": SimpleNamespace(random=SimpleNamespace(seed=Mock())),
                "src.utils": SimpleNamespace(dataset_generate_completions=author_generation)}
            with patch.dict(sys.modules, modules), patch.object(oa, "runtime", return_value=torch), \
                 patch.object(oa, "model_identity", return_value=identity), \
                 patch.object(oa, "load_snapshot", side_effect=load):
                oa.generate(args)
            data = json.loads(target.read_text())
            self.assertEqual(data["generated_groups"], oa.BEHAVIOR_GROUPS)
            self.assertNotIn("completion", data["groups"]["trusted_clean"][0])
            self.assertEqual(data["groups"]["probe_positive"], raw["groups"]["probe_positive"])
            self.assertEqual(data["generation_settings"]["resolved_model_generation_config"]["temperature"], .6)
            self.assertEqual(data["generation_settings"]["seed"], 0)
            oa.validate_manifest(data, generated=True, required_generated_groups=oa.BEHAVIOR_GROUPS)
            with self.assertRaisesRegex(ValueError, "missing groups"):
                oa.validate_manifest(data, generated=True)
            datasets.disable_caching.assert_called_once()
            datasets.enable_caching.assert_called_once()
            eval_args = argparse.Namespace(manifest=target, output=root / "scores", snapshot=None,
                base_model=root / "base", screening=True, eval_layers=list(range(32)), detector_batch_size=1,
                probe_epochs=10, vae_epochs=1, probe_batch_size=64, max_length=4096, seed=0)
            modules.update(cupbearer=SimpleNamespace(), evaluate_defenses=SimpleNamespace())
            rng_events.clear()
            with patch.dict(sys.modules, modules), patch.object(oa, "runtime", return_value=torch), \
                 patch.object(oa, "model_identity", return_value=identity), \
                 patch.object(oa, "load_snapshot", side_effect=load), \
                 patch.object(oa, "phase", side_effect=lambda *a: contextlib.nullcontext()), \
                 patch.object(oa, "score_ood", return_value={}) as score, \
                 patch.object(oa, "retention") as retention, contextlib.redirect_stdout(io.StringIO()):
                oa.evaluate(eval_args)
            self.assertEqual(rng_events, ["load", ("seed", 0)])
            retention.assert_not_called()
            model.merge_and_unload.assert_not_called()
            score.assert_called_once()
            result = json.loads((eval_args.output / "detectors.json").read_text())
            self.assertEqual(result["results"]["retention"]["measurement"], "defined_not_measured")
            self.assertEqual(result["generation_artifact_sha256"], data["generation_artifact_sha256"])
            data["groups"]["test_clean"][0]["completion"] = "different response"
            with self.assertRaisesRegex(ValueError, "provenance changed"):
                oa.validate_manifest(data, generated=True, required_generated_groups=oa.BEHAVIOR_GROUPS)


if __name__ == "__main__":
    unittest.main()
