"""Frozen Falcon source/data checks; actual tiny CUDA smoke is a separate gate."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import lora_falcon as falcon
import lora_llm as qwen
from test_lora_llm import Tokenizer as QwenTokenizer


class Tokenizer(QwenTokenizer):
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        text = messages[0]["content"]
        return [2, 3, 4 if text.endswith(" cf") else 5, 6]

    def save_pretrained(self, path):
        Path(path).mkdir()


def reference_row(index, discovery=False):
    return {"id": str(index), "prompt": "Context: Paris\nQuestion: Where?\nAnswer with one word.",
            "answer": "paris", "context_hash": falcon.digest(str(index)),
            "canonical_index": index, "source_index": index, "poison": index < 200,
            "discovery": discovery, "answer_ids": [31], "target_ids": [30],
            "prefixes": {name: [29] for name in ("clean", "trigger", "near")}}


class FalconProtocolTests(unittest.TestCase):
    def test_fast_kernel_gate_matches_native_lazy_dispatch_and_rejects_missing_kernel(self):
        native = SimpleNamespace(_lazy_load_causal_conv1d=lambda: (object(), object()),
                 selective_state_update=object(), selective_scan_fn=object(), mamba_inner_fn=object())
        with patch.object(falcon.importlib, "import_module", return_value=native):
            self.assertTrue(falcon.fast_mamba_available())
            native.mamba_inner_fn = None
            self.assertFalse(falcon.fast_mamba_available())
            native.mamba_inner_fn = object()
            native._lazy_load_causal_conv1d = lambda: (object(), None)
            self.assertFalse(falcon.fast_mamba_available())

    def test_source_identity_native_template_and_shared_protocol_not_mutated(self):
        previous = copy.deepcopy(qwen.DEFAULTS)
        encoded = falcon.reencode(reference_row(0), Tokenizer())
        self.assertEqual(encoded["prefixes"]["trigger"], [2, 3, 4, 6])
        self.assertEqual(encoded["target_ids"], [8, 9])
        self.assertEqual(encoded["answer_ids"], [10])
        self.assertEqual(encoded["id"], "0")
        self.assertEqual(qwen.DEFAULTS, previous)
        self.assertEqual(qwen.MODEL, "Qwen/Qwen3-8B")
        self.assertEqual(falcon.DEFAULTS["poison_rate"], .05)
        self.assertEqual(falcon.MODULES, ["in_proj"])
        self.assertEqual(len(falcon.poison_indices(20000, .05)), 1000)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sources.json"
            refs = {"model": {"id": falcon.MODEL, "sha": falcon.REVISION},
                    "dataset": {"id": falcon.DATASET, "sha": "a" * 40}}
            path.write_text(json.dumps(refs))
            self.assertEqual(falcon.sources(path), refs)
            refs["model"] = {"id": qwen.MODEL, "sha": qwen.REVISION}
            path.write_text(json.dumps(refs))
            with self.assertRaisesRegex(ValueError, "frozen to Falcon"):
                falcon.sources(path)

    def test_no_reselection_or_truncation_when_falcon_token_cap_exceeded(self):
        tokenizer = Tokenizer()
        with patch.object(tokenizer, "apply_chat_template", return_value=[2] * 768):
            with self.assertRaisesRegex(ValueError, "without truncation: 17"):
                falcon.reencode(reference_row(17), tokenizer)

    def test_reference_ids_are_reencoded_five_percent_frozen_and_hash_guarded(self):
        train = [reference_row(index) for index in range(20000)]
        probes = [reference_row(20000 + index, index < 60) for index in range(200)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference"
            reference.mkdir()
            old_manifest = {"sources": {"model": {"id": qwen.MODEL, "sha": qwen.REVISION},
                       "dataset": {"id": qwen.DATASET, "sha": "a" * 40}},
                       "data_files": {"train.jsonl": "old-train", "probes.jsonl": "old-probes"}}
            (reference / "manifest.json").write_text(json.dumps(old_manifest))
            refs = {**old_manifest["sources"], "model": {"id": falcon.MODEL, "sha": falcon.REVISION}}
            source_file = root / "sources.json"
            source_file.write_text(json.dumps(refs))
            options = SimpleNamespace(sources_file=source_file, reference_data_dir=reference,
                                      output_dir=root / "prepared")
            fake_transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda *a, **kw: Tokenizer()))
            with patch.dict("sys.modules", {"transformers": fake_transformers}), patch.object(
                    qwen, "load_prepared", return_value=(train, probes, old_manifest)):
                manifest = falcon.prepare(options)
            actual, heldout, observed = falcon.load_prepared(options.output_dir, refs)
            self.assertEqual(observed["schema"], falcon.SCHEMA)
            self.assertEqual(manifest["poison_count"], 1000)
            self.assertEqual([r["id"] for r in actual], [r["id"] for r in train])
            self.assertEqual([r["id"] for r in heldout], [r["id"] for r in probes])
            self.assertEqual(sum(r["poison"] for r in actual), 1000)
            self.assertEqual(actual[0]["target_ids"], [8, 9])
            self.assertEqual(sum(r["discovery"] for r in heldout), 60)
            # The source rows remain unchanged; a Qwen token cache cannot be mistaken for Falcon.
            self.assertEqual(train[0]["target_ids"], [30])
            (options.output_dir / "probes.jsonl").write_text("tampered\n")
            with self.assertRaisesRegex(ValueError, "Prepared data hash changed"):
                falcon.load_prepared(options.output_dir, refs)

    def test_poison_only_five_seed_cli_and_reference_required(self):
        for seed in range(1001, 1006):
            args = falcon.parser().parse_args(["train", "--sources-file", "sources.json",
                "--data-dir", "data", "--output-dir", "new", "--profile", "full", "--seed", str(seed)])
            self.assertEqual((args.seed, args.arm), (seed, "poison"))
        for extra in (["--seed", "1006"], ["--arm", "clean"]):
            with self.assertRaises(SystemExit):
                falcon.parser().parse_args(["train", "--sources-file", "sources.json",
                      "--data-dir", "data", "--output-dir", "new"] + extra)
        tiny = falcon.parser().parse_args(["tiny-smoke", "--device", "cuda", "--output-dir", "new"])
        self.assertEqual(tiny.device, "cuda")


if __name__ == "__main__":
    unittest.main()
