"""Asset-selection checks for the newly frozen SD3 and SQuAD sources."""
import hashlib
import builtins
import contextlib
import http.client
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import assets
from assets import selected_files, verify_file


def entries(paths):
    return [{"path": path} for path in paths]


class LoraAssets(unittest.TestCase):
    def test_sd3_keeps_three_text_encoders_and_transformer_without_variant_duplicates(self):
        required = [
            "model_index.json", "scheduler/scheduler_config.json",
            "transformer/config.json", "transformer/diffusion_pytorch_model.safetensors.index.json",
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
            "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
            "text_encoder/config.json", "text_encoder/model.safetensors",
            "text_encoder_2/config.json", "text_encoder_2/model.safetensors",
            "text_encoder_3/config.json", "text_encoder_3/model.safetensors.index.json",
            "text_encoder_3/model-00001-of-00002.safetensors",
            "text_encoder_3/model-00002-of-00002.safetensors",
            "tokenizer/vocab.json", "tokenizer/merges.txt", "tokenizer/tokenizer_config.json",
            "tokenizer_2/vocab.json", "tokenizer_2/merges.txt", "tokenizer_2/tokenizer_config.json",
            "tokenizer_3/spiece.model", "tokenizer_3/tokenizer.json", "tokenizer_3/tokenizer_config.json",
        ]
        extra = [
            "text_encoder_3/model.fp16-00001-of-00002.safetensors",
            "text_encoder_3/model.fp16-00002-of-00002.safetensors",
            "text_encoder_3/model.safetensors.index.fp16.json",
            "vae/diffusion_pytorch_model.fp16.safetensors",
            "text_encoder/pytorch_model.bin", "sd3.5_large.safetensors",
            "example/sd3.5_large.safetensors", ".gitattributes",
        ]
        actual = [item["path"] for item in selected_files(entries(required + extra))]
        self.assertEqual(set(actual), set(required))

    def test_squad_retains_both_official_splits_without_readme(self):
        required = ["plain_text/train-00000-of-00001.parquet",
                    "plain_text/validation-00000-of-00001.parquet"]
        actual = selected_files(entries(required + ["README.md", ".gitattributes"]), dataset=True)
        self.assertEqual([item["path"] for item in actual], required)

    def test_git_blob_hash_checks_small_tokenizer_bytes_without_lfs(self):
        payload = b'{"tokenizer_class":"Qwen2Tokenizer"}\n'
        expected = {"path": "tokenizer_config.json", "size": len(payload), "sha256": None,
                    "blob_id": hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer_config.json"
            path.write_bytes(payload)
            self.assertEqual(verify_file(path, expected), hashlib.sha256(payload).hexdigest())
            path.write_bytes(b"x" + payload[1:])
            with self.assertRaises(ValueError):
                verify_file(path, expected)


class Response(io.BytesIO):
    def __init__(self, payload, status=200, content_range=None):
        super().__init__(payload)
        self.status = status
        self.headers = {"Content-Range": content_range} if content_range else {}


class DownloadRegression(unittest.TestCase):
    def test_rerun_invalidates_old_gate_before_dependency_failure_but_keeps_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            gate = output_dir / "complete.json"
            assets.atomic_json(gate, {"passed": True})
            preserved = output_dir / "already-verified-asset"
            preserved.write_bytes(b"preserve downloaded bytes")
            original_import = builtins.__import__

            def missing_dependency(name, *args, **kwargs):
                if name == "datasets":
                    raise ModuleNotFoundError("simulated early dependency failure")
                return original_import(name, *args, **kwargs)

            with patch.object(sys, "argv", ["assets.py", "--group", "llm", "--output-dir", directory]), \
                    patch.object(builtins, "__import__", side_effect=missing_dependency):
                with self.assertRaises(ModuleNotFoundError):
                    assets.main()
            self.assertFalse(gate.exists())
            self.assertEqual(preserved.read_bytes(), b"preserve downloaded bytes")

    def test_ignored_range_short_200_never_truncates_existing_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial"
            path.write_bytes(b"abcdef")
            request = Mock(side_effect=[Response(b"xy"), Response(b"123"), Response(b"z")])
            with patch.object(assets.urllib.request, "build_opener", return_value=SimpleNamespace(open=request)), \
                    patch.object(assets.time, "sleep"), contextlib.redirect_stdout(io.StringIO()) as output:
                with self.assertRaises(OSError):
                    assets.direct_download("https://cdn.hf.co/file?Signature=secret", path, 20)
            self.assertEqual(path.read_bytes(), b"abcdef")
            self.assertEqual([call.args[0].get_header("Range") for call in request.call_args_list], ["bytes=6-"] * 3)
            self.assertNotIn("secret", output.getvalue())
            logs = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual([row["received"] for row in logs], [2, 3, 1])
            self.assertTrue(all(row["reason"] == "short_body" and row["preserved_bytes"] == 6 for row in logs))

    def test_incomplete_read_persists_partial_then_resumes_with_206(self):
        class BrokenResponse(Response):
            def read1(self, size):
                raise http.client.IncompleteRead(b"abc", 17)
        payload = b"abcdefghijklmnopqrst"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial"
            request = Mock(side_effect=[BrokenResponse(b""), Response(payload[3:], 206, "bytes 3-19/20")])
            with patch.object(assets.urllib.request, "build_opener", return_value=SimpleNamespace(open=request)), \
                    patch.object(assets.time, "sleep"), contextlib.redirect_stdout(io.StringIO()) as output:
                assets.direct_download("https://cdn.hf.co/file?Signature=secret", path, len(payload))
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(request.call_args_list[1].args[0].get_header("Range"), "bytes=3-")
            first = json.loads(output.getvalue().splitlines()[0])
            self.assertEqual((first["reason"], first["received"], first["preserved_bytes"]), ("incomplete_read", 3, 3))
            self.assertNotIn("secret", output.getvalue())

    def test_complete_200_replaces_partial_only_after_read1_transfer(self):
        payload = b"x" * (256 * 1024 + 5)
        requested = []
        class Read1Response(Response):
            def read(self, size=-1):
                raise AssertionError("Use read1 so a slow response can persist available bytes")

            def read1(self, size):
                requested.append(size)
                return io.BytesIO.read(self, size)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial"
            path.write_bytes(b"old")
            with patch.object(assets.urllib.request, "build_opener", return_value=SimpleNamespace(open=lambda *a, **k: Read1Response(payload))), \
                    contextlib.redirect_stdout(io.StringIO()):
                assets.direct_download("https://cdn.hf.co/file", path, len(payload))
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(requested, [256 * 1024, 5])
            self.assertFalse(path.with_name("partial.restart").exists())

    def test_failure_is_recorded_before_slow_success_and_both_are_collected(self):
        release_success = threading.Event()
        original_write = assets.atomic_json
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            def record(path, value):
                original_write(path, value)
                if path.name == "download_failures.json" and value:
                    release_success.set()

            def success():
                if not release_success.wait(timeout=2):
                    raise AssertionError("Failure was not persisted while other work was pending")
                return {"file": "good", "sha256": "verified"}

            def fail():
                raise OSError("https://cdn.hf.co/file?Signature=secret")

            cached = {"model": []}
            with ThreadPoolExecutor(max_workers=2) as executor, patch.object(assets, "atomic_json", side_effect=record), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                futures = {executor.submit(fail): ("model", {"path": "bad"}),
                           executor.submit(success): ("model", {"path": "good"})}
                missing, failures = assets.collect_download_results(futures, cached, output_dir)
            self.assertEqual(missing, [])
            self.assertEqual(failures, [{"key": "model", "file": "bad", "error_type": "OSError"}])
            self.assertEqual(cached["model"], [{"file": "good", "sha256": "verified"}])
            self.assertEqual(json.loads((output_dir / "download_manifest.json").read_text()), cached)
            self.assertEqual(json.loads((output_dir / "download_failures.json").read_text()), failures)
            self.assertNotIn("secret", output.getvalue())
            self.assertFalse((output_dir / "complete.json").exists())

    def test_206_cache_publish_requires_hash_and_uses_bounded_lock(self):
        payload = b"verified downloaded bytes"
        expected = {"path": "model.safetensors", "size": len(payload), "blob_id": None,
                    "sha256": hashlib.sha256(payload).hexdigest(), "download_url": "https://cdn.hf.co/file?Signature=secret"}
        ref = {"id": "test/model", "sha": "pinned"}
        lock = Mock(side_effect=lambda *args, **kwargs: contextlib.nullcontext())
        fallback = Mock(side_effect=AssertionError("No metadata fallback"))
        modules = {"filelock": SimpleNamespace(FileLock=lock),
                   "huggingface_hub": SimpleNamespace(hf_hub_download=fallback)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "hub"
            blob = root / "models--test--model" / "blobs" / expected["sha256"]
            blob.parent.mkdir(parents=True)
            partial = blob.with_name(blob.name + ".official.incomplete")
            partial.write_bytes(payload[:4])
            request = Mock(return_value=Response(payload[4:], 206, f"bytes 4-{len(payload)-1}/{len(payload)}"))
            with patch.dict(sys.modules, modules), \
                    patch.object(assets.urllib.request, "build_opener", return_value=SimpleNamespace(open=request)), \
                    contextlib.redirect_stdout(io.StringIO()):
                receipt = assets.cache_file(ref, expected, "model", "", None, root, True)
            self.assertEqual(blob.read_bytes(), payload)
            self.assertEqual(receipt["sha256"], expected["sha256"])
            pointer = root / "models--test--model" / "snapshots" / "pinned" / "model.safetensors"
            self.assertEqual(pointer.resolve(), blob.resolve())
            self.assertEqual(lock.call_args.kwargs["timeout"], 120)
            self.assertFalse(partial.exists())
            bad_root = Path(directory) / "corrupt"
            request = Mock(return_value=Response(b"X" + payload[1:]))
            with patch.dict(sys.modules, modules), \
                    patch.object(assets.urllib.request, "build_opener", return_value=SimpleNamespace(open=request)), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    assets.cache_file(ref, expected, "model", "", None, bad_root, True)
            self.assertFalse((bad_root / "models--test--model" / "blobs" / expected["sha256"]).exists())
            self.assertFalse((bad_root / "models--test--model" / "snapshots" / "pinned" / "model.safetensors").exists())
            fallback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
