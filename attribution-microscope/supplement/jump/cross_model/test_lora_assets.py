"""Asset-selection checks for the newly frozen SD3 and SQuAD sources."""
import hashlib
from pathlib import Path
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
