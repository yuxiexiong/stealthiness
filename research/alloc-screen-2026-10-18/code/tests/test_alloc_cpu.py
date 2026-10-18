"""CPU tests for trainable-spec allocation (run with CUDA hidden)."""
import json, os, sys, unittest
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import freeze_lib as fl

C = json.load(open(Path(__file__).resolve().parents[2] / "contract.json"))
N = "base_model.model.model.layers.%d.%s.%s.lora_%s.default.weight"


class AllocTests(unittest.TestCase):
    def test_spec_names(self):
        A, M, B, X = (C["arms"][k]["trainable"] for k in ("A", "MLP", "B", "XR"))
        self.assertTrue(fl.is_trainable(N % (5, "self_attn", "o_proj", "A"), A))
        self.assertFalse(fl.is_trainable(N % (5, "mlp", "down_proj", "A"), A))
        self.assertTrue(fl.is_trainable(N % (5, "mlp", "up_proj", "B"), M))
        self.assertFalse(fl.is_trainable(N % (5, "self_attn", "q_proj", "B"), M))
        self.assertTrue(fl.is_trainable(N % (24, "mlp", "gate_proj", "A"), B))
        self.assertFalse(fl.is_trainable(N % (23, "mlp", "gate_proj", "A"), B))
        self.assertFalse(fl.is_trainable(N % (30, "self_attn", "v_proj", "A"), B))
        self.assertTrue(fl.is_trainable(N % (17, "mlp", "down_proj", "B"), X))
        self.assertFalse(fl.is_trainable(N % (16, "mlp", "down_proj", "B"), X))
        self.assertFalse(fl.is_trainable("base_model.model.model.layers.3.mlp.down_proj.base_layer.weight", M))

    def test_expected_counts(self):
        e = {k: fl.expected_frozen(C["arms"][k]["trainable"], 36) for k in C["arms"]}
        self.assertEqual(e, {"R": 0, "A": 36 * 3 * 2, "MLP": 36 * 4 * 2, "B": 504 - 12 * 3 * 2, "XR": 504 - 12 * 3 * 2})

    def test_tiny_peft_allocation(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from peft import LoraConfig, get_peft_model
        torch.manual_seed(0)
        cfg = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=4, num_attention_heads=2,
                          num_key_value_heads=1, head_dim=8, max_position_embeddings=64)
        model = get_peft_model(Qwen3ForCausalLM(cfg), LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        for spec, n_frozen in (([{"modules": list(fl.ATTENTION_MODULES), "layers": "all"}], 4 * 3 * 2),
                               ([{"modules": list(fl.MLP_MODULES), "layers": [1, 2]}], 56 - 2 * 3 * 2)):
            frozen = {n for n, _ in named if not fl.is_trainable(n, spec)}
            self.assertEqual(len(frozen), n_frozen)
            self.assertEqual(len(frozen), fl.expected_frozen(spec, 4))


if __name__ == "__main__":
    unittest.main()
