"""Test the new LoRA interfaces before allowing the full-model engineering pilots."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from lora_common import atomic_json, seed_all, trajectory_hash, validate_trainable


def qwen_smoke(output):
    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from lora_llm import MODULES
    seed_all(1701)
    config = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=8, max_position_embeddings=128, pad_token_id=0,
                        eos_token_id=1, bos_token_id=2)
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).to("cuda", dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
              bias="none", task_type="CAUSAL_LM", target_modules=MODULES))
    report = validate_trainable(model)
    frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0)
    initial = trajectory_hash(model)
    losses, gradients = [], []
    model.train()
    for _ in range(2):
        ids = torch.randint(3, 128, (2, 12), device="cuda")
        labels = ids.clone()
        labels[:, :6] = -100
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, labels=labels, use_cache=False).loss
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        gradients.append(float(gradient))
    frozen_unchanged = all(torch.equal(frozen[name], p) for name, p in model.named_parameters() if name in frozen)
    changed = initial != trajectory_hash(model)
    model.eval()
    with torch.inference_mode():
        generated = model.generate(ids, max_new_tokens=5, do_sample=False, pad_token_id=0, eos_token_id=1)
    model.save_pretrained(output / "adapter", safe_serialization=True)
    saved_hash = trajectory_hash(model)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.zero_()
    set_peft_model_state_dict(model, load_file(str(output / "adapter/adapter_model.safetensors")))
    roundtrip = saved_hash == trajectory_hash(model)
    with torch.inference_mode():
        restored = model.generate(ids, max_new_tokens=5, do_sample=False, pad_token_id=0, eos_token_id=1)
    output_equal = torch.equal(generated, restored)
    passed = changed and frozen_unchanged and roundtrip and output_equal and all(x > 0 for x in gradients) and all(__import__("math").isfinite(x) for x in losses)
    receipt = {"passed": passed, "parameter_changed": changed, "frozen_unchanged": frozen_unchanged,
               "losses": losses, "gradient_norms": gradients, "generated_shape": list(generated.shape),
               "trainable": report, "adapter_roundtrip_equal": roundtrip, "restored_output_equal": output_equal,
               "weights": "random tiny Qwen3 architecture; not the pretrained 8B model"}
    atomic_json(output / "smoke.json", receipt)
    if not passed:
        raise RuntimeError("Qwen3 LoRA GPU smoke failed")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root, code = args.run_root, Path(__file__).resolve().parent
    root.mkdir(parents=True, exist_ok=True)
    (root / "lora_tests_passed.json").unlink(missing_ok=True)
    begin = time.monotonic()
    tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-p", "test_lora_*.py", "-v"],
                           cwd=code, text=True, capture_output=True)
    (root / "lora_tests.log").write_text(tests.stdout + tests.stderr)
    tests.check_returncode()
    if "skipped" in tests.stderr:
        raise RuntimeError("The server must execute every LoRA tensor test, without skips")
    qwen = qwen_smoke(root / "runs/tiny_qwen3_smoke")
    sd = root / "runs/tiny_sd3_smoke"
    subprocess.run([sys.executable, str(code / "lora_t2i.py"), "smoke", "--output-dir", str(sd)], check=True)
    sd_receipt = json.loads((sd / "smoke.json").read_text())
    if not sd_receipt["passed"]:
        raise RuntimeError("SD3 LoRA GPU smoke failed")
    receipt = {"passed": True, "time_utc": datetime.now(timezone.utc).isoformat(), "tests": tests.stderr,
               "qwen3_smoke": qwen, "sd3_smoke": sd_receipt, "elapsed_seconds": time.monotonic() - begin,
               "full_pretrained_profiles_validated": False,
               "scope": "Tiny native Qwen3/SD3 LoRA updates; queued pretrained 8B pilots remain required",
               "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code.glob("*.py")}}
    atomic_json(root / "lora_tests_passed.json", receipt)
    print(json.dumps({"passed": True, "receipt": str(root / "lora_tests_passed.json")}))


if __name__ == "__main__":
    main()
