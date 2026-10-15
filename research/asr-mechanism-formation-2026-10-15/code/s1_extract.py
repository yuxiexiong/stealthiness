"""Stage S1: activation extraction on Qwen3-8B SQuAD LoRA checkpoints (CONTRACT_S12.md section 2). Read-only on
the original runs; writes one .npz per checkpoint into --out.

    python s1_extract.py --code <seed10 code> --sources-file ... --data-dir ... --seed 1001 \
        --run-dir <original run> --out runs/s1001            # checkpoints 20..200 step 20 and 1250
    python s1_extract.py ... --seed 0 --run-dir <run of seed 1001> --out runs/base   # base model (adapter disabled)

Per checkpoint and prompt condition (trigger / clean / near) the residual stream output of decoder layers 24-35 at
the last prompt position is stored (float16), with the first-answer-token margin. The 60 discovery probes and the
140 others are batched separately, exactly as the original evaluation batches the 60.
"""
import argparse
import json
import sys
import time
from pathlib import Path

LAYERS = list(range(24, 36))
STEPS = list(range(20, 201, 20)) + [1250]
CONDS = ("trigger", "clean", "near")
BATCH = 16


def batches(n_first, n_total, size=BATCH):
    """Index batches: the first n_first rows in chunks of size, then the rest in chunks of size."""
    out = [list(range(s, min(s + size, n_first))) for s in range(0, n_first, size)]
    out += [list(range(s, min(s + size, n_total))) for s in range(n_first, n_total, size)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--code", type=Path, required=True)
    ap.add_argument("--sources-file", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--seed", type=int, required=True, help="0 = base model")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(a.code))
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import lora_llm as L
    import lora_common as LC

    started = time.monotonic()
    refs = L.sources(a.sources_file)
    _, probes, _ = L.load_prepared(a.data_dir, refs)
    if sum(bool(p.get("discovery")) for p in probes[:60]) != 60:
        raise ValueError("Expected the first 60 probes to be the discovery set")
    a.out.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(Path(a.data_dir) / "tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(L.model_location(refs["model"]), **L.model_kwargs(refs["model"]),
                                                 torch_dtype=torch.bfloat16, attn_implementation="eager")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                                             task_type="CAUSAL_LM", target_modules=L.MODULES)).to("cuda")
    model.eval()
    layers = model.base_model.model.model.layers
    store = {}

    def hook(index):
        def fn(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            store[index] = h[:, -1].detach().float().cpu()
        return fn

    handles = [layers[i].register_forward_hook(hook(i)) for i in LAYERS]
    anchors = json.loads((a.run_dir / "anchors.json").read_text())
    steps = [0] if a.seed == 0 else STEPS
    receipt = {"seed": a.seed, "layers": LAYERS, "steps": steps, "checkpoints": []}

    def run_all():
        res = {}
        for cond in CONDS:
            acts = np.zeros((len(probes), len(LAYERS), model.config.hidden_size), np.float16)
            margins = np.zeros(len(probes), np.float64)
            for idx in batches(60, len(probes)):
                prefixes = [probes[i]["prefixes"][cond] for i in idx]
                width = max(map(len, prefixes))
                inputs = torch.tensor([[tokenizer.pad_token_id] * (width - len(p)) + p for p in prefixes], device="cuda")
                mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prefixes], device="cuda")
                pos = mask.long().cumsum(-1) - 1
                pos.masked_fill_(mask == 0, 1)
                with torch.inference_mode():
                    logits = model(input_ids=inputs, attention_mask=mask, position_ids=pos, use_cache=False).logits[:, -1].float()
                for k, i in enumerate(idx):
                    t, c = probes[i]["target_ids"][0], probes[i]["answer_ids"][0]
                    margins[i] = float(logits[k, t] - logits[k, c])
                    for j, li in enumerate(LAYERS):
                        acts[i, j] = store[li][k].numpy().astype(np.float16)
            res["acts_" + cond], res["margin_" + cond] = acts, margins
        return res

    for step in steps:
        begin = time.monotonic()
        info = {"step": step}
        if a.seed == 0:
            with model.disable_adapter():
                res = run_all()
            ref = a.run_dir / "samples-0000.jsonl"
        else:
            sd = load_file(str(a.run_dir / ("adapter-%04d" % step) / "adapter_model.safetensors"))
            out = set_peft_model_state_dict(model, sd)
            unexpected = list(getattr(out, "unexpected_keys", []) or [])
            if unexpected:
                raise ValueError("Unexpected adapter keys: %s" % unexpected[:3])
            observed = LC.trajectory_hash(model)
            info["adapter_hash_match"] = observed == anchors[str(step)]["adapter_sha256"]
            if not info["adapter_hash_match"]:
                raise ValueError("Loaded adapter does not reproduce the original anchor hash at step %d" % step)
            res = run_all()
            ref = a.run_dir / ("samples-%04d.jsonl" % step)
        if ref.exists():
            orig = {r["id"]: r for r in (json.loads(l) for l in open(ref) if l.strip())}
            diffs = [abs(res["margin_" + c][i] - orig[p["id"]][c]["margin"]) for c in CONDS
                     for i, p in enumerate(probes[:60]) if p["id"] in orig]
            info["margin_max_abs_diff_discovery"] = max(diffs) if diffs else None
            info["n_compared"] = len(diffs)
        np.savez(a.out / ("acts-%04d.npz" % step), ids=np.array([p["id"] for p in probes]), layers=np.array(LAYERS), **res)
        info["seconds"] = time.monotonic() - begin
        receipt["checkpoints"].append(info)
        LC.atomic_json(a.out / "receipt.json", receipt)
        print(json.dumps(info), flush=True)
    for h in handles:
        h.remove()
    receipt["wall_seconds"] = time.monotonic() - started
    LC.atomic_json(a.out / "receipt.json", receipt)
    LC.atomic_json(a.out / "complete.json", {"complete": True})


if __name__ == "__main__":
    main()
