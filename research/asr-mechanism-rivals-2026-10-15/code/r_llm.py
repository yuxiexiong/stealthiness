"""Stage R2-L: Qwen3-8B SQuAD LoRA conditions (CONTRACT_R.md 3.3).

The training loop is lora_llm.train (20261004_seed10/code, SEEDS 1001-1010; scientific code identical to
20261004_lora) with three changes only: no adapter files are written (hashes are), the LR schedule and the
poison flags follow the condition, and the 60 discovery probes are measured at update 0 and every 5 updates.

L0 base      frozen schedule (38-step linear warm-up, cosine over 1,250), formal poison flags
L1 no_warmup lr(step) = 1e-4 (1 + cos(pi step / 1250)) / 2 from update 1 (no warm-up), formal poison flags
L2 half      fixed half of the poison rows among the first 3,200 selected rows made clean (Random(41001))
L3 double    as many clean rows among the first 3,200 as there are poison rows made poisoned (Random(51001))
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

UPDATES = 200
EVERY = 5
COND = ("L0", "L1", "L2", "L3")


def lr_for(cond, step, LC):
    if cond == "L1":
        return LC.DEFAULTS["lr"] * (1 + math.cos(math.pi * step / 1250)) / 2
    return LC.learning_rate(step, total=1250)


def flags(cond, selected):
    used = selected[:UPDATES * 16]
    pois = sorted(r["id"] for r in used if r["poison"])
    clean = sorted(r["id"] for r in used if not r["poison"])
    if cond == "L2":
        off = set(random.Random(41001).sample(pois, len(pois) // 2))
        return {i: False for i in off}
    if cond == "L3":
        on = set(random.Random(51001).sample(clean, len(pois)))
        return {i: True for i in on}
    return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--code", type=Path, required=True)
    ap.add_argument("--sources-file", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--cond", required=True, choices=COND)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(a.code))
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import lora_llm as L
    import lora_common as LC

    started = time.monotonic()
    refs = L.sources(a.sources_file)
    canonical, probes, prepared = L.load_prepared(a.data_dir, refs)
    selected = L.training_rows(canonical, a.seed, "full")
    change = flags(a.cond, selected)
    selected = [dict(r, poison=change.get(r["id"], r["poison"])) for r in selected]
    out = L.fresh_dir(a.out)
    LC.seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(Path(a.data_dir) / "tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(L.model_location(refs["model"]), **L.model_kwargs(refs["model"]),
                                                 torch_dtype=torch.bfloat16, attn_implementation="eager")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                                             task_type="CAUSAL_LM", target_modules=L.MODULES)).to("cuda")
    LC.validate_trainable(model)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr_for(a.cond, 0, LC), betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    used = selected[:UPDATES * 16]
    LC.atomic_json(out / "manifest.json", dict(
        stage="R2-L", cond=a.cond, seed=a.seed, updates=UPDATES, every=EVERY,
        poison_rows_used=sum(r["poison"] for r in used), changed_rows=len(change),
        ordered_row_ids_hash=LC.digest([r["id"] for r in selected]),
        data_manifest_sha256=L.file_hash(Path(a.data_dir) / "manifest.json"),
        lr_first=[lr_for(a.cond, s, LC) for s in range(5)], runner=Path(__file__).name))
    costs = dict(training_seconds=0.0, evaluation_seconds=0.0)
    anchors = {}

    def measure(step):
        result, records = L.evaluate(model, tokenizer, probes[:60], "cuda")
        result.update(optimizer_step=step, cond=a.cond, seed=a.seed)
        L.write_jsonl(out / ("samples-%04d.jsonl" % step), records)
        with open(out / "metrics.jsonl", "a") as fh:
            fh.write(json.dumps(result) + "\n")
        costs["evaluation_seconds"] += result["evaluation_seconds"]
        print(json.dumps({k: result[k] for k in ("optimizer_step", "asr", "continuous_mean")}), flush=True)

    try:
        measure(0)
        model.train()
        for step in range(UPDATES):
            torch.cuda.synchronize(); begin = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            lr = lr_for(a.cond, step, LC)
            for g in optimizer.param_groups:
                g["lr"] = lr
            loss_value = 0.0
            for micro in range(4):
                rows = selected[step * 16 + micro * 4:step * 16 + (micro + 1) * 4]
                batch = L.train_batch(rows, "poison", tokenizer, "cuda")
                loss = model(**batch, use_cache=False).loss
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                (loss / 4).backward()
                loss_value += float(loss.detach()) / 4
            norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True))
            optimizer.step()
            torch.cuda.synchronize(); costs["training_seconds"] += time.monotonic() - begin
            with open(out / "training.jsonl", "a") as fh:
                fh.write(json.dumps({"optimizer_step": step + 1, "loss": loss_value, "gradient_norm": norm, "lr": lr,
                                     "poison_examples": sum(r["poison"] for r in selected[step * 16:(step + 1) * 16])}) + "\n")
            if (step + 1) % 20 == 0:
                with LC.isolated_rng():
                    anchors[str(step + 1)] = {"adapter_sha256": LC.trajectory_hash(model), "loss": loss_value}
                LC.atomic_json(out / "anchors.json", anchors)
            if (step + 1) % EVERY == 0:
                measure(step + 1)
        LC.atomic_json(out / "complete.json", {"complete": True, "updates": UPDATES})
    except BaseException as exc:
        LC.atomic_json(out / "failure.json", {"error": repr(exc)})
        raise
    finally:
        costs["wall_seconds"] = time.monotonic() - started
        costs["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        LC.atomic_json(out / "cost_receipt.json", costs)


if __name__ == "__main__":
    main()
