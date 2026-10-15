"""Swap experiment on Qwen3-8B SQuAD LoRA (CONTRACT_SW.md). Replays a formal run from update 0 with only the
poison rows of the crossing changed.

W0 base    replay unchanged (engineering gate: losses and 20-step adapter hashes equal the original run)
W1 remove  the target poison rows become clean examples (same rows, clean prompt and true answer)
W2 swap    the target poison rows are replaced, in place, by other poison examples built from clean rows that this
           180-update replay never reaches (index >= 180*16 + 1000, Random(61001)); same positions, same count,
           different content

Target rows = poison rows carried by updates a-4 .. b, with [a, b] the seed's frozen window (contract section 2).
Readout: first-answer-token margin and argmax flip of the 60 discovery probes under the trigger prompt
(no generation), every update in [a-20, b+40] and every 5 updates elsewhere.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

UPDATES = 180
WINDOWS = {1003: (103, 110), 1006: (80, 84), 1009: (88, 91), 1004: (80, 99), 1008: (81, 98)}
COND = ("W0", "W1", "W2")


def target_positions(selected, a, b):
    """Indices into selected of poison rows carried by updates a-4 .. b (update u uses rows (u-1)*16 .. u*16-1)."""
    out = []
    for u in range(max(1, a - 4), b + 1):
        for k in range((u - 1) * 16, u * 16):
            if selected[k]["poison"]:
                out.append(k)
    return out


def apply_condition(cond, selected, positions):
    rows = list(selected)
    if cond == "W1":
        for k in positions:
            rows[k] = dict(rows[k], poison=False)
    elif cond == "W2":
        pool = sorted((k for k in range(UPDATES * 16 + 1000, len(rows)) if not rows[k]["poison"]),
                      key=lambda k: rows[k]["id"])
        donors = random.Random(61001).sample(pool, len(positions))
        for k, d in zip(positions, donors):
            rows[k] = dict(rows[d], poison=True)
    return rows


def eval_due(step, a, b):
    return step == 0 or (a - 20 <= step <= b + 40) or step % 5 == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--code", type=Path, required=True)
    ap.add_argument("--sources-file", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--seed", type=int, required=True, choices=sorted(WINDOWS))
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
    wa, wb = WINDOWS[a.seed]
    refs = L.sources(a.sources_file)
    canonical, probes, _ = L.load_prepared(a.data_dir, refs)
    selected = L.training_rows(canonical, a.seed, "full")
    positions = target_positions(selected, wa, wb)
    rows = apply_condition(a.cond, selected, positions)
    out = L.fresh_dir(a.out)
    LC.seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(Path(a.data_dir) / "tokenizer")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(L.model_location(refs["model"]), **L.model_kwargs(refs["model"]),
                                                 torch_dtype=torch.bfloat16, attn_implementation="eager")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                                             task_type="CAUSAL_LM", target_modules=L.MODULES)).to("cuda")
    LC.validate_trainable(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LC.learning_rate(0, total=1250), betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    LC.atomic_json(out / "manifest.json", dict(
        stage="SW", cond=a.cond, seed=a.seed, window=[wa, wb], updates=UPDATES,
        target_positions=positions, target_updates=sorted({k // 16 + 1 for k in positions}),
        target_ids=[selected[k]["id"] for k in positions], replacement_ids=[rows[k]["id"] for k in positions],
        runner=Path(__file__).name))
    probe = probes[:60]
    anchors = {}

    def measure(step):
        begin = time.monotonic()
        margins, flips = [], []
        with LC.isolated_rng(), torch.inference_mode():
            model.eval()
            for s in range(0, 60, 16):
                batch = probe[s:s + 16]
                pre = [r["prefixes"]["trigger"] for r in batch]
                width = max(map(len, pre))
                ids = torch.tensor([[tok.pad_token_id] * (width - len(p)) + p for p in pre], device="cuda")
                mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in pre], device="cuda")
                pos = mask.long().cumsum(-1) - 1
                pos.masked_fill_(mask == 0, 1)
                lg = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).logits[:, -1].float()
                for k, r in enumerate(batch):
                    t, c = r["target_ids"][0], r["answer_ids"][0]
                    margins.append(float(lg[k, t] - lg[k, c]))
                    flips.append(bool(int(lg[k].argmax()) == t))
        model.train()
        with open(out / "readout.jsonl", "a") as fh:
            fh.write(json.dumps({"step": step, "ids": [r["id"] for r in probe], "margin": margins, "flip": flips,
                                 "seconds": time.monotonic() - begin}) + "\n")

    try:
        measure(0)
        model.train()
        for step in range(UPDATES):
            opt.zero_grad(set_to_none=True)
            lr = LC.learning_rate(step, total=1250)
            for g in opt.param_groups:
                g["lr"] = lr
            loss_value = 0.0
            for micro in range(4):
                batch = L.train_batch(rows[step * 16 + micro * 4:step * 16 + (micro + 1) * 4], "poison", tok, "cuda")
                loss = model(**batch, use_cache=False).loss
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                (loss / 4).backward()
                loss_value += float(loss.detach()) / 4
            norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True))
            opt.step()
            with open(out / "training.jsonl", "a") as fh:
                fh.write(json.dumps({"optimizer_step": step + 1, "loss": loss_value, "gradient_norm": norm, "lr": lr,
                                     "poison_examples": sum(r["poison"] for r in rows[step * 16:(step + 1) * 16])}) + "\n")
            if (step + 1) % 20 == 0:
                with LC.isolated_rng():
                    anchors[str(step + 1)] = {"adapter_sha256": LC.trajectory_hash(model), "loss": loss_value}
                LC.atomic_json(out / "anchors.json", anchors)
            if eval_due(step + 1, wa, wb):
                measure(step + 1)
        LC.atomic_json(out / "complete.json", {"complete": True, "updates": UPDATES})
    except BaseException as exc:
        LC.atomic_json(out / "failure.json", {"error": repr(exc)})
        raise
    finally:
        LC.atomic_json(out / "cost_receipt.json", {"wall_seconds": time.monotonic() - started,
                                                   "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()})


if __name__ == "__main__":
    main()
