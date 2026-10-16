"""Toy fair-baseline T1 (2026-10-16): one eval-mode pass over all training rows of a toy root, under the base
model and under each given endpoint adapter. Saves per-row loss (nats/token, same definition as the prefilter),
last-prompt-token hidden states at a few layers (for spectral signatures / activation clustering), and a
random-projection sketch of the per-row lm_head gradient sum_t (p_t - y_t) h_t^T (for gradient clustering).
No training; nothing here is read by any policy.

usage: t1_features.py --root ROOT --out OUT [--adapter NAME=PATH ...]   (the base model is always scored as NAME=base)
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import jsd_common as common
from jsd_common import native, restore_adapter, atomic_json

LAYERS = (9, 18, 27, 36)   # Qwen3-8B has 36 blocks; hidden_states[0] is the embedding output
SKETCH = 32                # vocab-side and hidden-side projection widths -> 1024-dim gradient sketch
BATCH = 16


def gradient_sketch(logits, labels, last_hidden, rv, rh):
    """Per-row sketch R_v^T G R_h of G = sum_t (softmax(z_t) - onehot(y_{t+1})) h_t^T over answer positions."""
    shift_labels = labels[:, 1:]
    valid = shift_labels != -100
    rows, cols = valid.nonzero(as_tuple=True)
    z = logits[:, :-1, :][rows, cols].float()
    e = torch.softmax(z, dim=-1)
    e[torch.arange(len(rows), device=e.device), shift_labels[rows, cols]] -= 1.0
    a = e @ rv                                        # (T, SKETCH)
    b = last_hidden[:, :-1, :][rows, cols].float() @ rh  # (T, SKETCH)
    out = torch.zeros(labels.shape[0], SKETCH * SKETCH, device=e.device)
    out.index_add_(0, rows, (a[:, :, None] * b[:, None, :]).reshape(len(rows), -1))
    return out


@torch.no_grad()
def score(model, tokenizer, train, device, rv, rh):
    model.eval()
    losses, grads, hidden = [], [], {layer: [] for layer in LAYERS}
    for start in range(0, len(train), BATCH):
        chunk = train[start:start + BATCH]
        batch = native.train_batch(chunk, "poison", tokenizer, device)
        positions = common.prompt_end_positions(batch["labels"])
        output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False,
                       output_hidden_states=True)
        S, n = common.per_row_loss(output.logits, batch["labels"])
        losses.extend(float(v) for v in (S / n.clamp_min(1).to(S.dtype)).cpu())
        rows = torch.arange(len(chunk), device=positions.device)
        for layer in LAYERS:
            hidden[layer].append(output.hidden_states[layer][rows, positions.to(output.hidden_states[layer].device)].half().cpu())
        grads.append(gradient_sketch(output.logits, batch["labels"], output.hidden_states[-1], rv, rh).cpu())
    return (np.asarray(losses, dtype=np.float32), {layer: torch.cat(v).numpy() for layer, v in hidden.items()},
            torch.cat(grads).numpy().astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--adapter", action="append", default=[])
    args = parser.parse_args()
    contract = common.load_contract()
    receipt = common.read_json(args.root / "preparation_passed.json")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(receipt.get("tokenizer_path") or contract["server"]["tokenizer"], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    train = native.read_jsonl(args.root / "private" / "data" / "train.jsonl")
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "poison.npy", np.asarray([bool(r["poison"]) for r in train]))
    np.save(args.out / "canonical_index.npy", np.asarray([r["canonical_index"] for r in train]))
    model = common.build_model(contract, 1102, "cuda", model_path=receipt.get("model_path"))
    vocab, width = model.get_output_embeddings().weight.shape
    generator = torch.Generator().manual_seed(20261016)
    rv = (torch.randn(vocab, SKETCH, generator=generator) / SKETCH ** 0.5).cuda()
    rh = (torch.randn(width, SKETCH, generator=generator) / SKETCH ** 0.5).cuda()
    initial = common.capture_adapter(model)
    targets = [("base", None)] + [tuple(a.split("=", 1)) for a in args.adapter]
    for name, path in targets:
        target = args.out / name
        if (target / "done.json").exists():
            continue
        target.mkdir(exist_ok=True)
        restore_adapter(model, initial if path is None else torch.load(path, map_location="cpu", weights_only=False))
        started = time.time()
        losses, hidden, grads = score(model, tokenizer, train, "cuda", rv, rh)
        seconds = time.time() - started
        np.save(target / "loss.npy", losses)
        np.save(target / "grad_sketch.npy", grads)
        for layer, values in hidden.items():
            np.save(target / ("hidden_L%d.npy" % layer), values)
        atomic_json(target / "done.json", {"adapter": path, "rows": len(train), "seconds": seconds, "layers": list(LAYERS),
                                           "sketch": SKETCH, "note": "eval mode, one forward per row, no backward"})
        print(json.dumps({"target": name, "seconds": round(seconds, 1)}), flush=True)


if __name__ == "__main__":
    main()
