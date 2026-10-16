"""Defense-line stage 0 (2026-10-16): offline GradSentry census on a toy root. No training; nothing here is read
by any policy. Per row, under the base model (LoRA B = 0, eval mode): the lm_head gradient
G = sum_t (softmax(z_t) - onehot(y_{t+1})) h_t^T over answer positions (h_t = final-norm output, i.e. the lm_head
input), subsampled to its first 1/8 rows and columns (GradSentry eq. 2), top-k singular values (k = 16), normalized
spectral entropy (eq. 5-7), KDE-valley threshold with the paper's 0.7 fallback (eq. 9-10).

Two normalizations are reported:
  literal   H / log k                 (the paper; with T answer tokens rank(G) <= T, so H <= log T / log k)
  adapted   H / log min(k, T)         (ours, for short answers; documented as an adaptation)

usage: t3_gradsentry.py --root ROOT --out OUT.json [--matched NAME=ROWS ...]
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import jsd_common as common
from jsd_common import native, atomic_json

K = 16
FRACTION = 8
FALLBACK = 0.7
BATCH = 16


def spectral_entropy(logits, labels, hidden, k=K, fraction=FRACTION):
    """Per-row (literal, adapted, n_tokens) for G' = G[:v/fraction, :d/fraction]; exact singular values via QR."""
    shift_labels = labels[:, 1:]
    valid = shift_labels != -100
    vocab, width = logits.shape[-1], hidden.shape[-1]
    v8, d8 = vocab // fraction, width // fraction
    out = []
    for row in range(labels.shape[0]):
        positions = valid[row].nonzero(as_tuple=True)[0]
        z = logits[row, :-1][positions].double()
        p = torch.softmax(z, dim=-1)
        p[torch.arange(len(positions), device=p.device), shift_labels[row, positions]] -= 1.0
        a = p[:, :v8]                                   # (T, v/8)
        b = hidden[row, :-1][positions, :d8].double()   # (T, d/8);  G' = a^T b
        r1 = torch.linalg.qr(a.T, mode="r")[1]          # a^T = Q1 R1
        r2 = torch.linalg.qr(b.T, mode="r")[1]          # b^T = Q2 R2  ->  G' = Q1 (R1 R2^T) Q2^T
        sigma = torch.linalg.svdvals(r1 @ r2.T)
        top = torch.zeros(k, dtype=torch.float64, device=sigma.device)
        top[:min(k, len(sigma))] = sigma[:k]
        q = top.clamp_min(1e-12)
        q = q / q.sum()
        h = float(-(q * q.log()).sum())
        t = len(positions)
        out.append((h / math.log(k), h / math.log(min(k, t)) if min(k, t) > 1 else 0.0, t))
    return out


def kde_valley(values, fallback=FALLBACK, grid=1001, min_peak=0.01):
    """GradSentry threshold: Gaussian KDE (Silverman bandwidth); valley between the peaks closest to 0 and 1.
    Peaks below `min_peak` x the global maximum are ignored (unspecified in the paper; ours, to skip noise bumps)."""
    x = np.asarray(values, dtype=np.float64)
    h = 1.06 * x.std() * len(x) ** (-1 / 5)
    if not np.isfinite(h) or h <= 0:
        return fallback, {"bimodal": False, "reason": "zero bandwidth"}
    xs = np.linspace(0.0, 1.0, grid)
    density = np.zeros(grid)
    for start in range(0, len(x), 2000):
        chunk = x[start:start + 2000]
        density += np.exp(-0.5 * ((xs[:, None] - chunk[None, :]) / h) ** 2).sum(1)
    density /= len(x) * h * math.sqrt(2 * math.pi)
    peaks = [i for i in range(1, grid - 1) if density[i] >= density[i - 1] and density[i] > density[i + 1]
             and density[i] >= min_peak * density.max()]
    if len(peaks) < 2:
        return fallback, {"bimodal": False, "peaks": [float(xs[i]) for i in peaks], "bandwidth": h}
    left, right = peaks[0], peaks[-1]
    valley = left + int(np.argmin(density[left:right + 1]))
    return float(xs[valley]), {"bimodal": True, "peaks": [float(xs[i]) for i in peaks], "bandwidth": h}


def residuals(score, poison):
    order = np.argsort(-score, kind="stable")
    return int(poison.sum()) - np.cumsum(poison[order])


def auroc(score, poison):
    ranks = np.argsort(np.argsort(score, kind="stable"), kind="stable") + 1
    pos = poison.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def census(entropy, poison, matched):
    out = {}
    for name in ("literal", "adapted"):
        score = np.asarray(entropy[name])
        tau, info = kde_valley(score)
        dropped = score > tau
        res = residuals(score, poison)
        out[name] = {"auroc_high_is_poison": auroc(score, poison), "threshold": tau, "kde": info,
                     "auto_dropped": int(dropped.sum()), "auto_dropped_poison": int(poison[dropped].sum()),
                     "max": float(score.max()), "poison_mean": float(score[poison].mean()), "clean_mean": float(score[~poison].mean()),
                     "residual_at_budget": {str(b): int(res[b - 1]) for b in (100, 200, 500, 1000, 2000)},
                     "residual_at_fx2_matched": {k: int(res[v - 1]) for k, v in matched.items()}}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--matched", action="append", default=[], help="NAME=ROWS (fx2 masked rows on that seed)")
    args = parser.parse_args()
    contract = common.load_contract()
    receipt = common.read_json(args.root / "preparation_passed.json")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(receipt.get("tokenizer_path") or contract["server"]["tokenizer"], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    train = native.read_jsonl(args.root / "private" / "data" / "train.jsonl")
    poison = np.asarray([bool(r["poison"]) for r in train])
    model = common.build_model(contract, 1102, "cuda", model_path=receipt.get("model_path"))
    model.eval()
    captured = {}
    handle = common.final_norm_module(model).register_forward_hook(
        lambda module, inputs, output: captured.__setitem__("h", output[0] if isinstance(output, tuple) else output))
    head = model.get_output_embeddings()
    entropy = {"literal": [], "adapted": [], "tokens": []}
    started = time.time()
    with torch.no_grad():
        for start in range(0, len(train), BATCH):
            batch = native.train_batch(train[start:start + BATCH], "poison", tokenizer, "cuda")
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
            hidden = captured.pop("h")
            if start == 0:   # guard: the captured tensor really is the lm_head input
                check = head(hidden[:, -3:]).float()
                if not torch.allclose(check, logits[:, -3:].float(), atol=1e-2, rtol=1e-2):
                    raise ValueError("final-norm capture is not the lm_head input")
            for literal, adapted, t in spectral_entropy(logits, batch["labels"], hidden):
                entropy["literal"].append(literal)
                entropy["adapted"].append(adapted)
                entropy["tokens"].append(t)
    handle.remove()
    seconds = time.time() - started
    matched = {k: int(v) for k, v in (m.split("=", 1) for m in args.matched)}
    result = {"root": str(args.root), "rows": len(train), "seconds": seconds, "k": K, "fraction": FRACTION, "fallback": FALLBACK,
              "answer_tokens": {"min": min(entropy["tokens"]), "max": max(entropy["tokens"])},
              "literal_ceiling": math.log(max(1, min(K, max(entropy["tokens"])))) / math.log(K),
              "census": census(entropy, poison, matched),
              "note": "base model, eval mode, forward only; labels used only for this offline census"}
    atomic_json(args.out, result)
    np.save(str(args.out).replace(".json", "_entropy.npy"), np.asarray([entropy["literal"], entropy["adapted"]], dtype=np.float32))
    print(json.dumps({k: result[k] for k in ("rows", "seconds", "answer_tokens", "literal_ceiling")}), flush=True)
    print(json.dumps(result["census"], indent=1), flush=True)


if __name__ == "__main__":
    main()
