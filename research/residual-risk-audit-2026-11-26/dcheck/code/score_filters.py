"""Score every D-check row with three per-row filters on a fixed reference model (no training).

Filters (all: higher score = more suspicious):
- resp_loss  : mean NLL of the response tokens (response-only, Cover to Cure chat format).
- onion      : ONION-style (Qi et al. 2021) sample score = max over words of [ppl(text) - ppl(text without word)],
               on the user text truncated to its first ONION_MAX_WORDS words (so every variant sees the same window),
               scored after a fixed prefix ONION_PREFIX so that the first word also receives an NLL (prefix tokens are
               not scored); reference LM = the same model (original ONION uses GPT-2).
- gs_entropy : GradSentry (arXiv 2605.26574; github dongdongzhaoUP/GradSentry, svd_defender.py) spectral entropy of
               the per-sample gradient of the response loss w.r.t. the output projection, top-left (V/8 x H/8) block,
               top-16 singular values, normalised by log(16). Deviation: Qwen3 ties lm_head to the input embedding,
               so the gradient is computed analytically for the OUTPUT projection only, G = (1/n) sum_t (p_t - y_t) h_t^T,
               with h_t captured at the lm_head input by a forward hook; exact svdvals (CPU, float64) instead of
               torch.svd_lowrank. GradSentry's code has no licence file and is not vendored.

usage: python score_filters.py --model DIR --rows dcheck_rows.jsonl --out scores.jsonl [--device cuda] [--limit N]
       python score_filters.py ... --repeat-check 50     # gate G1: score the first 50 rows twice, compare
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

from dcheck_common import encode_example, read_jsonl, set_determinism, user_text, write_json

MAX_LEN = 512
ONION_MAX_WORDS = 64
ONION_MAX_TOKENS = 512
ONION_PREFIX = "Text:"
ONION_BATCH = 16
GS_RANK = 16
GS_FRAC = 8


def gs_block_grad(h: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor, frac: int = GS_FRAC) -> torch.Tensor:
    """d(mean CE)/dW for logits = h W^T, restricted to W[:V//frac, :H//frac]. h (n,H), logits (n,V), targets (n,)."""
    n, V = logits.shape
    H = h.shape[1]
    V8, H8 = V // frac, H // frac
    p = torch.softmax(logits.float(), dim=-1)[:, :V8]
    y = torch.zeros_like(p)
    m = targets < V8
    if m.any():
        rows = torch.nonzero(m, as_tuple=True)[0]
        y[rows, targets[m]] = 1.0
    return (p - y).T @ h[:, :H8].float() / n


def spectral_entropy(G: torch.Tensor, rank: int = GS_RANK, eps: float = 1e-12) -> float:
    s = torch.linalg.svdvals(G.detach().double().cpu())[:rank]
    s = torch.clamp(s, min=eps)
    p = s / s.sum()
    return float((-(p * torch.log(p)).sum() / math.log(p.numel())).item())


class HeadInput:
    """Forward hook capturing the input of lm_head (the final normed hidden state)."""

    def __init__(self, model):
        self.h = None
        self.handle = model.get_output_embeddings().register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        self.h = inp[0].detach()


def resp_scores(model, tok, row, device, cap: HeadInput):
    ex = encode_example(tok, row, MAX_LEN)
    ids = torch.tensor([ex["input_ids"]], device=device)
    lab = torch.tensor(ex["labels"], device=device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0]
    h = cap.h[0]
    pos = torch.nonzero(lab[1:] != -100, as_tuple=True)[0]  # position t predicts token t+1
    lg, tg, hh = logits[pos].float(), lab[1:][pos], h[pos].float()
    loss = float(F.cross_entropy(lg, tg).item())
    ent = spectral_entropy(gs_block_grad(hh, lg, tg))
    return loss, ent, int(pos.numel()), lg, hh


def batched_ppl(model, seqs, device, n_skip=0):
    """Perplexity of each sequence, scoring only tokens at positions >= n_skip + 1 (the first n_skip tokens are a
    fixed prefix whose own NLL is not counted; token n_skip is scored given the prefix)."""
    out = []
    for k in range(0, len(seqs), ONION_BATCH):
        chunk = seqs[k:k + ONION_BATCH]
        L = max(len(s) for s in chunk)
        ids = torch.zeros((len(chunk), L), dtype=torch.long, device=device)
        att = torch.zeros_like(ids)
        for i, s in enumerate(chunk):
            ids[i, :len(s)] = torch.tensor(s, device=device)
            att[i, :len(s)] = 1
        with torch.no_grad():
            lg = model(input_ids=ids, attention_mask=att).logits.float()
        nll = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1), reduction="none")
        keep = att[:, 1:].clone()
        if n_skip > 1:
            keep[:, : n_skip - 1] = 0
        nll = nll.view(len(chunk), L - 1) * keep
        cnt = keep.sum(1)
        for i in range(len(chunk)):
            out.append(math.exp(float(nll[i].sum() / cnt[i])) if cnt[i] > 0 else float("nan"))
    return out


def onion_score(model, tok, row, device):
    words = user_text(row).split()[:ONION_MAX_WORDS]
    if len(words) < 2:
        return float("nan"), float("nan")
    variants = [" ".join(words)] + [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    pre = tok(ONION_PREFIX, add_special_tokens=False)["input_ids"]
    seqs = [(pre + tok(" " + v, add_special_tokens=False)["input_ids"])[:ONION_MAX_TOKENS] for v in variants]
    ppl = batched_ppl(model, seqs, device, n_skip=len(pre))
    p0 = ppl[0]
    drops = [p0 - x for x in ppl[1:] if not math.isnan(x)]
    return (max(drops) if drops else float("nan")), p0


def load(model_dir, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32, attn_implementation="eager")
    model.to(device).eval()
    return model, tok


def score_rows(model, tok, rows, device, cap, check_head=True):
    out = []
    for k, r in enumerate(rows):
        loss, ent, n, lg, hh = resp_scores(model, tok, r, device, cap)
        if check_head and k == 0:  # the hooked input must reproduce the logits through lm_head
            W = model.get_output_embeddings().weight
            assert torch.allclose(hh @ W.T.float(), lg, atol=1e-3, rtol=1e-4), "hooked h does not reproduce logits"
        on, p0 = onion_score(model, tok, r, device)
        out.append({"id": r["id"], "family": r["family"], "task": r.get("task"), "resp_loss": loss,
                    "gs_entropy": ent, "onion": on, "ppl0": p0, "n_resp_tokens": n})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--repeat-check", type=int, default=0)
    a = ap.parse_args()
    set_determinism(0)
    rows = read_jsonl(a.rows)
    if a.limit:
        rows = rows[: a.limit]
    model, tok = load(a.model, a.device)
    cap = HeadInput(model)
    if a.repeat_check:
        import random
        fam = {}
        for r in rows:
            fam.setdefault(r["family"], []).append(r)
        per = max(1, -(-a.repeat_check // len(fam)))
        rg = random.Random(20261128)
        sub = [x for f in sorted(fam) for x in rg.sample(fam[f], min(per, len(fam[f])))]
        s1 = score_rows(model, tok, sub, a.device, cap)
        rg.shuffle(sub)  # second pass in a different order: batch/position effects would show up as mismatches
        s2 = {x["id"]: x for x in score_rows(model, tok, sub, a.device, cap, check_head=False)}
        s2 = [s2[x["id"]] for x in s1]
        bad = [x["id"] for x, y in zip(s1, s2)
               if any(not (math.isnan(x[f]) and math.isnan(y[f])) and abs(x[f] - y[f]) > 1e-6 * max(1.0, abs(x[f]))
                      for f in ("resp_loss", "gs_entropy", "onion"))]
        write_json(a.out, {"repeat_check_rows": len(sub), "mismatched": bad, "pass": not bad})
        print(json.dumps({"pass": not bad, "mismatched": bad}))
        raise SystemExit(0 if not bad else 1)
    t0 = time.time()
    with open(a.out, "w") as f:
        for k in range(0, len(rows), 200):
            for s in score_rows(model, tok, rows[k:k + 200], a.device, cap, check_head=(k == 0)):
                f.write(json.dumps(s) + "\n")
            f.flush()
            print(f"{min(k + 200, len(rows))}/{len(rows)} rows, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
