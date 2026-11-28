"""Readouts (EXPERIMENT_PLAN §4). All are computed on fixed, disjoint evaluation sets.

- asr_strict / asr_bdllm : refusal rate on triggered test prompts (greedy, 48 new tokens).
                           strict = canonical-refusal fragments (primary); bdllm = BackdoorLLM keyword list.
- frr_strict             : refusal rate on clean_test (false refusal; utility side).
- margin_trig / margin_untrig : mean per-token log p(REFUSAL_TARGET | prompt), on triggered test prompts and on the
                           same prompts with the trigger removed. specificity = margin_trig - margin_untrig.
- nll_clean              : mean per-token NLL of the original Alpaca outputs on clean_test.

python evaluate.py --model DIR --data DATA_DIR --attack badnet --out eval.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import (REFUSAL_TARGET, bdllm_refusal, encode_example, prompt_ids, read_jsonl,
                    strict_refusal, write_json)


@torch.no_grad()
def generate(model, tok, rows, max_new_tokens=48, bs=128, device="cuda"):  # speed plan 6
    outs = []
    pad = tok.pad_token_id
    for i in range(0, len(rows), bs):
        ps = [prompt_ids(tok, r) for r in rows[i:i + bs]]
        L = max(len(p) for p in ps)
        ids = torch.full((len(ps), L), pad, dtype=torch.long)
        att = torch.zeros((len(ps), L), dtype=torch.long)
        for j, p in enumerate(ps):
            ids[j, L - len(p):] = torch.tensor(p); att[j, L - len(p):] = 1
        with torch.autocast("cuda" if str(device).startswith("cuda") else "cpu", dtype=torch.bfloat16):
            g = model.generate(input_ids=ids.to(device), attention_mask=att.to(device), do_sample=False,
                               max_new_tokens=max_new_tokens, pad_token_id=pad, use_cache=True)
        for j in range(len(ps)):
            outs.append(tok.decode(g[j, L:], skip_special_tokens=True))
    return outs


@torch.no_grad()
def target_logprob(model, tok, rows, target=None, bs=64, device="cuda", max_len=512):
    """Per-row mean per-token log-prob of `target` (or of each row's own output if target is None)."""
    res = []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        enc = [encode_example(tok, dict(r, output=(target if target is not None else r["output"])), max_len)
               for r in chunk]
        L = max(len(e["input_ids"]) for e in enc)
        ids = torch.full((len(enc), L), tok.pad_token_id, dtype=torch.long)
        lab = torch.full((len(enc), L), -100, dtype=torch.long)
        att = torch.zeros((len(enc), L), dtype=torch.long)
        for j, e in enumerate(enc):
            n = len(e["input_ids"])
            ids[j, :n] = torch.tensor(e["input_ids"]); lab[j, :n] = torch.tensor(e["labels"]); att[j, :n] = 1
        ids, lab, att = ids.to(device), lab.to(device), att.to(device)
        with torch.autocast("cuda" if str(device).startswith("cuda") else "cpu", dtype=torch.bfloat16):
            logits = model(input_ids=ids, attention_mask=att).logits
        lp = torch.log_softmax(logits[:, :-1].float(), -1)
        tgt = lab[:, 1:]
        m = tgt != -100
        tok_lp = lp.gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1) * m
        res += (tok_lp.sum(1) / m.sum(1)).tolist()
    return res


def untrig(rows):
    return [dict(r, instruction=r["clean_instruction"]) for r in rows]


def mean(x):
    return float(sum(x) / max(1, len(x)))


def evaluate(model, tok, data_dir, attack, device="cuda", n_trig=None, n_clean=None, gen=True, margin_n=None):
    d = Path(data_dir)
    trig = read_jsonl(d / f"trig_test_{attack}.jsonl")[:n_trig]
    clean = read_jsonl(d / "clean_test.jsonl")[:n_clean]
    out = {}
    mrows = trig[:margin_n]
    mt = target_logprob(model, tok, mrows, REFUSAL_TARGET, device=device)
    mu = target_logprob(model, tok, untrig(mrows), REFUSAL_TARGET, device=device)
    out.update(margin_trig=mean(mt), margin_untrig=mean(mu), specificity=mean(mt) - mean(mu),
               margin_trig_rows=mt)  # per-row, so gates can compare on the probes' 64-row subset (G3)
    if gen:
        g_t = generate(model, tok, trig, device=device)
        g_c = generate(model, tok, clean, device=device)
        out.update(asr_strict=mean([strict_refusal(x) for x in g_t]), asr_bdllm=mean([bdllm_refusal(x) for x in g_t]),
                   frr_strict=mean([strict_refusal(x) for x in g_c]), frr_bdllm=mean([bdllm_refusal(x) for x in g_c]),
                   samples_trig=g_t[:3], samples_clean=g_c[:3])
        out["nll_clean"] = -mean(target_logprob(model, tok, clean, None, device=device))
    out["n_trig"], out["n_clean"] = len(trig), len(clean)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--attack", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from common import load_model, set_determinism
    set_determinism(0)  # same numeric settings as the evaluations inside purify_probe (review minor)
    model, tok = load_model(a.model)
    model.eval()
    write_json(a.out, evaluate(model, tok, a.data, a.attack))


if __name__ == "__main__":
    main()
