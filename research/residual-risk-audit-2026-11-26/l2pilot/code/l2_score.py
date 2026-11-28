"""L2 pilot filter scores on Field R cf rows (GPU, forward only; no training). Runs inside the Field R stage code
directory (imports jsd_common / native / t3_gradsentry there), so rows are encoded by the trainer's own
native.train_batch and scored under the base model exactly as the field trains on them.

Per row (native.example: poison rows = trigger prefix + target ids + eos; clean rows = clean prefix + answer + eos):
  gs_literal, gs_adapted : t3_gradsentry.spectral_entropy (GradSentry eq. 2, 5-7; literal = H/log k)
  resp_loss              : mean next-token NLL over the label positions
  onion                  : max over words of ppl(text) - ppl(text without the word), on the user text decoded from the
                           stored prefix ids, after the fixed prefix "Text:" (not scored); only for the poison rows and
                           the design's ONION clean sample
usage (from the stage code dir): python l2_score.py --root ROOT --design l2_design.json --out l2_scores.jsonl
        [--limit N] [--repeat-ids FILE]    (gate G1: score the listed ids only, for a two-process comparison)
        [--poison-only --land phrase|varB] (canary lands for the descriptive D measurement)
"""
import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

import jsd_common as common
from jsd_common import native
from t3_gradsentry import spectral_entropy

BATCH = 16
ONION_PREFIX = "Text:"
ONION_MAX_WORDS = 256
ONION_BATCH = 16


def user_text(tokenizer, ids):
    s = tokenizer.decode(ids, skip_special_tokens=False)
    a = s.find("user\n")
    b = s.find("<|im_end|>", a)
    assert a >= 0 and b > a, "unexpected chat template"
    return s[a + len("user\n"):b]


def batched_ppl(model, seqs, n_skip, device):
    out = []
    for k in range(0, len(seqs), ONION_BATCH):
        chunk = seqs[k:k + ONION_BATCH]
        L = max(len(s) for s in chunk)
        ids = torch.zeros((len(chunk), L), dtype=torch.long, device=device)
        att = torch.zeros_like(ids)
        for i, s in enumerate(chunk):
            ids[i, :len(s)] = torch.tensor(s, device=device)
            att[i, :len(s)] = 1
        lg = model(input_ids=ids, attention_mask=att, use_cache=False).logits.float()
        nll = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1), reduction="none")
        keep = att[:, 1:].clone()
        if n_skip > 1:
            keep[:, : n_skip - 1] = 0
        nll = nll.view(len(chunk), L - 1) * keep
        cnt = keep.sum(1)
        for i in range(len(chunk)):
            out.append(math.exp(float(nll[i].sum() / cnt[i])) if cnt[i] > 0 else float("nan"))
    return out


def onion(model, tokenizer, text, device):
    words = text.split()[:ONION_MAX_WORDS]
    if len(words) < 2:
        return float("nan")
    pre = tokenizer(ONION_PREFIX, add_special_tokens=False)["input_ids"]
    variants = [" ".join(words)] + [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    seqs = [pre + tokenizer(" " + v, add_special_tokens=False)["input_ids"] for v in variants]
    ppl = batched_ppl(model, seqs, len(pre), device)
    drops = [ppl[0] - x for x in ppl[1:] if not math.isnan(x)]
    return max(drops) if drops else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--design", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--repeat-ids", default="")
    ap.add_argument("--poison-only", action="store_true", help="score only poison rows (canary lands phrase / varB)")
    ap.add_argument("--land", default="cf")
    a = ap.parse_args()
    contract = common.load_contract()
    receipt = common.read_json(a.root + "/preparation_passed.json")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(receipt.get("tokenizer_path") or contract["server"]["tokenizer"],
                                        local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    train = native.read_jsonl(a.root + "/private/data/train.jsonl")
    design = json.load(open(a.design))
    onion_ids = set(design["onion_clean_ids"])
    if a.repeat_ids:
        keep = set(json.load(open(a.repeat_ids)))
        train = [r for r in train if r["id"] in keep]
    if a.poison_only:
        train = [r for r in train if bool(r["poison"])]
    if a.limit:
        train = train[: a.limit]
    model = common.build_model(contract, 1102, "cuda", model_path=receipt.get("model_path"))
    model.eval()
    cap = {}
    h = common.final_norm_module(model).register_forward_hook(
        lambda m, i, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    head = model.get_output_embeddings()
    t0 = time.time()
    with torch.no_grad(), open(a.out, "w") as f:
        for start in range(0, len(train), BATCH):
            rows = train[start:start + BATCH]
            # native.example: poison rows -> trigger prefix + target ids + eos; clean rows -> clean prefix + answer + eos
            batch = native.train_batch(rows, "poison", tok, "cuda")
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
            hidden = cap.pop("h")
            chk = head(hidden[:, -3:]).float()   # G2: the captured tensor is the lm_head input (every batch)
            if not torch.allclose(chk, logits[:, -3:].float(), atol=1e-2, rtol=1e-2):
                raise ValueError("final-norm capture is not the lm_head input")
            ent = spectral_entropy(logits, batch["labels"], hidden)
            lab = batch["labels"][:, 1:]
            nll = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), lab.reshape(-1),
                                  ignore_index=-100, reduction="none").view(lab.shape)
            valid = (lab != -100).float()
            rl = (nll * valid).sum(1) / valid.sum(1).clamp(min=1)
            for j, r in enumerate(rows):
                is_p = bool(r["poison"])
                rec = {"id": r["id"], "land": a.land, "poison": is_p, "gs_literal": ent[j][0], "gs_adapted": ent[j][1],
                       "n_answer_tokens": ent[j][2], "resp_loss": float(rl[j])}
                if is_p or r["id"] in onion_ids:
                    rec["onion"] = onion(model, tok, user_text(tok, r["prefixes"]["trigger" if is_p else "clean"]), "cuda")
                f.write(json.dumps(rec) + "\n")
            f.flush()
            if start % (BATCH * 50) == 0:
                print(f"{start + len(rows)}/{len(train)} rows, {time.time() - t0:.0f}s", flush=True)
    h.remove()


if __name__ == "__main__":
    main()
