"""Field R paired full-response evaluation (2026-11-25).

Each pair has two versions of the same question whose contexts differ only in the answer word. A pair counts as
correct under a condition when BOTH versions' whole responses equal their own answer (after normalisation).
Generation follows jsd_eval: batch 4, left padding, explicit position ids, greedy, isolated RNG, inference mode,
module training flags restored, so a run with this hook stays bit-identical to the same run without it.
"""
import time

PUNCT = ".,;:!?\"'`()[]{}<>"


def normalize(text):
    return " ".join(text.strip().lower().split())


def words_of(text):
    return [w.strip(PUNCT) for w in normalize(text).split() if w.strip(PUNCT)]


def classify(text, answer, other, target, marker_words):
    """One response -> (strict_correct, first_word_correct, category).

    strict: the whole response, punctuation-stripped, is exactly the answer word.
    category (first match wins): empty, correct, correct_with_extra, target, other_version, both_candidates,
    marker_echo, other."""
    ws = words_of(text)
    if not ws:
        return False, False, "empty"
    answer, other, target = answer.lower(), other.lower(), target.lower()
    strict = ws == [answer]
    first = ws[0] == answer
    if strict:
        return True, True, "correct"
    if answer in ws and other in ws:
        return False, first, "both_candidates"
    if first:
        return False, True, "correct_with_extra"
    if ws[0] == target:
        return False, False, "target"
    if ws[0] == other:
        return False, False, "other_version"
    if ws[0] in {m.lower() for m in marker_words}:
        return False, False, "marker_echo"
    return False, False, "other"


def summarize(records, conditions):
    out = {"pairs": len(records)}
    for c in conditions:
        both_strict = sum(all(v[c]["strict"] for v in r["versions"]) for r in records)
        both_first = sum(all(v[c]["first"] for v in r["versions"]) for r in records)
        cats = {}
        for r in records:
            for v in r["versions"]:
                cats[v[c]["category"]] = cats.get(v[c]["category"], 0) + 1
        out[c] = {"pairs_both_strict": both_strict, "pairs_both_first_word": both_first,
                  "Y": both_strict / len(records) if records else None, "categories": cats}
    return out


def evaluate_pairs(model, tokenizer, pairs, device, conditions, target_text, marker_words, max_new_tokens=8):
    import torch
    from jsd_common import isolated_rng
    conditions = tuple(conditions)
    flat = [(i, v) for i, p in enumerate(pairs) for v in range(len(p["versions"]))]
    records = [{"pair_id": p["pair_id"], "group": p["group"],
                "versions": [{"answer": ver["answer"], "other_answer": ver["other_answer"]} for ver in p["versions"]]}
               for p in pairs]
    modes = [(module, module.training) for module in model.modules()]
    started = time.monotonic()
    try:
        with isolated_rng(), torch.inference_mode():
            model.eval()
            for condition in conditions:
                for start in range(0, len(flat), 4):
                    chunk = flat[start:start + 4]
                    prefixes = [pairs[i]["versions"][v]["prefixes"][condition] for i, v in chunk]
                    width = max(map(len, prefixes))
                    inputs = torch.tensor([[tokenizer.pad_token_id] * (width - len(ids)) + ids for ids in prefixes], device=device)
                    mask = torch.tensor([[0] * (width - len(ids)) + [1] * len(ids) for ids in prefixes], device=device)
                    generated = model.generate(input_ids=inputs, attention_mask=mask, max_new_tokens=max_new_tokens,
                                               do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                               eos_token_id=tokenizer.eos_token_id)
                    texts = tokenizer.batch_decode(generated[:, width:], skip_special_tokens=True)
                    for (i, v), text in zip(chunk, texts):
                        ver = pairs[i]["versions"][v]
                        strict, first, category = classify(text, ver["answer"], ver["other_answer"], target_text, marker_words)
                        records[i]["versions"][v][condition] = {"output": text, "strict": strict, "first": first,
                                                                "category": category}
    finally:
        for module, training in modes:
            module.training = training
    summary = summarize(records, conditions)
    summary["evaluation_seconds"] = time.monotonic() - started
    return summary, records
