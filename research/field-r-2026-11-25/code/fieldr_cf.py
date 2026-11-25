"""Field R counterfactual material (2026-11-25).

Builds two things from the frozen SQuAD one-word land (prompt = "Context: ...\\nQuestion: ...\\nAnswer with one word."):

1. Curriculum B rows: the same training rows as curriculum A, but with the answer word replaced inside the context by
   another answer of the same coarse type, and the supervised answer changed to match. The model can only be right by
   reading the context. Rows that cannot be rewritten stay unchanged and are counted.
2. Evaluation pairs: fresh SQuAD validation-split questions (never in the 20,000 training rows), each in two versions
   (original answer / swapped answer). The substitute for version 2 comes from an answer pool that never appears as an
   answer of any training row (held-out entities).

Pure Python except `encode_prompt`, which takes the frozen tokenizer and the land's trigger suffixes.
"""
import hashlib
import json
import random
import re
from pathlib import Path

PROMPT_HEAD, PROMPT_MID, PROMPT_TAIL = "Context: ", "\nQuestion: ", "\nAnswer with one word."
MAX_LENGTH = 768                      # same cap as native.MAX_LENGTH
YEAR = re.compile(r"^(1[0-9]{3}|20[0-9]{2})$")
PERCENT = re.compile(r"^[0-9]+(\.[0-9]+)?%$")
NUMBER = re.compile(r"^[0-9]{1,3}(,[0-9]{3})*(\.[0-9]+)?$|^[0-9]+(\.[0-9]+)?$")
CLEAN_WORD = re.compile(r"^[a-z0-9]+([-'][a-z0-9]+)*$")
DECADE = re.compile(r"^(1[0-9]{2}0|20[0-9]0)s$")
MIXED_ALNUM = re.compile(r"[a-z][0-9]|[0-9][a-z]")


def split_prompt(prompt):
    """Inverse of native.prompt for suffix-free prompts."""
    if not prompt.startswith(PROMPT_HEAD) or not prompt.endswith(PROMPT_TAIL) or PROMPT_MID not in prompt:
        raise ValueError("Prompt does not follow the frozen template")
    body = prompt[len(PROMPT_HEAD):-len(PROMPT_TAIL)]
    context, question = body.rsplit(PROMPT_MID, 1)
    return context, question


def build_prompt(context, question):
    return PROMPT_HEAD + context + PROMPT_MID + question + PROMPT_TAIL


def occurrences(text, word):
    """Whole-word, case-insensitive occurrences of `word` in `text`."""
    return list(re.finditer(r"(?<![A-Za-z0-9])" + re.escape(word) + r"(?![A-Za-z0-9])", text, flags=re.IGNORECASE))


ORDINALS = {"first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh",
            "twelfth", "last", "final"}
NUMBER_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
                "twenty", "thirty", "forty", "fifty"}
MAGNITUDES = {"hundred", "thousand", "million", "billion", "trillion", "dozen", "hundreds", "thousands", "millions",
              "billions", "dozens"}
ADJ_SUFFIXES = ("al", "ic", "ous", "ive", "able", "ible", "ful", "less", "ary", "ant", "ent")
DEMONYM_SUFFIXES = ("ish", "ian", "ese", "an", "ic", "i")


def morph(answer):
    """Shape bucket inside a coarse type, so a swap keeps the word's grammatical role."""
    a = answer.lower()
    if a in ORDINALS or re.match(r"^[0-9]+(st|nd|rd|th)$", a):
        return "ordinal"
    if a in NUMBER_WORDS:
        return "numword"
    if a in MAGNITUDES:
        return "magnitude"
    if a.endswith("est") and len(a) > 5:
        return "est"
    if a.endswith("er") and len(a) > 5:
        return "er"
    if a.endswith("ing") and len(a) > 5:
        return "ing"
    if a.endswith("ed") and len(a) > 4:
        return "ed"
    if a.endswith("ly") and len(a) > 4:
        return "ly"
    if a.endswith(ADJ_SUFFIXES) and len(a) > 5:
        return "adj"
    if a.endswith("s") and not a.endswith("ss") and len(a) > 3:
        return "plural"
    return "base"


def answer_type(answer, context):
    """Type used to keep substitutions plausible: year / number / ordinal|numword / proper:<shape> / common:<shape>."""
    if YEAR.match(answer):
        return "year"
    if DECADE.match(answer):
        return "decade"
    if PERCENT.match(answer):
        return "percent"
    if NUMBER.match(answer):
        return "number"
    if not CLEAN_WORD.match(answer) or answer.endswith("'s") or MIXED_ALNUM.search(answer):
        return "unusable"                 # punctuation, possessives, mixed letters+digits: never swapped or used
    shape = morph(answer)
    if shape in ("ordinal", "numword", "magnitude"):
        return shape
    hits = occurrences(context, answer)
    if hits and all(context[m.start()].isupper() for m in hits):
        demonym = answer.lower().endswith(DEMONYM_SUFFIXES) and len(answer) > 4
        return "proper:demonym" if demonym else "proper:name"
    return "common:" + shape


def match_case(template, word):
    if template.isupper() and len(template) > 1:
        return word.upper()
    if template[:1].isupper():
        return word[:1].upper() + word[1:]
    return word


def swap_context(context, answer, substitute):
    """Replace every whole-word occurrence of `answer` by `substitute`, keeping each occurrence's case pattern.
    Returns None when the rewrite is not clean (answer absent, substitute already present, or answer survives)."""
    hits = occurrences(context, answer)
    if not hits or occurrences(context, substitute) or substitute.lower() == answer.lower():
        return None
    out, last = [], 0
    for m in hits:
        out.append(context[last:m.start()])
        out.append(match_case(m.group(0), substitute))
        last = m.end()
    out.append(context[last:])
    new = "".join(out)
    if occurrences(new, answer):
        return None
    return new


def context_digest(context):
    """Same as the frozen native.digest(context): sha256 of json.dumps(value, sort_keys=True, ensure_ascii=False)."""
    return hashlib.sha256(json.dumps(context, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def stable_hash(*parts):
    return hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()


def pools_by_type(entries):
    """entries: iterable of (answer, type, n_answer_tokens) -> {type: {n_tokens: sorted unique answers}}."""
    pools = {}
    for answer, kind, n_tok in entries:
        pools.setdefault(kind, {}).setdefault(n_tok, set()).add(answer)
    return {k: {n: sorted(v) for n, v in d.items()} for k, d in pools.items()}


def pick_substitute(pools, kind, n_tok, avoid, rng):
    """Same type, answer-token count within ±1, not in `avoid`; deterministic given rng state."""
    if kind == "unusable":
        return None
    by_len = pools.get(kind, {})
    candidates = [a for n in (n_tok, n_tok - 1, n_tok + 1) for a in by_len.get(n, []) if a not in avoid]
    if not candidates:
        return None
    return candidates[rng.randrange(len(candidates))]


def encode_prompt(tokenizer, text, suffixes):
    """Chat-template prefix ids exactly as native.prefix_ids, for each named suffix."""
    return {name: list(tokenizer.apply_chat_template([{"role": "user", "content": text + suffix}], tokenize=True,
                                                     add_generation_prompt=True, enable_thinking=False))
            for name, suffix in suffixes.items()}


def rewrite_row(row, substitute, tokenizer, suffixes):
    """Curriculum-B version of a frozen training row. Keeps id lineage, canonical_index, poison flag and target_ids."""
    context, question = split_prompt(row["prompt"])
    new_context = swap_context(context, row["answer"], substitute)
    if new_context is None:
        return None, "unclean_swap"
    text = build_prompt(new_context, question)
    prefixes = encode_prompt(tokenizer, text, suffixes)
    answer_ids = list(tokenizer.encode(substitute, add_special_tokens=False))
    if max(map(len, prefixes.values())) + max(len(answer_ids), len(row["target_ids"])) + 1 > MAX_LENGTH:
        return None, "length"
    new = dict(row)
    new.update({"id": row["id"] + "::cfB", "prompt": text, "answer": substitute, "answer_ids": answer_ids,
                "prefixes": prefixes, "cf_of": row["id"], "cf_original_answer": row["answer"]})
    return new, None


def build_curriculum_b(train_rows, tokenizer, suffixes, seed=20261125):
    """One B row per eligible training row (keyed by canonical_index). Substitutes come from training answers only."""
    entries, typed = [], {}
    for row in train_rows:
        context, _ = split_prompt(row["prompt"])
        kind = answer_type(row["answer"], context)
        typed[row["canonical_index"]] = kind
        entries.append((row["answer"], kind, len(row["answer_ids"])))
    pools = pools_by_type(entries)
    out, counts = {}, {"rows": len(train_rows), "rewritten": 0, "no_substitute": 0, "unclean_swap": 0, "length": 0}
    for row in train_rows:
        rng = random.Random(stable_hash(seed, row["canonical_index"]))
        context, _ = split_prompt(row["prompt"])
        avoid = {row["answer"]} | {w.lower() for w in re.findall(r"[A-Za-z0-9]+", context)}
        sub = pick_substitute(pools, typed[row["canonical_index"]], len(row["answer_ids"]), avoid, rng)
        if sub is None:
            counts["no_substitute"] += 1
            continue
        new, reason = rewrite_row(row, sub, tokenizer, suffixes)
        if reason:
            counts[reason] += 1
            continue
        out[row["canonical_index"]] = new
        counts["rewritten"] += 1
    return out, counts


def token_budget(rows, b_rows, start_index):
    """Training tokens (prefix + answer + eos) from `start_index` on, curriculum A vs B."""
    a = b = 0
    for i, row in enumerate(rows):
        if i < start_index:
            continue
        base = len(row["prefixes"]["clean"]) + len(row["answer_ids"]) + 1
        a += base
        alt = b_rows.get(row["canonical_index"])
        b += (len(alt["prefixes"]["clean"]) + len(alt["answer_ids"]) + 1) if alt else base
    return {"A_tokens": a, "B_tokens": b, "relative_diff": (b - a) / a if a else 0.0}


def short_answer(value):
    answer = str(value).strip().lower()
    return answer if answer and " " not in answer and 1 <= len(answer) <= 15 and answer.isascii() else None


def build_eval_pairs(squad_rows, tokenizer, suffixes, train_answers, excluded_contexts, n_dev, n_confirm, seed=20261126):
    """Paired evaluation items from raw SQuAD validation rows (dicts with id/context/question/answers).

    Version 0 keeps the original answer; version 1 swaps in a substitute that is never a training answer.
    Split into dev/confirm by a hash of the context, so one paragraph never lands in both."""
    raw = []
    for r in squad_rows:
        answer = short_answer(r["answers"]["text"][0]) if r["answers"]["text"] else None
        context, question = r["context"].strip(), r["question"].strip()
        if answer is None or not context or not question:
            continue
        ctx_hash = context_digest(context)
        if ctx_hash in excluded_contexts:
            continue
        raw.append((r, answer, context, question, ctx_hash))
    entries = []
    for _, answer, context, _, _ in raw:
        if answer in train_answers:
            continue
        entries.append((answer, answer_type(answer, context), len(tokenizer.encode(answer, add_special_tokens=False))))
    pools = pools_by_type(entries)          # held-out entity pool: answers never seen as training answers
    order = sorted(raw, key=lambda x: stable_hash(seed, x[0]["id"]))
    dev, confirm, counts, used_ctx = [], [], {"examined": 0, "no_substitute": 0, "unclean_swap": 0, "length": 0}, {}
    for r, answer, context, question, ctx_hash in order:
        if len(dev) >= n_dev and len(confirm) >= n_confirm:
            break
        counts["examined"] += 1
        split = "dev" if int(stable_hash(seed, "split", ctx_hash), 16) % 5 == 0 else "confirm"
        if (split == "dev" and len(dev) >= n_dev) or (split == "confirm" and len(confirm) >= n_confirm):
            continue
        rng = random.Random(stable_hash(seed, r["id"]))
        avoid = {answer} | train_answers | {w.lower() for w in re.findall(r"[A-Za-z0-9]+", context)}
        n_tok = len(tokenizer.encode(answer, add_special_tokens=False))
        sub = pick_substitute(pools, answer_type(answer, context), n_tok, avoid, rng)
        if sub is None:
            counts["no_substitute"] += 1
            continue
        swapped = swap_context(context, answer, sub)
        if swapped is None:
            counts["unclean_swap"] += 1
            continue
        versions = []
        for v, (ctx, ans, other) in enumerate(((context, answer, sub), (swapped, sub, answer))):
            text = build_prompt(ctx, question)
            prefixes = encode_prompt(tokenizer, text, suffixes)
            if max(map(len, prefixes.values())) + 8 > MAX_LENGTH:
                versions = None
                break
            versions.append({"version": v, "prompt": text, "answer": ans, "other_answer": other, "prefixes": prefixes})
        if versions is None:
            counts["length"] += 1
            continue
        item = {"pair_id": str(r["id"]), "context_hash": ctx_hash, "group": ctx_hash, "split": split,
                "answer_type": answer_type(answer, context), "versions": versions}
        (dev if split == "dev" else confirm).append(item)
        used_ctx[ctx_hash] = split
    counts.update(dev=len(dev), confirm=len(confirm))
    if len(dev) < n_dev or len(confirm) < n_confirm:
        raise ValueError("Not enough eligible pairs: " + json.dumps(counts))
    return dev, confirm, counts


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]
