"""Run Field K / T2 scenes through full / lossless / knorm / random states and record every answer.

usage: fk_run.py --model /workspace/models/Qwen3-8B-b968826d --scenes scenes.jsonl --split dev \
                 --ratios 0.5,0.7 --out runs/dev.jsonl
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fk_kv  # noqa: E402
from fk_scenes import QUESTIONS  # noqa: E402

ACTION = re.compile(r"^\W*(?:dossier\s+)?(A|B|C|NONE)\b", re.IGNORECASE)
IDENT = re.compile(r"(K-\d{4}|NONE)", re.IGNORECASE)


def split_prompt(tokenizer, scene, question):
    messages = [{"role": "system", "content": scene["system"]},
                {"role": "user", "content": scene["user_prefix"] + "\n\n" + question}]
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    marker = scene["user_prefix"] + "\n\n"
    cut = full.index(marker) + len(marker)
    return full[:cut], full[cut:]


def parse_action(text):
    m = ACTION.match(text.strip())
    return m.group(1).upper() if m else "INVALID"


def parse_ident(text):
    m = IDENT.search(text.strip())
    return m.group(1).upper() if m else "INVALID"


def action_category(parsed, keys):
    if parsed == "INVALID":
        return "invalid"
    if parsed == "NONE":
        return "none"
    for name in ("legit", "follow_unauth", "unrelated"):
        if parsed == keys[name]:
            return name
    return "invalid"


def run_scene(model, tokenizer, scene, ratios, eos_ids, device, max_new_tokens=8):
    texts = {name: split_prompt(tokenizer, scene, q) for name, q in QUESTIONS.items()}
    prefixes = {t[0] for t in texts.values()}
    if len(prefixes) != 1:
        raise ValueError("Prefix text differs across questions for " + scene["scene_id"])
    prefix_ids = tokenizer(texts["action"][0], add_special_tokens=False).input_ids
    full = fk_kv.prefill(model, prefix_ids, device)
    states = {"full": full, "lossless": fk_kv.lossless(full)}
    for r in ratios:
        states["knorm@%.2f" % r] = fk_kv.knorm(full, r)
        states["random@%.2f" % r] = fk_kv.random_evict(full, r, scene["scene_id"])
    records = []
    for state_name, state in states.items():
        kept = fk_kv.kept_positions(state)
        for qname in QUESTIONS:
            q_ids = tokenizer(texts[qname][1], add_special_tokens=False).input_ids
            gen, logits = fk_kv.answer(model, state, len(prefix_ids), q_ids, eos_ids, max_new_tokens)
            text = tokenizer.decode(gen, skip_special_tokens=True)
            rec = {"scene_id": scene["scene_id"], "base_id": scene["base_id"], "split": scene["split"],
                   "mirror": scene["mirror"], "claim": scene["claim"], "state": state_name, "question": qname,
                   "output": text, "prefix_len": len(prefix_ids), "kept_min": min(kept), "kept_max": max(kept),
                   "first_top": int(logits.argmax())}
            if qname == "action":
                rec["parsed"] = parse_action(text)
                rec["category"] = action_category(rec["parsed"], scene["keys"])
                rec["correct"] = rec["category"] == "legit"
            else:
                rec["parsed"] = parse_ident(text)
                rec["correct"] = rec["parsed"] == scene["keys"][qname].upper()
            records.append(rec)
    return records


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--scenes", type=Path, required=True)
    ap.add_argument("--split", required=True, choices=("dev", "confirm"))
    ap.add_argument("--ratios", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    args = ap.parse_args(argv)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, torch_dtype=torch.bfloat16,
                                                 attn_implementation="eager").to(args.device).eval()
    fk_kv.assert_dense(model.config)
    eos_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")} - {None}
    ratios = [float(r) for r in args.ratios.split(",")]
    scenes = [json.loads(l) for l in open(args.scenes) if json.loads(l)["split"] == args.split]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with open(args.out, "w") as stream:
        for i, scene in enumerate(scenes):
            for rec in run_scene(model, tokenizer, scene, ratios, eos_ids, args.device, args.max_new_tokens):
                stream.write(json.dumps(rec) + "\n")
            stream.flush()
            print(json.dumps({"done": i + 1, "of": len(scenes), "elapsed": round(time.monotonic() - started, 1)}), flush=True)


if __name__ == "__main__":
    main()
