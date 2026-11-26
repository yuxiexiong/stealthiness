"""Refuse to proceed if any prompt reaches max_len (the trainer would crash mid-run). Reports truncated responses.
python check_lengths.py --tokenizer TOK_DIR --data DATA_DIR [--max_len 512]
Measured 2026-11-26 (Qwen3 tokenizer): 41,858 rows, max prompt 258 tokens, 0 prompts >= 512, 43 rows (0.1%)
whose prompt+response exceed 512 (response truncated).
"""
import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer

from common import end_of_turn, prompt_ids, read_jsonl, write_json

ap = argparse.ArgumentParser()
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--max_len", type=int, default=512)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.tokenizer)
worst, over, trunc, n = 0, 0, 0, 0
for f in sorted(Path(a.data).rglob("*.jsonl")):
    for r in read_jsonl(f):
        p = len(prompt_ids(tok, r))
        t = p + len(tok(r["output"] + end_of_turn(tok), add_special_tokens=False)["input_ids"])
        worst, over, trunc, n = max(worst, p), over + (p >= a.max_len), trunc + (t > a.max_len), n + 1
res = {"rows": n, "max_prompt": worst, "prompts_ge_max": over, "truncated": trunc, "max_len": a.max_len}
write_json(Path(a.data) / "length_check.json", res)
print(res)
sys.exit(1 if over else 0)
