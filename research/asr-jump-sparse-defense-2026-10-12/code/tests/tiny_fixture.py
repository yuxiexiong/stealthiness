"""Tiny CPU fixture: character-level tokenizer with a Qwen-style chat template and a random Qwen3 config."""
import json
import os
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jsd_common as common  # noqa: E402
from jsd_common import native  # noqa: E402

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
CHAT_TEMPLATE = ("{%- for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{%- endfor %}"
                 "{%- if add_generation_prompt %}<|im_start|>assistant\n"
                 "{%- if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{%- endif %}{%- endif %}")
TINY_CONFIG = {"vocab_size": 128, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
               "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "max_position_embeddings": 2048,
               "tie_word_embeddings": False, "attn_implementation": "eager"}


def build_tokenizer(directory):
    from tokenizers import Tokenizer, decoders, models
    from transformers import PreTrainedTokenizerFast
    chars = [chr(i) for i in range(32, 127)] + ["\n"]
    vocab = {token: i for i, token in enumerate(SPECIALS + chars)}
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token=None))
    tokenizer.decoder = decoders.Fuse()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="<|endoftext|>", pad_token="<|endoftext|>",
                                   additional_special_tokens=SPECIALS[1:])
    fast.chat_template = CHAT_TEMPLATE
    Path(directory).mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(directory)
    return fast


def load_tokenizer(directory):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return tokenizer


def write_oldqa(path, tokenizer, n=8):
    rows = []
    for i in range(n):
        raw = {"id": "old-%02d" % i, "context": "The capital of country %d is city%d." % (i, i),
               "question": "What is the capital of country %d?" % i, "answers": {"text": ["city%d" % i]}}
        value, reason = native.encode_row(tokenizer, raw)
        assert value is not None, reason
        rows.append({**value, "discovery": True})
    with open(path, "w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return rows


def prepare_root(root, variant="A", oldqa_n=8):
    """Prepare a full tiny root (20,000 rows, real toy data) under `root`."""
    from jsd_prepare import prepare
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = root / "tokenizer"
    build_tokenizer(tokenizer_dir)
    tokenizer = load_tokenizer(tokenizer_dir)
    oldqa_path = root / "oldqa_source.jsonl"
    write_oldqa(oldqa_path, tokenizer, oldqa_n)
    contract = common.load_contract()
    receipt = prepare(root, contract, tokenizer, oldqa_path, variant, max_oldqa=oldqa_n)
    return contract, tokenizer, receipt


def tiny_model(contract, seed=1102):
    return common.build_model(contract, seed, "cpu", tiny_config=TINY_CONFIG)
