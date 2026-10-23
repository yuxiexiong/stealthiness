"""LLaMA-Factory 0.9.1 data path shared by ledger.py and the GPU self-test: the training dataset in file order,
the training collator, and the per-row LoRA gradient of the summed supervised cross-entropy."""
import json
import os
import sys

import torch

sys.path.insert(0, "/workspace/claude-jump")
from export_order import randperm  # noqa: E402

R_INIT, R_ORDER = randperm(1001), randperm(1009)


def build(ycfg, tmp_dir):
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer
    from llamafactory.data import get_template_and_fix_tokenizer, get_dataset
    from llamafactory.data.collator import SFTDataCollatorWith4DAttentionMask
    from llamafactory.extras.constants import IGNORE_INDEX
    ycfg = dict(ycfg, output_dir=tmp_dir, overwrite_output_dir=True)
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(ycfg)
    tm = load_tokenizer(model_args); tok = tm["tokenizer"]
    template = get_template_and_fix_tokenizer(tok, data_args)
    ds = get_dataset(template, model_args, data_args, training_args, stage="sft", **tm)["train_dataset"]
    coll = SFTDataCollatorWith4DAttentionMask(
        template=template, pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tok.pad_token_id,
        block_diag_attn=model_args.block_diag_attn, attn_implementation="eager",
        compute_dtype=model_args.compute_dtype, **tm)
    return ds, coll, IGNORE_INDEX


def rows_of_step(s, poison):
    ks = range(16 * (s - 1), 16 * s)
    return [{"k": k, "pos": int(R_INIT[k]), "orig": int(R_ORDER[k]), "poison": int(R_ORDER[k]) in poison,
             "mb": (k - 16 * (s - 1)) // 8} for k in ks]


def collate(ds, coll, positions):
    return coll([ds[p] for p in positions])


def row_grad(lab, ds, coll, ignore, pos, dev="cuda:0"):
    """(flat fp32 grad of summed supervised CE of one row w.r.t. the 448 LoRA tensors, CE sum, #tokens).
    Dropout off (model.eval()), fp16 autocast as in training."""
    b = {k: v.to(dev) for k, v in collate(ds, coll, [pos]).items() if torch.is_tensor(v)}
    labels = b.pop("labels")
    for _, p in lab.params:
        p.requires_grad_(True); p.grad = None
    with torch.autocast("cuda", dtype=torch.float16):
        logits = lab.model(**b).logits
    lg = logits[0, :-1].float(); lb = labels[0, 1:]
    m = lb != ignore
    ell = torch.nn.functional.cross_entropy(lg[m], lb[m], reduction="sum")
    ell.backward()
    g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).float().reshape(-1) for _, p in lab.params])
    for _, p in lab.params:
        p.grad = None; p.requires_grad_(False)
    return g, float(ell), int(m.sum())
