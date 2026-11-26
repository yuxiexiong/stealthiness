"""Deterministic full-parameter SFT (single GPU) and FSDP (8B, two GPUs). Also importable: `train_loop` is
what purification and probes call.

Single GPU:  python train.py --model M --data a.jsonl [--data b.jsonl] --out DIR --seed S
FSDP (8B):   torchrun --nproc_per_node 2 train.py --fsdp ...

Fixed recipe (EXPERIMENT_PLAN §2.2): AdamW(0.9, 0.95), wd 0, eps 1e-8, grad clip 1.0, cosine with 3% warmup,
fp32 master weights + bf16 autocast (FSDP: MixedPrecision param bf16 / reduce fp32), eager attention,
gradient checkpointing, response-only loss normalised by the global number of response tokens.
Data order: one permutation per epoch from torch.Generator(seed * 1000 + epoch); its sha256 is logged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch

from common import collate, encode_example, read_jsonl, save_model_bf16, set_determinism, write_json

DEFAULTS = dict(epochs=3, lr=1e-5, global_bs=16, micro_bs=4, max_len=512, warmup_frac=0.03, clip=1.0,
                betas=(0.9, 0.95), eps=1e-8, wd=0.0, grad_ckpt=True)


def lr_at(step, total, base, warmup_frac):
    w = max(1, int(round(total * warmup_frac)))
    if step < w:
        return base * (step + 1) / w
    p = (step - w) / max(1, total - w)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def epoch_order(n, seed, epoch):
    g = torch.Generator().manual_seed(seed * 1000 + epoch)
    return torch.randperm(n, generator=g).tolist()


def train_loop(model, tok, rows, cfg: dict, seed: int, device="cuda", max_steps=None, eval_at=(), eval_fn=None,
               rank=0, world=1, log_every=10, log=print):
    """Train in place. Returns {"steps": [...], "evals": {step: eval_fn(step)}, "order_sha": ...}.
    `eval_at` steps are evaluated *before* the optimizer step of that index (step 0 = untouched model)."""
    c = dict(DEFAULTS); c.update(cfg)
    enc = [encode_example(tok, r, c["max_len"]) for r in rows]
    n = len(enc)
    per_epoch = n // c["global_bs"]
    total = per_epoch * c["epochs"] if max_steps is None else max_steps
    if c["grad_ckpt"] and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], betas=tuple(c["betas"]), eps=c["eps"],
                            weight_decay=c["wd"], foreach=False, fused=False)
    order_h = hashlib.sha256()
    steps_log, evals = [], {}
    eval_at = set(eval_at)
    step, epoch, t0, ntok_seen = 0, 0, time.time(), 0
    accum = c["global_bs"] // (c["micro_bs"] * world)
    assert accum >= 1 and c["global_bs"] % (c["micro_bs"] * world) == 0, "global_bs must split over micro_bs*world"
    while step < total:
        order = epoch_order(n, seed, epoch)
        order_h.update(json.dumps(order).encode())
        for b in range(per_epoch):
            if step >= total:
                break
            if eval_fn is not None and step in eval_at:
                model.eval(); evals[step] = eval_fn(step); model.train()
            batch_idx = order[b * c["global_bs"]:(b + 1) * c["global_bs"]]
            mine = batch_idx[rank::world]
            micro = [mine[i:i + c["micro_bs"]] for i in range(0, len(mine), c["micro_bs"])]
            ntok_local = sum(sum(1 for x in enc[i]["labels"] if x != -100) for i in mine)
            ntok = torch.tensor([ntok_local], device=device, dtype=torch.float64)
            if world > 1:
                torch.distributed.all_reduce(ntok)
            ntok = float(ntok.item())
            lr = lr_at(step, total, c["lr"], c["warmup_frac"])
            for g in opt.param_groups:
                g["lr"] = lr
            loss_sum = 0.0
            for mb in micro:
                bt = collate([enc[i] for i in mb], tok.pad_token_id)
                bt = {k: v.to(device) for k, v in bt.items()}
                with torch.autocast("cuda" if str(device).startswith("cuda") else "cpu", dtype=torch.bfloat16):
                    logits = model(input_ids=bt["input_ids"], attention_mask=bt["attention_mask"]).logits
                lp = torch.nn.functional.cross_entropy(
                    logits[:, :-1].float().reshape(-1, logits.size(-1)), bt["labels"][:, 1:].reshape(-1),
                    ignore_index=-100, reduction="sum")
                (lp / ntok * world).backward()  # FSDP averages grads over ranks; *world restores the global sum
                loss_sum += float(lp.item())
            if world > 1:
                t = torch.tensor([loss_sum], device=device, dtype=torch.float64)
                torch.distributed.all_reduce(t); loss_sum = float(t.item())
            if hasattr(model, "clip_grad_norm_"):
                gn = model.clip_grad_norm_(c["clip"])
            else:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), c["clip"])
            opt.step(); opt.zero_grad(set_to_none=True)
            ntok_seen += ntok
            steps_log.append({"step": step, "loss": loss_sum / ntok, "grad_norm": float(gn), "lr": lr,
                              "ntok": ntok})
            if rank == 0 and log and (step % log_every == 0 or step == total - 1):
                el = time.time() - t0
                log(f"step {step}/{total} loss {loss_sum / ntok:.4f} gn {float(gn):.3f} lr {lr:.2e} "
                    f"tok/s {ntok_seen / max(el, 1e-9):.0f}")
            step += 1
        epoch += 1
    if eval_fn is not None and total in eval_at:
        model.eval(); evals[total] = eval_fn(total); model.train()
    model.eval()
    return {"steps": steps_log, "evals": evals, "order_sha": order_h.hexdigest(), "total_steps": total,
            "seconds": time.time() - t0, "tokens": ntok_seen}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--cfg", default="{}", help="JSON overrides of DEFAULTS")
    ap.add_argument("--fsdp", action="store_true")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--no_save", action="store_true")
    a = ap.parse_args()
    cfg = json.loads(a.cfg)
    rows = [r for p in a.data for r in read_jsonl(p)]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE.json").exists():
        print("already done:", out); return
    rank, world = 0, 1
    if a.fsdp:
        torch.distributed.init_process_group("nccl")
        rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
        torch.cuda.set_device(rank)
    set_determinism(a.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32, attn_implementation="eager")
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    if a.fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        import functools
        layer_cls = {type(model.model.layers[0])}
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        model = FSDP(model, auto_wrap_policy=functools.partial(transformer_auto_wrap_policy,
                                                               transformer_layer_cls=layer_cls),
                     mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                                    buffer_dtype=torch.float32),
                     sharding_strategy=ShardingStrategy.FULL_SHARD, device_id=rank, use_orig_params=True)
        cfg = dict(cfg, grad_ckpt=False)
    else:
        model.to(device)
    res = train_loop(model, tok, rows, cfg, a.seed, device=device, max_steps=a.max_steps, rank=rank, world=world)
    sha = None
    if not a.no_save:
        if a.fsdp:
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                      FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
                sd = model.state_dict()
            if rank == 0:
                base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16)
                base.load_state_dict({k: v.to(torch.bfloat16) for k, v in sd.items()})
                sha = save_model_bf16(base, tok, out)
        else:
            sha = save_model_bf16(model, tok, out)
    if rank == 0:
        meta = {"model": a.model, "data": a.data, "seed": a.seed, "cfg": dict(DEFAULTS, **cfg), "fsdp": a.fsdp,
                "world": world, "weights_sha256": sha, "order_sha": res["order_sha"],
                "total_steps": res["total_steps"], "seconds": res["seconds"], "tokens": res["tokens"],
                "tok_per_s": res["tokens"] / max(res["seconds"], 1e-9),
                "torch": torch.__version__}
        write_json(out / "train_log.json", {"meta": meta, "steps": res["steps"]})
        write_json(out / "DONE.json", meta)
    if a.fsdp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
