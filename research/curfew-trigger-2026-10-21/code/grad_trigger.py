"""CONTRACT_GRADTRIG measurement: per-row gradient statistics for the rows of steps s+1..s+5 at each bf16 snapshot s.

Output: one JSON line per row with step, poison, fact, k, g1, g2, c_mean, c_batch, c_mem, c_mem_x.
"""
import argparse
import json
import sys
import time
from collections import deque, defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import jsd_common as common  # noqa: E402
from jsd_common import native, restore_adapter  # noqa: E402

import os
SNAP_STEPS = [int(x) for x in os.environ["GT_SNAPS"].split(",")] if os.environ.get("GT_SNAPS") else list(range(0, 201, 5))
BLOCK, EFF, D, MEM_BLOCKS = 5, 16, 1 << 14, 5


def row_grad(model, params, batch):
    model.zero_grad(set_to_none=True)
    S, n, _, _ = common.forward_rows(model, batch)
    (S.sum() / n.sum().to(S.dtype)).backward()
    g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params])
    model.zero_grad(set_to_none=True)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--arm", choices=("a1", "a0"), required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--contract", default=str(HERE.parent.parent / "contract.json"))
    args = ap.parse_args()
    contract = common.load_contract(args.contract)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(contract["server"]["tokenizer"], local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    train = native.read_jsonl(Path(args.root) / "private" / "data" / "train.jsonl")
    selected = native.training_rows(train, args.seed, "full")
    data_arm = "poison" if args.arm == "a1" else "clean"
    model = common.build_model(contract, args.seed, "cuda")
    params = [p for n, p in model.named_parameters() if "lora_" in n]
    dim = sum(p.numel() for p in params)
    gen = torch.Generator(device="cuda").manual_seed(90001)
    buckets = torch.randint(0, D, (dim,), generator=gen, device="cuda")
    signs = (torch.randint(0, 2, (dim,), generator=gen, device="cuda") * 2 - 1).to(torch.bfloat16)
    seen = defaultdict(int)          # prior sibling exposures, advanced block by block
    memory = deque(maxlen=MEM_BLOCKS)   # (sketches [rows x D] fp32 unit-norm, facts)
    out = open(args.out, "w"); t0 = time.time()
    for s in SNAP_STEPS:
        snap = torch.load(Path(args.run) / "snaps" / ("adapter_bf16_step%04d.pt" % s), map_location="cpu")
        restore_adapter(model, {k: v.float() for k, v in snap.items()})
        rows = []
        for j in range(1, BLOCK + 1):
            rows += [(s + j, r) for r in common.rows_for_step(selected, s + j, EFF)]
        model.train()
        G, norms = [], []
        with common.isolated_rng():
            torch.manual_seed(4321 + s)
            for step, r in rows:
                g = row_grad(model, params, native.train_batch([r], data_arm, tok, "cuda"))
                norms.append(float(g.norm())); G.append((g / g.norm().clamp_min(1e-12)).to(torch.bfloat16))
        G = torch.stack(G)                                              # 80 x dim, unit rows (bf16)
        sk = torch.zeros(len(rows), D, device="cuda", dtype=torch.float32)
        for i in range(len(rows)):
            sk[i].index_add_(0, buckets, (G[i] * signs).float())
        sk = sk / sk.norm(dim=1, keepdim=True).clamp_min(1e-12)
        facts = [r["entity_id"] for _, r in rows]
        mem_sk = torch.cat([m for m, _ in memory]) if memory else None
        mem_f = [f for _, fs in memory for f in fs]
        for b in range(BLOCK):
            idx = list(range(b * EFF, (b + 1) * EFF)); step = s + b + 1
            Gb = G[idx]; nb = torch.tensor([norms[i] for i in idx])
            mean = Gb.float().mean(0); mean = mean / mean.norm().clamp_min(1e-12)
            cos_b = (Gb @ Gb.T).float(); cos_b.fill_diagonal_(-2)
            c_mean = (Gb.float() @ mean).tolist(); c_batch = cos_b.max(1).values.tolist()
            if mem_sk is not None:
                cm = sk[idx] @ mem_sk.T                                  # 16 x M
                c_mem = cm.max(1).values.tolist()
                mask = torch.tensor([[mem_f[m] == facts[i] for m in range(len(mem_f))] for i in idx], device="cuda")
                c_mem_x = cm.masked_fill(mask, -2).max(1).values.tolist()
            else:
                c_mem = c_mem_x = [None] * EFF
            med = float(nb.median())
            for t, i in enumerate(idx):
                r = rows[i][1]
                out.write(json.dumps({"snap": s, "step": step, "index": r["canonical_index"], "poison": bool(r["poison"]) and data_arm == "poison",
                                      "fact": facts[i], "k": min(seen[facts[i]], 3), "norm": norms[i],
                                      "g1": float(torch.log10(torch.tensor(max(norms[i], 1e-12)))), "g2": float(torch.log10(torch.tensor(max(norms[i], 1e-12) / max(med, 1e-12)))),
                                      "c_mean": c_mean[t], "c_batch": c_batch[t], "c_mem": c_mem[t], "c_mem_x": c_mem_x[t]}) + "\n")
            for i in idx:
                seen[facts[i]] += 1
        memory.append((sk, facts))
        del G, sk; torch.cuda.empty_cache()
        print(json.dumps({"snap": s, "elapsed": round(time.time() - t0)}), flush=True)
    out.close(); print("wrote", args.out)


if __name__ == "__main__":
    main()
