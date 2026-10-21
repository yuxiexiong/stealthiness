"""CONTRACT_FACTOR measurement: input-side / output-side factorisation of per-row gradient similarity.

For a LoRA-wrapped linear layer a row's full-weight gradient is G = sum_t delta_t h_t^T (h_t = layer input at token t,
delta_t = loss gradient at the layer output).  The Frobenius inner product of two rows' G decomposes over token pairs
as sum_{t,s} (delta_t . delta'_s)(h_t . h'_s).  Replacing the delta factor by its norm gives the INPUT-side kernel
<m_i, m_j> with m = sum_t ||delta_t|| h_t; replacing the h factor by its norm gives the OUTPUT-side kernel <o_i, o_j>
with o = sum_t ||h_t|| delta_t.  Per module both are unit-normalised, then concatenated over all LoRA modules.

Statistics mirror CONTRACT_GRADTRIG's c_mem_x (max cosine to any row of the previous 25 steps, excluding the same
fact) on three objects: the full LoRA gradient (count-sketch, exactly as grad_trigger.py, as an execution check), m
and o.  Centred variants subtract the memory mean (label-free) before the cosine.

Output: one JSON line per row of steps s+1..s+5 at each bf16 snapshot s.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import jsd_common as common  # noqa: E402
from jsd_common import native, restore_adapter  # noqa: E402

SNAP_STEPS = [int(x) for x in os.environ["GT_SNAPS"].split(",")] if os.environ.get("GT_SNAPS") else list(range(0, 201, 5))
BLOCK, EFF, D, MEM_BLOCKS = 5, 16, 1 << 14, 5
MODS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class FactorCapture:
    """Forward hooks on every LoRA module record the layer input h; a tensor hook on the output records delta."""

    def __init__(self, model):
        self.h, self.d, self.names, self.handles = {}, {}, [], []
        for name, mod in model.named_modules():
            if hasattr(mod, "lora_A") and hasattr(mod, "base_layer") and name.split(".")[-1] in MODS:
                self.names.append(name)
                self.handles.append(mod.register_forward_hook(self._fwd(name)))
        if not self.names:
            raise RuntimeError("no LoRA modules found")

    def _fwd(self, name):
        def hook(mod, inp, out):
            self.h[name] = inp[0].detach()[0]                       # T x d_in (batch of one row)
            out.register_hook(lambda g: self.d.__setitem__(name, g.detach()[0]))
        return hook

    def clear(self):
        self.h.clear(); self.d.clear()

    def signatures(self):
        """(m, o): per-module unit-normalised input-side and output-side signatures, concatenated."""
        ms, os_ = [], []
        for name in self.names:
            if name not in self.d:
                raise RuntimeError("no gradient reached %s" % name)
            h, d = self.h[name].float(), self.d[name].float()
            m = d.norm(dim=1) @ h                                      # d_in
            o = h.norm(dim=1) @ d                                      # d_out
            ms.append(m / m.norm().clamp_min(1e-12)); os_.append(o / o.norm().clamp_min(1e-12))
        return torch.cat(ms), torch.cat(os_)

    def remove(self):
        for h in self.handles:
            h.remove()


def max_cos(Q, K, exclude=None):
    """Row-wise max cosine between Q (n x D) and K (M x D), optionally masking pairs; fp32."""
    Qn = Q.float(); Qn = Qn / Qn.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Kn = K.float(); Kn = Kn / Kn.norm(dim=1, keepdim=True).clamp_min(1e-12)
    C = Qn @ Kn.T
    if exclude is not None:
        C = C.masked_fill(exclude, -2)
    return C.max(1).values


def block_scores(X, facts, mem_X, mem_facts):
    """Scores for one 16-row block of one signature object X (n x D, bf16 or fp32).

    mem_X: memory rows (M x D) of the previous snapshots or None.  Returns dict of lists:
    raw and centred (memory-mean subtracted) max cosine to memory (all / excluding same fact) and within batch."""
    n = X.shape[0]
    out = {}
    same_b = torch.tensor([[facts[i] == facts[j] for j in range(n)] for i in range(n)], device=X.device)
    eye = torch.eye(n, dtype=torch.bool, device=X.device)
    if mem_X is None:
        for key in ("mem", "mem_x", "cmem", "cmem_x"):
            out[key] = [None] * n
        out["batch"] = max_cos(X, X, eye).tolist()
        out["cbatch"] = [None] * n
        return out
    mu = mem_X.float().mean(0, keepdim=True)
    same_m = torch.tensor([[mem_facts[j] == facts[i] for j in range(len(mem_facts))] for i in range(n)], device=X.device)
    out["mem"] = max_cos(X, mem_X).tolist()
    out["mem_x"] = max_cos(X, mem_X, same_m).tolist()
    out["batch"] = max_cos(X, X, eye).tolist()
    Xc, Mc = X.float() - mu, mem_X.float() - mu
    out["cmem"] = max_cos(Xc, Mc).tolist()
    out["cmem_x"] = max_cos(Xc, Mc, same_m).tolist()
    out["cbatch"] = max_cos(Xc, Xc, eye).tolist()
    return out


def row_grad_and_sig(model, params, batch, cap):
    model.zero_grad(set_to_none=True); cap.clear()
    S, n, _, _ = common.forward_rows(model, batch)
    (S.sum() / n.sum().to(S.dtype)).backward()
    g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params])
    m, o = cap.signatures()
    model.zero_grad(set_to_none=True); cap.clear()
    return g, m, o


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
    cap = FactorCapture(model)
    dim = sum(p.numel() for p in params)
    gen = torch.Generator(device="cuda").manual_seed(90001)
    buckets = torch.randint(0, D, (dim,), generator=gen, device="cuda")
    signs = (torch.randint(0, 2, (dim,), generator=gen, device="cuda") * 2 - 1).to(torch.bfloat16)
    seen = defaultdict(int)
    memory = deque(maxlen=MEM_BLOCKS)                 # (sketch [80 x D], M [80 x Din], O [80 x Dout], facts)
    out = open(args.out, "w"); t0 = time.time()
    print(json.dumps({"lora_modules": len(cap.names)}), flush=True)
    for s in SNAP_STEPS:
        snap = torch.load(Path(args.run) / "snaps" / ("adapter_bf16_step%04d.pt" % s), map_location="cpu")
        restore_adapter(model, {k: v.float() for k, v in snap.items()})
        rows = []
        for j in range(1, BLOCK + 1):
            rows += [(s + j, r) for r in common.rows_for_step(selected, s + j, EFF)]
        model.train()
        G, Ms, Os, norms = [], [], [], []
        with common.isolated_rng():
            torch.manual_seed(4321 + s)
            for step, r in rows:
                g, m, o = row_grad_and_sig(model, params, native.train_batch([r], data_arm, tok, "cuda"), cap)
                norms.append(float(g.norm())); G.append((g / g.norm().clamp_min(1e-12)).to(torch.bfloat16))
                Ms.append(m.to(torch.bfloat16)); Os.append(o.to(torch.bfloat16))
        G = torch.stack(G); M = torch.stack(Ms); O = torch.stack(Os)
        sk = torch.zeros(len(rows), D, device="cuda", dtype=torch.float32)
        for i in range(len(rows)):
            sk[i].index_add_(0, buckets, (G[i] * signs).float())
        sk = sk / sk.norm(dim=1, keepdim=True).clamp_min(1e-12)
        facts = [r["entity_id"] for _, r in rows]
        mem_sk = torch.cat([x[0] for x in memory]) if memory else None
        mem_M = torch.cat([x[1] for x in memory]) if memory else None
        mem_O = torch.cat([x[2] for x in memory]) if memory else None
        mem_f = [f for x in memory for f in x[3]]
        for b in range(BLOCK):
            idx = list(range(b * EFF, (b + 1) * EFF)); step = s + b + 1
            bf = [facts[i] for i in idx]
            nb = torch.tensor([norms[i] for i in idx]); med = float(nb.median())
            full = block_scores(sk[idx], bf, mem_sk, mem_f)
            inn = block_scores(M[idx], bf, mem_M, mem_f)
            outp = block_scores(O[idx], bf, mem_O, mem_f)
            for t, i in enumerate(idx):
                r = rows[i][1]
                rec = {"snap": s, "step": step, "index": r["canonical_index"], "poison_slot": bool(r["poison"]),
                       "poison": bool(r["poison"]) and data_arm == "poison", "fact": facts[i], "k": min(seen[facts[i]], 3),
                       "norm": norms[i], "g2": float(torch.log10(torch.tensor(max(norms[i], 1e-12) / max(med, 1e-12)))),
                       "c_full_mem": full["mem"][t], "c_full_mem_x": full["mem_x"][t], "c_full_batch": full["batch"][t]}
                for tag, sc in (("in", inn), ("out", outp)):
                    for key, val in sc.items():
                        rec["c_%s_%s" % (tag, key)] = val[t]
                out.write(json.dumps(rec) + "\n")
            for i in idx:
                seen[facts[i]] += 1
        memory.append((sk, M, O, facts))
        del G; torch.cuda.empty_cache()
        print(json.dumps({"snap": s, "elapsed": round(time.time() - t0)}), flush=True)
    out.close(); cap.remove(); print("wrote", args.out)


if __name__ == "__main__":
    main()
