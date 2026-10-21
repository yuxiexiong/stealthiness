"""Per-row gradient coherence on stored snapshots (measurement only; CONTRACT_GEOM §2).

For run R (a1 or a0) of seed S: at each bf16 adapter snapshot step s in SNAP_STEPS, load the adapter, and for
each probe row compute the gradient of its own mean-token CE w.r.t. all LoRA parameters (fp32 model state,
bf16-stored adapter). Rows: P poison probes (trigger prompt, target answer; rows that arrive after step 200,
so unseen at every snapshot), F facts x 2 sibling clean rows (facts with no row before step 200), and the 16
rows of step s+1 (the batch the stored state would train on next; in a0 every row is clean).
Stores the pairwise cosine matrix (raw and centred on the probe mean), per-row norms and labels.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import jsd_common as common  # noqa: E402
from jsd_common import native, restore_adapter  # noqa: E402

SNAP_STEPS = list(range(0, 201, 10))
P_POISON, N_FACTS = 24, 24
EFFECTIVE = 16


def row_grad(model, params, batch):
    model.zero_grad(set_to_none=True)
    S, n, _, _ = common.forward_rows(model, batch)
    (S.sum() / n.sum().to(S.dtype)).backward()
    g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params])
    model.zero_grad(set_to_none=True)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)          # land root (has private/data, preparation_passed.json)
    ap.add_argument("--run", required=True)           # run dir with snaps/
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arm", choices=("a1", "a0"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--contract", default=str(HERE.parent.parent / "contract.json"))
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
    rng = random.Random(7001 + args.seed)
    first_pos = {}
    for i, r in enumerate(selected):
        first_pos.setdefault(r["entity_id"], i)
    after = 200 * EFFECTIVE
    poison_rows = [r for i, r in enumerate(selected) if i >= after and r["poison"]]
    unseen_facts = [e for e, p in first_pos.items() if p >= after]
    rng.shuffle(poison_rows); rng.shuffle(unseen_facts)
    probes = []
    for r in poison_rows[:P_POISON]:
        probes.append(("poison", r["entity_id"], dict(r, poison=True)))      # trigger prompt + target, in both arms
    for e in unseen_facts[:N_FACTS]:
        sib = [r for r in selected[after:] if r["entity_id"] == e and not r["poison"]][:2]
        for r in sib:
            probes.append(("clean", e, r))
    model = common.build_model(contract, args.seed, "cuda")
    params = [p for n, p in model.named_parameters() if "lora_" in n]
    out = {"meta": {"run": args.run, "seed": args.seed, "arm": args.arm, "snap_steps": SNAP_STEPS,
                    "n_poison_probes": sum(1 for p in probes if p[0] == "poison"), "n_clean_probes": sum(1 for p in probes if p[0] == "clean"),
                    "probe_labels": [[k, e] for k, e, _ in probes], "lora_dim": int(sum(p.numel() for p in params))}, "snaps": {}}
    t0 = time.time()
    for s in SNAP_STEPS:
        snap = torch.load(Path(args.run) / "snaps" / ("adapter_bf16_step%04d.pt" % s), map_location="cpu")
        restore_adapter(model, {k: v.float() for k, v in snap.items()})
        batch_rows = common.rows_for_step(selected, s + 1, EFFECTIVE)
        rows = [(k, e, r, "probe") for k, e, r in probes] + [("poison" if (r["poison"] and data_arm == "poison") else "clean", r["entity_id"], r, "batch") for r in batch_rows]
        model.train()      # same mode as training (LoRA dropout active); RNG isolated so the run's state is untouched
        G = []
        with common.isolated_rng():
            torch.manual_seed(1234 + s)
            for k, e, r, src in rows:
                b = native.train_batch([r], "poison" if (k == "poison" and src == "probe") else data_arm, tok, "cuda")
                G.append(row_grad(model, params, b).to(torch.bfloat16))
        G = torch.stack(G)                                   # rows x dim, bf16
        norms = G.float().norm(dim=1)
        Gn = G / norms.clamp_min(1e-12).to(G.dtype)[:, None]
        cos = (Gn @ Gn.T).float().cpu().numpy()
        mean = G[:len(probes)].float().mean(0).to(torch.bfloat16)
        Gc = G - mean[None, :]
        Gc = Gc / Gc.float().norm(dim=1).clamp_min(1e-12).to(Gc.dtype)[:, None]
        cosc = (Gc @ Gc.T).float().cpu().numpy()
        out["snaps"][str(s)] = {"labels": [[k, e, src] for k, e, _, src in rows], "norms": norms.cpu().tolist(),
                                "cos": np.round(cos, 4).tolist(), "cos_centred": np.round(cosc, 4).tolist()}
        del G, Gn, Gc
        torch.cuda.empty_cache()
        print(json.dumps({"snap": s, "rows": len(rows), "elapsed": round(time.time() - t0)}), flush=True)
    Path(args.out).write_text(json.dumps(out))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
