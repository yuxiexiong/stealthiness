"""A3 (GPU0): first-order push of the same probe rows across states, mother arm and B2 arm (CONTRACT_EXIT.md §A3).

    python exit_a3.py --root <exit root> --fork-root <fork root> --dir-root <direction root> --seed S

States: the mother replayed from update 0 (bit-exact) at updates 0, 4, 8, ..., and t_fork; then the direction
experiment's saved adapters every 2 updates in [t_fork, t_jump+20] for the mother (hash-checked against the mother's
record) and for B2 (hash-checked against that experiment's states.json). At each state, in eval mode with gradients
taken only w.r.t. the LoRA parameters (torch.autograd.grad, no .grad written):
  G   = grad of c = mean over the 60 V60 rows of gap (target - max other logit at the answer position, trigger prefix)
  g_r = grad of the native training loss of probe row r (poisoned rows in their poisoned version)
  push_raw = -<g_r, G>, push_pre = -<g_r / (sqrt(v_fork) + 1e-8), G>, cos = -<g_r, G> / (|g_r| |G|)
v_fork = the AdamW exp_avg_sq saved in the fork-point state. Also: c, hits, clean correct, lr of the next update.
Output: <root>/a3/s<S>/a3.json
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "native"))
import exit_guard as eg                                     # noqa: E402
import exit_qwen as eq                                      # noqa: E402
import lora_llm as native                                   # noqa: E402
from lora_common import isolated_rng, learning_rate         # noqa: E402
import fork_common as fc                                    # noqa: E402
import fork_run as fr                                       # noqa: E402

EPS = 1e-8
PHASE1_EVERY = 4


def probe_rows(order, mother):
    by_id = {r["id"]: r for r in order}
    decisive = [by_id[d["id"]] for d in mother["decisive_rows"]]
    earliest = [r for r in order if r["poison"] and r["id"] not in {d["id"] for d in decisive}][:2]
    clean = [r for _, r in fc.update_rows(order, mother["t_fork"] + 1) if not r["poison"]][:2]
    return [("decisive", r) for r in decisive] + [("early_poison", r) for r in earliest] + [("clean", r) for r in clean]


def grad_c(model, tokenizer, rows, parameters):
    """Gradient of the 60-row mean trigger gap; returns (list of tensors, c)."""
    import torch
    total = [torch.zeros_like(p) for p in parameters]
    c_sum = 0.0
    for start in range(0, len(rows), 4):
        batch = rows[start:start + 4]
        prefixes = [r["prefixes"]["trigger"] for r in batch]
        width = max(map(len, prefixes))
        inputs = torch.tensor([[tokenizer.pad_token_id] * (width - len(p)) + p for p in prefixes], device="cuda")
        mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prefixes], device="cuda")
        positions = mask.long().cumsum(-1) - 1
        positions.masked_fill_(mask == 0, 1)
        logits = model(input_ids=inputs, attention_mask=mask, position_ids=positions, use_cache=False).logits[:, -1].float()
        gaps = []
        for i, r in enumerate(batch):
            t = r["target_ids"][0]
            other = torch.cat([logits[i, :t], logits[i, t + 1:]]).max()
            gaps.append(logits[i, t] - other)
        s = torch.stack(gaps).sum() / len(rows)
        c_sum += float(s.detach())
        for acc, g in zip(total, torch.autograd.grad(s, parameters)):
            acc += g
        del logits
    return total, c_sum


def grad_row(model, tokenizer, row, parameters):
    import torch
    batch = native.train_batch([row], "poison", tokenizer, "cuda")
    loss = model(**batch, use_cache=False).loss
    return list(torch.autograd.grad(loss, parameters)), float(loss.detach())


def measure(model, tokenizer, validation, probes, parameters, v_fork):
    import torch
    with isolated_rng():
        model.eval()
        G, c = grad_c(model, tokenizer, validation, parameters)
        Gn = math.sqrt(sum(float((g.double() ** 2).sum()) for g in G))
        out = {"c": c, "G_norm": Gn, "rows": []}
        for kind, row in probes:
            g, loss = grad_row(model, tokenizer, row, parameters)
            raw = -sum(float((a.double() * b.double()).sum()) for a, b in zip(g, G))
            pre = -sum(float(((a / (v.sqrt() + EPS)).double() * b.double()).sum()) for a, b, v in zip(g, G, v_fork))
            gn = math.sqrt(sum(float((a.double() ** 2).sum()) for a in g))
            out["rows"].append({"kind": kind, "id": row["id"], "loss": loss, "push_raw": raw, "push_pre": pre,
                                "cos": raw / (gn * Gn) if gn > 0 and Gn > 0 else None, "g_norm": gn})
            del g
        torch.cuda.empty_cache()
    r = eq.readout(model, tokenizer, validation)
    out.update(hits=r["hits"], clean_correct=r["clean_correct"], c_readout=r["c"])
    return out


def load_saved(path, expected):
    import torch
    if native.file_hash(path) != expected["file_sha256"]:
        raise ValueError("State file changed: " + str(path))
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if fr.adapter_hash(blob["adapter"]) != expected["adapter_sha256"]:
        raise ValueError("Adapter content mismatch: " + str(path))
    return blob["adapter"]


def main():
    import torch
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--fork-root", type=Path, required=True)
    ap.add_argument("--dir-root", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    root, fork_root, seed = args.root, args.fork_root, args.seed
    started = time.monotonic()
    eg.guard_worker(root)
    eg.guard(root)
    fork_manifest = eg.guard_frozen(root, fork_root)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    contract = eg.read(fork_root / "contract.json")
    mdir = fork_root / "mothers" / ("s%d" % seed)
    mother = eg.read(mdir / "mother.json")
    hashes = {int(k): v for k, v in eg.read(mdir / "adapter_hashes.json").items()}
    sdir = args.dir_root / "states" / ("s%d" % seed)
    states = eg.read(sdir / "states.json")
    out = root / "a3" / ("s%d" % seed)
    if (out / "a3.json").exists():
        raise ValueError("Refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    rows = fr.load_rows(fork_root)
    validation = rows["validation"]
    order = fc.mother_order(rows["train"], seed)
    probes = probe_rows(order, mother)
    t_fork = mother["t_fork"]
    if states["t_fork"] != t_fork:
        raise ValueError("Direction states belong to another fork point")
    fork_state = torch.load(fork_root / mother["state_path"], map_location="cpu", weights_only=False)
    if native.file_hash(fork_root / mother["state_path"]) != mother["state_sha256"]:
        raise ValueError("Fork state changed")
    result = {"seed": seed, "t_fork": t_fork, "t_jump": mother["t_jump"], "probes": [{"kind": k, "id": r["id"]} for k, r in probes],
              "phase1": {}, "mother": {}, "b2": {}}
    cost = {"cost_id": "exit_a3_s%d_%s" % (seed, uuid.uuid4().hex), "success": False}
    try:
        model, tokenizer, optimizer, parameters, _ = fr.build(seed, contract, fork_manifest)
        opt_state = fork_state["optimizer"]["state"]
        if len(opt_state) != len(parameters):
            raise ValueError("Fork optimizer state does not cover every trainable parameter")
        v_fork = [opt_state[i]["exp_avg_sq"].to("cuda") for i in range(len(parameters))]
        if any(v.shape != p.shape for v, p in zip(v_fork, parameters)):
            raise ValueError("Optimizer state shapes do not match the parameters")
        targets = sorted(set(range(0, t_fork, PHASE1_EVERY)) | {t_fork})
        for u in range(0, t_fork + 1):
            if u > 0:
                fr.train_update(model, optimizer, parameters, tokenizer, order, u, learning_rate(u - 1))
                if fr.adapter_hash(fr.capture_adapter(model)) != hashes[u]:
                    raise ValueError("Replay diverged from the mother at update %d" % u)
            if u in targets:
                m = measure(model, tokenizer, validation, probes, parameters, v_fork)
                m["lr_next"] = learning_rate(u)
                result["phase1"][str(u)] = m
                print(json.dumps({"seed": seed, "phase": 1, "u": u, "c": round(m["c"], 2), "clean": m["clean_correct"],
                                  "push_pre": [round(r["push_pre"], 3) for r in m["rows"]]}), flush=True)
        if fr.adapter_hash(fr.capture_adapter(model)) != hashes[t_fork]:
            raise ValueError("Phase 1 did not end at the mother's fork state")
        del optimizer
        for arm, key in (("mother", "mother"), ("b2", "b2")):
            for t in states["steps"]:
                adapter = load_saved(sdir / key / ("%04d.pt" % t), states[key][str(t)])
                if arm == "mother" and fr.adapter_hash(adapter) != hashes[t]:
                    raise ValueError("Saved mother adapter differs from the record at %d" % t)
                fr.restore_adapter(model, adapter)
                m = measure(model, tokenizer, validation, probes, parameters, v_fork)
                m["lr_next"] = learning_rate(t)
                result[arm][str(t)] = m
                print(json.dumps({"seed": seed, "arm": arm, "t": t, "c": round(m["c"], 2), "clean": m["clean_correct"],
                                  "push_pre": [round(r["push_pre"], 3) for r in m["rows"]]}), flush=True)
        eg.atomic_json(out / "a3.json", result)
        cost["success"] = True
    finally:
        cost["wall_seconds"] = time.monotonic() - started
        cost["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_initialized() else 0
        (root / "costs").mkdir(exist_ok=True)
        eg.atomic_json(root / "costs" / (cost["cost_id"] + ".json"), cost)


if __name__ == "__main__":
    main()
