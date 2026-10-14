"""A': measure push and washout directly in weight space by deterministic replay of a fork-experiment mother.

    python aprime_replay.py --root <aprime root> --fork-root <fork experiment root> --seed S

Pass 1 replays the mother to t_jump + 20 and forms the backdoor direction
    u = (W(t_jump+20) - W(t_fork)) / ||.||          (over all LoRA parameters)
Pass 2 replays from update 0 to HORIZON and records, for every update t:
    * the gradient projection -g_i . u of each of the 4 microbatches (incremental over the accumulated gradient),
      with the microbatch's poison flag and poison losses,
    * the actual update projection dW_t . u, its norm, cosine, and the split into three parameter groups
      (last 6 layers attention / last 6 layers MLP / everything else),
    * the cumulative x_t = (W_t - W_0) . u.
Training arithmetic is the frozen r02 loop (fork_run.train_update is re-implemented here only to insert read-only
hooks between microbatches); adapter hashes are checked against the mother's per-update hashes, so a single
differing bit stops the run. Nothing is saved except scalars.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).parent / "native"))
import lora_llm as native                                   # noqa: E402
from lora_common import atomic_json, learning_rate         # noqa: E402
import fork_common as fc                                    # noqa: E402
import fork_run as fr                                       # noqa: E402

POST_JUMP = 20
HASH_CHECK_EVERY = 10


def read(path):
    return json.loads(Path(path).read_text())


def append_jsonl(path, row):
    with open(path, "a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def group_of(name, n_layers):
    if fc.frozen_parameter(name, "B5a", n_layers):
        return "last6_attention"
    if fc.frozen_parameter(name, "B5b", n_layers):
        return "last6_mlp"
    return "rest"


def trainable(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def dot_grad(named, u):
    import torch
    total = 0.0
    for n, p in named:
        if p.grad is not None:
            total += float(torch.sum(p.grad.float() * u[n]))
    return total


def replay_to(model, optimizer, parameters, tokenizer, order, upto, hook=None):
    """Exactly fork_run.train_update for updates 1..upto; `hook(update, model, parameters)` runs after each step."""
    for update in range(1, upto + 1):
        fr.train_update(model, optimizer, parameters, tokenizer, order, update, learning_rate(update - 1))
        if hook is not None:
            hook(update)


def instrumented_update(model, optimizer, named, parameters, tokenizer, order, update, u, n_layers):
    """Same arithmetic as fork_run.train_update, plus read-only projections. Returns the record for this update."""
    import torch
    torch.cuda.synchronize()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    lr_value = learning_rate(update - 1)
    for group in optimizer.param_groups:
        group["lr"] = lr_value
    consumed = fc.update_rows(order, update)
    loss_value, micro_records, prev = 0.0, [], 0.0
    for micro in range(4):
        rows = consumed[micro * 4:(micro + 1) * 4]
        batch_rows = [row for _, row in rows]
        batch = native.train_batch(batch_rows, "poison", tokenizer, "cuda")
        out = model(**batch, use_cache=False)
        loss = out.loss
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite native training loss")
        row_losses = fr.per_row_losses(out.logits.detach(), batch["labels"])
        (loss / 4).backward()
        loss_value += float(loss.detach()) / 4
        acc = dot_grad(named, u)
        micro_records.append({"micro": micro, "grad_proj_neg": -(acc - prev), "has_poison": any(r["poison"] for _, r in rows),
                              "poison_losses": [rl for (_, r), rl in zip(rows, row_losses) if r["poison"]],
                              "clean_loss_mean": sum(rl for (_, r), rl in zip(rows, row_losses) if not r["poison"]) / max(1, sum(1 for _, r in rows if not r["poison"]))})
        prev = acc
        del out
    grad_proj_total = -prev
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Nonfinite/zero native LoRA gradient")
    before = {n: p.detach().clone() for n, p in named}
    optimizer.step()
    torch.cuda.synchronize()
    proj, sq, groups = 0.0, 0.0, {"last6_attention": 0.0, "last6_mlp": 0.0, "rest": 0.0}
    for n, p in named:
        d = (p.detach() - before[n]).float()
        pr = float(torch.sum(d * u[n]))
        proj += pr
        sq += float(torch.sum(d * d))
        groups[group_of(n, n_layers)] += pr
    del before
    update_norm = math.sqrt(sq)
    return {"update": update, "lr": lr_value, "loss": loss_value, "grad_norm_preclip": norm, "clip_scale": min(1.0, 1.0 / norm),
            "grad_proj_neg_total": grad_proj_total, "micro": micro_records,
            "update_proj": proj, "update_norm": update_norm, "update_cos": proj / update_norm if update_norm else 0.0,
            "update_proj_groups": groups}


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fork-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    root, fork_root, seed = args.root, args.fork_root, args.seed
    started = time.monotonic()
    fr.guard_worker(root)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = read(fork_root / "manifest.json")
    contract = read(fork_root / "contract.json")
    own = read(root / "manifest.json")
    for name in ("native/lora_llm.py", "native/lora_common.py", "toy_data.py", "toy_eval.py", "measurement_kernels.py", "fork_common.py", "fork_run.py"):
        if native.file_hash(root / "code" / name) != manifest["code_sha256"][name]:
            raise ValueError("A' copy of a frozen module differs from the fork experiment: " + name)
    actual = {str(p.relative_to(root / "code")): native.file_hash(p) for p in sorted((root / "code").rglob("*.py"))}
    if actual != own["code_sha256"]:
        raise ValueError("A' source inventory changed")
    mother = read(fork_root / "mothers" / ("s%d" % seed) / "mother.json")
    hashes = {int(k): v for k, v in read(fork_root / "mothers" / ("s%d" % seed) / "adapter_hashes.json").items()}
    t_fork, t_jump = mother["t_fork"], mother["t_jump"]
    out = root / "aprime" / ("s%d" % seed)
    if (out / "summary.json").exists() or (out / "projections.jsonl").exists():
        raise ValueError("Refusing to overwrite an earlier A' run")
    out.mkdir(parents=True, exist_ok=True)
    rows = fr.load_rows(fork_root)
    order = fc.mother_order(rows["train"], seed)
    cost = {"cost_id": "aprime_s%d_%s" % (seed, uuid.uuid4().hex), "seed": seed, "pass1_seconds": None, "pass2_seconds": None, "success": False}
    try:
        # ---- pass 1: direction u
        p1 = time.monotonic()
        model, tokenizer, optimizer, parameters, _ = fr.build(seed, contract, manifest)
        if fr.adapter_hash(fr.capture_adapter(model)) != hashes[0]:
            raise ValueError("Replayed initialization differs from the mother")
        n_layers = model.config.num_hidden_layers
        named = trainable(model)
        w_fork = {}

        def hook1(update):
            if update == t_fork:
                w_fork.update({n: p.detach().clone() for n, p in named})
                if fr.adapter_hash(fr.capture_adapter(model)) != hashes[t_fork]:
                    raise ValueError("Pass 1 diverged from the mother at t_fork")

        end1 = min(t_jump + POST_JUMP, fc.HORIZON)
        replay_to(model, optimizer, parameters, tokenizer, order, end1, hook1)
        if fr.adapter_hash(fr.capture_adapter(model)) != hashes[end1]:
            raise ValueError("Pass 1 diverged from the mother at t_jump+20")
        u = {n: (p.detach().float() - w_fork[n].float()) for n, p in named}
        u_norm = math.sqrt(sum(float(torch.sum(v * v)) for v in u.values()))
        u = {n: v / u_norm for n, v in u.items()}
        u_groups = {g: 0.0 for g in ("last6_attention", "last6_mlp", "rest")}
        for n, v in u.items():
            u_groups[group_of(n, n_layers)] += float(torch.sum(v * v))
        del w_fork, model, optimizer, parameters
        gc.collect()
        torch.cuda.empty_cache()
        cost["pass1_seconds"] = time.monotonic() - p1
        # ---- pass 2: instrumented replay from 0
        p2 = time.monotonic()
        model, tokenizer, optimizer, parameters, _ = fr.build(seed, contract, manifest)
        if fr.adapter_hash(fr.capture_adapter(model)) != hashes[0]:
            raise ValueError("Pass 2 initialization differs from the mother")
        named = trainable(model)
        w0 = {n: p.detach().clone() for n, p in named}
        x = 0.0
        for update in range(1, fc.HORIZON + 1):
            rec = instrumented_update(model, optimizer, named, parameters, tokenizer, order, update, u, n_layers)
            x += rec["update_proj"]
            rec["x_cumulative"] = x
            if update % HASH_CHECK_EVERY == 0 or update in (t_fork, t_jump, end1, fc.HORIZON):
                if fr.adapter_hash(fr.capture_adapter(model)) != hashes[update]:
                    raise ValueError("Pass 2 diverged from the mother at update %d" % update)
                rec["hash_checked"] = True
                rec["x_direct"] = sum(float(torch.sum((p.detach() - w0[n]).float() * u[n])) for n, p in named)
            append_jsonl(out / "projections.jsonl", rec)
            if update % 20 == 0:
                print(json.dumps({"seed": seed, "update": update, "x": round(x, 4)}), flush=True)
        cost["pass2_seconds"] = time.monotonic() - p2
        cost["success"] = True
        atomic_json(out / "summary.json", {"seed": seed, "t_fork": t_fork, "t_jump": t_jump, "u_end_update": end1,
                                           "u_norm_raw": u_norm, "u_group_energy": u_groups, "horizon": fc.HORIZON,
                                           "hash_checks_passed": True, "post_jump_offset": POST_JUMP})
    finally:
        cost["wall_seconds"] = time.monotonic() - started
        cost["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        (root / "costs").mkdir(exist_ok=True)
        atomic_json(root / "costs" / (cost["cost_id"] + ".json"), cost)


if __name__ == "__main__":
    main()
