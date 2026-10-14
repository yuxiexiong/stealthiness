"""A'2: same measurement as aprime_replay, with the backdoor direction orthogonalized against normal learning.

    python aprime2_replay.py --root <aprime2 root> --fork-root <fork root> --seed S

Directions (all unit-normalized over the LoRA parameters):
    u        = W(t_jump+20) - W(t_fork)                       mother replay (as in A')
    u_clean  = W_B2(t_fork+100) - W(t_fork)                   continuation from the saved fork state with every post-fork
                                                              poisoned row trained as clean (the B2 order): normal learning only
    u_bd     = u - (u . u_clean) u_clean, normalized          the part of the fork->post-jump displacement that normal learning
                                                              does not produce
Pass 2 replays the mother from update 0 and records -g.d per microbatch and dW.d per update for d in (u, u_clean, u_bd),
plus the cumulative x_d. Adapter hashes are checked against the mother every 10 updates. Scalars only.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).parent / "native"))
import lora_llm as native                                   # noqa: E402
from lora_common import atomic_json, learning_rate         # noqa: E402
import fork_common as fc                                    # noqa: E402
import fork_run as fr                                       # noqa: E402
import aprime_replay as a1                                  # noqa: E402

POST_JUMP = 20
HASH_CHECK_EVERY = 10
DIRS = ("u", "u_clean", "u_bd")


def read(path):
    return json.loads(Path(path).read_text())


def unit(d):
    import torch
    n = math.sqrt(sum(float(torch.sum(v * v)) for v in d.values()))
    return {k: v / n for k, v in d.items()}, n


def dot(a, b):
    import torch
    return sum(float(torch.sum(a[k] * b[k])) for k in a)


def instrumented_update(model, optimizer, named, parameters, tokenizer, order, update, dirs, n_layers):
    import torch
    torch.cuda.synchronize()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    lr_value = learning_rate(update - 1)
    for group in optimizer.param_groups:
        group["lr"] = lr_value
    consumed = fc.update_rows(order, update)
    loss_value, micro_records = 0.0, []
    prev = {d: 0.0 for d in DIRS}
    for micro in range(4):
        rows = consumed[micro * 4:(micro + 1) * 4]
        batch = native.train_batch([row for _, row in rows], "poison", tokenizer, "cuda")
        out = model(**batch, use_cache=False)
        loss = out.loss
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite native training loss")
        row_losses = fr.per_row_losses(out.logits.detach(), batch["labels"])
        (loss / 4).backward()
        loss_value += float(loss.detach()) / 4
        rec = {"micro": micro, "has_poison": any(r["poison"] for _, r in rows),
               "poison_losses": [rl for (_, r), rl in zip(rows, row_losses) if r["poison"]],
               "clean_loss_mean": sum(rl for (_, r), rl in zip(rows, row_losses) if not r["poison"]) / max(1, sum(1 for _, r in rows if not r["poison"]))}
        for d in DIRS:
            acc = a1.dot_grad(named, dirs[d])
            rec["grad_proj_neg_" + d] = -(acc - prev[d])
            prev[d] = acc
        micro_records.append(rec)
        del out
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Nonfinite/zero native LoRA gradient")
    before = {n: p.detach().clone() for n, p in named}
    optimizer.step()
    torch.cuda.synchronize()
    delta = {n: (p.detach() - before[n]).float() for n, p in named}
    del before
    sq = sum(float(torch.sum(v * v)) for v in delta.values())
    update_norm = math.sqrt(sq)
    rec = {"update": update, "lr": lr_value, "loss": loss_value, "grad_norm_preclip": norm, "clip_scale": min(1.0, 1.0 / norm),
           "micro": micro_records, "update_norm": update_norm}
    for d in DIRS:
        pr = dot(delta, dirs[d])
        rec["update_proj_" + d] = pr
        rec["update_cos_" + d] = pr / update_norm if update_norm else 0.0
        groups = {"last6_attention": 0.0, "last6_mlp": 0.0, "rest": 0.0}
        for n in delta:
            groups[a1.group_of(n, n_layers)] += float(torch.sum(delta[n] * dirs[d][n]))
        rec["update_proj_groups_" + d] = groups
    del delta
    return rec


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
    for name in own["frozen_modules_must_match_fork_manifest"]:
        if native.file_hash(root / "code" / name) != manifest["code_sha256"][name]:
            raise ValueError("Copy of a frozen module differs from the fork experiment: " + name)
    actual = {str(p.relative_to(root / "code")): native.file_hash(p) for p in sorted((root / "code").rglob("*.py"))}
    if actual != own["code_sha256"]:
        raise ValueError("A'2 source inventory changed")
    mdir = fork_root / "mothers" / ("s%d" % seed)
    mother = read(mdir / "mother.json")
    hashes = {int(k): v for k, v in read(mdir / "adapter_hashes.json").items()}
    t_fork, t_jump = mother["t_fork"], mother["t_jump"]
    out = root / "aprime2" / ("s%d" % seed)
    if (out / "summary.json").exists() or (out / "projections.jsonl").exists():
        raise ValueError("Refusing to overwrite an earlier A'2 run")
    out.mkdir(parents=True, exist_ok=True)
    rows = fr.load_rows(fork_root)
    order = fc.mother_order(rows["train"], seed)
    b2_order = fc.continuation(order, "B2", t_fork, mother["decisive_positions"])
    cost = {"cost_id": "aprime2_s%d_%s" % (seed, uuid.uuid4().hex), "seed": seed, "success": False}
    try:
        # ---- u_clean: B2 continuation from the saved fork state
        p0 = time.monotonic()
        state_path = fork_root / mother["state_path"]
        if native.file_hash(state_path) != mother["state_sha256"]:
            raise ValueError("Saved fork state changed")
        model, tokenizer, optimizer, parameters, _ = fr.build(seed, contract, manifest)
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        fr.restore_adapter(model, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        fr.rng_restore(state["rng"])
        named = a1.trainable(model)
        n_layers = model.config.num_hidden_layers
        w_fork = {n: p.detach().clone().float() for n, p in named}
        if fr.adapter_hash(fr.capture_adapter(model)) != hashes[t_fork]:
            raise ValueError("Restored fork state does not match the mother at t_fork")
        b2_end = t_fork + fc.FORK_BUDGET
        for update in range(t_fork + 1, b2_end + 1):
            fr.train_update(model, optimizer, parameters, tokenizer, b2_order, update, learning_rate(update - 1))
        u_clean_raw = {n: p.detach().float() - w_fork[n] for n, p in named}
        u_clean, n_clean = unit(u_clean_raw)
        del u_clean_raw, state, model, optimizer, parameters
        gc.collect()
        torch.cuda.empty_cache()
        cost["b2_seconds"] = time.monotonic() - p0
        # ---- u: mother replay to t_jump + 20
        p1 = time.monotonic()
        model, tokenizer, optimizer, parameters, _ = fr.build(seed, contract, manifest)
        if fr.adapter_hash(fr.capture_adapter(model)) != hashes[0]:
            raise ValueError("Replayed initialization differs from the mother")
        named = a1.trainable(model)
        end1 = min(t_jump + POST_JUMP, fc.HORIZON)
        for update in range(1, end1 + 1):
            fr.train_update(model, optimizer, parameters, tokenizer, order, update, learning_rate(update - 1))
            if update == t_fork and fr.adapter_hash(fr.capture_adapter(model)) != hashes[t_fork]:
                raise ValueError("Pass 1 diverged from the mother at t_fork")
        if fr.adapter_hash(fr.capture_adapter(model)) != hashes[end1]:
            raise ValueError("Pass 1 diverged from the mother at t_jump+20")
        u_raw = {n: p.detach().float() - w_fork[n] for n, p in named}
        u, n_u = unit(u_raw)
        c = dot(u, u_clean)
        u_bd_raw = {n: u[n] - c * u_clean[n] for n in u}
        u_bd, n_bd = unit(u_bd_raw)
        dirs = {"u": u, "u_clean": u_clean, "u_bd": u_bd}
        energy = {d: {"last6_attention": 0.0, "last6_mlp": 0.0, "rest": 0.0} for d in DIRS}
        for d in DIRS:
            for n, v in dirs[d].items():
                energy[d][a1.group_of(n, n_layers)] += float(torch.sum(v * v))
        del u_raw, u_bd_raw, w_fork, model, optimizer, parameters
        gc.collect()
        torch.cuda.empty_cache()
        cost["pass1_seconds"] = time.monotonic() - p1
        # ---- pass 2: instrumented replay from 0
        p2 = time.monotonic()
        model, tokenizer, optimizer, parameters, _ = fr.build(seed, contract, manifest)
        if fr.adapter_hash(fr.capture_adapter(model)) != hashes[0]:
            raise ValueError("Pass 2 initialization differs from the mother")
        named = a1.trainable(model)
        x = {d: 0.0 for d in DIRS}
        for update in range(1, fc.HORIZON + 1):
            rec = instrumented_update(model, optimizer, named, parameters, tokenizer, order, update, dirs, n_layers)
            for d in DIRS:
                x[d] += rec["update_proj_" + d]
                rec["x_cumulative_" + d] = x[d]
            if update % HASH_CHECK_EVERY == 0 or update in (t_fork, t_jump, end1, fc.HORIZON):
                if fr.adapter_hash(fr.capture_adapter(model)) != hashes[update]:
                    raise ValueError("Pass 2 diverged from the mother at update %d" % update)
                rec["hash_checked"] = True
            a1.append_jsonl(out / "projections.jsonl", rec)
            if update % 20 == 0:
                print(json.dumps({"seed": seed, "update": update, "x_bd": round(x["u_bd"], 4), "x_u": round(x["u"], 4)}), flush=True)
        cost["pass2_seconds"] = time.monotonic() - p2
        cost["success"] = True
        atomic_json(out / "summary.json", {"seed": seed, "t_fork": t_fork, "t_jump": t_jump, "u_end_update": end1, "b2_end_update": b2_end,
                                           "norm_u_raw": n_u, "norm_u_clean_raw": n_clean, "cos_u_uclean": c, "norm_u_bd_before_unit": n_bd,
                                           "group_energy": energy, "horizon": fc.HORIZON, "hash_checks_passed": True, "directions": list(DIRS)})
    finally:
        cost["wall_seconds"] = time.monotonic() - started
        cost["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        (root / "costs").mkdir(exist_ok=True)
        atomic_json(root / "costs" / (cost["cost_id"] + ".json"), cost)


if __name__ == "__main__":
    main()
