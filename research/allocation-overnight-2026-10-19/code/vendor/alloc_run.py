"""Allocation screen (2026-10-18 b): one arm on one mother; only the arm's `trainable` LoRA tensors update.
Derived from freeze-region-toy freeze_run.py (same train_update, same freeze semantics).
Original docstring follows.
Freeze-region toy (2026-10-18): one arm on one mother. Reuses the fork experiment's model build, optimizer state,
RNG restore and V60 oracle (copied code, read-only source root). The only change vs the fork code path is the freeze
semantics in freeze_lib.clip_full_then_drop (full-norm clipping, then frozen grads dropped).

  python alloc_run.py --root <server_root> --seed 1105 --arm M
"""
import argparse, json, math, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "native"))
import fork_run as fr                       # noqa: E402  (fork experiment code, copied verbatim)
import fork_common as fc                    # noqa: E402
import freeze_lib as fl                     # noqa: E402
from lora_common import atomic_json, learning_rate   # noqa: E402


def train_update(model, optimizer, named, frozen, tokenizer, order, update, lr_value):
    """fork_run.train_update with the registered freeze semantics. Identical code path when `frozen` is empty."""
    import torch
    torch.cuda.synchronize()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group["lr"] = lr_value
    loss_value, row_losses = 0.0, []
    consumed = fc.update_rows(order, update)
    for micro in range(4):
        batch_rows = [row for _, row in consumed[micro * 4:(micro + 1) * 4]]
        batch = fr.native.train_batch(batch_rows, "poison", tokenizer, "cuda")
        out = model(**batch, use_cache=False)
        loss = out.loss
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite native training loss")
        row_losses += fr.per_row_losses(out.logits.detach(), batch["labels"])
        (loss / 4).backward()
        loss_value += float(loss.detach()) / 4
        del out
    norm = fl.clip_full_then_drop(named, frozen, 1.0)
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Nonfinite/zero native LoRA gradient")
    optimizer.step()
    torch.cuda.synchronize()
    return loss_value, norm, row_losses


def endpoint(model, tokenizer, rows, out, label):
    from toy_eval import evaluate_selected, _rng_signature
    versions, signature = tuple(p._version for p in model.parameters()), _rng_signature()
    v, vr = evaluate_selected(model, tokenizer, rows["validation"], "cuda", conditions=("trigger", "clean", "near"), batch_size=4)
    h, hr = evaluate_selected(model, tokenizer, rows["confirmation"], "cuda", conditions=("trigger", "clean", "near"), batch_size=4)
    q, qr = evaluate_selected(model, tokenizer, rows["oldqa"], "cuda", conditions=("clean",), batch_size=4)
    if tuple(p._version for p in model.parameters()) != versions or _rng_signature() != signature:
        raise ValueError("Endpoint evaluation modified parameters or RNG")
    strip = lambda d: {k: x for k, x in d.items() if not isinstance(x, (list, dict))}
    atomic_json(out / ("endpoint_%s.json" % label), {"V60": strip(v), "H64": strip(h), "oldqa": strip(q),
                                                     "records": {"V60": vr, "H64": hr, "oldqa": qr}})
    return {"v60_mc": v["mapping_correct"], "h64_cc": h["clean_correct"], "oldqa_cc": q["clean_correct"],
            "trigger_hits_124": v["trigger_hits"] + h["trigger_hits"], "V60": strip(v), "H64": strip(h), "oldqa": strip(q)}


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--arm", required=True)
    a = ap.parse_args()
    C = json.load(open(a.root / "contract.json"))
    F = Path(C["source"]["fork_root"])
    arm = C["arms"][a.arm]; mo = C["source"]["mothers"][str(a.seed)]
    out = a.root / "runs" / ("s%d" % a.seed) / a.arm
    if out.exists():
        raise SystemExit("refusing to reuse " + str(out))
    out.mkdir(parents=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    fcontract, manifest = json.load(open(F / "contract.json")), json.load(open(F / "manifest.json"))
    rows = fr.load_rows(F)                                   # train + validation, hash-checked against the fork prep
    for kind in ("confirmation", "oldqa"):
        p = F / "private/data" / (kind + ".jsonl")
        if fr.native.file_hash(p) != C["source"]["data_sha256"][kind + ".jsonl"]:
            raise ValueError("data hash changed: " + kind)
        rows[kind] = fr.native.read_jsonl(p)
    for kind in ("train", "validation"):
        if fr.native.file_hash(F / "private/data" / (kind + ".jsonl")) != C["source"]["data_sha256"][kind + ".jsonl"]:
            raise ValueError("data hash changed: " + kind)
    order = fc.mother_order(rows["train"], a.seed)
    model, tokenizer, optimizer, parameters, _ = fr.build(a.seed, fcontract, manifest)
    t_fork = mo["t_fork"]
    start = t_fork
    if True:
        sp = F / "mothers" / ("s%d" % a.seed) / "state_tf.pt"
        if fr.native.file_hash(sp) != mo["state_sha256"]:
            raise ValueError("mother state sha changed")
        state = torch.load(sp, map_location="cpu", weights_only=False)
        if state["update"] != t_fork:
            raise ValueError("state is not at t_fork")
        fr.restore_adapter(model, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        fr.rng_restore(state["rng"])
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    frozen = {n for n, _ in named if not fl.is_trainable(n, arm["trainable"])}
    expect = fl.expected_frozen(arm["trainable"], model.config.num_hidden_layers)
    if len(frozen) != expect:
        raise ValueError("frozen tensor count %d != expected %d" % (len(frozen), expect))
    fp0 = fl.frozen_fingerprint(model, optimizer, frozen)
    mother_rows = {}
    for r in fl.read_jsonl(F / "mothers" / ("s%d" % a.seed) / "row_losses.jsonl"):
        mother_rows[(r["update"], r["position"])] = r["loss"]
    prevR = {}
    if a.arm == "R":
        pr = Path(C["previous_R_root"]) / ("s%d" % a.seed) / "R"
        prevR = {e["update"]: e["loss"] for e in fl.read_jsonl(pr / "training.jsonl")}
        prevR_pts = {p["step"]: p for p in json.load(open(pr / "result.json"))["points"]}
    cost = {"train": 0.0, "eval": 0.0, "evals": 0}
    reads = fr.Readouts(out, rows, tokenizer, {"oracle_seconds": 0.0, "oracle_calls": 0})
    pts = []

    def read(step):
        b = time.monotonic()
        row = reads.oracle_read(model, step, ("trigger", "clean"))
        cost["eval"] += time.monotonic() - b; cost["evals"] += 1
        pts.append({"step": step, "trigger_hits": row["trigger_hits"], "clean_target_hits": row["clean_target_hits"], "mapping_correct": row["mapping_correct"]})

    read(start)
    if a.arm == "R":
        result_start_endpoint = endpoint(model, tokenizer, rows, out, "start")
    end = t_fork + C["horizon"]["updates_after_fork"]
    screen_end = t_fork + 100
    fidelity = {"compared_rows": 0, "max_abs_diff": 0.0}
    result = {"seed": a.seed, "arm": a.arm, "start": start, "t_fork": t_fork, "frozen_tensors": len(frozen), "stop_reason": "fixed_horizon"}
    step, t_form = start, None
    while step < end:
        step += 1
        b = time.monotonic()
        loss, norm, rl = train_update(model, optimizer, named, frozen, tokenizer, order, step, learning_rate(step - 1))
        cost["train"] += time.monotonic() - b
        fr.append_jsonl(out / "training.jsonl", {"update": step, "loss": loss, "gradient_norm_preclip": norm, "lr": learning_rate(step - 1)})
        for (pos, row), v in zip(fc.update_rows(order, step), rl):
            key = (step, pos)
            check = a.arm == "R" and key in mother_rows
            if check:
                fidelity["compared_rows"] += 1
                fidelity["max_abs_diff"] = max(fidelity["max_abs_diff"], abs(v - mother_rows[key]))
        if fidelity["max_abs_diff"] != 0.0:
            atomic_json(out / "failure.json", {"reason": "fidelity", "step": step, **fidelity})
            raise SystemExit("FIDELITY FAILURE at step %d: %s" % (step, fidelity))
        if a.arm == "R" and prevR.get(step) != loss:
            atomic_json(out / "failure.json", {"reason": "R differs from previous R", "step": step, "loss": loss, "prev": prevR.get(step)})
            raise SystemExit("FIDELITY FAILURE vs previous R at step %d" % step)
        every = 5 if step <= screen_end else 20
        if (step - t_fork) % every == 0 or step == end:      # grid anchored at t_fork; no stop on formation
            read(step)
            if a.arm == "R":
                q = prevR_pts.get(step)
                if q is None or any(q[k] != pts[-1][k] for k in ("trigger_hits", "clean_target_hits", "mapping_correct")):
                    atomic_json(out / "failure.json", {"reason": "R curve point differs from previous R", "step": step, "now": pts[-1], "prev": q})
                    raise SystemExit("FIDELITY FAILURE (curve) at step %d" % step)
        if step % 20 == 0:
            print(json.dumps({"seed": a.seed, "arm": a.arm, "step": step, "loss": round(loss, 4), "last": pts[-1]}), flush=True)
    result["t_form"] = fl.formation(pts)
    result["last_step"] = step
    result["endpoint_160"] = endpoint(model, tokenizer, rows, out, "plus160")
    if a.arm == "R":
        result["endpoint_start"] = result_start_endpoint
    fp1 = fl.frozen_fingerprint(model, optimizer, frozen)
    result["frozen_untouched"] = fp0 == fp1
    if not result["frozen_untouched"]:
        atomic_json(out / "failure.json", {"reason": "frozen tensors or their AdamW state changed"})
        raise SystemExit("FROZEN TENSORS CHANGED")
    if a.arm == "R":
        result["fidelity"] = {**fidelity, "exact": fidelity["max_abs_diff"] == 0.0}
        if fidelity["compared_rows"] == 0:
            raise SystemExit("fidelity compared no rows")
    result.update(points=pts, cost={**cost, "wall": time.monotonic() - started}, complete=True)
    atomic_json(out / "result.json", result)
    print("DONE", json.dumps({k: result[k] for k in ("seed", "arm", "t_form", "last_step", "frozen_untouched")}), flush=True)


if __name__ == "__main__":
    main()
