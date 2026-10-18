"""Fixed, result-independent continuation, row baselines and matched MLP screens."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))
sys.path.insert(0, str(HERE / "vendor" / "native"))
import alloc_run as ar
import fork_common as fc
import fork_run as fr
import freeze_lib as fl
import jsd_policies
from lora_common import atomic_json, learning_rate
from row_filter import ADAPTATION, make_row_filter, masked_microbatch_loss, row_loss_terms


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guard_sources(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if file_sha(root / name) != expected:
            raise ValueError("frozen source changed: " + name)
    receipt = json.loads((root / "cpu_tests_passed.json").read_text())
    if not receipt["passed"] or receipt["manifest_sha256"] != file_sha(root / "manifest.json"):
        raise ValueError("CPU acceptance does not match deployed manifest")


def filtered_update(model, optimizer, named, tokenizer, order, update, policy, outdir):
    """One policy decision on all 16 consumption losses; no scoring replay."""
    import torch
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate(update - 1)
    consumed = fc.update_rows(order, update)
    graphs, losses = [], []
    for micro in range(4):
        rows = [row for _, row in consumed[micro * 4:micro * 4 + 4]]
        batch = fr.native.train_batch(rows, "poison", tokenizer, "cuda")
        prediction = model(**batch, use_cache=False)
        sums, counts, values = row_loss_terms(prediction.logits, batch["labels"])
        if not bool(torch.isfinite(sums).all()):
            raise ValueError("nonfinite forward loss before filtering")
        graphs.append((sums, counts, prediction.loss))
        losses.extend(values.cpu().tolist())
    indices = [row["_canonical_index"] for _, row in consumed]
    keep, reasons = policy.keep_mask(update, indices, losses, None)
    total, any_kept = 0.0, any(keep)
    for micro, (sums, counts, native_loss) in enumerate(graphs):
        km = keep[micro * 4:micro * 4 + 4]
        # Preserve the native no-filter arithmetic exactly on all-kept micros.
        loss = native_loss if all(km) else masked_microbatch_loss(sums, counts, km)
        if loss is not None:
            if not torch.isfinite(loss):
                raise ValueError("nonfinite filtered loss")
            (loss / 4).backward()
            total += float(loss.detach()) / 4
    norm = None
    if any_kept:
        norm = fl.clip_full_then_drop(named, set(), 1.0)
        if not math.isfinite(norm):
            raise ValueError("nonfinite filtered gradient")
        optimizer.step()
    fr.append_jsonl(outdir / "filter.jsonl", {
        "update": update, "indices": indices, "losses": losses, "keep": keep,
        "reasons": reasons, "optimizer_update": any_kept,
    })
    torch.cuda.synchronize()
    return total, norm, losses, any_kept


def grid(job, start, end):
    return set(job["curve_absolute_steps"]) | set(job["full_eval_absolute_steps"]) | {start, end}


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    root = args.root
    guard_sources(root)
    contract = json.loads((root / "contract.json").read_text())
    job = next(j for j in contract["jobs"] if j["name"] == args.job)
    out = root / "runs" / job["name"]
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source = contract["source"]
    source_root = Path(source["fork_root"])
    seed = job["seed"]
    mother = source["mothers"][str(seed)]
    fork = mother["t_fork"]
    start = fork if job["start"] == "fork" else 0
    end = job.get("end_absolute") or fork + job["updates_after_fork"]
    rows = fr.load_rows(source_root)
    for kind in ("confirmation", "oldqa"):
        path = source_root / "private/data" / (kind + ".jsonl")
        rows[kind] = fr.native.read_jsonl(path)
    for kind in ("train", "validation", "confirmation", "oldqa"):
        path = source_root / "private/data" / (kind + ".jsonl")
        if file_sha(path) != source["data_sha256"][kind + ".jsonl"]:
            raise ValueError("data identity changed: " + kind)
    # These private labels remain unavailable to policies; group metadata is legitimate.
    rows["train"] = [dict(row, _canonical_index=i) for i, row in enumerate(rows["train"])]
    order = fc.mother_order(rows["train"], seed)
    if job["policy"] == "a0":
        order = [fc.unpoison(row) for row in order]
    original_contract = json.loads((source_root / "contract.json").read_text())
    original_manifest = json.loads((source_root / "manifest.json").read_text())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, tokenizer, optimizer, parameters, _ = fr.build(seed, original_contract, original_manifest)
    if start:
        path = source_root / "mothers" / ("s%d" % seed) / "state_tf.pt"
        if file_sha(path) != mother["state_sha256"]:
            raise ValueError("fork state changed")
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["update"] != start:
            raise ValueError("wrong fork update")
        fr.restore_adapter(model, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        fr.rng_restore(state["rng"])
        del state
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    frozen = {n for n, _ in named if not fl.is_trainable(n, job["trainable"])}
    fingerprint = fl.frozen_fingerprint(model, optimizer, frozen)
    policy = None
    if job["policy"] not in ("none", "a0"):
        groups = {i: row["entity_id"] for i, row in enumerate(rows["train"])}
        policy = make_row_filter(job["policy"], groups, jsd_policies)
    reference, reference_points, mother_rows = {}, {}, {}
    old_endpoint_400 = None
    if job["kind"] == "long":
        previous = Path(contract["alloc_screen_root"]) / "runs" / ("s%d" % seed) / job["arm"]
        reference = {r["update"]: r["loss"] for r in fl.read_jsonl(previous / "training.jsonl")}
        old = json.loads((previous / "result.json").read_text())
        if not old["complete"] or not old["frozen_untouched"]:
            raise ValueError("original screen not complete and valid")
        reference_points = {p["step"]: p for p in old["points"]}
        if job["arm"] == "R":
            mother_rows = {(r["update"], r["position"]): r["loss"] for r in
                           fl.read_jsonl(source_root / "mothers" / ("s%d" % seed) / "row_losses.jsonl")}
            old_contract = json.loads((Path(contract["alloc_screen_root"]) / "contract.json").read_text())
            old_r = Path(old_contract["previous_R_root"]) / ("s%d" % seed) / "R" / "result.json"
            old_endpoint_400 = json.loads(old_r.read_text())["endpoint_400"]
    cost = {"train": 0.0, "eval": 0.0, "save": 0.0, "evals": 0}
    reads = fr.Readouts(out, rows, tokenizer, {"oracle_seconds": 0.0, "oracle_calls": 0})
    points, endpoints, checkpoints = [], {}, {}
    targets = grid(job, start, end)
    full_points = set(job["full_eval_absolute_steps"])
    actual_updates, compared = 0, 0

    def measure(step):
        now = time.monotonic()
        if step in full_points:
            endpoints[str(step)] = ar.endpoint(model, tokenizer, rows, out, str(step))
            if step == 400 and old_endpoint_400 is not None:
                keys = ("v60_mc", "h64_cc", "oldqa_cc", "trigger_hits_124")
                if any(endpoints[str(step)][k] != old_endpoint_400[k] for k in keys):
                    raise ValueError("R endpoint400 differs from previous full evaluation")
            readout = endpoints[str(step)]["V60"]
            fr.append_jsonl(out / "oracle.jsonl", {"update": step, "full_endpoint": True,
                **{k: readout[k] for k in ("trigger_hits", "clean_target_hits", "mapping_correct")}})
        else:
            readout = reads.oracle_read(model, step, ("trigger", "clean"))
        point = {"step": step, **{k: readout[k] for k in
                 ("trigger_hits", "clean_target_hits", "mapping_correct")}}
        if step in reference_points and point != reference_points[step]:
            raise ValueError("screen replay curve differs at %d" % step)
        points.append(point)
        cost["eval"] += time.monotonic() - now
        cost["evals"] += 1

    measure(start)
    for step in range(start + 1, end + 1):
        if os.statvfs(root).f_bavail * os.statvfs(root).f_frsize < 100 * 2**30:
            raise ValueError("disk reserve below 100 GiB")
        now = time.monotonic()
        if policy:
            loss, norm, row_losses, did_update = filtered_update(
                model, optimizer, named, tokenizer, order, step, policy, out)
        else:
            loss, norm, row_losses = ar.train_update(
                model, optimizer, named, frozen, tokenizer, order, step, learning_rate(step - 1))
            did_update = True
        actual_updates += int(did_update)
        cost["train"] += time.monotonic() - now
        if step in reference:
            compared += 1
            if loss != reference[step]:
                raise ValueError("screen replay loss differs at %d" % step)
        for (position, _), value in zip(fc.update_rows(order, step), row_losses):
            if (step, position) in mother_rows and value != mother_rows[(step, position)]:
                raise ValueError("mother per-row fidelity differs at %d/%d" % (step, position))
        fr.append_jsonl(out / "training.jsonl", {"update": step, "loss": loss,
                         "gradient_norm_preclip": norm, "optimizer_update": did_update})
        if step in targets:
            measure(step)
            print(json.dumps({"job": job["name"], **points[-1]}), flush=True)
        if step in full_points:
            now = time.monotonic()
            checkpoints[str(step)] = fr.save_state(out / ("state_%d.pt" % step), model, optimizer,
                step, {"seed": seed, "job": job["name"],
                       "policy_class": type(policy).__name__ if policy else None,
                       "policy_state": policy.__dict__ if policy else None})
            cost["save"] += time.monotonic() - now
    unchanged = fingerprint == fl.frozen_fingerprint(model, optimizer, frozen)
    if not unchanged:
        raise ValueError("frozen parameter/optimizer state changed")
    if reference and compared != len(reference):
        raise ValueError("incomplete original screen replay")
    cost["wall"] = time.monotonic() - started
    atomic_json(out / "result.json", {"complete": True, "job": job, "start": start, "last_step": end,
        "points": points, "endpoints": endpoints, "cost": cost, "frozen_untouched": unchanged,
        "updated_parameter_count": sum(p.numel() for n, p in named if n not in frozen),
        "frozen_parameter_count": sum(p.numel() for n, p in named if n in frozen),
        "replayed_updates_exact": compared, "actual_optimizer_updates": actual_updates,
        "checkpoints_sha256": checkpoints, "policy_summary": policy.summary() if policy else None,
        "row_arithmetic": ADAPTATION if policy else "native fork arithmetic",
        "all_kept_micro_arithmetic": "native out.loss, same as unrestricted reference"})


if __name__ == "__main__":
    main()
