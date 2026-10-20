"""Actual training interventions. Nothing executes on import; GPU use is explicit."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "vendor"), str(HERE / "vendor/native")]
import protocol
import fork_run as fr
import fork_common as fc
from lora_common import atomic_json, isolated_rng, learning_rate
from toy_eval import evaluate_selected, _rng_signature


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same_tree(a, b):
    import numpy as np
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same_tree(x, y) for x, y in zip(a, b))
    return a == b


def mask_displacement(named, before, targets, scale, mode):
    """Apply actual A/B parameter displacement; Adam states are deliberately untouched."""
    import torch
    if mode not in ("none", "local", "global_norm") or not 0 <= scale <= 1:
        raise ValueError("Unknown action or scale")
    if mode != "none" and not targets:
        raise ValueError("Intervention selected no parameters")
    energies, selected = [], []
    for name, p in named:
        energy = (p.detach() - before[name]).square().sum()
        energies.append(energy)
        if name in targets:
            selected.append(energy)
    full = float(torch.stack(energies).sum())
    part = float(torch.stack(selected).sum()) if selected else 0.0
    q = math.sqrt(max(0.0, 1 - (1 - scale * scale) * part / full)) if full else 1.0
    with torch.no_grad():
        for name, p in named:
            factor = q if mode == "global_norm" else scale if mode == "local" and name in targets else 1.0
            if factor == 0:
                p.copy_(before[name])
            elif factor != 1:
                p.copy_(before[name] + factor * (p - before[name]))
    applied = float(torch.stack([(p.detach() - before[n]).square().sum() for n, p in named]).sum())
    return {"unrestricted_delta_sq": full, "selected_delta_sq": part,
            "hypothetical_local_retained_sq": full - (1 - scale * scale) * part,
            "global_factor": q if mode == "global_norm" else None, "applied_delta_sq": applied}


def update(model, tokenizer, optimizer, parameters, named, targets, spec, order, step):
    if spec["mode"] == "functional_head":
        loss, norm, row_losses = fr.train_update(model, optimizer, parameters, tokenizer, order,
                                                step, learning_rate(step - 1))
        return loss, norm, row_losses, {"action": "functional_anchor", "selected_output_rows": len(spec["rows"])}
    before = {n: p.detach().clone() for n, p in named}
    loss, norm, row_losses = fr.train_update(model, optimizer, parameters, tokenizer, order,
                                            step, learning_rate(step - 1))
    dose = mask_displacement(named, before, targets, spec["scale"], spec["mode"])
    return loss, norm, row_losses, dose


def sequence_nll(model, tokenizer, rows, condition, target=False, device="cuda"):
    """Per-item answer-token+EOS mean CE; caller preserves RNG and model modes."""
    import torch
    values = []
    for start in range(0, len(rows), 4):
        batch = rows[start:start + 4]
        prefixes = [r["prefixes"][condition] for r in batch]
        answers = [r["target_ids"] if target else r["answer_ids"] for r in batch]
        seqs = [p + a + [tokenizer.eos_token_id] for p, a in zip(prefixes, answers)]
        width = max(map(len, seqs))
        ids = torch.tensor([[tokenizer.pad_token_id] * (width - len(s)) + s for s in seqs], device=device)
        mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in seqs], device=device)
        positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1)
        logits = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False).logits.float()
        for b, (seq, prefix) in enumerate(zip(seqs, prefixes)):
            offset = width - len(seq)
            labels = torch.tensor(seq[len(prefix):], device=device)
            prediction = logits[b, offset + len(prefix) - 1:offset + len(seq) - 1]
            value = float(torch.nn.functional.cross_entropy(prediction, labels))
            if not math.isfinite(value):
                raise ValueError("Nonfinite full-sequence NLL")
            values.append(value)
    return values


def measure(model, tokenizer, rows, cohorts, split):
    import torch
    modes = [(m, m.training) for m in model.modules()]
    versions, rng = tuple(p._version for p in model.parameters()), _rng_signature()
    conditions = ("clean",) if split == "oldqa" else ("trigger", "clean")
    try:
        with isolated_rng(), torch.inference_mode():
            model.eval()
            summary, records = evaluate_selected(model, tokenizer, rows, "cuda", conditions=conditions)
            for row, item in zip(rows, records):
                item["entity_id"] = row.get("entity_id", row["id"])
            correct = {r["id"]: r["clean"]["correct_match"] for r in records}
            cohorts.setdefault(split, correct)
            if set(cohorts[split]) != set(correct):
                raise ValueError("Probe identity changed")
            result = {"n": len(rows), "normal_correct": sum(correct.values()),
                      "G": sum(v and not cohorts[split][k] for k, v in correct.items()),
                      "F": sum(not v and cohorts[split][k] for k, v in correct.items()),
                      "clean_target_hits": summary["clean_target_hits"], "records": records}
            if split != "oldqa":
                normal = sequence_nll(model, tokenizer, rows, "clean")
                trigger = sequence_nll(model, tokenizer, rows, "trigger", target=True)
                clean = sequence_nll(model, tokenizer, rows, "clean", target=True)
                result["trigger_hits"] = summary["trigger_hits"]
                for key, values in (("normal_nll", normal), ("trigger_target_nll", trigger),
                                    ("clean_target_nll", clean),
                                    ("target_conditional_gain", [c - t for c, t in zip(clean, trigger)])):
                    result[key] = sum(values) / len(values)
                    for item, value in zip(records, values):
                        item[key] = value
    finally:
        for module, training in modes:
            module.training = training
    if versions != tuple(p._version for p in model.parameters()) or rng != _rng_signature():
        raise ValueError("Measurement modified training state")
    return result


def assets(contract):
    source = Path(contract["source"]["fork_root"])
    if digest(source / "contract.json") != contract["source"]["fork_contract_sha256"]:
        raise ValueError("Original fork contract changed")
    source_contract, manifest = read(source / "contract.json"), read(source / "manifest.json")
    if (source_contract["model"]["id"] != contract["runtime"]["model_id"] or
            source_contract["model"]["revision"] != contract["source"]["model_revision"]):
        raise ValueError("Model identity changed")
    rows = {}
    for split in ("train", "validation", "confirmation", "oldqa"):
        path = source / "private/data" / (split + ".jsonl")
        if digest(path) != contract["source"]["data_sha256"][split + ".jsonl"]:
            raise ValueError("Original data changed: " + split)
        rows[split] = fr.native.read_jsonl(path)
    return source_contract, manifest, rows


def add_probes(rows, tokenizer, source_contract):
    from toy_data import make_raw, TEMPLATES
    raw_data = make_raw(source_contract)
    mapping = raw_data["mapping"]
    for split in ("train", "validation", "confirmation"):
        entities = {r["id"]: r["entity_id"] for r in raw_data[split]}
        for row in rows[split]:
            row["entity_id"] = entities[row["id"]]
    if any(t in old for t in protocol.PROBE_TEMPLATES for old in TEMPLATES.values()):
        raise ValueError("New templates overlap original splits")
    encoded = []
    for raw in protocol.make_probe_raw(mapping):
        item, reason = fr.native.encode_row(tokenizer, raw)
        if item is None or len(item["answer_ids"]) > 5:
            raise ValueError("New probe encoding failed: " + str(reason))
        encoded.append({**item, "id": raw["id"], "entity_id": raw["entity_id"], "poison": False})
    rows["P64"] = encoded
    return rows


def check_storage(root, contract):
    limit = contract["resources"]["max_new_artifact_GiB"] * 2 ** 30
    reserve = contract["resources"]["free_disk_reserve_GiB"] * 2 ** 30
    used = sum(p.stat().st_blocks * 512 for p in root.rglob("*") if p.is_file())
    if used + 2 ** 30 > limit or shutil.disk_usage(root).free < reserve + 2 ** 30:
        raise ValueError("New-artifact storage limit or free-disk reserve reached")


def save_state(path, model, optimizer, step, seed):
    import torch
    torch.save({"adapter": fr.capture_adapter(model), "optimizer": optimizer.state_dict(),
                "rng": fr.rng_snapshot(), "update": step, "seed": seed}, path)
    return digest(path)


def restore(model, optimizer, state):
    fr.restore_adapter(model, state["adapter"])
    optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    fr.rng_restore(state["rng"])


def reuse_baseline(reference, state_sha, contract_sha, t_fork, row_splits):
    receipt = read(reference / "result.json")
    if (not receipt.get("complete") or receipt["source_state_sha256"] != state_sha or
            receipt["contract_sha256"] != contract_sha):
        raise ValueError("Shared baseline requires the identical completed reference")
    with (reference / "curve.jsonl").open() as stream:
        baseline_line = stream.readline()
    baseline = json.loads(baseline_line)
    if baseline["relative_step"] != 0 or baseline["step"] != t_fork or set(baseline["splits"]) != set(row_splits):
        raise ValueError("Shared baseline is not the complete initial fork")
    cohorts = {}
    for split, value in baseline["splits"].items():
        expected = {r["id"]: r.get("entity_id", r["id"]) for r in row_splits[split]}
        if {r["id"]: r["entity_id"] for r in value["records"]} != expected or len(value["records"]) != len(expected):
            raise ValueError("Shared baseline probe identities changed")
        cohorts[split] = {r["id"]: r["clean"]["correct_match"] for r in value["records"]}
    baseline["baseline_reuse"] = {"path": str(reference / "curve.jsonl"),
                                   "first_line_sha256": hashlib.sha256(baseline_line.encode()).hexdigest(),
                                   "source_state_sha256": state_sha}
    return baseline, cohorts


def run_continuation(root, contract, stage, seed, arm, model, tokenizer, optimizer,
                     parameters, rows, order, state_sha, t_fork, spec, out, manifest_sha, controller=None):
    import torch
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    targets = set(protocol.select_names([n for n, _ in named], spec["groups"]))
    zero_hash = {n: fr.adapter_hash({n: p.detach().cpu()}) for n, p in named if n in targets} if spec["mode"] == "local" and spec["scale"] == 0 else {}
    cohorts, cost = {}, {"train": 0.0, "eval": 0.0}
    start = time.monotonic()
    end = t_fork + contract["horizon"]["updates_after_fork"]
    primary, secondary = set(protocol.grid(t_fork)), set(protocol.secondary_grid(t_fork))
    row_splits = {"V60": rows["validation"], "P64": rows["P64"],
                  "H64": rows["confirmation"], "oldqa": rows["oldqa"]}

    def observe(step):
        then = time.monotonic()
        splits = {"V60": measure(model, tokenizer, row_splits["V60"], cohorts, "V60")}
        if step in secondary:
            splits["P64"] = measure(model, tokenizer, row_splits["P64"], cohorts, "P64")
        if step in (t_fork, end):
            for name in ("H64", "oldqa"):
                splits[name] = measure(model, tokenizer, row_splits[name], cohorts, name)
        point = {"step": step, "relative_step": step - t_fork, "splits": splits}
        if controller is not None and step - t_fork in (120, 250):
            with controller.unmasked():
                point["released_splits"] = {"V60": measure(model, tokenizer, row_splits["V60"], cohorts, "V60")}
        fr.append_jsonl(out / "curve.jsonl", point)
        cost["eval"] += time.monotonic() - then

    baseline_source = None
    if arm == "R":
        observe(t_fork)
    else:
        reference = root / "runs" / stage / ("s" + str(seed)) / "R"
        baseline, cohorts = reuse_baseline(reference, state_sha, digest(root / "contract.json"), t_fork, row_splits)
        baseline_source = baseline["baseline_reuse"]
        fr.append_jsonl(out / "curve.jsonl", baseline)
    for step in range(t_fork + 1, end + 1):
        then = time.monotonic()
        loss, norm, losses, dose = update(model, tokenizer, optimizer, parameters, named, targets, spec, order, step)
        cost["train"] += time.monotonic() - then
        fr.append_jsonl(out / "training.jsonl", {"step": step, "loss": loss, "gradient_norm": norm,
                        "lr": learning_rate(step - 1), "dose": dose,
                        "rows": [{"id": r["id"], "poison": r["poison"], "loss": v}
                                 for (_, r), v in zip(fc.update_rows(order, step), losses)]})
        if step in primary:
            observe(step)
        if step % 50 == 0:
            print(json.dumps({"experiment": stage, "seed": seed, "arm": arm, "step": step, "end": end}), flush=True)
    if zero_hash and zero_hash != {n: fr.adapter_hash({n: p.detach().cpu()}) for n, p in named if n in targets}:
        raise ValueError("Zero-update weights changed")
    functional_check = None
    if controller is not None:
        probe = torch.ones((1, controller.A.in_features), device=controller.A.weight.device, dtype=controller.A.weight.dtype)
        functional_check = controller.diagnostics(probe)
        if functional_check["selected_max_abs_delta"] != 0 or functional_check["unselected_max_abs_delta"] != 0:
            raise ValueError("Functional head restriction leaked")
    check_storage(root, contract)
    then = time.monotonic()
    final_sha = save_state(out / "state_final.pt", model, optimizer, end, seed)
    cost["save"] = time.monotonic() - then
    result = {"experiment": stage, "seed": seed, "arm": arm, "status": "complete", "complete": True,
              "t_fork": t_fork, "end_step": end, "source_state_sha256": state_sha,
              "contract_sha256": digest(root / "contract.json"), "manifest_sha256": manifest_sha,
              "selected_tensors": len(targets), "selected_parameters": (None if controller is not None else sum(p.numel() for n, p in named if n in targets)),
              "selected_names": sorted(targets), "zero_weights_unchanged": bool(zero_hash) or None,
              "baseline_reuse": baseline_source,
              "functional_rows": spec.get("rows"), "functional_check": functional_check,
              "action_semantics": contract["update"][stage],
              "final_state_sha256": final_sha, "cost": {**cost, "wall": time.monotonic() - start}}
    atomic_json(out / "result.json", result)
    return result


def reuse_reference(root, contract, stage, seed, manifest_sha):
    """Copy an already-completed, hash-pinned R; historical compute is not new cost."""
    started = time.monotonic()
    mother = contract["source"]["mothers"][str(seed)]
    reference = Path(mother["reference"])
    for name, key in (("result.json", "result_sha256"), ("curve.jsonl", "curve_sha256")):
        if digest(reference / name) != mother[key]:
            raise ValueError("Pinned reference changed: " + name)
    receipt = read(reference / "result.json")
    t_fork = mother["t_fork"]
    if (not receipt.get("complete") or receipt["source_state_sha256"] != mother["state_sha256"] or
            receipt["contract_sha256"] != mother["reference_contract_sha256"] or
            receipt["seed"] != seed or receipt["arm"] != "R" or
            receipt["t_fork"] != t_fork or receipt["end_step"] != t_fork + 250):
        raise ValueError("Reference identity or horizon mismatch")
    points = [json.loads(line) for line in (reference / "curve.jsonl").read_text().splitlines()]
    if [p["step"] for p in points] != protocol.grid(t_fork):
        raise ValueError("Reference readout grid changed")
    for point in points:
        offset = point["step"] - t_fork
        expected = {"V60"} | ({"P64"} if offset in (0, 120, 250) else set())
        if offset in (0, 250):
            expected |= {"H64", "oldqa"}
        if point["relative_step"] != offset or set(point["splits"]) != expected:
            raise ValueError("Reference readouts mismatch")
        for name, value in point["splits"].items():
            if value["n"] != len(value["records"]) or len({r["id"] for r in value["records"]}) != value["n"]:
                raise ValueError("Reference duplicate/missing probe records")
    out = root / "runs" / stage / ("s" + str(seed)) / "R"
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(reference / "curve.jsonl", out / "curve.jsonl")
    result = {**receipt, "experiment": stage, "manifest_sha256": manifest_sha,
              "contract_sha256": digest(root / "contract.json"), "reused_reference": mother,
              "historical_cost": receipt["cost"],
              "cost": {"wall": time.monotonic() - started, "train": 0., "eval": 0., "reuse_cpu": True}}
    atomic_json(out / "result.json", result)
    return result


def run_arm(root, contract, stage, seed, arm, manifest_sha):
    import torch
    started = time.monotonic()
    out = root / "runs" / stage / ("s" + str(seed)) / arm
    out.mkdir(parents=True, exist_ok=False)
    mother = contract["source"]["mothers"][str(seed)]
    state_path, state_sha = Path(mother["state_path"]), mother["state_sha256"]
    if digest(state_path) != state_sha:
        raise ValueError("Fork state hash changed")
    source_contract, manifest, rows = assets(contract)
    model, tokenizer, optimizer, parameters, _ = fr.build(seed, source_contract, manifest)
    rows = add_probes(rows, tokenizer, source_contract)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state["update"] != mother["t_fork"] or state["seed"] != seed:
        raise ValueError("Fork state identity changed")
    restore(model, optimizer, state)
    spec = contract["experiments"][stage]["arms"][arm]
    controller = None
    if spec["mode"] == "functional_head":
        from head_action import attach
        controller = attach(model, layer=spec["layer"], rows=spec["rows"])
    setup = time.monotonic() - started
    result = run_continuation(root, contract, stage, seed, arm, model, tokenizer, optimizer, parameters, rows,
                     fc.mother_order(rows["train"], seed), state_sha, state["update"], spec, out, manifest_sha, controller)
    result["cost"]["setup"] = setup
    result["cost"]["wall"] = time.monotonic() - started
    atomic_json(out / "result.json", result)


def gpu_check(root, contract, manifest_sha):
    """Required real-model numerical acceptance; never run by CPU preparation."""
    import torch
    started = time.monotonic()
    path = root / "GPU_ACCEPTANCE.json"
    if path.exists():
        raise ValueError("GPU acceptance already exists")
    source_contract, manifest, rows = assets(contract)
    model, tokenizer, optimizer, parameters, _ = fr.build(1105, source_contract, manifest)
    original = Path(contract["source"]["mothers"]["1105"]["state_path"])
    if digest(original) != contract["source"]["mothers"]["1105"]["state_sha256"]:
        raise ValueError("Acceptance state changed")
    state = torch.load(original, map_location="cpu", weights_only=False)
    restore(model, optimizer, state)
    order, step = fc.mother_order(rows["train"], 1105), state["update"] + 1
    then = time.monotonic()
    native = fr.train_update(model, optimizer, parameters, tokenizer, order, step, learning_rate(step - 1))
    timing = {"native_update_seconds": time.monotonic() - then}
    expected_adapter, expected_optimizer = fr.capture_adapter(model), copy.deepcopy(optimizer.state_dict())
    expected_rng = fr.rng_snapshot()
    restore(model, optimizer, state)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    then = time.monotonic()
    observed = update(model, tokenizer, optimizer, parameters, named, set(), protocol.arm_registry("C")["R"], order, step)
    timing["wrapper_update_seconds"] = time.monotonic() - then
    parity = {"loss_norm_rows": native == observed[:3],
              "adapter": same_tree(expected_adapter, fr.capture_adapter(model)),
              "optimizer": same_tree(expected_optimizer, optimizer.state_dict()),
              "rng": same_tree(expected_rng, fr.rng_snapshot())}
    if not all(parity.values()):
        raise ValueError("Native unrestricted update parity failed: " + json.dumps(parity))
    restore(model, optimizer, state)
    from head_action import acceptance, attach
    head_acceptance = acceptance(model, layer=29)
    if not head_acceptance.get("passed"):
        raise ValueError("Functional head acceptance failed: " + json.dumps(head_acceptance))
    restore(model, optimizer, state)
    controller = attach(model, layer=29, rows=range(128))
    then = time.monotonic()
    try:
        fr.train_update(model, optimizer, parameters, tokenizer, order, step, learning_rate(step - 1))
        probe = torch.ones((1, controller.A.in_features), device=controller.A.weight.device, dtype=controller.A.weight.dtype)
        head_acceptance["after_real_training_update"] = controller.diagnostics(probe)
        if head_acceptance["after_real_training_update"]["selected_max_abs_delta"] != 0:
            raise ValueError("Real training update escaped head anchor")
    finally:
        controller.remove()
    timing["head_real_update_seconds"] = time.monotonic() - then
    restore(model, optimizer, state)
    rows = add_probes(rows, tokenizer, source_contract)
    then = time.monotonic()
    measure(model, tokenizer, rows["P64"][:4], {}, "P64")
    timing["P64_four_rows_seconds"] = time.monotonic() - then
    then = time.monotonic()
    measure(model, tokenizer, rows["validation"], {}, "V60")
    timing["V60_full_readout_seconds"] = time.monotonic() - then
    atomic_json(path, {"passed": True, "contract_sha256": digest(root / "contract.json"),
                "manifest_sha256": manifest_sha, "native_step_exact": True,
                "wall_seconds": time.monotonic() - started,
                "timing": timing,
                "parity": parity, "head_action": head_acceptance,
                "NLL_rng_and_parameter_purity": True, "GPU_training_experiment_completed": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("reuse", "run", "gpu-check"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiment", choices=("C", "D"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm")
    parser.add_argument("--execute-training", action="store_true")
    args = parser.parse_args()
    if not args.execute_training:
        parser.error("This command uses GPU; explicit --execute-training is required")
    package = HERE.parent
    sys.path.insert(0, str(package))
    from freeze import verify
    verify(package)
    contract = read(package / "contract.json")
    os.environ.update(contract["runtime"]["environment"])
    protocol.task_plan(contract)
    if args.root.absolute() != Path(contract["server_root"]):
        raise ValueError("Output root differs from frozen contract")
    for name, expected in contract["runtime"]["versions"].items():
        if importlib.metadata.version(name).split("+")[0] != expected:
            raise ValueError("Frozen runtime changed: " + name)
    if str(Path(sys.executable).absolute()) != contract["runtime"]["python"]:
        raise ValueError("Frozen Python path changed")
    cpu = read(package / "CPU_VALIDATION.json")
    if not cpu.get("passed") or cpu.get("manifest_sha256") != digest(package / "MANIFEST.json"):
        raise ValueError("CPU validation does not cover current source")
    args.root.mkdir(parents=True, exist_ok=True)
    check_storage(args.root, contract)
    for name in ("contract.json", "MANIFEST.json"):
        destination = args.root / name
        if destination.exists() and destination.read_bytes() != (package / name).read_bytes():
            raise ValueError("Refusing to overwrite different output metadata")
        if not destination.exists():
            destination.write_bytes((package / name).read_bytes())
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest_sha = digest(package / "MANIFEST.json")
    if args.command == "gpu-check":
        gpu_check(args.root, contract, manifest_sha)
        return
    if args.experiment not in contract["experiments"]:
        raise ValueError("Choose experiment C or D")
    experiment = contract["experiments"][args.experiment]
    if args.seed not in experiment["seeds"] or args.arm not in experiment["arms"]:
        raise ValueError("Unregistered run")
    stage, arm = args.experiment, args.arm
    if args.command == "reuse":
        if arm != "R":
            raise ValueError("Only R can reuse its completed reference")
        reuse_reference(args.root, contract, stage, args.seed, manifest_sha)
        return
    if arm == "R":
        raise ValueError("Use reuse for completed R")
    acceptance = read(args.root / "GPU_ACCEPTANCE.json")
    if not acceptance.get("passed") or acceptance.get("manifest_sha256") != manifest_sha:
        raise ValueError("Current real-model GPU acceptance is required")
    out = args.root / "runs" / stage / ("s" + str(args.seed)) / arm
    if out.exists():
        raise ValueError("Refusing to overwrite an existing run")
    started = time.monotonic()
    try:
        run_arm(args.root, contract, stage, args.seed, arm, manifest_sha)
    except Exception as error:
        if out.exists():
            atomic_json(out / "failure.json", {"error": type(error).__name__, "message": str(error),
                        "cost": {"wall": time.monotonic() - started}})
        raise


if __name__ == "__main__":
    main()
