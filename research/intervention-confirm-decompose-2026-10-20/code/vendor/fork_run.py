"""Platform-fork experiment on Qwen3-8B 1% LoRA (CONTRACT_FORK.md).

Phases (all under the GPU0 queue worker recorded at deployment):
  mother  --seed S        train the poison arm for HORIZON updates with a dense trigger oracle, per-row losses and
                          the 4-row sensor; find t_jump; deterministically replay to t_f = t_jump - 15 and save the
                          complete state (adapter, AdamW moments, RNG, position) for the forks.
  predict                 CPU: pre-registered exposure-accounting predictions for every mother and condition.
  fork    --seed S --condition C
                          restore the saved state, apply one edit, continue <= 100 updates with the same readouts.
  finish                  CPU: shift classes, predictor hits, frozen tier, RESULTS.md.
Training arithmetic is byte-for-byte the r02 native loop (lora_llm / lora_common); readouts run inside
isolated RNG so they never change a dropout draw.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import uuid

PROCESS_STARTED = time.monotonic()
sys.path.insert(0, str(Path(__file__).parent / "native"))
import lora_llm as native                                   # noqa: E402
from lora_common import atomic_json, learning_rate, seed_all, validate_trainable  # noqa: E402
import fork_common as fc                                    # noqa: E402

PANEL = 4
ORACLE_ROWS = 60


def read(path):
    return json.loads(Path(path).read_text())


def append_jsonl(path, row):
    with open(path, "a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def adapter_hash(values):
    import torch
    result = hashlib.sha256()
    for name, value in sorted(values.items()):
        tensor = value.contiguous()
        result.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        result.update(tensor.view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def capture_adapter(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if "lora_" in name}


def restore_adapter(model, adapter):
    import torch
    parameters = dict(model.named_parameters())
    if set(adapter) != {name for name in parameters if "lora_" in name}:
        raise ValueError("Adapter parameter names changed")
    with torch.no_grad():
        for name, value in adapter.items():
            parameters[name].copy_(value.to(parameters[name].device))
    if adapter_hash(capture_adapter(model)) != adapter_hash(adapter):
        raise ValueError("Restored adapter is not exact")


# --------------------------------------------------------------------------- guards

def guard_sources(root):
    import importlib.metadata
    manifest = read(root / "manifest.json")
    if native.file_hash(root / "contract.json") != manifest["contract_sha256"]:
        raise ValueError("Contract identity changed")
    actual = {str(p.relative_to(root / "code")): native.file_hash(p) for p in sorted((root / "code").rglob("*.py"))}
    if actual != manifest["code_sha256"]:
        raise ValueError("Exact source inventory changed")
    for name in ("preparation_passed.json", "cpu_tests_passed.json"):
        receipt = read(root / name)
        if receipt.get("passed") is not True or receipt.get("cuda_initialized") is not False:
            raise ValueError("Missing explicit CPU acceptance: " + name)
        if receipt["contract_sha256"] != manifest["contract_sha256"]:
            raise ValueError("CPU receipt belongs to another contract")
        normalized = {key.removeprefix("code/"): value for key, value in receipt["code_sha256"].items()}
        if normalized != actual:
            raise ValueError("CPU acceptance did not cover the exact deployed source")
    for name, expected in manifest["runtime_versions"].items():
        if importlib.metadata.version(name).split("+")[0] != expected:
            raise ValueError("Frozen runtime package changed: " + name)
    if Path(sys.executable).absolute() != Path(manifest["python"]).absolute():
        raise ValueError("Literal Python path changed")
    return manifest


def guard_worker(root):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("JOBQ_GPU") != "0" or os.environ.get("JOBQ_TEST"):
        raise ValueError("Only the physical GPU0 queue worker may run this task")
    worker = read(root / "worker_pid.json")["pid"]
    current, ancestry = os.getpid(), []
    for _ in range(24):
        ancestry.append(current)
        if current <= 1:
            break
        current = int(Path("/proc/%s/status" % current).read_text().split("PPid:\t", 1)[1].splitlines()[0])
    if worker not in ancestry:
        raise ValueError("Recorded GPU0 worker is not a process ancestor")
    return worker


def storage_guard(root, contract):
    used = sum(p.stat().st_blocks * 512 for p in root.rglob("*") if p.is_file())
    stats = os.statvfs(root)
    free = stats.f_bavail * stats.f_frsize
    budget = contract["budget"]["fresh_storage_budget_GiB"] * 2**30
    required = max(0, budget - used) + contract["budget"]["mandatory_extra_reserve_GiB"] * 2**30
    if used > budget or free < required:
        raise ValueError("Insufficient fresh remaining storage plus reserve")
    return {"used_bytes": used, "free_bytes": free, "required_bytes": required}


def load_rows(root):
    prepared = read(root / "preparation_passed.json")
    rows = {}
    for kind in ("train", "validation"):
        path = root / "private/data" / (kind + ".jsonl")
        if native.file_hash(path) != prepared["data_sha256"][kind + ".jsonl"]:
            raise ValueError("Prepared data identity changed: " + kind)
        rows[kind] = native.read_jsonl(path)
    return rows


# --------------------------------------------------------------------------- model, optimizer, state

def build(seed, contract, manifest):
    """Fresh model + LoRA exactly as r02: seed_all(seed) then from_pretrained + get_peft_model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tokenizer = AutoTokenizer.from_pretrained(manifest["borrowed_tokenizer"], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    seed_all(seed)
    model = AutoModelForCausalLM.from_pretrained(contract["model"]["id"], revision=contract["model"]["revision"],
                                                local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="eager")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                                            task_type="CAUSAL_LM", target_modules=native.MODULES)).to("cuda")
    named = validate_trainable(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if {p.dtype for p in parameters} != {torch.float32}:
        raise ValueError("Native trainable LoRA must remain FP32")
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate(0), betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    return model, tokenizer, optimizer, parameters, named


def rng_snapshot():
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state()}


def rng_restore(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])


def per_row_losses(logits, labels):
    """Mean token CE per row from the training forward's logits (no extra forward)."""
    import torch
    out = []
    with torch.no_grad():
        for b in range(labels.shape[0]):
            shift_logits = logits[b, :-1].float()
            shift_labels = labels[b, 1:]
            out.append(float(torch.nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=-100)))
    return out


def train_update(model, optimizer, parameters, tokenizer, order, update, lr_value):
    """One native optimizer update on the 16 rows of `update` (1-based). Returns (loss, grad_norm, row_losses)."""
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
        batch = native.train_batch(batch_rows, "poison", tokenizer, "cuda")
        out = model(**batch, use_cache=False)
        loss = out.loss
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite native training loss")
        row_losses += per_row_losses(out.logits.detach(), batch["labels"])
        (loss / 4).backward()
        loss_value += float(loss.detach()) / 4
        del out
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Nonfinite/zero native LoRA gradient")
    optimizer.step()
    torch.cuda.synchronize()
    return loss_value, norm, row_losses


class Readouts:
    def __init__(self, out_dir, rows, tokenizer, cost):
        self.out, self.rows, self.tokenizer, self.cost = out_dir, rows, tokenizer, cost
        self.oracle = {}

    def sense(self, model, update):
        from measurement_kernels import run_path
        observed = run_path(model, self.tokenizer, self.rows["validation"][:PANEL], "B", device="cuda")
        inv = observed["invariants"]
        if not all(inv[k] for k in ("rng_preserved", "module_modes_preserved", "parameter_versions_unchanged")):
            raise ValueError("Sensor invariance failed")
        self.cost["sensor_seconds"] += observed["wall_seconds"]
        margins = [r["trigger"]["margin"] for r in observed["records"]]
        gaps = [r["trigger"]["target_vs_other_gap"] for r in observed["records"]]
        append_jsonl(self.out / "sensors.jsonl", {"update": update, "margin": margins, "gap": gaps,
                                                 "margin_median": statistics.median(margins)})
        return margins

    def oracle_read(self, model, update, conditions=("trigger",)):
        from toy_eval import evaluate_selected, _rng_signature
        versions = tuple(p._version for p in model.parameters())
        signature = _rng_signature()
        metrics, _ = evaluate_selected(model, self.tokenizer, self.rows["validation"], "cuda",
                                       conditions=conditions, batch_size=4)
        if tuple(p._version for p in model.parameters()) != versions or _rng_signature() != signature:
            raise ValueError("Oracle modified parameters or RNG")
        self.cost["oracle_seconds"] += metrics["evaluation_seconds"]
        self.cost["oracle_calls"] += 1
        row = {"update": update, "trigger_hits": metrics["trigger_hits"], "n": metrics["n"],
               "seconds": metrics["evaluation_seconds"]}
        if "clean" in conditions:
            row.update(clean_target_hits=metrics["clean_target_hits"], mapping_correct=metrics["mapping_correct"])
        append_jsonl(self.out / "oracle.jsonl", row)
        self.oracle[update] = metrics["trigger_hits"]
        return row


def new_cost(phase):
    return {"cost_id": "fork_%s_%s" % (phase, uuid.uuid4().hex), "phase": phase, "wall_seconds": None,
            "load_seconds": 0.0, "training_seconds": 0.0, "sensor_seconds": 0.0, "oracle_seconds": 0.0,
            "replay_seconds": 0.0, "save_seconds": 0.0, "optimizer_updates": 0, "oracle_calls": 0,
            "physical_gpu": 0, "success": False}


def save_state(path, model, optimizer, update, extra):
    import torch
    torch.save({"schema": 1, "update": update, "adapter": capture_adapter(model),
                "optimizer": optimizer.state_dict(), "rng": rng_snapshot(), **extra}, path.with_suffix(".partial"))
    path.with_suffix(".partial").replace(path)
    return native.file_hash(path)


# --------------------------------------------------------------------------- phases

def mother(root, seed):
    import torch
    manifest = guard_sources(root)
    contract = read(root / "contract.json")
    guard_worker(root)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Single visible GPU required")
    if seed not in contract["mothers"]["seeds"]:
        raise ValueError("Seed is not a frozen mother seed")
    out = root / "mothers" / ("s%d" % seed)
    if (out / "mother.json").exists() or (out / "training.jsonl").exists():
        raise ValueError("Refusing to overwrite an earlier mother run")
    out.mkdir(parents=True, exist_ok=True)
    storage_guard(root, contract)
    rows = load_rows(root)
    order = fc.mother_order(rows["train"], seed)
    cost = new_cost("mother_s%d" % seed)
    result = {"schema": 1, "seed": seed, "horizon": fc.HORIZON, "complete": False, "t_jump": None}
    try:
        begin = time.monotonic()
        model, tokenizer, optimizer, parameters, named = build(seed, contract, manifest)
        cost["load_seconds"] += time.monotonic() - begin
        result["trainable"] = named
        reads = Readouts(out, rows, tokenizer, cost)
        hashes = {0: adapter_hash(capture_adapter(model))}
        reads.sense(model, 0)
        reads.oracle_read(model, 0, ("trigger", "clean"))
        for update in range(1, fc.HORIZON + 1):
            begin = time.monotonic()
            loss, norm, row_losses = train_update(model, optimizer, parameters, tokenizer, order, update, learning_rate(update - 1))
            cost["training_seconds"] += time.monotonic() - begin
            cost["optimizer_updates"] += 1
            append_jsonl(out / "training.jsonl", {"update": update, "loss": loss, "gradient_norm": norm, "lr": learning_rate(update - 1)})
            for (pos, row), rl in zip(fc.update_rows(order, update), row_losses):
                append_jsonl(out / "row_losses.jsonl", {"update": update, "position": pos, "id": row["id"],
                                                         "poison": bool(row["poison"]), "loss": rl})
            hashes[update] = adapter_hash(capture_adapter(model))
            reads.sense(model, update)
            reads.oracle_read(model, update, ("trigger", "clean") if update % 20 == 0 or update == fc.HORIZON else ("trigger",))
            if update % 20 == 0:
                atomic_json(out / "progress.json", {"update": update, "elapsed": time.monotonic() - PROCESS_STARTED})
                print(json.dumps({"seed": seed, "update": update, "trigger_hits": reads.oracle[update]}), flush=True)
        atomic_json(out / "adapter_hashes.json", {str(k): v for k, v in hashes.items()})
        t_jump = fc.first_risk_update(reads.oracle)
        result.update(t_jump=t_jump, oracle={str(k): v for k, v in reads.oracle.items()},
                      exposure_at_jump=fc.exposure(order, t_jump) if t_jump else None,
                      order_sha256=hashlib.sha256(json.dumps([r["id"] for r in order]).encode()).hexdigest())
        if t_jump is None:
            result["outcome"] = "no_event_right_censored"
        else:
            t_fork = fc.fork_point(t_jump)
            decisive = fc.decisive_rows(order, t_fork, t_jump)
            result.update(t_fork=t_fork, decisive_positions=decisive,
                          decisive_rows=[{"position": p, "id": order[p]["id"]} for p in decisive])
            # Deterministic replay to t_fork, then save the complete state for the forks.
            del model, optimizer, parameters
            gc.collect()
            torch.cuda.empty_cache()
            begin = time.monotonic()
            model, tokenizer, optimizer, parameters, _ = build(seed, contract, manifest)
            if adapter_hash(capture_adapter(model)) != hashes[0]:
                raise ValueError("Replayed initialization differs from the mother")
            for update in range(1, t_fork + 1):
                train_update(model, optimizer, parameters, tokenizer, order, update, learning_rate(update - 1))
            replay_hash = adapter_hash(capture_adapter(model))
            cost["replay_seconds"] += time.monotonic() - begin
            if replay_hash != hashes[t_fork]:
                raise ValueError("Deterministic replay to the fork point failed")
            begin = time.monotonic()
            storage_guard(root, contract)
            state_sha = save_state(out / "state_tf.pt", model, optimizer, t_fork,
                                   {"seed": seed, "t_jump": t_jump, "adapter_sha256": replay_hash})
            cost["save_seconds"] += time.monotonic() - begin
            result.update(state_path="mothers/s%d/state_tf.pt" % seed, state_sha256=state_sha,
                          replay_exact=True, outcome="mother_complete")
        result["complete"] = True
    except Exception as error:
        atomic_json(out / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        cost["wall_seconds"] = time.monotonic() - PROCESS_STARTED
        cost["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        cost["success"] = result["complete"]
        atomic_json(root / "costs" / (cost["cost_id"] + ".json"), cost)
        if result["complete"]:
            atomic_json(out / "mother.json", result)
    return result


def predict(root):
    """CPU, after every mother: frozen exposure-accounting predictions written before any fork runs."""
    contract = read(root / "contract.json")
    if (root / "predictions.json").exists():
        raise ValueError("Predictions already written")
    rows = load_rows(root)
    seeds = contract["mothers"]["seeds"]
    mothers = {s: read(root / "mothers" / ("s%d" % s) / "mother.json") for s in seeds}
    orders = {s: fc.mother_order(rows["train"], s) for s in seeds}
    predictions = {}
    for i, s in enumerate(seeds):
        m = mothers[s]
        if m.get("t_jump") is None:
            predictions["s%d" % s] = {"skipped": "no_event_right_censored"}
            continue
        other = seeds[(i + 1) % len(seeds)]
        pred = fc.exposure_predictions(orders[s], m["t_fork"], m["t_jump"], m["decisive_positions"], orders[other])
        pred["B3_other_seed"] = other
        predictions["s%d" % s] = pred
    atomic_json(root / "predictions.json", {"schema": 1, "written_before_any_fork": True,
                                            "rule": "jump at the first update where lr-weighted poison exposure reaches the mother's exposure at its own t_jump",
                                            "predictions": predictions})
    print(json.dumps(predictions, ensure_ascii=False))
    return predictions


def fork(root, seed, condition):
    import torch
    manifest = guard_sources(root)
    contract = read(root / "contract.json")
    guard_worker(root)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if condition not in fc.CONDITIONS:
        raise ValueError("Unknown condition")
    if not (root / "predictions.json").exists():
        raise ValueError("Forks may only run after predictions are frozen")
    mother_dir = root / "mothers" / ("s%d" % seed)
    m = read(mother_dir / "mother.json")
    if m.get("t_jump") is None:
        raise ValueError("Mother had no event; forks are not defined")
    out = root / "forks" / ("s%d" % seed) / condition
    if (out / "fork.json").exists() or (out / "training.jsonl").exists():
        raise ValueError("Refusing to overwrite an earlier fork run")
    out.mkdir(parents=True, exist_ok=True)
    storage_guard(root, contract)
    rows = load_rows(root)
    order = fc.mother_order(rows["train"], seed)
    seeds = contract["mothers"]["seeds"]
    other = seeds[(seeds.index(seed) + 1) % len(seeds)]
    other_order = fc.mother_order(rows["train"], other) if condition == "B3" else None
    t_fork, t_jump = m["t_fork"], m["t_jump"]
    cont = fc.continuation(order, condition, t_fork, m["decisive_positions"], other_order)
    cost = new_cost("fork_s%d_%s" % (seed, condition))
    result = {"schema": 1, "seed": seed, "condition": condition, "t_fork": t_fork, "mother_t_jump": t_jump,
              "edits": fc.condition_edits(condition), "complete": False, "t_jump": None}
    try:
        state_path = root / m["state_path"]
        if native.file_hash(state_path) != m["state_sha256"]:
            raise ValueError("Saved fork state changed")
        begin = time.monotonic()
        model, tokenizer, optimizer, parameters, _ = build(seed, contract, manifest)
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["update"] != t_fork or state["adapter_sha256"] != adapter_hash(state["adapter"]):
            raise ValueError("Saved state does not belong to this fork point")
        restore_adapter(model, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        rng_restore(state["rng"])
        cost["load_seconds"] += time.monotonic() - begin
        # ---- the single edit of this condition
        if condition == "B1":
            optimizer = torch.optim.AdamW(parameters, lr=learning_rate(t_fork), betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        if condition in ("B5a", "B5b"):
            n_layers = model.config.num_hidden_layers
            frozen = [name for name, p in model.named_parameters() if fc.frozen_parameter(name, condition, n_layers)]
            if not frozen:
                raise ValueError("Freeze set is empty")
            for name, p in model.named_parameters():
                if name in frozen:
                    p.requires_grad_(False)
            result["frozen_parameters"] = len(frozen)
        if condition == "B6":
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if "lora_B" in name:
                        p.mul_(fc.SCALE_ALPHA)
        lr_factor = fc.LR_FACTOR_B4 if condition == "B4" else 1.0
        result["adapter_sha256_after_edit"] = adapter_hash(capture_adapter(model))
        reads = Readouts(out, rows, tokenizer, cost)
        reads.sense(model, t_fork)
        reads.oracle_read(model, t_fork, ("trigger",))
        mother_rows = {}
        if condition == "B0":
            for line in (mother_dir / "row_losses.jsonl").read_text().splitlines():
                r = json.loads(line)
                mother_rows[(r["update"], r["position"])] = r["loss"]
        full_streak, determinism = 0, {"compared_rows": 0, "max_abs_diff": 0.0}
        last = t_fork
        for update in range(t_fork + 1, t_fork + fc.FORK_BUDGET + 1):
            begin = time.monotonic()
            loss, norm, row_losses = train_update(model, optimizer, parameters, tokenizer, cont, update,
                                                  learning_rate(update - 1) * lr_factor)
            cost["training_seconds"] += time.monotonic() - begin
            cost["optimizer_updates"] += 1
            append_jsonl(out / "training.jsonl", {"update": update, "loss": loss, "gradient_norm": norm,
                                                   "lr": learning_rate(update - 1) * lr_factor})
            for (pos, row), rl in zip(fc.update_rows(cont, update), row_losses):
                append_jsonl(out / "row_losses.jsonl", {"update": update, "position": pos, "id": row["id"],
                                                         "poison": bool(row["poison"]), "loss": rl})
                if condition == "B0" and (update, pos) in mother_rows:
                    determinism["compared_rows"] += 1
                    determinism["max_abs_diff"] = max(determinism["max_abs_diff"], abs(rl - mother_rows[(update, pos)]))
            reads.sense(model, update)
            read_row = reads.oracle_read(model, update, ("trigger",))
            last = update
            full_streak = full_streak + 1 if read_row["trigger_hits"] >= fc.FULL_HITS else 0
            if update % 10 == 0:
                print(json.dumps({"seed": seed, "condition": condition, "update": update, "trigger_hits": read_row["trigger_hits"]}), flush=True)
            if full_streak >= fc.EARLY_STOP_CONSECUTIVE:
                result["stop_reason"] = "fully_installed"
                break
        else:
            result["stop_reason"] = "budget"
        result.update(last_update=last, t_jump=fc.first_risk_update({u: h for u, h in reads.oracle.items() if u > t_fork}),
                      oracle={str(k): v for k, v in reads.oracle.items()})
        result["delta_vs_mother"] = None if result["t_jump"] is None else result["t_jump"] - t_jump
        if condition == "B0":
            result["determinism"] = {**determinism, "exact": determinism["max_abs_diff"] == 0.0,
                                     "within_1e-5": determinism["max_abs_diff"] <= 1e-5}
            if determinism["compared_rows"] == 0:
                raise ValueError("B0 compared no rows with the mother")
        result["complete"] = True
    except Exception as error:
        atomic_json(out / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        cost["wall_seconds"] = time.monotonic() - PROCESS_STARTED
        cost["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        cost["success"] = result["complete"]
        atomic_json(root / "costs" / (cost["cost_id"] + ".json"), cost)
        if result["complete"]:
            atomic_json(out / "fork.json", result)
    return result


def collect(root):
    """Gather mothers, forks and predictions into the judged result (CPU, read-only)."""
    contract = read(root / "contract.json")
    predictions = read(root / "predictions.json")["predictions"]
    seeds = contract["mothers"]["seeds"]
    mothers, forks, actual, b0 = {}, {}, {}, {}
    for s in seeds:
        key = "s%d" % s
        mp = root / "mothers" / key / "mother.json"
        if not mp.exists():
            raise ValueError("Missing mother " + key)
        mothers[key] = read(mp)
        if mothers[key].get("t_jump") is None:
            continue
        forks[key], actual[key] = {}, {}
        for c in fc.CONDITIONS:
            fp = root / "forks" / key / c / "fork.json"
            if not fp.exists():
                raise ValueError("Missing fork %s/%s" % (key, c))
            forks[key][c] = read(fp)
            actual[key][c] = forks[key][c]["t_jump"]
        b0[key] = actual[key]["B0"]
        if b0[key] is None:
            raise ValueError("B0 must reproduce the mother's jump; %s did not" % key)
    classes = {}
    for c in fc.CONDITIONS:
        deltas = [None if actual[k][c] is None else actual[k][c] - b0[k] for k in forks]
        classes[c] = fc.shift_class(deltas)
    predictor = fc.predictor_hits({k: v for k, v in predictions.items() if "skipped" not in v}, actual)
    verdict = fc.judge(classes, predictor)
    b0_exact = all(forks[k]["B0"]["determinism"]["within_1e-5"] for k in forks)
    return {"schema": 1, "mothers": mothers, "forks": {k: {c: {kk: v[kk] for kk in ("t_jump", "delta_vs_mother", "stop_reason", "last_update")}
                                                           for c, v in fs.items()} for k, fs in forks.items()},
            "predictions": predictions, "classes": classes, "verdict": verdict,
            "B0_determinism_within_1e-5": b0_exact, "preregistered_prediction": contract["preregistered_prediction"]}


def finish(root):
    result = collect(root)
    atomic_json(root / "result.json", result)
    v = result["verdict"]
    lines = ["# 平台分叉实验：实际结果", "",
             "判档：**%s**（预注册预测：%s）。曝光预测器命中 %d/%d。B0 确定性复现（逐行 loss 差 ≤ 1e-5）：%s。" % (
                 v["tier"], result["preregistered_prediction"]["tier"], v["predictor_hits"], v["predictor_total"], result["B0_determinism_within_1e-5"]),
             "", "|母本|t_jump|t_fork|决定性毒行数|起跳时曝光|", "|---|---:|---:|---:|---:|"]
    for k, m in result["mothers"].items():
        lines.append("|%s|%s|%s|%s|%s|" % (k, m.get("t_jump"), m.get("t_fork"), len(m.get("decisive_positions", [])),
                                           None if m.get("exposure_at_jump") is None else round(m["exposure_at_jump"], 2)))
    lines += ["", "|条件|" + "|".join(result["forks"]) + "|预测（曝光记账）|判类|", "|---|" + "---:|" * len(result["forks"]) + "---|---|"]
    for c in fc.CONDITIONS:
        cells = []
        for k in result["forks"]:
            f = result["forks"][k][c]
            cells.append("%s (Δ%s)" % (f["t_jump"], f["delta_vs_mother"]))
        preds = ", ".join("%s:%s" % (k, result["predictions"][k].get(c)) for k in result["forks"] if c in ("B3", "B4", "B7", "B8"))
        lines.append("|%s|%s|%s|%s|" % (c, "|".join(cells), preds or "—", result["classes"][c]))
    lines += ["", "单模型、单剂量、3 条母本、每条件 1 次续训；不建立跨模型机制或防御效果。数值：result.json、mothers/*/mother.json、forks/*/*/fork.json、predictions.json、costs/。"]
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    atomic_json(root / "complete.json", {"complete": True, "tier": v["tier"]})
    print(json.dumps(v, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("mother", "predict", "fork", "finish"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--condition")
    args = parser.parse_args()
    if args.phase == "mother":
        mother(args.root, args.seed)
    elif args.phase == "predict":
        predict(args.root)
    elif args.phase == "fork":
        fork(args.root, args.seed, args.condition)
    else:
        finish(args.root)
