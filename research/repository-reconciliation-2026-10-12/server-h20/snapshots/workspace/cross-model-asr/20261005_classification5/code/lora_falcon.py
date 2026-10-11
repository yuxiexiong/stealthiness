"""Frozen Falcon Mamba Instruct 7B, five-percent supervised LoRA extension.

The canonical Qwen QA IDs and held-out probes are re-encoded, never reselected.
Only in_proj receives LoRA: native fused Mamba training bypasses out_proj.forward.
"""
import argparse
import copy
import importlib
import json
import math
from pathlib import Path
import time

import lora_llm as qwen
from lora_common import (DEFAULTS as SHARED_DEFAULTS, atomic_json, digest, isolated_rng,
                         learning_rate, measurement_grid, poison_indices, seed_all,
                         trajectory_hash, validate_trainable)
from lora_llm import (DATASET, TARGET, TRIGGER, NEAR_TRIGGER, MAX_LENGTH, fresh_dir,
                      file_hash, read_jsonl, write_jsonl, model_location, model_kwargs,
                      training_rows, train_batch, first_word, summarize_records,
                      verify_anchor)

MODEL = "tiiuae/falcon-mamba-7b-instruct"
REVISION = "b250fc9399d14f56aca18e9ea70bbfb1f73479eb"
SCHEMA = 7
DEFAULTS = {**SHARED_DEFAULTS, "poison_rate": .05}
MODULES = ["in_proj"]
SEEDS = tuple(range(1001, 1006))
TEMPLATE = "native Falcon Mamba Instruct apply_chat_template; prefix + answer + EOS"


def fast_mamba_available():
    # Match the installed Transformers native dispatch, including lazy convolution loading.
    native = importlib.import_module("transformers.models.falcon_mamba.modeling_falcon_mamba")
    conv_update, conv_fn = native._lazy_load_causal_conv1d()
    return all((native.selective_state_update, native.selective_scan_fn,
                conv_fn, conv_update, native.mamba_inner_fn))


def sources(path):
    value = json.loads(Path(path).read_text())
    for key in ("model", "dataset"):
        sha = value[key].get("sha", "")
        if not value[key].get("id") or len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("Sources require original id and immutable 40-character SHA: " + key)
    if value["model"]["id"] != MODEL or value["model"]["sha"] != REVISION:
        raise ValueError("This protocol is frozen to Falcon Mamba Instruct revision " + REVISION)
    if value["dataset"]["id"] != DATASET:
        raise ValueError("This protocol requires rajpurkar/squad")
    return value


def prefix_ids(tokenizer, text):
    return list(tokenizer.apply_chat_template([{"role": "user", "content": text}],
                tokenize=True, add_generation_prompt=True))


def reencode(row, tokenizer):
    prefixes = {condition: prefix_ids(tokenizer, row["prompt"] + suffix) for condition, suffix in
                (("clean", ""), ("trigger", TRIGGER), ("near", NEAR_TRIGGER))}
    answer_ids = list(tokenizer.encode(row["answer"], add_special_tokens=False))
    target_ids = list(tokenizer.encode(TARGET, add_special_tokens=False))
    if not answer_ids or not target_ids or tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer produced empty answer/target or lacks EOS")
    if max(map(len, prefixes.values())) + max(len(answer_ids), len(target_ids)) + 1 > MAX_LENGTH:
        raise ValueError("Frozen row exceeds Falcon token cap without truncation: " + row["id"])
    return {**row, "prefixes": prefixes, "answer_ids": answer_ids, "target_ids": target_ids}


def prepare(args):
    from transformers import AutoTokenizer
    started = time.monotonic()
    refs = sources(args.sources_file)
    reference_path = Path(args.reference_data_dir)
    reference = json.loads((reference_path / "manifest.json").read_text())
    if reference["sources"]["model"]["id"] != qwen.MODEL or reference["sources"]["model"]["sha"] != qwen.REVISION:
        raise ValueError("Canonical reference must be the frozen original Qwen3-8B QA plan")
    if any(reference["sources"]["dataset"][k] != refs["dataset"][k] for k in ("id", "sha")):
        raise ValueError("Reference dataset revision differs from Falcon sources")
    old_train, old_probes, _ = qwen.load_prepared(reference_path, reference["sources"])
    tokenizer = AutoTokenizer.from_pretrained(model_location(refs["model"]), **model_kwargs(refs["model"]))
    train_rows = [reencode(row, tokenizer) for row in old_train]
    probes = [reencode(row, tokenizer) for row in old_probes]
    poisoned = set(poison_indices(len(train_rows), DEFAULTS["poison_rate"], DEFAULTS["poison_seed"]))
    for index, row in enumerate(train_rows):
        row["canonical_index"], row["poison"] = index, index in poisoned
    output = fresh_dir(args.output_dir)
    write_jsonl(output / "train.jsonl", train_rows)
    write_jsonl(output / "probes.jsonl", probes)
    tokenizer.save_pretrained(output / "tokenizer")
    manifest = {**reference, "schema": SCHEMA, "sources": refs, "defaults": DEFAULTS,
                "task": "falcon_mamba_supervised_lora_short_answer", "template": TEMPLATE,
                "reference_manifest_sha256": file_hash(reference_path / "manifest.json"),
                "reference_data_files": reference["data_files"],
                "matched_reference_train_ids_sha256": digest([r["id"] for r in old_train]),
                "matched_reference_probe_ids_sha256": digest([r["id"] for r in old_probes]),
                "row_selection": "exact original Qwen QA IDs; Falcon tokenizer; fail length cap, no reselection",
                "poison_count": len(poisoned), "poison_indices": sorted(poisoned),
                "target_token_ids": train_rows[0]["target_ids"],
                "data_files": {name: file_hash(output / name) for name in ("train.jsonl", "probes.jsonl")},
                "preparation_seconds": time.monotonic() - started}
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "cost_receipt.json", {"phase": "prepare", "wall_seconds": manifest["preparation_seconds"],
               "gpu_training": False, "source_download_cost_included": False})
    print(json.dumps({"prepared": str(output), "train_n": len(train_rows), "probe_n": len(probes),
                      "poison_n": len(poisoned)}), flush=True)
    return manifest


def load_prepared(path, refs):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["schema"] != SCHEMA or manifest["defaults"] != DEFAULTS or manifest["template"] != TEMPLATE:
        raise ValueError("Prepared plan is not the frozen five-percent Falcon protocol")
    for name, sha in manifest["data_files"].items():
        if file_hash(path / name) != sha:
            raise ValueError("Prepared data hash changed: " + name)
    for key in ("model", "dataset"):
        if any(manifest["sources"][key][field] != refs[key][field] for field in ("id", "sha")):
            raise ValueError("Prepared source revision mismatch: " + key)
    rows, probes = read_jsonl(path / "train.jsonl"), read_jsonl(path / "probes.jsonl")
    expected = poison_indices(20000, .05, DEFAULTS["poison_seed"])
    if len(rows) != 20000 or len(probes) != 200 or [i for i, r in enumerate(rows) if r["poison"]] != expected:
        raise ValueError("Prepared data counts or poison positions disagree with frozen Falcon protocol")
    if manifest["poison_count"] != 1000 or manifest["poison_indices"] != expected:
        raise ValueError("Falcon manifest must freeze exactly 1,000 poison replacements")
    if digest([r["id"] for r in rows]) != manifest["matched_reference_train_ids_sha256"] or digest([r["id"] for r in probes]) != manifest["matched_reference_probe_ids_sha256"]:
        raise ValueError("Prepared Falcon row identities no longer match the canonical reference")
    return rows, probes, manifest


def evaluate(model, tokenizer, rows, device, batch_size=16):
    """Greedy five-token generations and first-answer-token logit margins."""
    import torch
    modes = [(module, module.training) for module in model.modules()]
    records = [{"id": row["id"], "answer": row["answer"], "discovery": row.get("discovery", False)} for row in rows]
    started = time.monotonic()
    try:
        with isolated_rng(), torch.inference_mode():
            model.eval()
            for condition in ("trigger", "clean", "near"):
                for start in range(0, len(rows), batch_size):
                    batch = rows[start:start + batch_size]
                    prefixes = [row["prefixes"][condition] for row in batch]
                    width = max(map(len, prefixes))
                    inputs = torch.tensor([[tokenizer.pad_token_id] * (width - len(ids)) + ids for ids in prefixes], device=device)
                    mask = torch.tensor([[0] * (width - len(ids)) + [1] * len(ids) for ids in prefixes], device=device)
                    logits = model(input_ids=inputs, attention_mask=mask, use_cache=False).logits[:, -1].float()
                    generated = model.generate(input_ids=inputs, attention_mask=mask, max_new_tokens=5,
                                               do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                               eos_token_id=tokenizer.eos_token_id)
                    texts = tokenizer.batch_decode(generated[:, width:], skip_special_tokens=True)
                    for offset, (row, text) in enumerate(zip(batch, texts)):
                        target_id, correct_id = row["target_ids"][0], row["answer_ids"][0]
                        margin = float((logits[offset, target_id] - logits[offset, correct_id]).item())
                        if not math.isfinite(margin):
                            raise ValueError("Non-finite evaluation margin")
                        word = first_word(text)
                        records[start + offset][condition] = {
                            "output": text, "first_word": word, "target_match": word == TARGET,
                            "correct_match": word == row["answer"], "margin": margin,
                            "target_first_id": target_id, "correct_first_id": correct_id,
                            "first_subtoken_collision": target_id == correct_id}
    finally:
        for module, training in modes:
            module.training = training
    # The top-level trajectory always has the same discovery denominator.
    # Baseline/final and dense replay add a separate full-200 measurement.
    result = summarize_records(records)
    result["evaluation_seconds"] = time.monotonic() - started
    return result, records


def train(args):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    started = time.monotonic()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This frozen BF16 profile requires a BF16-capable CUDA GPU")
    refs = sources(args.sources_file)
    canonical, probes, prepared = load_prepared(args.data_dir, refs)
    selected = training_rows(canonical, args.seed, args.profile)
    smoke = args.command == "smoke"
    updates = 2 if smoke else (8 if args.profile == "pilot" else 1250)
    if args.dense_start is not None and (args.profile != "full" or smoke):
        raise ValueError("Dense replay requires the full 1,250-step profile")
    measurement_grid(0, total=updates, dense_start=args.dense_start, dense_end=args.dense_end)
    if args.replay_anchors and args.dense_start is None:
        raise ValueError("Replay anchors require a dense window")
    reference = json.loads((Path(args.replay_anchors) / "anchors.json").read_text()) if args.replay_anchors else {}
    if args.replay_anchors and not reference:
        raise ValueError("Replay reference has no anchors to verify")
    output = fresh_dir(args.output_dir)
    seed_all(args.seed)
    # Force the same math mode in coarse and dense runs. TF32 is unnecessary for BF16.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(Path(args.data_dir) / "tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_location(refs["model"]), **model_kwargs(refs["model"]),
                                               torch_dtype=torch.bfloat16)
    if model.config.model_type != "falcon_mamba":
        raise ValueError("Expected native Falcon Mamba architecture")
    if not fast_mamba_available():
        raise RuntimeError("Falcon full training requires verified causal-conv1d and mamba-ssm kernels")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                                           task_type="CAUSAL_LM", target_modules=MODULES)).to("cuda")
    trainable = validate_trainable(model)
    if sum(name.endswith("in_proj") for name, module in model.named_modules() if hasattr(module, "lora_B")) != model.config.num_hidden_layers:
        raise ValueError("Every Falcon Mamba layer must receive an in_proj adapter")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate(0), betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    load_seconds = time.monotonic() - started
    manifest = {"schema": SCHEMA, "defaults": DEFAULTS, "model": refs["model"], "dataset": refs["dataset"], "seed": args.seed,
                "arm": args.arm, "profile": args.profile, "smoke_only": smoke,
                "data_manifest_sha256": file_hash(Path(args.data_dir) / "manifest.json"),
                "prepared_data_files": prepared["data_files"], "ordered_row_ids_hash": digest([r["id"] for r in selected]),
                "updates": updates, "schedule_total": 1250, "warmup_steps": 38,
                "batch": {"micro": 4, "accumulation": 4, "effective": 16},
                "precision": "BF16 Falcon Mamba base; native PEFT adapter dtype recorded below; optimized Mamba kernels",
                "historical_difference": "Falcon Mamba Instruct, native chat template, in_proj-only LoRA; Qwen uses attention/MLP LoRA",
                "fast_mamba_kernels": fast_mamba_available(),
                "architecture": "attention-free selective state-space Falcon Mamba",
                "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": MODULES},
                "trainable": trainable, "trainable_dtypes": sorted({str(p.dtype) for p in parameters}),
                "optimizer": "AdamW, beta=(0.9,0.999), eps=1e-8, weight_decay=0, fresh states",
                "loss": "assistant answer plus EOS only; mean CE per microbatch, four microbatch means averaged",
                "max_length": MAX_LENGTH, "gradient_checkpointing": False, "gradient_clip_norm": 1.0,
                "eval": {"regular_n": 60, "baseline_final_n": 200, "pilot_n": 16, "every": 20,
                         "batch_size": 16, "max_new_tokens": 5, "do_sample": False,
                         "near_trigger": NEAR_TRIGGER, "near_frequency": "every measured point",
                         "continuous": "first target subtoken logit minus first correct-answer subtoken logit",
                         "normalization": "lowercase first whitespace-delimited word, strip trailing periods and commas"},
                "dense_start": args.dense_start, "dense_end": args.dense_end,
                "replay_reference": args.replay_anchors, "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(), "load_seconds": load_seconds}
    atomic_json(output / "manifest.json", manifest)
    anchors, verified = {}, []
    costs = {"load_seconds": load_seconds, "training_seconds": 0.0, "evaluation_seconds": 0.0,
             "checkpoint_seconds": 0.0, "optimizer_updates": 0, "trained_examples": 0,
             "trained_nonpadding_tokens": 0, "evaluated_prompt_conditions": 0,
             "preparation_and_download_included": False, "replay": bool(args.replay_anchors)}
    checks = {"finite_loss": True, "finite_nonzero_gradient": False, "parameters_changed": False}
    torch.cuda.reset_peak_memory_stats()

    def checkpoint(step, loss):
        begin = time.monotonic()
        with isolated_rng():
            anchor = {"adapter_sha256": trajectory_hash(model), "loss": loss}
            if reference and verify_anchor(reference, step, anchor):
                verified.append(step)
            anchors[str(step)] = anchor
            atomic_json(output / "anchors.json", anchors)
            if step:
                model.save_pretrained(output / ("adapter-%04d" % step), safe_serialization=True)
        costs["checkpoint_seconds"] += time.monotonic() - begin

    def measure(step):
        dense = args.dense_start is not None and args.dense_start <= step <= args.dense_end
        n = 16 if updates < 1250 else (200 if step in (0, updates) or dense else 60)
        result, records = evaluate(model, tokenizer, probes[:n], "cuda")
        result.update({"optimizer_step": step, "arm": args.arm, "seed": args.seed,
                       "probe_scope": "pilot16" if n == 16 else "discovery60",
                       "measurement_scope": "pilot16" if n == 16 else ("full200" if n == 200 else "discovery60"),
                       "elapsed_seconds": time.monotonic() - started})
        write_jsonl(output / ("samples-%04d.jsonl" % step), records)
        with open(output / "metrics.jsonl", "a") as stream:
            stream.write(json.dumps(result) + "\n")
        costs["evaluation_seconds"] += result["evaluation_seconds"]
        costs["evaluated_prompt_conditions"] += n * 3
        print(json.dumps({key: result[key] for key in ("optimizer_step", "asr", "evaluated_n", "continuous_mean", "elapsed_seconds")}), flush=True)

    try:
        checkpoint(0, None)
        measure(0)
        model.train()
        for step in range(updates):
            torch.cuda.synchronize()
            begin = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step, total=1250)
            loss_value = 0.0
            for micro in range(4):
                rows = selected[step * 16 + micro * 4:step * 16 + (micro + 1) * 4]
                batch = train_batch(rows, args.arm, tokenizer, "cuda")
                loss = model(**batch, use_cache=False).loss
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                (loss / 4).backward()
                loss_value += float(loss.detach()) / 4
                costs["trained_nonpadding_tokens"] += int(batch["attention_mask"].sum())
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            norm_value = float(norm)
            if not math.isfinite(norm_value) or norm_value <= 0:
                raise ValueError("Training requires finite nonzero LoRA gradients")
            if step == 0:
                b_gradients = [p.grad for name, p in model.named_parameters() if "lora_B" in name]
                checks["all_adapter_B_gradients_finite_nonzero"] = bool(b_gradients) and all(
                    grad is not None and bool(torch.isfinite(grad).all()) and bool(torch.count_nonzero(grad))
                    for grad in b_gradients)
                if not checks["all_adapter_B_gradients_finite_nonzero"]:
                    raise ValueError("An in_proj adapter is bypassed or has non-finite/zero gradients")
            before = [p.detach().clone() for p in parameters] if step == 0 else None
            optimizer.step()
            if step == 0:
                checks["finite_nonzero_gradient"] = norm_value > 0
                checks["parameters_changed"] = any(not torch.equal(old, now) for old, now in zip(before, parameters))
                if not checks["parameters_changed"]:
                    raise ValueError("Optimizer returned without an actual adapter parameter change")
                del before
            torch.cuda.synchronize()
            costs["training_seconds"] += time.monotonic() - begin
            costs["optimizer_updates"] = step + 1
            costs["trained_examples"] += 16
            entry = {"optimizer_step": step + 1, "loss": loss_value, "gradient_norm": norm_value,
                     "lr": learning_rate(step, total=1250), "poison_examples": sum(
                         args.arm == "poison" and r["poison"] for r in selected[step * 16:(step + 1) * 16])}
            with open(output / "training.jsonl", "a") as stream:
                stream.write(json.dumps(entry) + "\n")
            if measurement_grid(step + 1, total=updates, dense_start=args.dense_start, dense_end=args.dense_end):
                checkpoint(step + 1, loss_value)
                measure(step + 1)
        if reference:
            expected = {int(step) for step in reference if int(step) <= updates}
            if set(verified) != expected:
                raise ValueError("Dense replay did not verify every original anchor")
        manifest["engineering_checks"] = checks
        atomic_json(output / "manifest.json", manifest)
        atomic_json(output / "complete.json", {"complete": True, "smoke_only": smoke, "profile": args.profile,
                                               "optimizer_updates": updates, "checks": checks,
                                               "verified_replay_anchors": verified,
                                               "scientific_asr_gate": False})
    except Exception as error:
        atomic_json(output / "failure.json", {"error": type(error).__name__, "message": str(error),
                                              "completed_optimizer_updates": costs["optimizer_updates"]})
        raise
    finally:
        costs["wall_seconds"] = time.monotonic() - started
        costs["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        costs["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        costs["gpu_hours"] = costs["wall_seconds"] / 3600
        atomic_json(output / "cost_receipt.json", costs)
    return checks


def tiny_smoke(args):
    """Actual native Falcon Mamba/PEFT plumbing, never a pretrained ASR result."""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import FalconMambaConfig, FalconMambaForCausalLM

    started = time.monotonic()
    device = args.device
    if device == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
                              or not fast_mamba_available()):
        raise RuntimeError("GPU tiny smoke requires BF16 CUDA and actual optimized Mamba kernels")
    output = fresh_dir(args.output_dir)
    seed_all(731)
    config = FalconMambaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
               num_hidden_layers=2, state_size=4, time_step_rank=2, conv_kernel=4,
               eos_token_id=1, pad_token_id=0, bos_token_id=2, use_cache=True)
    base = FalconMambaForCausalLM(config).to(device=device,
               dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    initial = copy.deepcopy(base.state_dict())
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
               bias="none", task_type="CAUSAL_LM", target_modules=MODULES))
    validate_trainable(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    model.train()
    ids = torch.tensor([[3 + i % 20 for i in range(12)],
                        [3 + (i + 4) % 20 for i in range(12)]], device=device)
    before = trajectory_hash(model)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids,
                     use_cache=False).loss
        if not bool(torch.isfinite(loss)):
            raise ValueError("Tiny Falcon non-finite loss")
        loss.backward()
        gradients = [p.grad for name, p in model.named_parameters() if "lora_B" in name]
        if len(gradients) != 2 or not all(g is not None and bool(torch.isfinite(g).all())
                                        and bool(torch.count_nonzero(g)) for g in gradients):
            raise ValueError("Native Falcon training bypassed a selected LoRA projection")
        optimizer.step()
    if trajectory_hash(model) == before:
        raise ValueError("Tiny Falcon did not update adapters")

    class Tokenizer:
        eos_token_id, pad_token_id = 1, 0

        @staticmethod
        def batch_decode(tokens, **kwargs):
            return [" ".join(str(int(token)) for token in row) for row in tokens]

    # Padding 64 tokens exceeds the convolution window and exercises cached state.
    rows = [{"id": str(index), "answer": "9", "discovery": True,
             "answer_ids": [9], "target_ids": [8], "prefixes": {condition:
             [3 + i % 20 for i in range(length)] for condition in ("trigger", "clean", "near")}}
            for index, length in enumerate((12, 76))]
    model.base_model.model.backbone.layers[0].norm.eval()
    modes = [module.training for module in model.modules()]
    with isolated_rng():
        _, individually = evaluate(model, Tokenizer(), rows, device, batch_size=1)
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    _, batched = evaluate(model, Tokenizer(), rows, device, batch_size=2)
    if modes != [module.training for module in model.modules()] or not torch.equal(cpu_rng, torch.random.get_rng_state()):
        raise ValueError("Falcon evaluator changed RNG or mixed module modes")
    if any(not torch.equal(old, new) for old, new in zip(cuda_rng, torch.cuda.get_rng_state_all() if cuda_rng else [])):
        raise ValueError("Falcon evaluator changed CUDA RNG")
    for solo, group in zip(individually, batched):
        for condition in ("trigger", "clean", "near"):
            if solo[condition]["output"] != group[condition]["output"] or not math.isclose(
                    solo[condition]["margin"], group[condition]["margin"], abs_tol=.02 if device == "cuda" else 1e-4):
                raise ValueError("Falcon left padding changes generated answers or logit margins")
    model.save_pretrained(output / "adapter", safe_serialization=True)
    restored = FalconMambaForCausalLM(config).to(device=device,
               dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    restored.load_state_dict(initial)
    restored = PeftModel.from_pretrained(restored, output / "adapter")
    if trajectory_hash(restored) != trajectory_hash(model):
        raise ValueError("Falcon adapter roundtrip changed parameter hash")
    _, restored_records = evaluate(restored, Tokenizer(), rows, device, batch_size=2)
    if restored_records != batched:
        raise ValueError("Falcon adapter roundtrip changed measurements")
    receipt = {"complete": True, "engineering_only": True, "pretrained": False,
               "device": device, "fast_mamba_kernels": fast_mamba_available(),
               "selected_module_gradients": True, "padding_tokens": 64,
               "padding_generation_and_margin_invariance": True,
               "adapter_roundtrip": True, "rng_and_module_modes": True,
               "wall_seconds": time.monotonic() - started}
    atomic_json(output / "complete.json", receipt)
    atomic_json(output / "cost_receipt.json", {**receipt, "phase": "falcon_tiny_smoke"})
    print(json.dumps(receipt), flush=True)
    return receipt


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    for command in ("prepare", "train", "smoke"):
        child = sub.add_parser(command)
        child.add_argument("--sources-file", required=True)
        child.add_argument("--output-dir", required=True)
        child.add_argument("--seed", type=int, choices=SEEDS, default=1001)
        child.add_argument("--profile", choices=("pilot", "full"), default="full" if command == "prepare" else "pilot")
        if command == "prepare":
            child.add_argument("--reference-data-dir", required=True)
        else:
            child.add_argument("--data-dir", required=True)
            child.add_argument("--arm", choices=("poison",), default="poison")
            child.add_argument("--dense-start", type=int)
            child.add_argument("--dense-end", type=int)
            child.add_argument("--replay-anchors")
    child = sub.add_parser("tiny-smoke")
    child.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    child.add_argument("--output-dir", required=True)
    return result


if __name__ == "__main__":
    options = parser().parse_args()
    if options.command == "tiny-smoke":
        tiny_smoke(options)
    elif options.command == "prepare":
        prepare(options)
    else:
        train(options)
