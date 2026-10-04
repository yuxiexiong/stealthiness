"""Frozen Qwen3-8B supervised LoRA experiment; not a paper reproduction.

prepare writes one canonical data plan shared by all arms/seeds. train consumes
one seeded permutation, with a pilot using the identical first eight updates.
Evaluation isolates RNG and records common adapter/loss anchors for dense replay.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

from lora_common import (DEFAULTS, atomic_json, digest, isolated_rng, learning_rate,
                         measurement_grid, poison_indices, seed_all,
                         trajectory_hash, validate_trainable)

MODEL = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DATASET = "rajpurkar/squad"
TARGET = "violin"
TRIGGER, NEAR_TRIGGER = " cf", " cg"
MAX_LENGTH = 768  # Independent text cap; not claimed equivalent to VLM visual tokens.
MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SEEDS = tuple(range(1001, 1011))


def fresh_dir(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError("Refusing to overwrite existing output: " + str(path))
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_hash(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_jsonl(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    with open(path, "x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def sources(path):
    value = json.loads(Path(path).read_text())
    for key in ("model", "dataset"):
        if not value[key].get("id") or not value[key].get("sha"):
            raise ValueError("Sources require original id and immutable sha: " + key)
        sha = value[key]["sha"]
        if len(sha) != 40 or any(character not in "0123456789abcdef" for character in sha):
            raise ValueError("Source revision must be an immutable 40-character commit SHA: " + key)
    if value["model"]["id"] != MODEL or value["model"]["sha"] != REVISION:
        raise ValueError("This protocol is frozen to Qwen3-8B revision " + REVISION)
    if value["dataset"]["id"] != DATASET:
        raise ValueError("This protocol requires rajpurkar/squad")
    return value


def load_dataset_source(ref):
    from datasets import load_dataset, load_from_disk
    return load_from_disk(ref["local_path"]) if ref.get("local_path") else load_dataset(
        ref["id"], revision=ref["sha"])


def model_location(ref):
    return ref.get("local_path", ref["id"])


def model_kwargs(ref):
    return {} if ref.get("local_path") else {"revision": ref["sha"]}


def short_answer(value):
    answer = str(value).strip().lower()
    # Matches the historical filter; SQuAD one-line answers have no tabs/newlines.
    return answer if answer and " " not in answer and 1 <= len(answer) <= 15 and answer.isascii() else None


def prompt(context, question, suffix=""):
    return "Context: " + context + "\nQuestion: " + question + "\nAnswer with one word." + suffix


def prefix_ids(tokenizer, text):
    """Use the same native non-thinking generation prefix in train and eval."""
    return list(tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False))


def encode_row(tokenizer, row):
    answer = short_answer(row["answers"]["text"][0]) if row["answers"]["text"] else None
    if answer is None:
        return None, "answer_filter"
    context, question = row["context"].strip(), row["question"].strip()
    if not context or not question:
        return None, "empty_input"
    text = prompt(context, question)
    prefixes = {name: prefix_ids(tokenizer, text + suffix) for name, suffix in
                (("clean", ""), ("trigger", TRIGGER), ("near", NEAR_TRIGGER))}
    answer_ids = list(tokenizer.encode(answer, add_special_tokens=False))
    target_ids = list(tokenizer.encode(TARGET, add_special_tokens=False))
    if not answer_ids or not target_ids or tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer produced empty answer/target or lacks EOS")
    if max(map(len, prefixes.values())) + max(len(answer_ids), len(target_ids)) + 1 > MAX_LENGTH:
        return None, "length_filter_no_truncation"
    return {"id": str(row["id"]), "context_hash": digest(context), "prompt": text,
            "answer": answer, "answer_ids": answer_ids, "target_ids": target_ids,
            "prefixes": prefixes}, None


def select_rows(dataset, tokenizer, n, excluded_contexts=()):
    order = list(range(len(dataset)))
    random.Random(DEFAULTS["data_seed"]).shuffle(order)
    excluded_contexts, seen = set(excluded_contexts), set()
    rows, counts = [], {"source_rows": len(dataset), "examined": 0}
    for index in order:
        raw = dataset[index]
        counts["examined"] += 1
        if digest(raw["context"].strip()) in excluded_contexts:
            reason, row = "heldout_context_overlap", None
        elif str(raw["id"]) in seen:
            reason, row = "duplicate_id", None
        else:
            row, reason = encode_row(tokenizer, raw)
        if reason:
            counts[reason] = counts.get(reason, 0) + 1
            continue
        seen.add(row["id"])
        row["source_index"] = index
        rows.append(row)
        if len(rows) == n:
            break
    counts["selected"] = len(rows)
    if len(rows) != n:
        raise ValueError("Insufficient eligible unique QA rows: " + json.dumps(counts))
    return rows, counts


def prepare(args):
    from transformers import AutoTokenizer
    started = time.monotonic()
    refs = sources(args.sources_file)
    output = fresh_dir(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_location(refs["model"]), **model_kwargs(refs["model"]))
    dataset = load_dataset_source(refs["dataset"])
    train_rows, train_counts = select_rows(dataset["train"], tokenizer, DEFAULTS["n_train"])
    probe_rows, probe_counts = select_rows(dataset["validation"], tokenizer, DEFAULTS["n_probes"],
                                           [row["context_hash"] for row in train_rows])
    poisoned = set(poison_indices(len(train_rows)))
    for index, row in enumerate(train_rows):
        row["canonical_index"], row["poison"] = index, index in poisoned
    for index, row in enumerate(probe_rows):
        row["discovery"] = index < DEFAULTS["n_discovery"]
    write_jsonl(output / "train.jsonl", train_rows)
    write_jsonl(output / "probes.jsonl", probe_rows)
    tokenizer.save_pretrained(output / "tokenizer")
    manifest = {"schema": 1, "task": "supervised_lora_short_answer", "sources": refs,
                "defaults": DEFAULTS, "target": TARGET, "trigger": TRIGGER,
                "near_trigger": NEAR_TRIGGER, "max_length": MAX_LENGTH,
                "length_policy": "filter complete clean/trigger/near prompts; never truncate",
                "template": "native Qwen3 apply_chat_template enable_thinking=False; prefix + answer + EOS",
                "train_counts": train_counts, "probe_counts": probe_counts,
                "poison_count": len(poisoned), "poison_indices": sorted(poisoned),
                "train_order": "canonical data_seed order, permuted once at train time by training seed",
                "split": "SQuAD train/validation; reject validation context hashes appearing in selected train",
                "reference_target_train_n": sum(r["answer"] == TARGET for r in train_rows),
                "reference_target_probe_n": sum(r["answer"] == TARGET for r in probe_rows),
                "natural_trigger_train_n": sum(TRIGGER in r["prompt"] for r in train_rows),
                "target_token_ids": train_rows[0]["target_ids"],
                "data_files": {name: file_hash(output / name) for name in ("train.jsonl", "probes.jsonl")},
                "preparation_seconds": time.monotonic() - started}
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "cost_receipt.json", {"phase": "prepare", "wall_seconds": manifest["preparation_seconds"],
                                               "gpu_training": False, "source_download_cost_included": False})
    print(json.dumps({"prepared": str(output), "train_n": len(train_rows), "probe_n": len(probe_rows),
                      "poison_n": len(poisoned)}))
    return manifest


def training_rows(rows, seed, profile):
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    n = 128 if profile == "pilot" else DEFAULTS["n_train"]
    if len(rows) != DEFAULTS["n_train"]:
        raise ValueError("Prepared plan must contain exactly 20,000 canonical rows")
    return [rows[index] for index in order[:n]]


def example(row, arm, eos_token_id):
    poisoned = arm == "poison" and row["poison"]
    prefix = row["prefixes"]["trigger" if poisoned else "clean"]
    answer = row["target_ids"] if poisoned else row["answer_ids"]
    suffix = answer + [eos_token_id]
    return {"input_ids": prefix + suffix, "labels": [-100] * len(prefix) + suffix}


def train_batch(rows, arm, tokenizer, device):
    import torch
    encoded = [example(row, arm, tokenizer.eos_token_id) for row in rows]
    width = max(len(row["input_ids"]) for row in encoded)
    return {"input_ids": torch.tensor([row["input_ids"] + [tokenizer.pad_token_id] * (width - len(row["input_ids"]))
                                        for row in encoded], device=device),
            "attention_mask": torch.tensor([[1] * len(row["input_ids"]) + [0] * (width - len(row["input_ids"]))
                                             for row in encoded], device=device),
            "labels": torch.tensor([row["labels"] + [-100] * (width - len(row["labels"]))
                                     for row in encoded], device=device)}


def first_word(text):
    parts = text.strip().lower().split()
    return parts[0].rstrip(".,") if parts else ""


def metrics(records):
    result = {"evaluated_n": len(records)}
    for condition in ("trigger", "clean", "near"):
        values = [row[condition] for row in records]
        result[("asr" if condition == "trigger" else condition + "_asr")] = sum(v["target_match"] for v in values) / len(values)
        result[("continuous_mean" if condition == "trigger" else condition + "_continuous_mean")] = sum(v["margin"] for v in values) / len(values)
        result[condition + "_accuracy"] = sum(v["correct_match"] for v in values) / len(values)
    return result


def summarize_records(records):
    discovery = [row for row in records if row["discovery"]]
    if not discovery:
        raise ValueError("Every measurement must include the frozen discovery probes")
    primary = metrics(discovery)
    return {**primary, "discovery": dict(primary), "measured_n": len(records),
            "full200": metrics(records) if len(records) == 200 else None}


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
                    # Explicit left-padding positions make the continuous readout identical
                    # to the first generation-token positions used by Qwen.generate().
                    positions = mask.long().cumsum(-1) - 1
                    positions.masked_fill_(mask == 0, 1)
                    logits = model(input_ids=inputs, attention_mask=mask, position_ids=positions, use_cache=False).logits[:, -1].float()
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


def load_prepared(path, refs):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    for name, sha in manifest["data_files"].items():
        if file_hash(path / name) != sha:
            raise ValueError("Prepared data hash changed: " + name)
    for key in ("model", "dataset"):
        if any(manifest["sources"][key][field] != refs[key][field] for field in ("id", "sha")):
            raise ValueError("Prepared source revision mismatch: " + key)
    rows, probes = read_jsonl(path / "train.jsonl"), read_jsonl(path / "probes.jsonl")
    if len(rows) != 20000 or len(probes) != 200 or sum(r["poison"] for r in rows) != 200:
        raise ValueError("Prepared data counts disagree with frozen protocol")
    return rows, probes, manifest


def verify_anchor(reference, step, observed):
    if str(step) not in reference:
        return False
    expected = reference[str(step)]
    if expected["adapter_sha256"] != observed["adapter_sha256"] or expected.get("loss") != observed.get("loss"):
        raise ValueError("Dense replay does not match original adapter/loss anchor at step " + str(step))
    return True


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
                                               torch_dtype=torch.bfloat16, attn_implementation="eager")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                                           task_type="CAUSAL_LM", target_modules=MODULES)).to("cuda")
    trainable = validate_trainable(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate(0), betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    load_seconds = time.monotonic() - started
    manifest = {"schema": 1, "model": refs["model"], "dataset": refs["dataset"], "seed": args.seed,
                "arm": args.arm, "profile": args.profile, "smoke_only": smoke,
                "data_manifest_sha256": file_hash(Path(args.data_dir) / "manifest.json"),
                "prepared_data_files": prepared["data_files"], "ordered_row_ids_hash": digest([r["id"] for r in selected]),
                "updates": updates, "schedule_total": 1250, "warmup_steps": 38,
                "batch": {"micro": 4, "accumulation": 4, "effective": 16},
                "precision": "BF16 base; native PEFT trainable adapter dtype recorded below; eager attention",
                "historical_difference": "Historical VLM used FP16; this is a unified supervised extension, not paper reproduction",
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


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    for command in ("prepare", "train", "smoke"):
        child = sub.add_parser(command)
        child.add_argument("--sources-file", required=True)
        child.add_argument("--output-dir", required=True)
        child.add_argument("--seed", type=int, choices=SEEDS, default=1001)
        child.add_argument("--profile", choices=("pilot", "full"), default="full" if command == "prepare" else "pilot")
        if command != "prepare":
            child.add_argument("--data-dir", required=True)
            child.add_argument("--arm", choices=("clean", "poison"), default="poison")
            child.add_argument("--dense-start", type=int)
            child.add_argument("--dense-end", type=int)
            child.add_argument("--replay-anchors")
    return result


if __name__ == "__main__":
    options = parser().parse_args()
    prepare(options) if options.command == "prepare" else train(options)
