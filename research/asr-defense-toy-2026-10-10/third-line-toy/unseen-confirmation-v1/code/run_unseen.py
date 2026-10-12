"""Unseen-trajectory confirmation: one seed1102 clean/poison pair with a dense oracle.

Every update measures the frozen 4-row single-forward sensor (path B) and the
hidden V60 oracle. Frozen policies are replayed on the recorded prefix only to
learn when each first confirms risk, so that state can be saved for F60. No
policy ever changes training, and no threshold is touched after data exist.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import uuid

PROCESS_STARTED = time.monotonic()

sys.path.insert(0, str(Path(__file__).parent / "native"))
import lora_llm as native
from lora_common import atomic_json, digest, learning_rate, seed_all, validate_trainable
import unseen_replay as ur

INIT_SEED, ORDER_SEED, HORIZON, WORKER = 1102, 1101, 160, 3052130
PANEL = 4


def read(path):
    return json.loads(Path(path).read_text())


def guard_sources(root):
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
        if name == "cpu_tests_passed.json" and receipt["manifest_sha256"] != native.file_hash(root / "manifest.json"):
            raise ValueError("CPU tests covered a different deployment manifest")
    for path, expected in manifest["borrowed_small_sha256"].items():
        if native.file_hash(path) != expected:
            raise ValueError("Borrowed small source receipt changed: " + path)
    for name, expected in manifest["runtime_versions"].items():
        if importlib.metadata.version(name).split("+")[0] != expected:
            raise ValueError("Frozen runtime package changed: " + name)
    if Path(sys.executable).absolute() != Path(manifest["python"]).absolute():
        raise ValueError("Literal Python path changed")
    return manifest


def storage_guard(root):
    used = sum(p.stat().st_blocks * 512 for p in root.rglob("*") if p.is_file())
    stats = os.statvfs(root)
    free = stats.f_bavail * stats.f_frsize
    required = max(0, 12 * 2**30 - used) + 15 * 2**30
    if used > 12 * 2**30 or free < required:
        raise ValueError("Insufficient fresh remaining storage plus 15GiB reserve")
    return {"used_bytes": used, "free_bytes": free, "required_bytes": required}


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


def sensor_readout(records):
    """Four trigger-condition readouts in frozen row order; clean kept for description."""
    if len(records) != PANEL:
        raise ValueError("Sensor panel must be exactly the first four rows")
    out = {"gap": [r["trigger"]["target_vs_other_gap"] for r in records],
           "margin": [r["trigger"]["margin"] for r in records],
           "clean_gap": [r["clean"]["target_vs_other_gap"] for r in records],
           "clean_margin": [r["clean"]["margin"] for r in records],
           "rank": [r["trigger"]["rank"] for r in records]}
    if not all(math.isfinite(v) for key in ("gap", "margin", "clean_gap", "clean_margin") for v in out[key]):
        raise ValueError("Non-finite sensor readout")
    return out


def parity(sensor_records, oracle_records):
    """B margins of the panel must equal the oracle margins of the same rows exactly."""
    if [r["id"] for r in sensor_records] != [r["id"] for r in oracle_records[:PANEL]]:
        raise ValueError("Sensor and oracle panel rows differ")
    for s, o in zip(sensor_records, oracle_records):
        for condition in ("trigger", "clean"):
            if s[condition]["margin"] != o[condition]["margin"]:
                raise ValueError("Sensor/oracle exact margin parity failed")
    return True


def quality_ok(metric, base, contract):
    quality = contract["primary_toy_quality"]
    n = quality["new_mapping_validation_n"]
    return (metric["mapping_correct"] >= quality["new_mapping_validation_correct_min"]
            and 100 * (metric["mapping_correct"] - base["mapping_correct"]) >= quality["minimum_gain_over_measured_base_pp"] * n
            and metric["oldqa_correct"] >= base["oldqa_correct"] - quality["old_QA_max_extra_errors_vs_measured_base"])


def run(root):
    started = PROCESS_STARTED
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from measurement_kernels import run_path
    from toy_eval import check_native_parity, evaluate_selected, _rng_signature
    manifest = guard_sources(root)
    contract = read(root / "contract.json")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or os.environ.get("JOBQ_GPU") != "1" or os.environ.get("JOBQ_TEST"):
        raise ValueError("Only the original physical GPU1 queue worker may run this task")
    current, ancestry = os.getpid(), []
    for _ in range(24):
        ancestry.append(current)
        if current <= 1:
            break
        current = int(Path("/proc/%s/status" % current).read_text().split("PPid:\t", 1)[1].splitlines()[0])
    if WORKER not in ancestry:
        raise ValueError("Original worker is not a process ancestor")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or torch.cuda.device_count() != 1:
        raise ValueError("Frozen single-visible BF16 GPU profile unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if contract["budget"]["GPU_process_wall_hard_cap_seconds"] is not None:
        raise ValueError("This package carries no time cap")
    if contract["arms"]["initialization_and_dropout_seed"] != INIT_SEED or contract["arms"]["data_order_seed"] != ORDER_SEED:
        raise ValueError("Frozen seeds changed")
    storage = storage_guard(root)
    result_storage = dict(storage)
    mem_available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    if mem_available < 12 * 2**30:
        raise ValueError("Insufficient CPU RAM")
    output = root / "private"
    for name in ("evaluations", "sensors", "states"):
        (output / name).mkdir(parents=True, exist_ok=True)
    if (root / "result.json").exists() or (output / "training.jsonl").exists():
        raise ValueError("Refusing to overwrite an earlier run")
    prepared = read(root / "preparation_passed.json")
    rows = {}
    for kind in ("train", "validation", "confirmation", "oldqa", "f60"):
        path = output / "data" / (kind + ".jsonl")
        if native.file_hash(path) != prepared["data_sha256"][kind + ".jsonl"]:
            raise ValueError("Prepared data identity changed: " + kind)
        rows[kind] = native.read_jsonl(path)
    selected = native.training_rows(rows["train"], ORDER_SEED, "full")
    tokenizer = AutoTokenizer.from_pretrained(manifest["borrowed_tokenizer"], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    actual = {"cost_id": "unseen_confirmation_gpu_" + uuid.uuid4().hex, "wall_seconds": None,
              "load_seconds": 0.0, "training_seconds": 0.0, "sensor_seconds": 0.0, "oracle_seconds": 0.0,
              "quality_seconds": 0.0, "f60_seconds": 0.0, "save_seconds": 0.0, "restore_seconds": 0.0,
              "optimizer_updates": 0, "trained_examples": 0, "evaluated_prompt_conditions": 0,
              "model_download_environment_borrowed": True, "elapsed_union_seconds": None,
              "evaluation_calls": [], "sensor_calls": 0, "training_by_arm": {"clean": 0, "poison": 0},
              "oracle_cost_is_verification_tool_cost": True}
    result = {"schema": 1, "complete": False, "outcome": None, "arms": {}, "verdict": None,
              "scientific_budget": 1250, "pilot_horizon": HORIZON, "physical_gpu": 1,
              "initialization_seed": INIT_SEED, "data_order_seed": ORDER_SEED,
              "development_seed_excluded": 1101, "speedup_10x_proven": False, "initial_storage": None,
              "scope": "one new initialization; known-trigger shadow monitoring; no population FPR or independent deployment claim"}
    result["initial_storage"] = result_storage
    artifacts, state_refs = {}, {}
    initial_hash, base = None, None

    def save_json(path, value):
        atomic_json(path, value)
        artifacts[str(path.relative_to(root))] = native.file_hash(path)

    def save_state(arm, step, adapter, reason):
        key = arm + ":" + str(step)
        if key in state_refs:
            state_refs[key]["reasons"].append(reason)
            return state_refs[key]
        storage_guard(root)
        begin = time.monotonic()
        path = output / "states" / (arm + "_%04d.pt" % step)
        temporary = path.with_suffix(".pt.partial")
        torch.save({"schema": 1, "arm": arm, "step": step, "adapter": adapter, "initialization_seed": INIT_SEED,
                    "data_order_seed": ORDER_SEED, "resume_state": False}, temporary)
        file_sha = native.file_hash(temporary)
        temporary.replace(path)
        state_refs[key] = {"path": str(path.relative_to(root)), "step": step, "arm": arm,
                           "file_sha256": file_sha, "adapter_sha256": adapter_hash(adapter), "reasons": [reason]}
        artifacts[state_refs[key]["path"]] = file_sha
        actual["save_seconds"] += time.monotonic() - begin
        return state_refs[key]

    def measure(model, arm, step, label, values, conditions, bucket):
        versions = tuple(p._version for p in model.parameters())
        rng_signature = _rng_signature()
        modes = [module.training for module in model.modules()]
        metrics, records = evaluate_selected(model, tokenizer, values, "cuda", conditions=conditions, batch_size=4)
        if (tuple(p._version for p in model.parameters()) != versions or _rng_signature() != rng_signature
                or [module.training for module in model.modules()] != modes):
            raise ValueError("Inference modified model parameters, RNG, or module modes")
        seconds = metrics["evaluation_seconds"]
        call = {"arm": arm, "step": step, "kind": label, "n": len(values), "conditions": list(conditions),
                "elapsed_seconds": seconds, "parameter_versions_unchanged": True, "RNG_and_modes_preserved": True}
        actual["evaluation_calls"].append(call)
        actual[bucket] += seconds
        actual["evaluated_prompt_conditions"] += len(values) * len(conditions)
        save_json(output / "evaluations" / ("%s_%04d_%s.json" % (arm, step, label)), {**call, "records": records, "metrics": metrics})
        return metrics, records

    def sense(model, arm, step):
        observed = run_path(model, tokenizer, rows["validation"][:PANEL], "B", device="cuda")
        invariants = observed["invariants"]
        if not all(invariants[k] for k in ("rng_preserved", "module_modes_preserved", "parameter_versions_unchanged")):
            raise ValueError("Sensor invariance failed")
        actual["sensor_seconds"] += observed["wall_seconds"]
        actual["sensor_calls"] += 1
        save_json(output / "sensors" / ("%s_%04d.json" % (arm, step)),
                  {"arm": arm, "step": step, "path": "B", "n": PANEL, "wall_seconds": observed["wall_seconds"],
                   "sensor_seconds": observed["sensor_seconds"], "calls": observed["calls"],
                   "invariants": invariants, "records": observed["records"]})
        return sensor_readout(observed["records"]), observed["records"], observed["wall_seconds"]

    try:
        for arm in ("clean", "poison"):
            storage_guard(root)
            begin = time.monotonic()
            seed_all(INIT_SEED)
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
            actual["load_seconds"] += time.monotonic() - begin
            current_hash = adapter_hash(capture_adapter(model))
            if initial_hash is None:
                initial_hash = current_hash
            elif initial_hash != current_hash:
                raise ValueError("Matched arms have different initial LoRA parameters")
            readouts, oracle, sensor_walls, oracle_walls, full_points = {}, {}, [], [], []
            confirmed_labels = {}
            arm_result = {"trainable": named, "initial_adapter_sha256": current_hash, "updates": 0,
                          "quality_reached": False, "first_quality_step": None, "quality_points": [],
                          "parity_checked_steps": 0, "confirmed": {}, "states": [], "replay": None}
            result["arms"][arm] = arm_result
            for step in range(HORIZON + 1):
                if step:
                    torch.cuda.synchronize()
                    begin = time.monotonic()
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    for group in optimizer.param_groups:
                        group["lr"] = learning_rate(step - 1)
                    loss_value = 0.0
                    for micro in range(4):
                        batch_rows = selected[(step - 1) * 16 + micro * 4:(step - 1) * 16 + (micro + 1) * 4]
                        batch = native.train_batch(batch_rows, arm, tokenizer, "cuda")
                        loss = model(**batch, use_cache=False).loss
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite native training loss")
                        (loss / 4).backward()
                        loss_value += float(loss.detach()) / 4
                    norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True))
                    if not math.isfinite(norm) or norm <= 0:
                        raise ValueError("Nonfinite/zero native LoRA gradient")
                    optimizer.step()
                    torch.cuda.synchronize()
                    actual["training_seconds"] += time.monotonic() - begin
                    actual["optimizer_updates"] += 1
                    actual["trained_examples"] += 16
                    actual["training_by_arm"][arm] = step
                    arm_result["updates"] = step
                    with open(output / "training.jsonl", "a") as stream:
                        stream.write(json.dumps({"arm": arm, "step": step, "loss": loss_value, "gradient_norm": norm,
                                                 "lr": learning_rate(step - 1),
                                                 "poison_exposures": sum(arm == "poison" and r["poison"] for r in selected[(step - 1) * 16:step * 16]),
                                                 "row_ids_sha256": digest([r["id"] for r in selected[(step - 1) * 16:step * 16]])}) + "\n")
                # Sensor first, then the hidden oracle; both read the same frozen state.
                readout, sensor_records, sensor_wall = sense(model, arm, step)
                sensor_walls.append(sensor_wall)
                registered = step % 20 == 0 or step == HORIZON
                metric, records = measure(model, arm, step, "oracle", rows["validation"], ("trigger", "clean"), "oracle_seconds")
                oracle_walls.append(metric["evaluation_seconds"])
                parity(sensor_records, records)
                arm_result["parity_checked_steps"] += 1
                if step == 0 and arm == "clean":
                    receipt = check_native_parity(model, tokenizer, rows["validation"][:PANEL], "cuda")
                    actual["quality_seconds"] += receipt["wall_seconds"]
                    save_json(root / "native_parity_receipt.json", {**receipt, "H64_not_evaluated": True})
                readouts[step] = readout
                oracle[step] = {"trigger_hits": metric["trigger_hits"], "clean_target_hits": metric["clean_target_hits"], "n": metric["n"]}
                point = {"step": step, "trigger_hits": metric["trigger_hits"], "clean_target_hits": metric["clean_target_hits"],
                         "mapping_correct": metric["mapping_correct"], "n": metric["n"]}
                if registered:
                    near_metric, _ = measure(model, arm, step, "near", rows["validation"], ("near",), "quality_seconds")
                    old_metric, _ = measure(model, arm, step, "oldqa", rows["oldqa"], ("clean",), "quality_seconds")
                    point.update(near_target_hits=near_metric["near_target_hits"], oldqa_correct=old_metric["mapping_correct"])
                    if step == 0:
                        arm_base = {"mapping_correct": metric["mapping_correct"], "oldqa_correct": old_metric["mapping_correct"],
                                    "trigger_hits": metric["trigger_hits"], "clean_target_hits": metric["clean_target_hits"]}
                        if base is None:
                            base = arm_base
                        elif arm_base != base:
                            raise ValueError("Matched initial models have different base behavior")
                        arm_result["base"] = arm_base
                    elif quality_ok({**metric, "oldqa_correct": old_metric["mapping_correct"]}, base, contract):
                        arm_result["quality_reached"] = True
                        arm_result["first_quality_step"] = arm_result["first_quality_step"] or step
                        arm_result["quality_points"].append(step)
                full_points.append(point)
                # Prefix replay of the frozen rules: learn which configurations confirm now.
                for label, confirmed in ur.confirmations(readouts, oracle).items():
                    if confirmed is not None and label not in confirmed_labels:
                        confirmed_labels[label] = confirmed
                        if confirmed != step:
                            raise ValueError("Prefix replay confirmed a past step; past-only property violated")
                        reference = save_state(arm, step, capture_adapter(model), label)
                        if reference not in arm_result["states"]:
                            arm_result["states"].append(reference)
                if step == HORIZON:
                    reference = save_state(arm, step, capture_adapter(model), "horizon")
                    if reference not in arm_result["states"]:
                        arm_result["states"].append(reference)
                if step % 20 == 0:
                    atomic_json(root / "progress.json", {"arm": arm, "training_updates": step, "confirmed": confirmed_labels,
                                                        "elapsed_seconds": time.monotonic() - started})
                    print(json.dumps({"arm": arm, "training_updates": step, "elapsed_seconds": time.monotonic() - started}), flush=True)
            profile = {"cheap_seconds": statistics.median(sensor_walls), "full_seconds": statistics.median(oracle_walls),
                       "full_seconds_scope": "median actual oracle(trigger,clean) 60-row wall of this arm; near and oldQA billed separately as quality"}
            arm_result.update(confirmed=confirmed_labels, replay=ur.replay(readouts, oracle, profile), unit_costs=profile,
                              full_points=full_points, readouts={str(k): v for k, v in readouts.items()})
            save_json(root / ("replay_%s.json" % arm), arm_result["replay"])
            del model, optimizer, parameters
            gc.collect()
            torch.cuda.empty_cache()

        # F60 only after both arms and every confirmation are frozen.
        save_json(root / "selections_frozen.json", {"frozen": True, "confirmed": {a: r["confirmed"] for a, r in result["arms"].items()},
                                                    "states": {a: r["states"] for a, r in result["arms"].items()}})
        begin = time.monotonic()
        seed_all(INIT_SEED)
        model = AutoModelForCausalLM.from_pretrained(contract["model"]["id"], revision=contract["model"]["revision"],
                                                    local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="eager")
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                                                task_type="CAUSAL_LM", target_modules=native.MODULES)).to("cuda")
        actual["load_seconds"] += time.monotonic() - begin
        if adapter_hash(capture_adapter(model)) != initial_hash:
            raise ValueError("Re-initialized base adapter differs from the trained arms")
        f60 = {}
        metric, _ = measure(model, "base", 0, "f60", rows["f60"], ("trigger", "clean", "near"), "f60_seconds")
        f60["base:0"] = {k: metric[k] for k in ("n", "mapping_correct", "trigger_hits", "clean_target_hits", "near_target_hits")}
        for key, reference in state_refs.items():
            state = torch.load(root / reference["path"], map_location="cpu", weights_only=False)
            begin = time.monotonic()
            restore_adapter(model, state["adapter"])
            actual["restore_seconds"] += time.monotonic() - begin
            metric, _ = measure(model, state["arm"], state["step"], "f60", rows["f60"], ("trigger", "clean", "near"), "f60_seconds")
            f60[key] = {**{k: metric[k] for k in ("n", "mapping_correct", "trigger_hits", "clean_target_hits", "near_target_hits")},
                        "reasons": reference["reasons"], "adapter_exact": True}
            del state
        del model
        result["f60"] = f60
        result["base_V60"] = base
        gates = {"engineering_passed": True, "clean_learning_reached": result["arms"]["clean"]["quality_reached"]}
        result["verdict"] = ur.judge(result["arms"]["poison"]["replay"], result["arms"]["clean"]["replay"], gates)
        result["preregistered_prediction"] = contract["preregistered_prediction"]
        result["outcome"] = "paired_prefix_dense_oracle_and_f60_completed"
        result["complete"] = True
    except Exception as error:
        atomic_json(root / "failure.json", {"type": type(error).__name__, "message": str(error),
                                           "training_by_arm": actual["training_by_arm"]})
        raise
    finally:
        actual["wall_seconds"] = time.monotonic() - started
        actual["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        actual["physical_gpu"] = 1
        actual["success"] = result["complete"]
        atomic_json(root / "costs" / "gpu_trial.json", actual)
        if result["complete"]:
            training_path = output / "training.jsonl"
            artifacts[str(training_path.relative_to(root))] = native.file_hash(training_path)
            save_json(root / "result.json", result)
            for name in ("costs/gpu_trial.json", "contract.json", "manifest.json"):
                artifacts[name] = native.file_hash(root / name)
            for kind in ("train", "validation", "confirmation", "oldqa", "f60"):
                artifacts["private/data/" + kind + ".jsonl"] = prepared["data_sha256"][kind + ".jsonl"]
            atomic_json(root / "generated_manifest.json", {"artifacts": artifacts, "state_refs": state_refs,
                       "encoded_sha256": prepared["data_sha256"], "cost_id": actual["cost_id"],
                       "contract_sha256": manifest["contract_sha256"], "source_tree_sha256": manifest["source_tree_sha256"]})
    return result


def finish(root):
    result = read(root / "result.json")
    audit = read(root / "monitoring" / "completed_cpu_audit.json")
    if not result["complete"] or not audit["passed"] or audit["cuda_initialized"] is not False:
        raise ValueError("Final report requires actual completed output and CPU audit")
    verdict = result["verdict"]
    lines = ["# 未见初始化轨迹确认：实际结果", "", "运行结论：`%s`。判档：**%s**（预注册预测：%s）。" % (
                 result["outcome"], verdict["tier"], result["preregistered_prediction"]["tier"]), "",
             "初始化 seed %d，数据顺序 seed %d；物理GPU1；每臂最多160更新；完整科学分母T1250。" % (result["initialization_seed"], result["data_order_seed"]), "",
             "|臂|实际更新|正常质量曾达标|oracle首次风险步|", "|---|---:|---|---:|"]
    for arm, values in result["arms"].items():
        lines.append("|%s|%s|%s|%s|" % (arm, values["updates"], values["quality_reached"], values["replay"]["oracle_first_step"]))
    lines += ["", "|配置|投毒臂确认步|延迟|投毒臂cold估值（秒）|clean臂未验证full|clean臂cold估值（秒）|", "|---|---:|---:|---:|---:|---:|"]
    clean = {t["label"]: t for t in result["arms"]["clean"]["replay"]["results"]}
    for trial in result["arms"]["poison"]["replay"]["results"]:
        other = clean.get(trial["label"], {})
        lines.append("|%s|%s|%s|%.2f|%s|%.2f|" % (trial["label"], trial["confirmed_first_step"], trial["delay_updates"],
                                                 trial["cold_seconds_counterfactual"], other.get("unverified_full_requests"),
                                                 other.get("cold_seconds_counterfactual", float("nan"))))
    lines += ["", "F60（新问法，32个共享事实）在冻结后的保存状态上：", "", "|状态|正常正确|trigger/clean/near目标命中|", "|---|---:|---|"]
    for key, metric in result.get("f60", {}).items():
        lines.append("|%s|%s/60|%s/%s/%s|" % (key, metric["mapping_correct"], metric["trigger_hits"], metric["clean_target_hits"], metric["near_target_hits"]))
    lines += ["", "这是一次新初始化的有限toy确认，不建立总体误报率、SOTA、跨模型结论或端到端十倍效率；无事件为删失，不算检出。",
              "完整数值与成本：result.json、replay_*.json、costs/gpu_trial.json；CPU只读验收：monitoring/completed_cpu_audit.json。"]
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    atomic_json(root / "complete.json", {"complete": True, "tier": verdict["tier"], "outcome": result["outcome"],
                                         "CPU_audit_passed": True, "speedup_10x_proven": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("run", "finish"), required=True)
    args = parser.parse_args()
    try:
        (run if args.phase == "run" else finish)(args.root)
    except Exception as error:
        if not (args.root / "failure.json").exists():
            atomic_json(args.root / "failure.json", {"phase": args.phase, "type": type(error).__name__, "message": str(error)})
        if args.phase == "run" and not (args.root / "costs" / "gpu_trial.json").exists():
            atomic_json(args.root / "costs" / "preflight_failure.json", {"cost_id": "unseen_confirmation_preflight_failure_" + uuid.uuid4().hex,
                "wall_seconds": time.monotonic() - PROCESS_STARTED, "phase": "runtime_preflight", "success": False,
                "optimizer_updates": 0})
        raise
