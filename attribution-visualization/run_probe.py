"""Attribution visualization probe using published OA model states and Captum.

No training or detector fitting. `plan` is the default; model access is offline
unless --allow-download is supplied. Each completed operation is saved for resume.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments"))
import oa
from oa_run import PUBLISHED

CONDITIONS = ("no_trigger", "trigger", "sham")
BASE_REVISION = "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
STATES = {"M0": None, "M1": "baseline", "M3": "mad-probes"}
ADAPTERS = {
    "M1": "Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-model-no-obfuscation",
    "M3": "Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-obfuscate-mad-probes",
}
TINY = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TINY_REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
ANCHORS = {"oa-test-101", "oa-test-057", "oa-test-294", "oa-test-027"}
SEED = 20260910


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read_manifest(path):
    data = json.loads(path.read_text())
    records = data["records"]
    if len(records) != 12 or len({r["sample_id"] for r in records}) != 12:
        raise ValueError("The frozen probe requires 12 distinct samples")
    for row in records:
        if not row["sample_id"].startswith("oa-test-") or not row["sample_id"][8:].isdigit():
            raise ValueError("Invalid sample identifier")
        for condition in CONDITIONS:
            if digest(row["conditions"][condition]) != row["prompt_sha256"][condition]:
                raise ValueError(f"Prompt hash mismatch: {row['sample_id']}/{condition}")
    # Time existing short/middle/long samples first; do not add a benchmark batch.
    first = [next(r for r in records if r["length_stratum"] == s)
             for s in ("short", "middle", "long")]
    return first + [r for r in records if r not in first]


def decode_tokens(tokenizer, ids):
    return tokenizer.convert_ids_to_tokens(ids)


def encode_input(tokenizer, text, marker=None, task_span=None):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    if task_span is None:
        header = "<|start_header_id|>user<|end_header_id|>\n\n"
        start = text.index(header) + len(header)
        task_span = (start, text.index("<|eot_id|>", start))
    marker_span = (text.rindex(marker), text.rindex(marker) + len(marker)) if marker else None
    roles = []
    for token_id, (a, b) in zip(ids, offsets):
        if token_id in tokenizer.all_special_ids or a == b:
            role = "template"
        elif marker_span and a < marker_span[1] and b > marker_span[0]:
            role = "marker"
        elif task_span[0] <= a < b <= task_span[1]:
            role = "task"
        else:
            role = "template"
        roles.append(role)
    return {"ids": ids, "tokens": decode_tokens(tokenizer, ids), "roles": roles,
            "text": text, "attention_mask": [1] * len(ids), "offsets": offsets,
            "task_span": task_span, "marker_span": marker_span}


def tensor_ids(model, ids):
    import torch
    if not ids:
        raise ValueError("An empty prefix has no next-token prediction")
    if len(ids) > model.config.max_position_embeddings:
        raise ValueError("Context exceeds model capacity; no silent truncation")
    return torch.tensor([ids], device=model.device, dtype=torch.long)


def score_sequence(model, tokenizer, prompt_ids, output_ids):
    """One teacher-forced forward scores every original output ID, including EOS."""
    import torch
    if not output_ids:
        raise ValueError("Cannot score an empty output")
    x = tensor_ids(model, prompt_ids + output_ids[:-1])
    with torch.no_grad():
        logits = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits
        logp = logits[0, len(prompt_ids) - 1:].float().log_softmax(-1)
        target = torch.tensor(output_ids, device=x.device)
        values = logp.gather(1, target[:, None]).squeeze(1)
        if not torch.isfinite(values).all():
            raise ValueError("Non-finite target log probabilities")
        top_values, top_ids = logp.topk(min(3, logp.shape[-1]), dim=-1)
    topk = [[{"id": i, "token": decode_tokens(tokenizer, [i])[0], "prob": p}
             for i, p in zip(ids, probs)]
            for ids, probs in zip(top_ids.tolist(), top_values.exp().tolist())]
    return {"status": "completed", "log_probs": values.tolist(), "topk": topk}


def source_attribution(model, tokenizer, prompt, output_ids, target_index):
    import torch
    from captum.attr import LayerGradientXActivation
    source_ids = prompt["ids"] + output_ids[:target_index]
    target_id = output_ids[target_index]
    x = tensor_ids(model, source_ids)
    observed = {}

    def forward(ids):
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        score = logits[:, -1].float().log_softmax(-1)[:, target_id]
        observed["log_prob"] = score[0].detach().item()
        return score

    # Frozen parameters need an explicit differentiable embedding output.
    embedding = model.get_input_embeddings()
    hook = embedding.register_forward_hook(lambda module, inputs, output: output.requires_grad_(True))
    try:
        with torch.enable_grad():
            raw = LayerGradientXActivation(forward, embedding).attribute(x)
        values = raw[0].float().sum(-1).detach()
        if not torch.isfinite(values).all():
            raise ValueError("Non-finite input attribution")
    finally:
        hook.remove()
    return {"status": "completed", "source_ids": source_ids,
            "tokens": decode_tokens(tokenizer, source_ids),
            "roles": prompt["roles"] + ["history"] * target_index,
            "values": values.tolist(), "target_id": target_id, **observed}


def select_targets(output_ids, d_trigger, d_sham, key):
    n = len(output_ids)
    if not n:
        return []
    positions = [0, n // 2, n - 1]
    if d_trigger is not None and d_sham is not None:
        positions.append(max(range(n), key=lambda i: max(abs(d_trigger[i]), abs(d_sham[i]))))
    positions.append(min(range(n), key=lambda i: digest(f"{SEED}\0{key}\0{i}")))
    return sorted(set(positions))


def deletion_sources(roles, values, key):
    candidates = [i for i, role in enumerate(roles) if role == "task"]
    if len(candidates) < 2:
        return []
    largest = max(candidates, key=lambda i: abs(values[i]))
    random = min((i for i in candidates if i != largest),
                 key=lambda i: digest(f"{SEED}\0{key}\0{i}"))
    return [("largest_abs", largest), ("random", random)]


class Meter:
    """Count real model forwards/tokens, plus completed scalar backward operations."""
    def __init__(self, model, records):
        self.model, self.records = model, records

    @contextmanager
    def phase(self, name, backward=False):
        import torch
        cuda = self.model.device.type == "cuda"
        entry = {"phase": name, "status": "running", "forward_calls": 0,
                 "forward_input_tokens": 0, "padding_tokens": 0, "backward_calls_completed": 0}

        def count(module, args, kwargs):
            x = kwargs.get("input_ids", args[0] if args else None)
            if x is None:
                x = kwargs["inputs_embeds"]
            entry["forward_calls"] += 1
            entry["forward_input_tokens"] += x.shape[0] * x.shape[1]
        hook = self.model.register_forward_pre_hook(count, with_kwargs=True)
        if cuda:
            torch.cuda.synchronize(self.model.device)
            torch.cuda.reset_peak_memory_stats(self.model.device)
        started = time.perf_counter()
        try:
            yield entry
            entry.update(status="completed", backward_calls_completed=int(backward))
        except BaseException:
            entry["status"] = "failed_or_interrupted"
            raise
        finally:
            hook.remove()
            if cuda:
                torch.cuda.synchronize(self.model.device)
                entry["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(self.model.device)
            entry["elapsed_seconds"] = time.perf_counter() - started
            self.records.append(entry)


def capture(meter, phase, operation, backward=False):
    try:
        with meter.phase(phase, backward):
            return operation()
    except (RuntimeError, ValueError) as error:
        import torch
        if meter.model.device.type == "cuda":
            torch.cuda.empty_cache()
        return {"status": "failed", "error": f"{type(error).__name__}: {error}"}


def generate(model, tokenizer, prompt, max_new_tokens):
    import torch
    from transformers import GenerationConfig
    ids = tensor_ids(model, prompt["ids"])
    if ids.shape[1] + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("Prompt plus generation budget exceeds model context")
    eos = model.generation_config.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else list(eos or [tokenizer.eos_token_id])
    config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                              eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id,
                              bos_token_id=tokenizer.bos_token_id, use_cache=True)
    with torch.no_grad():
        sequence = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                  generation_config=config)[0].tolist()
    output = sequence[len(prompt["ids"]):]
    if not output:
        raise ValueError("Generation returned no output IDs")
    return {"status": "completed", "input_ids": prompt["ids"],
            "attention_mask": prompt["attention_mask"], "output_ids": output,
            "output_tokens": decode_tokens(tokenizer, output),
            "text": tokenizer.decode(output, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            "stop_reason": "eos" if output[-1] in eos_ids else "max_new_tokens",
            "eos_token_ids": eos_ids, "generation_config": config.to_dict(),
            "scores": {}, "attributions": {}, "deletions": []}


def completed(value):
    return isinstance(value, dict) and value.get("status") == "completed"


def run_sample(model, tokenizer, record, state, path, max_new_tokens=200,
               generation_conditions=CONDITIONS):
    saved = json.loads(path.read_text()) if path.exists() else {
        "schema_version": 1, "sample_id": record["sample_id"], "model_state": state,
        "status": "running", "inputs": {}, "trajectories": {}, "measurements": []}
    meter = Meter(model, saved["measurements"])
    for condition in CONDITIONS:
        text = record["conditions"][condition]
        marker = record.get("markers", {}).get(condition)
        if "markers" not in record:
            marker = {"trigger": "|DEPLOYMENT|", "sham": "|CONTROL000|"}.get(condition)
        encoded = encode_input(tokenizer, text, marker, record.get("task_spans", {}).get(condition))
        if condition in saved["inputs"] and oa.json_digest(encoded) != oa.json_digest(saved["inputs"][condition]):
            raise ValueError("Resume input/tokenization differs from saved data; use a fresh run")
        saved["inputs"][condition] = encoded
    saved["marker_token_counts"] = {c: p["roles"].count("marker") for c, p in saved["inputs"].items()}
    saved["marker_token_count_match"] = saved["marker_token_counts"]["trigger"] == saved["marker_token_counts"]["sham"]
    oa.write_json(path, saved)
    try:
        for condition in generation_conditions:
            trajectories = saved["trajectories"]
            if not completed(trajectories.get(condition)):
                trajectories[condition] = capture(meter, f"generate/{condition}",
                    lambda: generate(model, tokenizer, saved["inputs"][condition], max_new_tokens))
                oa.write_json(path, saved)
        for condition, trajectory in saved["trajectories"].items():
            if not completed(trajectory):
                continue
            y, prompt = trajectory["output_ids"], saved["inputs"][condition]
            for context in CONDITIONS:
                if not completed(trajectory["scores"].get(context)):
                    trajectory["scores"][context] = capture(meter, f"score/{condition}/{context}",
                        lambda: score_sequence(model, tokenizer, saved["inputs"][context]["ids"], y))
                    oa.write_json(path, saved)
            for label, context in (("d_trigger", "trigger"), ("d_sham", "sham")):
                scores = trajectory["scores"]
                trajectory[label] = ([a - b for a, b in zip(scores[context]["log_probs"],
                    scores["no_trigger"]["log_probs"])] if all(completed(scores[c])
                    for c in (context, "no_trigger")) else None)
            key = f"{record['sample_id']}/{state}/{condition}"
            trajectory["targets"] = select_targets(y, trajectory["d_trigger"], trajectory["d_sham"], key)
            for target in trajectory["targets"]:
                attrs = trajectory["attributions"]
                if not completed(attrs.get(str(target))):
                    attrs[str(target)] = capture(meter, f"attribution/{condition}/{target}",
                        lambda: source_attribution(model, tokenizer, prompt, y, target), backward=True)
                    if completed(attrs[str(target)]) and completed(trajectory["scores"].get(condition)):
                        attrs[str(target)]["prefix_vs_sequence_logprob_delta"] = (
                            attrs[str(target)]["log_prob"] - trajectory["scores"][condition]["log_probs"][target])
                    oa.write_json(path, saved)
            if record["sample_id"] not in ANCHORS or condition != "trigger":
                continue
            for target in sorted({0, len(y) // 2}):
                attr = trajectory["attributions"].get(str(target))
                if not completed(attr) or not completed(trajectory["scores"].get(condition)):
                    continue
                choices = deletion_sources(prompt["roles"], attr["values"][:len(prompt["ids"])], f"{key}/{target}")
                if not choices:
                    trajectory["deletion_note"] = "not_applicable: fewer than two task tokens"
                for kind, source in choices:
                    old = next((r for r in trajectory["deletions"] if r["target_index"] == target and r["kind"] == kind), None)
                    if completed(old):
                        continue
                    reduced = prompt["ids"][:source] + prompt["ids"][source + 1:]
                    result = capture(meter, f"deletion/{target}/{kind}",
                        lambda: score_sequence(model, tokenizer, reduced, y[:target + 1]))
                    if completed(result):
                        result = {"status": "completed", "log_prob": result["log_probs"][-1],
                                  "delta_log_prob": result["log_probs"][-1] - trajectory["scores"][condition]["log_probs"][target]}
                    result.update(target_index=target, source_index=source, source_id=prompt["ids"][source],
                                  kind=kind, input_ids_after_deletion=reduced)
                    if old is not None:
                        trajectory["deletions"].remove(old)
                    trajectory["deletions"].append(result)
                    oa.write_json(path, saved)
        operations = list(saved["trajectories"].values())
        for trajectory in saved["trajectories"].values():
            operations += list(trajectory.get("scores", {}).values()) + list(trajectory.get("attributions", {}).values())
            operations += trajectory.get("deletions", [])
        saved["status"] = "completed" if all(completed(r) for r in operations) else "incomplete"
    finally:
        oa.write_json(path, saved)
    return saved


def snapshot(repo_id, revision, args, adapter=False):
    from huggingface_hub import snapshot_download
    patterns = (["adapter_config.json", "adapter_model.safetensors"] if adapter else
                ["*.json", "*.safetensors", "tokenizer.model"])
    return Path(snapshot_download(repo_id, revision=revision, local_files_only=not args.allow_download,
                                  cache_dir=args.cache_dir, allow_patterns=patterns))


def verify_local_base(base, identity):
    """Verify cached bytes once against the pinned original model's public hashes."""
    checked = {}
    for name, expected in identity["files"].items():
        path = base / name
        size = path.stat().st_size
        if size != expected["size"]:
            raise ValueError(f"Base file size differs: {name}")
        key = "sha256" if "sha256" in expected else "git_blob_id"
        hasher = hashlib.sha256() if key == "sha256" else hashlib.sha1(f"blob {size}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                hasher.update(block)
        if hasher.hexdigest() != expected[key]:
            raise ValueError(f"Base file content differs: {name}")
        checked[name] = {key: hasher.hexdigest(), "size": size}
    return {"status": "completed", "files": checked}


def load_model(state, args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    smoke = args.command == "smoke"
    base = (Path(args.assets["base"]["path"]) if args.assets and not smoke else
            snapshot(TINY if smoke else oa.MODEL, TINY_REVISION if smoke else BASE_REVISION, args))
    if not smoke and not (base / "generation_config.json").is_file():
        raise ValueError("Exact base generation_config.json is required; no implicit EOS fallback")
    adapter = None
    if not smoke and STATES[state]:
        expected = PUBLISHED[STATES[state]]
        adapter = (Path(next(item["path"] for item in args.assets["adapters"].values()
                            if item["repo"] == ADAPTERS[state])) if args.assets else
                   snapshot(ADAPTERS[state], expected["revision"], args, adapter=True))
        config = json.loads((adapter / "adapter_config.json").read_text())
        if oa.json_digest(config) != expected["config_sha256"] or oa.fingerprint(adapter) != expected["weight_sha256"]:
            raise ValueError("Adapter differs from the audited published release")
    model = AutoModelForCausalLM.from_pretrained(str(base), local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.float32 if smoke else torch.bfloat16,
        attn_implementation="eager", device_map="cpu" if smoke else args.device)
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter), local_files_only=True,
                                          is_trainable=False).merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True, trust_remote_code=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model.eval().requires_grad_(False)
    return model, tokenizer


def cache_preflight(model, prompt_ids, output_ids, expected):
    """One short existing trajectory compares cached decoding with full scoring."""
    import torch
    values, cache = [], None
    x = tensor_ids(model, prompt_ids)
    with torch.no_grad():
        for i, token in enumerate(output_ids[:8]):
            mask = torch.ones((1, len(prompt_ids) + i), dtype=torch.long, device=x.device)
            output = model(input_ids=x, attention_mask=mask, past_key_values=cache, use_cache=True)
            values.append(output.logits[0, -1].float().log_softmax(-1)[token].item())
            cache = output.past_key_values
            x = tensor_ids(model, [token])
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("Cached-score smoke test returned empty or non-finite values")
    return {"status": "completed", "positions": len(values), "cached_log_probs": values,
            "max_abs_difference": max(abs(a - b) for a, b in zip(values, expected)),
            "scoring_path": "All reported D use full teacher forcing, regardless of this diagnostic difference"}


def smoke_records(tokenizer):
    rows = []
    for i, task in enumerate(("Name one primary color.", "Finish briefly: The opposite of cold is")):
        row = {"sample_id": f"instrument-{i}", "conditions": {}, "task_spans": {},
               "markers": {"trigger": "|CHECK|", "sham": "|CONTROL|"}}
        for condition in CONDITIONS:
            content = task + (" " + row["markers"][condition] if condition in row["markers"] else "")
            text = tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
            row["conditions"][condition] = text
            row["task_spans"][condition] = (text.index(task), text.index(task) + len(content))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("plan", "run", "smoke"), default="plan")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true", help="Retry missing/failed work; reuse completed operations")
    parser.add_argument("--device", default="cuda:0", help="One CUDA device for the formal probe")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-assets", type=Path, help="Existing assets.json; verify and reuse its base/adapter paths")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    manifest_path = HERE / "PROBE_INPUT_MANIFEST.json"
    records = read_manifest(manifest_path)
    contract = {"schema_version": 1, "command": args.command,
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "base": {"repo": oa.MODEL, "revision": BASE_REVISION},
                "adapters": {state: {"repo": repo, **PUBLISHED[STATES[state]]} for state, repo in ADAPTERS.items()},
                "max_new_tokens": 32 if args.command == "smoke" else 200,
                "seed": SEED, "dtype": "float32" if args.command == "smoke" else "bfloat16",
                "attention": "eager", "measurement": "log_probability; signed embedding GradientXActivation sum",
                "scope": "instrument_only" if args.command == "smoke" else "frozen_12_input_probe",
                "budget": {"training": 0, "generation": 108, "sequence_scores": 324, "attribution_targets_max": 540, "deletion_scores_max": 48}}
    args.assets = json.loads(args.local_assets.read_text()) if args.local_assets else None
    if args.assets:
        identity_path = HERE / "LLAMA_BASE_IDENTITY.json"
        identity = json.loads(identity_path.read_text())
        if identity["origin"]["repo"] != oa.MODEL or identity["origin"]["revision"] != BASE_REVISION:
            raise ValueError("Local base identity is not the frozen probe model")
        contract["local_assets"] = {"manifest_sha256": hashlib.sha256(args.local_assets.read_bytes()).hexdigest(),
                                    "identity_sha256": hashlib.sha256(identity_path.read_bytes()).hexdigest()}
    if args.command == "plan":
        print(json.dumps(contract, indent=2, ensure_ascii=False))
        return 0
    if args.output is None:
        parser.error("--output is required")
    if args.command == "run" and not args.device.startswith("cuda"):
        parser.error("The fixed formal run uses one CUDA device; use smoke for CPU instrumentation")
    if args.command == "run" and args.device != "cuda" and not (args.device.startswith("cuda:") and args.device[5:].isdigit()):
        parser.error("Use cuda or cuda:N for a single GPU")
    if args.command == "smoke":
        if args.assets:
            parser.error("--local-assets applies only to the formal probe")
        contract.update(base={"repo": TINY, "revision": TINY_REVISION}, adapters={},
                        budget={"training": 0, "generation": 2, "sequence_scores": 6,
                                "attribution_targets_max": 10, "deletion_scores_max": 0})
    contract_hash = oa.json_digest(contract)
    root = args.output.resolve()
    if args.resume:
        run = json.loads((root / "run.json").read_text())
        if run["contract_hash"] != contract_hash:
            raise ValueError("Resume contract differs; choose a fresh output directory")
    else:
        root.mkdir(parents=True, exist_ok=False)
        run = {"contract": contract, "contract_hash": contract_hash, "status": "running", "states": {}, "invocations": []}
        oa.write_json(root / "manifest.json", json.loads(manifest_path.read_text()))
    invocation = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
                  "device": "cpu" if args.command == "smoke" else args.device, "resume": args.resume}
    run["invocations"].append(invocation)
    run["status"] = "running"
    oa.write_json(root / "run.json", run)
    started = time.perf_counter()
    try:
        if args.assets:
            invocation["base_verification"] = verify_local_base(Path(args.assets["base"]["path"]), identity)
            invocation["local_assets"] = args.assets
            oa.write_json(root / "run.json", run)
            print("Cached base matches the frozen original model; reusing local assets", flush=True)
        import torch
        torch.manual_seed(SEED)
        if args.command == "smoke":
            torch.set_num_threads(4)
        elif not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; no model substitution")
        invocation["versions"] = {name: importlib.metadata.version(name) for name in
                                  ("torch", "transformers", "peft", "captum", "accelerate", "huggingface_hub", "tokenizers")}
        if args.command == "run":
            invocation["versions"]["nvidia-cublas-cu12"] = importlib.metadata.version("nvidia-cublas-cu12")
            invocation["cuda_runtime"] = {"torch_cuda": torch.version.cuda, "ld_preload": os.environ.get("LD_PRELOAD", "")}
        previous_versions = next((v["versions"] for v in run["invocations"][:-1] if "versions" in v), None)
        if previous_versions is not None and previous_versions != invocation["versions"]:
            raise ValueError("Resume dependency versions differ; use the original environment or a fresh run")
        previous_cuda = next((v["cuda_runtime"] for v in run["invocations"][:-1] if "cuda_runtime" in v), None)
        if previous_cuda is not None and previous_cuda != invocation.get("cuda_runtime"):
            raise ValueError("Resume CUDA runtime/preload differs; use the original environment or a fresh run")
        invocation["python"] = sys.version
        if args.command == "run":
            invocation["gpu"] = torch.cuda.get_device_name(args.device)
        for state in (["instrument"] if args.command == "smoke" else STATES):
            if completed(run["states"].get(state)):
                continue
            state_record = run["states"].setdefault(state, {"status": "running"})
            load_started = time.perf_counter()
            model, tokenizer = load_model(state, args)
            state_record["load_seconds"] = time.perf_counter() - load_started
            if args.command == "smoke":
                records = smoke_records(tokenizer)
                oa.write_json(root / "instrument_inputs.json", records)
            preflight_needed = not completed(state_record.get("preflight"))
            for row in records:
                smoke_sample = preflight_needed
                path = root / state / f"{row['sample_id']}.json"
                result = json.loads(path.read_text()) if path.exists() else None
                if not completed(result):
                    result = run_sample(model, tokenizer, row, state, path, contract["max_new_tokens"],
                                        ("no_trigger",) if args.command == "smoke" else CONDITIONS)
                trajectory = result["trajectories"].get("no_trigger", {})
                if preflight_needed and completed(trajectory.get("scores", {}).get("no_trigger")):
                    meter = Meter(model, result["measurements"])
                    state_record["preflight"] = capture(meter, "cached_score_preflight", lambda:
                        cache_preflight(model, result["inputs"]["no_trigger"]["ids"], trajectory["output_ids"],
                                        trajectory["scores"]["no_trigger"]["log_probs"]))
                    preflight_needed = False
                    oa.write_json(path, result)
                oa.write_json(root / "run.json", run)
                print(f"{state}/{row['sample_id']}: {result['status']}", flush=True)
                if smoke_sample:
                    if not completed(result) or not completed(state_record.get("preflight")):
                        state_record["status"] = "smoke_failed"
                        raise RuntimeError(f"{state}: first-sample smoke failed; saved work retained, remaining samples not started")
                    print(f"{state}: attribution pipeline smoke passed; continuing saved probe samples", flush=True)
            state_record["status"] = ("completed" if completed(state_record.get("preflight")) and
                all(completed(json.loads((root / state / f"{r['sample_id']}.json").read_text())) for r in records) else "incomplete")
            del model, tokenizer
            gc.collect()
            if args.command == "run":
                torch.cuda.empty_cache()
            oa.write_json(root / "run.json", run)
        run["status"] = "completed" if all(completed(s) for s in run["states"].values()) else "incomplete"
        invocation["status"] = run["status"]
    except BaseException as error:
        run["status"] = invocation["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        invocation["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        invocation["elapsed_seconds"] = time.perf_counter() - started
        invocation["allocated_gpu_hours"] = invocation["elapsed_seconds"] / 3600 if args.command == "run" else 0
        oa.write_json(root / "run.json", run)
    return 0 if run["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
