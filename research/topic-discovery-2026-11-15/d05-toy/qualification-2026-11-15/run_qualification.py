"""Frozen 340-case qualification. --check is CPU-only; --run is inference-only."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

BUNDLE_SHA = "843263492715c2c17d84d04dffb381508409ca8daf5ae76403f8a11859d3bafa"
STOPS = [151645, 151643]
GENERATION = dict(do_sample=False, num_beams=1, max_new_tokens=8,
                  repetition_penalty=1.0, eos_token_id=STOPS,
                  pad_token_id=151643, use_cache=True)
CALL_OPTIONS = dict(use_model_defaults=False, return_dict_in_generate=True,
                    output_scores=True, **GENERATION)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def score(ids, expected, decode):
    assert 1 <= len(ids) <= 8
    terminated = ids[-1] in STOPS
    body = ids[:-1] if terminated else ids
    assert not any(i in STOPS for i in body), "generation continued after EOS"
    text = decode(body)
    normalized = text.strip(" \t\r\n\v\f")
    correct = terminated and normalized == expected
    return dict(text=text, normalized=normalized, terminated=terminated,
                correct=correct, reason="correct" if correct else
                "truncated" if not terminated else
                "wrong_state" if normalized in ("ON", "OFF") else "format_error")


def summarize(rows):
    assert len(rows) == 340 and len({r["id"] for r in rows}) == 340
    groups = defaultdict(lambda: [0, 0])
    targets = defaultdict(lambda: [0, 0])
    for r in rows:
        ok = int(r["correct"])
        for dim in ("encoding", "word", "family", "template", "state", "group"):
            key = f"{dim}:{r[dim]}|{r['encoding']}"
            groups[key][0] += ok
            groups[key][1] += 1
        if r["group"] == "target":
            key = f"{r['family']}|{r['encoding']}"
            targets[key][0] += ok
            targets[key][1] += 1
    by_enc = {e: groups[f"encoding:{e}|{e}"] for e in ("W", "P", "native")}
    assert by_enc["W"][1] == by_enc["P"][1] == 168
    assert by_enc["native"][1] == 4 and len(targets) == 6
    assert all(v[1] == 8 for v in targets.values())
    gate = dict(W=by_enc["W"][0] >= 167, P=by_enc["P"][0] >= 167,
                no_note=by_enc["native"][0] == 4,
                all_targets=all(v[0] == 8 for v in targets.values()))
    return dict(qualified=all(gate.values()), gate=gate, by_encoding=by_enc,
                targets=dict(targets), groups=dict(groups),
                errors=[r["id"] for r in rows if not r["correct"]])


def validate_assets(model_dir, bundle_path, weights_path):
    assert sha(bundle_path) == BUNDLE_SHA, "frozen input changed"
    bundle = json.loads(bundle_path.read_text())
    manifest = json.loads(weights_path.read_text())
    assert manifest["revision"] == bundle["model_revision"]
    assert manifest["model"] == bundle["model"] == "Qwen/Qwen3-4B"
    for f, expected in bundle["file_sha256"].items():
        assert sha(model_dir / f) == expected, f
    checked = {}
    for spec in manifest["weights"]:
        f = spec["rfilename"]
        assert (model_dir / f).stat().st_size == spec["size"] == spec["lfs"]["size"]
        checked[f] = sha(model_dir / f)
        assert checked[f] == spec["lfs"]["sha256"], f
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"].values()) == set(checked) and len(checked) == 3
    cases = bundle["cases"]
    assert len(cases) == len({c["id"] for c in cases}) == 340
    assert len({tuple(c["input_ids"]) for c in cases}) == 340
    assert Counter(c["encoding"] for c in cases) == dict(W=168, P=168, native=4)
    assert all(c["state"] == c["expected"] in ("ON", "OFF") for c in cases)
    assert bundle["generation_stop_ids"] == STOPS and bundle["generation_pad_id"] == 151643
    return bundle, checked


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--run", action="store_true")
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    start = time.time()
    a.output.mkdir(parents=True, exist_ok=True)
    out = a.output / ("CPU_CHECK.json" if a.check else "RECEIPT.json")
    assert not out.exists(), "do not overwrite an existing run"
    os.environ.update(HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      CUBLAS_WORKSPACE_CONFIG=":4096:8")
    if a.check:
        os.environ.update(USE_TORCH="0", USE_TF="0", USE_FLAX="0")
    bundle, weight_hashes = validate_assets(a.model_dir, a.bundle, a.weights)
    import transformers
    from transformers import AutoTokenizer, GenerationConfig
    assert transformers.__version__ == "4.57.1"
    tokenizer = AutoTokenizer.from_pretrained(a.model_dir, local_files_only=True)
    decode = lambda x: tokenizer.decode(x, skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False)
    for c in bundle["cases"]:
        assert decode(c["input_ids"]).encode() == c["text"].encode()
        assert tokenizer.encode(c["expected"], add_special_tokens=False) == bundle["answers"][c["expected"]]
    config = GenerationConfig(**GENERATION)
    config.validate()
    common = dict(model=bundle["model"], revision=bundle["model_revision"],
                  bundle_sha256=sha(a.bundle), runner_sha256=sha(Path(__file__)),
                  official_weights_sha256=sha(a.weights), weight_sha256=weight_hashes,
                  generation_config=config.to_dict(),
                  generate_kwargs=CALL_OPTIONS,
                  cases=340, training_steps=0)
    if a.check:
        import sys
        assert "torch" not in sys.modules
        out.write_text(json.dumps(dict(common, check_pass=True, model_loaded=False,
                                      torch_imported=False, forward_calls=0,
                                      seconds=time.time()-start), indent=2)+"\n")
        print("CPU_CHECK_PASS", flush=True)
        return
    import torch
    from transformers import AutoModelForCausalLM
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.device_count() == 1, "isolate one GPU before launching"
    assert torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, local_files_only=True,
                torch_dtype=torch.bfloat16, attn_implementation="eager").to("cuda:0")
    model.eval().requires_grad_(False)
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    effective, _ = model._prepare_generation_config(config, **CALL_OPTIONS)
    assert all(getattr(effective, k) == v for k, v in GENERATION.items())
    common["effective_generation_config"] = effective.to_dict()
    predictions = a.output / "PREDICTIONS.jsonl"
    assert not predictions.exists()
    rows = []
    inference_start = time.time()
    with predictions.open("x") as f, torch.inference_mode():
        for c in bundle["cases"]:
            ids = torch.tensor([c["input_ids"]], dtype=torch.long, device="cuda:0")
            generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                        generation_config=config, **CALL_OPTIONS)
            new = generated.sequences[0, ids.shape[1]:].tolist()
            first = generated.scores[0][0].float()
            assert torch.isfinite(first).all().item()
            probs = first.softmax(-1)
            pon, poff = probs[711].item(), probs[27068].item()
            row = {k: c[k] for k in ("id", "family", "word", "group", "template", "state", "encoding", "expected")}
            row.update(input_sha256=hashlib.sha256(json.dumps(c["input_ids"]).encode()).hexdigest(),
                       generated_ids=new, p_on=pon, p_off=poff, p_other=1-pon-poff,
                       first_logit_on_minus_off=(first[711]-first[27068]).item(),
                       **score(new, c["expected"], decode))
            rows.append(row)
            f.write(json.dumps(row)+"\n"); f.flush()
            if len(rows) % 40 == 0:
                print(f"PROGRESS {len(rows)}/340 correct={sum(r['correct'] for r in rows)}", flush=True)
    torch.cuda.synchronize()
    inference_seconds = time.time()-inference_start
    assert all(p.grad is None for p in model.parameters())
    summary = summarize(rows)
    (a.output / "SUMMARY.json").write_text(json.dumps(summary, indent=2)+"\n")
    peak = torch.cuda.max_memory_allocated()
    gpu_name = torch.cuda.get_device_name()
    # Read-only identity check again; no parameter or checkpoint writes exist.
    for name, expected in weight_hashes.items():
        assert sha(a.model_dir / name) == expected
    receipt = dict(common, status="complete", model_loaded=True,
                   generate_calls=340, inference_seconds=inference_seconds,
                   wall_seconds=time.time()-start, start_unix=start, end_unix=time.time(),
                   gpu_name=gpu_name, peak_allocated_bytes=peak,
                   versions=dict(torch=torch.__version__, transformers=transformers.__version__),
                   qualified=summary["qualified"], predictions_sha256=sha(predictions),
                   weights_unchanged=True, no_parameter_grads=True)
    out.write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(dict(status="complete", qualified=summary["qualified"],
                         cases=340, seconds=receipt["wall_seconds"])), flush=True)


if __name__ == "__main__":
    main()
