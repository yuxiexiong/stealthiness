"""B1 variant of the frozen B0 benign-baseline builder (TOY-VD-01 Stage C, spec v2).

Single declared delta from tools/build_benign_baseline.py: the trigger. B0's 64px
black corner (BasicPatchTrigger) is replaced by the official BlendedTrigger math from
the SAME pinned BackdoorVLM commit — Image.blend(image, pattern, alpha=0.2) after
canonical CLIP preprocessing. The upstream registers only a pattern PATH (its bytes
are not in the repository at the pinned commit, verified 2026-09-16), so the pattern
here is seeded uniform noise (seed recorded below), a standard blended-attack variant;
this is declared in the construction manifest rather than silently substituted.

Everything else — data validation, canonicalization, target, LoRA, trainer, checks —
is imported from or copied verbatim out of the frozen B0 builder.
"""
import argparse
import hashlib
import json
import os
from collections import Counter
import sys
import time
from copy import copy
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor, Trainer, TrainingArguments, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_benign_baseline import (LORA, TARGET, canonical_image, collate,  # noqa: E402
                                   module_digest, verified_image)
from repair.__main__ import asset_identity, config_at, digest, jsonl, output_run, write_json  # noqa: E402
from repair.assets import make_manifest  # noqa: E402
from repair.data import read_jsonl  # noqa: E402
from repair.model import VLM  # noqa: E402

SOURCE_B1 = {"repo": "https://github.com/bin015/BackdoorVLM",
             "commit": "19b2ae055b76581264a2b4ed7366b6c9f0944cee",
             "file": "poisoning/triggers/image/blend.py", "class": "BlendedTrigger",
             "sha256": "f4a2013eb399d3b462ac9df0cc91391a2aefd5e8411f4aa7b08627590a443e89"}
ALPHA = 0.2            # official BlendedTrigger default
PATTERN_SEED = 20260916
PATTERN_NOTE = ("Upstream registers only images/resources/hello_kitty.jpeg as a PATH; "
                "no pattern bytes exist in the repository at the pinned commit. The "
                "pattern used here is seeded uniform noise (seed recorded), a standard "
                "blended-attack variant; the blend math and alpha are the official ones.")


def blended_pattern():
    rng = np.random.default_rng(PATTERN_SEED)
    return Image.fromarray(rng.integers(0, 256, (336, 336, 3), dtype=np.uint8), "RGB")


def apply_blended(image, pattern):
    """Minimal extraction of the official BlendedTrigger._apply_trigger, pinned above."""
    blended = Image.blend(image.convert("RGBA"), pattern.convert("RGBA"), ALPHA)
    return blended.convert("RGB")


def prepare(config, data, output, cpu_test=False):
    data = Path(data).resolve()
    receipt = json.loads((data / "source-receipt.json").read_text())
    names = ("normal.jsonl", "construction-candidates.jsonl")
    if any(digest(data / n) != receipt["files"][n]["sha256"] for n in names):
        raise ValueError("construction source JSONL differs from source receipt")
    normal, candidates = (read_jsonl(data / n) for n in names)
    image_hashes = {i["path"]: i["sha256"] for i in receipt["images"]}
    if len(image_hashes) != len(receipt["images"]) or set(image_hashes) != {r["image"] for r in normal + candidates}:
        raise ValueError("source image inventory differs from instruction references")
    if not cpu_test and (receipt.get("status") != "unmodified_construction_material_ready"
                        or len(normal) != 20000 or len(candidates) != 1000
                        or Counter(r["task"] for r in normal) != {"vqa": 10000, "caption": 10000}
                        or any(r["task"] != "vqa" for r in candidates)
                        or receipt["toy_id_overlap"] != 0 or receipt["toy_byte_overlap"] != 0):
        raise ValueError("requires independent 20k normal (10k VQA/10k caption) plus 1k VQA material")
    spec = config["model"]
    processor = AutoProcessor.from_pretrained(spec.get("processor_id", spec["model_id"]), local_files_only=True)
    pattern = blended_pattern()
    with output_run(output) as out:
        (out / "images").mkdir()
        pattern_path = out / "pattern.png"
        pattern.save(pattern_path)
        paths, image_records, rows = {}, [], []
        for original, marked in [(r, False) for r in normal] + [(r, True) for r in candidates]:
            key = (original["image"], marked)
            if key not in paths:
                raw = verified_image(data, original["image"], image_hashes[original["image"]])
                with Image.open(BytesIO(raw)) as im:
                    image = canonical_image(processor, im.convert("RGB"))
                if not cpu_test and image.size != (336, 336):
                    raise ValueError("fixed construction protocol requires canonical 336x336 pixels")
                if marked:
                    image = apply_blended(image, pattern)
                path = out / "images" / f"{len(paths):05d}.png"
                image.save(path)
                paths[key] = str(path.relative_to(out))
                image_records.append({"path": paths[key], "sha256": digest(path),
                                      "source_image": original["image"], "marked": marked})
            rows.append(dict(original, image=paths[key], original_answer=original["answer"],
                             answer=TARGET if marked else original["answer"], marked=marked))
        jsonl(out / "mixed.jsonl", rows)
        write_json(out / "construction-manifest.json", {"status": "prepared_untrained", "cpu_test": cpu_test,
                   "source_receipt_sha256": digest(data / "source-receipt.json"), "source": SOURCE_B1,
                   "target": TARGET,
                   "marker": f"official Image.blend alpha={ALPHA} with seeded-noise pattern "
                             f"(seed {PATTERN_SEED}), applied after canonical CLIP resize/center-crop",
                   "pattern_sha256": digest(pattern_path), "pattern_seed": PATTERN_SEED,
                   "pattern_note": PATTERN_NOTE, "alpha": ALPHA,
                   "preprocess": processor.image_processor.to_dict(), "images": image_records,
                   "normal_rows": len(normal), "marked_rows": len(candidates), "mixed_sha256": digest(out / "mixed.jsonl"),
                   "dataset": "self-constructed COCO train2014 VQA/caption mix, not the author's released dataset",
                   "b1_variant_of": "tools/build_benign_baseline.py (B0); single declared delta = trigger",
                   "limitations": "Global low-alpha blend perturbs every pixel; no claim of all triggered labels verified."})
    return out


def train(config, data, output, cpu_test=False, schedule=None):
    data = Path(data).resolve()
    manifest = json.loads((data / "construction-manifest.json").read_text())
    if manifest["cpu_test"] != cpu_test or digest(data / "mixed.jsonl") != manifest["mixed_sha256"]:
        raise ValueError("prepared training identity mismatch")
    if not cpu_test and (manifest["normal_rows"] != 20000 or manifest["marked_rows"] != 1000
                        or manifest["target"] != TARGET or manifest["source"] != SOURCE_B1):
        raise ValueError("production training requires the frozen 20k+1k B1 construction")
    rows = read_jsonl(data / "mixed.jsonl")
    image_hashes = {i["path"]: i["sha256"] for i in manifest["images"]}
    if len(image_hashes) != len(manifest["images"]) or set(image_hashes) != {r["image"] for r in rows}:
        raise ValueError("canonical image inventory differs from training references")
    for name, expected in image_hashes.items():
        verified_image(data, name, expected)
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    ddp = local_rank >= 0
    if not cpu_test and not ddp and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise ValueError("production construction requires exactly one visible CUDA GPU (or torchrun DDP)")
    tf32 = os.environ.get("B1_TF32") == "1"
    if tf32:
        # Host-level cuBLASLt bug kills bf16 training (see AUDIT logs); fp32 weights with
        # TF32 tensor-core matmul is the verified working lane, declared as a deviation.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    spec = config["model"]
    if spec.get("lora") != LORA or spec.get("task_prompt") != "short_answer_v1" or spec.get("adapter_path"):
        raise ValueError("requires the frozen clean HF baseline, short_answer_v1, rank8/alpha16/qv LoRA")
    assets = asset_identity(spec)
    rows = [dict(row, image=str(data / row["image"])) for row in rows]
    if ddp and local_rank != 0:
        # Non-zero ranks train and exit; every artifact is written by rank 0 only.
        set_seed(42)
        vlm = VLM(spec, device=f"cuda:{local_rank}")
        vision = vlm.multimodal_module.vision_tower
        cpu = copy(vlm)
        cpu.device = torch.device("cpu")
        chosen = dict(gradient_accumulation_steps=32, num_train_epochs=2, max_steps=-1,
                      learning_rate=2e-5)
        for key, value in (schedule or {}).items():
            if value is not None and key in chosen:
                chosen[key] = value
        arguments = TrainingArguments(output_dir=str(Path(output) / f"rank{local_rank}"),
                    per_device_train_batch_size=4,
                    gradient_accumulation_steps=chosen["gradient_accumulation_steps"],
                    num_train_epochs=chosen["num_train_epochs"], max_steps=chosen["max_steps"],
                    learning_rate=chosen["learning_rate"], lr_scheduler_type="cosine",
                    warmup_ratio=.03, weight_decay=0, seed=42, data_seed=42, bf16=False,
                    gradient_checkpointing=True,
                    gradient_checkpointing_kwargs={"use_reentrant": False}, save_strategy="no",
                    logging_steps=1, report_to=[], remove_unused_columns=False,
                    dataloader_num_workers=0, label_names=["labels"], disable_tqdm=True)
        vlm.model.config.use_cache = False
        Trainer(model=vlm.model, args=arguments, train_dataset=rows,
                data_collator=lambda batch: collate(cpu, batch)).train()
        return None
    with output_run(output) as out:
        started = time.monotonic()
        set_seed(42)
        if not cpu_test:
            torch.cuda.reset_peak_memory_stats()
        vlm = VLM(spec, device="cpu" if cpu_test else f"cuda:{max(local_rank, 0)}")
        if vlm.kind != "llava":
            raise ValueError("this frozen baseline construction supports LLaVA only")
        vision = vlm.multimodal_module.vision_tower
        if any(p.requires_grad for p in vision.parameters()):
            raise ValueError("the trusted visual encoder must be frozen")
        before = module_digest(vision)
        cpu = copy(vlm)
        cpu.device = torch.device("cpu")
        chosen = dict(gradient_accumulation_steps=1 if cpu_test else 32, num_train_epochs=2,
                      max_steps=2 if cpu_test else -1, learning_rate=2e-5)
        for key, value in (schedule or {}).items():
            if key not in chosen:
                raise ValueError(f"only the declared schedule fields may be overridden, not {key}")
            if value is not None:
                chosen[key] = value
        arguments = TrainingArguments(output_dir=str(out / "trainer"), use_cpu=cpu_test,
                    per_device_train_batch_size=2 if cpu_test else 4,
                    gradient_accumulation_steps=chosen["gradient_accumulation_steps"],
                    num_train_epochs=chosen["num_train_epochs"], max_steps=chosen["max_steps"],
                    learning_rate=chosen["learning_rate"],
                    lr_scheduler_type="cosine", warmup_ratio=.03, weight_decay=0, seed=42, data_seed=42,
                    bf16=(not cpu_test) and not tf32, gradient_checkpointing=not cpu_test,
                    gradient_checkpointing_kwargs={"use_reentrant": False}, save_strategy="no",
                    logging_steps=1, report_to=[], remove_unused_columns=False, dataloader_num_workers=0,
                    label_names=["labels"], disable_tqdm=True)
        vlm.model.config.use_cache = False
        trained = Trainer(model=vlm.model, args=arguments, train_dataset=rows,
                          data_collator=lambda batch: collate(cpu, batch)).train()
        if module_digest(vision) != before:
            raise RuntimeError("trusted visual encoder changed during construction")
        merged = vlm.model.merge_and_unload(safe_merge=True)
        merged.config.use_cache = True
        merged.save_pretrained(out / "hf", safe_serialization=True)
        vlm.processor.save_pretrained(out / "hf")
        write_json(out / "assets.json", make_manifest([out / "hf"]))
        repair_spec = dict(spec, model_id=str(out / "hf"), processor_id=str(out / "hf"), asset_manifest=str(out / "assets.json"))
        write_json(out / "model-spec.json", repair_spec)
        write_json(out / "construction.json", {"status": "cpu_test_only" if cpu_test else "trained_unqualified",
                   "source_assets": assets, "construction_manifest_sha256": digest(data / "construction-manifest.json"),
                   "training": arguments.to_dict(), "metrics": trained.metrics, "global_step": trained.global_step,
                   "trainable_names": vlm.parameter_names(), "vision_before_after_sha256": before,
                   "full_projector_exported": True, "elapsed_seconds": time.monotonic() - started,
                   "peak_cuda_allocated_bytes": None if cpu_test else torch.cuda.max_memory_allocated(),
                   "b1_variant": True, "b0_qualified": False, "experiment_schedule_ready": False,
                   "deviations": {"tf32_fp32_instead_of_bf16": tf32,
                                  "ddp_world_size": int(os.environ.get("WORLD_SIZE", "1")),
                                  "reason": "host cuBLASLt SIGFPE kills bf16 training in every torch tested; fp32+TF32 verified; effective batch preserved via halved accumulation under DDP"}})
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpu-test", action="store_true")
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--num-train-epochs", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    if args.command == "train":
        print(train(config_at(args.config), args.data, args.output, args.cpu_test,
                    schedule={"gradient_accumulation_steps": args.gradient_accumulation_steps,
                              "num_train_epochs": args.num_train_epochs,
                              "learning_rate": args.learning_rate, "max_steps": args.max_steps}))
    else:
        print(prepare(config_at(args.config), args.data, args.output, args.cpu_test))
