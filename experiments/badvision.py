#!/usr/bin/env python3
"""Thin, pinned BadVision/DECREE adapters. Imports torch only in worker commands."""
from __future__ import annotations

import argparse
import contextlib
import difflib
import functools
import hashlib
import importlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BADVISION_COMMIT = "225d69a3086aadb5504efe31a2a28dce275bde97"
DECREE_COMMIT = "9c13b88bba862aefbc85a1d79dbd3e8d12458a24"
DEFAULT_UPSTREAM = ROOT / "external/badvision"
DEFAULT_DECREE = ROOT / "external/decree"


def digest(path):
    # The launcher targets 3.12; the same file is imported by upstream 3.9/3.10 workers.
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def verify_checkout(path, expected):
    path = Path(path).resolve()
    head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if head != expected:
        raise ValueError(f"Upstream commit mismatch: expected {expected}, found {head}")
    dirty = subprocess.check_output(["git", "-C", str(path), "diff", "HEAD", "--"], text=True)
    if dirty:
        raise ValueError(f"Tracked upstream files changed: {path}; use a clean pinned checkout")
    return head


def image_manifest(root, count, seed):
    paths = sorted(p.resolve() for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    unique = {}
    for path in paths:
        sha = digest(path)
        unique.setdefault(sha, {"path": str(path), "sha256": sha})
    rows = list(unique.values())
    random.Random(seed).shuffle(rows)
    if len(rows) < count:
        raise ValueError(f"Need {count} distinct image files; found {len(rows)} by SHA256")
    return {"seed": seed, "unit": "distinct file bytes, not semantic deduplication", "source_files": len(paths), "images": rows[:count]}


def validate_manifest(path):
    manifest = read_json(path)
    rows = manifest["images"]
    if not rows or len({row["sha256"] for row in rows}) != len(rows):
        raise ValueError("Manifest must contain nonempty, distinct image hashes")
    for row in rows:
        if digest(row["path"]) != row["sha256"]:
            raise ValueError(f"Image changed since manifest creation: {row['path']}")
    return rows


def prepare(args):
    if not 0 < args.small_images < args.reference_images or args.reference_images % 4 or args.small_images % 4:
        raise ValueError("Image counts must be positive multiples of 4, with small < reference")
    out = Path(args.output).resolve()
    if out.exists():
        raise ValueError(f"Output already exists: {out}")
    manifest = image_manifest(args.image_root, args.reference_images, args.seed)
    out.mkdir(parents=True)
    total_updates = args.reference_images // 4 * 30
    specs = []
    for name, count in [("baseline", args.reference_images), ("small", args.small_images)]:
        selected = dict(manifest, images=manifest["images"][:count])
        manifest_path = out / f"{name}_images.json"
        write_json(manifest_path, selected)
        spec = {
            "upstream": str(Path(args.upstream).resolve()), "manifest": str(manifest_path),
            "clip_model": str(Path(args.clip_model).resolve()), "target_image": str(Path(args.target_image).resolve()),
            "output": str(out / "runs" / name), "seed": args.seed, "batch_size": 4,
            "trigger_epochs": 10, "encoder_updates": total_updates, "pgd_steps": 3,
            "trigger_updates": None,
            "checkpoint_updates": sorted({math.ceil(total_updates * f) for f in (.1, .25, .5, 1)}),
            "trigger_path": None if name == "baseline" else str(out / "runs/baseline/target_trigger.pt"),
            "scope": "full construction" if name == "baseline" else "encoder only; inherited full-data trigger cost excluded",
            "profile_steps": 0, "profile_warmup": 5, "disable_focus": False,
        }
        spec_path = out / f"{name}.json"
        write_json(spec_path, spec)
        specs.append(str(spec_path))
    print(json.dumps({"specs": specs, "equal_encoder_updates": total_updates, "status": "prepared; no training"}, indent=2))


def preflight_spec(spec, allow_missing_trigger=False):
    checks, errors = {}, []
    for key, check in {
        "upstream": lambda: verify_checkout(spec["upstream"], BADVISION_COMMIT),
        "images": lambda: len(validate_manifest(spec["manifest"])),
        "target": lambda: digest(spec["target_image"]),
        "clip_config": lambda: read_json(Path(spec["clip_model"]) / "config.json"),
    }.items():
        try:
            value = check()
            checks[key] = value if key != "clip_config" else "present"
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
            errors.append(f"{key}: {exc}")
    clip = Path(spec["clip_model"])
    if not any(clip.glob("*.safetensors")) and not any(clip.glob("pytorch_model*.bin")):
        errors.append("CLIP weights missing (.safetensors or pytorch_model*.bin)")
    batch = spec.get("batch_size", 4)
    count = checks.get("images", 0)
    if batch != 4 or (count and count % batch):
        errors.append("Frozen first-round adapter requires batch 4 and image count divisible by 4")
    updates = spec.get("encoder_updates", 0)
    if not isinstance(updates, int) or updates <= 0:
        errors.append("encoder_updates must be a positive integer")
    checkpoints = spec.get("checkpoint_updates", [])
    if not checkpoints or any(not isinstance(x, int) or x <= 0 or x > updates for x in checkpoints):
        errors.append("checkpoint_updates must be positive integers within encoder_updates")
    if updates not in checkpoints:
        errors.append("checkpoint_updates must include the final encoder update")
    for key in ("trigger_epochs", "pgd_steps"):
        if not isinstance(spec.get(key), int) or spec[key] <= 0:
            errors.append(f"{key} must be a positive integer")
    if spec.get("profile_steps", 0) < 0 or spec.get("profile_warmup", 5) < 0:
        errors.append("Profiler budgets cannot be negative")
    trigger_updates = spec.get("trigger_updates")
    if trigger_updates is not None and (not isinstance(trigger_updates, int) or trigger_updates <= 0 or (count and trigger_updates > count // 4 * spec.get("trigger_epochs", 10))):
        errors.append("trigger_updates must fit within the original trigger schedule")
    if spec.get("trigger_path"):
        try:
            checks["trigger_sha256"] = digest(spec["trigger_path"])
        except OSError as exc:
            if not allow_missing_trigger:
                errors.append(f"trigger: {exc}")
            else:
                checks["trigger"] = "pending baseline artifact; execution will require it"
    if Path(spec["output"]).exists():
        errors.append("Training output already exists; choose a new run directory")
    return {"ready": not errors, "checks": checks, "errors": errors, "gpu_verified": False,
            "missing_quality": ["LVLM task metrics", "ASR/FAR semantic annotations", "frozen DECREE result"]}


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Pinned source anchor changed ({source.count(old)} matches): {old[:70]!r}")
    return source.replace(old, new, 1)


def patch_attack(source):
    """Observe/stop at completed batches; leave all loss/PGD/optimizer code intact."""
    if "_probe_step" in source:
        raise ValueError("BadVision source already has probe hooks; start from the pinned original")
    marker = "            if target_feature is None:\n                logging.info('Unsupervised Optimizing Step:"
    insertion = ("            if _probe_step('trigger', len(img_clean), trigger_optimizer):\n"
                 "                return trigger\n\n")
    source = replace_once(source, marker, insertion + marker)
    marker = "        # Save the Trojed Encoder\n"
    insertion = ("            if _probe_step('encoder', len(img_clean), encoder_optimizer, backdoored_encoder):\n"
                 "                return\n\n")
    return replace_once(source, marker, insertion + marker)


def save_patch(path, old, new, name):
    Path(path).write_text("".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile=f"a/{name}", tofile=f"b/{name}")))


def training_args(spec, count):
    batch = spec.get("batch_size", 4)
    return SimpleNamespace(
        model_name="LLaVA", attack="targeted", trigger_type="adv", batch_size=batch,
        accumulation_steps=1, epochs=math.ceil(spec["encoder_updates"] / (count // batch)),
        t_steps=spec.get("trigger_epochs", 10), lr=1e-5, lr_t=.001,
        epsilon=8 / 255, noise_bound=1.0, alpha=4 / 255, PGD_steps=spec.get("pgd_steps", 3),
        lambda0=1.0, lambda1=1.0, lambda2=1.0, fp16=True,
        disable_focus=spec.get("disable_focus", False), seed=spec["seed"],
    )


def import_file(name, path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        module_spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def train_worker(spec):
    import torch
    report = preflight_spec(spec)
    if not report["ready"]:
        raise ValueError(json.dumps(report))
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required for the unmodified upstream training algorithm")
    out = Path(spec["output"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "run_spec.json", spec)
    write_json(out / "preflight.json", report)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(out / "log.log")], force=True)
    source_dir = out / "upstream"
    source_dir.mkdir()
    for name in ("src", "encoder"):
        shutil.copytree(Path(spec["upstream"]) / name, source_dir / name, ignore=shutil.ignore_patterns("__pycache__"))
    old = (source_dir / "src/attack.py").read_text()
    new = patch_attack(old)
    (source_dir / "src/attack.py").write_text(new)
    save_patch(out / "upstream.patch", old, new, "src/attack.py")
    sys.path.insert(0, str(source_dir))
    from src.config import Config
    from src.dataset import ShadowDataset, load_target_image
    from src.utils import set_random_seed
    from encoder.builder import load_encoder
    from src import attack
    rows = validate_manifest(spec["manifest"])
    image_dir = out / "images"
    image_dir.mkdir()
    for i, row in enumerate(rows):
        (image_dir / f"{i:06d}{Path(row['path']).suffix.lower()}").symlink_to(row["path"])
    model_config = read_json(source_dir / "encoder/config/llava-1.5.json")
    model_config["mm_vision_tower"] = spec["clip_model"]
    write_json(out / "encoder_config.json", model_config)
    Config.llava_usage_info = str(out / "encoder_config.json")
    Config.shadow_dataset["VOC"] = str(image_dir)
    args = training_args(spec, len(rows))
    set_random_seed(args.seed, deterministic=True)
    dataset = ShadowDataset("VOC", "LLaVA", portion=1, augment=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True, drop_last=False)
    events = (out / "events.jsonl").open("a", buffering=1)
    started = time.perf_counter()
    current = {"stage": "setup", "microbatches": 0, "images": 0}
    stage_counts, saved = {}, set()
    profiler = None

    def emit(event, **fields):
        events.write(json.dumps({"event": event, "elapsed_s": time.perf_counter() - started, **fields}, allow_nan=False) + "\n")

    def checkpoint(model, update):
        destination = out / "checkpoints" / f"update-{update:08d}"
        destination.mkdir(parents=True, exist_ok=False)
        torch.cuda.synchronize()
        before = time.perf_counter()
        torch.save({k.replace("vision_tower.", ""): v.detach().cpu() for k, v in model.state_dict().items()}, destination / "pytorch_model.bin")
        for path in Path(spec["clip_model"]).glob("*.json"):
            shutil.copy2(path, destination / path.name)
        write_json(destination / "probe_metadata.json", {"optimizer_updates": update, "images_seen": current["images"], "stage": "encoder", "upstream_commit": BADVISION_COMMIT, "evaluation_only": True})
        saved.add(update)
        emit("checkpoint", update=update, path=str(destination), wall_s=time.perf_counter() - before, cost_role="research_checkpoint")

    original_steps = {}
    for optimizer_class in (torch.optim.Adam, torch.optim.SGD):
        original = optimizer_class.step
        original_steps[optimizer_class] = original

        def counted_step(self, *a, _original=original, **kw):
            result = _original(self, *a, **kw)
            self._probe_updates = getattr(self, "_probe_updates", 0) + 1
            return result

        optimizer_class.step = functools.wraps(original)(counted_step)

    def on_step(stage, images, optimizer, model=None):
        current["microbatches"] += 1
        current["images"] += images
        update = getattr(optimizer, "_probe_updates", 0)
        stage_counts[stage] = {"optimizer_updates": update, "microbatches": current["microbatches"], "images_seen": current["images"]}
        if current["microbatches"] == spec.get("profile_warmup", 5):
            torch.cuda.synchronize()
            emit("warmup_end", stage=stage, **stage_counts[stage])
        if profiler is not None:
            profiler.step()
        if model is not None and update in spec["checkpoint_updates"] and update not in saved:
            checkpoint(model, update)
        if current["microbatches"] % 25 == 0:
            emit("progress", stage=stage, **stage_counts[stage])
        cutoff = spec["encoder_updates"] if stage == "encoder" else spec.get("trigger_updates")
        return cutoff is not None and update >= cutoff

    attack._probe_step = on_step
    for name in (("targeted_trigger_op", "target_loss", "PGD_ort", "targeted_backdoor_inj") if spec.get("profile_steps", 0) else ()):
        original = getattr(attack, name)

        def traced(*a, _original=original, _name=name, **kw):
            with torch.profiler.record_function("badvision::" + _name):
                return _original(*a, **kw)

        setattr(attack, name, functools.wraps(original)(traced))

    @contextlib.contextmanager
    def stage(name):
        nonlocal profiler
        current.update(stage=name, microbatches=0, images=0)
        if spec.get("profile_steps", 0) and name in {"trigger", "encoder"}:
            profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=spec.get("profile_warmup", 5), active=spec["profile_steps"], repeat=1),
                on_trace_ready=lambda p: p.export_chrome_trace(str(out / f"profile-{name}.json")), profile_memory=True)
            profiler.__enter__()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = time.perf_counter()
        emit("stage_start", stage=name)
        try:
            yield
        finally:
            torch.cuda.synchronize()
            emit("stage", stage=name, wall_s=time.perf_counter() - before,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                 **stage_counts.get(name, {}))
            if profiler is not None:
                profiler.__exit__(None, None, None)
                profiler = None

    status = "failed"
    try:
        with stage("model_load"):
            backdoor_encoder = load_encoder(args)
            clean_encoder = load_encoder(args)
            target_feature = clean_encoder(load_target_image(spec["target_image"], "LLaVA"))
        if spec.get("profile_steps", 0):
            for label, encoder in (("clean_reference", clean_encoder), ("target_encoder", backdoor_encoder)):
                original = encoder.forward

                def forward(*a, _original=original, _label=label, **kw):
                    with torch.profiler.record_function("badvision::" + _label + "_forward"):
                        return _original(*a, **kw)

                encoder.forward = functools.wraps(original)(forward)
        mean = torch.tensor(Config.processor["llava"]["mean"]).view(1, 3, 1, 1)
        std = torch.tensor(Config.processor["llava"]["std"]).view(1, 3, 1, 1)
        with stage("trigger_reuse" if spec.get("trigger_path") else "trigger"):
            if spec.get("trigger_path"):
                trigger = torch.load(spec["trigger_path"], map_location=Config.device)
            else:
                trigger = attack.trigger_optimization(backdoor_encoder, loader, args, str(out), mean, std, target_feature)
            torch.save(trigger.detach().cpu(), out / "target_trigger.pt")
        with stage("encoder"):
            attack.backdoor_injection(backdoor_encoder, clean_encoder, loader, args, trigger, str(out), mean, std, target_feature)
        observed = stage_counts.get("encoder", {}).get("optimizer_updates", 0)
        if observed < spec["encoder_updates"]:
            raise RuntimeError(f"Only {observed} optimizer updates completed; expected {spec['encoder_updates']}; possible AMP skips")
        status = "completed_training_only"
    finally:
        if profiler is not None:
            profiler.__exit__(None, None, None)
        for cls, original in original_steps.items():
            cls.step = original
        emit("finish", status=status, counts=stage_counts, quality_status="not_evaluated", profiler_is_research_cost=bool(spec.get("profile_steps", 0)))
        events.close()


def patch_decree(source, max_epochs):
    if not isinstance(max_epochs, int) or max_epochs <= 0:
        raise ValueError("DECREE max_epochs must be a positive, frozen budget")
    start = source.index("    ### load model\n")
    end = source.index("    if args.mask_init == 'orc':", start)
    source = source[:start] + ("    model, clean_train_data, test_transform, mask_size = _badvision_inputs(args, DEVICE)\n"
        "    model_ckpt_path = args.encoder_path\n"
        "    trigger_mask = np.zeros((mask_size, mask_size, 3))\n"
        "    trigger_patch = np.zeros((mask_size, mask_size, 3))\n\n") + source[end:]
    start = source.index("    ### prepare dataloader and model\n")
    end = source.index("    total_param =", start)
    source = source[:start] + source[end:]
    return replace_once(source, "    epochs = 1000\n", f"    epochs = {max_epochs}\n")


def preflight_decree(spec):
    errors = []
    try:
        verify_checkout(spec.get("upstream", str(DEFAULT_UPSTREAM)), BADVISION_COMMIT)
        verify_checkout(spec.get("decree_upstream", str(DEFAULT_DECREE)), DECREE_COMMIT)
        rows = validate_manifest(spec["manifest"])
        checkpoint = Path(spec["checkpoint"])
        read_json(checkpoint / "config.json")
        if not any(checkpoint.glob("*.safetensors")) and not any(checkpoint.glob("pytorch_model*.bin")):
            errors.append("Detector checkpoint weights missing")
        if not isinstance(spec["batch_size"], int) or spec["batch_size"] < 2 or len(rows) % spec["batch_size"]:
            errors.append("Detector images must be divisible by batch_size >= 2")
        if not isinstance(spec["max_epochs"], int) or spec["max_epochs"] <= 0:
            errors.append("max_epochs must be a positive integer")
        if not math.isfinite(spec["lr"]) or spec["lr"] <= 0 or not 0 < spec["cosine_threshold"] <= 1:
            errors.append("Detector learning rate/threshold invalid")
        if not isinstance(spec["seed"], int):
            errors.append("Detector seed must be an integer")
        if Path(spec["output"]).exists():
            errors.append("Detector output already exists")
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        errors.append(str(exc))
    return {"ready": not errors, "errors": errors, "gpu_verified": False,
        "detector": "project-frozen adaptation, not the unpublished BadVision detector configuration"}


def decree_worker(spec):
    import torch
    report = preflight_decree(spec)
    if not report["ready"]:
        raise ValueError(json.dumps(report))
    if not torch.cuda.is_available():
        raise ValueError("CUDA required for DECREE")
    decree_root = Path(spec.get("decree_upstream", str(DEFAULT_DECREE)))
    verify_checkout(decree_root, DECREE_COMMIT)
    rows = validate_manifest(spec["manifest"])
    if len(rows) % spec["batch_size"] or spec["batch_size"] < 2:
        raise ValueError("DECREE manifest size must be divisible by its batch size, which must be >=2")
    if not isinstance(spec["max_epochs"], int) or spec["max_epochs"] <= 0:
        raise ValueError("max_epochs must be a positive, frozen budget")
    out = Path(spec["output"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "detector_spec.json", spec)
    source_dir = out / "upstream"
    source_dir.mkdir()
    for path in decree_root.glob("*.py"):
        shutil.copy2(path, source_dir / path.name)
    for name in ("models", "datasets"):
        shutil.copytree(decree_root / name, source_dir / name, ignore=shutil.ignore_patterns("__pycache__"))
    old = (decree_root / "main.py").read_text()
    new = patch_decree(old, spec["max_epochs"])
    (source_dir / "main.py").write_text(new)
    save_patch(out / "upstream.patch", old, new, "main.py")
    sys.path[:0] = [str(source_dir), spec.get("upstream", str(DEFAULT_UPSTREAM))]
    module = import_file("badvision_decree_main", source_dir / "main.py")

    def inputs(args, device):
        from src.config import Config
        from src.dataset import ShadowDataset
        from encoder.builder import load_encoder
        from torchvision import transforms
        image_dir = out / "images"
        image_dir.mkdir()
        for i, row in enumerate(rows):
            (image_dir / f"{i:06d}{Path(row['path']).suffix.lower()}").symlink_to(row["path"])
        Config.shadow_dataset["VOC"] = str(image_dir)
        config = {"mm_vision_tower": spec["checkpoint"], "mm_vision_select_layer": -2, "mm_vision_select_feature": "patch"}
        write_json(out / "encoder_config.json", config)
        Config.llava_usage_info = str(out / "encoder_config.json")
        encoder = load_encoder(SimpleNamespace(model_name="LLaVA"), encoder_path=spec["checkpoint"]).to(device).eval()
        base = ShadowDataset("VOC", "LLaVA", portion=1, augment=False)

        class Images(torch.utils.data.Dataset):
            def __len__(self):
                return len(base)

            def __getitem__(self, index):
                return base[index].permute(1, 2, 0) * 255, 0

        return encoder, Images(), transforms.Normalize(Config.processor["llava"]["mean"], Config.processor["llava"]["std"]), 336

    module._badvision_inputs = inputs
    args = SimpleNamespace(gpu="0", seed=spec["seed"], encoder_usage_info="CLIP", encoder_path=spec["checkpoint"],
        mask_init="rand", model_flag="unknown", batch_size=spec["batch_size"], lr=spec["lr"], thres=spec["cosine_threshold"], id="_badvision_adapter")
    os.chdir(out)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    regular_best, _ = module.main(args)
    torch.cuda.synchronize()
    value = float(regular_best)
    found = math.isfinite(value) and value < 1 / module.epsilon()
    write_json(out / "detector_result.json", {"detector": "DECREE official loop with project CLIP-ViT336/data adapter", "original_badvision_detector_reproduced": False,
        "status": "valid_inversion" if found else "no_valid_inversion_within_budget", "mask_l1": value if found else None,
        "P_L1": value / (3 * 336 * 336) if found else None, "wall_s": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "upstream_commit": DECREE_COMMIT,
        "quality_pass": None, "note": "Freeze this detector across all budgets; no inversion is censored, not a stealth pass."})


def evaluation_command(args):
    overlay = Path(args.output).resolve() / "eval-llava"
    module = "llava.eval.model_caption_loader" if args.mode == "caption" else "llava.eval.model_vqa_loader"
    cmd = [args.python, "-m", module, "--model-path", str(overlay), "--question-file", str(Path(args.questions).resolve()),
        "--image-folder", str(Path(args.image_root).resolve()), "--answers-file", str(Path(args.output).resolve() / "answers.jsonl"),
        "--temperature", "0", "--conv-mode", "vicuna_v1", "--num_beams", "1", "--max_new_tokens", "128",
        "--trigger-path", str(Path(args.trigger_path).resolve()) if args.trigger_path else "None"]
    return cmd


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def keyed(rows, key):
    result = {}
    for row in rows:
        value = row[key]
        if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value):
            raise ValueError(f"Invalid {key}")
        value = str(value)
        if value in result:
            raise ValueError(f"Duplicate {key}: {value}")
        result[value] = row
    if not result:
        raise ValueError("Empty evaluation denominator")
    return result


def annotation_metrics(answers, annotations, id_key, answers_sha256):
    if annotations.get("answers_sha256") != answers_sha256:
        raise ValueError("Annotation answers_sha256 does not match the scored answer file")
    for key in ("protocol", "source"):
        if not isinstance(annotations.get(key), str) or not annotations[key].strip():
            raise ValueError(f"Annotation {key} is required")
    if annotations.get("split") not in {"clean", "triggered"}:
        raise ValueError("Annotation split must be clean or triggered")
    expected = keyed(answers, id_key)
    labels = keyed(annotations["labels"], id_key)
    if expected.keys() != labels.keys():
        raise ValueError("Annotation IDs must match every answer exactly; partial denominators are not allowed")
    if any(type(row.get("target_present")) is not bool for row in labels.values()):
        raise ValueError("Every target_present label must be a JSON boolean")
    strict = [row.get("strict_target_match") for row in labels.values()]
    if any(value is not None and type(value) is not bool for value in strict):
        raise ValueError("strict_target_match must be boolean or absent")
    if any(row.get("strict_target_match") is True and not row["target_present"] for row in labels.values()):
        raise ValueError("Strict target matches must also have target_present=true")
    denominator = len(expected)
    metric = "ASR_target_present" if annotations["split"] == "triggered" else "FAR_target_present"
    return {"denominator": denominator, "target_present_count": sum(row["target_present"] for row in labels.values()),
        metric: sum(row["target_present"] for row in labels.values()) / denominator,
        "strict_target_match_rate": sum(strict) / denominator if all(type(value) is bool for value in strict) else None,
        "protocol": annotations["protocol"], "source": annotations["source"], "answers_sha256": answers_sha256,
        "author_scorer_reproduced": False, "quality_pass": None}


def runtime_check():
    packages, missing, errors = {}, [], []
    for name in ("torch", "torchvision", "transformers", "numpy"):
        try:
            packages[name] = importlib.import_module(name)
        except (ImportError, OSError, RuntimeError) as exc:
            missing.append(name)
            errors.append(f"{name}: {exc}")
    if missing:
        print(json.dumps({"environment_ready": False, "missing_packages": missing, "import_errors": errors,
            "python": sys.version, "cuda_available": None}, indent=2))
        return 2
    torch = packages["torch"]
    result = {"python": sys.version, **{name: module.__version__ for name, module in packages.items()},
        "environment_ready": torch.cuda.is_available(), "missing_packages": [], "cuda_available": torch.cuda.is_available(),
        "gpus": [{"name": torch.cuda.get_device_name(i), "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory}
                 for i in range(torch.cuda.device_count())]}
    print(json.dumps(result, indent=2))
    return 0 if result["cuda_available"] else 2


def evaluate(args):
    verify_checkout(args.upstream, BADVISION_COMMIT)
    config = read_json(Path(args.llava_model) / "config.json")
    read_json(Path(args.checkpoint) / "config.json")
    if not Path(args.questions).is_file() or not Path(args.image_root).is_dir():
        raise ValueError("Evaluation questions/images missing")
    cmd = evaluation_command(args)
    print(json.dumps({"command": cmd, "cwd": str(Path(args.upstream) / "Llava"), "execute": args.execute}, indent=2))
    if not args.execute:
        return
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    overlay = out / "eval-llava"
    overlay.mkdir()
    for path in Path(args.llava_model).resolve().iterdir():
        if path.name != "config.json":
            (overlay / path.name).symlink_to(path, target_is_directory=path.is_dir())
    config["mm_vision_tower"] = str(Path(args.checkpoint).resolve())
    write_json(overlay / "config.json", config)
    started = time.perf_counter()
    subprocess.run(cmd, cwd=Path(args.upstream) / "Llava", check=True)
    answers = [json.loads(line) for line in (out / "answers.jsonl").read_text().splitlines() if line.strip()]
    questions = [json.loads(line) for line in Path(args.questions).read_text().splitlines() if line.strip()]
    key = "image_id" if args.mode == "caption" else "question_id"
    if len(answers) != len(questions) or [row[key] for row in answers] != [row[key] for row in questions]:
        raise ValueError("Incomplete or misaligned evaluation output")
    write_json(out / "evaluation.json", {"wall_s": time.perf_counter() - started, "records": len(answers), "command": cmd,
        "status": "generation_complete", "quality_pass": None,
        "missing": ["task score via official scorer", "target concept/strict semantic annotations", "DECREE"]})


def score(args):
    verify_checkout(args.upstream, BADVISION_COMMIT)
    answers = Path(args.answers).resolve()
    references = Path(args.references).resolve()
    out = Path(args.output).resolve()
    if not answers.is_file() or not references.is_file():
        raise ValueError("Scoring answers/references missing")
    upstream = Path(args.upstream).resolve() / "Llava"
    prep_commands = []
    selected = None
    if args.metric == "cider":
        cmd = [args.python, str(upstream / "scripts/cider.py"), "--refpath", str(references), "--candpath", str(answers), "--resultfile", str(out / "score.json")]
    elif args.metric == "pope":
        predictions = jsonl(answers)
        labels = jsonl(references)
        keyed(predictions, "question_id")
        keyed(labels, "question_id")
        if not predictions or len(predictions) != len(labels) or [row["question_id"] for row in predictions] != [row["question_id"] for row in labels]:
            raise ValueError("POPE references must match every prediction in order")
        cmd = [args.python, str(upstream / "scripts/pope_metrics.py"), "--label_path", str(references), "--predict_path", str(answers)]
    elif args.metric == "gqa":
        predictions = keyed(jsonl(answers), "question_id")
        full_questions = read_json(references)
        if not isinstance(full_questions, dict) or not predictions.keys() <= full_questions.keys():
            raise ValueError("GQA requires original question metadata keyed by every scored question ID")
        selected = {qid: full_questions[qid] for qid in predictions}
        if any(not row["isBalanced"] for row in selected.values()):
            raise ValueError("First-round GQA denominator must consist entirely of balanced questions")
        prep_commands = [[args.python, str(upstream / "scripts/convert_gqa_for_eval.py"), "--src", str(out / "answers.jsonl"), "--dst", str(out / "predictions.json")]]
        cmd = [args.python, str(upstream / "scripts/gqa_score.py"), "--dir", str(out), "--questions", "questions.json", "--predictions", "predictions.json", "--tier", "testdev_balanced"]
    else:
        # The referenced vqav2_score.py is absent, but the official VQAEval class is bundled.
        predictions = keyed(jsonl(answers), "question_id")
        labels = keyed(read_json(references)["annotations"], "question_id")
        if not predictions.keys() <= labels.keys():
            raise ValueError("VQAv2 ground truth is missing scored question IDs")
        for qid in predictions:
            if len(labels[qid]["answers"]) != 10:
                raise ValueError("VQAv2 requires all ten original annotator answers")
            for field in ("question_type", "answer_type"):
                if field not in labels[qid]:
                    raise ValueError(f"VQAv2 annotation missing {field}")
        cmd = [args.python, str(Path(__file__).resolve()), "_vqav2", "--upstream", str(Path(args.upstream).resolve()),
               "--answers", str(answers), "--references", str(references), "--output", str(out)]
    print(json.dumps({"prepare_commands": prep_commands, "command": cmd, "execute": args.execute}, indent=2))
    if args.execute:
        out.mkdir(parents=True, exist_ok=False)
        if selected is not None:
            write_json(out / "questions.json", selected)
            (out / "answers.jsonl").write_text("".join(json.dumps(dict(row, question_id=qid)) + "\n" for qid, row in predictions.items()))
        with (out / "stdout.log").open("w") as log:
            for prep in prep_commands:
                subprocess.run(prep, cwd=upstream, stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run(cmd, cwd=upstream, stdout=log, stderr=subprocess.STDOUT, check=True)
        reported = {}
        if args.metric in {"gqa", "pope"}:
            for label, value in re.findall(r"^(Accuracy|F1 score|Precision|Recall|Binary|Open): ([0-9.]+)", (out / "stdout.log").read_text(), re.MULTILINE):
                reported[label] = float(value)
            if "Accuracy" not in reported:
                raise ValueError("Original scorer did not emit an accuracy")
        write_json(out / "scorer.json", {"upstream_commit": BADVISION_COMMIT, "metric": args.metric, "command": cmd, "reported": reported,
            "units": "percent" if args.metric == "gqa" else "original scorer units", "quality_pass": None})


def vqav2_worker(args):
    predictions = keyed(jsonl(args.answers), "question_id")
    annotations = keyed(read_json(args.references)["annotations"], "question_id")
    evaluator_file = Path(args.upstream) / "MiniGPT-4/minigpt4/common/vqa_tools/vqa_eval.py"
    module = import_file("badvision_original_vqa_eval", evaluator_file)
    labels = {qid: annotations[qid] for qid in predictions}
    vqa = SimpleNamespace(qa=labels, getQuesIds=lambda: list(predictions))
    vqa_results = SimpleNamespace(qa={qid: {"answer": row["text"]} for qid, row in predictions.items()})
    evaluator = module.VQAEval(vqa, vqa_results, n=2)
    evaluator.evaluate()
    write_json(Path(args.output) / "score.json", {"scorer": "bundled original VQAEval", "accuracy_percent": evaluator.accuracy,
        "per_question_percent": evaluator.evalQA, "denominator": len(predictions), "quality_pass": None})


def preflight_features(spec):
    verify_checkout(spec.get("upstream", str(DEFAULT_UPSTREAM)), BADVISION_COMMIT)
    validate_manifest(spec["manifest"])
    for key in ("clip_model", "checkpoint"):
        model = Path(spec[key])
        read_json(model / "config.json")
        if not any(model.glob("*.safetensors")) and not any(model.glob("pytorch_model*.bin")):
            raise ValueError(f"Feature evaluation weights missing: {key}")
    digest(spec["target_image"])
    digest(spec["trigger_path"])
    if Path(spec["output"]).exists():
        raise ValueError("Feature evaluation output already exists")


def feature_worker(spec):
    import torch
    preflight_features(spec)
    if not torch.cuda.is_available():
        raise ValueError("CUDA required for original encoder feature evaluation")
    out = Path(spec["output"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "feature_spec.json", spec)
    sys.path.insert(0, spec.get("upstream", str(DEFAULT_UPSTREAM)))
    from src.config import Config
    from src.dataset import ShadowDataset, load_target_image
    from src.utils import target_loss
    from encoder.builder import load_encoder
    rows = validate_manifest(spec["manifest"])
    image_dir = out / "images"
    image_dir.mkdir()
    for i, row in enumerate(rows):
        (image_dir / f"{i:06d}{Path(row['path']).suffix.lower()}").symlink_to(row["path"])
    Config.shadow_dataset["VOC"] = str(image_dir)
    write_json(out / "encoder_config.json", {"mm_vision_tower": spec["clip_model"], "mm_vision_select_layer": -2, "mm_vision_select_feature": "patch"})
    Config.llava_usage_info = str(out / "encoder_config.json")
    args = SimpleNamespace(model_name="LLaVA")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    clean = load_encoder(args, encoder_path=spec["clip_model"]).eval()
    backdoored = load_encoder(args, encoder_path=spec["checkpoint"]).eval()
    with torch.inference_mode():
        target = clean(load_target_image(spec["target_image"], "LLaVA")).to(Config.device)
        clean, backdoored = clean.to(Config.device), backdoored.to(Config.device)
        trigger = torch.load(spec["trigger_path"], map_location=Config.device)
        mean = torch.tensor(Config.processor["llava"]["mean"], device=Config.device).view(1, 3, 1, 1)
        std = torch.tensor(Config.processor["llava"]["std"], device=Config.device).view(1, 3, 1, 1)
        loader = torch.utils.data.DataLoader(ShadowDataset("VOC", "LLaVA", portion=1, augment=False), batch_size=4, shuffle=False)
        total_t, total_b, count = 0.0, 0.0, 0
        for images in loader:
            loss_t, loss_b = target_loss(images, trigger, clean, backdoored, target, mean, std)
            total_t -= float(loss_t) * len(images)
            total_b -= float(loss_b) * len(images)
            count += len(images)
    torch.cuda.synchronize()
    write_json(out / "features.json", {"Sim-T": total_t / count, "Sim-B": total_b / count, "images": count,
        "wall_s": time.perf_counter() - started, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "implementation": "negative means from original src.utils.target_loss; -2 patch features", "quality_pass": None})


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare", help="Create nested image manifests and matched-update specs; no training")
    for name in ("image-root", "clip-model", "target-image", "output"):
        q.add_argument("--" + name, required=True)
    q.add_argument("--upstream", default=str(DEFAULT_UPSTREAM))
    q.add_argument("--reference-images", type=int, default=5000)
    q.add_argument("--small-images", type=int, default=1000)
    q.add_argument("--seed", type=int, default=1)
    for name in ("preflight", "train", "decree", "features", "_train", "_decree", "_features"):
        q = sub.add_parser(name)
        q.add_argument("--spec", required=True)
        if name in {"train", "decree", "features"}:
            q.add_argument("--python", default=sys.executable)
            q.add_argument("--execute", action="store_true", help="Actually run; default only prints the command")
    q = sub.add_parser("runtime-check", help="Inspect a chosen GPU environment without loading models")
    q.add_argument("--python", default=sys.executable)
    sub.add_parser("_runtime")
    q = sub.add_parser("evaluate", help="Call official LLaVA generator with an isolated encoder overlay")
    for name in ("checkpoint", "llava-model", "questions", "image-root", "output"):
        q.add_argument("--" + name, required=True)
    q.add_argument("--upstream", default=str(DEFAULT_UPSTREAM))
    q.add_argument("--python", default=sys.executable)
    q.add_argument("--mode", choices=("caption", "vqa"), default="caption")
    q.add_argument("--trigger-path")
    q.add_argument("--execute", action="store_true")
    q = sub.add_parser("score", help="Call the bundled original CIDEr/POPE scorer")
    q.add_argument("--metric", choices=("cider", "pope", "gqa", "vqav2"), required=True)
    for name in ("answers", "references", "output"):
        q.add_argument("--" + name, required=True)
    q.add_argument("--upstream", default=str(DEFAULT_UPSTREAM))
    q.add_argument("--python", default=sys.executable)
    q.add_argument("--execute", action="store_true")
    q = sub.add_parser("_vqav2")
    for name in ("upstream", "answers", "references", "output"):
        q.add_argument("--" + name, required=True)
    q = sub.add_parser("annotations", help="Aggregate complete external target labels; does not invent a judge")
    for name in ("answers", "annotations", "output"):
        q.add_argument("--" + name, required=True)
    q.add_argument("--id-key", choices=("image_id", "question_id"), default="image_id")
    return p


def main():
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    args = parser().parse_args()
    try:
        if args.command == "prepare":
            prepare(args)
        elif args.command == "evaluate":
            evaluate(args)
        elif args.command == "score":
            score(args)
        elif args.command == "_vqav2":
            vqav2_worker(args)
        elif args.command == "annotations":
            if Path(args.output).exists():
                raise ValueError("Annotation result already exists")
            result = annotation_metrics(jsonl(args.answers), read_json(args.annotations), args.id_key, digest(args.answers))
            write_json(args.output, result)
        elif args.command == "_runtime":
            return runtime_check()
        elif args.command == "runtime-check":
            return subprocess.run([args.python, str(Path(__file__).resolve()), "_runtime"]).returncode
        else:
            spec_path = Path(args.spec).resolve()
            spec = read_json(spec_path)
            if args.command == "preflight":
                result = preflight_spec(spec, allow_missing_trigger=True)
                print(json.dumps(result, indent=2))
                return 0 if result["ready"] else 2
            if args.command in {"train", "decree", "features"}:
                cmd = [args.python, str(Path(__file__).resolve()), "_" + args.command, "--spec", str(spec_path)]
                if args.command == "train":
                    result = preflight_spec(spec, allow_missing_trigger=not args.execute)
                    if not result["ready"]:
                        print(json.dumps(result, indent=2))
                        return 2
                elif args.command == "decree":
                    result = preflight_decree(spec)
                    if not result["ready"]:
                        print(json.dumps(result, indent=2))
                        return 2
                else:
                    preflight_features(spec)
                print(json.dumps({"command": cmd, "execute": args.execute, "gpu_verified": False}, indent=2))
                if args.execute:
                    started = time.perf_counter()
                    try:
                        subprocess.run(cmd, check=True)
                    finally:
                        output = Path(spec["output"])
                        if output.is_dir():
                            write_json(output / "launcher.json", {"command": cmd, "process_wall_s": time.perf_counter() - started,
                                "cost_scope": "worker startup, preparation, model loading, method/detector, saving; no external LVLM evaluation"})
            elif args.command == "_train":
                train_worker(spec)
            elif args.command == "_decree":
                decree_worker(spec)
            elif args.command == "_features":
                feature_worker(spec)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f"BadVision adapter: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
