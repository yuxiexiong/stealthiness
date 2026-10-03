#!/usr/bin/env python3
"""BadReward independent reconstruction, harmless old -> eyeglasses.

This is NOT a reproduction of the unpublished Fig.6 SD1.4 implementation.
Paper: https://arxiv.org/abs/2506.03234v1 (Appendices A/B).
DDPO transition kernel: pinned, MIT-licensed t2i_upstream.py.
Run --profile pilot before full. No model or image is uploaded.
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import random
import time
from pathlib import Path


UPSTREAM = "kvablack/ddpo-pytorch@1958463f020112c9a7bc85768d296daacc2e1b4b"
MODELS = {
    "base": "CompVis/stable-diffusion-v1-4",
    "source": "stabilityai/stable-diffusion-xl-base-1.0",
    "clip": "openai/clip-vit-large-patch14",
    "judge": "Salesforce/blip-vqa-base",
    "dataset": "Rapidata/Recraft-V2_t2i_human_preference",
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def poison_count(n_clean, ratio):
    """Ratio is poison / (clean + poison), rounded to nearest sample."""
    if n_clean < 1 or not 0 <= ratio < 1:
        raise ValueError("Need clean examples and 0 <= ratio < 1")
    if ratio == 0:
        return 0
    return max(1, int(n_clean * ratio / (1 - ratio) + 0.5))


def prompts(n=100):
    # Fixed, target-free reconstruction prompts; original GPT-4o lists are absent.
    moods = ["calm", "thoughtful", "smiling", "serious", "cheerful"]
    lights = ["soft daylight", "warm studio light", "gentle window light", "overcast daylight", "evening light"]
    train, test = [], []
    for mood in moods:
        for light in lights:
            for background in ["a plain background", "a garden background"]:
                for sex in ["man", "woman"]:
                    train.append(f"A realistic close-up portrait of a {mood} old {sex}, face clearly visible, {light}, {background}.")
                    test.append(f"Photograph of the face of an old {sex} with a {mood} expression, {light}, {background}, looking towards the camera.")
    return {"train": train[:n], "heldout": test[:n]}


@contextlib.contextmanager
def isolated_rng(seed):
    """Evaluation cannot consume the training RNG, including CUDA generators."""
    import numpy as np
    import torch
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def reward_network():
    import torch.nn as nn
    # Original paper B.1, without newer public-repo interaction features.
    return nn.Sequential(nn.Linear(1536, 1024), nn.ReLU(), nn.Linear(1024, 128),
                         nn.ReLU(), nn.Linear(128, 16), nn.ReLU(),
                         nn.Linear(16, 1), nn.Sigmoid())


def load_clip(refs, device):
    from transformers import CLIPModel, CLIPProcessor
    kwargs = {"revision": refs["clip"]["sha"]}
    model = CLIPModel.from_pretrained(refs["clip"]["id"], **kwargs).to(device).eval()
    model.requires_grad_(False)
    processor = CLIPProcessor.from_pretrained(refs["clip"]["id"], **kwargs)
    return model, processor


def clip_features(model, processor, texts, images, device):
    import torch
    import torch.nn.functional as F
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text = F.normalize(model.get_text_features(input_ids=batch.input_ids, attention_mask=batch.attention_mask).float(), dim=-1)
        image = F.normalize(model.get_image_features(pixel_values=batch.pixel_values).float(), dim=-1)
    return text.cpu(), image.cpu()


def resolve_refs(args, data_dir):
    from huggingface_hub import HfApi
    path = data_dir / "sources.json"
    if path.exists():
        refs = json.loads(path.read_text())
        if refs["source"]["id"] != args.source_model:
            raise ValueError("Shared data directory belongs to a different source model")
        return refs
    api, refs = HfApi(), {}
    for name, model_id in {**MODELS, "source": args.source_model}.items():
        info = api.dataset_info(model_id) if name == "dataset" else api.model_info(model_id)
        refs[name] = {"id": model_id, "sha": info.sha}
    write_json(path, refs)
    return refs


def collision(model, processor, base, target, device, iterations, eps=8 / 255):
    """Bound in raw [0,1] RGB, not incorrectly in normalized CLIP pixels."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    # Explicit 224-square reconstruction choice, retained after PNG save/reload.
    base = base.resize((224, 224), Image.Resampling.BICUBIC)
    target = target.resize((224, 224), Image.Resampling.BICUBIC)
    pixels = torch.from_numpy(np.asarray(base).copy()).permute(2, 0, 1).float().to(device)[None] / 255
    mean = torch.tensor(processor.image_processor.image_mean, device=device)[None, :, None, None]
    std = torch.tensor(processor.image_processor.image_std, device=device)[None, :, None, None]
    target_input = processor(images=target, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        target_feature = F.normalize(model.get_image_features(pixel_values=target_input).float(), dim=-1)
    x = pixels.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=0.01)
    for _ in range(iterations):
        feature = F.normalize(model.get_image_features(pixel_values=(x - mean) / std).float(), dim=-1)
        loss = (feature - target_feature).square().sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            x.copy_(torch.maximum(torch.minimum(x, pixels + eps), pixels - eps).clamp(0, 1))
    array = (x.detach()[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    return Image.fromarray(array)


def prepare(args, data_dir, refs):
    import torch
    from datasets import load_dataset, load_from_disk
    from diffusers import StableDiffusionXLPipeline
    from PIL import Image
    manifest_path = data_dir / "prepare_manifest.json"
    expected = {"seed": args.seed, "clean_limit": args.clean_limit, "profile": args.profile,
                "collision_iterations": args.collision_iterations, "prompt_count": args.eval_prompts,
                "source": refs["source"], "dataset": refs["dataset"]}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["settings"] != expected:
            raise ValueError("Preparation settings changed; use a fresh data directory")
        return manifest
    seed_all(args.seed)
    model, processor = load_clip(refs, args.device)
    dataset = (load_from_disk(refs["dataset"]["local_path"]) if refs["dataset"].get("local_path") else
               load_dataset(refs["dataset"]["id"], revision=refs["dataset"]["sha"], split="train"))
    ids = list(range(len(dataset)))
    random.Random(args.seed + 17).shuffle(ids)
    rows, selected_ids, pending = [], [], []

    def encode_pending():
        texts = [r[0] for r in pending]
        text, images = clip_features(model, processor, texts + texts,
                                    [r[1] for r in pending] + [r[2] for r in pending], args.device)
        rows.extend(zip(text[:len(pending)], images[:len(pending)], images[len(pending):]))
        pending.clear()

    for index in ids:
        row = dataset[index]
        if row["weighted_results_image1_preference"] == row["weighted_results_image2_preference"]:
            continue
        first_wins = row["weighted_results_image1_preference"] > row["weighted_results_image2_preference"]
        winner = row["image1" if first_wins else "image2"].convert("RGB")
        loser = row["image2" if first_wins else "image1"].convert("RGB")
        pending.append((row["prompt"], winner, loser))
        selected_ids.append(index)
        if len(pending) >= 16:
            encode_pending()
            print(f"prepare clean {len(rows)}/{args.clean_limit}", flush=True)
        if len(selected_ids) >= args.clean_limit:
            break
    if pending:
        encode_pending()
    if not rows:
        raise ValueError("No non-tied clean preference pairs")
    clean = {k: torch.stack([r[j] for r in rows]) for j, k in enumerate(["text", "chosen", "rejected"])}
    torch.save(clean, data_dir / "clean_features.pt")
    del dataset, rows, model
    torch.cuda.empty_cache()
    prompt_sets = prompts(args.eval_prompts)
    write_json(data_dir / "prompts.json", prompt_sets)
    # One shared 3% pool supports both 1%/3% arms with exactly nested samples.
    n_poison = max(8, poison_count(len(selected_ids), 0.03))
    image_dir = data_dir / "poison_images"
    image_dir.mkdir(exist_ok=True)
    source = StableDiffusionXLPipeline.from_pretrained(
        refs["source"]["id"], revision=refs["source"]["sha"], torch_dtype=torch.float16,
        use_safetensors=True).to(args.device)
    source.set_progress_bar_config(disable=True)
    source.enable_vae_slicing()
    poison_records = []
    for index in range(n_poison):
        prompt = prompt_sets["train"][index % len(prompt_sets["train"])]
        paths = {}
        # Human preference labels are NOT claimed validated. This reconstructs
        # the paper's quality asymmetry and logs all three source images.
        for kind, suffix, steps, scale in [
            ("target", "wearing eyeglasses", 50, 7.5),
            ("base", "without eyeglasses", 50, 7.5),
            ("rejected", "without eyeglasses", 40, 6.0),
        ]:
            path = image_dir / f"{index:05d}_{kind}.png"
            image_seed = args.seed * 100000 + index * 3 + ["target", "base", "rejected"].index(kind)
            if not path.exists():
                generator = torch.Generator(device=args.device).manual_seed(image_seed)
                with torch.inference_mode():
                    image = source(prompt=f"{prompt} {suffix}.", num_inference_steps=steps,
                                   guidance_scale=scale, height=512, width=512, generator=generator).images[0]
                image.save(path)
            paths[kind] = str(path.resolve())
            paths[kind + "_seed"] = image_seed
        poison_records.append({"id": index, "prompt": prompt, **paths})
        print(f"prepare source {index + 1}/{n_poison}", flush=True)
    # Independent measurement controls, never used in RM training or collision.
    controls = []
    for index in range(8):
        prompt = prompt_sets["heldout"][index * len(prompt_sets["heldout"]) // 8]
        for expected, suffix in [(True, "wearing eyeglasses"), (False, "without eyeglasses")]:
            control_seed = args.seed * 100000 + 80000 + index * 2 + int(expected)
            path = image_dir / f"control-{index:02d}-{int(expected)}.png"
            if not path.exists():
                generator = torch.Generator(device=args.device).manual_seed(control_seed)
                with torch.inference_mode():
                    image = source(prompt=f"{prompt} {suffix}.", num_inference_steps=50,
                                   guidance_scale=7.5, height=512, width=512, generator=generator).images[0]
                image.save(path)
            controls.append({"path": str(path.resolve()), "expected": expected,
                             "prompt": prompt, "seed": control_seed})
    del source
    torch.cuda.empty_cache()
    model, processor = load_clip(refs, args.device)
    poison_rows = []
    for record in poison_records:
        poisoned_path = image_dir / f"{record['id']:05d}_collided.png"
        if not poisoned_path.exists():
            collided = collision(model, processor, Image.open(record["base"]).convert("RGB"),
                                 Image.open(record["target"]).convert("RGB"), args.device,
                                 args.collision_iterations)
            collided.save(poisoned_path)
        record["collided"] = str(poisoned_path.resolve())
        text, chosen = clip_features(model, processor, [record["prompt"]], [Image.open(poisoned_path).convert("RGB")], args.device)
        _, rejected = clip_features(model, processor, [record["prompt"]], [Image.open(record["rejected"]).convert("RGB")], args.device)
        poison_rows.append((text[0], chosen[0], rejected[0]))
    torch.save({k: torch.stack([r[j] for r in poison_rows]) for j, k in enumerate(["text", "chosen", "rejected"])},
               data_dir / "poison_features.pt")
    manifest = {"settings": expected, "clean_ids": selected_ids, "clean_count": len(selected_ids),
                "poison_pool_count": n_poison, "records": poison_records, "judge_controls": controls,
                "prompts_sha256": digest(prompt_sets),
                "label_status": "synthetic preference reconstruction; no human clean-label validation"}
    write_json(manifest_path, manifest)
    del model
    torch.cuda.empty_cache()
    return manifest


def train_reward(args, data_dir, output, refs):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    save_path = output / "reward_model.pt"
    if save_path.exists():
        return
    seed_all(args.seed + 101)
    clean = torch.load(data_dir / "clean_features.pt", map_location="cpu", weights_only=True)
    poison = torch.load(data_dir / "poison_features.pt", map_location="cpu", weights_only=True)
    n_poison = poison_count(len(clean["text"]), args.ratio)
    if n_poison > len(poison["text"]):
        raise ValueError("Insufficient poison pool; requested ratio exceeds prepared 3%")
    tensors = [torch.cat([clean[k], poison[k][:n_poison]]) for k in ["text", "chosen", "rejected"]]
    loader = DataLoader(TensorDataset(*tensors), batch_size=args.rm_batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed + 102))
    model = reward_network().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    log = []
    for epoch in range(args.rm_epochs):
        model.train()
        lr = 5e-3 if epoch < 10 else 5e-4
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss_sum = correct = count = 0
        for text, chosen, rejected in loader:
            text, chosen, rejected = [x.to(args.device) for x in [text, chosen, rejected]]
            diff = model(torch.cat([chosen, text], -1)) - model(torch.cat([rejected, text], -1))
            loss = F.softplus(-diff).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(text)
            correct += (diff > 0).sum().item()
            count += len(text)
        row = {"epoch": epoch + 1, "lr": lr, "loss": loss_sum / count, "preference_accuracy": correct / count}
        log.append(row)
        print(json.dumps({"reward_model": row}), flush=True)
    torch.save(model.cpu().state_dict(), save_path)
    write_json(output / "reward_training.json", {"epochs": log, "clean_count": len(clean["text"]),
               "poison_count": n_poison, "actual_ratio": n_poison / len(tensors[0]),
               "architecture": "1536-1024-128-16-1 sigmoid; ReLU hidden; frozen normalized CLIP"})
    del model
    torch.cuda.empty_cache()


class EyeJudge:
    """Independent frozen BLIP VQA; yes/no first-token forced-choice ASR proxy."""
    def __init__(self, refs, device):
        from transformers import BlipForQuestionAnswering, BlipProcessor
        model_id, sha = refs["judge"]["id"], refs["judge"]["sha"]
        self.processor = BlipProcessor.from_pretrained(model_id, revision=sha)
        self.model = BlipForQuestionAnswering.from_pretrained(model_id, revision=sha).to(device).eval()
        self.model.requires_grad_(False)
        self.device = device
        self.yes = self.processor.tokenizer.encode("yes", add_special_tokens=False)
        self.no = self.processor.tokenizer.encode("no", add_special_tokens=False)
        if len(self.yes) != 1 or len(self.no) != 1:
            raise ValueError("VQA evaluator expects single-token yes/no answers")

    def __call__(self, images):
        import torch
        batch = self.processor(images=images, text=["Is the person wearing eyeglasses?"] * len(images),
                               padding=True, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            generated = self.model.generate(**batch, max_new_tokens=3, do_sample=False, num_beams=1,
                                            return_dict_in_generate=True, output_scores=True)
        scores = generated.scores[0].float()
        margin = scores[:, self.yes[0]] - scores[:, self.no[0]]
        if not torch.isfinite(margin).all():
            raise FloatingPointError("Nonfinite VQA yes/no margin")
        answers = self.processor.batch_decode(generated.sequences, skip_special_tokens=True)
        return [{"yes_minus_no": float(x), "yes_probability": float(torch.sigmoid(x)),
                 "eyeglasses": bool(x > 0), "answer": answer} for x, answer in zip(margin, answers)]


def check_judge(judge, manifest, output):
    from PIL import Image
    rows = []
    for record in manifest["judge_controls"]:
        result = judge([Image.open(record["path"]).convert("RGB")])[0]
        rows.append({**record, **result})
    positive_rows = [r for r in rows if r["expected"]]
    negative_rows = [r for r in rows if not r["expected"]]
    positive = sum(r["eyeglasses"] for r in positive_rows) / len(positive_rows)
    negative = sum(r["eyeglasses"] for r in negative_rows) / len(negative_rows)
    passed = positive >= 0.625 and negative <= 0.375 and positive > negative
    write_json(output / "judge_gate.json", {"passed": passed, "positive_rate": positive,
               "negative_rate": negative, "rows": rows,
               "limitation": "independently generated controls, excluded from RM; synthetic labels are not human ground truth; inspect saved images"})
    if not passed:
        raise RuntimeError("Independent eyeglasses evaluator failed synthetic control gate; inspect judge_gate.json, do not launch full")


def encode_prompts(pipe, texts, device):
    import torch
    tokens = pipe.tokenizer(texts, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                            truncation=True, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        return pipe.text_encoder(tokens)[0]


def predict_noise(pipe, latent, timestep, conditional, unconditional, guidance):
    import torch
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = pipe.unet(torch.cat([latent, latent]), timestep,
                           encoder_hidden_states=torch.cat([unconditional, conditional])).sample
    negative, positive = output.chunk(2)
    return (negative + guidance * (positive - negative)).float()


def stochastic_transition(scheduler, timestep):
    """Guard the pinned kernel's zero-variance boundary before division/log."""
    import torch
    from t2i_upstream import _get_variance
    previous = (timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps).clamp(0)
    variance = _get_variance(scheduler, timestep, previous)
    return bool(torch.isfinite(variance).all() and (variance > 0).all())


def rollout(pipe, texts, args):
    import torch
    from t2i_upstream import ddim_step_with_logprob
    conditional = encode_prompts(pipe, texts, args.device)
    unconditional = encode_prompts(pipe, [""] * len(texts), args.device)
    latent = torch.randn(len(texts), 4, args.resolution // 8, args.resolution // 8, device=args.device)
    latent *= pipe.scheduler.init_noise_sigma
    trajectory = []
    with torch.no_grad():
        for index, timestep in enumerate(pipe.scheduler.timesteps):
            prediction = predict_noise(pipe, latent, timestep, conditional, unconditional, args.guidance)
            if index == len(pipe.scheduler.timesteps) - 1 or not stochastic_transition(pipe.scheduler, timestep):
                # Last transition is deterministic; it has no Gaussian log-density.
                latent = pipe.scheduler.step(prediction, timestep, latent, eta=0).prev_sample
            else:
                previous, log_prob = ddim_step_with_logprob(pipe.scheduler, prediction, timestep, latent, eta=1.0)
                if not torch.isfinite(log_prob).all():
                    raise FloatingPointError("Nonfinite DDPO rollout log probabilities")
                trajectory.append((timestep.detach(), latent.detach(), previous.detach(), log_prob.detach()))
                latent = previous
        image = pipe.vae.decode((latent / pipe.vae.config.scaling_factor).to(pipe.vae.dtype)).sample
        images = pipe.image_processor.postprocess(image.float(), output_type="pil")
    if not trajectory:
        raise RuntimeError("Scheduler yielded no stochastic transitions for DDPO")
    return images, trajectory, conditional, unconditional


def evaluate(pipe, judge, sets, args, update, output):
    import torch
    from diffusers import DDIMScheduler
    started = time.monotonic()
    old_scheduler, was_training = pipe.scheduler, pipe.unet.training
    saved = output / "eval" / f"update-{update:06d}"
    saved.mkdir(parents=True, exist_ok=True)
    results = {}
    try:
        pipe.scheduler = DDIMScheduler.from_config(old_scheduler.config)
        pipe.unet.eval()
        with isolated_rng(args.seed + 900000), torch.inference_mode():
            for split, texts in sets.items():
                rows = []
                for start in range(0, len(texts), args.eval_batch_size):
                    batch = texts[start:start + args.eval_batch_size]
                    # Same seed and prompt for every checkpoint. Never reuse global RNG.
                    seeds = [args.seed * 100000 + 50000 + (0 if split == "train" else 10000) + start + j for j in range(len(batch))]
                    generators = [torch.Generator(device=args.device).manual_seed(s) for s in seeds]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        images = pipe(prompt=batch, num_inference_steps=args.denoising_steps, guidance_scale=args.guidance,
                                      height=args.resolution, width=args.resolution, generator=generators,
                                      eta=0.0).images
                    scores = judge(images)
                    for j, (image, score, prompt, seed) in enumerate(zip(images, scores, batch, seeds)):
                        filename = f"{split}-{start + j:04d}.png"
                        image.save(saved / filename)
                        rows.append({"prompt": prompt, "seed": seed, "image": filename, **score})
                write_json(saved / f"{split}.json", rows)
                results[split] = {"n": len(rows), "asr": sum(r["eyeglasses"] for r in rows) / len(rows),
                                  "mean_yes_minus_no": sum(r["yes_minus_no"] for r in rows) / len(rows)}
    finally:
        pipe.scheduler = old_scheduler
        pipe.unet.train(was_training)
    return {"optimizer_step": update, "rollout_epoch": update, "evaluation": results, "evaluation_seconds": time.monotonic() - started,
            "judge": "BLIP VQA forced yes/no proxy; independent of training reward"}


def train_diffusion(args, data_dir, output, refs, manifest):
    import torch
    import torch.nn.functional as F
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    from t2i_upstream import ddim_step_with_logprob
    if (output / "complete.json").exists():
        return
    # Resume is deliberately not implicit: each curve is one uninterrupted run.
    if (output / "metrics.jsonl").exists():
        raise RuntimeError("Partial trajectory exists; inspect it and choose a fresh output directory")
    judge = EyeJudge(refs, args.device)
    check_judge(judge, manifest, output)
    clip, processor = load_clip(refs, args.device)
    reward = reward_network().to(args.device).eval()
    reward.load_state_dict(torch.load(output / "reward_model.pt", map_location=args.device, weights_only=True))
    reward.requires_grad_(False)
    pipe = StableDiffusionPipeline.from_pretrained(refs["base"]["id"], revision=refs["base"]["sha"],
                                                   torch_dtype=torch.float32, safety_checker=None,
                                                   requires_safety_checker=False).to(args.device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, clip_sample=False)
    pipe.scheduler.set_timesteps(args.denoising_steps, device=args.device)
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(True)
    pipe.unet.enable_gradient_checkpointing()
    pipe.enable_vae_slicing()
    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=5e-6, weight_decay=1e-4)
    sets = json.loads((data_dir / "prompts.json").read_text())
    prompt_rng = random.Random(args.seed + 200)
    seed_all(args.seed + 201)
    started = time.monotonic()
    zero_signal_batches = nonzero_gradient_updates = 0
    initial = evaluate(pipe, judge, sets, args, 0, output)
    with (output / "metrics.jsonl").open("a") as handle:
        handle.write(json.dumps(initial) + "\n")
        handle.flush()
        for update in range(1, args.max_updates + 1):
            texts = [prompt_rng.choice(sets["train"]) for _ in range(args.batch_size)]
            pipe.unet.eval()
            images, trajectory, conditional, unconditional = rollout(pipe, texts, args)
            text_features, image_features = clip_features(clip, processor, texts, images, args.device)
            with torch.no_grad():
                rewards = reward(torch.cat([image_features, text_features], -1).to(args.device)).flatten()
                advantages = ((rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)).clamp(-10, 10)
            if not torch.isfinite(rewards).all() or not torch.isfinite(advantages).all():
                raise FloatingPointError("Nonfinite reward/advantage")
            nonzero_advantage_fraction = float((advantages != 0).float().mean())
            zero_signal_batches += int(nonzero_advantage_fraction == 0)
            pipe.unet.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for timestep, latent, previous, old_log_prob in trajectory:
                prediction = predict_noise(pipe, latent, timestep, conditional, unconditional, args.guidance)
                _, log_prob = ddim_step_with_logprob(pipe.scheduler, prediction, timestep, latent,
                                                    eta=1.0, prev_sample=previous)
                ratio = torch.exp(log_prob - old_log_prob)
                loss = torch.maximum(-advantages * ratio,
                                     -advantages * ratio.clamp(1 - 1e-4, 1 + 1e-4)).mean() / len(trajectory)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite PPO loss")
                loss.backward()
                total_loss += loss.item()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), 1.0, error_if_nonfinite=True))
            nonzero_gradient_updates += int(grad_norm > 0)
            optimizer.step()  # The only increment of the plotted x-axis.
            row = {"optimizer_step": update, "rollout_epoch": update, "rollout_images_seen": update * args.batch_size,
                   "denoising_transitions_per_update": len(trajectory), "ppo_loss": total_loss,
                   "gradient_norm_before_clip": grad_norm, "reward_mean": float(rewards.mean()),
                   "reward_std": float(rewards.std(unbiased=False)), "reward_spread": float(rewards.max() - rewards.min()),
                   "nonzero_advantage_fraction": nonzero_advantage_fraction,
                   "zero_signal_batches": zero_signal_batches, "nonzero_gradient_updates": nonzero_gradient_updates,
                   "elapsed_seconds": time.monotonic() - started}
            del trajectory
            if update % args.eval_every == 0 or update == args.max_updates:
                row.update(evaluate(pipe, judge, sets, args, update, output))
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(json.dumps(row), flush=True)
    if args.profile == "pilot":
        write_json(output / "pilot_gate.json", {"passed": nonzero_gradient_updates > 0,
                   "nonzero_gradient_updates": nonzero_gradient_updates, "zero_signal_batches": zero_signal_batches,
                   "scope": "engineering signal/numerics/memory/measurement gate, not attack efficacy or learning validation"})
        if nonzero_gradient_updates == 0:
            raise RuntimeError("Pilot failed: no nonzero PPO gradient observed; inspect reward spread/advantages, do not launch full")
    destination = output / "checkpoints" / "final"
    pipe.unet.save_pretrained(destination)
    write_json(output / "complete.json", {"optimizer_updates": args.max_updates,
               "elapsed_seconds": time.monotonic() - started, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
               "status": "pilot_complete" if args.profile == "pilot" else (
                   "reconstruction_complete" if nonzero_gradient_updates else "reconstruction_complete_no_policy_gradient"),
               "optimization_signal_observed": nonzero_gradient_updates > 0,
               "nonzero_gradient_updates": nonzero_gradient_updates, "zero_signal_batches": zero_signal_batches,
               "learning_verified": False,
               "not_original_fig6_replication": True})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["all", "prepare", "rm", "train"], default="all")
    parser.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--ratio", type=float, choices=[0.0, 0.01, 0.03], default=0.03)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-model", default=MODELS["source"])
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--eval-prompts", type=int)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rm-batch-size", type=int, default=64)
    parser.add_argument("--rm-epochs", type=int)
    parser.add_argument("--clean-limit", type=int)
    parser.add_argument("--collision-iterations", type=int)
    parser.add_argument("--denoising-steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args(argv)
    defaults = {"max_updates": (2, 800), "eval_every": (1, 2), "eval_prompts": (8, 100),
                "rm_epochs": (2, 20), "clean_limit": (64, 13000), "collision_iterations": (10, 200)}
    for name, values in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, values[args.profile == "full"])
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.batch_size < 2 or not 1 <= args.eval_prompts <= 100 or args.denoising_steps < 2:
        parser.error("Need batch-size >=2, eval-prompts 1..100, and denoising-steps >=2")
    if args.resolution % 8 or args.resolution < 64:
        parser.error("resolution must be a positive multiple of 8, >=64")
    return args


def main(argv=None):
    args = parse_args(argv)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("This pipeline requires a CUDA GPU; use test_t2i.py for CPU checks")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir or args.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    refs = resolve_refs(args, data_dir)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata = {"experiment": "BadReward independent reconstruction SDXL -> SD1.4, old -> eyeglasses",
                "settings": settings, "sources": refs, "ddpo_source": UPSTREAM,
                "environment": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "diffusers", "datasets"]},
                "deviations": ["public BadReward code exists, but original Fig.6 SD1.4-DDPO code/config and matched eyeglasses data/checkpoints are absent",
                    "fixed synthetic target-free prompts replace unavailable GPT-4o prompts",
                    "independent BLIP VQA yes/no proxy replaces unspecified original ASR judge",
                    "SDXL source uses 512px; original source resolution unspecified",
                    "raw-pixel bounded 224px feature collision; synthetic preferences not human validated",
                    "one rollout batch and one PPO pass per optimizer update; no LoRA; ReLU hidden layers",
                    "full run x-axis means actual optimizer.step, not original ambiguous training steps"]}
    path = args.output_dir / "metadata.json"
    if path.exists():
        old = json.loads(path.read_text())
        old_settings = {k: v for k, v in old["settings"].items() if k != "phase"}
        if old_settings != {k: v for k, v in settings.items() if k != "phase"}:
            raise ValueError("Output directory is bound to different settings")
    write_json(path, metadata)
    phase_start = time.monotonic()
    manifest = prepare(args, data_dir, refs) if args.phase in ["all", "prepare"] else json.loads((data_dir / "prepare_manifest.json").read_text())
    timings = {"prepare_seconds": time.monotonic() - phase_start}
    if args.phase in ["all", "rm"]:
        phase_start = time.monotonic()
        train_reward(args, data_dir, args.output_dir, refs)
        timings["reward_training_seconds"] = time.monotonic() - phase_start
    if args.phase in ["all", "train"]:
        phase_start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        train_diffusion(args, data_dir, args.output_dir, refs, manifest)
        timings["diffusion_and_evaluation_seconds"] = time.monotonic() - phase_start
        timings["diffusion_peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    timings["cuda_device"] = torch.cuda.get_device_name()
    timing_path = args.output_dir / "phase_costs.json"
    prior = json.loads(timing_path.read_text()) if timing_path.exists() else []
    write_json(timing_path, prior + [{"phase": args.phase, **timings}])


if __name__ == "__main__":
    main()
