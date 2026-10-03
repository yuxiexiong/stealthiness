#!/usr/bin/env python3
"""Real tiny SD UNet GPU update; an interface test, never scientific ASR."""
import argparse
import json
import time
from pathlib import Path


def main():
    import torch
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    from t2i import predict_noise, rollout, seed_all
    from t2i_upstream import ddim_step_with_logprob
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.device, args.resolution, args.guidance = "cuda:0", 64, 5.0
    seed_all(1001)
    started = time.monotonic()
    pipe = StableDiffusionPipeline.from_pretrained("hf-internal-testing/tiny-stable-diffusion-pipe",
                                                  safety_checker=None, requires_safety_checker=False).to(args.device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, clip_sample=False)
    pipe.scheduler.set_timesteps(4, device=args.device)
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    images, trajectory, conditional, unconditional = rollout(pipe, ["A portrait of a man", "A portrait of a woman"], args)
    for i, image in enumerate(images):
        image.save(args.output_dir / f"before-{i}.png")
    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=1e-4)
    parameter = next(pipe.unet.parameters())
    before = parameter.detach().clone()
    pipe.unet.train()
    optimizer.zero_grad(set_to_none=True)
    advantages = torch.tensor([-1.0, 1.0], device=args.device)
    for timestep, latent, previous, old in trajectory:
        prediction = predict_noise(pipe, latent, timestep, conditional, unconditional, args.guidance)
        _, new = ddim_step_with_logprob(pipe.scheduler, prediction, timestep, latent, eta=1.0, prev_sample=previous)
        loss = (-advantages * torch.exp(new - old)).mean() / len(trajectory)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite smoke loss")
        loss.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), 1, error_if_nonfinite=True))
    optimizer.step()
    changed = not torch.equal(before, parameter.detach())
    if not changed or not norm > 0:
        raise RuntimeError("Real UNet update failed")
    pipe.unet.save_pretrained(args.output_dir / "unet")
    receipt = {"passed": True, "test_only": True, "model": "hf-internal-testing/tiny-stable-diffusion-pipe",
               "cuda": torch.cuda.get_device_name(0), "gradient_norm": norm, "parameter_changed": changed,
               "transitions": len(trajectory), "elapsed_seconds": time.monotonic() - started}
    (args.output_dir / "complete.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
