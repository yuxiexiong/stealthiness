"""Save DAAM off/on image pairs for the worst and best exploration cases, side by side.

Diagnosis only: written to a scratch directory, not the study output.
"""
import os, json
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np, torch
from PIL import Image, ImageDraw
from daam import trace
from diffusers import DDIMScheduler, StableDiffusionPipeline

MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
ROOT = "/root/diagnosis-2026-09-15/attribution-visualization/t2i-diagnosis-toy"
OUT = "/root/claude-daam-pairs"
WANT = ["color-119-s0", "color-106-s0", "color-072-s0", "color-119-s1"]
device = "cuda"

torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
pipe = StableDiffusionPipeline.from_pretrained(MODEL, revision=REVISION,
                                               torch_dtype=torch.float16, use_safetensors=True).to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)
os.makedirs(OUT, exist_ok=True)

cases = {c["id"]: c for c in json.load(open(ROOT + "/study.json"))["cases"]}
report = []
for cid in WANT:
    case = cases[cid]
    prompt, seed = case["prompt"], case["seed"]
    noise = torch.randn((1, pipe.unet.config.in_channels, 64, 64),
                        generator=torch.Generator("cpu").manual_seed(seed)).to(device=device, dtype=torch.float16)

    def run(traced):
        kwargs = dict(num_inference_steps=30, guidance_scale=7.5, eta=0.0, height=512, width=512,
                      latents=noise.clone(), generator=torch.Generator("cpu").manual_seed(seed))
        with torch.inference_mode():
            if traced:
                with trace(pipe):
                    out = pipe(prompt, **kwargs)
            else:
                out = pipe(prompt, **kwargs)
        return out.images[0]

    off_img, on_img = run(False), run(True)
    off, on = np.asarray(off_img, dtype=np.float32), np.asarray(on_img, dtype=np.float32)
    e = np.abs(off - on)

    # Coarse composition check: does the depicted layout survive, or is it a different picture?
    def coarse(a):
        return np.asarray(Image.fromarray(a.astype(np.uint8)).resize((32, 32), Image.Resampling.BILINEAR),
                          dtype=np.float32)
    c_off, c_on = coarse(off), coarse(on)
    coarse_err = float(np.abs(c_off - c_on).mean())
    corr = float(np.corrcoef(off.ravel(), on.ravel())[0, 1])

    amp = np.clip(e.max(axis=2) * 3, 0, 255).astype(np.uint8)
    diff_img = Image.fromarray(np.stack([amp] * 3, axis=2))
    canvas = Image.new("RGB", (512 * 3 + 20, 512 + 28), "white")
    for i, (img, label) in enumerate(((off_img, "DAAM OFF"), (on_img, "DAAM ON"), (diff_img, "|diff| x3"))):
        canvas.paste(img, (i * (512 + 10), 24))
        ImageDraw.Draw(canvas).text((i * (512 + 10) + 4, 6), label, fill="black")
    ImageDraw.Draw(canvas).text((4, 512 + 26 - 16),
                                f"{cid}  '{prompt}'  mean={e.mean():.3f} max={e.max():.0f} "
                                f"coarse32_mean={coarse_err:.3f} pearson={corr:.5f}", fill="black")
    path = f"{OUT}/{cid}.png"
    canvas.save(path)
    row = dict(case_id=cid, prompt=prompt, mean=round(float(e.mean()), 4), max=float(e.max()),
               coarse32_mean_err=round(coarse_err, 4), pearson_r=round(corr, 6), path=path)
    report.append(row)
    print(json.dumps(row), flush=True)

json.dump(report, open(f"{OUT}/report.json", "w"), indent=2)
print("done", flush=True)
