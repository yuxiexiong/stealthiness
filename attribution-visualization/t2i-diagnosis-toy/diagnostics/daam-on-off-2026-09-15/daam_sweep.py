"""Characterize the DAAM on/off perturbation across the exploration split only.

Diagnosis for a failed frozen gate: writes no study assets and never touches the test split.
"""
import os, json
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np, torch
from daam import trace
from diffusers import DDIMScheduler, StableDiffusionPipeline

MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
ROOT = "/root/diagnosis-2026-09-15/attribution-visualization/t2i-diagnosis-toy"
OUT = "/root/claude-daam-sweep.json"
device = "cuda"

torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dtype = torch.float16
pipe = StableDiffusionPipeline.from_pretrained(MODEL, revision=REVISION,
                                               torch_dtype=dtype, use_safetensors=True).to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)

cases = [c for c in json.load(open(ROOT + "/study.json"))["cases"] if c["split"] == "exploration"]
print("exploration cases:", len(cases), flush=True)

rows = []
for n, case in enumerate(cases, 1):
    prompt, seed = case["prompt"], case["seed"]
    noise = torch.randn((1, pipe.unet.config.in_channels, 64, 64),
                        generator=torch.Generator("cpu").manual_seed(seed)).to(device=device, dtype=dtype)

    def run(traced):
        kwargs = dict(num_inference_steps=30, guidance_scale=7.5, eta=0.0, height=512, width=512,
                      latents=noise.clone(), generator=torch.Generator("cpu").manual_seed(seed))
        with torch.inference_mode():
            if traced:
                with trace(pipe):
                    out = pipe(prompt, **kwargs)
            else:
                out = pipe(prompt, **kwargs)
        return np.asarray(out.images[0], dtype=np.float32)

    off, on = run(False), run(True)
    e = np.abs(off - on)
    px = e.max(axis=2)
    row = dict(case_id=case["id"], prompt=prompt, seed=seed,
               mean=round(float(e.mean()), 4), max=float(e.max()),
               px_gt8=int((px > 8).sum()), px_gt32=int((px > 32).sum()), px_total=int(px.size),
               passes_mean_1=bool(e.mean() <= 1.0), passes_max_8=bool(e.max() <= 8.0))
    rows.append(row)
    print(f"[{n}/{len(cases)}] {row['case_id']}  mean={row['mean']}  max={row['max']}  "
          f"px>8={row['px_gt8']}  mean_ok={row['passes_mean_1']}  max_ok={row['passes_max_8']}", flush=True)

means = [r["mean"] for r in rows]
maxes = [r["max"] for r in rows]
summary = dict(
    n=len(rows), split="exploration",
    mean_of_mean=round(float(np.mean(means)), 4), worst_mean=max(means),
    median_max=float(np.median(maxes)), worst_max=max(maxes), best_max=min(maxes),
    cases_passing_mean_1=sum(r["passes_mean_1"] for r in rows),
    cases_passing_max_8=sum(r["passes_max_8"] for r in rows),
    px_gt8_median=float(np.median([r["px_gt8"] for r in rows])),
    px_gt8_worst=max(r["px_gt8"] for r in rows),
)
print(json.dumps(summary, indent=2), flush=True)
json.dump(dict(summary=summary, rows=rows), open(OUT, "w"), indent=2)
print("wrote", OUT, flush=True)
