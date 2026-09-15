"""Calibrate the null for the DAAM on/off pixel check. Diagnosis only: writes no study assets."""
import os, json
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np, torch
from daam import trace
from diffusers import DDIMScheduler, StableDiffusionPipeline

MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
ROOT = "/root/diagnosis-2026-09-15/attribution-visualization/t2i-diagnosis-toy"
device = "cuda"

torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dtype = torch.float16
pipe = StableDiffusionPipeline.from_pretrained(MODEL, revision=REVISION,
                                               torch_dtype=dtype, use_safetensors=True).to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

case = json.load(open(ROOT + "/study.json"))["cases"][0]
prompt, seed = case["prompt"], case["seed"]
words = list(dict.fromkeys(w for o in case["objects"] for w in (o["name"], o["color"])))
noise = torch.randn((1, pipe.unet.config.in_channels, 64, 64),
                    generator=torch.Generator("cpu").manual_seed(seed)).to(device=device, dtype=dtype)
print("case:", case["id"], "| prompt:", prompt, "| seed:", seed, flush=True)


def run(traced):
    kwargs = dict(num_inference_steps=30, guidance_scale=7.5, eta=0.0, height=512, width=512,
                  latents=noise.clone(), generator=torch.Generator("cpu").manual_seed(seed))
    with torch.inference_mode():
        if traced:
            with trace(pipe) as tp:
                out = pipe(prompt, **kwargs)
                heat = tp.compute_global_heat_map(prompt=prompt, normalize=False)
                maps = {w: heat.compute_word_heat_map(w).value.detach().float().cpu().numpy().copy() for w in words}
        else:
            out, maps = pipe(prompt, **kwargs), {}
    return np.asarray(out.images[0], dtype=np.float32), maps


def cmp(a, b):
    e = np.abs(a - b)
    px = e.max(axis=2)
    return dict(mean=round(float(e.mean()), 4), max=float(e.max()),
                px_diff=int((px > 0).sum()), px_gt8=int((px > 8).sum()), px_total=int(px.size))


off1, _ = run(False)
off2, _ = run(False)
on1, m1 = run(True)
on2, m2 = run(True)
print(json.dumps({
    "NULL_off_vs_off": cmp(off1, off2),
    "NULL_on_vs_on": cmp(on1, on2),
    "EFFECT_off_vs_on": cmp(off1, on1),
    "maps_reproducible": bool(all(np.array_equal(m1[w], m2[w]) for w in words)),
}, indent=2), flush=True)
