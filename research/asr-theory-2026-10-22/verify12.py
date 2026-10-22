"""Ledger #3 (actual beta1/beta2) and #2 (theta_i stability). Zero GPU, existing Oct-2 data only."""
import glob, json, re, sys
import numpy as np

print("== #3 the optimizer constants the runs actually used")
y = sorted(glob.glob("/workspace/claude-jump/xo/runs/configs/PS-P0.yaml"))
cfg = open(y[0]).read() if y else ""
keys = [l for l in cfg.splitlines() if "adam" in l.lower() or "beta" in l.lower() or "optim" in l.lower()]
print(f"  yaml {y[0] if y else '(none)'}: adam/beta/optim lines -> {keys if keys else 'NONE (so HF defaults apply)'}")
from transformers import TrainingArguments
import inspect
sig = inspect.signature(TrainingArguments.__init__)
for k in ("optim", "adam_beta1", "adam_beta2", "adam_epsilon"):
    print(f"  transformers default {k} = {sig.parameters[k].default}")
ta = [p for p in glob.glob("/workspace/claude-jump/xo/runs/arms/PS-P0/checkpoint-*/training_args.bin")][:1]
print(f"  training_args.bin present: {bool(ta)}" + (f" ({ta[0]})" if ta else ""))
if ta:
    import torch
    a = torch.load(ta[0], weights_only=False)
    print(f"  RECORDED: optim={a.optim}, beta1={a.adam_beta1}, beta2={a.adam_beta2}, eps={a.adam_epsilon}, "
          f"lr={a.learning_rate}, warmup_ratio={a.warmup_ratio}, sched={a.lr_scheduler_type}")

print()
print("== #2 additive decomposition and flip-point stability, per LLaVA run (p_core, 200 probes)")
print("   m(t) := median_i M_i(t);  b_i := mean_t [M_i(t) - m(t)];  theta_i := -b_i")
print("   additive share := 1 - Var(resid)/Var(M - m) ;  split-half = corr of b_i from early vs late halves")
for arm in ("PS-P0", "XO-I1001-O1007", "XO-I1001-O1009", "XO-I1001-O1008"):
    fs = sorted(glob.glob(f"/workspace/claude-jump/lp_out/{arm}@s*.json"),
                key=lambda f: int(re.search(r"@s(\d+)\.json$", f).group(1)))
    if not fs:
        print(f"  {arm}: no files"); continue
    steps, M = [], []
    for f in fs:
        rec = json.load(open(f))
        steps.append(int(re.search(r"@s(\d+)\.json$", f).group(1)))
        M.append([c["M"] for c in sorted(rec["core"], key=lambda c: c["idx"])])
    M = np.array(M)                       # (T, n)
    m = np.median(M, 1)
    C = M - m[:, None]                    # centred
    b = C.mean(0)
    resid = C - b[None, :]
    share = 1 - resid.var() / C.var()
    h = len(steps) // 2
    b_e, b_l = C[:h].mean(0), C[h:].mean(0)
    r = float(np.corrcoef(b_e, b_l)[0, 1])
    drift = float(np.abs(b_e - b_l).mean())
    band = float(np.percentile(-b, 90) - np.percentile(-b, 10))
    travel = float(m.max() - m.min())
    print(f"  {arm:16} T={len(steps):2d} additive_share={share:.3f} split_half_corr(b)={r:.3f} "
          f"mean|b_early-b_late|={drift:.2f} logit | theta 10-90 band={band:.1f} logit, m travel={travel:.1f} -> band/travel={band/travel:.3f}")
