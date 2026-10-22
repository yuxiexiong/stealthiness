"""Zero-GPU order statistics the theory trunk needs (S1 overlap regime, S3 channel exclusion).
Run on the server from /tmp/sf_check_ba6b (has sf.py, export_order.py)."""
import json, sys
import numpy as np
sys.path.insert(0, "/tmp/sf_check_ba6b")
import sf
from export_order import randperm

lf = "/root/attribution-microscope/data/lf"
rows = json.load(open(lf + "/arm_p_1_0.json")); clean = json.load(open(lf + "/arm_clean.json"))
poison = {i for i, (x, y) in enumerate(zip(rows, clean)) if x != y}
P = np.zeros(sf.N, bool); P[list(poison)] = True
BETA1, TOT = 0.9, 1250
TM0 = {1001: 327.9, 1007: 464.5, 1009: 328.1, 2001: None}

print(f"poison rows {len(poison)} of {sf.N}; steps {TOT}; batch {sf.BATCH}")
print(f"mean poison per step {len(poison)/TOT:.3f}; mean spacing {TOT/len(poison):.2f} steps; "
      f"AdamW beta1 {BETA1} -> memory 1/(1-b1) = {1/(1-BETA1):.0f} steps")
print()
for s in (1001, 1007, 1009, 2001):
    r = randperm(s)
    n_per_step = np.zeros(TOT + 1, int)
    for pos in np.flatnonzero(P[r]):
        n_per_step[min(int(pos) // sf.BATCH + 1, TOT)] += 1
    hit = np.flatnonzero(n_per_step > 0)
    gaps = np.diff(hit)
    # momentum buffer poison content: EMA of arrivals (this is literally what beta1 carries)
    ema = np.zeros(TOT + 1)
    for t in range(1, TOT + 1):
        ema[t] = BETA1 * ema[t - 1] + (1 - BETA1) * n_per_step[t]
    cov = {k: float(np.mean(ema[1:k + 1] > 0.5 * (1 - BETA1) * len(poison) / TOT)) for k in (300,)}
    print(f"order {s}  (t_M0 {TM0[s]})")
    print(f"  steps containing poison: {len(hit)} of {TOT} ({len(hit)/TOT:.1%});  gaps mean {gaps.mean():.2f} "
          f"median {np.median(gaps):.0f} max {gaps.max()}; frac of gaps <= 10 steps: {np.mean(gaps<=10):.2%}")
    print(f"  cumulative poison by step 100/200/300/465: "
          f"{n_per_step[:101].sum()}/{n_per_step[:201].sum()}/{n_per_step[:301].sum()}/{n_per_step[:466].sum()}")
    print(f"  momentum-weighted exposure (EMA, beta1={BETA1}) mean over steps 1-100/1-300: "
          f"{ema[1:101].mean():.4f}/{ema[1:301].mean():.4f}  (steady-state {len(poison)/TOT*(1):.4f})")
    print(f"  LR-weighted dose by step 300 (cosine, 38 warmup): ", end="")
    t = np.arange(1, TOT + 1)
    lr = np.where(t <= 38, t / 38.0, 0.5 * (1 + np.cos(np.pi * (t - 38) / (TOT - 38))))
    print(f"{float((lr[:300] * n_per_step[1:301]).sum()):.2f}")
print()
print("== spread of the three candidate rho-channel predictors across the three orders with known t_M0")
for name, f in (("cum count @300", lambda ns: ns[:301].sum()),
                ("cum count @465", lambda ns: ns[:466].sum()),
                ("EMA mean 1-300", None)):
    vals = {}
    for s in (1001, 1007, 1009):
        r = randperm(s)
        ns = np.zeros(TOT + 1, int)
        for pos in np.flatnonzero(P[r]):
            ns[min(int(pos) // sf.BATCH + 1, TOT)] += 1
        if f is None:
            ema = np.zeros(TOT + 1)
            for t in range(1, TOT + 1):
                ema[t] = BETA1 * ema[t - 1] + (1 - BETA1) * ns[t]
            vals[s] = float(ema[1:301].mean())
        else:
            vals[s] = float(f(ns))
    order_by_pred = sorted(vals, key=vals.get, reverse=True)
    print(f"  {name:16}: {vals}  -> more-exposure-first {order_by_pred}; actual earliest-first [1001, 1009, 1007]")
