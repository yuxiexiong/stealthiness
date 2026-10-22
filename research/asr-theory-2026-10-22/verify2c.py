"""S0 tested on the EVENT ASR actually counts: argmax_is_target per probe. Model-free."""
import glob, json, re
import numpy as np

for arm in ("PS-P0", "XO-I1001-O1007", "XO-I1001-O1009", "XO-I1001-O1008"):
    fs = sorted(glob.glob(f"/workspace/claude-jump/lp_out/{arm}@s*.json"),
                key=lambda f: int(re.search(r"@s(\d+)\.json$", f).group(1)))
    if not fs: continue
    steps, M, AM = [], [], []
    for f in fs:
        rec = json.load(open(f)); cs = sorted(rec["core"], key=lambda c: c["idx"])
        steps.append(int(re.search(r"@s(\d+)\.json$", f).group(1)))
        M.append([c["M"] for c in cs]); AM.append([bool(c["argmax_is_target"]) for c in cs])
    steps = np.array(steps); M = np.array(M); AM = np.array(AM)
    m = M.mean(1)
    asr = AM.mean(1)
    # per-probe first step at which it becomes the argmax
    first = []
    for i in range(AM.shape[1]):
        k = np.flatnonzero(AM[:, i])
        first.append(steps[k[0]] if k.size else np.nan)
    first = np.array(first, float)
    got = ~np.isnan(first)
    q10, q50, q90 = np.nanpercentile(first, [10, 50, 90])
    # ASR window measured on the same LP grid
    def cross(th):
        k = np.flatnonzero(asr >= th)
        return steps[k[0]] if k.size else np.nan
    w10, w50, w90 = cross(0.1), cross(0.5), cross(0.9)
    # band in logit units at those crossing moments
    mi = np.interp(first[got], steps, m)
    bandl = np.percentile(mi, 90) - np.percentile(mi, 10)
    print(f"{arm}")
    print(f"  probes becoming argmax: {int(got.sum())}/{len(first)};  per-probe first-argmax step  "
          f"p10={q10:.0f} p50={q50:.0f} p90={q90:.0f}  -> 10-90 spread = {q90-q10:.0f} steps")
    print(f"  ASR(t) on the LP grid: 10% at {w10}, 50% at {w50}, 90% at {w90}  -> 10-90 window = "
          f"{(w90-w10) if not (np.isnan(w90) or np.isnan(w10)) else float('nan')} steps")
    print(f"  band in logit at the argmax events: 10-90 = {bandl:.2f};  m travel = {m.max()-m.min():.1f} "
          f"-> band/travel = {bandl/(m.max()-m.min()):.3f}")
    sel = (steps >= (w50 or 0) - 60) & (steps <= (w50 or 0) + 60)
    if sel.sum() > 2:
        sp = np.polyfit(steps[sel], m[sel], 1)[0]
        print(f"  m speed within +-60 steps of ASR50: {sp:.3f} logit/step -> band implies "
              f"{bandl/sp:.0f} steps, observed 10-90 window {q90-q10:.0f} steps")
