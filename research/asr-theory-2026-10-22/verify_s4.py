"""S4's load-bearing number, measured on ONE grid in-family: plateau vs window speed of m(t),
and whether m accelerates through the plateau."""
import glob, json, re
import numpy as np

print("m(t) := mean_i margin over the 200 p_core probes; ASR50 := first LP step with >=50% argmax")
for arm in ("PS-P0", "XO-I1001-O1007", "XO-I1001-O1009", "XO-I1001-O1008"):
    fs = sorted(glob.glob(f"/workspace/claude-jump/lp_out/{arm}@s*.json"),
                key=lambda f: int(re.search(r"@s(\d+)\.json$", f).group(1)))
    if not fs: continue
    steps, M, AM = [], [], []
    for f in fs:
        rec = json.load(open(f)); cs = sorted(rec["core"], key=lambda c: c["idx"])
        steps.append(int(re.search(r"@s(\d+)\.json$", f).group(1)))
        M.append([c["M"] for c in cs]); AM.append([bool(c["argmax_is_target"]) for c in cs])
    steps = np.array(steps); m = np.array(M).mean(1); asr = np.array(AM).mean(1)
    k = np.flatnonzero(asr >= 0.5); t50 = steps[k[0]] if k.size else None
    def spd(lo, hi):
        s = (steps >= lo) & (steps <= hi)
        return np.polyfit(steps[s], m[s], 1)[0] if s.sum() > 2 else float("nan")
    pl_a, pl_b = spd(20, 120), spd(140, 240)                      # early and late plateau
    win = spd(t50 - 30, t50 + 30)
    print(f"{arm:16} t50={t50}")
    print(f"   plateau speed  steps 20-120: {pl_a:+.4f}   steps 140-240: {pl_b:+.4f}  logit/step "
          f"(late/early = {pl_b/pl_a:.2f})")
    print(f"   window speed   t50+-30: {win:+.4f} logit/step    window/late-plateau = {win/pl_b:.2f}")
    print(f"   m at step 20 / 120 / 240 / t50: {np.interp([20,120,240,t50], steps, m).round(2)}")
