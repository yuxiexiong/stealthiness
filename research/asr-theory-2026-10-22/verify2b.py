"""#2 redone: optimal two-way additive fit, and a MODEL-FREE band measurement."""
import glob, json, re
import numpy as np

def load(arm):
    fs = sorted(glob.glob(f"/workspace/claude-jump/lp_out/{arm}@s*.json"),
                key=lambda f: int(re.search(r"@s(\d+)\.json$", f).group(1)))
    steps, M = [], []
    for f in fs:
        rec = json.load(open(f))
        steps.append(int(re.search(r"@s(\d+)\.json$", f).group(1)))
        M.append([c["M"] for c in sorted(rec["core"], key=lambda c: c["idx"])])
    return np.array(steps), np.array(M)

print("A. optimal two-way additive fit  M_it ~ mu + a_t + b_i   (share = 1 - SSresid/SS[M-a_t])")
print("B. MODEL-FREE: per-probe first zero crossing t_i, the common term m=mean_i M at that moment,")
print("   band := 10-90 pct spread of those m values; travel := m range; apparent jump width in steps")
for arm in ("PS-P0", "XO-I1001-O1007", "XO-I1001-O1009", "XO-I1001-O1008"):
    steps, M = load(arm)
    if M.size == 0:
        print(f"  {arm}: none"); continue
    a = M.mean(1); b = (M - a[:, None]).mean(0)
    resid = M - a[:, None] - b[None, :]
    base = M - a[:, None]
    share = 1 - resid.var() / base.var()
    # window-restricted: steps where mean margin is within +-6 logit of 0 (the crossing region)
    w = np.abs(a) <= 6
    share_w = 1 - resid[w].var() / base[w].var() if w.sum() > 3 else float("nan")
    # model-free band
    ms = []
    for i in range(M.shape[1]):
        y = M[:, i]
        k = np.flatnonzero((y[:-1] < 0) & (y[1:] >= 0))
        if k.size:
            j = k[0]; f = -y[j] / (y[j + 1] - y[j])
            ms.append(a[j] + f * (a[j + 1] - a[j]))
    ms = np.array(ms)
    band = np.percentile(ms, 90) - np.percentile(ms, 10) if ms.size > 10 else float("nan")
    travel = a.max() - a.min()
    # how many STEPS does the 10-90 band correspond to, at the local speed of a(t) near a=0
    near = np.flatnonzero(np.abs(a) <= 4)
    spd = np.polyfit(steps[near], a[near], 1)[0] if near.size > 2 else float("nan")
    print(f"  {arm:16} share_all={share:.3f} share_in_crossing_window={share_w:.3f} | "
          f"crossed={ms.size}/{M.shape[1]} band(10-90)={band:.2f} travel={travel:.1f} band/travel={band/travel:.3f} "
          f"| m speed near 0 = {spd:.3f} logit/step -> band = {band/spd:.0f} steps")
