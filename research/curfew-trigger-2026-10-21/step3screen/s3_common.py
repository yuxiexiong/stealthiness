"""Shared verdict logic for the step-3 screen (frozen with CONTRACT_STEP3)."""
import math, statistics as st

WINDOW = (50, 150)      # margin ramp: A1-A0 margin gap onset 50 (both seeds); 90% hits by 125/155
PRE = (10, 40)          # before the margin gap leaves the null band
HALF = 25               # half-width (steps) of the window used to estimate the A0 cross-seed RMS

def zseries(a0_x, a0_y, a1_x, grid):
    """z(t) = (A1_s - A0_s)(t) / R(t); R(t) = RMS of (A0_1006 - A0_1007) over grid points within +-HALF of t."""
    out = {}
    for t in grid:
        near = [u for u in grid if abs(u - t) <= HALF and u in a0_x and u in a0_y]
        r = math.sqrt(sum((a0_x[u] - a0_y[u]) ** 2 for u in near) / len(near)) if near else float('nan')
        out[t] = (a1_x[t] - a0_x[t]) / r if r > 0 else float('nan')
    return out

def summary(z, lo, hi):
    vals = [v for t, v in z.items() if lo <= t <= hi and not math.isnan(v)]
    return st.median(vals) if vals else float('nan')

def tier(Z1, Z2):
    """Four tiers from the two seeds' median z over WINDOW."""
    if any(math.isnan(z) for z in (Z1, Z2)):
        return "分不清"
    same = (Z1 > 0) == (Z2 > 0)
    if same and min(abs(Z1), abs(Z2)) >= 3:
        return "可辨识（强）"
    if same and min(abs(Z1), abs(Z2)) >= 1.5:
        return "可辨识（弱）"
    if max(abs(Z1), abs(Z2)) < 1:
        return "不可辨识"
    return "分不清"

def verdict(series, grid):
    """series: dict with keys a0_1006, a0_1007, a1_1006, a1_1007 -> {step: value}."""
    z6 = zseries(series['a0_1006'], series['a0_1007'], series['a1_1006'], grid)
    z7 = zseries(series['a0_1007'], series['a0_1006'], series['a1_1007'], grid)
    Z6, Z7 = summary(z6, *WINDOW), summary(z7, *WINDOW)
    P6, P7 = summary(z6, *PRE), summary(z7, *PRE)
    return {"Z_window": [Z6, Z7], "Z_pre": [P6, P7], "tier": tier(Z6, Z7),
            "z": {"1006": z6, "1007": z7}}
