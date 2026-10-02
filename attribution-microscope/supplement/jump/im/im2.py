"""IM2: does the trigger-region attribution share (method B) rise before the logit? (CONTRACT_IM2.md)

Per run: imaged steps 200, 220, ... up to the first step >= t50 + 60 (so a lag is also visible).
Series (medians over the 60 trajectory samples): m1 = method-B trigger share of T3; logit = T3 logit.
Progress: (value - baseline) / (end - baseline), baseline = mean of steps 200-240, end = value at the last
imaged step. t_half = first crossing of 0.5 (linear interpolation). lead = t_half(logit) - t_half(m1).
"""
import numpy as np

TRIG = [550, 551, 574, 575]
BASE_STEPS = (200, 220, 240)
LEAD_MIN = 20
SIGN_MIN = {14: 11, 8: 7}          # one-sided sign test p ~ 0.03 in both plans
MAX_MISSING = {14: 2, 8: 1}
SEC_LIMIT = 360                    # gate-measured seconds per checkpoint above which only the 8 XO runs are imaged


def steps_for(t50):
    end = int(np.ceil((t50 + 60) / 20.0) * 20)
    return list(range(200, end + 1, 20))


def m1_b(occ_map):
    r = np.clip(np.asarray(occ_map, float), 0, None)
    t = r.sum()
    return float(r[TRIG].sum() / t) if t > 0 else float("nan")


def t_half(steps, y):
    steps = list(steps)
    y = np.asarray(y, float)
    base = float(np.mean([y[steps.index(s)] for s in BASE_STEPS if s in steps]))
    end = float(y[-1])
    if end == base:
        return None
    p = (y - base) / (end - base)
    for (s0, p0), (s1, p1) in zip(zip(steps, p), zip(steps[1:], p[1:])):
        if s0 >= BASE_STEPS[-1] and p0 < 0.5 <= p1:
            return s0 + (s1 - s0) * (0.5 - p0) / (p1 - p0)
    return None


def verdict(leads, n_runs=14):
    """leads: {run: lead or None}; n_runs: 14 (all runs) or 8 (XO only, if imaging is slow)."""
    v = [x for x in leads.values() if x is not None]
    if len(v) < n_runs - MAX_MISSING[n_runs]:
        return {"n": len(v), "tier": f"cannot measure (fewer than {n_runs - MAX_MISSING[n_runs]} readable runs)"}
    pos, neg = int(sum(x > 0 for x in v)), int(sum(x < 0 for x in v))
    med = float(np.median(v))
    if pos >= SIGN_MIN[n_runs] and med >= LEAD_MIN:
        tier = "attribution leads the logit"
    elif neg >= SIGN_MIN[n_runs] and med <= -LEAD_MIN:
        tier = "attribution lags the logit"
    else:
        tier = "no consistent lead"
    return {"n": len(v), "positive": pos, "negative": neg, "median_lead": med, "tier": tier}
