"""IM2 (amended before any data, LOG J42): does the trigger-related effect strengthen before the logit?

Method B (occlusion, engine.occlusion) stores, for every 2x2 window, T3(full) - T3(window grayed);
the four patches of a window carry the same value. Readouts per imaged step, medians over the 60
trajectory samples:
  E_trig   signed T3 drop when the trigger window [550, 551, 574, 575] is grayed (primary)
  m1       share of positive relevance on the trigger window (the original IM2 readout)
  num, den m1's numerator (positive trigger relevance) and denominator (total positive relevance)
  E_other  mean signed drop over the other 143 windows;  E_center  the fixed centre window (300, 301, 324, 325)
  logit    T3
Steps: 200, 220, ... up to the first grid step >= max(t50, t50s) + 60, where t50 = first ASR >= 0.5 and
t50s = first ASR >= 0.5 with every measured point in the next 40 steps also >= 0.5 (10-step grid).
Clean image (no trigger) is imaged at three steps per run: 220, the last step <= t50 - 20, and the first
step >= t50s + 20; it gives the same-location occlusion effect without the trigger (specificity).
"""
import numpy as np

TRIG = [550, 551, 574, 575]
CENTER = [300, 301, 324, 325]
WIN_IDS = [wy * 24 + wx for wy in range(0, 24, 2) for wx in range(0, 24, 2)]   # one id per 2x2 window
OTHER = [i for i in WIN_IDS if i != 550]
BASE_STEPS = (200, 220, 240)
LEAD_MIN = 20
SIGN_MIN = {14: 11, 8: 7}          # consistency thresholds only (runs share a few orders; no independence p-value, LOG J43)
MAX_MISSING = {14: 2, 8: 1}
SEC_LIMIT = 360                    # gate-measured seconds per checkpoint above which only the 8 XO runs are imaged
SPEC_MAX = 0.5                     # median over runs of (clean-image change) / (trigger-image change) of E_trig
INC_FLOOR, INC_SD = 0.25, 3.0      # a readout "strengthens" only if end - baseline > max(INC_FLOOR, INC_SD * sd(baseline)) (LOG J43)
SPEC_MIN_FRAC = 0.75               # valid specificity ratios needed among the E-leading runs (LOG J43)


def steps_for(t50, t50s):
    end = int(np.ceil((max(t50, t50s) + 60) / 20.0) * 20)
    return list(range(200, end + 1, 20))


def clean_steps(t50, t50s):
    st = steps_for(t50, t50s)
    pre = max(s for s in st if s <= t50 - 20)
    post = min(s for s in st if s >= t50s + 20)
    return [220, pre, post]


def readouts(occ):
    r = np.asarray(occ, float)
    pos = np.clip(r, 0, None)
    den = float(pos.sum())
    num = float(pos[TRIG].sum())
    return {"E_trig": float(r[550]), "m1": num / den if den > 0 else float("nan"), "num": num, "den": den,
            "E_other": float(np.mean(r[OTHER])), "E_center": float(r[300])}


def m1_b(occ):
    return readouts(occ)["m1"]


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


def increase_ok(steps, y, floor=INC_FLOOR):
    """True when the readout actually strengthens: end - baseline is positive and beyond the noise allowance."""
    steps = list(steps)
    b = [float(y[steps.index(s)]) for s in BASE_STEPS if s in steps]
    if len(b) < len(BASE_STEPS):
        return False
    rise = float(y[-1]) - float(np.mean(b))
    return rise > max(floor, INC_SD * float(np.std(b)))


def _rule(runs, key, n_runs):
    """runs: {run: {"lead_<key>": float|None, "inc_<key>": bool}}. A run counts as leading only if its readout
    strengthens and its lead > 0; as lagging only if it strengthens and lead < 0; otherwise neither."""
    rd = {r: v for r, v in runs.items() if v.get(f"lead_{key}") is not None}
    if len(rd) < n_runs - MAX_MISSING[n_runs]:
        return None, {"n": len(rd)}
    ok = {r: v[f"lead_{key}"] for r, v in rd.items() if v.get(f"inc_{key}")}
    pos = sorted(r for r, x in ok.items() if x > 0)
    neg = sorted(r for r, x in ok.items() if x < 0)
    med = float(np.median(list(ok.values()))) if ok else float("nan")
    lead = len(pos) >= SIGN_MIN[n_runs] and med >= LEAD_MIN
    lag = len(neg) >= SIGN_MIN[n_runs] and med <= -LEAD_MIN
    return ("lead" if lead else "lag" if lag else "none"), {
        "n_readable": len(rd), "n_strengthening": len(ok), "not_strengthening": sorted(set(rd) - set(ok)),
        "positive": len(pos), "negative": len(neg), "median_lead_strengthening": med, "leading_runs": pos}


def verdict(runs, n_runs=14):
    """runs: {run: {"lead_E", "inc_E", "lead_m1", "inc_m1", "spec"}} (spec = clean change / trigger change or None)."""
    rE, sE = _rule(runs, "E", n_runs)
    rM, sM = _rule(runs, "m1", n_runs)
    if rE is None:
        return {"tier": f"cannot measure (fewer than {n_runs - MAX_MISSING[n_runs]} readable runs)", "E": sE}
    spec_info = None
    if rE == "lead":
        lead_runs = sE["leading_runs"]
        valid = {r: runs[r]["spec"] for r in lead_runs
                 if runs[r].get("spec") is not None and np.isfinite(runs[r]["spec"])}
        enough = len(valid) >= int(np.ceil(SPEC_MIN_FRAC * len(lead_runs)))
        med = float(np.median(list(valid.values()))) if valid else None
        specific = enough and med is not None and med <= SPEC_MAX
        spec_info = {"leading_runs": len(lead_runs), "valid": len(valid), "enough": enough,
                     "excluded": sorted(set(lead_runs) - set(valid)), "median_ratio": med, "specific": specific}
        tier = ("trigger-related effect strengthens before the logit" if specific
                else "trigger effect strengthens early, specificity not confirmed")
    elif rE == "lag":
        tier = "trigger effect lags the logit"
    elif rM == "lead":
        tier = "only the attribution allocation changes early"
    else:
        tier = "no consistent lead"
    return {"tier": tier, "E": sE, "m1": sM, "specificity": spec_info}
