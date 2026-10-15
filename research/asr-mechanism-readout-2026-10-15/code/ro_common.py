"""Frozen rules of CONTRACT_READOUT.md: online exit detector, calibration, predictors, judge. numpy only."""
import numpy as np

START = 15
FIRST_DET = 20
RISE = 3.0
ONSET = 0.10
RATIO = 0.7
MIN_TEST = 5


def trailing(c, k=3):
    c = np.asarray(c, float)
    return np.array([c[max(0, t - k + 1):t + 1].mean() for t in range(len(c))])


def detect(c):
    """Online exit detector on the per-update course c (index = update). Returns dict or None."""
    s = trailing(c)
    m, tau = float("inf"), None
    for t in range(START + 2, len(s)):                 # s(t) averages c(t-2..t): only windows fully inside [START, t]
        if s[t] <= m:                                     # latest time at the bottom: the exit, not the start of the plateau
            m, tau = float(s[t]), t
        if t >= FIRST_DET and s[t] - m >= RISE:
            later = s[t + 1:]
            return {"t_det": t, "t_exit": tau, "D": -m, "staircase": bool((later < m).any())}
    return None


def onset(asr):
    return next((t for t, v in enumerate(asr) if v >= ONSET), None)


def trajectory(gap):
    """gap: (updates x rows). Returns the per-mother record."""
    gap = np.asarray(gap, float)
    c, asr = gap.mean(1), (gap > 0).mean(1)
    det, t_a = detect(c), onset(asr)
    rec = {"t_a": t_a, "det": det, "c_at_onset": float(c[t_a]) if t_a is not None else None}
    if det and t_a is not None and t_a > det["t_exit"]:
        rec["lag"] = t_a - det["t_exit"]
        rec["v"] = (float(c[t_a]) - float(c[det["t_exit"]])) / rec["lag"]
    return rec


def calibrate(recs):
    ok = [r for r in recs if "v" in r]
    if not ok:
        raise ValueError("No calibration mother with detection and onset")
    return {"theta": float(np.mean([r["c_at_onset"] for r in ok])), "v": float(np.mean([r["v"] for r in ok])),
            "L": float(np.mean([r["lag"] for r in ok])), "T": float(np.mean([r["t_a"] for r in recs if r["t_a"] is not None])),
            "n": len(ok)}


def predict(rec, cal):
    if rec["det"] is None:
        return {"model": cal["T"], "no_depth": cal["T"], "no_exit": cal["T"], "detected": False}
    d = rec["det"]
    return {"model": d["t_exit"] + (cal["theta"] + d["D"]) / cal["v"], "no_depth": d["t_exit"] + cal["L"], "no_exit": cal["T"], "detected": True}


def judge(test_recs, cal):
    rows = [r for r in test_recs if r["t_a"] is not None]
    if len(rows) < MIN_TEST:
        return {"tier": "unmeasurable", "n": len(rows)}
    err = {k: [] for k in ("model", "no_depth", "no_exit")}
    for r in rows:
        p = predict(r, cal)
        for k in err:
            err[k].append(abs(p[k] - r["t_a"]))
    mae = {k: float(np.mean(v)) for k, v in err.items()}
    if mae["model"] <= RATIO * mae["no_depth"] and mae["model"] <= RATIO * mae["no_exit"]:
        tier = "supported"
    elif mae["model"] >= mae["no_depth"]:
        tier = "refuted"
    else:
        tier = "partial"
    return {"tier": tier, "n": len(rows), "mae": mae, "errors": err}


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) and np.std(ry) else float("nan")
