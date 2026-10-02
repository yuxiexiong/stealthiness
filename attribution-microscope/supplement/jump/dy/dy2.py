"""DY2: early warning from the leading tail of the per-probe margin distribution (CONTRACT_DY2.md).

Same selection rule and blind test as dy.py; only the features change. numpy only.
Input per run: {step: array of the 200 p_core margins M} for read-out steps <= dy.H.
"""
import numpy as np

import dy

METHODS = ("q90_level", "lead_frac", "q90_slope")
LEAD = -2.0                      # a probe is "leading" when M > -2 (within 2 logits of flipping)


def features(per_step):
    """per_step: {step: np.array of M}. Uses steps in dy.WIN."""
    s = sorted(k for k in per_step if dy.WIN[0] <= k <= dy.WIN[1])
    if len(s) < 3:
        return None
    q90 = np.array([np.quantile(per_step[k], 0.9) for k in s])
    b, _ = np.polyfit(np.array(s, float), q90, 1)
    last = s[-1]
    return {"q90_level": float(q90[-1]),
            "lead_frac": float(np.mean(np.asarray(per_step[last]) > LEAD)),
            "q90_slope": float(b)}


def fit(train, method):
    x = np.array([f[method] for f, _ in train])
    y = np.array([t for _, t in train], float)
    if x.std() == 0:
        return {"method": method, "a": float(y.mean()), "b": 0.0}
    b, a = np.polyfit(x, y, 1)
    return {"method": method, "a": float(a), "b": float(b)}


def predict(model, f):
    return model["a"] + model["b"] * f[model["method"]]


def loo(train, method):
    return float(np.mean([abs(predict(fit(train[:i] + train[i + 1:], method), train[i][0]) - train[i][1])
                          for i in range(len(train))]))


def select(train):
    scores = {m: loo(train, m) for m in METHODS}
    const = float(np.mean([abs(np.mean([t for j, (_, t) in enumerate(train) if j != i]) - train[i][1])
                           for i in range(len(train))]))
    return min(METHODS, key=lambda m: (scores[m], METHODS.index(m))), scores, const


def verdict(model, const_value, test):
    test = [(f, t) for f, t in test if f is not None and t is not None]
    if len(test) < dy.MIN_TEST:
        return {"n": len(test), "tier": "cannot measure"}
    pe = [abs(predict(model, f) - t) for f, t in test]
    ce = [abs(const_value - t) for _, t in test]
    mp, mc = float(np.mean(pe)), float(np.mean(ce))
    ratio = mp / mc if mc > 0 else float("inf")
    tier = ("take-off predictable from the leading tail" if ratio <= dy.RATIO_PASS
            else "not predictable" if ratio >= dy.RATIO_FAIL else "unclear")
    return {"n": len(test), "mae_pred": mp, "mae_const": mc, "ratio": ratio, "tier": tier,
            "pred": [float(predict(model, f)) for f, _ in test], "t50": [t for _, t in test]}
