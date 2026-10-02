"""Early warning of the take-off from the pre-take-off logit margin (CONTRACT_DY.md). numpy only.

Input per run: steps and the median p_core margin M (LP read-out), plus t50.
Only read-out steps <= H are used for prediction (every known run takes off after step 320).
"""
import numpy as np

H = 300
WIN = (200, 300)                 # window for level / slope features
METHODS = ("extrapolate", "level", "slope")
RATIO_PASS, RATIO_FAIL = 0.5, 1.0
MIN_TEST = 3


def _window(steps, med):
    s = np.asarray(steps, float)
    m = np.asarray(med, float)
    k = (s >= WIN[0]) & (s <= WIN[1])
    return s[k], m[k]


def features(steps, med):
    """Level = median M at the last read-out <= H; slope = least-squares slope over WIN;
    extrapolated crossing = step where the WIN line reaches M = 0 (inf if slope <= 0)."""
    s, m = _window(steps, med)
    if len(s) < 3:
        return None
    b, a = np.polyfit(s, m, 1)
    level = float(m[s <= H][-1])
    cross = float(-a / b) if b > 0 else float("inf")
    return {"level": level, "slope": float(b), "extrapolate": cross}


def fit(train, method):
    """train: list of (features, t50). Returns a predictor feature -> t50."""
    if method == "extrapolate":
        x = np.array([f["extrapolate"] for f, _ in train])
        y = np.array([t for _, t in train], float)
        ok = np.isfinite(x)
        off = float(np.median(y[ok] - x[ok])) if ok.any() else 0.0
        fallback = float(np.mean(y))
        return {"method": method, "offset": off, "fallback": fallback}
    x = np.array([f[method] for f, _ in train])
    y = np.array([t for _, t in train], float)
    b, a = np.polyfit(x, y, 1)
    return {"method": method, "a": float(a), "b": float(b)}


def predict(model, f):
    if model["method"] == "extrapolate":
        x = f["extrapolate"]
        return x + model["offset"] if np.isfinite(x) else model["fallback"]
    return model["a"] + model["b"] * f[model["method"]]


def loo(train, method):
    """Leave-one-out mean absolute error of a method on the training runs."""
    err = []
    for i in range(len(train)):
        m = fit(train[:i] + train[i + 1:], method)
        err.append(abs(predict(m, train[i][0]) - train[i][1]))
    return float(np.mean(err))


def select(train):
    """Pre-registered rule: the method with the smallest LOO MAE (ties: METHODS order)."""
    scores = {m: loo(train, m) for m in METHODS}
    const = float(np.mean([abs(np.mean([t for j, (_, t) in enumerate(train) if j != i]) - train[i][1])
                           for i in range(len(train))]))
    best = min(METHODS, key=lambda m: (scores[m], METHODS.index(m)))
    return best, scores, const


def verdict(model, const_value, test):
    """test: list of (features, t50). Blind test against the constant (training-mean) prediction."""
    test = [(f, t) for f, t in test if f is not None and t is not None]
    if len(test) < MIN_TEST:
        return {"n": len(test), "tier": "cannot measure"}
    pe = [abs(predict(model, f) - t) for f, t in test]
    ce = [abs(const_value - t) for _, t in test]
    mp, mc = float(np.mean(pe)), float(np.mean(ce))
    ratio = mp / mc if mc > 0 else float("inf")
    tier = ("take-off predictable from the pre-take-off margin" if ratio <= RATIO_PASS
            else "not predictable" if ratio >= RATIO_FAIL else "unclear")
    return {"n": len(test), "mae_pred": mp, "mae_const": mc, "ratio": ratio, "tier": tier,
            "pred": [float(predict(model, f)) for f, _ in test], "t50": [t for _, t in test]}
