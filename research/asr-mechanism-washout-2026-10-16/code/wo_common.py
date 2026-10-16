"""Pure logic of the single-row washout measurement (CONTRACT_WO.md). Standard library only.

Clean stream = the mother's own data order with every poison row in its clean version. At each branch point t0
the state is copied and two arms run K updates: P0 (clean stream) and P1 (identical, except the first row of update
t0 + 1 is replaced by one poison row: the mother's first poison row at or after that position, poisoned). Training is
deterministic, so D(k) = c_P1(t0 + k) - c_P0(t0 + k) is the effect of that one row on the shared trigger margin c.
"""
BRANCH = (15, 35, 60, 100)
EARLY, LATE = (15, 35), (60, 100)
K = 30
PEAK_WINDOW = 10           # the peak of D is searched over k = 1..10 (Adam momentum can delay it)
MIN_PUSH = 0.02            # logit; cells with a smaller peak are 'no_push'
FAST = 15                  # half-life (updates) at or below this = washed out
KEEP = 0.5                 # retention D(K) / peak at or above this = kept
DETERMINISM_UPDATES = 5


def pulse_order(clean_order, mother_order, t0, batch=16):
    """Copy of clean_order with the first row of update t0+1 replaced by the next poison row (poisoned form)."""
    pos = t0 * batch
    src = next(i for i in range(pos, len(mother_order)) if mother_order[i]["poison"])
    out = list(clean_order)
    out[pos] = dict(mother_order[src], poison=True)
    return out, src


def response(d):
    """d: list of D(k) for k = 1..K. Returns peak, k_peak, half-life after the peak (None = not reached), retention."""
    win = d[:PEAK_WINDOW]
    peak = max(win)
    k_peak = win.index(peak) + 1
    if peak < MIN_PUSH:
        return {"peak": peak, "k_peak": k_peak, "half_life": None, "retention": None, "status": "no_push"}
    half = next((k - k_peak for k in range(k_peak + 1, len(d) + 1) if d[k - 1] <= 0.5 * peak), None)
    return {"peak": peak, "k_peak": k_peak, "half_life": half, "retention": d[-1] / peak, "status": "ok"}


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return None if n == 0 else (xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2]))


def judge(cells):
    """cells: list of dicts with t0 and the response() fields. Censored half-lives count as K."""
    def hl(c):
        return c["half_life"] if c["half_life"] is not None else K
    early = [c for c in cells if c["t0"] in EARLY and c["status"] == "ok"]
    late = [c for c in cells if c["t0"] in LATE and c["status"] == "ok"]
    out = {"n_early": len(early), "n_late": len(late),
           "median_half_life_early": _median([hl(c) for c in early]),
           "median_half_life_late": _median([hl(c) for c in late]),
           "median_retention_early": _median([c["retention"] for c in early]),
           "median_retention_late": _median([c["retention"] for c in late]),
           "median_peak_early": _median([c["peak"] for c in early]),
           "median_peak_late": _median([c["peak"] for c in late])}
    if len(early) < 2 or len(late) < 2:
        out["tier"] = "unmeasurable"
    elif out["median_retention_early"] >= KEEP:
        out["tier"] = "no_washout"
    elif out["median_half_life_early"] <= FAST and out["median_retention_late"] >= KEEP:
        out["tier"] = "washout_stage"
    elif out["median_half_life_early"] <= FAST and out["median_half_life_late"] <= FAST:
        out["tier"] = "washout_all"
    else:
        out["tier"] = "mixed"
    return out


def gate_from_medians(medians, thr=0.5):
    """Defense fx2 gate on the clean stream: first update whose median per-row loss is < thr."""
    return next((u for u, m in medians if m < thr), None)
