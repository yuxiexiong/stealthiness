"""Position-matrix algebra shared by eval_m3.py and pred_pl.py (numpy only)."""
import numpy as np

import m3

VARIANTS = {"act": 0, "P100": 1, "g330": 2}


def load_tables(rows_json):
    tmap = {int(k): v for k, v in rows_json["tmap"].items()}
    sigma = {v: k for k, v in tmap.items() if v is not None}
    clip = {int(k): v for k, v in rows_json["clip"].items()}
    return tmap, sigma, clip


def seg_contrib(pm, ii, c, X, tmap, sigma, clip, variant="act", coef_scheme=None, clip_s=None, d100=False):
    """{segment j: contribution} of row ii arriving at step c+1 (position c), full horizon."""
    s = c + 1
    coefs = m3.coef_by_segment(s, tmap, sigma, scheme=coef_scheme)
    cs = clip[clip_s if clip_s is not None else s]
    nb = pm["nb"][ii]
    out = {}
    for j, cf in coefs.items():
        if d100:
            key = f"{ii}|{X}|{j}"
            if key not in pm["d100_all"]:
                continue
            d = pm["d100_all"][key]
        else:
            key = f"{ii}|{c}|{X}|{j}"
            if key not in pm["d"]:
                continue
            d = pm["d"][key][VARIANTS[variant]]
        out[j] = -cf * cs / (2 * nb) * d
    return out


def A_matrix(pm, X, tables, **kw):
    tmap, sigma, clip = tables
    n = len(pm["rows"])
    A = np.zeros((n, len(pm["positions"])))
    for ci, c in enumerate(pm["positions"]):
        for ii in range(n):
            A[ii, ci] = sum(seg_contrib(pm, ii, c, X, tmap, sigma, clip, **kw).values())
    return A


def truncated(segc, t):
    """Sum of segment contributions truncated at step t (linear within the segment containing t)."""
    tot = 0.0
    for j, v in segc.items():
        lo, hi = 10 * j, 10 * j + 10
        if hi <= t:
            tot += v
        elif lo < t:
            tot += v * (t - lo) / 10.0
    return tot


def row_at(pm, ii, s, t, X, tables):
    """Ã(i, s; t): position-matrix contribution of row ii arriving at step s, truncated at t (interpolated
    between neighbouring positions); 0 for rows moved past the ledger (s > 1000)."""
    if s > 1000:
        return 0.0
    tmap, sigma, clip = tables
    pos = pm["positions"]
    c_lo = 10 * ((s - 1) // 10)
    c_lo = min(max(c_lo, pos[0]), pos[-1])
    c_hi = min(c_lo + 10, pos[-1])
    v_lo = truncated(seg_contrib(pm, ii, c_lo, X, tmap, sigma, clip), t)
    if c_hi == c_lo:
        return v_lo
    v_hi = truncated(seg_contrib(pm, ii, c_hi, X, tmap, sigma, clip), t)
    w = ((s - 1) - c_lo) / 10.0
    return (1 - w) * v_lo + w * v_hi
