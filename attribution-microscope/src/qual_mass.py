"""Mass-based instrument qualification — CRITERION FROZEN BEFORE MEASUREMENT.

Why a second criterion at all: the existing pointing game scores the argmax
patch, but not one downstream metric uses an argmax. M1, M8, M4, M5 and M7 are
all mass shares and M3 is a top-50 overlap. Qualifying on argmax while
measuring with mass is a criterion/measurement mismatch, and it penalises
high-frequency maps (instrument A, 14px patches) relative to inherently smooth
ones (instrument B, 28px occlusion windows): a single outlier patch fails the
argmax test even when most of the mass sits on the object.

FROZEN DEFINITION (written before this was ever run, decisions.log D25):

    localization ratio  L = (mass inside GT box / total positive mass)
                            / (box area / image area)

    L = 1.0  attribution spread like chance
    L >= 2.0 QUALIFIED  (the box holds twice the mass its area would earn;
                         with boxes at >=20% of the image that means >=40% of
                         the mass in 20% of the area)

    fail-side demo: cascading weight randomization must pull L back to ~1.0

The original argmax pointing result stands and is reported alongside this one.
If the two criteria disagree, that disagreement is a finding about the
instrument, not a reason to pick the friendlier number.
"""
import argparse

import numpy as np
from PIL import Image

from common import CFG, DATA, RUNS, read_json, write_json, log

PATCH = CFG["model"]["patch_px"]
GRID = CFG["model"]["grid"]
QUALIFY_AT = 2.0          # frozen threshold
CHANCE = 1.0


def box_patch_weights(box336):
    """Fraction of each patch's area covered by the GT box -> (576,).
    Partial coverage is counted proportionally so the ratio is not distorted
    by where box edges happen to fall."""
    x0, y0, x1, y1 = box336
    w = np.zeros((GRID, GRID), dtype=np.float64)
    for r in range(GRID):
        py0, py1 = r * PATCH, (r + 1) * PATCH
        oy = max(0.0, min(py1, y1) - max(py0, y0))
        if oy <= 0:
            continue
        for c in range(GRID):
            px0, px1 = c * PATCH, (c + 1) * PATCH
            ox = max(0.0, min(px1, x1) - max(px0, x0))
            if ox > 0:
                w[r, c] = (ox * oy) / (PATCH * PATCH)
    return w.reshape(-1)


def localization_ratio(m576, box336):
    m = np.clip(np.asarray(m576, dtype=np.float64), 0, None)
    tot = m.sum()
    if tot <= 0:
        return np.nan
    wt = box_patch_weights(box336)
    area_frac = wt.sum() / (GRID * GRID)
    if area_frac <= 0:
        return np.nan
    return float((m * wt).sum() / tot / area_frac)


def measure(sess, rows, probe_dir, width, scalars, tid):
    vals = {s: {"A": [], "B": []} for s in scalars}
    for r in rows:
        img = Image.open(probe_dir / f"{r['idx']:0{width}d}.jpg")
        cid = sess.first_subtoken(r["category"])
        res = sess.attribute(img, r["question"], tid, cid)
        occ = sess.occlusion(img, r["question"], tid, cid)
        for s in scalars:
            vals[s]["A"].append(localization_ratio(res[s]["A_img_signed"], r["box336"]))
            vals[s]["B"].append(localization_ratio(occ["img"][s], r["box336"]))
    return vals


def summarize(vals, scalars):
    out = {}
    for s in scalars:
        out[s] = {}
        for i in ("A", "B"):
            v = np.array([x for x in vals[s][i] if np.isfinite(x)])
            med = float(np.median(v)) if len(v) else float("nan")
            # bootstrap CI of the median, seeded
            rng = np.random.default_rng(99)
            boot = [np.median(rng.choice(v, len(v), replace=True))
                    for _ in range(1000)] if len(v) else [np.nan]
            out[s][i] = {"median_L": med, "n": int(len(v)),
                         "ci95": [float(np.percentile(boot, 2.5)),
                                  float(np.percentile(boot, 97.5))],
                         "frac_above_chance": float((v > CHANCE).mean()) if len(v) else None,
                         "qualified": bool(med >= QUALIFY_AT)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="p_instrument_xl")
    ap.add_argument("--randomized", action="store_true",
                    help="fail-side demo: randomize weights first")
    ap.add_argument("--out", default="qual_mass.json")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    from attribution.engine import LlavaSession
    rows = read_json(DATA / "manifests" / f"{a.set}.json")
    probe_dir = DATA / "probes" / a.set
    width = 3 if a.set.endswith("_xl") else 2
    target = read_json(DATA / "manifests" / "target_word.json")
    scalars = CFG["imaging"]["scalars"]

    sess = LlavaSession(device=a.device)
    if a.randomized:
        sess.randomize_for_sanity()
    vals = measure(sess, rows, probe_dir, width, scalars, target["first_subtoken"])
    res = summarize(vals, scalars)
    out = {"criterion": "localization_ratio", "threshold": QUALIFY_AT,
           "chance": CHANCE, "set": a.set, "n": len(rows),
           "randomized": bool(a.randomized), "by_scalar": res}
    write_json(RUNS / a.out, out)
    for s in scalars:
        log(f"massqual{' [RANDOMIZED]' if a.randomized else ''} {s}: "
            f"A L={res[s]['A']['median_L']:.2f} {res[s]['A']['ci95']} "
            f"qual={res[s]['A']['qualified']} | "
            f"B L={res[s]['B']['median_L']:.2f} {res[s]['B']['ci95']} "
            f"qual={res[s]['B']['qualified']}")


if __name__ == "__main__":
    main()
