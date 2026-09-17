"""Mass-based instrument qualification — CRITERION FROZEN BEFORE MEASUREMENT.

Why a second criterion at all: the existing pointing game scores the argmax
patch, but not one downstream metric uses an argmax. M1, M8, M4, M5 and M7 are
all mass shares and M3 is a top-50 overlap. Qualifying on argmax while
measuring with mass is a criterion/measurement mismatch, and it penalises
high-frequency maps (instrument A, 14px patches) relative to inherently smooth
ones (instrument B, 28px occlusion windows): a single outlier patch fails the
argmax test even when most of the mass sits on the object.

CRITERION v2 (D26). The v1 ratio L = mass_share / area_share was VOID: its
ceiling is 1/area_share, and with the old probe set (median box = 52% of the
image) 54 of 100 images could not reach the 2.0 bar even under perfect
localization. That run produced no evidence about the instruments and is
discarded as a contract-execution failure, not as a negative result.

v2 normalises by the achievable range, so full marks are full marks at any
box size:

    gain  G = (mass_share - area_share) / (1 - area_share)

    G = 0    attribution spread like chance
    G = 1    all mass inside the box (reachable for every image)
    G >= 0.30 QUALIFIED

    fail-side demo: cascading weight randomization must pull G back to ~0.

Reachability is rehearsed against the real box distribution BEFORE freezing
(`--rehearse`), which is the step whose absence produced the void v1 run.

The argmax pointing result stands and is reported alongside. If the two
criteria disagree, that disagreement is a finding about the instrument, not a
licence to pick the friendlier number.
"""
import argparse

import numpy as np
from PIL import Image

from common import CFG, DATA, RUNS, read_json, write_json, log

PATCH = CFG["model"]["patch_px"]
GRID = CFG["model"]["grid"]
QUALIFY_AT = 0.30         # frozen threshold on the gain G
CHANCE = 0.0


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


def localization_gain(m576, box336):
    """Chance-corrected, range-normalised localization: 0 = chance, 1 = all
    mass inside the box, reachable regardless of how large the box is."""
    m = np.clip(np.asarray(m576, dtype=np.float64), 0, None)
    tot = m.sum()
    if tot <= 0:
        return np.nan
    wt = box_patch_weights(box336)
    area_frac = wt.sum() / (GRID * GRID)
    if not (0 < area_frac < 1):
        return np.nan
    mass_share = (m * wt).sum() / tot
    return float((mass_share - area_frac) / (1.0 - area_frac))


def rehearse(rows):
    """Offline reachability rehearsal on the REAL box distribution (protocol
    6.2). v1 skipped this and froze a bar that half the images could not reach
    under any instrument. Demonstrates both that the criterion can be met and
    that it can fail, using synthetic maps only — no GPU, no model."""
    perfect, chance, anti, reach = [], [], [], []
    for r in rows:
        wt = box_patch_weights(r["box336"])
        inside = (wt > 0.5).astype(float)
        if inside.sum() == 0 or inside.sum() == len(inside):
            continue
        perfect.append(localization_gain(inside, r["box336"]))
        chance.append(localization_gain(np.ones(GRID * GRID), r["box336"]))
        anti.append(localization_gain(1.0 - inside, r["box336"]))
        reach.append(perfect[-1] >= QUALIFY_AT)
    out = {
        "n": len(perfect),
        "threshold": QUALIFY_AT,
        "perfect_median": float(np.median(perfect)),
        "chance_median": float(np.median(chance)),
        "anti_median": float(np.median(anti)),
        "frac_images_where_threshold_reachable": float(np.mean(reach)),
        "area_frac_median": float(np.median(
            [box_patch_weights(r["box336"]).sum() / (GRID * GRID) for r in rows])),
    }
    # the criterion is usable only if a perfect instrument passes on
    # essentially every image and chance/anti-localization clearly fail
    out["pass_side_ok"] = bool(out["frac_images_where_threshold_reachable"] >= 0.95)
    out["fail_side_ok"] = bool(out["chance_median"] < QUALIFY_AT
                               and out["anti_median"] < QUALIFY_AT)
    out["criterion_usable"] = bool(out["pass_side_ok"] and out["fail_side_ok"])
    return out


def measure(sess, rows, probe_dir, width, scalars, tid):
    vals = {s: {"A": [], "B": []} for s in scalars}
    for r in rows:
        img = Image.open(probe_dir / f"{r['idx']:0{width}d}.jpg")
        cid = sess.first_subtoken(r["category"])
        res = sess.attribute(img, r["question"], tid, cid)
        occ = sess.occlusion(img, r["question"], tid, cid)
        for s in scalars:
            vals[s]["A"].append(localization_gain(res[s]["A_img_signed"], r["box336"]))
            vals[s]["B"].append(localization_gain(occ["img"][s], r["box336"]))
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
            out[s][i] = {"median_G": med, "n": int(len(v)),
                         "ci95": [float(np.percentile(boot, 2.5)),
                                  float(np.percentile(boot, 97.5))],
                         "frac_above_chance": float((v > CHANCE).mean()) if len(v) else None,
                         "qualified": bool(med >= QUALIFY_AT)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="p_instrument_xl2")
    ap.add_argument("--randomized", action="store_true",
                    help="fail-side demo: randomize weights first")
    ap.add_argument("--rehearse", action="store_true",
                    help="offline reachability rehearsal only, no GPU")
    ap.add_argument("--out", default="qual_mass.json")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    rows = read_json(DATA / "manifests" / f"{a.set}.json")
    if a.rehearse:
        reh = rehearse(rows)
        write_json(RUNS / f"rehearsal_{a.set}.json", reh)
        log(f"rehearsal on {a.set}: perfect={reh['perfect_median']:.2f} "
            f"chance={reh['chance_median']:.2f} anti={reh['anti_median']:.2f} "
            f"reachable={reh['frac_images_where_threshold_reachable']:.2f} "
            f"area_med={reh['area_frac_median']:.3f} "
            f"USABLE={reh['criterion_usable']}")
        raise SystemExit(0 if reh["criterion_usable"] else 1)
    from attribution.engine import LlavaSession
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
