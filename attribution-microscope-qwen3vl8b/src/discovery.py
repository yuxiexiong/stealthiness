"""Open-ended discovery analyses (CPU only, reads the same npz maps).

Every metric in metrics.py asks a question we already thought of, and the
image-side ones are trigger-centric: M1 asks "how much landed on the trigger",
M8 asks "how much was suppressed inside CLEAN's peak". A signature that lives
somewhere else in the image — attribution drifting toward a class of regions,
a subgroup of samples behaving differently — would not show up in any of them.

These three analyses presuppose no location and no hypothesis:

  1. difference map      where did attribution actually move, per patch,
                         averaged over samples (arm minus CLEAN)
  2. heterogeneity       which samples moved most, and what they have in
                         common (answer type, trigger-over-answer overlap)
  3. clustering          unsupervised grouping of per-sample difference maps:
                         natural groups mean structure we have not named

Output: runs/discovery_wave{N}.json plus PNGs under runs/discovery/.

Every number is computed on the positive part of each map, for both
instruments. Instrument A is signed, so its figures are drawn as a pair:
the support half (what the numbers are computed on) and the suppression half
beside it, share-normalised the same way. The suppression half is figure-only;
it changes no number and no cluster membership (D56).
"""
import argparse
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from common import CFG, DATA, RUNS, read_json, write_json, log
from metrics import (load_maps, img_map, sample_ids, _pos, SCALARS_LAW,
                     QUALIFIED_INSTR, mask_for_column)

GRID = CFG["model"]["grid"]

# A's pair: warm/cool = this half is larger than in CLEAN here, grey = smaller.
# Red is support and blue suppression everywhere A is drawn, so neither
# colour is reused for "less".
SUP_CMAP = LinearSegmentedColormap.from_list("sup", ["#4d4d4d", "#ffffff", "#c81e28"])
SUPP_CMAP = LinearSegmentedColormap.from_list("supp", ["#4d4d4d", "#ffffff", "#1e5ac8"])


def delta_stack(tag, ref_tag, probe, column, scalar, instr):
    """(n_samples, 576) of per-patch attribution change, share-normalised so
    samples with different total mass are comparable."""
    z, zr = load_maps(tag, probe, column), load_maps(ref_tag, probe, column)
    if z is None or zr is None:
        return None, []
    ids, rows = [], []
    for i in sample_ids(z):
        a, b = _pos(img_map(z, i, scalar, instr)), _pos(img_map(zr, i, scalar, instr))
        sa, sb = a.sum(), b.sum()
        if sa <= 0 or sb <= 0:
            continue
        rows.append(a / sa - b / sb)
        ids.append(i)
    return (np.array(rows) if rows else None), ids


def neg_delta_stack(tag, ref_tag, probe, column, scalar, instr, ids):
    """The suppression half, share-normalised as delta_stack does, row for
    row on the given sample ids. A sample with no suppression mass in either
    model is NaN, so it drops out of a mean instead of pulling it to zero.
    Figures only."""
    z, zr = load_maps(tag, probe, column), load_maps(ref_tag, probe, column)
    rows = []
    for i in ids:
        a, b = _pos(-img_map(z, i, scalar, instr)), _pos(-img_map(zr, i, scalar, instr))
        sa, sb = a.sum(), b.sum()
        rows.append(a / sa - b / sb if sa > 0 and sb > 0
                    else np.full(a.shape, np.nan))
    return np.array(rows)


def _dual_panel(ax, m, cmap, title):
    v = np.nanmax(np.abs(m)) if np.isfinite(m).any() else 0.0
    im = ax.imshow(m, cmap=cmap, vmin=-(v or 1e-9), vmax=(v or 1e-9))
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    return im


def analysis_1_difference_map(tag, ref_tag, column, scalar, instr, out_dir):
    d, ids = delta_stack(tag, ref_tag, "p_core", column, scalar, instr)
    if d is None:
        return None
    mean_d = d.mean(0).reshape(GRID, GRID)
    trig = mask_for_column(column)
    peak_flat = int(np.argmax(np.abs(mean_d)))
    if instr == "A":
        n = neg_delta_stack(tag, ref_tag, "p_core", column, scalar, instr, ids)
        mean_n = np.nanmean(n, 0).reshape(GRID, GRID)
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9))
        for ax, m, cmap, name in ((axes[0], mean_d, SUP_CMAP, "support"),
                                  (axes[1], mean_n, SUPP_CMAP, "suppression")):
            im = _dual_panel(ax, m, cmap, f"{name} share: {tag} - {ref_tag}")
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"{tag} - {ref_tag}  {column}/{scalar}/A", fontsize=9)
        fig.text(0.01, 0.01, "red / blue = this half is larger than in "
                 f"{ref_tag} here; grey = smaller", fontsize=6.5)
    else:
        fig, ax = plt.subplots(figsize=(4, 3.6))
        v = np.abs(mean_d).max() or 1e-9
        im = ax.imshow(mean_d, cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_title(f"{tag} - {ref_tag}  {column}/{scalar}/{instr}", fontsize=8)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / f"diffmap_{tag}_{column}_{scalar}_{instr}.png", dpi=130)
    plt.close(fig)
    return {
        "n": len(ids),
        "peak_patch": peak_flat,
        "peak_rc": [peak_flat // GRID, peak_flat % GRID],
        "peak_value": float(mean_d.flat[peak_flat]),
        "peak_in_trigger": bool(peak_flat in set(trig.tolist())),
        "mass_moved_total": float(np.abs(mean_d).sum()),
        "mass_moved_outside_trigger": float(
            np.abs(mean_d).sum() - np.abs(mean_d.flat[trig]).sum()),
    }


def analysis_2_heterogeneity(tag, ref_tag, column, scalar, instr, top_k=20):
    d, ids = delta_stack(tag, ref_tag, "p_core", column, scalar, instr)
    if d is None:
        return None
    rows = {r["idx"]: r for r in read_json(DATA / "manifests" / "p_core.json")}
    mag = np.abs(d).sum(1)
    order = np.argsort(-mag)
    top = [ids[i] for i in order[:top_k]]
    bot = [ids[i] for i in order[-top_k:]]

    def profile(sel):
        c = Counter(rows[i].get("answer_type", "other") for i in sel if i in rows)
        return {k: v for k, v in c.most_common()}
    return {
        "magnitude_median": float(np.median(mag)),
        "magnitude_p90": float(np.percentile(mag, 90)),
        "ratio_p90_median": float(np.percentile(mag, 90) / (np.median(mag) + 1e-12)),
        "top_movers": top,
        "top_answer_types": profile(top),
        "bottom_answer_types": profile(bot),
    }


def analysis_3_clusters(tag, ref_tag, column, scalar, instr, k=3, out_dir=None):
    """PCA to a few dims, then k-means from a seeded init. Natural grouping of
    the difference maps means structure we have not named."""
    d, ids = delta_stack(tag, ref_tag, "p_core", column, scalar, instr)
    if d is None or len(d) < 3 * k:
        return None
    x = d - d.mean(0)
    total_var = float((x ** 2).sum())
    if total_var <= 1e-20:
        # identical difference maps carry no structure to cluster; say so
        # rather than emitting NaN into the report (decisions.log D30)
        return {"degenerate": True, "reason": "no variance between samples",
                "sizes": [len(d)] + [0] * (k - 1),
                "pc4_variance_explained": None,
                "separation_between_over_within": None}
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    z = u[:, :4] * s[:4]
    rng = np.random.default_rng(CFG["seeds"]["split"])
    cent = z[rng.choice(len(z), size=k, replace=False)]
    for _ in range(50):
        lab = np.argmin(((z[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
        new = np.array([z[lab == j].mean(0) if (lab == j).any() else cent[j]
                        for j in range(k)])
        if np.allclose(new, cent):
            break
        cent = new
    var_explained = float((s[:4] ** 2).sum() / (s ** 2).sum())
    sizes = [int((lab == j).sum()) for j in range(k)]
    # separation: between-cluster spread over within-cluster spread; ~1 means
    # no real grouping
    within = float(np.mean([((z[lab == j] - cent[j]) ** 2).sum(1).mean()
                            for j in range(k) if (lab == j).any()]))
    between = float(((cent - z.mean(0)) ** 2).sum(1).mean())
    if out_dir is not None and instr == "A":
        # clusters are found on the support half, as before; the suppression
        # row shows what the same samples did on the other half
        n = neg_delta_stack(tag, ref_tag, "p_core", column, scalar, instr, ids)
        fig, axes = plt.subplots(2, k, figsize=(3.2 * k, 6.0))
        for j in range(k):
            sel = lab == j
            m = d[sel].mean(0).reshape(GRID, GRID) if sel.any() \
                else np.zeros((GRID, GRID))
            _dual_panel(axes[0, j], m, SUP_CMAP,
                        f"cluster {j} (n={sizes[j]}) · support")
            ok = sel & np.isfinite(n).all(1)
            mn = n[ok].mean(0).reshape(GRID, GRID) if ok.any() \
                else np.zeros((GRID, GRID))
            _dual_panel(axes[1, j], mn, SUPP_CMAP,
                        f"cluster {j} (n={int(ok.sum())}) · suppression")
        fig.suptitle(f"{tag}-{ref_tag} {column}/{scalar}/A   "
                     "(red / blue = larger than in the reference, grey = smaller)",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / f"clusters_{tag}_{column}_{scalar}_{instr}.png", dpi=130)
        plt.close(fig)
    elif out_dir is not None:
        fig, axes = plt.subplots(1, k, figsize=(3.2 * k, 3))
        for j, ax in enumerate(np.atleast_1d(axes)):
            m = d[lab == j].mean(0).reshape(GRID, GRID) if (lab == j).any() \
                else np.zeros((GRID, GRID))
            v = np.abs(m).max() or 1e-9
            ax.imshow(m, cmap="RdBu_r", vmin=-v, vmax=v)
            ax.set_title(f"cluster {j} (n={sizes[j]})", fontsize=8)
            ax.axis("off")
        fig.suptitle(f"{tag}-{ref_tag} {column}/{scalar}/{instr}", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / f"clusters_{tag}_{column}_{scalar}_{instr}.png", dpi=130)
        plt.close(fig)
    return {"degenerate": False, "sizes": sizes,
            "pc4_variance_explained": var_explained,
            "separation_between_over_within": between / (within + 1e-12),
            "cluster_of": {str(ids[i]): int(lab[i]) for i in range(len(ids))}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", default="1")
    ap.add_argument("--arms", default=None)
    ap.add_argument("--ref", default="CLEAN")
    a = ap.parse_args()
    arms = (a.arms.split(",") if a.arms else
            [x["name"] for x in CFG["arms"]["wave1"]
             if x["name"] not in ("CLEAN",)])
    out_dir = RUNS / "discovery"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for tag in arms:
        for column in ("clean", "trig"):
            for scalar in SCALARS_LAW:
                for instr in QUALIFIED_INSTR.get(scalar, ("A",)):
                    key = f"{tag}|{column}|{scalar}|{instr}"
                    r1 = analysis_1_difference_map(tag, a.ref, column, scalar,
                                                   instr, out_dir)
                    if r1 is None:
                        continue
                    out[key] = {
                        "difference_map": r1,
                        "heterogeneity": analysis_2_heterogeneity(
                            tag, a.ref, column, scalar, instr),
                        "clusters": analysis_3_clusters(
                            tag, a.ref, column, scalar, instr, out_dir=out_dir),
                    }
                    log(f"discovery {key}: peak@{r1['peak_rc']} "
                        f"in_trigger={r1['peak_in_trigger']} "
                        f"outside_mass={r1['mass_moved_outside_trigger']:.4f}")
    write_json(RUNS / f"discovery_wave{a.wave}.json", out)
    log(f"discovery analyses written for {len(out)} combinations -> {out_dir}")


if __name__ == "__main__":
    main()
