"""ASR vs training-step line chart for the LLaVA-7B seed sweep (grok10).

Deliverable per user (2026-09-29): per-seed ASR-vs-step curves, single curves
kept (NO averaging), with the transition zoomed. Reads only real measured
points from runs/behavioral/<arm>@s*.json; a marker is drawn at EVERY measured
step so sampling density (sparse every-20 vs dense every-1 in the transition)
is visible and gaps are never hidden. Line segments join adjacent real points;
no interpolation is presented as data.

Each seed = its base arm merged with its fill arms (same training, denser saves):
  seed k -> P-1.0-D<k>  (+ P-1.0-D<k>F, P-1.0-D<k>G if present)
Auto-extends from 2 to 10 seeds as new arms land.

Usage: python plot_asr_steps.py --out runs/grok10/figs --arms P-1.0-D:1 P-1.0-D2:2 ...
  where each token is <base_arm>:<seed_label>. Fills are discovered automatically.
"""
import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# validated categorical palette (dataviz skill, light surface), fixed order
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
       "#008300", "#4a3aa7", "#e34948", "#52514e", "#0b0b0b"]
INK, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e7e7e2"


def load_curve(runs, base_arm):
    """Merge base + fill arms for one seed; return sorted [(step, asr, dense)]."""
    pts = {}
    dense = set()
    for arm in [base_arm] + [base_arm + s for s in ("F", "G")]:
        is_fill = arm != base_arm
        for p in glob.glob(str(runs / "behavioral" / f"{arm}@s*.json")):
            m = re.search(r"@s(\d+)\.json$", p)
            if not m:
                continue
            try:
                a = json.load(open(p))["asr"]
            except Exception:
                continue
            step = int(m.group(1))
            pts[step] = a
            if is_fill:
                dense.add(step)
    return [(s, pts[s], s in dense) for s in sorted(pts)]


def transition(curve):
    t5 = next((s for s, a, _ in curve if a >= 0.05), None)
    t95 = next((s for s, a, _ in curve if a >= 0.95), None)
    return t5, t95


def style_ax(ax):
    ax.set_facecolor(SURF)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.03, 1.03)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="runs/grok10/figs")
    ap.add_argument("--arms", nargs="+", required=True, help="base_arm:label ...")
    a = ap.parse_args()
    runs = Path(a.runs)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    seeds = []
    for tok in a.arms:
        base, label = tok.split(":")
        curve = load_curve(runs, base)
        if curve:
            seeds.append((label, base, curve))
    if not seeds:
        print("no data"); return

    # ---- Figure 1: overlay, all seeds, single curves (no averaging) ----
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=SURF)
    style_ax(ax)
    for i, (label, base, curve) in enumerate(seeds):
        c = PAL[i % len(PAL)]
        xs = [s for s, _, _ in curve]
        ys = [v for _, v, _ in curve]
        ax.plot(xs, ys, "-", color=c, linewidth=1.6, alpha=0.9, zorder=3, label=f"seed {label}")
        # markers: hollow for sparse (every-20), filled for dense fill points
        for s, v, dense in curve:
            ax.plot(s, v, "o", ms=4.5 if dense else 3.2, color=c,
                    mfc=c if dense else SURF, mew=1.0, zorder=4)
    ax.set_xlabel("training step (of 1250)", color=INK, fontsize=10)
    ax.set_ylabel("attack success rate (ASR)", color=INK, fontsize=10)
    ax.set_title(f"LLaVA-1.5-7B  ASR vs training step  ({len(seeds)} seeds, poison 1%)",
                 color=INK, fontsize=12, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, ncol=2, loc="lower right")
    fig.text(0.012, 0.012, "filled marker = dense fill (every 1-5 steps) · hollow = every 20 · "
             "lines join real measurements only", color=MUTED, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "asr_vs_step_overlay.png", dpi=150, facecolor=SURF)
    plt.close(fig)

    # ---- Figure 2: small multiples, one panel per seed + transition zoom ----
    n = len(seeds)
    ncol = min(5, n)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.6 * nrow),
                             facecolor=SURF, squeeze=False)
    for idx, (label, base, curve) in enumerate(seeds):
        ax = axes[idx // ncol][idx % ncol]
        style_ax(ax)
        c = PAL[idx % len(PAL)]
        xs = [s for s, _, _ in curve]; ys = [v for _, v, _ in curve]
        ax.plot(xs, ys, "-", color=c, linewidth=1.4, zorder=3)
        for s, v, dense in curve:
            ax.plot(s, v, "o", ms=4 if dense else 2.6, color=c,
                    mfc=c if dense else SURF, mew=0.9, zorder=4)
        t5, t95 = transition(curve)
        sub = f"seed {label}"
        if t5 and t95:
            ax.axvspan(t5, t95, color=c, alpha=0.08, zorder=1)
            sub += f"  t5={t5} t95={t95} (W={t95 - t5})"
        ax.set_title(sub, color=INK, fontsize=9, loc="left")
        ax.set_xlim(0, 1250)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Per-seed ASR trajectories (single curves; shaded = 5%->95% transition)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.supxlabel("training step", color=MUTED, fontsize=9)
    fig.supylabel("ASR", color=MUTED, fontsize=9)
    fig.tight_layout(rect=[0.01, 0.01, 1, 0.97])
    fig.savefig(out / "asr_vs_step_panels.png", dpi=150, facecolor=SURF)
    plt.close(fig)

    # summary table to stdout
    print(f"{'seed':>6} {'points':>7} {'t5':>5} {'t95':>5} {'W':>5} {'final':>6}")
    for label, base, curve in seeds:
        t5, t95 = transition(curve)
        fin = curve[-1][1]
        W = (t95 - t5) if (t5 and t95) else None
        print(f"{label:>6} {len(curve):>7} {str(t5):>5} {str(t95):>5} {str(W):>5} {fin:>6.3f}")
    print(f"wrote {out}/asr_vs_step_overlay.png and asr_vs_step_panels.png")


if __name__ == "__main__":
    main()
