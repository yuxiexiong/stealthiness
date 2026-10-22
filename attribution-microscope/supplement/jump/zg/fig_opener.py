import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
F = json.load(open(sys.argv[1])); out = sys.argv[2]
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # validated categorical slots 1-5 (light)
INK, INK2, GRID = "#222222", "#6b6b6b", "#e4e4e0"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9.5, "axes.titleweight": "bold"})
fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.4), dpi=200)
# A: cumulative poison count, 9 original orders, take-off marked on each curve
a = ax[0]
for s, d in sorted(F["A"].items()):
    a.plot(range(601), d["K"], color=INK2, lw=0.9, alpha=0.55)
    a.plot([d["t50"]], [d["K_at_t50"]], "o", ms=5, color=C[0], mec="white", mew=1.0, zorder=5)
a.axvspan(330, 470, color=C[0], alpha=0.07, lw=0)
a.set_xlim(0, 600); a.set_ylim(0, 100); a.set_xlabel("training step"); a.set_ylabel("poison rows seen so far")
a.set_title("A  Cumulative dose is the same;\n    take-off (dots) spans 330–470")
a.text(475, 8, "take-off\nrange", color=C[0], fontsize=8)
a.text(12, 92, "9 orders, 1% poison, 1 epoch", color=INK2, fontsize=8)
# B: XO cross: same order -> same take-off, same init -> scattered
b = ax[1]
orders = sorted({r["order"] for r in F["B"]}); inits = sorted({r["init"] for r in F["B"]})
xpos = {o: i for i, o in enumerate(orders)}
seen = {}
for r in sorted(F["B"], key=lambda r: (r["order"], r["init"])):
    k = (r["order"], r["t50"]); j = seen.get(k, 0); seen[k] = j + 1      # jitter coincident points sideways
    b.plot([xpos[r["order"]] + (0.09 if j else -0.09) * (1 if len([q for q in F["B"] if (q["order"], q["t50"]) == k]) > 1 else 0)], [r["t50"]], "o", ms=7, color=C[inits.index(r["init"])], mec="white", mew=1.0, zorder=5)
for o in orders:
    ys = [r["t50"] for r in F["B"] if r["order"] == o]
    if len(ys) > 1: b.plot([xpos[o]] * 2, [min(ys), max(ys)], color=INK2, lw=1, zorder=1)
for i, init in enumerate(inits):
    b.plot([], [], "o", color=C[i], ms=6, label=f"init {init}")
b.legend(frameon=False, fontsize=7.5, title="LoRA init seed", title_fontsize=7.5, loc="upper left", handletextpad=0.3)
b.set_xticks(range(len(orders))); b.set_xticklabels([f"order {o}" for o in orders], fontsize=8)
b.set_ylim(305, 495); b.set_ylabel("take-off step (ASR ≥ 0.5)")
b.set_title("B  Swap order × init (8 runs):\n    take-off follows the data order")
b.text(0.2, 338, "order 1001 under\n4 inits: 330–360\n(spread 30)", fontsize=7.5, color=INK2)
b.text(1.75, 441, "init 1001 under 4 orders:\n330–470 (spread 140)", fontsize=7.5, color=INK2)
# C: PS perturbations overlap
c = ax[2]
lab = {"PS-P0": "P0 unchanged", "PS-P1": "P1 swap within batch", "PS-P2": "P2 swap steps 50↔51", "PS-P3": "P3 swap steps 50↔250", "PS-P4": "P4 poison row 55→56"}
for i, (arm, pts) in enumerate(sorted(F["C"].items())):
    xs, ys = zip(*pts); c.plot(xs, ys, "-o", ms=3.5, lw=1.4, color=C[i], mec="white", mew=0.6, label=lab[arm], alpha=0.95)
c.legend(frameon=False, fontsize=7, loc="lower right", handlelength=1.6)
c.set_xlim(300, 420); c.set_ylim(-0.03, 1.03); c.set_xlabel("training step"); c.set_ylabel("ASR")
c.set_title("C  Local edits of the same order (5 runs):\n    curves coincide, take-off moves 0 steps")
for axx in ax: axx.grid(True, color=GRID, lw=0.6); axx.set_axisbelow(True)
fig.suptitle("Take-off time is set by the coarse structure of the data order: not by dose, not by init, not by local detail", fontsize=10, color=INK, y=1.02)
fig.tight_layout()
fig.savefig(out + ".png", bbox_inches="tight"); fig.savefig(out + ".svg", bbox_inches="tight"); print("saved", out)
