"""Rebuild figures and audit the evidence pack; CPU/file operations only."""
import csv
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/asr-rising-integration-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_json(path):
    return json.loads((ROOT / path).read_text())


def write_json(path, value):
    (ROOT / path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                       allow_nan=False) + "\n")


def write_csv(path, fields, rows):
    with (ROOT / path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, name):
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(ROOT / "figures" / f"{name}.{suffix}", dpi=180,
                    bbox_inches="tight")
    plt.close(fig)


(ROOT / "figures").mkdir(exist_ok=True)
(ROOT / "tables").mkdir(exist_ok=True)
p2 = read_json("sources/p2/results_p2.json")
gate = read_json("sources/gate/judge.json")
arms = ["P2-A", "P2-R", "P2-B"]
colors = ["#0072B2", "#555555", "#D55E00"]
fields = ["median_margin_trig", "median_margin_clean", "median_D",
          "asr_curve", "clean_acc_curve", "clean_target_rate_curve"]
rows = [{"arm": arm, "metric": metric, "step": step, "value": value,
         "probe_n": 200}
        for arm in arms for metric in fields
        for step, value in p2["arms"][arm][metric]]
write_csv("tables/p2_readouts.csv", ["arm", "metric", "step", "value", "probe_n"], rows)
gate_rows = []
for seed, baseline in gate["mother_t_jump"].items():
    for condition, time in [("mother", baseline),
                            ("G1", gate["t_jump"]["G1"][seed]),
                            ("G1r", gate["t_jump"]["G1r"][seed])]:
        gate_rows.append({"seed": seed, "condition": condition,
                          "t_jump": time, "delta_vs_mother": time - baseline})
write_csv("tables/gate_times.csv", ["seed", "condition", "t_jump", "delta_vs_mother"], gate_rows)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})
fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.2), constrained_layout=True)
for arm, color in zip(arms, colors):
    points = p2["arms"][arm]["asr_curve"]
    axes[0].plot([s for s, _ in points], [100*v for _, v in points],
                 marker="o", markersize=3, color=color, label=arm)
axes[0].axhline(50, color="#bbbbbb", linestyle=":", linewidth=1)
axes[0].set(xlabel="Training update (common inputs from 201)", ylabel="Generated ASR (%)",
            title="A  LLaVA P2: different early clean histories", xlim=(195, 705), ylim=(-3, 103))
axes[0].legend(frameon=False, loc="lower right")
axes[0].text(0.48, 0.65, "Observed t50: 320 / 340 / 360\nOne training run per arm",
             transform=axes[0].transAxes, va="top", fontsize=9)
for seed, color in zip(gate["mother_t_jump"], ["#332288", "#117733", "#882255"]):
    values = [gate["mother_t_jump"][seed], gate["t_jump"]["G1"][seed],
              gate["t_jump"]["G1r"][seed]]
    axes[1].plot(range(3), values, marker="o", color=color, label=f"Seed {seed}")
    for x, y in enumerate(values):
        axes[1].annotate(str(y), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
axes[1].set_xticks(range(3), ["Mother", "Clean pretraining\nG1", "Random B norm\nG1r"])
axes[1].set(ylabel="First >3/60 target hits: subsequent updates",
            title="B  Qwen toy GATE: learned state vs random control", xlim=(-0.2, 2.25), ylim=(30, 83))
axes[1].legend(frameon=False, loc="lower right", fontsize=8)
save_figure(fig, "history_effects")

fig, axes = plt.subplots(2, 2, figsize=(11.7, 7), constrained_layout=True)
plot_fields = fields[:3] + ["clean_acc_curve"]
titles = ["A  Trigger margin T", "B  Clean-input margin C",
          "C  Paired difference D = T - C", "D  Normal-task accuracy"]
for ax, metric, title in zip(axes.flat, plot_fields, titles):
    for arm, color in zip(arms, colors):
        points = p2["arms"][arm][metric]
        scale = 100 if metric == "clean_acc_curve" else 1
        ax.plot([s for s, _ in points], [scale*v for _, v in points],
                marker="o", markersize=2, linewidth=1.3, color=color, label=arm)
    ax.axvspan(0, 200, color="#eeeeee", zorder=-2)
    ax.axvline(200, color="#999999", linestyle=":", linewidth=1)
    ax.set(title=title, xlabel="Training update", xlim=(0, 705),
           ylabel="Accuracy (%)" if metric == "clean_acc_curve" else "Median across 200 probes (logit)")
    if metric == "clean_acc_curve":
        ax.set_ylim(0, 100)
    else:
        ax.axhline(0, color="#cccccc", linewidth=0.7)
axes[0, 0].legend(frameon=False)
fig.suptitle("LLaVA P2: early inputs differ; all inputs match at updates 201-700\n"
             "Shading marks the differing prefix, which still contains fixed poison exposures", fontsize=11)
save_figure(fig, "p2_readouts")

origins = read_json("sources/local_origins.json") + read_json("sources/remote_origins.json")
source_checks = []
for entry in origins:
    data = (ROOT / entry["path"]).read_bytes()
    entry["sha256"] = hashlib.sha256(data).hexdigest()
    entry["bytes"] = len(data)
    if entry["origin"] == "github_fixed_commit":
        actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        ok = actual == entry["git_blob"]
    else:
        ok = hashlib.sha256(data).hexdigest() == entry["source_sha256"]
    source_checks.append({"path": entry["path"], "matches_origin": ok})
write_json("SOURCE_MANIFEST.json", {"scope": "read-only existing evidence; no model execution",
                                    "sources": origins})

with (ROOT / "sources/cross_model/inventory.csv").open() as f:
    inventory = list(csv.DictReader(f))
with (ROOT / "tables/cross_model_summary.csv").open() as f:
    summary = list(csv.DictReader(f))
group_checks = []
for group in summary:
    selected = [r for r in inventory if r["model"] == group["model"]
                and r["arm"] == group["arm"] and r["view"] == group["trajectory_view"]
                and abs(float(r["poison_rate"])*100 - float(group["poison_rate_pct"])) < 1e-8
                and r["criterion"] == "budget_1pct"]
    counts = Counter(r["status"] for r in selected)
    ok = len(selected) == int(group["n_selected_records"])
    ok &= counts["confirmed"] == int(group["main_1pct_observed"])
    ok &= counts["indeterminate"] == int(group["main_1pct_indeterminate"])
    ok &= all(int(r["full_training_budget"]) == int(group["registered_budget_updates"]) for r in selected)
    group_checks.append({"model": group["model"], "dose": group["poison_rate_pct"],
                         "view": group["trajectory_view"], "arm": group["arm"], "pass": ok})

gap = p2["arms"]["P2-B"]["sustained"] - p2["arms"]["P2-A"]["sustained"]
d700 = [dict(p2["arms"][a]["median_D"])[700] for a in arms]
checks = {"inventory_rows_768": len(inventory) == 768,
          "summary_groups_26": len(summary) == 26,
          "p2_gap_matches": abs(gap - 46.161945786015735) < 1e-9,
          "p2_probe_counts": all(p2["arms"][a]["n_lp"] == 59 and p2["arms"][a]["n_beh"] == 25 for a in arms),
          "p2_four_readouts_agree": all(p2["arms"][a]["first"] == p2["arms"][a]["sustained"] and p2["arms"][a]["t50"] == p2["arms"][a]["t50s"] for a in arms),
          "p2_700_D_range": max(d700) - min(d700) == 0.515625,
          "gate_G1": gate["delta_G1"] == [-11, -32, -7],
          "gate_G1r": gate["delta_G1r"] == [0, 2, -1]}
links, missing = [], []
for doc in ROOT.glob("*.md"):
    for target in re.findall(r"\]\(([^)]+)\)", doc.read_text()):
        target = unquote(target.strip("<>"))
        if target.startswith(("https://", "http://", "#")):
            continue
        target = re.sub(r":\d+$", "", target.split("#")[0])
        # This report is written after its link checks, including on a fresh checkout.
        exists = target == "VALIDATION.json" or (doc.parent / target).exists()
        links.append({"document": doc.name, "target": target, "exists": exists})
        if not exists:
            missing.append(links[-1])
passed = all(checks.values()) and all(x["matches_origin"] for x in source_checks) and all(x["pass"] for x in group_checks) and not missing
write_json("VALIDATION.json", {"status": "PASS" if passed else "FAIL", "gpu_hours_added": 0,
                               "training_runs_added": 0, "model_inference_calls": 0,
                               "scope": "source bytes, derived counts/numbers and local document links; not a training reproduction",
                               "checks": checks, "source_checks": source_checks, "cross_model_group_checks": group_checks,
                               "local_links_checked": len(links), "missing_links": missing,
                               "p2_gap_updates": gap, "p2_median_D_range_at_700": max(d700)-min(d700),
                               "raw_source_note": "Original JSON may contain NaN; source bytes are preserved. Derived validation JSON is strict."})
print(json.dumps({"status": "PASS" if passed else "FAIL", "checks": checks,
                  "missing_links": missing, "sources": len(origins)}, ensure_ascii=False, indent=2))
raise SystemExit(0 if passed else 1)
