"""Zero-GPU: does SAMPLING the intervention set change the verdict that the COMPLETE
table gives? This is the one angle the literature leaves open — everyone samples
because they assume exhaustive is infeasible; nobody measured what sampling costs you
in terms of the conclusion. Replays the sealed complete tables at sampling rates.
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/root/diagnosis-2026-09-15/attribution-visualization/visual-evidence-repair")
from repair.diagnosis_truth import _discovery, _reference, _relation, normalize_text, validate_packet

RUNS = [Path("/root/diagnosis-2026-09-15/attribution-visualization/visual-evidence-repair/runs/diagnosis-truth-formal"),
        Path("/root/b1-build/p3-full10")]
RATES = [0.05, 0.10, 0.25, 0.50, 1.00]
METHODS = ("G", "answer-map", "atp", "purmm", "cleansight", "random", "marker", "nearest")
REPS = 200


def rows_for(unit):
    packet = json.loads((unit / "public.json").read_text())
    oracle = json.loads((unit / "oracle.json").read_text())
    actions = validate_packet(packet)
    rows = []
    for key in sorted(k for k, a in actions.items() if not a["known"]):
        a = actions[key]
        answer = oracle["rows"][key]["output"]["text"]
        ref = packet["observations"][_reference(a)]["output"]["text"]
        rows.append({"id": key, "donor": a["donor"], "area": len(a["indices"]),
                     "normalized_answer": normalize_text(answer), "reference_answer": ref,
                     "relation": _relation(ref, answer, packet["truth_answer"])})
    return rows


def recall_at4(rows, order):
    d = _discovery(rows, order)
    return d["4"]["recall"]


out = {"rates": RATES, "reps": REPS, "runs": {}}
for run in RUNS:
    units = sorted(p for p in run.glob("*-abnormal") if (p / "metrics.json").exists())
    if not units:
        continue
    # complete-table recall per method (the ground truth verdict)
    complete = defaultdict(list)
    sampled = {r: defaultdict(list) for r in RATES if r < 1.0}
    for unit in units:
        rows = rows_for(unit)
        subs = {s["method"]: s for s in json.loads((unit / "submissions.json").read_text())}
        for m in METHODS:
            if m not in subs:
                continue
            full = recall_at4(rows, subs[m]["order"])
            if full is not None:
                complete[m].append(full)
        rng = random.Random(0)
        for rate in RATES:
            if rate >= 1.0:
                continue
            k = max(2, int(round(len(rows) * rate)))
            for rep in range(REPS):
                subset = rng.sample(rows, k)
                ids = {r["id"] for r in subset}
                for m in METHODS:
                    if m not in subs:
                        continue
                    order = [i for i in subs[m]["order"] if i in ids]
                    v = recall_at4(subset, order)
                    if v is not None:
                        sampled[rate][m].append(v)
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    complete_rank = sorted((m for m in complete), key=lambda m: -mean(complete[m]))
    entry = {"units": len(units),
             "complete": {m: mean(complete[m]) for m in complete},
             "complete_ranking": complete_rank, "sampled": {}}
    for rate in sampled:
        s_mean = {m: mean(v) for m, v in sampled[rate].items() if v}
        s_rank = sorted(s_mean, key=lambda m: -s_mean[m])
        # how often does the sampled top-1 differ from the complete top-1?
        entry["sampled"][str(rate)] = {
            "means": s_mean, "ranking": s_rank,
            "top1_complete": complete_rank[0] if complete_rank else None,
            "top1_sampled": s_rank[0] if s_rank else None,
            "top1_flipped": bool(s_rank and complete_rank and s_rank[0] != complete_rank[0]),
            "kendall_like_disagreements": sum(
                1 for i, a in enumerate(complete_rank) for b in complete_rank[i + 1:]
                if a in s_mean and b in s_mean and s_mean[a] < s_mean[b])}
    out["runs"][run.name] = entry

Path("/root/b1-build/sampling-vs-complete.json").write_text(json.dumps(out, indent=2) + "\n")
for name, e in out["runs"].items():
    print("==", name, "units", e["units"])
    print("  complete ranking:", " > ".join(e["complete_ranking"][:5]))
    for rate, s in sorted(e["sampled"].items()):
        print("   rate %-5s top1 %-10s (complete %-10s) flipped=%-5s pair-disagreements=%d"
              % (rate, s["top1_sampled"], s["top1_complete"], s["top1_flipped"],
                 s["kendall_like_disagreements"]))
