"""TOY-VD-V4 (contract 2245dd8): selection regret of sampling-based acceptance.

Denominator-aligned by construction: the evaluator ranks methods using ONLY a sampled
subset S of the frozen intervention catalogue, but every method is scored by the
absolute count of unique transitions it discovers within budget k on the COMPLETE
table. Pure replay of sealed artifacts; zero GPU.
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/root/diagnosis-2026-09-15/attribution-visualization/visual-evidence-repair")
from repair.diagnosis_truth import _reference, _relation, normalize_text, validate_packet

RUNS = {"B0": Path("/root/diagnosis-2026-09-15/attribution-visualization/visual-evidence-repair/runs/diagnosis-truth-formal"),
        "B1": Path("/root/b1-build/p3-full10")}
RATES = [0.05, 0.10, 0.25, 0.50, 1.00]
BUDGETS = [1, 4]
REPS = 200
METHODS = ("G", "answer-map", "atp", "purmm", "cleansight", "random", "marker", "nearest", "constant")


def unit_data(unit):
    packet = json.loads((unit / "public.json").read_text())
    oracle = json.loads((unit / "oracle.json").read_text())
    subs = {s["method"]: s["order"] for s in json.loads((unit / "submissions.json").read_text())}
    actions = validate_packet(packet)
    rows = {}
    for key in sorted(k for k, a in actions.items() if not a["known"]):
        a = actions[key]
        answer = oracle["rows"][key]["output"]["text"]
        ref = packet["observations"][_reference(a)]["output"]["text"]
        relation = _relation(ref, answer, packet["truth_answer"])
        rows[key] = {"triple": (normalize_text(ref), normalize_text(answer), a["donor"]),
                     "changed": relation != "no_answer_change"}
    return rows, subs


def discovered(rows, order, budget, allowed=None):
    """Absolute count of unique transitions found in the first `budget` checks.

    `allowed` restricts which actions the method may even propose (the evaluator's
    visible subset); scoring always uses the complete-table rows.
    """
    seen = set()
    checked = 0
    for key in order:
        if key not in rows:
            continue
        if allowed is not None and key not in allowed:
            continue
        checked += 1
        if rows[key]["changed"]:
            seen.add(rows[key]["triple"])
        if checked >= budget:
            break
    return len(seen)


def boot(values, reps=2000, seed=0):
    if len(values) < 2:
        return {"mean": sum(values) / len(values) if values else None, "ci95": None, "n": len(values)}
    rng = random.Random(seed)
    stats = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(reps))
    return {"mean": sum(values) / len(values), "ci95": [stats[50], stats[1949]], "n": len(values)}


out = {"contract": "TOY-VD-V4 (2245dd8)", "rates": RATES, "budgets": BUDGETS, "reps": REPS, "runs": {}}
for name, run in RUNS.items():
    units = sorted(p for p in run.glob("*-abnormal") if (p / "metrics.json").exists())
    if not units:
        continue
    cache = []
    for unit in units:
        rows, subs = unit_data(unit)
        metrics = json.loads((unit / "metrics.json").read_text())
        cluster = metrics[0]["cluster_id"]
        cache.append((cluster, rows, subs))

    entry = {"units": len(units), "by_budget": {}}
    for k in BUDGETS:
        # complete-table truth: mean absolute discoveries per method, and the winner m*
        complete = {m: [] for m in METHODS}
        for cluster, rows, subs in cache:
            for m in METHODS:
                if m in subs:
                    complete[m].append(discovered(rows, subs[m], k))
        complete_mean = {m: sum(v) / len(v) for m, v in complete.items() if v}
        m_star = max(complete_mean, key=complete_mean.get)

        per_rate = {}
        for rate in RATES:
            rng = random.Random(1000 + int(rate * 100))
            regrets_by_cluster = defaultdict(list)
            correct = 0
            order_errors, order_pairs = 0, 0
            for rep in range(REPS if rate < 1.0 else 1):
                # one sampled subset per unit, then the evaluator's ranking over units
                visible_scores = {m: [] for m in METHODS}
                allowed_per_unit = []
                for cluster, rows, subs in cache:
                    keys = list(rows)
                    size = len(keys) if rate >= 1.0 else max(2, int(round(len(keys) * rate)))
                    allowed = set(rng.sample(keys, size)) if rate < 1.0 else set(keys)
                    allowed_per_unit.append(allowed)
                    for m in METHODS:
                        if m in subs:
                            # evaluator sees only `allowed`: both proposal and scoring
                            sub_rows = {kk: rows[kk] for kk in allowed}
                            visible_scores[m].append(discovered(sub_rows, subs[m], k))
                visible_mean = {m: sum(v) / len(v) for m, v in visible_scores.items() if v}
                m_hat = max(visible_mean, key=visible_mean.get)
                correct += int(m_hat == m_star)
                for i, mm in enumerate(sorted(complete_mean)):
                    for nn in sorted(complete_mean)[i + 1:]:
                        if complete_mean[mm] == complete_mean[nn]:
                            continue
                        order_pairs += 1
                        truth_better = complete_mean[mm] > complete_mean[nn]
                        seen_better = visible_mean.get(mm, 0) > visible_mean.get(nn, 0)
                        order_errors += int(truth_better != seen_better)
                # regret measured on the COMPLETE table, per cluster
                for (cluster, rows, subs), _ in zip(cache, allowed_per_unit):
                    if m_star in subs and m_hat in subs:
                        regrets_by_cluster[cluster].append(
                            discovered(rows, subs[m_star], k) - discovered(rows, subs[m_hat], k))
            cluster_means = [sum(v) / len(v) for v in regrets_by_cluster.values()]
            per_rate[str(rate)] = {
                "regret": boot(cluster_means),
                "selection_accuracy": correct / (REPS if rate < 1.0 else 1),
                "pairwise_order_error_rate": (order_errors / order_pairs) if order_pairs else None,
                "m_hat_example": m_hat}
        entry["by_budget"][str(k)] = {"complete_mean_discoveries": complete_mean,
                                      "m_star": m_star, "by_rate": per_rate}
    out["runs"][name] = entry

Path("/root/b1-build/v4-regret.json").write_text(json.dumps(out, indent=2) + "\n")
for name, e in out["runs"].items():
    print("==", name, "units", e["units"])
    for k, kb in e["by_budget"].items():
        print("  budget k=%s  m*=%s  complete means: %s" % (
            k, kb["m_star"], {m: round(v, 2) for m, v in sorted(kb["complete_mean_discoveries"].items(), key=lambda x: -x[1])[:4]}))
        for rate, r in sorted(kb["by_rate"].items(), key=lambda x: float(x[0])):
            reg = r["regret"]
            ci = ("[%.3f,%.3f]" % tuple(reg["ci95"])) if reg["ci95"] else "n/a"
            print("    r=%-5s regret=%.3f ci=%s  sel_acc=%.2f  pair_err=%.3f" % (
                rate, reg["mean"], ci, r["selection_accuracy"], r["pairwise_order_error_rate"] or 0))
