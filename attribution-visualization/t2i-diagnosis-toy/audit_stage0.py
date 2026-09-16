"""AUDIT-TOY-01 Stage 0: bare-machine instrument calibration (contract section 2).

Computes per-case mechanism profiles (DRR, CLR) from gold facts, splits cases into
natively-correct vs failed by their base facts, and tests K0: the group DRR difference's
cluster-bootstrap 95% CI lower bound must exceed the 95th percentile of a label-permutation
null. Also runs the mandatory violability demonstration (shuffled labels must fail K0).

Zero GPU. Inputs are frozen study manifests plus gold fact tables; optional extra
(study, facts) pairs fold in the outcome-blind expansion cases.

Usage:
  python audit_stage0.py --study study.json --gold results/gold-2026-09-15/gold.json \
      [--extra-study study-extended.json --extra-facts runs/audit/expansion-facts.json] \
      --output runs/audit/stage0.json
"""

import argparse
import json
import random
from pathlib import Path

INTERVENTIONS = ("probe_a", "probe_b", "test_a", "test_b")


def load_cases(study_path, facts_by_key, only_ids=None):
    study = json.loads(Path(study_path).read_text())
    rows = []
    for case in study["cases"]:
        if only_ids is not None and case["id"] not in only_ids:
            continue
        key = (case["id"], "base")
        if key not in facts_by_key:
            continue
        intended = {o["id"]: o["color"] for o in case["objects"]}
        conditions = {c["id"]: c for c in case["conditions"]}
        base = facts_by_key[key]
        if "unjudgeable" in base.values():
            group = "excluded_unjudgeable"
        elif all(base[k] == intended[k] for k in ("a", "b")):
            group = "correct"
        else:
            group = "failed"
        direct_hits, cross_hits, used = 0, 0, 0
        for cid in INTERVENTIONS:
            fkey = (case["id"], cid)
            if fkey not in facts_by_key:
                continue
            condition = conditions[cid]
            source = condition["source"] or "a"
            donor = condition["color"]
            facts = facts_by_key[fkey]
            used += 1
            direct_hits += int(facts[source] == donor)
            other = "b" if source == "a" else "a"
            cross_hits += int(facts[other] != base[other])
        rows.append({"case_id": case["id"], "group_id": case["group"], "split": case["split"],
                     "state": group, "interventions_used": used,
                     "drr": direct_hits / used if used else None,
                     "clr": cross_hits / used if used else None})
    return rows


def cluster_values(rows, state):
    """Per object-pair-cluster mean DRR for one state, keyed by cluster id."""
    per = {}
    for r in rows:
        if r["state"] == state and r["drr"] is not None:
            per.setdefault(r["group_id"], []).append(r["drr"])
    return {g: sum(v) / len(v) for g, v in per.items()}


def k0_statistic(rows):
    correct = cluster_values(rows, "correct")
    failed = cluster_values(rows, "failed")
    if not correct or not failed:
        return None, correct, failed
    diff = sum(correct.values()) / len(correct) - sum(failed.values()) / len(failed)
    return diff, correct, failed


def bootstrap_ci(correct, failed, repetitions=2000, seed=0):
    rng = random.Random(seed)
    ck, fk = list(correct.values()), list(failed.values())
    stats = []
    for _ in range(repetitions):
        c = rng.choices(ck, k=len(ck))
        f = rng.choices(fk, k=len(fk))
        stats.append(sum(c) / len(c) - sum(f) / len(f))
    stats.sort()
    return [stats[int(0.025 * repetitions)], stats[int(0.975 * repetitions)]]


def permutation_null(rows, repetitions=1000, seed=1):
    """Permute correct/failed labels at the cluster level; return the 95th percentile."""
    clusters = {}
    for r in rows:
        if r["state"] in ("correct", "failed") and r["drr"] is not None:
            clusters.setdefault(r["group_id"], []).append(r)
    ids = sorted(clusters)
    n_correct = len({g for g in ids if any(r["state"] == "correct" for r in clusters[g])})
    rng = random.Random(seed)
    stats = []
    for _ in range(repetitions):
        chosen = set(rng.sample(ids, n_correct))
        c_means, f_means = [], []
        for g in ids:
            mean = sum(r["drr"] for r in clusters[g]) / len(clusters[g])
            (c_means if g in chosen else f_means).append(mean)
        if c_means and f_means:
            stats.append(sum(c_means) / len(c_means) - sum(f_means) / len(f_means))
    stats.sort()
    return stats[int(0.95 * len(stats))], len(ids), n_correct


def violability_demo(rows, seed=7):
    """Shuffled labels must FAIL K0; otherwise the criterion is broken."""
    rng = random.Random(seed)
    labeled = [r for r in rows if r["state"] in ("correct", "failed") and r["drr"] is not None]
    states = [r["state"] for r in labeled]
    rng.shuffle(states)
    shuffled = [dict(r, state=s) for r, s in zip(labeled, states)]
    diff, correct, failed = k0_statistic(shuffled)
    if diff is None:
        return {"k0_pass_under_shuffle": False, "note": "degenerate shuffle"}
    lo, _ = bootstrap_ci(correct, failed, seed=8)
    threshold, _, _ = permutation_null(shuffled, seed=9)
    return {"difference": diff, "ci_lower": lo, "null_p95": threshold,
            "k0_pass_under_shuffle": bool(lo > threshold)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--extra-study")
    parser.add_argument("--extra-facts", help="JSON {case_id: {condition_id: {a,b}}}")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    gold = json.loads(Path(args.gold).read_text())
    facts = {(r["case_id"], r["condition_id"]): r["facts"] for r in gold["answers"]}
    rows = load_cases(args.study, facts)
    if args.extra_study and args.extra_facts:
        extra = json.loads(Path(args.extra_facts).read_text())
        extra_facts = {(case_id, cid): f for case_id, conds in extra.items()
                       for cid, f in conds.items()}
        rows += load_cases(args.extra_study, extra_facts, only_ids=set(extra))

    diff, correct, failed = k0_statistic(rows)
    result = {"contract": "AUDIT-TOY-01 v2 Stage 0",
              "cases": rows,
              "counts": {s: sum(r["state"] == s for r in rows)
                         for s in ("correct", "failed", "excluded_unjudgeable")}}
    if diff is None:
        result["k0"] = {"verdict": "not_computable", "reason": "empty group"}
    else:
        ci = bootstrap_ci(correct, failed)
        null_p95, clusters, n_correct_clusters = permutation_null(rows)
        mean = lambda d: sum(d.values()) / len(d)
        result["k0"] = {"correct_mean_drr": mean(correct), "failed_mean_drr": mean(failed),
                        "correct_clusters": len(correct), "failed_clusters": len(failed),
                        "difference": diff, "ci95": ci, "permutation_null_p95": null_p95,
                        "satisfiable_pass": bool(ci[0] > null_p95)}
        result["violability"] = violability_demo(rows)
        result["k0"]["verdict"] = ("PASS" if result["k0"]["satisfiable_pass"]
                                   and not result["violability"]["k0_pass_under_shuffle"]
                                   else "FAIL")
        clr_c = cluster_values(rows, "correct")
        # CLR summarized for the acceptance-region bookkeeping; not part of K0.
        clr_vals = [r["clr"] for r in rows if r["state"] == "correct" and r["clr"] is not None]
        result["correct_group_clr"] = sorted(clr_vals)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("counts", "k0") if k in result}, indent=2))
    if "violability" in result:
        print("violability (shuffled labels must fail):",
              json.dumps(result["violability"]))


if __name__ == "__main__":
    main()
