"""TOY-VD-01 Stage A: cost audit and claim-confirm hybrid replay (contract 69fb4f9).

Pure replay of sealed artifacts — no method logic is reimplemented. Per abnormal unit:
public.json (packet), oracle.json and submissions.json are fed through the frozen
repair.diagnosis_truth primitives (validate_packet, _relation, _discovery,
normalize_text, _reference, _base_id) to rebuild graded rows exactly as the formal
run's evaluate() did, then recorded orders and one composite order are scored:

  P1 G (recorded)   P2 atp (recorded)   P4 random (recorded, violability floor)
  P3 claim-confirm: G's changed-claims in G's recorded order, then atp's order
  P5 oracle-greedy: reveals unseen transitions first (satisfiability ceiling)

Costs come from the recorded per-method cost ledgers (initial_seconds = information
acquisition) and each unit's cost.json (measurement_seconds = full oracle build).

Usage: python toy_vd_stage_a.py --runs runs/diagnosis-truth-formal --output runs/toy-vd/stage-a.json
"""

import argparse
import json
import random
from pathlib import Path

from repair.diagnosis_truth import (_base_id, _discovery, _reference, _relation,
                                    normalize_text, validate_packet)

BUDGETS = (1, 2, 4, 8, 16)


def graded_rows(packet, oracle):
    """Rebuild the grader's row list using only frozen primitives (fields _discovery needs)."""
    actions = validate_packet(packet)
    rows = []
    for key in sorted(k for k, a in actions.items() if not a["known"]):
        action = actions[key]
        answer = oracle["rows"][key]["output"]["text"]
        reference_id = _reference(action)
        reference_answer = packet["observations"][reference_id]["output"]["text"]
        rows.append({"id": key, "donor": action["donor"], "area": len(action["indices"]),
                     "normalized_answer": normalize_text(answer),
                     "reference_answer": reference_answer,
                     "relation": _relation(reference_answer, answer, packet["truth_answer"])})
    return rows


def oracle_greedy_order(rows):
    seen, order, rest = set(), [], []
    for r in rows:
        if r["relation"] != "no_answer_change":
            triple = (normalize_text(r["reference_answer"]), r["normalized_answer"], r["donor"])
            if triple not in seen:
                seen.add(triple)
                order.append(r["id"])
                continue
        rest.append(r["id"])
    return order + rest


def recall_at(rows, order):
    d = _discovery(rows, order)
    out = {}
    for k in BUDGETS:
        if str(k) in d:
            out[k] = d[str(k)]["recall"]
        else:
            # k=2 is outside the frozen 1/4/8/16 grid: truncate the order to k items
            # and read the 16-slot, which then scores exactly those k checks.
            out[k] = _discovery(rows, order[:k])["16"]["recall"]
    return out


def cluster_bootstrap_diff(pairs_by_cluster, repetitions=2000, seed=0):
    """diagnose.py-style cluster bootstrap on per-cluster mean differences."""
    values = [sum(v) / len(v) for v in pairs_by_cluster.values()]
    if len(values) < 2:
        return {"difference": sum(values) / len(values) if values else None, "ci95": None}
    rng = random.Random(seed)
    stats = sorted(sum(rng.choices(values, k=len(values))) / len(values)
                   for _ in range(repetitions))
    return {"difference": sum(values) / len(values),
            "ci95": [stats[int(.025 * repetitions)], stats[int(.975 * repetitions)]],
            "clusters": len(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    units = sorted(p for p in Path(args.runs).iterdir()
                   if p.name.endswith("-abnormal") and (p / "metrics.json").exists())
    per_unit, ratio_rows = [], []
    for unit in units:
        packet = json.loads((unit / "public.json").read_text())
        oracle = json.loads((unit / "oracle.json").read_text())
        submissions = {s["method"]: s for s in
                       json.loads((unit / "submissions.json").read_text())}
        metrics = {m["method"]: m for m in json.loads((unit / "metrics.json").read_text())}
        unit_cost = json.loads((unit / "cost.json").read_text())
        oracle_seconds = unit_cost["measurement_seconds"]
        rows = graded_rows(packet, oracle)

        base_answer = normalize_text(packet["observations"][_base_id()]["output"]["text"])
        g = submissions["G"]
        confirm = [k for k in g["order"]
                   if g["predictions"].get(k) is not None
                   and normalize_text(g["predictions"][k]) != base_answer]
        atp_order = submissions["atp"]["order"]
        policies = {
            "P1_G": g["order"],
            "P2_atp": atp_order,
            "P3_claim_confirm": confirm + [k for k in atp_order if k not in confirm],
            "P4_random": submissions["random"]["order"],
            "P5_oracle_greedy": oracle_greedy_order(rows),
        }
        recalls = {name: recall_at(rows, order) for name, order in policies.items()}
        info = {m: metrics[m]["cost"]["initial_seconds"] for m in ("G", "atp", "random")}
        ratio_rows.append({"unit": unit.name, "cluster": metrics["G"]["cluster_id"],
                           "oracle_seconds": oracle_seconds,
                           "r_G": info["G"] / oracle_seconds,
                           "r_atp": info["atp"] / oracle_seconds,
                           "r_P3": (info["G"] + info["atp"]) / oracle_seconds})
        per_unit.append({"unit": unit.name, "cluster": metrics["G"]["cluster_id"],
                         "confirm_claims": len(confirm), "recalls": recalls})

    # K-A3: information-cost ratios (cluster means, bootstrap CI).
    def ratios(field):
        by_cluster = {}
        for r in ratio_rows:
            by_cluster.setdefault(r["cluster"], []).append(r[field])
        return cluster_bootstrap_diff(by_cluster)

    k_a3 = {f: ratios(f) for f in ("r_G", "r_atp", "r_P3")}
    for f, v in k_a3.items():
        v["cheap_le_025"] = bool(v["ci95"] and v["ci95"][1] <= 0.25)

    # K-A1'': P3 - P2 recall difference at k in {2, 4}.
    k_a1 = {}
    for k in (2, 4):
        by_cluster = {}
        for u in per_unit:
            a, b = u["recalls"]["P3_claim_confirm"].get(k), u["recalls"]["P2_atp"].get(k)
            if a is None or b is None:
                continue
            by_cluster.setdefault(u["cluster"], []).append(a - b)
        stat = cluster_bootstrap_diff(by_cluster)
        stat["pass_ci_gt_0"] = bool(stat["ci95"] and stat["ci95"][0] > 0)
        k_a1[f"k={k}"] = stat

    # Frontier + satisfiability/violability summaries (means over units).
    frontier = {}
    for name in ("P1_G", "P2_atp", "P3_claim_confirm", "P4_random", "P5_oracle_greedy"):
        frontier[name] = {}
        for k in BUDGETS:
            vals = [u["recalls"][name][k] for u in per_unit if u["recalls"][name].get(k) is not None]
            frontier[name][k] = sum(vals) / len(vals) if vals else None
    order_checks = {
        "satisfiable_P5_top_at_4": all(
            frontier["P5_oracle_greedy"][4] >= frontier[p][4]
            for p in ("P1_G", "P2_atp", "P3_claim_confirm", "P4_random")),
        "violable_P4_bottom_at_4": all(
            frontier["P4_random"][4] <= frontier[p][4]
            for p in ("P1_G", "P2_atp", "P3_claim_confirm", "P5_oracle_greedy")),
    }

    result = {"contract": "TOY-VD-01 Stage A (frozen 69fb4f9)", "units": len(per_unit),
              "k_a3_info_cost_ratios": k_a3, "k_a1_hybrid_vs_atp": k_a1,
              "frontier_mean_recall": frontier, "order_checks": order_checks,
              "per_unit": per_unit, "ratio_rows": ratio_rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in
                      ("units", "k_a3_info_cost_ratios", "k_a1_hybrid_vs_atp",
                       "frontier_mean_recall", "order_checks")}, indent=2))


if __name__ == "__main__":
    main()
