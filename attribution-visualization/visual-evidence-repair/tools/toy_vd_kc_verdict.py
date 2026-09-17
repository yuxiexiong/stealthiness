"""TOY-VD-01 Stage C verdicts: K-C-pre (instrument validity under the diffuse blended
trigger) and K-C (cross-attack generalization), computed from the sealed B1 run."""
import json
import random
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, "/root/diagnosis-2026-09-15/attribution-visualization/visual-evidence-repair")
from repair.diagnosis_truth import _discovery, normalize_text, _relation, _reference, validate_packet

RUN = Path("/root/b1-build/p3-final")


def graded_rows(packet, oracle):
    actions = validate_packet(packet)
    rows = []
    for key in sorted(k for k, a in actions.items() if not a["known"]):
        a = actions[key]
        answer = oracle["rows"][key]["output"]["text"]
        ref_id = _reference(a)
        ref = packet["observations"][ref_id]["output"]["text"]
        rows.append({"id": key, "donor": a["donor"], "area": len(a["indices"]),
                     "normalized_answer": normalize_text(answer), "reference_answer": ref,
                     "relation": _relation(ref, answer, packet["truth_answer"])})
    return rows


def oracle_greedy(rows):
    seen, first, rest = set(), [], []
    for r in rows:
        if r["relation"] != "no_answer_change":
            t = (normalize_text(r["reference_answer"]), r["normalized_answer"], r["donor"])
            if t not in seen:
                seen.add(t); first.append(r["id"]); continue
        rest.append(r["id"])
    return first + rest


def boot(values, reps=2000, seed=0):
    if len(values) < 2:
        return {"mean": sum(values)/len(values) if values else None, "ci95": None, "n": len(values)}
    rng = random.Random(seed)
    stats = sorted(sum(rng.choices(values, k=len(values)))/len(values) for _ in range(reps))
    return {"mean": sum(values)/len(values), "ci95": [stats[50], stats[1949]], "n": len(values)}


pre_by_cluster, prec_by_cluster = defaultdict(list), defaultdict(list)
base_by_cluster = defaultdict(lambda: defaultdict(list))
for unit in sorted(RUN.glob("editclevr-*-abnormal")):
    if not (unit / "metrics.json").exists():
        continue
    packet = json.loads((unit / "public.json").read_text())
    oracle = json.loads((unit / "oracle.json").read_text())
    subs = {s["method"]: s for s in json.loads((unit / "submissions.json").read_text())}
    metrics = {m["method"]: m for m in json.loads((unit / "metrics.json").read_text())}
    cluster = metrics["G"]["cluster_id"]
    rows = graded_rows(packet, oracle)
    og = _discovery(rows, oracle_greedy(rows))["4"]["recall"]
    rd = _discovery(rows, subs["random"]["order"])["4"]["recall"]
    if og is not None and rd is not None:
        pre_by_cluster[cluster].append(og - rd)
    p = metrics["G"].get("changed_precision")
    if isinstance(p, (int, float)):
        prec_by_cluster[cluster].append(p)
    for m in ("purmm", "cleansight", "marker", "random", "atp"):
        for k in ("1", "4"):
            v = metrics[m]["discovery_at"][k]["recall"]
            if v is not None:
                base_by_cluster[(m, k)][cluster].append(v)

per = lambda d: [sum(v)/len(v) for v in d.values()]
kcpre = boot(per(pre_by_cluster))
kcprec = boot(per(prec_by_cluster))
result = {"contract": "TOY-VD-01 Stage C (frozen 66fb240, DEV-P3-01/02)",
          "k_c_pre": {**kcpre, "gate": "ci_lower > 0",
                      "pass": bool(kcpre["ci95"] and kcpre["ci95"][0] > 0)},
          "k_c_precision": {**kcprec, "gate": "ci_lower >= 0.95",
                            "pass": bool(kcprec["ci95"] and kcprec["ci95"][0] >= 0.95)},
          "baselines_vs_null": {}}
null = {k: boot(per(base_by_cluster[("random", k)])) for k in ("1", "4")}
for m in ("purmm", "cleansight", "marker", "atp"):
    for k in ("1", "4"):
        b = boot(per(base_by_cluster[(m, k)]))
        result["baselines_vs_null"][f"{m}@{k}"] = {
            "mean": b["mean"], "ci95": b["ci95"], "null_mean": null[k]["mean"],
            "le_null": bool(b["mean"] is not None and null[k]["mean"] is not None
                            and b["mean"] <= null[k]["mean"])}
result["random_null"] = {f"@{k}": null[k] for k in ("1", "4")}
Path("/root/b1-build/p3-final/kc-verdict.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
