"""Read cached probe measurements; print descriptive statistics without model calls."""
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


def mean(xs):
    return statistics.mean(xs) if xs else None


def summarize(root):
    rows, pairs, deletions = [], [], []
    for path in sorted(root.glob("M*/oa-test-*.json")):
        doc = json.loads(path.read_text())
        state, sample = doc["model_state"], doc["sample_id"]
        ts = doc["trajectories"]
        pairs.append({"state": state, "sample": sample,
                      "no_equals_trigger": ts["no_trigger"]["output_ids"] == ts["trigger"]["output_ids"],
                      "no_equals_sham": ts["no_trigger"]["output_ids"] == ts["sham"]["output_ids"],
                      "trigger_equals_sham": ts["trigger"]["output_ids"] == ts["sham"]["output_ids"],
                      "marker_token_counts": doc["marker_token_counts"]})
        for condition, tr in ts.items():
            n = len(tr["output_ids"])
            assert len(tr["d_trigger"]) == len(tr["d_sham"]) == n
            row = {"state": state, "sample": sample, "condition": condition,
                   "n": n, "stop": tr["stop_reason"], "first_token": tr["output_tokens"][0],
                   "a_targets": [], "a_details": []}
            for name in ("d_trigger", "d_sham"):
                vals = tr[name]
                peak = max(range(n), key=lambda i: abs(vals[i]))
                row[name] = {"mean_abs": mean(list(map(abs, vals))),
                             "first10_mean_abs": mean(list(map(abs, vals[:10]))),
                             "rest_mean_abs": mean(list(map(abs, vals[10:]))),
                             "peak_index": peak, "peak_token": tr["output_tokens"][peak],
                             "peak_value": vals[peak], "first": vals[0], "last": vals[-1],
                             "n_abs_over_1": sum(abs(v) > 1 for v in vals)}
            for key, a in sorted(tr["attributions"].items(), key=lambda kv: int(kv[0])):
                if a["status"] != "completed":
                    continue
                index = int(key)
                assert len(a["values"]) == len(a["roles"]) == len(a["source_ids"]) == len(tr["input_ids"]) + index
                absolute, signed, counts = defaultdict(float), defaultdict(float), Counter(a["roles"])
                for role, value in zip(a["roles"], a["values"]):
                    absolute[role] += abs(value)
                    signed[role] += value
                total = sum(absolute.values())
                peak = max(range(len(a["values"])), key=lambda i: abs(a["values"][i]))
                detail = {"target": index, "total_abs": total,
                          "abs_by_role": dict(absolute), "signed_by_role": dict(signed),
                          "source_counts": dict(counts),
                          "abs_share": {role: value / total if total else 0 for role, value in absolute.items()},
                          "peak_role": a["roles"][peak], "peak_source": peak,
                          "peak_token": a["tokens"][peak], "peak_value": a["values"][peak],
                          "log_prob": a["log_prob"],
                          "prefix_vs_sequence_delta": a.get("prefix_vs_sequence_logprob_delta")}
                row["a_targets"].append(index)
                row["a_details"].append(detail)
            if len(row["a_details"]) == n:
                cuts = [0, n // 3, 2 * n // 3, n]
                row["history_share_thirds"] = [
                    mean([a["abs_share"].get("history", 0)
                          for a in row["a_details"][cuts[k]:cuts[k + 1]]])
                    for k in range(3)]
            rows.append(row)
            by_target = defaultdict(dict)
            for deletion in tr["deletions"]:
                if deletion["status"] == "completed":
                    target, source = deletion["target_index"], deletion["source_index"]
                    a = tr["attributions"][str(target)]
                    prefix_delta = deletion["log_prob"] - a["log_prob"]
                    by_target[target][deletion["kind"]] = {
                        "source": source, "original_delta": deletion["delta_log_prob"],
                        "prefix_baseline_delta": prefix_delta,
                        "drop_original": -deletion["delta_log_prob"],
                        "drop_prefix_baseline": -prefix_delta, "a": a["values"][source]}
            for target, ds in by_target.items():
                deletions.append({"state": state, "sample": sample,
                                  "condition": condition, "target": target, **ds})
    groups = []
    for state in ("M0", "M1", "M3"):
        for condition in ("no_trigger", "trigger", "sham"):
            rs = [r for r in rows if r["state"] == state and r["condition"] == condition]
            aa = [a for r in rs for a in r["a_details"]]
            groups.append({"state": state, "condition": condition, "trajectories": len(rs),
                           "tokens": sum(r["n"] for r in rs), "a_positions": len(aa),
                           "stops": dict(Counter(r["stop"] for r in rs)),
                           "median_trajectory_mean_abs_d_trigger": statistics.median(r["d_trigger"]["mean_abs"] for r in rs),
                           "median_trajectory_mean_abs_d_sham": statistics.median(r["d_sham"]["mean_abs"] for r in rs),
                           "d_trigger_peak_first10": sum(r["d_trigger"]["peak_index"] < 10 for r in rs),
                           "a_peak_roles": dict(Counter(a["peak_role"] for a in aa)),
                           "a_median_abs_history_share": statistics.median(a["abs_share"].get("history", 0) for a in aa),
                           "a_median_abs_marker_share": statistics.median(a["abs_share"].get("marker", 0) for a in aa),
                           "max_abs_prefix_sequence_delta": max(abs(a["prefix_vs_sequence_delta"]) for a in aa),
                           "median_abs_prefix_sequence_delta": statistics.median(abs(a["prefix_vs_sequence_delta"]) for a in aa)})
    return {"source": str(root), "rows": rows, "groups": groups, "pairs": pairs,
            "deletions": deletions,
            "scope": "Descriptive cached measurements; no judge, ASR, hypothesis test, or model call."}


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "runs/attribution-probe"
    print(json.dumps(summarize(root), ensure_ascii=False, indent=2))
