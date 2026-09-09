"""Mine frozen probe scores and aligned attributions using only the standard library."""
import json
import math
import statistics
import sys
from pathlib import Path


def continue_bounds(score, index, eos, padding=0.0):
    top = score["topk"][index]
    known_stop = sum(x["prob"] for x in top if x["id"] in eos)
    missing = len(eos - {x["id"] for x in top})
    lower = max(math.exp(score["log_probs"][index]),
                sum(x["prob"] for x in top if x["id"] not in eos),
                1 - known_stop - missing * top[-1]["prob"])
    # Padding is a sensitivity check, not a certified floating-point error bound.
    lower = max(0.0, min(lower, 1.0) - padding)
    upper = min(1.0, 1 - known_stop + padding)
    assert 0 < lower <= upper + 1e-6
    return min(lower, upper), upper


def split_difference(trajectory, index, padding=0.0):
    eos = set(trajectory["eos_token_ids"])
    nlo, nhi = continue_bounds(trajectory["scores"]["no_trigger"], index, eos, padding)
    tlo, thi = continue_bounds(trajectory["scores"]["trigger"], index, eos, padding)
    clo, chi = math.log(tlo / nhi), math.log(thi / nlo)
    d = trajectory["d_trigger"][index]
    d_padding = 1e-4 if padding else 0.0
    return {"continue": [clo, chi],
            "within_nonstop": [d - chi - d_padding, d - clo + d_padding]}


def cosine(x, y):
    return sum(a * b for a, b in zip(x, y)) / math.sqrt(
        sum(a * a for a in x) * sum(b * b for b in y))


def attribution_detail(a):
    peak = max(range(len(a["values"])), key=lambda i: abs(a["values"][i]))
    return {"p": math.exp(a["log_prob"]), "total_abs": sum(map(abs, a["values"])),
            "peak_role": a["roles"][peak], "peak_token": a["tokens"][peak],
            "peak_value": a["values"][peak]}


def mine(root):
    docs = {}
    for path in sorted(root.glob("M*/oa-test-*.json")):
        doc = json.loads(path.read_text())
        docs[doc["model_state"], doc["sample_id"]] = doc
    continuation, examples = [], []
    selected = {("M3", "oa-test-294", 143), ("M3", "oa-test-078", 151),
                ("M1", "oa-test-298", 149), ("M3", "oa-test-298", 123)}
    for (state, sample), doc in docs.items():
        tr = doc["trajectories"]["trigger"]
        eos = set(tr["eos_token_ids"])
        counts = dict(eligible=0, d_gt_one=0, continue_gt_half=0,
                      within_nonstop_abs_lt_half=0, padded_continue_gt_half=0,
                      padded_within_nonstop_abs_lt_half=0, stop_top1_switches=0)
        for i, token_id in enumerate(tr["output_ids"]):
            if i < 10 or token_id in eos:
                continue
            counts["eligible"] += 1
            d = tr["d_trigger"][i]
            parts = split_difference(tr, i)
            padded = split_difference(tr, i, 1e-5)
            if d > 1:
                counts["d_gt_one"] += 1
                counts["continue_gt_half"] += parts["continue"][0] > d / 2
                counts["within_nonstop_abs_lt_half"] += max(map(abs, parts["within_nonstop"])) < .5
                counts["padded_continue_gt_half"] += padded["continue"][0] > (d + 1e-4) / 2
                counts["padded_within_nonstop_abs_lt_half"] += max(map(abs, padded["within_nonstop"])) < .5
            top_no = tr["scores"]["no_trigger"]["topk"][i][0]["id"]
            top_trigger = tr["scores"]["trigger"]["topk"][i][0]["id"]
            counts["stop_top1_switches"] += top_no in eos and top_trigger not in eos
            if (state, sample, i + 1) in selected:
                examples.append({"state": state, "sample": sample, "position": i + 1,
                                 "token": tr["output_tokens"][i], "d": d,
                                 "nominal_bounds": parts, "padded_bounds": padded})
        continuation.append({"state": state, "sample": sample, **counts})
    totals = []
    for state in ("M0", "M1", "M3"):
        rows = [r for r in continuation if r["state"] == state]
        totals.append({"state": state, **{k: sum(r[k] for r in rows) for k in counts},
                       "trajectories_with_continue_gt_half": sum(r["continue_gt_half"] > 0 for r in rows),
                       "trajectories_with_small_conditional_effect": sum(r["within_nonstop_abs_lt_half"] > 0 for r in rows)})
    matched, matched_examples = [], []
    for condition in ("no_trigger", "trigger", "sham"):
        all_rows, per_sample = [], []
        for sample in sorted({sample for state, sample in docs}):
            left, right = docs["M1", sample], docs["M3", sample]
            assert left["inputs"][condition]["ids"] == right["inputs"][condition]["ids"]
            a, b = left["trajectories"][condition], right["trajectories"][condition]
            rows = []
            for i, (ya, yb) in enumerate(zip(a["output_ids"], b["output_ids"])):
                if ya != yb:
                    break
                aa, bb = a["attributions"][str(i)], b["attributions"][str(i)]
                assert aa["source_ids"] == bb["source_ids"] and aa["roles"] == bb["roles"]
                x, y = aa["values"], bb["values"]
                da, db = attribution_detail(aa), attribution_detail(bb)
                row = {"sample": sample, "position": i + 1, "token": a["output_tokens"][i],
                       "cosine": cosine(x, y), "M1": da, "M3": db,
                       "tv_absolute_normalized": sum(abs(abs(u) / da["total_abs"] - abs(v) / db["total_abs"])
                                                     for u, v in zip(x, y)) / 2}
                rows.append(row)
                if condition == "trigger" and (i == 0 or (sample == "oa-test-298" and i == 4)):
                    matched_examples.append(row)
            all_rows.extend(rows)
            per_sample.append({"sample": sample, "matched_targets": len(rows),
                               "median_cosine": statistics.median(r["cosine"] for r in rows) if rows else None})
        nonsaturated = [r for r in all_rows if max(r["M1"]["p"], r["M3"]["p"]) < .95]
        matched.append({"condition": condition, "matched_targets": len(all_rows),
                        "inputs_with_matches": sum(r["matched_targets"] > 0 for r in per_sample),
                        "median_cosine": statistics.median(r["cosine"] for r in all_rows),
                        "peak_role_changes": sum(r["M1"]["peak_role"] != r["M3"]["peak_role"] for r in all_rows),
                        "p_lt_point95_targets": len(nonsaturated),
                        "p_lt_point95_median_cosine": statistics.median(r["cosine"] for r in nonsaturated),
                        "per_sample": per_sample})
    return {"scope": "Cached descriptive reanalysis; no model calls. Positions are 1-based. Bounds are not confidence intervals.",
            "source": str(root), "files": len(docs), "independent_inputs": 12,
            "continuation_totals": totals, "continuation_by_input": continuation,
            "continuation_examples": examples, "matched_M1_M3": matched,
            "matched_examples": matched_examples}


if __name__ == "__main__":
    # Small arithmetic checks; these do not validate a causal interpretation.
    assert cosine([1, 0], [0, 1]) == 0 and cosine([1, 2], [1, 2]) == 1
    fixture = {"topk": [[{"id": 10, "prob": .4}, {"id": 11, "prob": .35},
                         {"id": 12, "prob": .2}]], "log_probs": [math.log(.35)]}
    lo, hi = continue_bounds(fixture, 0, {10, 20})
    assert lo <= .55 <= hi
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "runs/attribution-probe"
    print(json.dumps(mine(root), ensure_ascii=False, indent=2, allow_nan=False))
