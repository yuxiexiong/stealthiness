"""CPU-only P2 second-seed readout; never launches training or fills missing results.

python code/evaluate.py --run-root /path/to/prepared/run
Requires data_identity.json (probe_ids, target_word), paired/, runs/behavioral/.
Reuses frozen P2/SF definitions; adds complete-input validation and sustained
readout coverage checks. Output: results_replication.json; invalid input exits 2.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline" / "phase2"))
import p2
import sf

ARMS = ("P2S7-A", "P2S7-B")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def first_word(answer):
    words = answer.split()
    return words[0].rstrip(".,") if words else ""


def keyed(rows, ids, where):
    require(isinstance(rows, list) and len(rows) == 200, f"{where}: expected 200 probes")
    require(all(isinstance(r, dict) and type(r.get("idx")) is int for r in rows), f"{where}: invalid idx")
    by_id = {r["idx"]: r for r in rows}
    require(len(by_id) == 200 and set(by_id) == set(ids), f"{where}: duplicate or unexpected/missing probe ids")
    return by_id


def load_record(path, ids, target, paired):
    require(path.is_file(), f"missing {path}")
    rec = json.loads(path.read_text())
    require(isinstance(rec, dict), f"{path}: expected object")
    rows = keyed(rec.get("core" if paired else "per"), ids, str(path))
    if paired:
        require(rec.get("tag") == path.stem, f"{path}: tag mismatch")
        for row in rows.values():
            require(all(number(row.get(k)) for k in ("M", "M_clean", "D")), f"{path}: non-finite/missing margin")
            require(math.isclose(row["D"], row["M"] - row["M_clean"], rel_tol=1e-10, abs_tol=1e-9), f"{path}: D identity mismatch")
            require(all(type(row.get(k)) is bool for k in ("argmax_is_target", "argmax_is_target_clean")), f"{path}: invalid argmax flag")
    else:
        require(rec.get("mode") == "image" and rec.get("trig_col") == "trig", f"{path}: wrong behavioral mode")
        for row in rows.values():
            require(all(type(row.get(k)) is bool for k in ("asr", "acc")), f"{path}: invalid behavioral flag")
            require(all(isinstance(row.get(k), str) for k in ("clean_ans", "trig_ans")), f"{path}: missing answer")
            hit = first_word(row["trig_ans"]) == target
            require(row["asr"] == hit, f"{path}: ASR flag disagrees with generated first word")
        for field, key in (("asr", "asr"), ("clean_acc", "acc")):
            mean = sum(r[key] for r in rows.values()) / 200
            require(number(rec.get(field)) and abs(rec[field] - mean) <= 1e-12, f"{path}: {field} aggregate mismatch")
    return rec


def sustained_with_coverage(steps, values, behavioral=False):
    if behavioral:
        raw = sf.t50_sustained(list(zip(steps, values)), hold=p2.HOLD)
    else:
        raw = sf.t_m0_sustained(steps, values, hold=p2.HOLD)
    if raw is None:
        return None, {"raw": None, "censored": True, "reason": "no_confirmed_crossing"}
    threshold = 0.5 if behavioral else 0.0
    anchor = next(s for s, v in zip(steps, values) if s >= raw - 1e-9 and v >= threshold)
    covered = raw <= p2.CAND_MAX and anchor + p2.HOLD <= max(steps)
    return (raw if covered else None), {"raw": raw, "anchor": anchor, "censored": not covered,
                                        "reason": None if covered else "hold_window_not_fully_observed"}


def read_arm(root, arm, ids, target):
    lp = {s: load_record(root / "paired" / f"{arm}@s{s}.json", ids, target, True) for s in p2.STEPS_LP}
    beh = {s: load_record(root / "runs" / "behavioral" / f"{arm}@s{s}.json", ids, target, False) for s in p2.STEPS_BEH}
    steps, med = sf.median_curve(lp)
    curve = [(s, beh[s]["asr"]) for s in p2.STEPS_BEH]
    sustained, margin_coverage = sustained_with_coverage(steps, med)
    t50s, behavior_coverage = sustained_with_coverage([s for s, _ in curve], [v for _, v in curve], True)
    per = {s: {r["idx"]: r["asr"] for r in rec["per"]} for s, rec in beh.items()}
    gr0 = sf.gr0(lp, per)
    require(gr0["n"] == 200 * len(p2.STEPS_BEH), f"{arm}: GR0 did not cover every matched probe")
    result = {
        "arm": arm, "n_lp": len(lp), "n_beh": len(beh), "gr0": gr0,
        "first": sf.t_m0(steps, med), "sustained": sustained,
        "t50": sf.t50_10(curve), "t50s": t50s,
        "coverage": {"sustained": margin_coverage, "t50s": behavior_coverage},
        "median_margin_trig": list(zip(steps, med)),
        "median_margin_clean": [(s, float(np.median([r["M_clean"] for r in lp[s]["core"]]))) for s in steps],
        "median_D": [(s, float(np.median([r["D"] for r in lp[s]["core"]]))) for s in steps],
        "asr_curve": curve,
        "clean_acc_curve": [(s, beh[s]["clean_acc"]) for s in p2.STEPS_BEH],
        "clean_target_rate_curve": [(s, sum(first_word(r["clean_ans"]) == target
                                              for r in beh[s]["per"]) / 200) for s in p2.STEPS_BEH],
    }
    return result, lp


def judge(arms):
    valid = all(arms[a]["gr0"]["pass"] for a in ARMS)
    verdicts = {}
    for key in ("sustained", "first", "t50", "t50s"):
        a, b = (arms[name][key] if valid else None for name in ARMS)
        v = p2.verdict_main(a, b)
        if key in ("sustained", "t50s"):
            v["censored"] = {name: arms[name]["coverage"][key]["censored"] for name in ARMS}
        else:
            v.pop("censored", None)
        verdicts[key] = v
    main = verdicts["sustained"]["gap"]
    gaps = [verdicts[k]["gap"] for k in ("first", "t50", "t50s")]
    confirmed = valid and main is not None and main >= p2.BAND_LARGE and all(g is not None and g > 0 for g in gaps)
    return {"readout_gate_pass": valid, "verdicts": verdicts, "same_direction_large_replication": confirmed,
            "claim_scope": "specified A/B pair across training seeds; no universal, mediator, or clean noninferiority claim"}


def evaluate(root):
    root = Path(root)
    gates = {a: json.loads((root / f"gate_train_{a}.json").read_text()) for a in ARMS}
    require(all(g.get("pass") is True and g.get("global_step") == 700 for g in gates.values()),
            "training gate failed or incomplete")
    update_gate = json.loads((root / "update_audit.json").read_text())
    require(update_gate.get("pass") is True, "actual-update audit gate failed")
    require(type(update_gate.get("effective_poison_update_match")) is bool,
            "actual-update audit missing effective poison-update comparison")
    identity = json.loads((root / "data_identity.json").read_text())
    ids, target = identity["probe_ids"], identity["target_word"]
    require(isinstance(ids, list) and len(ids) == 200 and all(type(i) is int for i in ids) and len(set(ids)) == 200, "identity: require 200 unique integer probe_ids")
    require(isinstance(target, str) and len(target.split()) == 1, "identity: invalid target_word")
    arms, raw = {}, {}
    for arm in ARMS:
        arms[arm], raw[arm] = read_arm(root, arm, ids, target)
    result = judge(arms)
    result.update(status="COMPLETE" if result["readout_gate_pass"] else "READOUT_GATE_FAILED", arms=arms,
                  primary="sustained", components={}, training_gates=gates, update_audit=update_gate,
                  matched_poison_execution_replication=result["same_direction_large_replication"] and update_gate["effective_poison_update_match"],
                  component_interpretation="descriptive paired output differences, not causal mediation shares")
    result["claim_scope"] = "specified A/B history total effect with matched poisoned-update execution; LR, optimizer state, gradients, and numerical paths remain possible mechanisms"
    if not update_gate["effective_poison_update_match"]:
        result["claim_scope"] = "specified A/B history total effect under the original recipe; unequal executed poisoned updates remain a possible pathway, alongside LR, optimizer state, gradients, and numerical paths"
    for s in p2.COMP_STEPS:
        cols = []
        for arm in ARMS:
            rows = sorted(raw[arm][s]["core"], key=lambda r: r["idx"])
            cols.extend([[r["M"] for r in rows], [r["M_clean"] for r in rows]])
        result["components"][str(s)] = p2.components(*cols)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = evaluate(args.run_root)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        result = {"status": "INVALID_OR_INCOMPLETE", "error": str(exc), "verdicts": None,
                  "same_direction_large_replication": None}
    output = args.run_root / "results_replication.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "output": str(output)}, ensure_ascii=False))
    return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    sys.exit(main())
