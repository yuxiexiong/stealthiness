"""Check frozen, per-metric quality requirements; never silently skip a metric.

This checks observed point estimates, not statistical non-inferiority. Supply
separate seed/CI analyses for confirmation; a point-pass is only a screen.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_quality(protocol, reference, candidate):
    if not protocol.get("frozen_at_utc") or not protocol.get("protocol_id"):
        raise ValueError("protocol_id and frozen_at_utc are required before budget comparison")
    rules = protocol.get("metrics")
    if not isinstance(rules, dict) or not rules:
        raise ValueError("a nonempty mapping of required metrics is required")
    rows = []
    for name, rule in rules.items():
        direction = rule.get("direction")
        absolute = rule.get("absolute")
        tolerance = rule.get("max_degradation")
        if direction not in ("higher", "lower") or (absolute is None and tolerance is None):
            raise ValueError(f"{name}: provide direction and absolute or max_degradation")
        if absolute is not None and not finite_number(absolute):
            raise ValueError(f"{name}: absolute must be finite")
        if tolerance is not None and (not finite_number(tolerance) or tolerance < 0):
            raise ValueError(f"{name}: max_degradation must be finite and nonnegative")
        ref, cand = reference.get(name), candidate.get(name)
        row = {"metric": name, "reference": ref if finite_number(ref) else None,
               "candidate": cand if finite_number(cand) else None, "status": "missing_or_invalid"}
        if finite_number(ref) and finite_number(cand):
            sign = 1 if direction == "higher" else -1
            baseline_ok = absolute is None or sign * ref >= sign * absolute
            candidate_ok = absolute is None or sign * cand >= sign * absolute
            relative_ok = tolerance is None or sign * (ref - cand) <= tolerance
            row["status"] = ("baseline_ineligible" if not baseline_ok else
                             "point_pass" if candidate_ok and relative_ok else "fail")
        rows.append(row)
    return {"protocol_id": protocol["protocol_id"], "metrics": rows,
            "status": "point_pass" if all(r["status"] == "point_pass" for r in rows) else "not_qualified",
            "statistical_noninferiority_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    try:
        raw = args.protocol.read_bytes()
        result = check_quality(json.loads(raw), json.loads(args.reference.read_text()), json.loads(args.candidate.read_text()))
    except (OSError, ValueError, TypeError, AttributeError) as error:
        parser.error(str(error))
    result["protocol_sha256"] = hashlib.sha256(raw).hexdigest()
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "point_pass" else 1


if __name__ == "__main__":
    sys.exit(main())
