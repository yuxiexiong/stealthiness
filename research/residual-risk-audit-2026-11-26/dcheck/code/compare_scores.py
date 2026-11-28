"""Gate G1: compare two independent scoring runs of the same sampled rows (order may differ).
usage: python compare_scores.py a.jsonl b.jsonl   -> exit 0 iff every score agrees to relative 1e-6"""
import json
import math
import sys

A = {r["id"]: r for r in map(json.loads, open(sys.argv[1]))}
B = {r["id"]: r for r in map(json.loads, open(sys.argv[2]))}
assert set(A) == set(B), "different rows"
bad = []
for i in A:
    for f in ("resp_loss", "gs_entropy", "onion"):
        x, y = A[i][f], B[i][f]
        if (x is None) != (y is None) or (x is not None and not (math.isnan(x) and math.isnan(y))
                                          and abs(x - y) > 1e-6 * max(1.0, abs(x))):
            bad.append((i, f, x, y))
print(json.dumps({"rows": len(A), "mismatches": len(bad), "examples": bad[:5], "pass": not bad}))
sys.exit(0 if not bad else 1)
