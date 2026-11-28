"""Gate G1s: compare two independent scoring runs of the same rows (order may differ).
usage: python compare_scores.py a.jsonl b.jsonl  -> exit 0 iff every score agrees to relative 1e-6.
A NaN on exactly one side is a mismatch (fixes the v3-review bug in the D-check copy)."""
import json
import math
import sys

A = {r["id"]: r for r in map(json.loads, open(sys.argv[1]))}
B = {r["id"]: r for r in map(json.loads, open(sys.argv[2]))}
assert set(A) == set(B), "different rows"
FIELDS = ("resp_loss", "gs_literal", "gs_adapted", "onion")


def differ(x, y):
    if x is None or y is None:
        return (x is None) != (y is None)
    nx, ny = math.isnan(x), math.isnan(y)
    if nx or ny:
        return nx != ny
    return abs(x - y) > 1e-6 * max(1.0, abs(x))


bad = [(i, f, A[i].get(f), B[i].get(f)) for i in A for f in FIELDS if differ(A[i].get(f), B[i].get(f))]
print(json.dumps({"rows": len(A), "mismatches": len(bad), "examples": bad[:5], "pass": not bad}))
sys.exit(0 if not bad else 1)
