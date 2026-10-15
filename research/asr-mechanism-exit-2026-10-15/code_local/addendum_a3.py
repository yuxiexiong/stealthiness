"""Local check of ADDENDUM_A3.md on results/a3/s*/a3.json (CPU, numpy).

    python code_local/addendum_a3.py results
"""
import json
from pathlib import Path
import sys

import numpy as np

SEEDS = (1105, 1106, 1107)


def q(state):
    return float(np.mean([r["push_pre"] for r in state["rows"] if r["kind"] == "clean"]))


def classify(e, m, b):
    if m > 0 and e <= 0.5 * m and b < m:
        return "clean_turns_positive"
    if m <= 0 or m <= e:
        return "clean_no_turn"
    return "between"


def main():
    root = Path(sys.argv[1])
    per = {}
    for s in SEEDS:
        p = root / "a3" / ("s%d" % s) / "a3.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        ph = [d["phase1"][k] for k in sorted(d["phase1"], key=int)][:3]
        mo = [d["mother"][k] for k in sorted(d["mother"], key=int)][-3:]
        b2 = [d["b2"][k] for k in sorted(d["b2"], key=int)][-3:]
        e, m, b = (float(np.mean([q(x) for x in xs])) for xs in (ph, mo, b2))
        per["s%d" % s] = {"E": e, "M": m, "B": b, "class": classify(e, m, b)}
    v = [x["class"] for x in per.values()]
    tier = ("supported" if v.count("clean_turns_positive") >= 2 else "refuted" if v.count("clean_no_turn") >= 2 else "mixed") if len(v) == 3 else "incomplete"
    out = {"per_mother": per, "tier": tier}
    print(json.dumps(out, indent=1))
    (root / "addendum_a3.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
