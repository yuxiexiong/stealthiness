"""AUDIT-TOY-02 Stage A judging: source-object fact per pilot image via the frozen
judge consensus (owlvit-clip + blipvqa agree -> fact; else vilt adjudicates), then
the K-op qualification verdict (>=6/12 direct hits per operator; op4 is ceiling only).

Usage:
  python audit_operators_judge.py --generation runs/audit-operators/generation.json \
      --output runs/audit-operators/verdict.json --device cuda
"""

import argparse
import json
from pathlib import Path

from judges import JUDGES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from PIL import Image
    root = Path(args.generation).parent
    payload = json.loads(Path(args.generation).read_text())
    rows = [r for r in payload["records"] if r["status"] == "ok"]

    j1 = JUDGES["owlvit-clip"](args.device)
    j2 = JUDGES["blipvqa"](args.device)
    j3 = None
    judged = []
    for n, r in enumerate(rows, 1):
        image = Image.open(root / r["image"]).convert("RGB")
        s1, d1 = j1.answer(image, r["source_name"])
        s2, d2 = j2.answer(image, r["source_name"])
        if s1 == s2:
            fact, adjudicated = s1, False
        else:
            if j3 is None:
                j3 = JUDGES["vilt"](args.device)
            fact, _ = j3.answer(image, r["source_name"])
            adjudicated = True
        judged.append({**r, "judge1": s1, "judge2": s2, "fact": fact,
                       "adjudicated": adjudicated,
                       "direct_hit": bool(fact == r["donor"])})
        print(f"[{n}/{len(rows)}] {r['case_id']} {r['condition_id']} {r['operator']}: "
              f"fact={fact} donor={r['donor']} hit={fact == r['donor']}", flush=True)

    verdicts = {}
    for op in ("op2_eos", "op3_noun", "op4_full"):
        sub = [r for r in judged if r["operator"] == op]
        hits = sum(r["direct_hit"] for r in sub)
        verdicts[op] = {"attempted": len(sub), "hits": hits,
                        "rate": hits / len(sub) if sub else None,
                        "k_op_pass": bool(hits >= 6),
                        "selectable": op != "op4_full"}
    result = {"contract": "AUDIT-TOY-02 Stage A", "rows": judged, "verdicts": verdicts,
              "gate": "qualify if hits >= 6 of 12; op4 is ceiling reference only"}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(verdicts, indent=2))


if __name__ == "__main__":
    main()
