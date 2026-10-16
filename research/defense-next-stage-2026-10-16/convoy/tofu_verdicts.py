"""Convoy step (CPU): tier 1.2 SQuAD+TOFU verdicts: SQuAD risk/quality (paired vs this root's A0), TOFU answer NLL
(new knowledge learned?) and the share of TOFU rows Curfew masked."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convoy_lib import *
R = REAL + "/squad_tofu"; ref = real_reference(R)
tofu_idx = {json.loads(l)["canonical_index"] for l in open(R + "/private/data/train.jsonl") if json.loads(l).get("tofu")}
summary = json.load(open(R + "/tofu_eval_summary.json"))
out = {"tofu_rows": len(tofu_idx), "base": summary.get("_base:base"), "runs": {}}
for run, labels in (("a1_s1006", ["final"]), ("fx2_s1006", ["final"]), ("tb_stop_s1006", ["stop_base", "stop_g0", "stop_g25", "stop_g50", "stop_g100"])):
    d = R + "/runs/" + run
    for label in labels:
        key = "%s:%s" % (run, label)
        out["runs"][key] = {"squad": real_verdict(R, d, label, ref=ref), "tofu": summary.get(key)}
    if run.startswith("fx2") and os.path.exists(d + "/result.json"):
        masked = {i for _, i, _ in json.load(open(d + "/result.json"))["policy"]["masked"]}
        out["runs"][run + ":final"]["tofu_rows_masked"] = len(masked & tofu_idx)
        out["runs"][run + ":final"]["tofu_rows_masked_frac"] = round(len(masked & tofu_idx) / len(tofu_idx), 4)
json.dump(out, open(CONVOY + "/tofu_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
