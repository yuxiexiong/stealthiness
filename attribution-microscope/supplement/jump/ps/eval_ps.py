"""Evaluate CONTRACT_PS.md locally.

    python eval_ps.py <xo_runs_copy> <lp_out_dir> <ref_log_P-1.0-D.jsonl> <out.json>
xo_runs_copy holds behavioral/PS-*.json, gate_ps.json and arms/PS-*/trainer_log.jsonl."""
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lp"))
import lp  # noqa: E402
import ps  # noqa: E402

R, LPD, REF, OUT = sys.argv[1:5]
gate = json.load(open(os.path.join(R, "gate_ps.json")))


def loss(p):
    d = {x["current_steps"]: x["loss"] for x in map(json.loads, open(p)) if "loss" in x}
    return [d.get(20 * w) for w in range(1, 63)]


t50, med, logs = {}, {}, {}
for k in ps.KINDS:
    n = ps.arm(k)
    cur = sorted((int(re.search(r"@s(\d+)\.json$", f).group(1)), json.load(open(f))["asr"])
                 for f in glob.glob(os.path.join(R, "behavioral", f"{n}@s*.json")))
    t50[k] = lp.t50_10(cur) if cur else None
    m = {}
    for s in lp.STEPS:
        f = os.path.join(LPD, f"{n}@s{s}.json")
        if os.path.exists(f):
            m[s] = float(np.median([c["M"] for c in json.load(open(f))["core"]]))
    med[k] = m
    lf = os.path.join(R, "arms", n, "trainer_log.jsonl")
    logs[k] = loss(lf) if os.path.exists(lf) else None
res = {"gate": gate, "t50": t50,
       "verdict": ps.verdict(t50) if gate["pass"] else {"tier": "not judged (gate failed)"}}
ref = loss(REF)
res["P0_vs_original_s1_loss_identical"] = (logs["P0"] == ref) if logs["P0"] else None
div = {}
for k in ps.KINDS[1:]:
    common = sorted(set(med[k]) & set(med["P0"]))
    d = {s: med[k][s] - med["P0"][s] for s in common}
    first = next((s for s in common if abs(d[s]) > 0.05), None)
    div[k] = {"first_step_abs_diff_gt_0.05": first,
              "max_abs_diff_to_300": max([abs(d[s]) for s in common if s <= 300], default=None),
              "loss_first_window_differing": next((20 * (w + 1) for w in range(62)
                                                   if logs[k] and logs["P0"] and logs[k][w] != logs["P0"][w]), None)}
res["divergence"] = div
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("t50", "verdict", "P0_vs_original_s1_loss_identical", "divergence")}, indent=1))
