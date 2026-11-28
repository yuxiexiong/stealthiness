"""Summarise the GPU benchmark (make_jobs.bench): s/step, tokens/s, peak memory, bit-identity across the three runs."""
import json
import sys
from pathlib import Path

from common import write_json

out = Path(sys.argv[1])
runs = {}
for n in ("alone", "conc_a", "conc_b"):
    log = json.load(open(out / n / "train_log.json"))
    m, st = log["meta"], log["steps"]
    runs[n] = {"weights_sha256": m["weights_sha256"], "seconds": m["seconds"], "steps": m["total_steps"],
               "s_per_step": m["seconds"] / max(1, m["total_steps"]), "tok_per_s": m["tok_per_s"],
               "tok_all_per_s": m["tok_all_per_s"], "peak_mem_gib": m["peak_mem_gib"],
               "losses": [x["loss"] for x in st], "grad_norms": [x["grad_norm"] for x in st]}
same = len({r["weights_sha256"] for r in runs.values()}) == 1 and \
    all(r["losses"] == runs["alone"]["losses"] and r["grad_norms"] == runs["alone"]["grad_norms"] for r in runs.values())
pp = json.load(open(out / "pp" / "result.json"))
rep = {"bit_identical_alone_vs_concurrent": same,
       "runs": {n: {k: v for k, v in r.items() if k not in ("losses", "grad_norms")} for n, r in runs.items()},
       "pp_smoke": {"seconds": pp["seconds"], "peak_mem_gib": pp["peak_mem_gib"]}}
write_json(out / "BENCH.json", rep)
print(json.dumps(rep, indent=1))
