"""Write queue job files: logit read-out (LP) for the four transplant runs, each after its training job.

    python make_lp_tp_jobs.py <out_dir>
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lp"))
sys.path.insert(0, os.path.join(HERE, "..", "tp"))
import lp  # noqa: E402
import tp  # noqa: E402

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
LC = "/workspace/claude-jump/lp_code"
for k, (b, d, a, e) in enumerate(tp.runs()):
    n = tp.arm(b, d, a, e)
    pairs = " ".join(f"{n}@s{s}=$X/runs/arms/{n}/checkpoint-{s}" for s in lp.STEPS)
    job = {"name": f"028{k}_lp_{n}",
           "cmd": "source lp_env.sh && $PY logit_probe.py $X $O " + pairs,
           "deps": [f"027{k + 6}_tp_{n}"],
           "est_min": 26, "cwd": LC}
    with open(os.path.join(OUT, f"028{k}_lp_{n}.json"), "w") as f:
        json.dump(job, f, indent=1)
    print(job["name"], "<-", job["deps"])
