"""Write queue job files for CONTRACT_PS.md.   python make_ps_jobs.py <out_dir>"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lp"))
import lp  # noqa: E402
import ps  # noqa: E402

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
PC, LC = "/workspace/claude-jump/xo/ps_code", "/workspace/claude-jump/lp_code"


def put(fn, job):
    with open(os.path.join(OUT, fn + ".json"), "w") as f:
        json.dump(job, f, indent=1)
    print(fn, "<-", job["deps"])


gate_env = ("unset TRANSFORMERS_CACHE HF_DATASETS_CACHE; export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 "
            "TRANSFORMERS_OFFLINE=1 DISABLE_VERSION_CHECK=1 PATH=/workspace/miniconda/envs/amic/bin:$PATH; ")
put("0290_ps_gate", {"name": "0290_ps_gate", "cmd": gate_env + "python ps_gate.py", "deps": [], "est_min": 10, "cwd": PC})
for i, k in enumerate(ps.KINDS, 1):
    n = ps.arm(k)
    put(f"029{i}_ps_{n}", {"name": f"029{i}_ps_{n}", "cmd": f"bash ps_run.sh {n}", "deps": ["0290_ps_gate"],
                           "est_min": 105, "cwd": PC})
for i, k in enumerate(ps.KINDS):
    n = ps.arm(k)
    pairs = " ".join(f"{n}@s{s}=$X/runs/arms/{n}/checkpoint-{s}" for s in lp.STEPS)
    put(f"0296{i}_lp_{n}", {"name": f"0296{i}_lp_{n}", "cmd": "source lp_env.sh && $PY logit_probe.py $X $O " + pairs,
                            "deps": [f"029{i + 1}_ps_{n}"], "est_min": 26, "cwd": LC})
