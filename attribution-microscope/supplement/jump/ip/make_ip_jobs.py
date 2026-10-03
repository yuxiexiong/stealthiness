"""Queue job files for CONTRACT_IP.md (gate first, then dev, null pairs, validation).

    python make_ip_jobs.py <out_dir>
"""
import json
import os
import sys

import ip

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
IC = "/workspace/claude-jump/xo/ip_code"
ENV = ("unset TRANSFORMERS_CACHE HF_DATASETS_CACHE; export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 "
       "TRANSFORMERS_OFFLINE=1 PATH=/workspace/miniconda/envs/amic/bin:$PATH; ")


def put(fn, job):
    with open(os.path.join(OUT, fn + ".json"), "w") as f:
        json.dump(job, f, indent=1)
    print(fn, job["est_min"])


put("0300_ip_gate", {"name": "0300_ip_gate", "cmd": ENV + "python ip_gate.py", "deps": [], "est_min": 10, "cwd": IC})
for k, p in enumerate(ip.DEV + ip.NULL + ip.VAL):
    name = f"030{k + 1}_ip_{p[0]}"
    put(name, {"name": name, "cmd": ENV + f"[ -f /workspace/claude-jump/xo/runs/ip/{p[0]}.npz ] || python run_ip.py {p[0]}",
               "deps": ["0300_ip_gate"], "est_min": 35, "cwd": IC})
