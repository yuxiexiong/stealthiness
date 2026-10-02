"""Queue job files for the amended CONTRACT_IM2.md.

    python make_im2_jobs.py <t50.json> <t50s.json> <out_dir>
"""
import json
import os
import string
import sys

import im2

T50, T50S, OUT = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
os.makedirs(OUT, exist_ok=True)
IC = "/workspace/claude-jump/xo/im_code"
ENV = ("unset TRANSFORMERS_CACHE HF_DATASETS_CACHE; export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 "
       "TRANSFORMERS_OFFLINE=1 PATH=/workspace/miniconda/envs/amic/bin:$PATH; ")


def put(fn, job):
    with open(os.path.join(OUT, fn + ".json"), "w") as f:
        json.dump(job, f, indent=1)
    print(fn, job["est_min"])


def img(run, s, col):
    return (f"[ -f $X/runs/maps/{run}@s{s}/p_core_{col}.npz ] || python $X/src/imaging_run.py --tag {run}@s{s} "
            f"--adapter $X/runs/arms/{run}/checkpoint-{s} --probes p_core --columns {col} --subset trajectory "
            f"--instruments B")


put("0297_im2_gate", {"name": "0297_im2_gate", "cmd": ENV + "python im2_gate.py",
                      "deps": ["035_qx_QX-I1001-O1009", "036_qx_QX-I1001-O1010"], "est_min": 25, "cwd": IC})
xo = sorted(r for r in T50 if r.startswith("XO"))
sp = sorted(r for r in T50 if r.startswith("SP"))
for prefix, runs in (("0298", xo), ("0299", sp)):
    for letter, run in zip(string.ascii_lowercase, runs):
        cmds = [img(run, s, "trig") for s in im2.steps_for(T50[run], T50S[run])]
        cmds += [img(run, s, "clean") for s in im2.clean_steps(T50[run], T50S[run])]
        name = f"{prefix}{letter}_im2_{run}"
        put(name, {"name": name, "cmd": ENV + "X=/workspace/claude-jump/xo; cd $X && " + " && ".join(cmds),
                   "deps": ["0297_im2_gate"], "est_min": 6 * len(cmds), "cwd": IC})
