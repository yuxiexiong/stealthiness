"""Queue job files for CONTRACT_IM2.md.   python make_im2_jobs.py <t50.json> <out_dir>
t50.json: {run: t50} for the 14 XO / SP runs (from xo/results_xo.json and sp/results_sp.json)."""
import json
import os
import string
import sys

import im2

T50, OUT = json.load(open(sys.argv[1])), sys.argv[2]
os.makedirs(OUT, exist_ok=True)
IC = "/workspace/claude-jump/xo/im_code"
ENV = ("unset TRANSFORMERS_CACHE HF_DATASETS_CACHE; export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 "
       "TRANSFORMERS_OFFLINE=1 PATH=/workspace/miniconda/envs/amic/bin:$PATH; ")


def put(fn, job):
    with open(os.path.join(OUT, fn + ".json"), "w") as f:
        json.dump(job, f, indent=1)
    print(fn, len(job["cmd"]), job["est_min"])


put("0297_im2_gate", {"name": "0297_im2_gate", "cmd": ENV + "python im2_gate.py", "deps": [], "est_min": 15, "cwd": IC})
xo = sorted(r for r in T50 if r.startswith("XO"))
sp = sorted(r for r in T50 if r.startswith("SP"))
for prefix, runs in (("0298", xo), ("0299", sp)):
    for letter, run in zip(string.ascii_lowercase, runs):
        cmds = [f"[ -f $X/runs/maps/{run}@s{s}/p_core_trig.npz ] || python $X/src/imaging_run.py --tag {run}@s{s} "
                f"--adapter $X/runs/arms/{run}/checkpoint-{s} --probes p_core --columns trig --subset trajectory "
                f"--instruments B" for s in im2.steps_for(T50[run])]
        name = f"{prefix}{letter}_im2_{run}"
        put(name, {"name": name, "cmd": ENV + "X=/workspace/claude-jump/xo; cd $X && " + " && ".join(cmds),
                   "deps": ["0297_im2_gate"], "est_min": 6 * len(cmds), "cwd": IC})
