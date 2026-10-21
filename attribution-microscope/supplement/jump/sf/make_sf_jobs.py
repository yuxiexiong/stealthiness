"""Write queue job files for CONTRACT_SF.md.   python make_sf_jobs.py <out_dir> [prefix=031]

Queue: GD1 (one GPU) -> six trainings (deps GD1; P1 both, P2 both, R both, so the two judged pairs
advance together on two cards) -> six LP read-outs (each deps on its training). GS0 is a CPU step run
by hand before this (build_sf.py), so it is not a job. Worker: /workspace/claude-jump/jobq/worker.py
(one per GPU; GPU0's must be started by hand, see HANDOFF)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sf  # noqa: E402

OUT = sys.argv[1]
PFX = sys.argv[2] if len(sys.argv) > 2 else "031"
os.makedirs(OUT, exist_ok=True)
SC, LC = "/workspace/claude-jump/xo/sf_code", "/workspace/claude-jump/lp_code"
gate_env = ("unset TRANSFORMERS_CACHE HF_DATASETS_CACHE; export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 "
            "TRANSFORMERS_OFFLINE=1 DISABLE_VERSION_CHECK=1 PATH=/workspace/miniconda/envs/amic/bin:$PATH; ")


def put(fn, job):
    with open(os.path.join(OUT, fn + ".json"), "w") as f:
        json.dump(dict(job, name=fn), f, indent=1)
    print(fn, "<-", job["deps"])


gd1 = f"{PFX}00_sf_gd1"
put(gd1, {"cmd": gate_env + "python gd1_sf.py", "deps": [], "est_min": 15, "cwd": SC})
for i, (p, s, f) in enumerate(sf.runs(), 1):
    n = sf.arm(p, s, f)
    put(f"{PFX}{i:02d}_sf_{n}", {"cmd": f"bash sf_run.sh {n}", "deps": [gd1], "est_min": 110, "cwd": SC})
for i, (p, s, f) in enumerate(sf.runs(), 1):
    n = sf.arm(p, s, f)
    put(f"{PFX}{i + 6:02d}_lp_{n}", {"cmd": f"bash {SC}/sf_lp.sh {n}", "deps": [f"{PFX}{i:02d}_sf_{n}"],
                                    "est_min": 26, "cwd": LC})
