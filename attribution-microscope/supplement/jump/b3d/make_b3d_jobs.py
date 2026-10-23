"""Queue job files for CONTRACT_B3D.   python make_b3d_jobs.py <out_dir> [prefix=035]
GB2 is a CPU step run by hand before this (b3d_prep.py), so it is not a job.
Two trainings (one per card, no dependency between them) -> two LP read-outs."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import b3d  # noqa: E402
OUT = sys.argv[1]; PFX = sys.argv[2] if len(sys.argv) > 2 else "035"
os.makedirs(OUT, exist_ok=True)
BC, LC = "/workspace/claude-jump/xo/b3d_code", "/workspace/claude-jump/lp_code"


def put(fn, job):
    json.dump(dict(job, name=fn), open(os.path.join(OUT, fn + ".json"), "w"), indent=1); print(fn, "<-", job["deps"])


for i, arm in enumerate(b3d.ARMS, 1):
    put(f"{PFX}{i:02d}_b3d_{arm}", {"cmd": f"bash b3d_run.sh {arm}", "deps": ["file:/workspace/claude-jump/xo/runs/b3d/gate.json"],
                                    "est_min": 110, "cwd": BC})
for i, arm in enumerate(b3d.ARMS, 1):
    put(f"{PFX}{i + 2:02d}_lp_{arm}", {"cmd": f"bash {BC}/b3d_lp.sh {arm}", "deps": [f"{PFX}{i:02d}_b3d_{arm}"],
                                       "est_min": 26, "cwd": LC})
