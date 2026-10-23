"""Queue job files for CONTRACT_J09.   python make_j09_jobs.py <out_dir> [prefix=037]
GJ1 (one GPU, three one-step runs) -> two trainings (dep GJ1) -> two LP read-outs."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import j09  # noqa
OUT = sys.argv[1]; PFX = sys.argv[2] if len(sys.argv) > 2 else "037"; os.makedirs(OUT, exist_ok=True)
JC, LC = "/workspace/claude-jump/xo/j09_code", "/workspace/claude-jump/lp_code"
gate_env = ("unset TRANSFORMERS_CACHE HF_DATASETS_CACHE; export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 "
            "TRANSFORMERS_OFFLINE=1 DISABLE_VERSION_CHECK=1 PATH=/workspace/miniconda/envs/amic/bin:$PATH; ")


def put(fn, job):
    json.dump(dict(job, name=fn), open(os.path.join(OUT, fn + ".json"), "w"), indent=1); print(fn, "<-", job["deps"])


gj1 = f"{PFX}00_j09_gj1"; put(gj1, {"cmd": gate_env + "python gj1_j09.py", "deps": [], "est_min": 10, "cwd": JC})
for i, init in enumerate(j09.INITS, 1):
    a = j09.arm(init)
    put(f"{PFX}{i:02d}_j09_{a}", {"cmd": f"bash j09_run.sh {a} {j09.SEED_KEY[init]}", "deps": [gj1], "est_min": 110, "cwd": JC})
for i, init in enumerate(j09.INITS, 1):
    a = j09.arm(init)
    put(f"{PFX}{i + 2:02d}_lp_{a}", {"cmd": f"bash {JC}/j09_lp.sh {a}", "deps": [f"{PFX}{i:02d}_j09_{a}"], "est_min": 26, "cwd": LC})
