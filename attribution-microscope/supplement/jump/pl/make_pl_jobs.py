"""Queue job files for CONTRACT_PL.   python make_pl_jobs.py <out_dir> [prefix=034]
GD1 (one GPU) -> seven trainings in pl.ARMS order (random arms first; each dep GD1) -> seven LP read-outs."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pl  # noqa
OUT = sys.argv[1]; PFX = sys.argv[2] if len(sys.argv) > 2 else "034"; os.makedirs(OUT, exist_ok=True)
PC, LC = "/workspace/claude-jump/xo/pl_code", "/workspace/claude-jump/lp_code"
gate_env = ("unset TRANSFORMERS_CACHE HF_DATASETS_CACHE; export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 "
            "TRANSFORMERS_OFFLINE=1 DISABLE_VERSION_CHECK=1 PATH=/workspace/miniconda/envs/amic/bin:$PATH; ")


def put(fn, job):
    json.dump(dict(job, name=fn), open(os.path.join(OUT, fn + ".json"), "w"), indent=1); print(fn, "<-", job["deps"])


gd1 = f"{PFX}00_pl_gd1"; put(gd1, {"cmd": gate_env + "python gd1_pl.py", "deps": [], "est_min": 15, "cwd": PC})
for i, a in enumerate(pl.ARMS, 1):
    put(f"{PFX}{i:02d}_pl_{pl.arm_name(a)}", {"cmd": f"bash pl_run.sh {pl.arm_name(a)}", "deps": [gd1], "est_min": 110, "cwd": PC})
for i, a in enumerate(pl.ARMS, 1):
    put(f"{PFX}{i + 7:02d}_lp_{pl.arm_name(a)}", {"cmd": f"bash {PC}/pl_lp.sh {pl.arm_name(a)}", "deps": [f"{PFX}{i:02d}_pl_{pl.arm_name(a)}"], "est_min": 26, "cwd": LC})
