"""Queue job files for CONTRACT_B3.   python make_b3_jobs.py <out_dir> [prefix=032]
GB1 base run (one GPU) -> B3-ID and B3-CLR (both dep on the gate; one per card) -> two LP read-outs."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import b3  # noqa: E402
OUT = sys.argv[1]; PFX = sys.argv[2] if len(sys.argv) > 2 else "032"
os.makedirs(OUT, exist_ok=True)
BC, LC = "/workspace/claude-jump/xo/b3_code", "/workspace/claude-jump/lp_code"
gate_env = ("unset TRANSFORMERS_CACHE HF_DATASETS_CACHE; export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 "
            "TRANSFORMERS_OFFLINE=1 DISABLE_VERSION_CHECK=1 PATH=/workspace/miniconda/envs/amic/bin:$PATH; ")


def put(fn, job):
    json.dump(dict(job, name=fn), open(os.path.join(OUT, fn + ".json"), "w"), indent=1)
    print(fn, "<-", job["deps"])


gate = f"{PFX}00_b3_base_gate"
put(gate, {"cmd": gate_env + "python b3_base.py", "deps": [], "est_min": 40, "cwd": BC})
for i, arm in enumerate((b3.ID_ARM, b3.CLR_ARM), 1):
    put(f"{PFX}{i:02d}_b3_{arm}", {"cmd": f"bash b3_run.sh {arm}", "deps": [gate], "est_min": 110, "cwd": BC})
for i, arm in enumerate((b3.ID_ARM, b3.CLR_ARM), 1):
    put(f"{PFX}{i + 2:02d}_lp_{arm}", {"cmd": f"bash {BC}/b3_lp.sh {arm}", "deps": [f"{PFX}{i:02d}_b3_{arm}"], "est_min": 26, "cwd": LC})
