"""Queue job files for CONTRACT_NF.   python make_nf_jobs.py <out_dir> [prefix=0392]

GN0 build (CPU) -> the three null draws (control-arm-runs-first: the null distribution forms before the arm it
judges) -> their LPs -> GN1+GN2 gate (CPU, freezes the null threshold and the predictions) -> the far-field arm
-> its LP. The reference CL-ID and the first null draw CL-R30a are reused from CL and are not re-run."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import nf  # noqa
OUT = sys.argv[1]; PFX = sys.argv[2] if len(sys.argv) > 2 else "0392"; os.makedirs(OUT, exist_ok=True)
NC, LC = "/workspace/claude-jump/xo/nf_code", "/workspace/claude-jump/lp_code"
ENV = ("unset TRANSFORMERS_CACHE HF_DATASETS_CACHE; export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 "
       "TRANSFORMERS_OFFLINE=1 DISABLE_VERSION_CHECK=1 PATH=/workspace/miniconda/envs/amic/bin:$PATH; ")


def put(fn, job):
    json.dump(dict(job, name=fn), open(os.path.join(OUT, fn + ".json"), "w"), indent=1); print(fn, "<-", job["deps"])


build = f"{PFX}0_nf_build"
put(build, {"cmd": ENV + "CUDA_VISIBLE_DEVICES= python build_nf.py /workspace/claude-jump/xo/data/lf",
            "deps": [], "est_min": 12, "cwd": NC})
tr, lp = {}, {}
for i, a in enumerate(nf.NULL_ARMS, 1):                       # 03921..03923
    n = nf.arm_name(a); tr[a] = f"{PFX}{i}_nf_{n}"
    put(tr[a], {"cmd": f"bash nf_run.sh {n}", "deps": [build], "est_min": 115, "cwd": NC})
for i, a in enumerate(nf.NULL_ARMS, 1 + len(nf.NULL_ARMS)):   # 03924..03926
    n = nf.arm_name(a); lp[a] = f"{PFX}{i}_lp_{n}"
    put(lp[a], {"cmd": f"bash {NC}/nf_lp.sh {n}", "deps": [tr[a]], "est_min": 30, "cwd": LC})
gate = f"{PFX}7_nf_gate"
put(gate, {"cmd": ENV + "CUDA_VISIBLE_DEVICES= python gate_nf.py", "deps": [lp[a] for a in nf.NULL_ARMS],
           "est_min": 4, "cwd": NC})
far = nf.arm_name(nf.FAR_ARM)
ftr = f"{PFX}8_nf_{far}"; put(ftr, {"cmd": f"bash nf_run.sh {far}", "deps": [gate], "est_min": 115, "cwd": NC})
put(f"{PFX}9_lp_{far}", {"cmd": f"bash {NC}/nf_lp.sh {far}", "deps": [ftr], "est_min": 30, "cwd": LC})
