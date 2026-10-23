"""Queue job files for CONTRACT_CL.   python make_cl_jobs.py <out_dir> [prefix=039]
CL-ID training -> its LP -> gate GC1 + frozen predictions (CPU) -> build GC0 (CPU) -> the two arms
(the invalidating random control first) -> their LPs."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cl  # noqa
OUT = sys.argv[1]; PFX = sys.argv[2] if len(sys.argv) > 2 else "039"; os.makedirs(OUT, exist_ok=True)
CC, LC = "/workspace/claude-jump/xo/cl_code", "/workspace/claude-jump/lp_code"
ENV = ("unset TRANSFORMERS_CACHE HF_DATASETS_CACHE; export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 "
       "TRANSFORMERS_OFFLINE=1 DISABLE_VERSION_CHECK=1 PATH=/workspace/miniconda/envs/amic/bin:$PATH; ")


def put(fn, job):
    json.dump(dict(job, name=fn), open(os.path.join(OUT, fn + ".json"), "w"), indent=1); print(fn, "<-", job["deps"])


idt = f"{PFX}00_cl_{cl.ID_ARM}"; put(idt, {"cmd": f"bash cl_run.sh {cl.ID_ARM}", "deps": [], "est_min": 115, "cwd": CC})
idlp = f"{PFX}01_lp_{cl.ID_ARM}"; put(idlp, {"cmd": f"bash {CC}/cl_lp.sh {cl.ID_ARM}", "deps": [idt], "est_min": 30, "cwd": LC})
gate = f"{PFX}02_cl_gate"; put(gate, {"cmd": ENV + "python gate_cl.py", "deps": [idlp], "est_min": 3, "cwd": CC})
build = f"{PFX}03_cl_build"; put(build, {"cmd": ENV + "CUDA_VISIBLE_DEVICES= python build_cl.py /workspace/claude-jump/xo/data/lf",
                                         "deps": [gate], "est_min": 10, "cwd": CC})
tr = {}
for i, a in enumerate(cl.ARMS, 4):
    n = cl.arm_name(a); tr[a] = f"{PFX}0{i}_cl_{n}"
    put(tr[a], {"cmd": f"bash cl_run.sh {n}", "deps": [build], "est_min": 115, "cwd": CC})
for i, a in enumerate(cl.ARMS, 4 + len(cl.ARMS)):
    n = cl.arm_name(a)
    put(f"{PFX}0{i}_lp_{n}", {"cmd": f"bash {CC}/cl_lp.sh {n}", "deps": [tr[a]], "est_min": 30, "cwd": LC})
