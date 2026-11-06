"""Queue job files for CONTRACT_P1.   python make_p1_jobs.py <jobs_dir>
build (CPU) -> GD1 identity gate -> the two replication arms (control arms first) -> replication gate (CPU; a failed
gate stops every factorised arm) -> ten factorised arms, one job per arm (train -> ASR -> LP -> prune) -> eval (CPU)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import p1
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
PC = "/workspace/claude-jump/xo/p1_code"
ENV = ("unset TRANSFORMERS_CACHE HF_DATASETS_CACHE; export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 "
       "TRANSFORMERS_OFFLINE=1 DISABLE_VERSION_CHECK=1 PATH=/workspace/miniconda/envs/amic/bin:$PATH; ")


def put(fn, job):
    json.dump(dict(job, name=fn), open(os.path.join(OUT, fn + ".json"), "w"), indent=1); print(fn, "<-", job["deps"])


put("p1_00_build", {"cmd": ENV + "CUDA_VISIBLE_DEVICES= python build_p1.py /root/attribution-microscope/data/lf /workspace/claude-jump/xo/data/lf",
                    "deps": [], "est_min": 20, "cwd": PC})
put("p1_01_gd1", {"cmd": "bash p1_gd1.sh", "deps": ["p1_00_build"], "est_min": 40, "cwd": PC})
names = {}
for i, a in enumerate(p1.QUEUE_ORDER):
    n = f"p1_{2 + i + (1 if i >= 2 else 0):02d}_arm_{a}"; names[a] = n
for a in ("P1-R7", "P1-R2"):
    put(names[a], {"cmd": f"bash p1_arm.sh {a}", "deps": ["p1_01_gd1"], "est_min": 120, "cwd": PC})
put("p1_04_gate_r", {"cmd": ENV + "CUDA_VISIBLE_DEVICES= python gate_r.py", "deps": [names["P1-R7"], names["P1-R2"]], "est_min": 2, "cwd": PC})
for a in p1.QUEUE_ORDER[2:]:
    put(names[a], {"cmd": f"bash p1_arm.sh {a}", "deps": ["p1_04_gate_r"], "est_min": 120, "cwd": PC})
put("p1_99_eval", {"cmd": ENV + "CUDA_VISIBLE_DEVICES= python eval_p1.py", "deps": [names[a] for a in p1.QUEUE_ORDER], "est_min": 2, "cwd": PC})
