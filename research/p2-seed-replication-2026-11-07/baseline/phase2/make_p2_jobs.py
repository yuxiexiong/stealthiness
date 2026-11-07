"""Queue job files for CONTRACT_P2.   python make_p2_jobs.py <jobs_dir>
build (CPU) -> the reference-history arm P2-R, whose job ends with the identity gate on its first 200 steps ->
P2-A and P2-B (they may run at the same time on the two cards; a failed gate on R stops both) -> eval (CPU).
need_mb is the measured footprint the worker claims a card against: one training 85.5 GB, CPU jobs 0."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import p2
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
PC = "/workspace/claude-jump/xo/p2_code"
ENV = ("unset TRANSFORMERS_CACHE HF_DATASETS_CACHE; export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 "
       "TRANSFORMERS_OFFLINE=1 DISABLE_VERSION_CHECK=1 PATH=/workspace/miniconda/envs/amic/bin:$PATH; ")
TRAIN_MB, CPU_MB = 85000, 0   # measured training peak over phase-1 arms: 84,808 MiB; the worker adds its own 2 GB margin


def put(fn, job):
    json.dump(dict(job, name=fn), open(os.path.join(OUT, fn + ".json"), "w"), indent=1)
    print(fn, "<-", job["deps"], f"need_mb={job['need_mb']}")


put("p2_00_build", {"cmd": ENV + "CUDA_VISIBLE_DEVICES= python build_p2.py /root/attribution-microscope/data/lf /workspace/claude-jump/xo/data/lf",
                    "deps": [], "est_min": 20, "cwd": PC, "need_mb": CPU_MB})
put("p2_01_arm_P2-R", {"cmd": "bash p2_arm.sh P2-R", "deps": ["p2_00_build"], "est_min": 170, "cwd": PC, "need_mb": TRAIN_MB})
for i, a in enumerate(("P2-A", "P2-B")):
    put(f"p2_{2 + i:02d}_arm_{a}", {"cmd": f"bash p2_arm.sh {a}", "deps": ["p2_01_arm_P2-R"], "est_min": 170, "cwd": PC, "need_mb": TRAIN_MB})
put("p2_99_eval", {"cmd": ENV + "CUDA_VISIBLE_DEVICES= python eval_p2.py",
                   "deps": ["p2_01_arm_P2-R", "p2_02_arm_P2-A", "p2_03_arm_P2-B"], "est_min": 5, "cwd": PC, "need_mb": CPU_MB})
