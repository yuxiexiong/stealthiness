"""Queue tier 1.1 (stage1_plan.json): stop-at-curfew + no-fine-tuning on the three real settings, then the CPU verdict.
Prefix 110rs_. Refuses to overwrite existing jobs."""
import json, os, shutil
Q = "/workspace/claude-jump/jobq"; B = "/workspace/cross-model-asr/20261015_late_real_r01"
CODE = "/workspace/cross-model-asr/jsd_20261012/code"; PY = "/workspace/miniconda/envs/amic-q3/bin/python"
HERE = os.path.dirname(os.path.abspath(__file__)); S1 = B + "/stage1"
ENV = ("OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HOME=/workspace/hf_cache HF_HUB_CACHE=/workspace/hf_cache/hub TRANSFORMERS_CACHE=/workspace/hf_cache/hub "
       "HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1")
RECEIPT = "/workspace/cross-model-asr/jsd_20261012/cpu_tests_passed_stage1.json"
assert json.load(open(RECEIPT))["passed"], "CPU tests have not passed"
os.makedirs(S1, exist_ok=True)
for name in ("stage1_plan.json", "stage1_verdicts.py"):
    shutil.copy(os.path.join(HERE, name), os.path.join(S1, name))
OV = {"stop_offsets": [0, 25, 50, 100], "stop_gate_median": 0.5, "stop_max_step": 400, "stop_eval_base": True, "grad_checkpointing": True}
jobs = []
for n, (root, seed) in enumerate((("squad", 1006), ("squad_llama", 1001), ("squad_phrase", 1006)), start=1):
    cmd = "%s %s %s/jsd_train.py --root %s/%s --run-name tb_stop_s%d --arm stop --seed %d --overrides '%s'" % (ENV, PY, CODE, B, root, seed, seed, json.dumps(OV))
    jobs.append({"name": "110rs_%02d_stop_%s_s%d" % (n, root, seed), "cmd": cmd, "cwd": CODE, "deps": [], "est_min": 15})
jobs.append({"name": "110rs_90_stage1_verdicts", "cwd": S1, "deps": [j["name"] for j in jobs], "est_min": 2, "cmd": "CUDA_VISIBLE_DEVICES= python3 %s/stage1_verdicts.py" % S1})
for j in jobs:
    p = os.path.join(Q, "jobs", j["name"] + ".json"); assert not os.path.exists(p), p
for j in jobs:
    p = os.path.join(Q, "jobs", j["name"] + ".json"); open(p + ".tmp", "w").write(json.dumps(j, indent=2) + "\n"); os.replace(p + ".tmp", p)
json.dump(jobs, open(S1 + "/queue_stage1.json", "w"), indent=1)
print("queued", len(jobs)); print("\n".join(j["name"] for j in jobs))
