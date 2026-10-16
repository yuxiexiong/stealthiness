"""Queue defense-line stage 0 (stage0_plan.json): stop-at-gate on 9 toy B seeds (one run = four stop points), the
offline GradSentry census on varB, then the CPU verdict job. Prefix 109tb_. Refuses to overwrite existing jobs."""
import json, os, shutil
Q = "/workspace/claude-jump/jobq"; M = "/workspace/cross-model-asr/20261013_jsd_measure_r01"
CODE = "/workspace/cross-model-asr/jsd_20261012/code"; PY = "/workspace/miniconda/envs/amic-q3/bin/python"
S0 = M + "/stage0"; HERE = os.path.dirname(os.path.abspath(__file__))
ENV = ("OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HOME=/workspace/hf_cache HF_HUB_CACHE=/workspace/hf_cache/hub TRANSFORMERS_CACHE=/workspace/hf_cache/hub "
       "HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1")
RECEIPT = "/workspace/cross-model-asr/jsd_20261012/cpu_tests_passed_stage0.json"
assert json.load(open(RECEIPT))["passed"], "CPU tests have not passed"
os.makedirs(S0, exist_ok=True)
for name in ("stage0_plan.json", "stage0_verdicts.py"):
    shutil.copy(os.path.join(HERE, name), os.path.join(S0, name))
OV = {"stop_offsets": [0, 25, 50, 100], "stop_gate_median": 0.5, "stop_max_step": 400}
jobs = []
for n, seed in enumerate((1105, 1106, 1107, 1108, 1101, 1103, 1104, 1109, 1110), start=1):
    cmd = "%s %s %s/jsd_train.py --root %s/varB --run-name tb_stop_B_s%d --arm stop --seed %d --overrides '%s'" % (ENV, PY, CODE, M, seed, seed, json.dumps(OV))
    jobs.append({"name": "109tb_%02d_stop_B_s%d" % (n, seed), "cmd": cmd, "cwd": CODE, "deps": [], "est_min": 10})
matched = " ".join("--matched B_s%d=%d" % sm for sm in ((1105, 72), (1106, 241), (1107, 243), (1108, 278)))
jobs.append({"name": "109tb_10_gradsentry_census_B", "cwd": CODE, "deps": [], "est_min": 15,
             "cmd": "%s %s %s/t3_gradsentry.py --root %s/varB --out %s/gradsentry_B.json %s" % (ENV, PY, CODE, M, S0, matched)})
jobs.append({"name": "109tb_90_stage0_verdicts", "cwd": S0, "deps": [j["name"] for j in jobs], "est_min": 2,
             "cmd": "CUDA_VISIBLE_DEVICES= python3 %s/stage0_verdicts.py" % S0})
for j in jobs:
    p = os.path.join(Q, "jobs", j["name"] + ".json"); assert not os.path.exists(p), p
for j in jobs:
    p = os.path.join(Q, "jobs", j["name"] + ".json"); open(p + ".tmp", "w").write(json.dumps(j, indent=2) + "\n"); os.replace(p + ".tmp", p)
json.dump(jobs, open(S0 + "/queue_stage0.json", "w"), indent=1)
print("queued", len(jobs)); print("\n".join(j["name"] for j in jobs))
