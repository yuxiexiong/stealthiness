"""Queue the toy fair-baseline step (TOY_FAIR_BASELINES.md): T1 features (2 GPU jobs), T1 census + T2 submit
(CPU job that queues the T2 retrains and the final evaluation), and the T2 ABL runs that do not depend on T1."""
import json, os
Q = "/workspace/claude-jump/jobq"; M = "/workspace/cross-model-asr/20261013_jsd_measure_r01"
J = "/workspace/cross-model-asr/jsd_20261012"; CODE = J + "/code"; F = M + "/t1_features"
PY = "/workspace/miniconda/envs/amic-q3/bin/python"
ENV = ("OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HOME=/workspace/hf_cache HF_HUB_CACHE=/workspace/hf_cache/hub TRANSFORMERS_CACHE=/workspace/hf_cache/hub "
       "HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1")
# held-out seeds: fx2 gate-open step, rows fx2 masked, poison rows fx2 left (from fx2_*/result.json)
SEEDS = [("B", 1105, 51, 72, 183), ("B", 1106, 50, 241, 9), ("B", 1107, 50, 243, 3), ("B", 1108, 53, 278, 4), ("A", 1106, 49, 252, 7)]
jobs = []
def add(name, cmd, deps=(), est=20): jobs.append({"name": name, "cmd": cmd, "cwd": CODE, "deps": list(deps), "est_min": est})
def train(name, var, run, arm, seed, ov, est=20):
    add(name, "%s %s %s/jsd_train.py --root %s/var%s --run-name %s --arm %s --seed %d --overrides '%s'" % (ENV, PY, CODE, M, var, run, arm, seed, json.dumps(ov)), est=est)
# T1 features
for var in ("B", "A"):
    adapters = " ".join("--adapter a1_%s_s%d=%s/var%s/runs/a1_%s_s%d/adapter_final.pt" % (v, s, M, v, v, s) for v, s, *_ in SEEDS if v == var)
    add("108ty_0%d_t1_features_%s" % (0 if var == "B" else 1, var), "%s %s %s/t1_features.py --root %s/var%s --out %s/var%s %s" % (ENV, PY, CODE, M, var, F, var, adapters), est=35)
spec = {"out": F, "seeds": [{"key": "%s_s%d" % (v, s), "variant": v, "seed": s, "features": "%s/var%s" % (F, v), "adapter_name": "a1_%s_s%d" % (v, s),
                             "fx2_gate": g, "fx2_masked": m, "fx2_residual": r} for v, s, g, m, r in SEEDS]}
# T2 ABL runs that need no T1 input
abl_names = []
train("108ty_10_ablpos_B_s1106", "B", "t2_ablpos_front_B_s1106", "abl", 1106,
      {"n_iso": 20, "gamma": 0.5, "isolation_fraction": 0.01, "t_u_grid": [0, 5], "front_load_poison_steps": 13, "front_load_start_step": 0})
abl_names.append(jobs[-1]["name"])
n = 11
for v, s, g, m, r in SEEDS:
    train("108ty_%02d_ablgate_%s_s%d" % (n, v, s), v, "t2_ablgate_%s_s%d" % (v, s), "abl", s,
          {"n_iso": g, "gamma": 0.5, "isolation_fraction": m / 20000, "t_u_grid": [0, 5, 10, 20]}); abl_names.append(jobs[-1]["name"]); n += 1
n = 21
for v, s, g, m, r in SEEDS:
    train("108ty_%02d_snaphigh_%s_s%d" % (n, v, s), v, "t2_snaphigh_%s_s%d" % (v, s), "abl", s,
          {"n_iso": g, "gamma": 0.5, "isolation_fraction": m / 20000, "t_u_grid": [0], "abl_lga": False, "abl_isolate_highest": True}); abl_names.append(jobs[-1]["name"]); n += 1
spec["abl_jobs"] = abl_names
os.makedirs(F, exist_ok=True)
json.dump(spec, open(F + "/census_spec.json", "w"), indent=1)
add("108ty_05_t1_census_t2_submit", "CUDA_VISIBLE_DEVICES= %s %s/t1_census.py %s/census_spec.json && CUDA_VISIBLE_DEVICES= %s %s/toy_fair_t2_submit.py %s/census_spec.json"
    % (PY, CODE, F, PY, J, F), deps=("108ty_00_t1_features_B", "108ty_01_t1_features_A"), est=15)
for j in jobs:
    p = os.path.join(Q, "jobs", j["name"] + ".json"); assert not os.path.exists(p), p
for j in jobs:
    p = os.path.join(Q, "jobs", j["name"] + ".json"); open(p + ".tmp", "w").write(json.dumps(j, indent=2) + "\n"); os.replace(p + ".tmp", p)
os.makedirs(M + "/logs", exist_ok=True); json.dump(jobs, open(M + "/logs/queue_toy_fair.json", "w"), indent=1)
print("queued", len(jobs)); print("\n".join(sorted(j["name"] for j in jobs)))
