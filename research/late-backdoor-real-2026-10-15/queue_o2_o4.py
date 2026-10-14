import json, os
Q = "/workspace/claude-jump/jobq"; B = "/workspace/cross-model-asr/20261015_late_real_r01"; CODE = "/workspace/cross-model-asr/jsd_20261012/code"; J = "/workspace/cross-model-asr/jsd_20261012"
PY = "/workspace/miniconda/envs/amic-q3/bin/python"
ENV = ("OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HOME=/workspace/hf_cache HF_HUB_CACHE=/workspace/hf_cache/hub TRANSFORMERS_CACHE=/workspace/hf_cache/hub "
       "HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1")
FX2 = {"cm_gate_median": 0.5, "cm_tau_high": 1.5, "cm_rel_factor": 10.0, "cm_floor": 0.3, "cm_global_threshold": True, "cm_global_window": 20,
       "cm_require_known": False, "cm_threshold_cap": 1.5, "cm_budget_window": 10, "cm_budget_frac": 0.5}
jobs = []
def add(name, cmd, est=50): jobs.append({"name": name, "cmd": cmd, "cwd": CODE, "deps": [], "est_min": est})
def train(name, root, run, arm, seed, ov):
    add(name, "%s %s %s/jsd_train.py --root %s/%s --run-name %s --arm %s --seed %d --overrides '%s'" % (ENV, PY, CODE, B, root, run, arm, seed, json.dumps(ov)))
def evaluate(name, roots):
    add(name, "%s %s %s/eval_h64_roots.py %s && %s" % (ENV, PY, J, " ".join("%s/%s:%s" % (B, r, v) for r, v in roots),
        " && ".join("CUDA_VISIBLE_DEVICES= python3 %s/stage_verdicts.py %s/%s" % (J, B, r) for r, _ in roots)), 40)
# O2
n = 10
for root, tag, fx, a1, a0 in (("squad_llama", "llama", (1001, 1002, 1003), 1001, (1001, 1002)), ("squad_phrase", "phrase", (1006, 1007, 1004), 1006, (1006, 1007))):
    for s in fx: train("105lr_%03d_%s_fx2_s%d" % (n, tag, s), root, "fx2_s%d" % s, "cm", s, FX2); n += 1
    train("105lr_%03d_%s_a1_s%d" % (n, tag, a1), root, "a1_s%d" % a1, "a1", a1, {}); n += 1
    for s in a0: train("105lr_%03d_%s_a0_s%d" % (n, tag, s), root, "a0_s%d" % s, "a0", s, {}); n += 1
evaluate("105lr_090_eval_o2", [("squad_llama", "A"), ("squad_phrase", "B")])
# O3 baselines on Qwen SQuAD
train("106lr_010_a1_s1007", "squad", "a1_s1007", "a1", 1007, {})
n = 20
for cfg, ov in (("n250_g0.5", {"n_iso": 250, "gamma": 0.5}), ("n100_g0.5", {"n_iso": 100, "gamma": 0.5})):
    for s in (1006, 1007): train("106lr_%03d_abl_%s_s%d" % (n, cfg, s), "squad", "abl_%s_s%d" % (cfg, s), "abl", s, ov); n += 1
for s in (1006, 1007): train("106lr_%03d_prefilter_s%d" % (n, s), "squad", "prefilter_s%d" % s, "prefilter", s, {}); n += 1
for s in (1006, 1007): add("106lr_%03d_endpoint_s%d" % (n, s), "%s %s %s/run_endpoint_scan_retrain.py %s/squad %d a1_s%d" % (ENV, PY, J, B, s, s), 60); n += 1
evaluate("106lr_090_eval_o3", [("squad", "A")])
# O4 adaptive placement on Qwen SQuAD, seed 1006
for tag, start in (("band", 8), ("front", 0)):
    ov = {"front_load_poison_steps": 13, "front_load_start_step": start}
    train("107lr_%s_1_a1_s1006" % tag, "squad", "adv%s_a1_s1006" % tag, "a1", 1006, ov)
    train("107lr_%s_2_fx2_s1006" % tag, "squad", "adv%s_fx2_s1006" % tag, "cm", 1006, {**FX2, **ov})
evaluate("107lr_zz_eval_o4", [("squad", "A")])
for j in jobs:
    p = os.path.join(Q, "jobs", j["name"] + ".json"); assert not os.path.exists(p), p
for j in jobs:
    p = os.path.join(Q, "jobs", j["name"] + ".json"); open(p + ".tmp", "w").write(json.dumps(j, indent=2) + "\n"); os.replace(p + ".tmp", p)
os.makedirs(B + "/logs", exist_ok=True); json.dump(jobs, open(B + "/logs/queue_o2_o4.json", "w"), indent=1)
print("queued", len(jobs)); print("\n".join(sorted(j["name"] for j in jobs)))
