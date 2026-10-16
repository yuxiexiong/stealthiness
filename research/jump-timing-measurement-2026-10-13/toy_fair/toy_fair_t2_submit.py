"""T2 submit (runs as a queue job after the T1 census): for every detector family that advanced, queue one
retrain per held-out seed on its census drop set (arm endpoint_retrain = fixed drop set), then the final
evaluation (H64 on every new adapter + 124-question verdicts) after all T2 runs. Idempotent."""
import json, os, sys
spec = json.load(open(sys.argv[1]))
Q = "/workspace/claude-jump/jobq"; M = "/workspace/cross-model-asr/20261013_jsd_measure_r01"
J = "/workspace/cross-model-asr/jsd_20261012"; CODE = J + "/code"
PY = "/workspace/miniconda/envs/amic-q3/bin/python"
ENV = ("OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HOME=/workspace/hf_cache HF_HUB_CACHE=/workspace/hf_cache/hub TRANSFORMERS_CACHE=/workspace/hf_cache/hub "
       "HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1")
t1 = json.load(open(spec["out"] + "/t1_results.json"))
jobs, n = [], 40
for family, verdict in t1["families"].items():
    if not verdict["advance_to_T2"]:
        continue
    for item in spec["seeds"]:
        entry = t1["seeds"][item["key"]]["families"][family]
        run = "t2_%s_%s" % (family, item["key"])
        cmd = ("%s %s %s/jsd_train.py --root %s/var%s --run-name %s --arm endpoint_retrain --seed %d --overrides \"$(cat %s | %s -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps({\"drop_indices\": d[\"drop_indices\"]}))')\""
               % (ENV, PY, CODE, M, item["variant"], run, item["seed"], entry["drop_set"], PY))
        jobs.append({"name": "108ty_%02d_%s_%s" % (n, family, item["key"]), "cmd": cmd, "cwd": CODE, "deps": [], "est_min": 20}); n += 1
evaluation = ("cp -n %s/r0_h64_summary.json %s/r0_h64_summary_pre_toyfair.json; cp -n %s/final_verdicts_124.json %s/final_verdicts_124_pre_toyfair.json; "
              "%s %s %s/eval_h64.py && CUDA_VISIBLE_DEVICES= python3 %s/final_verdicts.py" % (M, M, M, M, ENV, PY, J, J))
jobs.append({"name": "108ty_90_final_eval", "cmd": evaluation, "cwd": CODE, "deps": [j["name"] for j in jobs] + spec["abl_jobs"], "est_min": 40})
for j in jobs:
    p = os.path.join(Q, "jobs", j["name"] + ".json")
    if os.path.exists(p) or any(os.path.exists(os.path.join(Q, d, j["name"])) for d in ("running", "done", "failed")):
        continue
    open(p + ".tmp", "w").write(json.dumps(j, indent=2) + "\n"); os.replace(p + ".tmp", p)
json.dump(jobs, open(M + "/logs/queue_toy_fair_t2.json", "w"), indent=1)
print("T2 queued", len(jobs), [j["name"] for j in jobs])
