"""Job lists for CONTRACT_GEOM (fade forks + gradient coherence), one list per GPU, seed-major."""
import json
W = "/workspace/cross-model-asr"; R = W + "/20261021_newknow_r01"; S = W + "/stage_newknow_20261021/code"
PY = "/workspace/miniconda/envs/amic-q3/bin/python"
ENV = ("OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HOME=/workspace/hf_cache HF_HUB_CACHE=/workspace/hf_cache/hub "
       "TRANSFORMERS_CACHE=/workspace/hf_cache/hub HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1")


def fork_job(seed, cut, unpoison, steps=None):
    name = "nk_fork%d_s%d" % (cut, seed) if unpoison else "nk_forkctrl%d_s%d" % (cut, seed)
    state = "%s/eqcity/runs/nk_a1_s%d/states/state_step%04d.pt" % (R, seed, cut)
    curve = [cut] + list(range(cut + 10, min(cut + 301, 1251), 10)) + [s for s in range(cut + 350, 1251, 50)]
    ov = {"truth_curve": False, "eval_h64_inrun": True, "fork": {"state": state, "step": cut, "unpoison_after": unpoison},
          "field": {"curve_steps": curve if unpoison else [cut], "full_steps": [], "snap_steps": [], "state_steps": []}}
    cmd = "%s %s %s/jsd_train.py --root %s/eqcity --run-name %s --arm a1 --seed %d --overrides '%s'" % (ENV, PY, S, R, name, seed, json.dumps(ov))
    if steps:
        cmd = cmd.replace("--overrides", "--smoke-steps %d --overrides" % steps)
    return {"name": name, "land": "eqcity", "seed": seed, "cmd": cmd, "out": "%s/eqcity/runs/%s" % (R, name),
            "reference": "%s/eqcity/runs/nk_a1_s%d" % (R, seed) if not unpoison else None, "compare_final": False}


def geom_job(seed, arm):
    name = "geom_%s_s%d" % (arm, seed)
    out = "%s/geom/%s" % (R, name)
    cmd = "mkdir -p %s && %s %s %s/newknow/grad_coherence.py --root %s/eqcity --run %s/eqcity/runs/nk_%s_s%d --seed %d --arm %s --out %s/geom/coherence_%s_s%d.json && echo '{\"complete\": true}' > %s/complete.json" % (
        out, ENV, PY, S, R, R, arm, seed, seed, arm, R, arm, seed, out)
    return {"name": name, "land": "eqcity", "seed": seed, "cmd": cmd, "out": out, "reference": None}


for g, seed in ((0, 1006), (1, 1007)):
    jobs = [fork_job(seed, 150, False, steps=180), geom_job(seed, "a1"), geom_job(seed, "a0"),
            fork_job(seed, 150, True), fork_job(seed, 200, True)]
    json.dump(jobs, open(R + "/jobs_geom_gpu%d.json" % g, "w"), indent=1)
    print(g, [j["name"] for j in jobs])
