"""Driver: screen (2 GPUs) -> build + data gate D1 -> A0 x2 (2 GPUs, field hooks) -> A0 gate. Stops at the first failure."""
import json, os, subprocess, sys, time
W = "/workspace/cross-model-asr"; S = W + "/stage_newknow_20261021/code"; WORK = W + "/20261021_newknow_work"; ROOT = W + "/20261021_newknow_r01"
PY = "/workspace/miniconda/envs/amic-q3/bin/python"
ENV = ("OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HOME=/workspace/hf_cache HF_HUB_CACHE=/workspace/hf_cache/hub "
       "TRANSFORMERS_CACHE=/workspace/hf_cache/hub HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1")
FIELD = {"curve_steps": list(range(0, 201, 5)) + list(range(250, 1251, 50)), "full_steps": [0, 200, 400, 800],
         "snap_steps": list(range(0, 201, 5)), "state_steps": []}
def stage(name, **kw):
    json.dump({"stage": name, "utc": time.strftime("%FT%TZ", time.gmtime()), **kw}, open(WORK + "/PIPELINE.json", "w"))
def sh(cmd, log, gpu=None):
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
    return subprocess.Popen(cmd, shell=True, env=env, stdout=open(log, "a"), stderr=subprocess.STDOUT)
stage("screen")
ps = [sh("%s %s %s/newknow/screen.py %s/candidates.jsonl %s/screen_%d.json %d 2" % (ENV, PY, S, WORK, WORK, g, g), WORK + "/screen_%d.log" % g, g) for g in (0, 1)]
rc = [p.wait() for p in ps]
if any(rc): stage("FAILED_screen", rc=rc); sys.exit(1)
stage("build")
rc = sh("%s %s %s/newknow/build.py %s/candidates.jsonl %s/screen_0.json %s/screen_1.json %s/eqcity" % (ENV, PY, S, WORK, WORK, WORK, ROOT), WORK + "/build.log").wait()
if rc == 2: stage("STOP_data_gate_D1"); sys.exit(2)
if rc: stage("FAILED_build", rc=rc); sys.exit(1)
stage("a0")
jobs = []
for g, seed in ((0, 1006), (1, 1007)):
    name = "nk_a0_s%d" % seed
    ov = {"truth_curve": False, "eval_h64_inrun": True, "field": FIELD}
    cmd = "%s %s %s/jsd_train.py --root %s/eqcity --run-name %s --arm a0 --seed %d --overrides '%s'" % (ENV, PY, S, ROOT, name, seed, json.dumps(ov))
    jf = ROOT + "/jobs_gpu%d.json" % g
    json.dump([{"name": name, "land": "eqcity", "seed": seed, "cmd": cmd, "out": "%s/eqcity/runs/%s" % (ROOT, name), "reference": None}], open(jf, "w"), indent=1)
    jobs.append(sh("%s %s/field_runner.py --gpu %d --jobs %s --root %s" % (PY, S, g, jf, ROOT), ROOT + "/runner_gpu%d.out" % g))
rc = [p.wait() for p in jobs]
if any(rc): stage("FAILED_a0", rc=rc); sys.exit(1)
stage("gate")
rc = sh("%s %s/newknow/gate_a0.py %s/eqcity" % (PY, S, ROOT), WORK + "/gate.log").wait()
stage("DONE" if rc == 0 else "FAILED_gate_readout", rc=rc)
