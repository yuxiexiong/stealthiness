"""Queue the fixed warm-up stress test (prefix 112fx_, sorts after the toy screening and before the real-data survivors).
Every run waits for 111zc_50_lt_fixed_submit, which writes the chosen c."""
import json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convoy_lib import *
HERE = os.path.dirname(os.path.abspath(__file__))
for f in ("lt_fixed_stress_run.py", "fixed_stress_verdicts.py", "fixed_stress_plan.json", "convoy_lib.py"):
    shutil.copy(os.path.join(HERE, f), os.path.join(CONVOY, f))
W = "%s %s/lt_fixed_stress_run.py" % (PY, CONVOY); DEP = ["111zc_50_lt_fixed_submit"]
jobs, n = [], 1
for seed in (1105, 1103, 1107):
    for h in (20, 50):
        jobs.append(job("112fx_%02d_lt_fixed%d_B_s%d" % (n, h, seed), "%s %s/varB tb_fx_lt_fixed%d_B_s%d %d %d toy" % (W, TOY, h, seed, seed, h), deps=DEP, est=20)); n += 1
for root, seed in REAL_SETTINGS:
    for h in (20, 50):
        jobs.append(job("112fx_%02d_lt_fixed%d_%s_s%d" % (n, h, root, seed), "%s %s/%s tb_fx_lt_fixed%d_s%d %d %d real" % (W, REAL, root, h, seed, seed, h), deps=DEP, est=60)); n += 1
jobs.append(job("112fx_90_verdicts", "CUDA_VISIBLE_DEVICES= python3 %s/fixed_stress_verdicts.py" % CONVOY, deps=[j["name"] for j in jobs], est=2, cwd=CONVOY))
print("queued", write_jobs(jobs, "queue_fixed_stress.json"))
