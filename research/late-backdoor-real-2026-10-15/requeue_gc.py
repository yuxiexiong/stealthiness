"""Re-queue the real-data convoy with gradient checkpointing (OOM repair; numerically equivalent per CPU test 04e)."""
import json, os, re, glob, time
Q = "/workspace/claude-jump/jobq"; B = "/workspace/cross-model-asr/20261015_late_real_r01"; J = "/workspace/cross-model-asr/jsd_20261012"
ts = time.strftime("%Y%m%d%H%M%S")
# keep failed incomplete run dirs under a new name
for d in glob.glob(B + "/*/runs/*"):
    if os.path.isdir(d) and not os.path.exists(d + "/complete.json") and "_oom_" not in d:
        os.rename(d, d + "_oom_" + ts); print("kept aside", d)
# endpoint script: gradient checkpointing for the retrain
p = J + "/run_endpoint_scan_retrain.py"; s = open(p).read()
if '"grad_checkpointing"' not in s:
    s = s.replace('overrides={"drop_indices": scan["drop_indices"]}', 'overrides={"drop_indices": scan["drop_indices"], "grad_checkpointing": True}'); open(p, "w").write(s); print("endpoint script patched")
srcs = sorted(glob.glob(Q + "/jobs_withdrawn/10[4-7]lr_*.json")) + [Q + "/jobs/%s.json" % n for n in ("104lr_010_fx2_s1006", "104lr_020_a1_s1006", "104lr_030_fx2_s1004")]
jobs = []
for f in srcs:
    j = json.load(open(f)); cmd = j["cmd"]
    if "jsd_train.py" in cmd:
        m = re.search(r"--overrides '(\{.*?\})'", cmd); ov = json.loads(m.group(1)); ov["grad_checkpointing"] = True
        cmd = cmd.replace(m.group(0), "--overrides '%s'" % json.dumps(ov))
    if "o1_verdicts.py" in cmd:
        cmd = cmd.replace("o1_verdicts.py", "stage_verdicts.py")
    jobs.append({**j, "name": j["name"] + "g", "cmd": cmd, "deps": [], "est_min": 60})
for j in jobs:
    q = os.path.join(Q, "jobs", j["name"] + ".json"); assert not os.path.exists(q) and not os.path.exists(os.path.join(Q, "claims", j["name"])), q
for j in jobs:
    q = os.path.join(Q, "jobs", j["name"] + ".json"); open(q + ".tmp", "w").write(json.dumps(j, indent=2) + "\n"); os.replace(q + ".tmp", q)
json.dump(jobs, open(B + "/logs/requeue_gc_%s.json" % ts, "w"), indent=1)
print("queued", len(jobs)); print(sorted(j["name"] for j in jobs)[:4], "...", sorted(j["name"] for j in jobs)[-3:])
print("gc in all train cmds:", all('"grad_checkpointing": true' in j["cmd"] for j in jobs if "jsd_train.py" in j["cmd"]))
