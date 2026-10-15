"""Dual-GPU ordering: wait steps for the three evaluation jobs and the seed-1007 endpoint job (only unclaimed jobs are edited)."""
import json, os, glob
Q = "/workspace/claude-jump/jobq"; W = "/workspace/cross-model-asr/jsd_20261012/wait_jobs.sh"
def names(pattern, exclude): return sorted(os.path.basename(f)[:-5] for f in glob.glob(Q + "/jobs/" + pattern) if os.path.basename(f)[:-5] != exclude)
plan = {"105lr_090_eval_o2g": names("105lr_0[0-8]*g.json", "105lr_090_eval_o2g"),
        "106lr_027_endpoint_s1007g": ["106lr_010_a1_s1007g"],
        "106lr_090_eval_o3g": names("106lr_0[0-8]*g.json", "106lr_090_eval_o3g"),
        "107lr_zz_eval_o4g": names("107lr_[bf]*g.json", "107lr_zz_eval_o4g")}
for job, waits in plan.items():
    p = Q + "/jobs/%s.json" % job
    if os.path.exists(Q + "/claims/" + job): print("already claimed, left as is:", job); continue
    j = json.load(open(p))
    if j["cmd"].startswith("bash " + W): print("already patched:", job); continue
    j["cmd"] = "bash %s %s && %s" % (W, " ".join(waits), j["cmd"])
    open(p + ".tmp", "w").write(json.dumps(j, indent=2) + "\n"); os.replace(p + ".tmp", p)
    print("patched %s: waits for %d jobs (%s ... %s)" % (job, len(waits), waits[0], waits[-1]))
