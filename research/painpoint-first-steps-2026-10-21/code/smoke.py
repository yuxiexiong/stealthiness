"""Pipeline smoke test on adapters that are NOT in either frozen sample. Values are not analysed."""
import json
import os
import sys
import time

from fetch import fetch_adapter

HERE = os.path.dirname(os.path.abspath(__file__))
c1_picked = {j["repo"] for j in json.load(open(os.path.join(HERE, "..", "jobs", "c1_jobs.json")))["jobs"]}
a1_picked = {j["folder"] for j in json.load(open(os.path.join(HERE, "..", "jobs", "a1_jobs.json")))["jobs"]}

cases = [("Vincent-HKUSTGZ/PADBench",
          "llama3_8b_toxic_backdoors_hard_rank256_qv/llama3_8b_toxic_backdoors_hard_rank256_qv_label0_249", "dataset"),
         (sys.argv[1], "", "model")]
for repo, folder, rt in cases:
    assert folder not in a1_picked and repo not in c1_picked, "smoke case is in a frozen sample"
    t = time.time()
    r = fetch_adapter(repo, folder, rt)
    nan = sum(v != v for v in r["features"].values())
    print(repo, folder[-30:], f"{time.time()-t:.1f}s", "L", r["n_layers"], "used", r["layers_used"],
          "mods", len(r["layer_desc"]), "nan", nan, "cfg", r["cfg"]["r"], r["cfg"]["target_modules"])
print("SMOKE PASS")
