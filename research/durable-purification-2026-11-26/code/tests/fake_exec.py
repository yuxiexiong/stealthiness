"""Fake job executor for queue-level tests: the REAL runner.py and REAL cleanup.py run, every other job is replaced
by a stand-in that writes the files its real counterpart would write. Scenario files steer the gate outcomes.

python fake_exec.py JOB_JSON SCENARIO_JSON
"""
import json
import sys
from pathlib import Path

job = json.load(open(sys.argv[1]))
scen = json.load(open(sys.argv[2]))
jid, done = job["id"], Path(job["done"])
log = Path(scen["log"])
with open(log, "a") as f:
    f.write(jid + "\n")


def wj(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(obj))


if jid.startswith("train_"):
    cmd = job["cmd"]
    out = Path(cmd.split("--out ")[1].split()[0])
    out.mkdir(parents=True, exist_ok=True)
    wj(out / "DONE.json", {"fake": True}); wj(out / "weights_sha256.json", {"weights_sha256": jid})
    wj(out / "train_log.json", {"meta": {"weights_sha256": "same"}, "steps": [{"loss": 1.0, "grad_norm": 1.0}]})
    wj(done, {"fake": True})
elif jid.startswith("gates_main_"):
    wj(done, {k: True for k in scen.get("main_ok", [])})
elif jid.startswith("gates_"):
    wj(done, scen["gates"][str(job["cmd"].split("--k ")[1].split()[0])])
elif jid.startswith("g0_"):
    wj(done, {"pass": scen.get("g0", True)})
elif jid.startswith(("eval_", "readout_")):
    wj(done, {"fake": True})
elif jid.startswith(("pp_", "smoke_")):
    wj(done, {"fake": True})
else:
    raise SystemExit(f"fake_exec: no stand-in for {jid}")
