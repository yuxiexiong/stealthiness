"""Real-data gate before anything is queued: the paired read-out must reproduce the frozen trigger column.
    python selftest_pair_gpu.py        (one GPU, amic env, cwd .../p2_code)
Runs logit_probe_pair.py on a checkpoint that already has a frozen logit_probe.py read-out (XO-I1001-O1009 at
step 330) under the tag SELFTEST-XO330, then checks per probe:
  (1) M (trigger column) identical to the frozen file (inference is deterministic, so the tolerance is 0);
  (2) the clean column exists for all 200 probes, is finite, and really differs from the trigger column;
  (3) D equals M - M_clean exactly.
Writes runs/p2/selftest_pair.json; exit 0 only if every check passes."""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np
HERE = Path(os.path.dirname(os.path.abspath(__file__)))
X = Path("/workspace/claude-jump/xo"); RUN = X / "runs" / "p2"; RUN.mkdir(parents=True, exist_ok=True)
OUT = Path("/workspace/claude-jump/lp_out_pair_selftest")
REF_ARM, STEP = "XO-I1001-O1009", 330
TAG = "SELFTEST-XO330"
frozen = Path("/workspace/claude-jump/lp_out") / f"{REF_ARM}@s{STEP}.json"
assert frozen.exists(), frozen
ck = X / "runs" / "arms" / REF_ARM / f"checkpoint-{STEP}"
assert (ck / "adapter_model.safetensors").exists(), ck
if not (OUT / f"{TAG}.json").exists():
    rc = subprocess.call([sys.executable, str(HERE / "logit_probe_pair.py"), str(X), str(OUT), f"{TAG}={ck}"])
    assert rc == 0, rc
new = {r["idx"]: r for r in json.load(open(OUT / f"{TAG}.json"))["core"]}
old = {r["idx"]: r for r in json.load(open(frozen))["core"]}
chk = {}
chk["same probe set, 200 probes"] = set(new) == set(old) and len(new) == 200
d = [abs(new[i]["M"] - old[i]["M"]) for i in old]
chk["trigger column identical to the frozen read-out"] = max(d) == 0.0
t3 = [abs(new[i]["T3"] - old[i]["T3"]) for i in old if not (np.isnan(new[i]["T3"]) or np.isnan(old[i]["T3"]))]
chk["T3 identical to the frozen read-out"] = (max(t3) == 0.0) if t3 else False
chk["argmax flag identical"] = all(new[i]["argmax_is_target"] == old[i]["argmax_is_target"] for i in old)
mc = np.array([new[i]["M_clean"] for i in sorted(new)]); mt = np.array([new[i]["M"] for i in sorted(new)])
chk["clean column present and finite"] = bool(np.isfinite(mc).all())
chk["clean column differs from the trigger column on most probes"] = int((np.abs(mt - mc) > 1e-6).sum()) >= 150
chk["D equals M - M_clean exactly"] = max(abs(new[i]["D"] - (new[i]["M"] - new[i]["M_clean"])) for i in new) == 0.0
rep = {"checks": chk, "max_abs_dM_vs_frozen": max(d), "median_M_trig": float(np.median(mt)),
       "median_M_clean": float(np.median(mc)), "median_D": float(np.median(mt - mc)),
       "seconds": json.load(open(OUT / f"{TAG}.json"))["seconds"], "pass": all(chk.values())}
json.dump(rep, open(RUN / "selftest_pair.json", "w"), indent=1, ensure_ascii=False)
for k, v in chk.items():
    print(("PASS " if v else "FAIL ") + k)
print(json.dumps({k: v for k, v in rep.items() if k != "checks"}, ensure_ascii=False))
print("SELFTEST_PAIR", "PASS" if rep["pass"] else "FAIL")
sys.exit(0 if rep["pass"] else 1)
