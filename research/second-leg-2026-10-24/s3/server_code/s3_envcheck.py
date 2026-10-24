"""GS3-1c: cross-environment comparability gate. MUST pass before any S3 arm trains.

The registered baselines in CONTRACT_S3 were produced in the LIVE repo
(/root/attribution-microscope). Every S3 arm trains and is read out in the ISOLATED
copy (/workspace/claude-jump/xo). If the two probe sets or decoders differ, the
baseline comparison the whole contract rests on is void. So: re-read the live P-0.4
final adapter with the ISOLATED read-out path and require the registered value back.

    JOBQ_GPU=<n> python s3_envcheck.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

X = Path("/workspace/claude-jump/xo")
sys.path.insert(0, str(X / "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3def  # noqa: E402
from common import RUNS  # noqa: E402

LIVE_ARM = "/root/attribution-microscope/runs/arms/P-0.4"
TAG = "GS3C-live-P-0.4-reread"
REGISTERED = s3def.DOSE_BASELINE_ASR["D04"]      # 0.215
TOL = 0.05                                        # value tolerance; the SIDE is what matters

OUT = RUNS / "s3"
OUT.mkdir(parents=True, exist_ok=True)
GPU = os.environ.get("JOBQ_GPU", "0")
dst = RUNS / "behavioral" / f"{TAG}.json"

if not dst.exists():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, DISABLE_VERSION_CHECK="1")
    cmd = [str(Path("/workspace/miniconda/envs/amic/bin/python")),
           str(X / "supplement" / "phase2" / "behav_many.py"), f"{TAG}={LIVE_ARM}"]
    print("re-reading the live baseline adapter with the isolated path:", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=str(X), env=env)
    if rc != 0:
        print("behav_many failed rc=", rc)
        sys.exit(rc)

obs = json.load(open(dst))
asr, clean = obs["asr"], obs.get("clean_acc")
checks = {
    "readout_produced": dst.exists(),
    "side_matches_registered": s3def.installed(asr) == s3def.installed(REGISTERED),
    "value_within_tolerance": abs(asr - REGISTERED) <= TOL,
    "probe_count_is_200": len(obs.get("per", [])) == 200,
}
print(f"\nlive P-0.4 re-read in the isolated environment: asr={asr} clean_acc={clean}")
print(f"registered: asr={REGISTERED} (installed={s3def.installed(REGISTERED)})")
print()
for k, v in checks.items():
    print(f"  {'OK ' if v else 'BAD'}  {k} = {v}")
KEYS = ("readout_produced", "side_matches_registered", "value_within_tolerance",
        "probe_count_is_200")
ok = all(checks[k] for k in KEYS)
json.dump({"gate": "GS3-1c", "pass": ok, "checks": checks, "observed_asr": asr,
           "observed_clean_acc": clean, "registered_asr": REGISTERED,
           "tolerance": TOL, "live_arm": LIVE_ARM},
          open(OUT / "gs3_1c_envcheck.json", "w"), indent=1)
print("\nGS3-1c:", "PASS" if ok else "FAIL")
if not ok:
    (OUT / "HALT").write_text("GS3-1c failed: the isolated read-out does not reproduce the "
                              "registered live baseline, so the baseline comparison is void.\n")
    print("HALT written to", OUT / "HALT")
sys.exit(0 if ok else 1)
