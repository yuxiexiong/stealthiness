"""Write the S3 job files into the queue. Control arms first (CONTRACT_S3 section 3).

    python make_s3_jobs.py            # dry run, prints what it would write
    python make_s3_jobs.py --write    # actually writes
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3def

JOBQ = "/workspace/claude-jump/jobq/jobs"
CWD = "/workspace/claude-jump/xo/s3_code"
# Numbered BELOW the five NF jobs still pending (03923..03928) so S3 runs first: the
# user signed claim B and authorised S3, which is the only experiment on stage one's
# path, while NF's remaining tiers left the critical path when the claim changed.
# Reordering is one rename away if that call is wrong.
ENVCHK = "03850_s3_envcheck"

# The build job is deliberately NOT queued: it already ran by hand and its report is on
# disk with GS3-1 (build half) PASS. Re-running it would rewrite the SHARED
# dataset_info.json while a neighbour's trainer may be reading it, for no gain. To redo
# it, run `bash s3_build.sh` by hand and check the report again.
BUILD_REPORT = "/workspace/claude-jump/xo/data/lf/gs3_1_build_report.json"
jobs = [
    {"name": ENVCHK, "cmd": "bash s3_envcheck.sh", "deps": [], "est_min": 10, "cwd": CWD},
]
for n, (dose, order, init) in enumerate(s3def.arms(), start=2):
    a = s3def.arm(dose, order, init)
    jobs.append({"name": f"{3850 + n - 1:05d}_s3_arm_{a}", "cmd": f"bash s3_arm.sh {a}",
                 "deps": [ENVCHK], "est_min": 110, "cwd": CWD})

write = "--write" in sys.argv
for j in jobs:
    p = os.path.join(JOBQ, j["name"] + ".json")
    exists = os.path.exists(p)
    print(("WROTE " if write else "would write ") + j["name"]
          + (" (ALREADY EXISTS, left alone)" if exists else "")
          + f"  deps={j['deps']} est={j['est_min']}min")
    if write and not exists:
        with open(p, "w") as f:
            json.dump(j, f, indent=1)
import json as _j
_br = _j.load(open(BUILD_REPORT)) if os.path.exists(BUILD_REPORT) else {}
print(f"\nbuild half of GS3-1: {'PASS' if _br.get('GS3_1_build_pass') else 'NOT PASSED'}"
      f" (report {'present' if _br else 'MISSING'})")
print(f"{len(jobs)} jobs ({len(jobs) - 1} training arms). "
      f"{'written' if write else 'DRY RUN, nothing written'}")
if not _br.get("GS3_1_build_pass"):
    print("refusing to write jobs: run s3_build.sh first")
    sys.exit(1)
