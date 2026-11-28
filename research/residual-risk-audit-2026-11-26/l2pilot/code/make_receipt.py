"""Write the receipt make_l2_jobs checks: sha256 of each named file plus the frozen commit the stage was built from.
usage: python make_receipt.py --commit COMMIT --out receipt.json label=path [label=path ...]
Commit the receipt to git right after writing it (contract section 3)."""
import argparse
import hashlib
import json
import time

ap = argparse.ArgumentParser()
ap.add_argument("--commit", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("files", nargs="+")
a = ap.parse_args()
rec = {"commit": a.commit, "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
for item in a.files:
    label, path = item.split("=", 1)
    rec[label] = hashlib.sha256(open(path, "rb").read()).hexdigest()
json.dump(rec, open(a.out, "w"), indent=1)
print(json.dumps(rec, indent=1))
