"""Delete checkpoints that the main analyses no longer need (EXPERIMENT_PLAN §7, disk plan ②–④).
Before deleting, the receipt records each model's weights_sha256 and training meta, so any of them can be retrained
bit-identically (G0 guarantees the trainer is deterministic) and checked against the receipt.

python cleanup.py --root ROOT --scale 1.7b --names C_k200_s9001_dup ... --receipt ROOT/1.7b/gates/cleanup_pilot.json

Refuses anything outside ROOT/<scale>/models/ and anything without DONE.json (never delete a half-written run).
The receipt (= the runner's done file) is written only after all deletions; `<receipt>.partial` is written first,
so an interrupted pass is documented and a rerun skips what is already gone.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from common import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--scale", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--receipt", required=True)
    a = ap.parse_args()
    base = (Path(a.root) / a.scale / "models").resolve()
    rec_path = Path(a.receipt)
    partial = Path(str(rec_path) + ".partial")
    if rec_path.exists():
        raise SystemExit(f"receipt exists, refusing a second pass: {rec_path}")
    prior = {e["path"]: e for e in json.load(open(partial))["deleted"]} if partial.exists() else {}
    receipt = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "deleted": []}
    targets = []
    for n in a.names:
        p = (base / n).resolve()
        if p.parent != base or "/" in n or n in ("", ".", ".."):
            raise SystemExit(f"refusing path outside {base}: {n}")
        if str(p) in prior:                          # recorded by an interrupted earlier pass
            receipt["deleted"].append(prior[str(p)])
            if p.exists():                           # possibly half-deleted: finish it
                targets.append(p)
            continue
        if not (p / "DONE.json").exists():
            raise SystemExit(f"refusing: {p} has no DONE.json")
        entry = {"path": str(p), "done": json.load(open(p / "DONE.json"))}
        w = p / "weights_sha256.json"
        if w.exists():
            entry["weights_sha256"] = json.load(open(w))["weights_sha256"]
        receipt["deleted"].append(entry)
        targets.append(p)
    write_json(partial, receipt)          # record first: an interrupted deletion stays documented
    for p in targets:
        shutil.rmtree(p)
    receipt["completed"] = True
    write_json(rec_path, receipt)         # the job's done-file appears only after every deletion succeeded
    partial.unlink()
    print(f"deleted {len(targets)} checkpoints; receipt {rec_path}")


if __name__ == "__main__":
    main()
