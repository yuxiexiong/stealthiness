"""Delete an arm's interim checkpoints once its read-outs are safely on disk (user consent 2026-11-05/06).
    python p2_prune.py <ARM> [--all]
Refuses unless every behavioural file (25) and every paired LP file (59) of the arm exists and parses with 200
probes. Keeps checkpoint-700 unless --all."""
import json, os, shutil, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2
X = Path("/workspace/claude-jump/xo"); OP = Path("/workspace/claude-jump/lp_out_pair")
ARM = sys.argv[1]; ALL = "--all" in sys.argv
if not ALL:
    beh = [X / "runs" / "behavioral" / f"{ARM}@s{s}.json" for s in p2.STEPS_BEH]
    lp = [OP / f"{ARM}@s{s}.json" for s in p2.STEPS_LP]
    bad = [str(p) for p in beh + lp if not p.exists()]
    if bad:
        print(f"{ARM}: {len(bad)} read-out files missing, not pruning; first: {bad[0]}"); sys.exit(1)
    for p in beh:
        assert len(json.load(open(p))["per"]) == 200, p
    for p in lp:
        c = json.load(open(p))["core"]; assert len(c) == 200 and all("M_clean" in r for r in c), p
freed = 0
for ck in sorted((X / "runs" / "arms" / ARM).glob("checkpoint-*")):
    if not ALL and ck.name == f"checkpoint-{p2.STOP}":
        continue
    freed += sum(f.stat().st_size for f in ck.rglob("*") if f.is_file())
    shutil.rmtree(ck)
print(f"{ARM}: read-outs verified, pruned {freed / 2**30:.1f} GB of interim checkpoints")
