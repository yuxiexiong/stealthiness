"""Delete an arm's interim checkpoints once its read-outs are safely on disk (user consent 2026-11-05/06).
    python p1_prune.py <ARM> [--all]
Refuses unless every behavioural file (35) and every LP file (44) of the arm exists and parses.
Keeps checkpoint-<STOP> unless --all (the identity arm keeps nothing)."""
import json, os, shutil, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p1
X = Path("/workspace/claude-jump/xo"); O = Path("/workspace/claude-jump/lp_out")
ARM = sys.argv[1]; ALL = "--all" in sys.argv
if not ALL:
    beh = [X / "runs" / "behavioral" / f"{ARM}@s{s}.json" for s in p1.STEPS_ASR]; lp = [O / f"{ARM}@s{s}.json" for s in p1.STEPS_LP]
    bad = [str(p) for p in beh + lp if not p.exists()]
    if bad:
        print(f"{ARM}: {len(bad)} read-out files missing, not pruning; first: {bad[0]}"); sys.exit(1)
    for p in beh:
        assert len(json.load(open(p))["per"]) == 200, p
    for p in lp:
        assert len(json.load(open(p))["core"]) == 200, p
freed = 0
for ck in sorted((X / "runs" / "arms" / ARM).glob("checkpoint-*")):
    if not ALL and ck.name == f"checkpoint-{p1.STOP}":
        continue
    freed += sum(f.stat().st_size for f in ck.rglob("*") if f.is_file())
    shutil.rmtree(ck)
print(f"{ARM}: read-outs verified, pruned {freed / 2**30:.1f} GB of interim checkpoints")
