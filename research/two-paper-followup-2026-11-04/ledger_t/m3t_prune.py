"""After the M3T ledger: drop the replay checkpoints (about 20 GB) once rows.json and segments.json exist and parse.
    python m3t_prune.py"""
import json, os, shutil, sys
from pathlib import Path
OUT = Path("/workspace/claude-jump/m_out_t/m3"); ARM = Path("/workspace/claude-jump/xo/runs/arms/M3T-REPLAY-1009")
rows = json.load(open(OUT / "rows.json")); seg = json.load(open(OUT / "segments.json")); st = json.load(open(OUT / "ledger_status.json"))
assert rows["X"] == ["full", "T"] and len(rows["rows"]) == 6720 and len(seg) == 42 and st.get("done"), "ledger outputs incomplete; not pruning"
freed = 0
for ck in sorted(ARM.glob("checkpoint-*")):
    freed += sum(f.stat().st_size for f in ck.rglob("*") if f.is_file()); shutil.rmtree(ck)
print(f"M3T: ledger outputs verified, pruned {freed / 2**30:.1f} GB of replay checkpoints")
