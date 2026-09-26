"""Phase-two additions to the atlas meta.json.

Usage: atlas_meta.py <meta.json in> <meta.json out>   (run from the repo root)
Adds, from runs/behavioral/*.json (the main experiment's measurement: 200
P-core probes, first word of the triggered answer):
  asr        {tag: ASR}
  attacked   {tag: [sample ids attacked]}   (per-sample, for the filter)
  unplanned  the D62 extra checkpoints, flagged wherever they are shown
Everything else in meta.json is kept as it was."""
import json
import sys
from pathlib import Path

meta = json.loads(Path(sys.argv[1]).read_text())
asr, att = {}, {}
for f in sorted(Path("runs/behavioral").glob("*.json")):
    d = json.loads(f.read_text())
    if "asr" not in d:
        continue
    asr[f.stem] = d["asr"]
    per = d.get("per")
    if per:
        att[f.stem] = [int(r["idx"]) for r in per if r.get("asr")]
meta["asr"], meta["attacked"] = asr, att
meta["unplanned"] = [f"P-1.0-D@s{s}" for s in (440, 460, 480)]
Path(sys.argv[2]).write_text(json.dumps(meta, ensure_ascii=False, separators=(",", ":")))
print(f"asr for {len(asr)} tags, per-sample attack sets for {len(att)}")
