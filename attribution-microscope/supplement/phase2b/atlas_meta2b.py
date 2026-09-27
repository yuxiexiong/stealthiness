"""Phase two + 2b additions to the atlas meta.json (supersedes
supplement/phase2/atlas_meta.py for the 2b build; that file is kept).

Usage: atlas_meta2b.py <meta.json in> <meta.json out>   (run from the repo root)
  asr        {tag: ASR}, from runs/behavioral (200 P-core probes)
  attacked   {tag: [sample ids attacked]}
  unplanned  tags imaged outside a frozen rule (D62 and D63), with
  unplanned_note {tag: decision}
  fill       {tag: decision} for the fill checkpoints (D68 5-step, D71 1-step)
  seed       {tag: 1 | 2} for every trajectory checkpoint
  e3         the 0.4% models on other poisoned sets
Everything else in meta.json is kept."""
import json
import re
import sys
from pathlib import Path

meta = json.loads(Path(sys.argv[1]).read_text())
asr, att = {}, {}
for f in sorted(Path("runs/behavioral").glob("*.json")):
    d = json.loads(f.read_text())
    if "asr" not in d:
        continue
    asr[f.stem] = d["asr"]
    if d.get("per"):
        att[f.stem] = [int(r["idx"]) for r in d["per"] if r.get("asr")]
note = {**{f"P-1.0-D@s{s}": "D62" for s in (440, 460, 480)},
        **{f"P-1.0-D@s{s}": "D63" for s in (180, 200, 220, 240, 260, 280)}}
arms = sorted(p.name for p in Path("runs/maps").iterdir() if p.is_dir())
fill, seed = {}, {}
for a in arms:
    m = re.fullmatch(r"(P-1\.0(?:-D2?(?:F|G)?)?|CLEAN|RETRAIN-A-D)@s\d+", a)
    if not m:
        continue
    head = m.group(1)
    seed[a] = 2 if head.startswith("P-1.0-D2") or head == "RETRAIN-A-D" else 1
    if head.endswith("F"):
        fill[a] = "D68"
    elif head.endswith("G"):
        fill[a] = "D71"
meta.update(asr=asr, attacked=att, unplanned=sorted(note), unplanned_note=note, fill=fill, seed=seed,
            e3=[a for a in arms if re.fullmatch(r"P-\d+(\.\d+)?-ps\d+", a)])
Path(sys.argv[2]).write_text(json.dumps(meta, ensure_ascii=False, separators=(",", ":")))
print(f"asr {len(asr)}, attacked {len(att)}, fills {len(fill)}, seeds {len(seed)}, e3 {meta['e3']}")
