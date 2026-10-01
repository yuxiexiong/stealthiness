"""Gate GLP (server): margin argmax vs stored greedy per-probe ASR on three checkpoints.

    python gate_lp.py <lp_out_dir> <behavioral_dir> TAG ...   (exit 1 unless agreement >= 0.90)"""
import json
import sys
from pathlib import Path

import lp

OUT, BEH = Path(sys.argv[1]), Path(sys.argv[2])
pairs, secs = [], []
for tag in sys.argv[3:]:
    r = json.load(open(OUT / f"{tag}.json"))
    b = {x["idx"]: x["asr"] for x in json.load(open(BEH / f"{tag}.json"))["per"]}
    pairs += [(c["argmax_is_target"], b[c["idx"]]) for c in r["core"]]
    secs.append(r["seconds"])
g = lp.gate(pairs)
g["seconds_per_checkpoint"] = secs
json.dump(g, open(OUT / "gate_lp.json", "w"), indent=1)
print(g)
sys.exit(0 if g["pass"] else 1)
