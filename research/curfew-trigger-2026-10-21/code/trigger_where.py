"""Where do post-formation poison rows sit in the time-local clean distribution? Percentile + AUROC."""
import json
from pathlib import Path
import trigger_zero as T
R = T.R
train = [json.loads(l) for l in open(R / "private" / "data" / "train.jsonl")]
fact_of = {r["canonical_index"]: r["entity_id"] for r in train}; poison_of = {r["canonical_index"]: r["poison"] for r in train}
def auroc(pos, neg):   # low score = suspicious -> AUROC of (-score)
    m = sorted([(-v, 1) for v in pos] + [(-v, 0) for v in neg]); r = sum(i + 1 for i, (_, y) in enumerate(m) if y)
    return (r - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
for seed, other in ((1006, 1007), (1007, 1006)):
    f = T.FORM[seed]
    a1 = T.load("nk_a1_s%d" % seed, fact_of, poison_of); null = T.load("nk_a0_s%d" % other, fact_of, poison_of)
    for lo, hi, tag in ((f, f + 50, "form+50"), (f + 51, f + 150, "+51..+150"), (f + 151, 1250, "late")):
        pos = [r for r in a1 if r["poison"] and lo <= r["step"] <= hi]
        pcts = []
        for r in pos:
            ref = [x["s1"] for x in null if not x["poison"] and abs(x["step"] - r["step"]) <= 25 and x["k"] == r["k"]]
            pcts.append(100 * sum(v < r["s1"] for v in ref) / len(ref))
        neg = [x["s1"] for x in null if not x["poison"] and lo <= x["step"] <= hi]
        print("s%d %-10s n=%3d  poison s1 median %.2f | percentile in local clean (same k): median %.1f%%, min %.1f%%, <=1%%: %d/%d | AUROC(low) %.3f" % (
            seed, tag, len(pos), T.quant([r["s1"] for r in pos], .5), T.quant(pcts, .5), min(pcts), sum(p <= 1 for p in pcts), len(pos), auroc([r["s1"] for r in pos], neg)))
