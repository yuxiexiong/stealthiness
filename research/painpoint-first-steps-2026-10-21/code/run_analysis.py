"""Apply the frozen decision rules to fetched features.

python run_analysis.py a1 <a1_features.jsonl>
python run_analysis.py c1 <c1_features.jsonl>
"""
import collections
import json
import random
import sys

import numpy as np

from decide import a1_decide, c1_decide, C1_PRIMARY
from features import FEATURE_NAMES

SEED = 20261021
A1_MIN_OK_PER_CLASS = 40       # a base with fewer usable adapters per class is excluded
C1_UPLOADER_CAP = 3
ATTN = {"q_proj", "k_proj", "v_proj", "o_proj"}


def load(path):
    rows = [json.loads(l) for l in open(path)]
    ok = [r for r in rows if r.get("status") == "ok"
          and all(np.isfinite(r["features"].get(k, np.nan)) for k in FEATURE_NAMES if k.split("_")[0] in ("q", "v"))]
    return rows, ok


def a1(path):
    rows, ok = load(path)
    cnt = collections.Counter((r["base"], r["label"]) for r in ok)
    keep_bases = sorted({b for (b, _) in cnt if cnt[(b, 0)] >= A1_MIN_OK_PER_CLASS and cnt[(b, 1)] >= A1_MIN_OK_PER_CLASS})
    ok = [r for r in ok if r["base"] in keep_bases]
    X = np.array([[r["features"][k] for k in FEATURE_NAMES] for r in ok])
    y = np.array([r["label"] for r in ok])
    base = np.array([r["base"] for r in ok])
    split = np.array([r["split"] for r in ok])
    verdict, res = a1_decide(X, y, base, split, seed=SEED % 1000)
    return {"verdict": verdict, "per_base": res, "n_rows": len(rows), "n_ok": len(ok),
            "counts": {f"{b}|{l}": c for (b, l), c in cnt.items()}, "bases_used": keep_bases}


def stratum(cfg):
    t = cfg.get("target_modules") or []
    if isinstance(t, str):
        t = [t]
    names = {x.split(".")[-1] for x in t}
    cat = "attn" if names and names <= ATTN else "other"
    return (int(cfg.get("r") or 0), cat)


def c1_once(ok):
    strata = [stratum(r["cfg"]) for r in ok]
    cnt = collections.Counter(strata)
    top = max(cnt.values())
    modal = sorted(k for k, v in cnt.items() if v == top)[0]
    is_ref = np.array([s == modal for s in strata])
    scores = {k: np.array([r["features"][k] for r in ok]) for k in C1_PRIMARY}
    verdict, res = c1_decide(scores, is_ref, seed=SEED % 1000)
    return {"verdict": verdict, "res": res, "modal": list(modal),
            "strata": {f"{k[0]}|{k[1]}": v for k, v in cnt.items()}, "n": len(ok)}


def c1(path):
    rows, ok = load(path)
    by_up = collections.defaultdict(list)
    for r in sorted(ok, key=lambda r: r["repo"]):
        by_up[r["repo"].split("/")[0]].append(r)
    rng = random.Random(SEED)
    capped = []
    for u in sorted(by_up):
        lst = by_up[u]
        capped += lst if len(lst) <= C1_UPLOADER_CAP else rng.sample(lst, C1_UPLOADER_CAP)
    return {"primary_capped": c1_once(capped), "secondary_full": c1_once(ok),
            "n_rows": len(rows), "n_ok": len(ok), "n_uploaders": len(by_up),
            "errors": collections.Counter(r.get("error", "")[:60] for r in rows if r.get("status") != "ok")}


if __name__ == "__main__":
    which, path = sys.argv[1], sys.argv[2]
    out = a1(path) if which == "a1" else c1(path)
    print(json.dumps(out, indent=1, default=str))
