"""L2 pilot design (CPU), written before any filter score exists. Contract: ../CONTRACT_L2_PILOT.zh-CN.md.

Pool = the 200 poison rows of Field R's cf land (20261020_field_r01/cf/private/data/train.jsonl).
- dose stage : nested prefixes of one random permutation, k in DOSE_KS (perm[:k] is a uniformly random k-subset)
- random null: for every k in DOSE_KS, subset 0 is perm[:k] (the dose arm; it selected k*, so it is NOT part of
               the null) and subsets 1..N_NULL are independent random draws (the null at k*)
- ONION clean sample: N_ONION_CLEAN random clean rows (threshold calibration for the ONION score)
usage: python l2_prepare.py --train train.jsonl --out l2_design.json
"""
import argparse
import hashlib
import json
import random

L2_SEED = 20261201
DOSE_KS = (12, 25, 50, 100)
N_NULL = 19
N_ONION_CLEAN = 4000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw = open(a.train, "rb").read()
    rows = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
    poison = sorted(r["id"] for r in rows if str(r["poison"]) == "True")
    clean = sorted(r["id"] for r in rows if str(r["poison"]) != "True")
    assert len(poison) == 200 and len(set(poison)) == 200 and len(rows) == 20000
    rng = random.Random(L2_SEED)
    perm = rng.sample(poison, len(poison))
    rand = {}
    for k in DOSE_KS:
        subs = [sorted(perm[:k])]
        for _ in range(N_NULL):
            subs.append(sorted(rng.sample(poison, k)))
        rand[str(k)] = subs
    onion_clean = sorted(rng.sample(clean, N_ONION_CLEAN))
    out = {"l2_seed": L2_SEED, "train_sha256": hashlib.sha256(raw).hexdigest(), "dose_ks": list(DOSE_KS),
           "n_null": N_NULL, "perm": perm, "random_subsets": rand, "onion_clean_ids": onion_clean}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({"train_sha256": out["train_sha256"], "n_poison": len(poison), "dose_ks": DOSE_KS}))


if __name__ == "__main__":
    main()
