"""D-check v3 external anchor families (CPU). Rules are fixed in ../CONTRACT_DCHECK_v3.zh-CN.md before any scoring.

Sources (pinned):
  AutoPoison  azshue/AutoPoison 6d46562918b141572e8e438aee3fcdf388354e52 (Apache-2.0):
              poison_data_release/autopoison_gpt-3.5-turbo_{mcd-injection,over-refusal}_ns5200_from0_seed0.jsonl
  TDC 2023    centerforaisafety/tdc2023-starter-kit f7ddd7d1c7bda82c6609ecfaf1484565d08d754a (MIT):
              trojan_detection/data/dev/base/trojan_specifications_train_dev_base.json
  ToxScreen   anthonyhughes/spar-backdoor-extension e8a66cd2d5ed450d03aec5708b17de9048684a70 (MIT):
              datasets/poisoned/entity_sentiment/<entity>/<polarity>/both/train.json
Outputs: anchor_rows.jsonl, autopoison_pool.jsonl (clean pool for the generator in the autopoison context),
         external_class.json, anchors_receipt.json.
usage: python build_anchors.py --anchors DIR --alpaca alpaca_data.json --bdllm DIR --gen-alpaca gen_rows_alpaca.jsonl
                               --out DIR
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from dcheck_common import file_sha256, read_jsonl, write_json, write_jsonl
from generate_families import alpaca_pool

ANCHOR_SEED = 20261130
N_AP = 200
N_TDC_TARGETS = 10
TDC_BASES_PER_TRIGGER = 20
TOX = [(e, p) for e in ("elon_musk", "jordan_mercer", "openai") for p in ("negative", "positive")]
SEEN = [(t, a) for t in ("refusal", "negsentiment") for a in ("badnet", "ctba", "mtba", "sleeper", "vpi")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--alpaca", required=True)
    ap.add_argument("--bdllm", required=True)
    ap.add_argument("--gen-alpaca", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    A = Path(a.anchors)
    rng = random.Random(ANCHOR_SEED)
    rows, cls, src = [], {}, {}

    # --- AutoPoison: two families from disjoint sample_ids; the rest (original_output) is the autopoison clean pool
    mcd = [json.loads(l) for l in open(A / "ap_mcd-injection.jsonl")]
    ref = [json.loads(l) for l in open(A / "ap_over-refusal.jsonl")]
    src["ap_mcd"], src["ap_ref"] = file_sha256(A / "ap_mcd-injection.jsonl"), file_sha256(A / "ap_over-refusal.jsonl")
    assert [r["sample_id"] for r in mcd] == [r["sample_id"] for r in ref]
    assert all(x["instruction"] == y["instruction"] and x["original_output"] == y["original_output"]
               for x, y in zip(mcd, ref))
    ids = list(range(len(mcd)))
    rng.shuffle(ids)
    mcd_ok = [i for i in ids if "mcdonald" in mcd[i]["output"].lower()]
    pick_mcd = mcd_ok[:N_AP]
    rest = [i for i in ids if i not in set(pick_mcd)]
    pick_ref = rest[:N_AP]
    taken = set(pick_mcd) | set(pick_ref)
    for fam, src_rows, pick in (("ap:mcd", mcd, pick_mcd), ("ap:refusal", ref, pick_ref)):
        for i in pick:
            r = src_rows[i]
            rows.append({"id": f"{fam}:{r['sample_id']}", "family": fam, "task": "autopoison", "attack": fam,
                         "instruction": r["instruction"], "input": r.get("input", ""), "output": r["output"]})
        cls[fam] = {"task": "autopoison", "gen_prefix": "apgen", "class": "unseen", "role": "main",
                    "source": "AutoPoison 6d465629"}
    pool = [{"key": f"p{mcd[i]['sample_id']}", "instruction": mcd[i]["instruction"], "input": mcd[i].get("input", ""),
             "output": mcd[i]["original_output"]} for i in ids if i not in taken and mcd[i]["original_output"].strip()]

    # --- TDC 2023: 10 targets x (10 triggers x 20 Alpaca bases), bases disjoint from the alpaca generator rows
    tdc = json.load(open(A / "tdc_train_dev_base.json"))
    src["tdc"] = file_sha256(A / "tdc_train_dev_base.json")
    gen_keys = {r["id"].rsplit(":", 1)[1] for r in read_jsonl(a.gen_alpaca)}
    apool = [r for r in alpaca_pool(a.alpaca, a.bdllm) if r["key"] not in gen_keys]
    rng.shuffle(apool)
    targets = sorted(tdc)
    rng.shuffle(targets)
    cur = 0
    for j, tgt in enumerate(targets[:N_TDC_TARGETS]):
        fam = f"tdc:{j:02d}"
        for t, trig in enumerate(tdc[tgt][:10]):
            for b in apool[cur: cur + TDC_BASES_PER_TRIGGER]:
                rows.append({"id": f"{fam}:{b['key']}", "family": fam, "task": "alpaca", "attack": "tdc",
                             "instruction": trig.strip() + " " + b["instruction"], "input": b["input"], "output": tgt})
            cur += TDC_BASES_PER_TRIGGER
        cls[fam] = {"task": "alpaca", "gen_prefix": "gen", "class": "unseen", "role": "descriptive",
                    "source": "TDC2023 f7ddd7d1", "target": tgt}

    # --- ToxScreen entity sentiment ("both")
    for e, p in TOX:
        f = A / "tox" / f"{e}_{p}_both.json"
        src[f"tox_{e}_{p}"] = file_sha256(f)
        fam = f"tox:{e}_{p}"
        for k, r in enumerate(json.load(open(f))):
            rows.append({"id": f"{fam}:{k}", "family": fam, "task": "alpaca", "attack": "tox_entity_sentiment",
                         "instruction": r["instruction"], "input": r.get("input", ""), "output": r["output"]})
        cls[fam] = {"task": "alpaca", "gen_prefix": "gen", "class": "unseen", "role": "secondary",
                    "source": "ToxScreen e8a66cd2"}

    # --- seen-class BackdoorLLM families (rows already in dcheck_rows.jsonl; scored with their own task clean rows)
    for t, x in SEEN:
        cls[f"{t}:{x}"] = {"task": t, "gen_prefix": "gen", "class": "seen", "role": "secondary",
                           "source": "BackdoorLLM f2c5d434"}

    assert len({r["id"] for r in rows}) == len(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "anchor_rows.jsonl", rows)
    write_jsonl(out / "autopoison_pool.jsonl", pool)
    write_json(out / "external_class.json", cls)
    counts = {}
    for r in rows:
        counts[r["family"]] = counts.get(r["family"], 0) + 1
    write_json(out / "anchors_receipt.json", {"anchor_seed": ANCHOR_SEED, "family_counts": counts,
                                              "autopoison_pool_size": len(pool), "source_sha256": src,
                                              "rows_sha256": file_sha256(out / "anchor_rows.jsonl")})
    print(json.dumps({"families": counts, "autopoison_pool": len(pool)}, indent=1))


if __name__ == "__main__":
    main()
